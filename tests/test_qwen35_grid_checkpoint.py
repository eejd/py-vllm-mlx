# SPDX-License-Identifier: Apache-2.0
"""Grid replay against a tiny, randomly initialized real hybrid model."""

from types import SimpleNamespace

import mlx.core as mx
import pytest

from tests.mllm_cache_sim import make_request
from vllm_mlx.memory_cache import MemoryCacheConfig
from vllm_mlx.mllm_batch_generator import MLLMBatchGenerator


@pytest.mark.parametrize(("shared", "boundary"), [(7, 4), (8, 8), (9, 8)])
def test_qwen35_grid_replay_matches_cold_greedy_and_logprobs(shared, boundary):
    config_module = pytest.importorskip("mlx_vlm.models.qwen3_5.config")
    language_module = pytest.importorskip("mlx_vlm.models.qwen3_5.language")
    if not hasattr(config_module, "TextConfig") or not hasattr(language_module, "LanguageModel"):
        pytest.skip("installed mlx-vlm lacks Qwen3.5 language model classes")

    mx.random.seed(17)
    config = config_module.TextConfig(
        model_type="qwen3_5",
        hidden_size=64,
        intermediate_size=128,
        linear_num_value_heads=2,
        linear_num_key_heads=2,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        rms_norm_eps=1e-6,
        vocab_size=64,
        max_position_embeddings=128,
        full_attention_interval=2,
        rope_parameters={
            "type": "default",
            "mrope_section": [4, 4, 8],
            "rope_theta": 10000,
            "partial_rotary_factor": 1.0,
        },
    )
    model = language_module.LanguageModel(config)
    mx.eval(model.parameters())
    cache_types = {type(cache).__name__ for cache in model.make_cache()}
    assert "ArraysCache" in cache_types and "KVCache" in cache_types

    def generator():
        gen = MLLMBatchGenerator(
            SimpleNamespace(language_model=model),
            SimpleNamespace(),
            max_tokens=4,
            prefill_step_size=4,
            hybrid_prefix_checkpoints=True,
            prefix_cache_config=MemoryCacheConfig(max_memory_mb=64, min_prefix_tokens=1),
            enable_vision_cache=False,
        )

        # Text-only positions have zero mRoPE delta; this language-only model
        # has no enclosing multimodal config for get_rope_index to inspect.
        def text_rope_positions(request):
            length = request.input_ids.shape[1]
            request.prompt_position_ids = mx.broadcast_to(
                mx.arange(length)[None, None, :], (3, 1, length)
            )
            return mx.zeros((1, 1), dtype=mx.int32)

        gen._derive_request_rope_deltas = text_rope_positions
        return gen

    def generate(gen, request_id, tokens):
        gen.insert([make_request(request_id, tokens, 4)])
        result = []
        for _ in range(100):
            for response in gen.next():
                if response.request_id == request_id:
                    result.append((response.token, float(response.logprobs[response.token].item())))
                    if response.finish_reason is not None:
                        return result
        pytest.fail("generation did not finish")

    source = list(range(1, 16))
    query = source[:shared] + [40, 41, 42]
    warm = generator()
    generate(warm, "source", source)
    cached, remaining = warm.prefix_cache.fetch_grid(query, 4)
    assert cached is not None
    assert len(query) - len(remaining) == boundary
    resumed = generate(warm, "resumed", query)
    cold = generate(generator(), "cold", query)
    assert resumed == cold
