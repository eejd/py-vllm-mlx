# SPDX-License-Identifier: Apache-2.0
"""MLXWorker against vLLM's engine-core worker contract (vLLM >= 0.31).

The engine core talks to its worker only through ``collective_rpc`` method
names, resolved by ``vllm.v1.serial_utils.run_method``: a name the worker lacks
aborts engine start-up with ``NotImplementedError: Method ... is not
implemented``. These tests drive the worker through that same resolution.
"""

import importlib.util
import json
import os
import tempfile
from types import SimpleNamespace

import pytest

needs_vllm_and_torch = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None
    or importlib.util.find_spec("vllm") is None,
    reason="vllm and torch are not installed",
)
pytestmark = needs_vllm_and_torch

_TINY_LLAMA_CONFIG = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "vocab_size": 100,
    "max_position_embeddings": 256,
    "torch_dtype": "float16",
}


def _layout_api():
    """vLLM's KV-cache-layout negotiation (vLLM >= 0.31), or skip."""
    try:
        from vllm.v1.attention.backends.utils import resolve_kv_cache_layout
    except ImportError:
        pytest.skip("this vLLM predates KV cache layout negotiation")
    return resolve_kv_cache_layout


@pytest.fixture
def worker():
    """An MLXWorker without a model: layout/config RPCs need only the config."""
    from vllm.config import CacheConfig
    from vllm_mlx_plugin.worker import MLXWorker

    w = object.__new__(MLXWorker)
    w.cache_config = CacheConfig()
    w.vllm_config = SimpleNamespace(
        cache_config=w.cache_config, kv_transfer_config=None
    )
    w.model_runner = None
    return w


def _rpc(worker, method, *args):
    from vllm.v1.serial_utils import run_method

    return run_method(worker, method, args, {})


def test_engine_core_layout_negotiation_resolves_and_publishes(worker):
    """Engine-core order: workers report layouts, core resolves, workers adopt."""
    resolve_kv_cache_layout = _layout_api()

    supported = _rpc(worker, "get_supported_kv_cache_layouts")
    # An empty answer is a hard error in resolve_kv_cache_layout.
    assert supported

    layout = resolve_kv_cache_layout(worker.vllm_config, [supported])
    _rpc(worker, "set_kv_cache_layout", layout.name)

    assert worker.cache_config.kv_cache_layout == layout.name
    assert layout.name in supported


def test_every_layout_is_acceptable_to_the_user_override(worker, monkeypatch):
    """MLX consumes no vLLM-side KV tensor, so VLLM_KV_CACHE_LAYOUT is never refused."""
    resolve_kv_cache_layout = _layout_api()
    from vllm.v1.attention.backends.utils import KVCacheLayout

    supported = _rpc(worker, "get_supported_kv_cache_layouts")
    assert set(supported) == {layout.name for layout in KVCacheLayout}

    monkeypatch.setenv("VLLM_KV_CACHE_LAYOUT", "BHLNC")
    assert resolve_kv_cache_layout(worker.vllm_config, [supported]).name == "BHLNC"


def test_set_kv_cache_layout_refuses_to_change_a_resolved_layout(worker):
    _layout_api()
    _rpc(worker, "set_kv_cache_layout", "LBNHC")
    _rpc(worker, "set_kv_cache_layout", "LBNHC")  # same answer again is fine
    with pytest.raises(ValueError, match="already resolved"):
        _rpc(worker, "set_kv_cache_layout", "BLNHC")


def test_initialize_from_config_adopts_the_resolved_layout(worker):
    """A worker created after resolution only learns the layout from the config."""
    _layout_api()
    kv_cache_config = SimpleNamespace(num_blocks=7, kv_cache_layout="BLNHC")

    _rpc(worker, "initialize_from_config", kv_cache_config)

    assert worker.cache_config.kv_cache_layout == "BLNHC"
    assert worker.cache_config.num_gpu_blocks == 7


def test_initialize_from_config_without_a_layout_leaves_it_unset(worker):
    _layout_api()
    _rpc(worker, "initialize_from_config", SimpleNamespace(num_blocks=3))
    assert worker.cache_config.kv_cache_layout is None


def test_profile_takes_the_prefix_the_executor_sends(worker):
    # Executor.profile -> collective_rpc("profile", args=(is_start, profile_prefix))
    assert _rpc(worker, "profile", True, "prefix") is None


@pytest.mark.parametrize(
    "method",
    [
        # Called on every engine start-up / step
        "get_kv_cache_spec",
        "get_supported_kv_cache_layouts",
        "set_kv_cache_layout",
        "determine_available_memory",
        "initialize_from_config",
        "compile_or_warm_up_model",
        "get_supported_tasks",
        "execute_model",
        "sample_tokens",
        "take_draft_token_ids",
        "execute_dummy_batch",
        "check_health",
        "shutdown",
        # Control plane (pause, cache reset, sleep, LoRA, weight-update gate)
        "synchronize_device",
        "reset_mm_cache",
        "reset_encoder_cache",
        "reset_prefix_cache",
        "update_max_model_len",
        "supports_draft_weight_updates",
        "profile",
        "sleep",
        "wake_up",
        "add_lora",
        "remove_lora",
        "pin_lora",
        "list_loras",
    ],
)
def test_worker_resolves_every_engine_core_rpc(method):
    from vllm.v1.worker.worker_base import WorkerBase
    from vllm_mlx_plugin.worker import MLXWorker

    if not hasattr(MLXWorker, method) and not hasattr(WorkerBase, method):
        # Defined only by a newer WorkerBase (set_kv_cache_layout and
        # supports_draft_weight_updates arrived after 0.27): MLXWorker inherits
        # them there, and the engine of an older vLLM never calls them.
        pytest.skip(f"this vLLM's WorkerBase predates {method}")
    assert callable(getattr(MLXWorker, method, None))
    # WorkerBase leaves these as ``raise NotImplementedError`` or as a default
    # that does not fit MLX (``synchronize_device`` waits on torch.accelerator,
    # which does not cover MLX's queue): the worker must supply its own.
    if method in {
        "get_kv_cache_spec",
        "compile_or_warm_up_model",
        "execute_model",
        "sample_tokens",
        "add_lora",
        "remove_lora",
        "pin_lora",
        "list_loras",
        "synchronize_device",
    }:
        assert getattr(MLXWorker, method) is not getattr(WorkerBase, method, None)


def test_synchronize_device_waits_on_mlx(worker, monkeypatch):
    # The engine core's pause path calls this RPC (EngineCore._finish_pause).
    mx = pytest.importorskip("mlx.core")
    calls = []
    monkeypatch.setattr(mx, "synchronize", lambda *a, **k: calls.append(1))

    assert _rpc(worker, "synchronize_device") is None
    assert calls == [1]


def test_sample_tokens_hands_the_grammar_output_to_the_runner(worker):
    seen = []
    worker.model_runner = SimpleNamespace(
        sample_tokens=lambda grammar_output: seen.append(grammar_output) or "output"
    )
    grammar = object()

    assert _rpc(worker, "sample_tokens", grammar) == "output"
    assert seen == [grammar]


def test_constructor_runs_workerbase_init_and_keeps_the_ranks():
    """MLXWorker(...) as ``WorkerWrapperBase.init_worker`` builds it."""
    from vllm.config import ModelConfig, VllmConfig
    from vllm_mlx_plugin.worker import MLXWorker

    with tempfile.TemporaryDirectory() as model_dir:
        with open(os.path.join(model_dir, "config.json"), "w") as f:
            json.dump(_TINY_LLAMA_CONFIG, f)
        model_config = ModelConfig(
            model=model_dir, tokenizer=model_dir, skip_tokenizer_init=True
        )
        vllm_config = VllmConfig(model_config=model_config)

        w = MLXWorker(
            vllm_config,
            local_rank=1,
            rank=3,
            distributed_init_method="file:///unused",
            is_driver_worker=False,
        )

    assert (w.rank, w.local_rank, w.is_driver_worker) == (3, 1, False)
    assert w.model_config is model_config
    assert w.cache_config is vllm_config.cache_config
    # WorkerBase.__init__ records the rank on the shared parallel config.
    assert w.parallel_config.rank == 3
    assert w.model_runner is None
