# SPDX-License-Identifier: Apache-2.0
"""Opt-in real-checkpoint check of py-vllm-mlx#104: paged hits answer like a cold request.

The consistency probe's cases (chain swap, shift, deletion, repeat, extension) at
temperature 0 through the production scheduler with ``--use-paged-cache``: the warm
request must emit the same tokens as the same request served by a scheduler whose
cache has never seen the other prompts, and must reuse exactly the blocks that were
computed behind its own prefix.

    VLLM_MLX_REAL_CACHE_MODEL_TESTS=1 VLLM_MLX_KV_MODEL=<local model dir> \
        pytest tests/test_paged_cache_real_model_consistency.py -m slow
"""

import os
import random
from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core")

from mlx_lm import load  # noqa: E402
from mlx_lm.models.cache import KVCache, make_prompt_cache  # noqa: E402

from vllm_mlx.request import Request, SamplingParams  # noqa: E402
from vllm_mlx.scheduler import Scheduler, SchedulerConfig  # noqa: E402

pytestmark = pytest.mark.slow

BS = 16
MAX_TOKENS = 12


@pytest.fixture(scope="module")
def model_and_tokenizer():
    if os.environ.get("VLLM_MLX_REAL_CACHE_MODEL_TESTS") != "1":
        pytest.skip("set VLLM_MLX_REAL_CACHE_MODEL_TESTS=1 for real checkpoints")
    path = os.environ.get("VLLM_MLX_KV_MODEL")
    if not path or not Path(path).joinpath("config.json").is_file():
        pytest.fail("VLLM_MLX_KV_MODEL must name a complete local model artifact")
    model, tokenizer = load(path)
    assert all(isinstance(layer, KVCache) for layer in make_prompt_cache(model))
    return model, tokenizer


def _scheduler(model, tokenizer):
    return Scheduler(
        model,
        tokenizer,
        SchedulerConfig(
            enable_prefix_cache=True,
            use_paged_cache=True,
            paged_cache_block_size=BS,
            max_cache_blocks=512,
            max_num_seqs=1,
            prefill_batch_size=1,
            completion_batch_size=1,
        ),
    )


def _generate(scheduler, request_id, prompt):
    request = Request(
        request_id=request_id,
        prompt=list(prompt),
        sampling_params=SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0),
    )
    scheduler.add_request(request)
    emitted = []
    for _ in range(MAX_TOKENS + 40):
        result = scheduler.step()
        for output in result.outputs:
            emitted.extend(output.new_token_ids)
        if request_id in result.finished_request_ids:
            return emitted, request
    raise AssertionError(f"request {request_id} did not finish")


def _blocks(seed, blocks, vocab=20000):
    rng = random.Random(seed)
    return [rng.randrange(1000, vocab) for _ in range(blocks * BS)]


def _cold(model, tokenizer, prompt):
    tokens, request = _generate(_scheduler(model, tokenizer), "cold", prompt)
    assert request.cached_tokens == 0
    return tokens


A, B, C, Z, Z1 = (_blocks(seed, 2) for seed in (1, 2, 3, 4, 5))
TAIL = [int(1234)]


def test_chain_swap_answers_like_a_cold_request(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    x3 = A + C + TAIL
    cold = _cold(model, tokenizer, x3)

    warm = _scheduler(model, tokenizer)
    _generate(warm, "x1", Z + C + TAIL)
    _generate(warm, "x2", A + B + TAIL)
    tokens, request = _generate(warm, "x3", x3)

    assert request.cached_tokens == len(A)  # C was only computed behind Z
    assert tokens == cold


def test_shift_and_deletion_answer_like_a_cold_request(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    x3 = A + C + TAIL
    cold = _cold(model, tokenizer, x3)

    shifted = _scheduler(model, tokenizer)
    _generate(shifted, "x1", Z + Z1 + C + TAIL)
    tokens, request = _generate(shifted, "x3", x3)
    assert request.cached_tokens == 0
    assert tokens == cold

    deleted = _scheduler(model, tokenizer)
    _generate(deleted, "x4", A + Z1 + C + TAIL)
    tokens, request = _generate(deleted, "x3", x3)
    assert request.cached_tokens == len(A)
    assert tokens == cold


def test_repeat_and_extension_reuse_everything_and_answer_like_cold(
    model_and_tokenizer,
):
    model, tokenizer = model_and_tokenizer
    p = A + B + C + TAIL

    warm = _scheduler(model, tokenizer)
    first, _ = _generate(warm, "first", p)
    repeat, repeat_request = _generate(warm, "repeat", p)
    assert repeat_request.cached_tokens == len(A + B + C)
    assert repeat == first == _cold(model, tokenizer, p)

    longer = p + Z1 + TAIL
    extension, extension_request = _generate(warm, "extension", longer)
    assert extension_request.cached_tokens >= len(A + B + C)
    assert extension == _cold(model, tokenizer, longer)
