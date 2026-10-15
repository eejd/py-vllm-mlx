# SPDX-License-Identifier: Apache-2.0
"""Opt-in real-checkpoint checks of the persisted prefix cache and the SSD tier.

Cold-versus-warm at temperature 0 through the production scheduler: an entry
written under another model, or flushed, must not change the answer.

    VLLM_MLX_REAL_CACHE_MODEL_TESTS=1 VLLM_MLX_KV_MODEL=<local model dir> \
        [VLLM_MLX_ROTATING_MODEL=<local model dir>] \
        pytest tests/test_cache_identity_real_model.py --run-slow -m slow

``VLLM_MLX_KV_MODEL`` is a plain-KV text model; ``VLLM_MLX_ROTATING_MODEL``
(optional) one with sliding-window (rotating) layers, e.g. Gemma-3, to exercise
the persisted and SSD formats on them too.
"""

import os
import time
from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core")

from mlx_lm import load  # noqa: E402

from vllm_mlx.request import Request, SamplingParams  # noqa: E402
from vllm_mlx.scheduler import Scheduler, SchedulerConfig  # noqa: E402

pytestmark = pytest.mark.slow

MAX_TOKENS = 12


def _model_dir(variable):
    if os.environ.get("VLLM_MLX_REAL_CACHE_MODEL_TESTS") != "1":
        pytest.skip("set VLLM_MLX_REAL_CACHE_MODEL_TESTS=1 for real checkpoints")
    value = os.environ.get(variable)
    if not value:
        pytest.skip(f"{variable} not set")
    if not Path(value).joinpath("config.json").is_file():
        pytest.fail(f"{variable} must name a complete local model artifact")
    return value


@pytest.fixture(
    params=["VLLM_MLX_KV_MODEL", "VLLM_MLX_ROTATING_MODEL"],
    ids=["plain_kv", "rotating"],
)
def loader(request):
    path = _model_dir(request.param)
    return lambda: load(path)


def _perturbed(model):
    """The same architecture and shapes with other weights: a stand-in for another
    fine-tune or a new revision downloaded over the same path."""
    for layer in model.layers:
        proj = layer.self_attn.o_proj
        for name in ("scales", "weight"):
            value = getattr(proj, name, None)
            if value is not None and value.dtype in (mx.float16, mx.bfloat16):
                setattr(proj, name, value * 1.25)
                break
    mx.eval(model.parameters())
    return model


def _prompt(tokenizer, n=200):
    seed = tokenizer.encode(
        "The persisted cache must never serve keys and values that another model "
        "computed. "
    )
    return [int(t) for t in (seed * (n // len(seed) + 1))[:n]]


def _scheduler(model, tokenizer, **cfg):
    return Scheduler(
        model,
        tokenizer,
        SchedulerConfig(
            enable_prefix_cache=True,
            use_memory_aware_cache=True,
            cache_memory_mb=512,
            max_num_seqs=1,
            prefill_batch_size=1,
            completion_batch_size=1,
            **cfg,
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
        if request.request_id in result.finished_request_ids:
            return emitted, request
    raise AssertionError(f"request {request_id} did not finish")


def _wait_spilled(scheduler, count=1):
    tier = scheduler._ssd_tier
    deadline = time.monotonic() + 30
    while tier.get_stats()["spill_count"] < count and time.monotonic() < deadline:
        time.sleep(0.05)
    assert tier.get_stats()["spill_count"] >= count


def test_persisted_cache_of_another_checkpoint_does_not_change_the_answer(
    loader, tmp_path
):
    model_a, tokenizer = loader()
    model_b, _ = loader()
    _perturbed(model_b)
    prompt = _prompt(tokenizer)

    sched_a = _scheduler(model_a, tokenizer)
    out_a, _ = _generate(sched_a, "a", prompt)
    assert sched_a.save_cache_to_disk(str(tmp_path))

    cold_b, _ = _generate(_scheduler(model_b, tokenizer), "b-cold", prompt)
    assert (
        cold_b != out_a
    ), "the perturbation must change the answer or this proves nothing"

    sched_b = _scheduler(model_b, tokenizer)
    assert sched_b.load_cache_from_disk(str(tmp_path)) == 0
    warm_b, request = _generate(sched_b, "b-warm", prompt)
    assert request.cached_tokens == 0
    assert warm_b == cold_b


def test_persisted_cache_of_the_same_checkpoint_is_reused_and_loses_nothing(
    loader, tmp_path
):
    model, tokenizer = loader()
    prompt = _prompt(tokenizer)
    sched = _scheduler(model, tokenizer)
    out, _ = _generate(sched, "a", prompt)
    saved = {
        k: [lyr.offset for lyr in e.cache]
        for k, e in sched.memory_aware_cache._entries.items()
    }
    assert saved and sched.save_cache_to_disk(str(tmp_path))

    reloaded, _ = loader()  # the same checkpoint, loaded again
    sched2 = _scheduler(reloaded, tokenizer)
    assert sched2.load_cache_from_disk(str(tmp_path)) == len(saved)
    restored = {
        k: [lyr.offset for lyr in e.cache]
        for k, e in sched2.memory_aware_cache._entries.items()
    }
    assert restored == saved
    # a request that extends the persisted prompt reuses it
    follow_up = prompt + out + prompt[:8]
    _, request = _generate(sched2, "b", follow_up)
    assert request.cached_tokens >= len(prompt)


def test_ssd_tier_shared_by_two_checkpoints_does_not_promote_across(loader, tmp_path):
    model_a, tokenizer = loader()
    model_b, _ = loader()
    _perturbed(model_b)
    prompt = _prompt(tokenizer)

    sched_a = _scheduler(model_a, tokenizer, ssd_cache_dir=str(tmp_path))
    out_a, _ = _generate(sched_a, "a", prompt)
    sched_a.memory_aware_cache._evict_lru()
    _wait_spilled(sched_a)
    sched_a.close_ssd_tier()
    follow_up = prompt + out_a + prompt[:20]

    cold_b, _ = _generate(_scheduler(model_b, tokenizer), "b-cold", follow_up)

    sched_b = _scheduler(model_b, tokenizer, ssd_cache_dir=str(tmp_path))
    warm_b, request = _generate(sched_b, "b-warm", follow_up)
    try:
        assert sched_b._ssd_tier.get_stats()["ssd_hits"] == 0
        assert request.cache_hit_type not in ("ssd_hit", "ssd_pending")
        assert warm_b == cold_b
    finally:
        sched_b.close_ssd_tier()


def test_ssd_tier_of_the_same_checkpoint_is_promoted_after_a_restart(loader, tmp_path):
    model, tokenizer = loader()
    prompt = _prompt(tokenizer)
    sched = _scheduler(model, tokenizer, ssd_cache_dir=str(tmp_path))
    out, _ = _generate(sched, "a", prompt)
    sched.memory_aware_cache._evict_lru()
    _wait_spilled(sched)
    sched.close_ssd_tier()

    reloaded, _ = loader()
    restarted = _scheduler(reloaded, tokenizer, ssd_cache_dir=str(tmp_path))
    try:
        _, request = _generate(restarted, "b", prompt + out + prompt[:20])
        assert request.cache_hit_type == "ssd_hit"
        assert request.cached_tokens >= len(prompt)
    finally:
        restarted.close_ssd_tier()


@pytest.mark.parametrize("flush", ["clear_runtime_caches", "_recover_from_cache_error"])
def test_a_flushed_cache_is_not_refilled_from_the_ssd_tier(loader, tmp_path, flush):
    model, tokenizer = loader()
    prompt = _prompt(tokenizer)
    sched = _scheduler(model, tokenizer, ssd_cache_dir=str(tmp_path))
    try:
        out, _ = _generate(sched, "a", prompt)
        sched.memory_aware_cache._evict_lru()
        _wait_spilled(sched)
        getattr(sched, flush)()

        follow_up = prompt + out + prompt[:20]
        warm, request = _generate(sched, "b", follow_up)
        assert request.cached_tokens == 0, "the flush left an entry to promote"
        cold, _ = _generate(_scheduler(model, tokenizer), "cold", follow_up)
        assert warm == cold
    finally:
        sched.close_ssd_tier()
