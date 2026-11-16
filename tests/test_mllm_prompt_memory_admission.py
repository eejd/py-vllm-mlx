# SPDX-License-Identifier: Apache-2.0
"""Hybrid prompt admission accounts for measured KV growth and queued work."""
from collections import deque
from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")

from vllm_mlx.engine.base import RequestPreprocessingError
from vllm_mlx.mllm_scheduler import MLLMScheduler, MLLMSchedulerConfig


@pytest.fixture
def scheduler(monkeypatch):
    from vllm_mlx import mllm_scheduler

    sched = MLLMScheduler.__new__(MLLMScheduler)
    sched.processor = SimpleNamespace(tokenizer=object())
    sched.config = MLLMSchedulerConfig(prefill_step_size=2048)
    sched.model_config = SimpleNamespace(text_config=SimpleNamespace(
        num_hidden_layers=32, full_attention_interval=4,
        num_key_value_heads=4, head_dim=256,
    ))
    sched.requests = {}
    sched.waiting = deque()
    monkeypatch.setattr(mx, "get_memory_limit", lambda: 24_159_191_040)
    monkeypatch.setattr(mx, "device_info", lambda: {
        "max_recommended_working_set_size": 26_843_545_600
    })
    monkeypatch.setattr(mx, "get_active_memory", lambda: 8_200_000_000)
    # raising=False makes this behavior test reach its assertion on the base
    # revision, which did not import this helper into the scheduler module.
    monkeypatch.setattr(mllm_scheduler, "_get_available_memory", lambda: 20_000_000_000, raising=False)
    return sched


@pytest.mark.parametrize("tokens", [210_111, 283_839, 431_295])
def test_measured_passing_prompts_are_admitted(scheduler, monkeypatch, tokens):
    from vllm_mlx import mllm_scheduler
    monkeypatch.setattr(mllm_scheduler, "encode_prompt", lambda *args: range(tokens))
    request_id = scheduler.add_request(prompt="long", max_tokens=128)
    if tokens == 431_295:
        soft_budget = 24_159_191_040 * 105 // 100 - 8_200_000_000
        assert soft_budget - ((tokens + 128) * 32768 + 2_000_000_000) >= 1_000_000_000
    assert request_id in scheduler.requests


def test_original_prompt_is_refused_before_admission(scheduler, monkeypatch):
    from vllm_mlx import mllm_scheduler
    monkeypatch.setattr(mllm_scheduler, "encode_prompt", lambda *args: range(578_751))
    with pytest.raises(RequestPreprocessingError, match="memory"):
        scheduler.add_request(prompt="long", max_tokens=128)
    assert not scheduler.requests
    assert not scheduler.waiting


def test_running_kv_is_not_reserved_twice_but_queued_kv_is(scheduler, monkeypatch):
    from vllm_mlx import mllm_scheduler
    monkeypatch.setattr(mx, "get_active_memory", lambda: 14_000_000_000)
    monkeypatch.setattr(mllm_scheduler, "encode_prompt", lambda *args: range(210_111))
    running = SimpleNamespace(num_prompt_tokens=210_111, sampling_params=SimpleNamespace(max_tokens=128))
    scheduler.requests["running"] = running
    request_id = scheduler.add_request(prompt="next", max_tokens=128)
    assert request_id in scheduler.requests
    with pytest.raises(RequestPreprocessingError, match="memory"):
        scheduler.add_request(prompt="queued", max_tokens=128)


def test_activation_allowance_scales_with_prefill_chunk(scheduler, monkeypatch):
    from vllm_mlx import mllm_scheduler
    monkeypatch.setattr(mllm_scheduler, "_get_available_memory", lambda: 15_500_000_000)
    monkeypatch.setattr(mllm_scheduler, "encode_prompt", lambda *args: range(431_295))
    with pytest.raises(RequestPreprocessingError, match="memory"):
        scheduler.add_request(prompt="long", max_tokens=128)
    scheduler.config.prefill_step_size = 512
    assert scheduler.add_request(prompt="long", max_tokens=128) in scheduler.requests


def test_admission_ceiling_cannot_exceed_recommended_working_set(scheduler, monkeypatch):
    from vllm_mlx import mllm_scheduler
    monkeypatch.setattr(mx, "get_memory_limit", lambda: 26_843_545_600)
    monkeypatch.setattr(mllm_scheduler, "_get_available_memory", lambda: 30_000_000_000)
    monkeypatch.setattr(mllm_scheduler, "encode_prompt", lambda *args: range(525_000))
    # This fits 105% of the soft limit, but not the device working set.
    with pytest.raises(RequestPreprocessingError, match="memory"):
        scheduler.add_request(prompt="long", max_tokens=128)
