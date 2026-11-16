# SPDX-License-Identifier: Apache-2.0
"""Oversized hybrid prompts must be rejected before entering the scheduler."""
from collections import deque
from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")

from vllm_mlx.engine.base import RequestPreprocessingError
from vllm_mlx.mllm_scheduler import MLLMScheduler


def test_hybrid_prompt_memory_admission(monkeypatch):
    from vllm_mlx import mllm_scheduler

    sched = MLLMScheduler.__new__(MLLMScheduler)
    sched.processor = SimpleNamespace(tokenizer=object())
    sched.model_config = SimpleNamespace(text_config=SimpleNamespace(
        num_hidden_layers=32, full_attention_interval=4,
        num_key_value_heads=4, head_dim=256,
    ))
    sched.requests = {}
    sched.waiting = deque()
    monkeypatch.setattr(mx, "get_memory_limit", lambda: 24_159_191_040)
    monkeypatch.setattr(mx, "get_active_memory", lambda: 8_200_000_000)
    monkeypatch.setattr(mllm_scheduler, "_get_available_memory", lambda: 20_000_000_000)
    monkeypatch.setattr(mllm_scheduler, "encode_prompt", lambda *args: range(505_023))

    with pytest.raises(RequestPreprocessingError, match="memory"):
        sched.add_request(prompt="long", max_tokens=128)
    assert not sched.requests
    assert not sched.waiting

    monkeypatch.setattr(mllm_scheduler, "encode_prompt", lambda *args: range(210_111))
    request_id = sched.add_request(prompt="short", max_tokens=128)
    assert request_id in sched.requests

    # Queued requests reserve their future KV allocations, not just memory
    # already materialized by MLX.
    with pytest.raises(RequestPreprocessingError, match="memory"):
        sched.add_request(prompt="another", max_tokens=128)
    assert list(sched.requests) == [request_id]
