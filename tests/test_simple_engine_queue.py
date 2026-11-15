# SPDX-License-Identifier: Apache-2.0
"""Bounded FIFO admission for SimpleEngine's serialized generation slot."""

import asyncio
from unittest.mock import patch

import pytest

from vllm_mlx.engine import base as engine_base
from vllm_mlx.engine.base import EngineBusy, EngineQueueFull, EngineQueueTimeout
from vllm_mlx.engine.simple import SimpleEngine, _generation_admission_hook

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _queue_engine(monkeypatch, *, limit=4, timeout=None):
    monkeypatch.setenv("VLLM_MLX_SIMPLE_ENGINE_LOCK_ADMISSION", "queue")
    monkeypatch.setenv("VLLM_MLX_SIMPLE_ENGINE_MAX_QUEUED_REQUESTS", str(limit))
    if timeout is None:
        monkeypatch.delenv("VLLM_MLX_SIMPLE_ENGINE_QUEUE_TIMEOUT_S", raising=False)
    else:
        monkeypatch.setenv("VLLM_MLX_SIMPLE_ENGINE_QUEUE_TIMEOUT_S", str(timeout))
    with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False):
        engine = SimpleEngine("test-model")
    engine._loaded = True
    return engine


@pytest.mark.anyio
async def test_queue_admits_waiters_in_fifo_order(monkeypatch):
    engine = _queue_engine(monkeypatch, limit=4)
    order = []
    release = asyncio.Event()

    async def run(name):
        async with engine._acquire_generation_slot(name):
            order.append(name)
            await release.wait()

    first = asyncio.create_task(run("first"))
    await asyncio.sleep(0)
    second = asyncio.create_task(run("second"))
    third = asyncio.create_task(run("third"))
    while engine._generation_waiters < 2:
        await asyncio.sleep(0)
    release.set()
    await asyncio.gather(first, second, third)
    assert order == ["first", "second", "third"]
    assert engine._generation_waiters == 0
    assert not engine._generation_lock.locked()


@pytest.mark.anyio
async def test_queue_full_rejects_with_retryable_error(monkeypatch):
    engine = _queue_engine(monkeypatch, limit=1)
    release = asyncio.Event()

    async def hold():
        async with engine._acquire_generation_slot("holder"):
            await release.wait()

    holder = asyncio.create_task(hold())
    await asyncio.sleep(0)
    queued = asyncio.create_task(hold())
    while engine._generation_waiters < 1:
        await asyncio.sleep(0)

    with pytest.raises(EngineQueueFull) as excinfo:
        async with engine._acquire_generation_slot("overflow"):
            pass
    assert excinfo.value.code == "text_generation_queue_full"
    assert excinfo.value.retry_after_s >= 1
    assert isinstance(excinfo.value, EngineBusy)
    assert engine._generation_busy_rejections == 1
    assert engine._generation_waiters == 1

    release.set()
    await asyncio.gather(holder, queued)
    assert engine._generation_waiters == 0


@pytest.mark.anyio
async def test_cancelled_queued_request_never_runs(monkeypatch):
    engine = _queue_engine(monkeypatch, limit=4)
    release = asyncio.Event()
    ran = []

    async def holder():
        async with engine._acquire_generation_slot("holder"):
            await release.wait()

    async def waiter():
        async with engine._acquire_generation_slot("dropped"):
            ran.append("dropped")

    holder_task = asyncio.create_task(holder())
    await asyncio.sleep(0)
    dropped = asyncio.create_task(waiter())
    while engine._generation_waiters < 1:
        await asyncio.sleep(0)

    dropped.cancel()
    with pytest.raises(asyncio.CancelledError):
        await dropped
    assert engine._generation_waiters == 0
    assert ran == []

    release.set()
    await holder_task
    async with engine._acquire_generation_slot("after"):
        ran.append("after")
    assert ran == ["after"]


@pytest.mark.anyio
async def test_queue_timeout_returns_retryable_error(monkeypatch):
    engine = _queue_engine(monkeypatch, limit=4, timeout=0.05)
    release = asyncio.Event()

    async def holder():
        async with engine._acquire_generation_slot("holder"):
            await release.wait()

    holder_task = asyncio.create_task(holder())
    await asyncio.sleep(0)
    with pytest.raises(EngineQueueTimeout) as excinfo:
        async with engine._acquire_generation_slot("slow"):
            pass
    assert excinfo.value.code == "text_generation_queue_timeout"
    assert engine._generation_waiters == 0
    assert engine._generation_busy_rejections == 1
    release.set()
    await holder_task


@pytest.mark.anyio
async def test_admission_hook_counts_each_admitted_request_once(monkeypatch):
    engine = _queue_engine(monkeypatch, limit=4)
    release = asyncio.Event()
    admitted = []
    token = _generation_admission_hook.set(lambda: admitted.append(1))
    try:
        holder = asyncio.create_task(_hold_gen(engine, release))
        await asyncio.sleep(0)
        queued = asyncio.create_task(_hold_gen(engine, release))
        while engine._generation_waiters < 1:
            await asyncio.sleep(0)
        release.set()
        await asyncio.gather(holder, queued)
    finally:
        _generation_admission_hook.reset(token)
    assert len(admitted) == 2


async def _hold_gen(engine, release):
    async with engine._acquire_generation_slot("req"):
        await release.wait()


@pytest.mark.anyio
async def test_default_fail_fast_still_rejects_without_queueing(monkeypatch):
    monkeypatch.delenv("VLLM_MLX_SIMPLE_ENGINE_LOCK_ADMISSION", raising=False)
    monkeypatch.delenv("VLLM_MLX_SIMPLE_ENGINE_MAX_QUEUED_REQUESTS", raising=False)
    with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False):
        engine = SimpleEngine("test-model")
    engine._loaded = True
    assert engine._generation_lock_admission == "fail_fast"
    release = asyncio.Event()

    async def holder():
        async with engine._acquire_generation_slot("holder"):
            await release.wait()

    holder_task = asyncio.create_task(holder())
    await asyncio.sleep(0)
    with pytest.raises(EngineBusy) as excinfo:
        async with engine._acquire_generation_slot("late"):
            pass
    assert excinfo.value.code == "text_generation_busy"
    assert engine._generation_waiters == 0
    release.set()
    await holder_task


def test_queue_mode_without_limit_falls_back_to_fail_fast(monkeypatch):
    monkeypatch.setenv("VLLM_MLX_SIMPLE_ENGINE_LOCK_ADMISSION", "queue")
    monkeypatch.delenv("VLLM_MLX_SIMPLE_ENGINE_MAX_QUEUED_REQUESTS", raising=False)
    with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False):
        engine = SimpleEngine("test-model")
    assert engine._generation_lock_admission == "fail_fast"


def test_queue_error_codes_are_distinct_busy_subclasses():
    assert issubclass(engine_base.EngineQueueFull, EngineBusy)
    assert issubclass(engine_base.EngineQueueTimeout, EngineBusy)
    assert engine_base.EngineQueueFull.code != engine_base.EngineQueueTimeout.code
