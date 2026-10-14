# SPDX-License-Identifier: Apache-2.0
"""The SpecPrefill draft model is loaded on the thread that scores with it.

MLX streams are thread-local: an array built on one thread cannot be evaluated
on another ("There is no Stream(gpu, N) in current thread").  ``SimpleEngine``
runs every MLX call, SpecPrefill scoring included, on its generation worker and
loads the target model there, but it used to load the draft model on the event
loop -- so on a real server every ``--specprefill`` request logged
"SpecPrefill failed, falling back to normal path" and silently ran the full
prefill.
"""

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import pytest

from vllm_mlx.engine.simple import SimpleEngine

pytestmark = pytest.mark.anyio

DRAFT_PATH = "/models/draft"


class FakeLanguageModel:
    """Stands in for ``MLXLanguageModel``: nothing to load, nothing to run."""

    def __init__(self, *_args, **_kwargs):
        self.model = SimpleNamespace()
        self.tokenizer = SimpleNamespace()

    def load(self):
        pass


def _engine(**kwargs):
    return SimpleEngine(
        "target",
        specprefill_enabled=True,
        specprefill_draft_model=DRAFT_PATH,
        **kwargs,
    )


async def _worker_thread(engine):
    return await asyncio.get_running_loop().run_in_executor(
        engine._generation_worker(), threading.get_ident
    )


def _lazy_draft_load(loaded_on):
    """``mlx_lm.load`` stand-in: weights left as a pending graph, as real loads do."""

    def load(path):
        loaded_on.append((path, threading.get_ident()))
        return {"weight": mx.ones((8, 8)) * 2}, None

    return load


async def test_draft_model_is_loaded_on_the_generation_worker():
    loaded_on = []
    engine = _engine()
    try:
        with (
            patch("vllm_mlx.models.llm.MLXLanguageModel", FakeLanguageModel),
            patch("mlx_lm.load", _lazy_draft_load(loaded_on)),
        ):
            await engine.start()
        assert loaded_on == [(DRAFT_PATH, await _worker_thread(engine))]
        assert loaded_on[0][1] != threading.get_ident()
    finally:
        await engine.stop()


async def test_draft_weights_can_be_evaluated_by_the_scoring_thread():
    # What score_tokens does with the draft on the worker: evaluate its arrays.
    engine = _engine()
    try:
        with (
            patch("vllm_mlx.models.llm.MLXLanguageModel", FakeLanguageModel),
            patch("mlx_lm.load", _lazy_draft_load([])),
        ):
            await engine.start()
        weight = engine._draft_model["weight"]
        await engine._run_blocking_serialized(lambda: mx.eval(weight))
        assert weight.tolist() == [[2.0] * 8] * 8
    finally:
        await engine.stop()


async def test_failed_draft_load_leaves_specprefill_off_and_the_engine_up():
    def broken_load(_path):
        raise OSError("no such model")

    engine = _engine()
    try:
        with (
            patch("vllm_mlx.models.llm.MLXLanguageModel", FakeLanguageModel),
            patch("mlx_lm.load", broken_load),
        ):
            await engine.start()
        assert engine._draft_model is None
        assert engine._loaded is True
    finally:
        await engine.stop()


async def test_no_draft_is_loaded_unless_specprefill_is_enabled():
    loaded_on = []
    engine = SimpleEngine(
        "target", specprefill_enabled=False, specprefill_draft_model=DRAFT_PATH
    )
    try:
        with (
            patch("vllm_mlx.models.llm.MLXLanguageModel", FakeLanguageModel),
            patch("mlx_lm.load", _lazy_draft_load(loaded_on)),
        ):
            await engine.start()
        assert loaded_on == []
        assert engine._draft_model is None
    finally:
        await engine.stop()


async def test_no_draft_is_loaded_without_a_draft_path():
    loaded_on = []
    engine = SimpleEngine("target", specprefill_enabled=True)
    try:
        with (
            patch("vllm_mlx.models.llm.MLXLanguageModel", FakeLanguageModel),
            patch("mlx_lm.load", _lazy_draft_load(loaded_on)),
        ):
            await engine.start()
        assert loaded_on == []
        assert engine._draft_model is None
    finally:
        await engine.stop()
