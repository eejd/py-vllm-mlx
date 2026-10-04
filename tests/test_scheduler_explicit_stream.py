# SPDX-License-Identifier: Apache-2.0
"""A Scheduler's BatchGenerator must not depend on the global generation_stream.

``bind_generation_streams`` rebinds ``mlx_lm.generate.generation_stream`` to a
stream owned by the calling worker thread. A BatchGenerator built without an
explicit stream captures whatever that global names at construction time, so a
generator (re)built on one thread after another engine's worker rebound the
global would hold a stream its own thread cannot enter.
"""

import importlib

import mlx.core as mx

from vllm_mlx.scheduler import Scheduler, SchedulerConfig
from vllm_mlx.request import SamplingParams


class _TinyModel:
    """Never evaluated: only BatchGenerator construction is under test."""

    layers = []

    def make_cache(self):
        return []


def test_batch_generator_uses_threads_default_stream_not_the_global(monkeypatch):
    own = mx.new_stream(mx.default_device())
    foreign = mx.new_stream(mx.default_device())
    mx.set_default_stream(own)
    # Another engine's worker rebound the process-global handle.
    # ``mlx_lm.generate`` as an attribute is the function; import the module.
    mlx_generate = importlib.import_module("mlx_lm.generate")
    monkeypatch.setattr(mlx_generate, "generation_stream", foreign)

    captured = {}

    class _Recorder:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("vllm_mlx.scheduler.BatchGenerator", _Recorder)

    scheduler = Scheduler.__new__(Scheduler)
    scheduler.model = _TinyModel()
    scheduler.config = SchedulerConfig()
    scheduler.memory_aware_cache = None
    scheduler.uid_to_request_id = {}
    scheduler._get_stop_tokens = lambda: set()
    scheduler._bounded_kv_size = lambda: None
    scheduler._create_batch_generator(SamplingParams())

    assert captured["stream"] == own
    assert captured["stream"] != foreign
