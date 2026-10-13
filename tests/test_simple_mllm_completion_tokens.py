# SPDX-License-Identifier: Apache-2.0
"""``usage.completion_tokens`` of the Simple engine's MLLM routes.

vLLM reports ``len(token_ids)``: every generated token, the one that ended the
request (EOS / stop token) included, nothing else. mlx-vlm's ``stream_generate``
yields one chunk per decoded token and then a last chunk that only flushes the
detokenizer and carries ``generation_tokens`` (the model's own count, stop token
included). The Simple engine used to count chunks, twice: the flush chunk, and
again the terminal chunk ``MLXMultimodalLM.stream_chat`` appends. A reply of 5
tokens plus ``<end_of_turn>`` was billed 7 (Gemma-3 tool round trip: 2 for a reply
that is the stop token alone), and a reply cut by ``max_tokens`` one over the cap.

The fakes below reproduce mlx-vlm's chunk sequence (``generate/dispatch.py``
``stream_generate``) and run it through the real ``_stream_mllm_generated_outputs``
and the real ``SimpleEngine`` routes.
"""

import asyncio
from types import SimpleNamespace

import pytest

from vllm_mlx.engine import simple as engine_module
from vllm_mlx.models.mllm import MLLMOutput, _stream_mllm_generated_outputs


def _mlx_vlm_chunks(n_content, *, flush_count):
    """``n_content`` per-token chunks, then the detokenizer-flush chunk."""
    chunks = [
        SimpleNamespace(text=f"t{i} ", prompt_tokens=9, generation_tokens=i + 1)
        for i in range(n_content)
    ]
    chunks.append(
        SimpleNamespace(
            text="", prompt_tokens=9, generation_tokens=flush_count, finish_reason="stop"
        )
    )
    return chunks


def _wrapped(chunks, stop=None):
    return list(
        _stream_mllm_generated_outputs(
            stream_generate_fn=lambda *a, **k: iter(chunks),
            model=None,
            processor=None,
            formatted_prompt="p",
            images=None,
            audio=None,
            max_tokens=16,
            temperature=0.0,
            prompt_cache=None,
            draft_kwargs={},
            generation_kwargs={},
            stop=stop,
            draft_metrics=lambda: {},
        )
    )


class TestStreamWrapper:
    def test_natural_stop_counts_content_plus_the_stop_token(self):
        # 5 content tokens, then <end_of_turn>: n=5 at the stop, so 6.
        outputs = _wrapped(_mlx_vlm_chunks(5, flush_count=6))

        assert outputs[-1].finish_reason == "stop"
        assert outputs[-1].completion_tokens == 6

    def test_a_reply_that_is_only_the_stop_token_is_one_token(self):
        outputs = _wrapped(_mlx_vlm_chunks(0, flush_count=1))

        assert outputs[-1].completion_tokens == 1

    def test_length_stop_is_exactly_the_cap(self):
        # mlx-vlm: n runs to max_tokens-1, the flush chunk repeats n+1 == cap.
        outputs = _wrapped(_mlx_vlm_chunks(3, flush_count=3))

        assert outputs[-1].completion_tokens == 3

    def test_running_count_follows_the_tokens(self):
        outputs = _wrapped(_mlx_vlm_chunks(3, flush_count=4))

        assert [o.completion_tokens for o in outputs] == [1, 2, 3, 4, 4]

    def test_stop_sequence_cut_counts_tokens_seen_so_far(self):
        outputs = _wrapped(_mlx_vlm_chunks(5, flush_count=6), stop=["t1"])

        # The consumer left after "t1 " (the second token).
        assert outputs[-1].completion_tokens == 2

    def test_a_chunk_without_a_count_still_counts_one_per_chunk(self):
        chunks = [SimpleNamespace(text="a"), SimpleNamespace(text="b")]

        outputs = _wrapped(chunks)

        assert [o.completion_tokens for o in outputs] == [1, 2, 2]


class _Model:
    """What ``SimpleEngine`` needs from an MLLM, driven by canned chunks."""

    _video_native = False

    def __init__(self, chunks):
        self._chunks = chunks

    def _collect_video_inputs(self, messages):
        return []

    def stream_chat(self, **kwargs):
        return iter(_wrapped(self._chunks))


class _NativeVideoModel(_Model):
    _video_native = True

    def _collect_video_inputs(self, messages):
        return ["clip"]

    def stream_chat(self, **kwargs):
        return iter(
            [
                MLLMOutput(
                    text="a clip",
                    finish_reason="stop",
                    prompt_tokens=30,
                    completion_tokens=7,
                )
            ]
        )


def _engine(model, monkeypatch):
    engine = engine_module.SimpleEngine("test-model", force_mllm=True)
    engine._loaded = True
    engine._model = model
    engine._text_model_initialization_attempted = True
    monkeypatch.setattr(engine, "_bind_generation_streams_once", lambda: None)
    return engine


def _drive(engine, route):
    messages = [{"role": "user", "content": "hi"}]

    async def scenario():
        if route == "chat":
            return await engine.chat(messages, max_tokens=16)
        return [c async for c in engine.stream_chat(messages, max_tokens=16)]

    try:
        return asyncio.run(scenario())
    finally:
        if engine._generation_executor is not None:
            engine._generation_executor.shutdown(wait=True)


@pytest.mark.parametrize("route", ["chat", "stream_chat"])
class TestSimpleEngineRoutes:
    def _count(self, engine, route):
        result = _drive(engine, route)
        return result.completion_tokens if route == "chat" else result[-1].completion_tokens

    def test_plain_reply_with_stop_token(self, monkeypatch, route):
        engine = _engine(_Model(_mlx_vlm_chunks(5, flush_count=6)), monkeypatch)

        assert self._count(engine, route) == 6

    def test_stop_token_only_reply(self, monkeypatch, route):
        engine = _engine(_Model(_mlx_vlm_chunks(0, flush_count=1)), monkeypatch)

        assert self._count(engine, route) == 1

    def test_length_capped_reply(self, monkeypatch, route):
        engine = _engine(_Model(_mlx_vlm_chunks(3, flush_count=3)), monkeypatch)

        assert self._count(engine, route) == 3

    def test_native_video_reports_the_models_count_not_one(self, monkeypatch, route):
        engine = _engine(_NativeVideoModel(None), monkeypatch)

        assert self._count(engine, route) == 7


def test_streamed_counts_never_exceed_the_final_count(monkeypatch):
    engine = _engine(_Model(_mlx_vlm_chunks(5, flush_count=6)), monkeypatch)

    counts = [c.completion_tokens for c in _drive(engine, "stream_chat")]

    assert counts == sorted(counts)
    assert counts[-1] == 6
