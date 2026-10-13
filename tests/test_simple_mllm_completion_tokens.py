# SPDX-License-Identifier: Apache-2.0
"""``usage.completion_tokens`` and ``finish_reason`` of the Simple engine's MLLM routes.

vLLM reports ``len(token_ids)``: every generated token, the one that ended the
request (EOS / stop token) included, nothing else. mlx-vlm's ``stream_generate``
yields one chunk per decoded token and then a last chunk that only flushes the
detokenizer and carries ``generation_tokens`` (the model's own count, stop token
included). The Simple engine used to count chunks, twice: the flush chunk, and
again the terminal chunk ``MLXMultimodalLM.stream_chat`` appends. A reply of 5
tokens plus ``<end_of_turn>`` was billed 7 (Gemma-3 tool round trip: 2 for a reply
that is the stop token alone), and a reply cut by ``max_tokens`` one over the cap.

``finish_reason`` had the same family of fault: the MLLM routes hardcoded ``stop``,
so a reply cut at ``max_tokens`` looked complete (``/v1/completions`` and vLLM say
``length``). mlx-vlm's final chunk / ``GenerationResult`` carries the reason.

The fakes below reproduce mlx-vlm's chunk sequence (``generate/dispatch.py``
``stream_generate``) and run it through the real ``_stream_mllm_generated_outputs``
and the real ``SimpleEngine`` routes.
"""

import asyncio
import sys
from types import ModuleType, SimpleNamespace

import pytest

from vllm_mlx.engine import simple as engine_module
from vllm_mlx.models.mllm import (
    MLLMOutput,
    MLXMultimodalLM,
    _stream_mllm_generated_outputs,
)


def _mlx_vlm_chunks(n_content, *, flush_count, finish="stop"):
    """``n_content`` per-token chunks, then the detokenizer-flush chunk."""
    chunks = [
        SimpleNamespace(text=f"t{i} ", prompt_tokens=9, generation_tokens=i + 1)
        for i in range(n_content)
    ]
    chunks.append(
        SimpleNamespace(
            text="",
            prompt_tokens=9,
            generation_tokens=flush_count,
            finish_reason=finish,
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
        return (
            result.completion_tokens
            if route == "chat"
            else result[-1].completion_tokens
        )

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


class TestFinishReason:
    """``stop`` for a natural end, ``length`` for the cap, on every MLLM route."""

    def test_wrapper_natural_stop(self):
        assert _wrapped(_mlx_vlm_chunks(5, flush_count=6))[-1].finish_reason == "stop"

    def test_wrapper_length_cap(self):
        chunks = _mlx_vlm_chunks(3, flush_count=3, finish="length")

        outputs = _wrapped(chunks)

        assert outputs[-1].finish_reason == "length"
        # Only the terminal output finishes the request.
        assert all(o.finish_reason is None for o in outputs[:-1])

    def test_wrapper_stop_sequence_is_a_stop_even_if_the_cap_was_reached(self):
        # The consumer leaves at the stop sequence, before the flush chunk; if
        # the cap was reached by that very chunk the reply still ended on the
        # client's stop string.
        chunks = _mlx_vlm_chunks(3, flush_count=3, finish="length")
        chunks[-2].finish_reason = "length"

        outputs = _wrapped(chunks, stop=["t2"])

        assert outputs[-1].finish_reason == "stop"

    def test_wrapper_stop_sequence_cut(self):
        outputs = _wrapped(_mlx_vlm_chunks(5, flush_count=6), stop=["t1"])

        assert outputs[-1].finish_reason == "stop"
        assert outputs[-1].completion_tokens == 2

    def test_wrapper_without_chunks_is_a_stop(self):
        assert _wrapped([])[-1].finish_reason == "stop"

    def test_a_reason_mlx_vlm_does_not_define_is_a_stop(self):
        chunks = _mlx_vlm_chunks(2, flush_count=3, finish="abort")

        assert _wrapped(chunks)[-1].finish_reason == "stop"


@pytest.mark.parametrize("route", ["chat", "stream_chat"])
@pytest.mark.parametrize(
    "chunks, expected",
    [
        (_mlx_vlm_chunks(5, flush_count=6), "stop"),
        (_mlx_vlm_chunks(0, flush_count=1), "stop"),
        (_mlx_vlm_chunks(3, flush_count=3, finish="length"), "length"),
    ],
    ids=["natural", "stop-token-only", "cap"],
)
def test_simple_engine_reports_the_models_finish_reason(
    monkeypatch, route, chunks, expected
):
    engine = _engine(_Model(chunks), monkeypatch)

    result = _drive(engine, route)
    final = result if route == "chat" else result[-1]

    assert final.finish_reason == expected


@pytest.fixture
def wrapper(monkeypatch):
    """The real ``MLXMultimodalLM`` with only mlx-vlm inference replaced."""
    result = {"finish_reason": "stop"}

    def generate(*args, **kwargs):
        return SimpleNamespace(
            text="reply",
            prompt_tokens=3,
            generation_tokens=3,
            finish_reason=result["finish_reason"],
        )

    vlm = ModuleType("mlx_vlm")
    vlm.generate = generate
    vlm.stream_generate = lambda *a, **k: iter(
        _mlx_vlm_chunks(2, flush_count=3, finish=result["finish_reason"])
    )
    prompt_utils = ModuleType("mlx_vlm.prompt_utils")
    prompt_utils.apply_chat_template = lambda *a, **k: "prompt"
    prompt_utils.get_chat_template = lambda *a, **k: "prompt"
    cache = ModuleType("mlx_vlm.models.cache")
    cache.make_prompt_cache = lambda *a, **k: []
    models = ModuleType("mlx_vlm.models")
    models.cache = cache
    video = ModuleType("mlx_vlm.video_generate")
    video.generate = generate
    # Where the wrapper tokenizes the prompt itself (#94) mlx-vlm's ``prepare_inputs``
    # runs before ``generate``; without a stub the real one loads the fake image.
    utils = ModuleType("mlx_vlm.utils")
    utils.prepare_inputs = lambda *a, **k: {
        "input_ids": [[1, 2, 3]],
        "attention_mask": [[1, 1, 1]],
    }
    for module in (vlm, prompt_utils, models, cache, video, utils):
        monkeypatch.setitem(sys.modules, module.__name__, module)

    model = MLXMultimodalLM("test-model", enable_cache=False)
    model._loaded = True
    model.model = SimpleNamespace(language_model=object())
    model.processor = SimpleNamespace(encode=lambda text: [1, 2, 3])
    model.config = {}
    monkeypatch.setattr(model, "_prepare_images", lambda images: ["prepared-image"])
    monkeypatch.setattr(
        model, "_prepare_native_video_inputs", lambda *a, **k: ("prompt", {})
    )
    model._finish = result
    return model


_IMAGE_CHAT = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this image"},
            {"type": "image_url", "image_url": {"url": "test-image"}},
        ],
    }
]


@pytest.mark.parametrize("finish", ["stop", "length"])
@pytest.mark.parametrize("route", ["generate", "chat", "native_video", "stream_chat"])
def test_wrapper_routes_report_the_finish_reason_of_mlx_vlm(wrapper, route, finish):
    wrapper._finish["finish_reason"] = finish
    if route == "generate":
        result = wrapper.generate(prompt="p", images=["test-image"])
    elif route == "chat":
        result = wrapper.chat(messages=_IMAGE_CHAT)
    elif route == "native_video":
        result = wrapper._generate_native_video(messages=_IMAGE_CHAT)
    else:
        result = list(wrapper.stream_chat(messages=_IMAGE_CHAT))[-1]

    assert result.finish_reason == finish


class _RawModel(_Model):
    """``model.stream_generate`` (raw ``/v1/completions``) yielding mlx-vlm chunks."""

    def __init__(self, chunks):
        super().__init__(chunks)
        self.calls = []

    def stream_generate(self, **kwargs):
        self.calls.append(kwargs)
        return iter(self._chunks)


def _drive_raw(engine, max_tokens):
    async def scenario():
        return [c async for c in engine.stream_generate("p", max_tokens=max_tokens)]

    try:
        return asyncio.run(scenario())
    finally:
        if engine._generation_executor is not None:
            engine._generation_executor.shutdown(wait=True)


def _flush_chunks(n_content, *, flush_text, flush_count, finish):
    """mlx-vlm's sequence where the detokenizer still holds text after the last token."""
    chunks = _mlx_vlm_chunks(n_content, flush_count=flush_count, finish=finish)
    chunks[-1].text = flush_text
    return chunks


class TestRawCompletionsStream:
    """``SimpleEngine.stream_generate`` on an MLLM: the flush chunk is part of the reply.

    At ``max_tokens`` the engine stopped on the last token chunk, one chunk before
    mlx-vlm's flush chunk, so the text still buffered in the detokenizer (the last
    word piece) never reached the client: Gemma-3 cap 3 answered ``, in`` where
    the batched engine answered ``, in a``.
    """

    def test_text_held_by_the_detokenizer_at_the_cap_is_delivered(self, monkeypatch):
        chunks = _flush_chunks(3, flush_text="LAST", flush_count=3, finish="length")
        engine = _engine(_RawModel(chunks), monkeypatch)

        outputs = _drive_raw(engine, max_tokens=3)

        assert outputs[-1].text == "t0 t1 t2 LAST"
        assert outputs[-1].new_text == "LAST"

    def test_the_cap_is_reported_as_length_with_exactly_the_cap_tokens(
        self, monkeypatch
    ):
        chunks = _flush_chunks(3, flush_text="LAST", flush_count=3, finish="length")
        engine = _engine(_RawModel(chunks), monkeypatch)

        outputs = _drive_raw(engine, max_tokens=3)

        assert outputs[-1].finish_reason == "length"
        assert outputs[-1].completion_tokens == 3
        # Only the flush chunk ends the request.
        assert [o.finished for o in outputs] == [False, False, False, True]

    def test_natural_stop_counts_the_stop_token_and_keeps_the_flush_text(
        self, monkeypatch
    ):
        chunks = _flush_chunks(5, flush_text="END", flush_count=6, finish="stop")
        engine = _engine(_RawModel(chunks), monkeypatch)

        outputs = _drive_raw(engine, max_tokens=16)

        assert outputs[-1].text.endswith("t4 END")
        assert outputs[-1].finish_reason == "stop"
        assert outputs[-1].completion_tokens == 6

    def test_a_stop_token_only_reply(self, monkeypatch):
        chunks = _flush_chunks(0, flush_text="", flush_count=1, finish="stop")
        engine = _engine(_RawModel(chunks), monkeypatch)

        (final,) = _drive_raw(engine, max_tokens=16)

        assert (final.text, final.completion_tokens, final.finish_reason) == (
            "",
            1,
            "stop",
        )

    def test_a_stream_without_counts_still_stops_at_the_cap(self, monkeypatch):
        # A custom ``stream_generate`` (no ``generation_tokens``) is counted per chunk.
        chunks = [SimpleNamespace(text=f"w{i}") for i in range(10)]
        engine = _engine(_RawModel(chunks), monkeypatch)

        outputs = _drive_raw(engine, max_tokens=3)

        assert outputs[-1].completion_tokens == 3
        assert outputs[-1].finish_reason == "length"
        assert outputs[-1].text == "w0w1w2"

    def test_a_pure_llm_engine_keeps_its_own_cap_rule(self, monkeypatch):
        # Only the MLLM route reads mlx-vlm's flush chunk; an LLM stream whose chunks
        # happen to carry a count still ends at ``max_tokens`` on the token chunk.
        chunks = _flush_chunks(5, flush_text="LAST", flush_count=5, finish="length")
        for i, chunk in enumerate(chunks[:-1]):
            chunk.finish_reason = None
        engine = _engine(_RawModel(chunks), monkeypatch)
        engine._is_mllm = False

        outputs = _drive_raw(engine, max_tokens=3)

        assert outputs[-1].text == "t0 t1 t2 "
        assert outputs[-1].completion_tokens == 3
        assert outputs[-1].finish_reason == "length"
