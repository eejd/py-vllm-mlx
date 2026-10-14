# SPDX-License-Identifier: Apache-2.0
"""A prompt's BOS must reach the batching schedulers exactly as its request kind says.

Chat prompts are rendered by the chat template, which owns every marker (the
BOS is literal text: ``<bos>`` for Gemma, ``<|begin_of_text|>`` for Llama-3).
Tokenizing that string with a HF tokenizer's default ``add_special_tokens=True``
prepends a second BOS, so the text scheduler fed ``[BOS, BOS, ...]`` to the
model, keyed its prefix cache on it, reported ``usage.prompt_tokens`` one too
high and (via the LCP in ``_compute_prefix_boundary``) put the shared-prefix
boundary one past the real one.  Upstream vLLM tokenizes chat prompts with
``add_special_tokens=False`` and raw completion prompts with the tokenizer
default (``vllm/renderers/base.py`` ``default_chat_tok_params`` /
``default_cmpl_tok_params``).

The engines therefore pass the request kind down: ``add_special_tokens=False``
for a chat-rendered prompt, ``None`` (tokenizer default, minus a duplicated
leading BOS) for a raw ``/v1/completions`` prompt.  The MLLM batch generator
used to hand ``mlx_vlm.utils.prepare_inputs`` ``add_special_tokens=False`` for
everything, so a raw completion on Gemma-3 reached the model without any BOS.

The fake tokenizer reproduces the HF contract: ``encode`` prepends the BOS id
when ``add_special_tokens`` is true (the default) and splits the BOS marker
text into its single id either way.
"""

import asyncio
from collections import deque
from types import SimpleNamespace

import mlx.core as mx
import pytest

from vllm_mlx.engine.batched import BatchedEngine
from vllm_mlx.mllm_batch_generator import (
    MLLMBatchGenerator,
    MLLMBatchRequest,
    MLLMBatchStats,
)
from vllm_mlx.mllm_scheduler import MLLMScheduler
from vllm_mlx.prompt_warmup import warm_prefix_cache
from vllm_mlx.request import Request, SamplingParams
from vllm_mlx.scheduler import Scheduler
from vllm_mlx.utils.tokenizer import (
    encode_prompt,
    prompt_adds_special_tokens,
    resolve_add_special_tokens,
)

BOS = "<bos>"
BOS_ID = 2


class FakeHFTokenizer:
    """HF-style tokenizer: words -> ids, BOS text -> BOS_ID, optional BOS prefix."""

    def __init__(self, bos_token=BOS):
        self.bos_token = bos_token
        self._vocab = {}

    def _word(self, word):
        if word == self.bos_token:
            return BOS_ID
        return self._vocab.setdefault(word, 100 + len(self._vocab))

    def encode(self, text, add_special_tokens=True):
        ids = [self._word(w) for w in text.split()]
        if add_special_tokens and self.bos_token is not None:
            ids = [BOS_ID] + ids
        return ids


# A chat prompt whose template renders the BOS (Llama-3 / Gemma style) and one
# whose template does not (ChatML on a Llama-3 tokenizer, e.g. Hermes-3).
CHAT_PROMPT = f"{BOS} <start_of_turn>user hello <end_of_turn> <start_of_turn>model"
CHATML_PROMPT = "<|im_start|>user hello <|im_end|> <|im_start|>assistant"
RAW_PROMPT = "once upon a time"


def _no_special(tok, prompt):
    return tok.encode(prompt, add_special_tokens=False)


class TestBosRule:
    """The rule is mlx-lm's / the Simple engine's, character for character."""

    def test_bos_text_in_the_middle_of_a_prompt_does_not_count(self):
        tok = FakeHFTokenizer()
        assert prompt_adds_special_tokens(tok, f"hello {BOS} world") is True

    def test_leading_whitespace_defeats_the_rule(self):
        tok = FakeHFTokenizer()
        assert prompt_adds_special_tokens(tok, f" {BOS} hello") is True

    def test_empty_bos_token_matches_every_prompt_like_mlx_lm(self):
        tok = FakeHFTokenizer(bos_token="")
        assert prompt_adds_special_tokens(tok, "hello") is False

    def test_non_string_bos_token_is_ignored(self):
        tok = FakeHFTokenizer(bos_token=None)
        tok.bos_token = object()
        assert prompt_adds_special_tokens(tok, "hello") is True

    def test_duck_typed_tokenizer_without_the_keyword_uses_its_defaults(self):
        class PlainTokenizer:
            bos_token = BOS

            def encode(self, text):
                return [len(w) for w in text.split()]

        assert encode_prompt(PlainTokenizer(), CHAT_PROMPT) == [
            len(w) for w in CHAT_PROMPT.split()
        ]

    def test_tokenizer_with_var_keyword_encode_gets_the_flag(self):
        seen = {}

        class KwTokenizer:
            bos_token = BOS

            def encode(self, text, **kwargs):
                seen.update(kwargs)
                return [1]

        encode_prompt(KwTokenizer(), CHAT_PROMPT)
        assert seen == {"add_special_tokens": False}


class TestEncodePrompt:
    def test_rendered_chat_prompt_with_bos_gets_a_single_bos(self):
        tok = FakeHFTokenizer()
        for kind in (False, None):
            ids = encode_prompt(tok, CHAT_PROMPT, kind)
            assert ids == _no_special(tok, CHAT_PROMPT)
            assert ids.count(BOS_ID) == 1

    def test_chat_kind_never_adds_the_tokenizer_bos(self):
        # The template owns the markers even when it renders no BOS at all.
        tok = FakeHFTokenizer()
        assert encode_prompt(tok, CHATML_PROMPT, False) == _no_special(
            tok, CHATML_PROMPT
        )

    def test_raw_completion_prompt_keeps_its_tokenizer_bos(self):
        tok = FakeHFTokenizer()
        ids = encode_prompt(tok, RAW_PROMPT, None)
        assert ids[0] == BOS_ID
        assert ids.count(BOS_ID) == 1

    def test_explicit_true_always_uses_the_tokenizer_default(self):
        tok = FakeHFTokenizer()
        assert resolve_add_special_tokens(tok, CHAT_PROMPT, True) is True

    def test_tokenizer_without_bos_is_untouched(self):
        tok = FakeHFTokenizer(bos_token=None)
        assert resolve_add_special_tokens(tok, "<|im_start|>user hi") is True


def _bare_scheduler(tokenizer):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.tokenizer = tokenizer
    scheduler.requests = {}
    scheduler.waiting = deque()
    scheduler.block_aware_cache = None
    scheduler.memory_aware_cache = None
    scheduler.prefix_cache = None
    return scheduler


def _request(prompt, add_special_tokens=None):
    return Request(
        request_id="r1",
        prompt=prompt,
        sampling_params=SamplingParams(max_tokens=4),
        add_special_tokens=add_special_tokens,
    )


class TestSchedulerAddRequest:
    @pytest.mark.parametrize("kind", [False, None])
    def test_chat_prompt_ids_and_count_match_reference(self, kind):
        tok = FakeHFTokenizer()
        scheduler = _bare_scheduler(tok)
        request = _request(CHAT_PROMPT, kind)
        scheduler.add_request(request)
        reference = _no_special(tok, CHAT_PROMPT)
        assert request.prompt_token_ids == reference
        assert request.num_prompt_tokens == len(reference)
        # The prefix-cache lookup/key is computed from these ids.
        assert request.remaining_tokens == reference

    def test_raw_completion_prompt_still_gets_bos(self):
        tok = FakeHFTokenizer()
        scheduler = _bare_scheduler(tok)
        request = _request(RAW_PROMPT)
        scheduler.add_request(request)
        assert request.prompt_token_ids == [BOS_ID, 100, 101, 102, 103]

    def test_chat_prompt_without_template_bos_gets_none(self):
        tok = FakeHFTokenizer()
        scheduler = _bare_scheduler(tok)
        request = _request(CHATML_PROMPT, add_special_tokens=False)
        scheduler.add_request(request)
        assert request.prompt_token_ids == _no_special(tok, CHATML_PROMPT)
        assert BOS_ID not in request.prompt_token_ids

    def test_processor_wrapped_tokenizer_honours_the_kind(self):
        class Processor:
            def __init__(self, inner):
                self.tokenizer = inner

        tok = FakeHFTokenizer()
        scheduler = _bare_scheduler(Processor(tok))
        chat = _request(CHATML_PROMPT, add_special_tokens=False)
        scheduler.add_request(chat)
        assert chat.prompt_token_ids == _no_special(tok, CHATML_PROMPT)
        raw = _request(RAW_PROMPT)
        raw.request_id = "r2"
        scheduler.add_request(raw)
        assert raw.prompt_token_ids[0] == BOS_ID


def _bare_mllm_scheduler(tokenizer):
    scheduler = MLLMScheduler.__new__(MLLMScheduler)
    scheduler.processor = tokenizer
    scheduler.requests = {}
    scheduler.waiting = deque()
    return scheduler


class TestMLLMSchedulerPromptCount:
    def test_chat_prompt_count_matches_prepare_inputs(self):
        tok = FakeHFTokenizer()
        scheduler = _bare_mllm_scheduler(tok)
        request_id = scheduler.add_request(
            CHAT_PROMPT, max_tokens=4, add_special_tokens=False
        )
        assert scheduler.requests[request_id].num_prompt_tokens == len(
            _no_special(tok, CHAT_PROMPT)
        )

    def test_raw_completion_count_includes_the_bos_the_model_sees(self):
        tok = FakeHFTokenizer()
        scheduler = _bare_mllm_scheduler(tok)
        request_id = scheduler.add_request(RAW_PROMPT, max_tokens=4)
        assert scheduler.requests[request_id].num_prompt_tokens == 5

    def test_processor_wrapped_tokenizer(self):
        class Processor:
            def __init__(self, inner):
                self.tokenizer = inner

        tok = FakeHFTokenizer()
        scheduler = _bare_mllm_scheduler(Processor(tok))
        request_id = scheduler.add_request(
            CHAT_PROMPT, max_tokens=4, add_special_tokens=False
        )
        assert scheduler.requests[request_id].num_prompt_tokens == len(
            _no_special(tok, CHAT_PROMPT)
        )

    def test_kind_reaches_the_batch_request(self, monkeypatch):
        inserted = []
        scheduler = _bare_mllm_scheduler(FakeHFTokenizer())
        scheduler.running = {}
        scheduler.request_id_to_uid = {}
        scheduler.uid_to_request_id = {}
        scheduler.total_prompt_tokens = 0
        scheduler.config = SimpleNamespace(max_num_seqs=4)
        scheduler._ensure_batch_generator = lambda: None
        scheduler.batch_generator = SimpleNamespace(
            insert=lambda reqs: inserted.extend(reqs) or list(range(len(reqs)))
        )
        scheduler.add_request(CHAT_PROMPT, max_tokens=4, add_special_tokens=False)
        scheduler.add_request(RAW_PROMPT, max_tokens=4)
        scheduler._schedule_waiting()
        assert [r.add_special_tokens for r in inserted] == [False, None]


class TestMLLMPreprocessTokenization:
    """The batch generator is the code that tokenizes MLLM prompts for the model."""

    def _generator(self, tokenizer):
        gen = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
        gen.processor = SimpleNamespace(tokenizer=tokenizer)
        gen.model = SimpleNamespace(config=SimpleNamespace(image_token_index=7))
        gen.vision_cache = SimpleNamespace(
            get_pixel_cache=lambda *a, **k: None, set_pixel_cache=lambda **k: None
        )
        gen._stats = MLLMBatchStats()
        return gen

    def _preprocess(self, monkeypatch, prompt, kind):
        seen = {}

        def fake_prepare_inputs(processor, **kwargs):
            seen.update(kwargs)
            return {
                "input_ids": mx.array([[1, 2, 3]]),
                "attention_mask": mx.array([[1, 1, 1]]),
            }

        monkeypatch.setattr("mlx_vlm.utils.prepare_inputs", fake_prepare_inputs)
        gen = self._generator(FakeHFTokenizer())
        request = MLLMBatchRequest(
            uid=1, request_id="r", prompt=prompt, add_special_tokens=kind
        )
        gen._preprocess_request(request)
        return seen["add_special_tokens"]

    def test_chat_prompt_is_tokenized_without_special_tokens(self, monkeypatch):
        assert self._preprocess(monkeypatch, CHAT_PROMPT, False) is False
        assert self._preprocess(monkeypatch, CHATML_PROMPT, False) is False

    def test_raw_completion_prompt_gets_the_tokenizer_bos(self, monkeypatch):
        assert self._preprocess(monkeypatch, RAW_PROMPT, None) is True

    def test_raw_prompt_carrying_its_bos_does_not_get_a_second(self, monkeypatch):
        assert self._preprocess(monkeypatch, CHAT_PROMPT, None) is False


class _RecordingLLMEngine:
    def __init__(self):
        self.calls = []

    async def generate(self, prompt, sampling_params, **kwargs):
        self.calls.append(("generate", prompt, kwargs))
        return SimpleNamespace(
            output_text="x",
            output_token_ids=[1],
            prompt_tokens=1,
            completion_tokens=1,
            finish_reason="stop",
            mtp_drafts=0,
            mtp_accepted=0,
            cached_tokens=None,
        )

    async def add_request(self, prompt, sampling_params, **kwargs):
        self.calls.append(("add_request", prompt, kwargs))
        return "rid"

    async def stream_outputs(self, request_id):
        yield SimpleNamespace(
            output_text="x",
            new_text="x",
            prompt_tokens=1,
            completion_tokens=1,
            finished=True,
            finish_reason="stop",
            mtp_drafts=0,
            mtp_accepted=0,
            cached_tokens=None,
        )


def _engine(tokenizer):
    engine = BatchedEngine.__new__(BatchedEngine)
    engine._loaded = True
    engine._is_mllm = False
    engine._processor = None
    engine._tokenizer = tokenizer
    engine._engine = _RecordingLLMEngine()
    engine._mllm_scheduler = None
    engine._apply_chat_template = lambda messages, *a, **k: CHAT_PROMPT
    return engine


MESSAGES = [{"role": "user", "content": "hello"}]


def _run(coro):
    return asyncio.run(coro)


async def _drain(agen):
    return [item async for item in agen]


class TestBatchedEngineRequestKind:
    def test_chat_passes_the_chat_kind(self):
        engine = _engine(FakeHFTokenizer())
        _run(engine.chat(MESSAGES, max_tokens=2))
        _, _, kwargs = engine._engine.calls[0]
        assert kwargs["add_special_tokens"] is False

    def test_stream_chat_passes_the_chat_kind(self):
        engine = _engine(FakeHFTokenizer())
        _run(_drain(engine.stream_chat(MESSAGES, max_tokens=2)))
        _, _, kwargs = engine._engine.calls[0]
        assert kwargs["add_special_tokens"] is False

    def test_completion_passes_no_kind(self):
        engine = _engine(FakeHFTokenizer())
        _run(engine.generate(RAW_PROMPT, max_tokens=2))
        _run(_drain(engine.stream_generate(RAW_PROMPT, max_tokens=2)))
        assert [c[2]["add_special_tokens"] for c in engine._engine.calls] == [
            None,
            None,
        ]

    def test_mllm_requests_carry_the_kind(self):
        calls = []

        class FakeMLLMScheduler:
            async def add_request_async(self, **kwargs):
                calls.append(kwargs)
                return "rid"

            async def stream_outputs(self, request_id):
                yield SimpleNamespace(
                    output_text="x",
                    new_text="x",
                    prompt_tokens=1,
                    completion_tokens=1,
                    finished=True,
                    finish_reason="stop",
                    mtp_drafts=0,
                    mtp_accepted=0,
                )

        engine = _engine(FakeHFTokenizer())
        engine._is_mllm = True
        engine._mllm_scheduler = FakeMLLMScheduler()
        engine._default_mllm_draft = False
        _run(_drain(engine.stream_chat(MESSAGES, max_tokens=2)))
        _run(_drain(engine.stream_generate(RAW_PROMPT, max_tokens=2)))
        assert [c["add_special_tokens"] for c in calls] == [False, None]


class TestPrefixBoundary:
    @pytest.mark.parametrize("bos_token", [BOS, None])
    @pytest.mark.parametrize("template_bos", [True, False])
    def test_boundary_is_the_shared_prefix_of_the_ids_the_model_sees(
        self, bos_token, template_bos
    ):
        tok = FakeHFTokenizer(bos_token=bos_token)
        engine = _engine(tok)
        lead = [BOS] if template_bos else []
        prefix = lead + ["<sys>", "be", "terse"]
        engine._apply_chat_template = lambda messages, tools=None, **kw: " ".join(
            prefix
            + [f"<{m['role']}> {m['content']}" for m in messages[1:]]
            + ["<model>"]
        )
        messages = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "second question"},
        ]
        boundary = engine._compute_prefix_boundary(messages)

        dummy = list(messages)
        dummy[3] = {**messages[3], "content": "XXXXXXXXXX"}
        # Chat requests are tokenized with add_special_tokens=False.
        real_ids = _no_special(tok, engine._apply_chat_template(messages))
        dummy_ids = _no_special(tok, engine._apply_chat_template(dummy))
        expected = 0
        for a, b in zip(real_ids, dummy_ids):
            if a != b:
                break
            expected += 1
        assert boundary == expected
        assert real_ids[:boundary] == dummy_ids[:boundary]
        assert real_ids[boundary] != dummy_ids[boundary]


class TestWarmupTokenizesLikeChat:
    def test_strict_prefix_request_uses_the_chat_kind(self):
        calls = []

        class Tok:
            def apply_chat_template(self, messages, **kwargs):
                user = messages[-1]["content"]
                return f"{BOS} system-prompt-with-enough-prefix {user}"

        class Engine:
            tokenizer = Tok()

            async def stream_generate(self, **kwargs):
                calls.append(kwargs)
                yield SimpleNamespace(finished=True, prompt_tokens=3)

        messages = [{"role": "system", "content": "be terse"}]
        _run(warm_prefix_cache(Engine(), [messages]))
        assert calls and calls[0]["add_special_tokens"] is False


def _bare_engine_core():
    """The real ``EngineCore.add_request`` with a scheduler that records the request."""
    from vllm_mlx.engine_core import EngineCore

    core = EngineCore.__new__(EngineCore)
    core.config = SimpleNamespace(stream_interval=1)
    core._output_collectors = {}
    core._stream_states = {}
    core._finished_events = {}
    core.scheduler = SimpleNamespace(add_request=lambda request: seen.append(request))
    seen = []
    core.seen = seen
    return core


class TestEngineCoreHop:
    """The kind must survive ``EngineCore`` / ``AsyncEngineCore`` into the ``Request``."""

    @pytest.mark.parametrize("kind", [False, None])
    def test_engine_core_puts_the_kind_on_the_request(self, kind):
        core = _bare_engine_core()
        _run(core.add_request(CHAT_PROMPT, add_special_tokens=kind))
        (request,) = core.seen
        assert request.add_special_tokens is kind

    @pytest.mark.parametrize("kind", [False, None])
    def test_the_non_streaming_generate_path_carries_the_kind(self, kind):
        from vllm_mlx.engine_core import AsyncEngineCore
        from vllm_mlx.request import RequestOutput

        core = _bare_engine_core()
        core._cleanup_request = lambda request_id: None

        def finish_at_once(request):
            core.seen.append(request)
            core._output_collectors[request.request_id].put(
                RequestOutput(
                    request_id=request.request_id, output_text="x", finished=True
                )
            )
            core._finished_events[request.request_id].set()

        core.scheduler = SimpleNamespace(add_request=finish_at_once)
        wrapper = AsyncEngineCore.__new__(AsyncEngineCore)
        wrapper.engine = core
        _run(wrapper.generate(CHAT_PROMPT, add_special_tokens=kind))
        (request,) = core.seen
        assert request.add_special_tokens is kind

    def test_async_engine_core_forwards_the_kind(self):
        from vllm_mlx.engine_core import AsyncEngineCore

        core = _bare_engine_core()
        wrapper = AsyncEngineCore.__new__(AsyncEngineCore)
        wrapper.engine = core
        _run(wrapper.add_request(CHAT_PROMPT, add_special_tokens=False))
        (request,) = core.seen
        assert request.add_special_tokens is False


class _MLLMSchedulerStub:
    def __init__(self):
        self.calls = []

    async def generate(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            output_text="x",
            output_token_ids=[1],
            prompt_tokens=1,
            completion_tokens=1,
            finish_reason="stop",
            mtp_drafts=0,
            mtp_accepted=0,
        )


class TestBatchedMultimodalNonStreaming:
    """``chat`` / ``generate`` on a multimodal model carry the kind to the scheduler."""

    def _engine(self):
        engine = _engine(FakeHFTokenizer())
        engine._is_mllm = True
        engine._mllm_scheduler = _MLLMSchedulerStub()
        engine._default_mllm_draft = False
        return engine

    def test_raw_generate_carries_no_kind(self):
        engine = self._engine()
        _run(engine.generate(RAW_PROMPT, max_tokens=2))
        (call,) = engine._mllm_scheduler.calls
        assert call["add_special_tokens"] is None

    def test_chat_carries_the_chat_kind(self):
        engine = self._engine()
        _run(engine.chat(MESSAGES, max_tokens=2))
        (call,) = engine._mllm_scheduler.calls
        assert call["add_special_tokens"] is False


class TestMLLMSchedulerCountForTemplatesWithoutBos:
    def test_chat_count_ignores_the_tokenizer_bos(self):
        # A ChatML-style template renders no BOS on a BOS-adding tokenizer: the
        # model is fed no BOS, so the count must not include one.
        tok = FakeHFTokenizer()
        scheduler = _bare_mllm_scheduler(tok)
        request_id = scheduler.add_request(
            CHATML_PROMPT, max_tokens=4, add_special_tokens=False
        )
        assert scheduler.requests[request_id].num_prompt_tokens == len(
            _no_special(tok, CHATML_PROMPT)
        )

    def test_raw_count_includes_the_tokenizer_bos(self):
        tok = FakeHFTokenizer()
        scheduler = _bare_mllm_scheduler(tok)
        request_id = scheduler.add_request(RAW_PROMPT, max_tokens=4)
        assert scheduler.requests[request_id].num_prompt_tokens == (
            len(_no_special(tok, RAW_PROMPT)) + 1
        )


class TestSimpleEngineSwallowsTheKind:
    """``prompt_warmup`` passes the kind to every engine; Simple must not forward it."""

    def test_the_hint_does_not_leak_into_the_model_call(self):
        from vllm_mlx.engine.simple import SimpleEngine

        seen = []

        def stream_generate(**kwargs):
            seen.append(kwargs)
            return iter(
                [SimpleNamespace(text="x", finished=True, finish_reason="stop")]
            )

        engine = SimpleEngine("test-model")
        engine._loaded = True
        engine._model = SimpleNamespace(
            tokenizer=FakeHFTokenizer(), stream_generate=stream_generate
        )

        async def run():
            return [
                o
                async for o in engine.stream_generate(
                    CHAT_PROMPT, max_tokens=2, add_special_tokens=False
                )
            ]

        outputs = _run(run())
        assert outputs
        (kwargs,) = seen
        assert "add_special_tokens" not in kwargs
