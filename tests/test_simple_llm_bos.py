# SPDX-License-Identifier: Apache-2.0
"""The Simple engine's mlx-lm routes tokenize by request kind, like the batched one.

A chat prompt the template rendered owns its markers: vLLM tokenizes it with
``add_special_tokens=False`` (``apply_chat_template(tokenize=True)`` gives the same
ids).  The Simple engine instead handed mlx-lm the *string* and let mlx-lm decide:
its rule keeps the tokenizer's BOS whenever the prompt does not start with the BOS
text, so Hermes-3-Llama-3.2 (ChatML template, no BOS, on a tokenizer that adds one)
was fed ``[BOS, <|im_start|>, ...]`` -- 21 ids -- by the streaming and the
non-streaming route alike, while the non-streaming route *reported* the reference's
20 and the batched engine (#93) both fed and reported 20.

Now ``MLXLanguageModel`` encodes the prompt itself (``encode_prompt``) and passes the
ids to mlx-lm, so the count reported is the count fed, on every Simple route:
``chat``/``stream_chat`` (cached, prefix-trie cached and uncached), the MLLM engine's
text route, and the raw ``generate``/``stream_generate`` routes, which keep the
tokenizer's BOS (never doubled) unless the caller says the prompt is rendered
(``add_special_tokens=False``, what prompt warm-up sends).
"""

from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import pytest

from vllm_mlx.engine.simple import SimpleEngine
from vllm_mlx.models.llm import MLXLanguageModel

pytestmark = pytest.mark.anyio

BOS_TEXT = "<|begin_of_text|>"
BOS_ID = 1000
RAW = "Once upon a time"
MESSAGES = [{"role": "user", "content": "say hi"}]


class FakeTokenizer:
    """A tokenizer that adds a BOS to every encode unless told not to (Llama-3).

    ``renders_bos``: whether the chat template opens the prompt with the BOS text
    (Llama-3) or not (Hermes-3's ChatML).  The BOS text is one id wherever it
    appears, so a prompt that carries it *and* is encoded with the tokenizer's
    default gets two -- the defect.
    """

    bos_token = BOS_TEXT
    eos_token_ids = {2}
    eos_token_id = 2

    def __init__(self, renders_bos: bool):
        self.renders_bos = renders_bos

    def apply_chat_template(
        self, messages, tokenize=True, add_generation_prompt=False, **_kwargs
    ):
        text = BOS_TEXT if self.renders_bos else ""
        for message in messages:
            text += f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
        return self.encode(text, add_special_tokens=False) if tokenize else text

    def encode(self, text, add_special_tokens=True):
        ids = [BOS_ID] if add_special_tokens else []
        if text.startswith(BOS_TEXT):
            ids.append(BOS_ID)
            text = text[len(BOS_TEXT) :]
        return ids + [ord(ch) for ch in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids if i < 256)


TEMPLATES = pytest.mark.parametrize(
    "renders_bos", [False, True], ids=["chatml-no-bos", "llama-bos"]
)


def reference_ids(tokenizer):
    return tokenizer.apply_chat_template(
        MESSAGES, tokenize=True, add_generation_prompt=True
    )


class Recorder:
    """Stands in for ``mlx_lm.stream_generate`` / ``generate``."""

    def __init__(self):
        self.prompts = []

    @staticmethod
    def _ids(prompt):
        assert not isinstance(prompt, str), "mlx-lm must be handed ids, not a string"
        return prompt.tolist() if hasattr(prompt, "tolist") else list(prompt)

    def patched(self):
        """Both routes: ``MLXLanguageModel.generate`` drives whichever mlx-lm
        entry point the checkout uses (``generate``, or ``stream_generate`` once
        it counts the ids itself), so a stub must cover the pair."""
        return patch.multiple(
            "mlx_lm", generate=self.generate, stream_generate=self.stream_generate
        )

    def stream_generate(self, _model, _tokenizer, *, prompt, **_kwargs):
        self.prompts.append(self._ids(prompt))
        yield SimpleNamespace(text="hi.", token=ord("h"), finish_reason="stop")

    def generate(self, _model, _tokenizer, *, prompt, **_kwargs):
        self.prompts.append(self._ids(prompt))
        return "hi."


class FakeModel:
    def __call__(self, *_args, **_kwargs):
        return mx.zeros((1, 1, 4), dtype=mx.float32)


class FakeCache:
    def __init__(self):
        self.state = (mx.array([[1]], dtype=mx.float32),) * 2
        self.nbytes = 8

    def is_trimmable(self):
        return False


def _engine(renders_bos, **kwargs):
    tokenizer = FakeTokenizer(renders_bos)
    model = MLXLanguageModel("test-model")
    model.model, model.tokenizer, model._loaded = FakeModel(), tokenizer, True
    engine = SimpleEngine("test-model", **kwargs)
    engine._loaded = True
    engine._supports_system_kv_cache = True
    engine._model = model
    return engine, tokenizer


async def _last(stream):
    return [chunk async for chunk in stream][-1]


@TEMPLATES
async def test_stream_chat_feeds_and_counts_the_reference_ids(renders_bos):
    engine, tokenizer = _engine(renders_bos)
    recorder = Recorder()
    with patch("mlx_lm.stream_generate", recorder.stream_generate):
        last = await _last(engine.stream_chat(MESSAGES, max_tokens=4, temperature=0.0))
    assert recorder.prompts == [reference_ids(tokenizer)]
    assert last.prompt_tokens == len(reference_ids(tokenizer))


@TEMPLATES
async def test_chat_feeds_and_counts_the_reference_ids(renders_bos):
    engine, tokenizer = _engine(renders_bos)
    recorder = Recorder()
    with recorder.patched():
        output = await engine.chat(MESSAGES, max_tokens=4, temperature=0.0)
    assert recorder.prompts == [reference_ids(tokenizer)]
    assert output.prompt_tokens == len(reference_ids(tokenizer))


@TEMPLATES
async def test_stream_and_non_stream_chat_agree(renders_bos):
    engine, _ = _engine(renders_bos)
    streaming, non_streaming = Recorder(), Recorder()
    with patch("mlx_lm.stream_generate", streaming.stream_generate):
        streamed = await _last(engine.stream_chat(MESSAGES, max_tokens=4))
    with non_streaming.patched():
        plain = await engine.chat(MESSAGES, max_tokens=4)
    assert streaming.prompts == non_streaming.prompts
    assert streamed.prompt_tokens == plain.prompt_tokens


@TEMPLATES
async def test_prefix_trie_cached_chat_feeds_and_counts_the_reference_ids(renders_bos):
    engine, tokenizer = _engine(renders_bos, prefix_trie_cache=True)
    recorder = Recorder()
    with (
        patch("mlx_lm.models.cache.make_prompt_cache", return_value=[FakeCache()]),
        patch("mlx_lm.stream_generate", recorder.stream_generate),
    ):
        last = await _last(engine.stream_chat(MESSAGES, max_tokens=4, temperature=0.0))
    assert recorder.prompts == [reference_ids(tokenizer)]
    assert last.prompt_tokens == len(reference_ids(tokenizer))


@TEMPLATES
async def test_cache_failure_falls_back_to_uncached_route_with_the_same_ids(
    renders_bos,
):
    # The cache branch dies before its first token: the request is re-run through
    # the public stream_generate, which must tokenize the rendered prompt as a chat.
    engine, tokenizer = _engine(renders_bos, prefix_trie_cache=True)
    recorder = Recorder()
    with (
        patch("mlx_lm.models.cache.make_prompt_cache", side_effect=RuntimeError("x")),
        patch("mlx_lm.stream_generate", recorder.stream_generate),
    ):
        last = await _last(engine.stream_chat(MESSAGES, max_tokens=4, temperature=0.0))
    assert recorder.prompts == [reference_ids(tokenizer)]
    assert last.prompt_tokens == len(reference_ids(tokenizer))


@TEMPLATES
async def test_system_kv_cached_chat_feeds_the_reference_ids(renders_bos):
    engine, tokenizer = _engine(renders_bos)
    engine._system_kv_cache_enabled = True
    messages = [{"role": "system", "content": "Be brief"}, *MESSAGES]
    reference = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    recorder = Recorder()
    with (
        patch("mlx_lm.models.cache.make_prompt_cache", return_value=[FakeCache()]),
        patch("mlx_lm.stream_generate", recorder.stream_generate),
    ):
        last = await _last(engine.stream_chat(messages, max_tokens=4, temperature=0.0))
    # the snapshot covers the system prefix; the rest is what mlx-lm is handed
    assert last.prompt_tokens == len(reference)
    (suffix,) = recorder.prompts
    assert reference[-len(suffix) :] == suffix


async def test_raw_completion_keeps_the_tokenizer_bos():
    engine, _ = _engine(renders_bos=False)
    expected = [BOS_ID, *[ord(c) for c in RAW]]
    recorder = Recorder()
    with patch("mlx_lm.stream_generate", recorder.stream_generate):
        streamed = await _last(engine.stream_generate(RAW, max_tokens=4))
        plain = await engine.generate(RAW, max_tokens=4)
    assert recorder.prompts == [expected, expected]
    assert streamed.prompt_tokens == plain.prompt_tokens == len(expected)


async def test_raw_completion_that_opens_with_the_bos_is_not_doubled():
    engine, _ = _engine(renders_bos=False)
    recorder = Recorder()
    with patch("mlx_lm.stream_generate", recorder.stream_generate):
        last = await _last(engine.stream_generate(BOS_TEXT + RAW, max_tokens=4))
    assert recorder.prompts == [[BOS_ID, *[ord(c) for c in RAW]]]
    assert last.prompt_tokens == len(RAW) + 1


@pytest.mark.parametrize("route", ["stream_generate", "generate"])
async def test_rendered_prompt_flagged_by_the_caller_gets_no_bos(route):
    # What prompt warm-up sends: a rendered chat prefix, tokenized like a chat.
    engine, _ = _engine(renders_bos=False)
    recorder = Recorder()
    rendered = "<|im_start|>system\nS<|im_end|>\n"
    with patch("mlx_lm.stream_generate", recorder.stream_generate):
        result = getattr(engine, route)(
            rendered, max_tokens=4, add_special_tokens=False
        )
        last = await (_last(result) if route == "stream_generate" else result)
    assert recorder.prompts == [[ord(c) for c in rendered]]
    assert last.prompt_tokens == len(rendered)


async def test_specprefill_tokens_follow_the_request_kind():
    engine, _ = _engine(renders_bos=False)
    engine._draft_model = object()
    engine._specprefill_threshold = 0
    seen = []

    async def fake_specprefill(prompt, tokens, *args, **kwargs):
        seen.append((list(tokens), kwargs.get("add_special_tokens")))
        yield SimpleNamespace(finished=True)

    engine._stream_generate_specprefill = fake_specprefill
    rendered = "<|im_start|>user\nhello<|im_end|>\n"
    await _last(engine.stream_generate(RAW, max_tokens=4))
    await _last(
        engine.stream_generate(rendered, max_tokens=4, add_special_tokens=False)
    )
    assert seen == [
        ([BOS_ID, *[ord(c) for c in RAW]], None),
        ([ord(c) for c in rendered], False),
    ]


@TEMPLATES
async def test_mllm_engine_text_route_feeds_and_counts_the_reference_ids(renders_bos):
    tokenizer = FakeTokenizer(renders_bos)
    engine = SimpleEngine("test-model", force_mllm=True)
    engine._loaded = True
    engine._text_model = FakeModel()
    engine._text_tokenizer = tokenizer
    recorder = Recorder()
    with (
        patch("vllm_mlx.engine.simple._bind_worker_generation_streams"),
        patch("mlx_lm.stream_generate", recorder.stream_generate),
    ):
        outputs = [
            chunk
            async for chunk in engine._stream_generate_text(
                MESSAGES, max_tokens=4, temperature=0.0, top_p=1.0
            )
        ]
    assert recorder.prompts == [reference_ids(tokenizer)]
    assert outputs[-1].prompt_tokens == len(reference_ids(tokenizer))


SYSTEM = {"role": "system", "content": "Be brief"}


def _mllm_text_engine(renders_bos):
    tokenizer = FakeTokenizer(renders_bos)
    engine = SimpleEngine("test-model", force_mllm=True)
    engine._loaded = True
    engine._text_model = FakeModel()
    engine._text_tokenizer = tokenizer
    engine._supports_system_kv_cache = True
    engine._system_kv_cache_enabled = True
    return engine, tokenizer


async def _text_route(engine, messages):
    return [
        chunk
        async for chunk in engine._stream_generate_text(
            messages, max_tokens=4, temperature=0.0, top_p=1.0
        )
    ]


def _system_prefix_len(tokenizer, renders_bos):
    prefix = (
        BOS_TEXT if renders_bos else ""
    ) + "<|im_start|>system\nBe brief<|im_end|>\n"
    return len(tokenizer.encode(prefix, add_special_tokens=False))


@TEMPLATES
async def test_mllm_text_route_system_kv_cache_keeps_working_on_a_bos_adding_tokenizer(
    renders_bos,
):
    # The system prefix is tokenized as a rendered chat prompt, like the full
    # prompt: with the tokenizer's BOS added to the prefix only, the prefix is no
    # longer a token prefix of the prompt and the cache silently switches itself
    # off (no snapshot, no hit, the whole prompt prefilled every time).
    engine, tokenizer = _mllm_text_engine(renders_bos)
    messages = [SYSTEM, *MESSAGES]
    reference = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    suffix = reference[_system_prefix_len(tokenizer, renders_bos) :]
    recorder = Recorder()
    with (
        patch("vllm_mlx.engine.simple._bind_worker_generation_streams"),
        patch("mlx_lm.models.cache.make_prompt_cache", return_value=[FakeCache()]),
        patch("mlx_lm.stream_generate", recorder.stream_generate),
    ):
        miss = await _text_route(engine, messages)
        hit = await _text_route(engine, messages)
    assert engine._system_kv_cache_stats["stores"] == 1
    assert engine._system_kv_cache_stats["hits"] == 1
    # mlx-lm is handed only what the snapshot does not cover, both times ...
    assert recorder.prompts == [suffix, suffix]
    # ... and the request is still billed for the whole prompt.
    assert miss[-1].prompt_tokens == hit[-1].prompt_tokens == len(reference)


@pytest.mark.parametrize("with_system", [False, True], ids=["no-system", "system-kv"])
@TEMPLATES
async def test_mllm_text_route_specprefill_failure_resends_the_counted_ids(
    renders_bos, with_system
):
    # SpecPrefill dies after the system snapshot was taken: the retry is a plain
    # decode of the whole prompt on a fresh cache, fed the ids that were counted
    # (a string would be tokenized again by mlx-lm's own BOS rule).
    engine, tokenizer = _mllm_text_engine(renders_bos)
    engine._draft_model = object()
    engine._specprefill_threshold = 0
    messages = [SYSTEM, *MESSAGES] if with_system else MESSAGES
    reference = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    recorder = Recorder()
    with (
        patch("vllm_mlx.engine.simple._bind_worker_generation_streams"),
        patch("mlx_lm.models.cache.make_prompt_cache", return_value=[FakeCache()]),
        patch("vllm_mlx.specprefill.score_tokens", side_effect=RuntimeError("boom")),
        patch("vllm_mlx.specprefill.cleanup_rope"),
        patch("mlx_lm.stream_generate", recorder.stream_generate),
    ):
        outputs = await _text_route(engine, messages)
    assert recorder.prompts == [reference]
    assert outputs[-1].prompt_tokens == len(reference)


async def test_stream_that_ends_unfinished_counts_the_prompt_by_request_kind():
    # A stream that never reports a prompt size and stops before its last chunk
    # is closed with a final chunk whose count is re-derived from the prompt:
    # by the same kind it was generated with.
    engine, _ = _engine(renders_bos=False)

    def silent_stream(**_kwargs):
        yield SimpleNamespace(text="a", token=ord("a"), finished=False)

    engine._model.stream_generate = silent_stream
    rendered = "<|im_start|>user\nhello<|im_end|>\n"
    chat = await _last(
        engine.stream_generate(rendered, max_tokens=4, add_special_tokens=False)
    )
    raw = await _last(engine.stream_generate(RAW, max_tokens=4))
    assert chat.prompt_tokens == len(rendered)
    assert raw.prompt_tokens == len(RAW) + 1


class TestModelWrapper:
    """``MLXLanguageModel`` on its own: the kind arrives as ``add_special_tokens``."""

    def _model(self, renders_bos=False):
        model = MLXLanguageModel("test-model")
        model.model, model._loaded = FakeModel(), True
        model.tokenizer = FakeTokenizer(renders_bos)
        return model

    def test_token_ids_are_passed_through_untouched(self):
        model, recorder = self._model(), Recorder()
        with patch("mlx_lm.stream_generate", recorder.stream_generate):
            chunks = list(model.stream_generate([5, 6, 7], max_tokens=2))
        assert recorder.prompts == [[5, 6, 7]]
        assert chunks[0].prompt_tokens == 3

    def test_chat_without_a_template_follows_the_raw_prompt_rule(self):
        model, recorder = self._model(), Recorder()
        model.tokenizer = SimpleNamespace(
            bos_token=BOS_TEXT, encode=FakeTokenizer(False).encode
        )
        with recorder.patched():
            model.chat(MESSAGES, max_tokens=2)
        assert recorder.prompts[0][0] == BOS_ID
