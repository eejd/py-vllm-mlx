# SPDX-License-Identifier: Apache-2.0
"""SimpleEngine tokenizes a rendered chat prompt with the BOS rule, at every site.

The chat template renders the BOS as text (``<bos>``); a HF tokenizer's default
``add_special_tokens=True`` then prepends a second one. ``SimpleEngine`` encodes the
rendered prompt in several places (the mlx-lm text route, the LLM ``stream_chat``
prefix caches, the SpecPrefill threshold, the ``prompt_tokens`` fallback of raw
completions); each must give the ids the model sees: one BOS.
"""

import hashlib
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import pytest

from vllm_mlx.engine.simple import SimpleEngine

pytestmark = pytest.mark.anyio

BOS = "<bos>"
BOS_ID = 2


class BosTokenizer:
    """HF contract: ``encode`` prepends the BOS id unless told not to; the BOS text is one id."""

    bos_token = BOS
    eos_token_ids = []

    def _ids(self, text):
        return [BOS_ID if word == BOS else 100 + ord(word[0]) for word in text.split()]

    def apply_chat_template(self, messages, **_kwargs):
        body = " ".join(f"<|{m['role']}|> {m.get('content', '')}" for m in messages)
        return f"{BOS} {body} <|assistant|>"

    def encode(self, text, add_special_tokens=True):
        return ([BOS_ID] if add_special_tokens else []) + self._ids(text)


def _reference_ids(prompt):
    """What the model must see: the rendered prompt, no tokenizer BOS on top."""
    return BosTokenizer().encode(prompt, add_special_tokens=False)


class FakeCache:
    def __init__(self):
        self.state = (
            mx.array([[1]], dtype=mx.float32),
            mx.array([[2]], dtype=mx.float32),
        )
        self.nbytes = 8

    def is_trimmable(self):
        return False


class FakeModel:
    def __call__(self, *_args, **_kwargs):
        return mx.zeros((1, 1, 4), dtype=mx.float32)


def _engine(**kwargs):
    engine = SimpleEngine("test-model", **kwargs)
    engine._loaded = True
    engine._supports_system_kv_cache = True
    engine._model = SimpleNamespace(model=FakeModel(), tokenizer=BosTokenizer())
    return engine


def _mlx_lm_stream():
    def fake_stream_generate(*_args, **kwargs):
        fake_stream_generate.prompts.append(kwargs["prompt"].tolist())
        yield SimpleNamespace(text="x", token=ord("x"), finish_reason="stop")

    fake_stream_generate.prompts = []
    return fake_stream_generate


async def _chat(engine, messages):
    return [
        chunk
        async for chunk in engine.stream_chat(
            messages, max_tokens=4, temperature=0.0, top_p=1.0
        )
    ]


MESSAGES = [
    {"role": "system", "content": "Rules"},
    {"role": "user", "content": "first"},
]


class TestLlmStreamChat:
    async def test_prefix_trie_path_feeds_one_bos(self):
        engine = _engine(prefix_trie_cache=True, prefix_trie_cache_size=8)
        engine._system_kv_cache_enabled = False
        stream = _mlx_lm_stream()
        prompt = BosTokenizer().apply_chat_template(MESSAGES)

        with (
            patch("mlx_lm.models.cache.make_prompt_cache", return_value=[FakeCache()]),
            patch("mlx_lm.stream_generate", side_effect=stream),
        ):
            chunks = await _chat(engine, MESSAGES)

        (sent,) = stream.prompts
        assert sent == _reference_ids(prompt)
        assert chunks[-1].prompt_tokens == len(_reference_ids(prompt))

    async def test_system_prefix_snapshot_is_keyed_on_the_ids_the_model_sees(self):
        # A snapshot stored for the system prefix is a hit only if the prefix's token
        # count is computed with the same rule as the full prompt's.
        tokenizer = BosTokenizer()
        engine = _engine()
        rendered = [
            tokenizer.apply_chat_template(
                [MESSAGES[0], {"role": "user", "content": content}]
            )
            for content in ("Alpha", "Bravo")
        ]
        boundary = next(i for i, (a, b) in enumerate(zip(*rendered)) if a != b)
        prefix = rendered[0][:boundary]
        engine._system_kv_cache[hashlib.sha256(prefix.encode()).hexdigest()[:16]] = (
            [FakeCache().state],
            len(_reference_ids(prefix)),
        )
        stream = _mlx_lm_stream()

        with (
            patch("mlx_lm.models.cache.make_prompt_cache", return_value=[FakeCache()]),
            patch("mlx_lm.stream_generate", side_effect=stream),
        ):
            chunks = await _chat(engine, MESSAGES)

        assert chunks[-1].cached_tokens == len(_reference_ids(prefix))
        # The suffix handed to mlx-lm starts where the cached prefix ends.
        full = _reference_ids(tokenizer.apply_chat_template(MESSAGES))
        assert stream.prompts == [full[len(_reference_ids(prefix)) :]]


class TestTextRoute:
    async def test_prompt_tokens_are_the_ids_the_model_sees(self):
        # MLLM engine, text-only request on the mlx-lm text model.
        engine = SimpleEngine("test-model", force_mllm=True)
        engine._loaded = True
        engine._is_mllm = True
        engine._text_tokenizer = BosTokenizer()
        engine._text_model = SimpleNamespace(model=FakeModel())
        engine._supports_system_kv_cache = False
        prompt = BosTokenizer().apply_chat_template(MESSAGES)

        with (
            patch("mlx_lm.stream_generate", return_value=iter(())),
            patch("mlx_lm.models.cache.make_prompt_cache", return_value=[]),
            patch("mlx_lm.sample_utils.make_sampler", return_value=object()),
        ):
            chunks = [
                chunk
                async for chunk in engine._stream_generate_text(
                    MESSAGES, max_tokens=4, temperature=0.0, top_p=1.0
                )
            ]

        assert chunks[-1].prompt_tokens == len(_reference_ids(prompt))


class TestRawCompletions:
    async def test_prompt_tokens_fallback_does_not_double_the_bos(self):
        # A stream that reports no prompt count and ends unfinished: the engine counts
        # the prompt itself.
        engine = SimpleEngine("test-model")
        engine._loaded = True
        engine._model = SimpleNamespace(
            tokenizer=BosTokenizer(),
            stream_generate=lambda **kwargs: iter([SimpleNamespace(text="x")]),
        )
        prompt = f"{BOS} once upon"

        chunks = [c async for c in engine.stream_generate(prompt, max_tokens=8)]

        assert chunks[-1].prompt_tokens == len(_reference_ids(prompt))

    async def test_a_raw_prompt_without_the_bos_text_gets_the_tokenizer_bos(self):
        engine = SimpleEngine("test-model")
        engine._loaded = True
        engine._model = SimpleNamespace(
            tokenizer=BosTokenizer(),
            stream_generate=lambda **kwargs: iter([SimpleNamespace(text="x")]),
        )

        chunks = [c async for c in engine.stream_generate("once upon", max_tokens=8)]

        assert chunks[-1].prompt_tokens == 3  # BOS + two words

    async def test_specprefill_threshold_counts_the_ids_the_model_sees(self):
        engine = SimpleEngine("test-model")
        engine._loaded = True
        engine._model = SimpleNamespace(tokenizer=BosTokenizer())
        engine._draft_model = object()
        engine._specprefill_threshold = 0
        seen = {}

        async def fake_specprefill(prompt, tokens_list, *args, **kwargs):
            seen["tokens"] = tokens_list
            yield SimpleNamespace(finished=True)

        engine._stream_generate_specprefill = fake_specprefill
        prompt = f"{BOS} once upon"

        async for _ in engine.stream_generate(prompt, max_tokens=8):
            pass

        assert seen["tokens"] == _reference_ids(prompt)


class BoslessTemplateTokenizer(BosTokenizer):
    """A template that renders no BOS (ChatML-style) on a tokenizer that adds one."""

    def apply_chat_template(self, messages, **_kwargs):
        body = " ".join(f"<|{m['role']}|> {m.get('content', '')}" for m in messages)
        return f"{body} <|assistant|>"


class TestTemplateWithoutBos:
    """The mlx-lm rule keeps the tokenizer's BOS when the prompt carries none."""

    async def test_text_route(self):
        engine = SimpleEngine("test-model", force_mllm=True)
        engine._loaded = True
        engine._is_mllm = True
        engine._text_tokenizer = BoslessTemplateTokenizer()
        engine._text_model = SimpleNamespace(model=FakeModel())
        engine._supports_system_kv_cache = False
        prompt = BoslessTemplateTokenizer().apply_chat_template(MESSAGES)

        with (
            patch("mlx_lm.stream_generate", return_value=iter(())),
            patch("mlx_lm.models.cache.make_prompt_cache", return_value=[]),
            patch("mlx_lm.sample_utils.make_sampler", return_value=object()),
        ):
            chunks = [
                chunk
                async for chunk in engine._stream_generate_text(
                    MESSAGES, max_tokens=4, temperature=0.0, top_p=1.0
                )
            ]

        assert chunks[-1].prompt_tokens == len(_reference_ids(prompt)) + 1

    async def test_llm_stream_chat_prefix_trie_path(self):
        engine = _engine(prefix_trie_cache=True, prefix_trie_cache_size=8)
        engine._system_kv_cache_enabled = False
        engine._model.tokenizer = BoslessTemplateTokenizer()
        stream = _mlx_lm_stream()
        prompt = BoslessTemplateTokenizer().apply_chat_template(MESSAGES)

        with (
            patch("mlx_lm.models.cache.make_prompt_cache", return_value=[FakeCache()]),
            patch("mlx_lm.stream_generate", side_effect=stream),
        ):
            await _chat(engine, MESSAGES)

        assert stream.prompts == [[BOS_ID, *_reference_ids(prompt)]]
