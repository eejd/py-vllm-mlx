# SPDX-License-Identifier: Apache-2.0
"""A chat prompt's BOS must reach the batching schedulers exactly once.

The chat template renders the BOS as literal text (``<bos>`` for Gemma,
``<|begin_of_text|>`` for Llama-3).  Tokenizing that string with a HF
tokenizer's default ``add_special_tokens=True`` prepends a second BOS, so the
text-only scheduler fed ``[BOS, BOS, ...]`` to the model, keyed its prefix
cache on it, reported ``usage.prompt_tokens`` one too high and (via the LCP in
``_compute_prefix_boundary``) put the shared-prefix boundary one past the real
one.  Upstream vLLM tokenizes rendered chat prompts with
``add_special_tokens=False`` (``vllm/renderers/base.py`` ``default_chat_tok_params``).
The MLLM scheduler's tokens come from ``mlx_vlm.utils.prepare_inputs`` which also
uses ``add_special_tokens=False``.

The fake tokenizer reproduces the HF contract: ``encode`` prepends the BOS id
when ``add_special_tokens`` is true (the default) and splits the BOS marker
text into its single id either way.
"""

from collections import deque

import pytest

from vllm_mlx.engine.batched import BatchedEngine
from vllm_mlx.mllm_scheduler import MLLMScheduler
from vllm_mlx.request import Request, SamplingParams
from vllm_mlx.scheduler import Scheduler
from vllm_mlx.utils.tokenizer import encode_prompt, prompt_adds_special_tokens

BOS = "<bos>"
BOS_ID = 2


class FakeHFTokenizer:
    """HF-style tokenizer: words -> ids, BOS text -> BOS_ID, optional BOS prefix."""

    def __init__(self, bos_token=BOS):
        self.bos_token = bos_token
        self.calls = []
        self._vocab = {}

    def _word(self, word):
        if word == self.bos_token:
            return BOS_ID
        return self._vocab.setdefault(word, 100 + len(self._vocab))

    def encode(self, text, add_special_tokens=True):
        self.calls.append(add_special_tokens)
        ids = [self._word(w) for w in text.split()]
        if add_special_tokens and self.bos_token is not None:
            ids = [BOS_ID] + ids
        return ids


CHAT_PROMPT = f"{BOS} <start_of_turn>user hello <end_of_turn> <start_of_turn>model"


def _reference_ids(tok, prompt):
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


def _bare_scheduler(tokenizer):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.tokenizer = tokenizer
    scheduler.requests = {}
    scheduler.waiting = deque()
    scheduler.block_aware_cache = None
    scheduler.memory_aware_cache = None
    scheduler.prefix_cache = None
    return scheduler


def _request(prompt):
    return Request(
        request_id="r1", prompt=prompt, sampling_params=SamplingParams(max_tokens=4)
    )


class TestPromptEncoding:
    def test_rendered_chat_prompt_gets_a_single_bos(self):
        tok = FakeHFTokenizer()
        ids = encode_prompt(tok, CHAT_PROMPT)
        assert ids == _reference_ids(tok, CHAT_PROMPT)
        assert ids.count(BOS_ID) == 1

    def test_raw_completion_prompt_keeps_its_tokenizer_bos(self):
        tok = FakeHFTokenizer()
        ids = encode_prompt(tok, "once upon a time")
        assert ids[0] == BOS_ID
        assert ids.count(BOS_ID) == 1

    def test_tokenizer_without_bos_is_untouched(self):
        tok = FakeHFTokenizer(bos_token=None)
        assert prompt_adds_special_tokens(tok, "<|im_start|>user hi") is True
        assert encode_prompt(tok, "<|im_start|>user hi") == tok.encode(
            "<|im_start|>user hi"
        )


class TestSchedulerAddRequest:
    def test_chat_prompt_ids_and_count_match_reference(self):
        tok = FakeHFTokenizer()
        scheduler = _bare_scheduler(tok)
        request = _request(CHAT_PROMPT)
        scheduler.add_request(request)
        reference = _reference_ids(tok, CHAT_PROMPT)
        assert request.prompt_token_ids == reference
        assert request.num_prompt_tokens == len(reference)
        # The prefix-cache lookup/key is computed from these ids.
        assert request.remaining_tokens == reference

    def test_raw_completion_prompt_still_gets_bos(self):
        tok = FakeHFTokenizer()
        scheduler = _bare_scheduler(tok)
        request = _request("once upon a time")
        scheduler.add_request(request)
        assert request.prompt_token_ids == [BOS_ID, 100, 101, 102, 103]

    def test_processor_wrapped_tokenizer_is_also_deduplicated(self):
        class Processor:
            def __init__(self, inner):
                self.tokenizer = inner

        tok = FakeHFTokenizer()
        scheduler = _bare_scheduler(Processor(tok))
        request = _request(CHAT_PROMPT)
        scheduler.add_request(request)
        assert request.prompt_token_ids == _reference_ids(tok, CHAT_PROMPT)


class TestMLLMSchedulerPromptCount:
    def test_prompt_token_count_matches_prepare_inputs(self, monkeypatch):
        tok = FakeHFTokenizer()
        scheduler = MLLMScheduler.__new__(MLLMScheduler)
        scheduler.processor = tok
        scheduler.requests = {}
        scheduler.waiting = deque()
        request_id = scheduler.add_request(CHAT_PROMPT, max_tokens=4)
        # mlx_vlm.utils.prepare_inputs tokenizes with add_special_tokens=False.
        assert scheduler.requests[request_id].num_prompt_tokens == len(
            _reference_ids(tok, CHAT_PROMPT)
        )

    def test_processor_wrapped_tokenizer(self):
        class Processor:
            def __init__(self, inner):
                self.tokenizer = inner

        tok = FakeHFTokenizer()
        scheduler = MLLMScheduler.__new__(MLLMScheduler)
        scheduler.processor = Processor(tok)
        scheduler.requests = {}
        scheduler.waiting = deque()
        request_id = scheduler.add_request(CHAT_PROMPT, max_tokens=4)
        assert scheduler.requests[request_id].num_prompt_tokens == len(
            _reference_ids(tok, CHAT_PROMPT)
        )


class TestPrefixBoundary:
    @pytest.mark.parametrize("bos_token", [BOS, None])
    def test_boundary_is_the_shared_prefix_of_the_ids_the_model_sees(self, bos_token):
        tok = FakeHFTokenizer(bos_token=bos_token)
        engine = BatchedEngine.__new__(BatchedEngine)
        engine._is_mllm = False
        engine._processor = None
        engine._tokenizer = tok
        prefix = f"{BOS if bos_token else ''} <sys> be terse".split()
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
        real_ids = _reference_ids(tok, engine._apply_chat_template(messages))
        dummy_ids = _reference_ids(tok, engine._apply_chat_template(dummy))
        expected = 0
        for a, b in zip(real_ids, dummy_ids):
            if a != b:
                break
            expected += 1
        assert boundary == expected
        # Everything before the boundary is identical in the ids actually fed.
        assert real_ids[:boundary] == dummy_ids[:boundary]
        assert real_ids[boundary] != dummy_ids[boundary]
