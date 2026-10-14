# SPDX-License-Identifier: Apache-2.0
"""BOS/PAD/UNK/EOS tokens a model generates are never rendered into the text.

Hermes-3-Llama-3.2 declares ``<|im_end|>`` as its only EOS. Asked to continue a
raw prompt it ends the document with ``<|end_of_text|>`` (its PAD), which is no
stop token for it, and goes on with ``<|begin_of_text|>`` and the next document;
mlx-lm's detokenizers rendered both markers into the completion text. vLLM
generates the very same ids and prints neither (``skip_special_tokens`` defaults
to true), so the text of ``/v1/completions`` must not carry them. The tokens are
still generated and counted.

These run a real tiny random-weight model through the real batching scheduler and
the real ``mlx_lm.stream_generate`` / ``generate``.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")
tokenizers = pytest.importorskip("tokenizers")

from transformers import PreTrainedTokenizerFast  # noqa: E402

from tests.test_ignore_eos_engines import (  # noqa: E402
    CAP,
    PROMPT,
    VOCAB,
    _first_occurrences,
    _request,
    _run,
    _tiny_model,
    _tokenizer,
)
from vllm_mlx.scheduler import Scheduler, SchedulerConfig  # noqa: E402
from vllm_mlx.utils.tokenizer import (  # noqa: E402
    boundary_token_ids,
    decode_visible,
    hide_boundary_tokens,
    load_model_with_fallback,
)

TOOL_MARKER = "<tool_marker>"


def _special_tokenizer(*, pad_id, bos_id, eos_id, unk_id=VOCAB, marker_id=None):
    """A tokenizer that prints every id as ``w<id>`` except the special ones.

    ``decode`` does not skip special tokens, like the tokenizers of mlx-lm's
    detokenizers, so any leaked id shows up under its name.
    """
    names = {pad_id: "<pad>", bos_id: "<bos>", eos_id: "<eos>"}
    if marker_id is not None:
        names[marker_id] = TOOL_MARKER
    vocab = {f"w{i}": i for i in range(VOCAB) if i not in names}
    vocab.update({name: i for i, name in names.items()})
    vocab["<unk>"] = unk_id
    raw = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token="<unk>"))
    raw.pre_tokenizer = tokenizers.pre_tokenizers.WhitespaceSplit()
    kwargs = {}
    if marker_id is not None:
        kwargs["additional_special_tokens"] = [TOOL_MARKER]
    return PreTrainedTokenizerFast(
        tokenizer_object=raw,
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
        unk_token="<unk>",
        **kwargs,
    )


@pytest.fixture(scope="module")
def model():
    return _tiny_model()


@pytest.fixture(scope="module")
def greedy_tokens(model):
    sched = Scheduler(
        model=model,
        tokenizer=_tokenizer(),
        config=SchedulerConfig(enable_prefix_cache=False),
    )
    return _run(sched, [_request("probe", max_tokens=CAP)])["probe"].tokens


@pytest.fixture(scope="module")
def hermes(greedy_tokens):
    """ids shaped like the Hermes case: the model generates PAD, then BOS.

    EOS is a token the model never generates (``<|im_end|>`` in the real case),
    so neither marker ends the request.
    """
    distinct = [t for _, t in _first_occurrences(greedy_tokens)]
    if len(distinct) < 3:
        pytest.skip("greedy sequence of the tiny model has fewer than 3 distinct ids")
    eos_id = next(i for i in range(VOCAB - 1, 0, -1) if i not in greedy_tokens)
    return SimpleNamespace(
        pad_id=distinct[0], bos_id=distinct[1], eos_id=eos_id, marker_id=distinct[2]
    )


def _ids(hermes):
    """The three boundary ids, without the marker id."""
    return dict(pad_id=hermes.pad_id, bos_id=hermes.bos_id, eos_id=hermes.eos_id)


def _words(greedy_tokens, hidden):
    return [f"w{t}" for t in greedy_tokens if t not in hidden]


class TestBoundaryTokenIds:
    def test_bos_pad_unk_and_eos_are_boundary_tokens(self):
        tok = _special_tokenizer(pad_id=3, bos_id=4, eos_id=5, unk_id=VOCAB)
        assert boundary_token_ids(tok) == {3, 4, 5, VOCAB}

    def test_other_special_tokens_stay_visible_to_the_parsers(self):
        """Tool-call/reasoning markers are special tokens the parsers read."""
        tok = _special_tokenizer(pad_id=3, bos_id=4, eos_id=5, marker_id=6)
        assert 6 in set(tok.all_special_ids)
        assert 6 not in boundary_token_ids(tok)
        assert decode_visible(tok, [7, 6, 3, 8]) == f"w7 {TOOL_MARKER} w8"

    def test_decode_visible_drops_boundary_tokens(self):
        tok = _special_tokenizer(pad_id=3, bos_id=4, eos_id=5)
        assert decode_visible(tok, [7, 3, 4, 8, 5]) == "w7 w8"


class TestBatchedScheduler:
    def _scheduler(self, model, tokenizer):
        return Scheduler(
            model=model,
            tokenizer=tokenizer,
            config=SchedulerConfig(enable_prefix_cache=False),
        )

    def test_generated_pad_and_bos_are_not_in_the_text(
        self, model, greedy_tokens, hermes
    ):
        tok = _special_tokenizer(**_ids(hermes))
        out = _run(self._scheduler(model, tok), [_request("a")])["a"]

        # Neither marker is an EOS: the request runs to the cap, markers counted.
        assert out.finish_reason == "length"
        assert out.completion_tokens == CAP
        assert out.tokens == greedy_tokens

        hidden = {hermes.pad_id, hermes.bos_id}
        for text in (out.text, out.streamed):
            assert "<pad>" not in text and "<bos>" not in text
            assert text.split() == _words(greedy_tokens, hidden)

    def test_parser_visible_special_token_still_reaches_the_text(
        self, model, greedy_tokens, hermes
    ):
        tok = _special_tokenizer(**vars(hermes))
        out = _run(self._scheduler(model, tok), [_request("a")])["a"]
        hidden = {hermes.pad_id, hermes.bos_id}
        assert TOOL_MARKER in out.text.split()
        assert [
            TOOL_MARKER if t == hermes.marker_id else f"w{t}"
            for t in greedy_tokens
            if t not in hidden
        ] == out.text.split()


class TestSimpleEngineTokenizer:
    """``load_model_with_fallback`` is how the Simple engine gets its tokenizer."""

    def _loaded(self, model, tokenizer):
        from mlx_lm.tokenizer_utils import TokenizerWrapper

        with (
            patch(
                "vllm_mlx.utils.tokenizer._prefer_fast_mistral_tokenizer",
                side_effect=lambda name, config: config,
            ),
            patch(
                "vllm_mlx.utils.tokenizer._load_model_with_fallback",
                return_value=(model, TokenizerWrapper(tokenizer)),
            ),
            patch(
                "vllm_mlx.utils.tokenizer._install_custom_chat_template",
                side_effect=lambda name, tok: tok,
            ),
        ):
            return load_model_with_fallback("test-model")

    def _language_model(self, model, tokenizer):
        from vllm_mlx.models.llm import MLXLanguageModel

        loaded_model, loaded_tokenizer = self._loaded(model, tokenizer)
        wrapper = MLXLanguageModel.__new__(MLXLanguageModel)
        wrapper.model = loaded_model
        wrapper.tokenizer = loaded_tokenizer
        wrapper._loaded = True
        wrapper._mtp = False
        return wrapper

    def _tokenizer(self, hermes):
        return _special_tokenizer(**_ids(hermes))

    def test_streamed_text_hides_generated_pad_and_bos(
        self, model, greedy_tokens, hermes
    ):
        lm = self._language_model(model, self._tokenizer(hermes))
        chunks = list(
            lm.stream_generate(prompt=mx.array(PROMPT), max_tokens=CAP, temperature=0.0)
        )
        assert [c.token for c in chunks] == greedy_tokens
        text = "".join(c.text for c in chunks)
        assert "<pad>" not in text and "<bos>" not in text
        assert text.split() == _words(greedy_tokens, {hermes.pad_id, hermes.bos_id})

    def test_generate_text_hides_generated_pad_and_bos(
        self, model, greedy_tokens, hermes
    ):
        lm = self._language_model(model, self._tokenizer(hermes))
        out = lm.generate(prompt=mx.array(PROMPT), max_tokens=CAP, temperature=0.0)
        assert "<pad>" not in out.text and "<bos>" not in out.text
        assert out.text.split() == _words(greedy_tokens, {hermes.pad_id, hermes.bos_id})

    def test_concurrent_detokenizers_do_not_share_state(self, hermes):
        """The wrapper's per-caller copy must also copy the wrapped detokenizer."""
        tok = hide_boundary_tokens(_special_tokenizer(**_ids(hermes)))
        first, second = tok.detokenizer, tok.detokenizer
        first.add_token(1)
        second.add_token(2)
        first.finalize()
        second.finalize()
        assert first.text == "w1"
        assert second.text == "w2"

    def test_hiding_twice_is_a_no_op(self, hermes):
        tok = hide_boundary_tokens(_special_tokenizer(**_ids(hermes)))
        inner = tok._detokenizer
        assert hide_boundary_tokens(tok) is tok
        assert tok._detokenizer is inner
