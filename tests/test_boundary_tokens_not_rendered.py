# SPDX-License-Identifier: Apache-2.0
"""BOS/PAD/UNK/EOS tokens a model generates are not rendered into the text.

Hermes-3-Llama-3.2 declares ``<|im_end|>`` as its only EOS. Asked to continue a
raw prompt it ends the document with ``<|end_of_text|>`` (its PAD), which is no
stop token for it, and goes on with ``<|begin_of_text|>`` and the next document;
mlx-lm's detokenizers rendered both markers into the completion text. vLLM
generates the very same ids and prints neither (``skip_special_tokens`` defaults
to true), so the text of ``/v1/completions`` must not carry them. The tokens are
still generated and counted. A request with ``skip_special_tokens: false`` gets
them rendered, as in vLLM; the token that ends a request (the stop) never is.

These run a real tiny random-weight model through the real batching scheduler and
the real ``mlx_lm.stream_generate`` / ``generate``, on both layouts of mlx-lm's
``TokenizerWrapper`` (0.32: a detokenizer instance; 0.31: a detokenizer class).
"""

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")
tokenizers = pytest.importorskip("tokenizers")

from transformers import PreTrainedTokenizerFast  # noqa: E402

from tests.test_ignore_eos_engines import (  # noqa: E402
    CAP,
    CHAT,
    CHAT_TEMPLATE,
    PROMPT,
    VOCAB,
    _RealEngines,
    _first_occurrences,
    _request,
    _run,
    _tiny_model,
    _tokenizer,
)
from vllm_mlx.request import SamplingParams  # noqa: E402
from vllm_mlx.scheduler import Scheduler, SchedulerConfig  # noqa: E402
from vllm_mlx.utils.tokenizer import (  # noqa: E402
    boundary_token_ids,
    decode_visible,
    eos_ignoring_tokenizer,
    hide_boundary_tokens,
    load_model_with_fallback,
    unfiltered_tokenizer,
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


def _wrapper(tokenizer, layout="current"):
    """An mlx-lm ``TokenizerWrapper`` in the layout of the installed mlx-lm
    (``current``) or of mlx-lm 0.31 (``0.31``: ``_detokenizer_class``, a fresh
    detokenizer on every ``tokenizer.detokenizer`` access; see ``layout``)."""
    from mlx_lm.tokenizer_utils import NaiveStreamingDetokenizer, TokenizerWrapper

    wrapper = TokenizerWrapper(tokenizer)
    if layout == "0.31":
        state = wrapper.__dict__
        state.pop("_detokenizer", None)
        state.setdefault("_detokenizer_class", NaiveStreamingDetokenizer)
    return wrapper


@pytest.fixture(params=["current", "0.31"])
def layout(request, monkeypatch):
    """Run under each layout of mlx-lm's ``TokenizerWrapper``: 0.32 keeps one
    detokenizer instance and copies it per caller; 0.31 builds one from
    ``_detokenizer_class`` on every access."""
    if request.param == "0.31":
        from mlx_lm.tokenizer_utils import TokenizerWrapper

        monkeypatch.setattr(
            TokenizerWrapper,
            "detokenizer",
            property(lambda self: self._detokenizer_class(self)),
        )
    return request.param


def _render(wrapper, ids):
    """What ``mlx_lm.stream_generate`` makes of ``ids``: feed, finalize, read."""
    detokenizer = wrapper.detokenizer
    for token in ids:
        detokenizer.add_token(token)
    detokenizer.finalize()
    return detokenizer.text


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
def distinct(greedy_tokens):
    ids = [t for _, t in _first_occurrences(greedy_tokens)]
    if len(ids) < 3:
        pytest.skip("greedy sequence of the tiny model has fewer than 3 distinct ids")
    return ids


@pytest.fixture(scope="module")
def unused(greedy_tokens):
    """Two ids the model never generates."""
    free = [i for i in range(VOCAB - 1, 0, -1) if i not in greedy_tokens]
    return free[0], free[1]


@pytest.fixture(scope="module")
def hermes(distinct, unused):
    """ids shaped like the Hermes case: the model generates PAD, then BOS.

    EOS is a token the model never generates (``<|im_end|>`` in the real case),
    so neither marker ends the request. ``marker_id`` is a third generated id.
    """
    return SimpleNamespace(
        pad_id=distinct[0], bos_id=distinct[1], eos_id=unused[0], marker_id=distinct[2]
    )


@pytest.fixture(scope="module")
def eos_generated(distinct, unused):
    """The model generates the EOS (second id), then PAD; BOS is never generated."""
    return SimpleNamespace(eos_id=distinct[1], pad_id=distinct[2], bos_id=unused[0])


def _ids(ids):
    """The three boundary ids, without the marker id."""
    return dict(pad_id=ids.pad_id, bos_id=ids.bos_id, eos_id=ids.eos_id)


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

    def test_decode_visible_drops_boundary_tokens_unless_asked_not_to(self):
        tok = _special_tokenizer(pad_id=3, bos_id=4, eos_id=5)
        assert decode_visible(tok, [7, 3, 4, 8, 5]) == "w7 w8"
        assert (
            decode_visible(tok, [7, 3, 4, 8, 5], skip_special_tokens=False)
            == "w7 <pad> <bos> w8 <eos>"
        )

    def test_config_files_are_read_once_per_tokenizer(self):
        from vllm_mlx.utils import tokenizer as tokenizer_utils

        tok = _wrapper(_special_tokenizer(pad_id=3, bos_id=4, eos_id=5))
        with patch.object(
            tokenizer_utils,
            "collect_eos_token_ids",
            wraps=tokenizer_utils.collect_eos_token_ids,
        ) as spy:
            for _ in range(3):
                boundary_token_ids(tok)
            decode_visible(tok, [7, 3])
        assert spy.call_count == 1


class TestDetokenizerLayouts:
    def _tok(self, layout):
        return _wrapper(_special_tokenizer(pad_id=3, bos_id=4, eos_id=5), layout)

    def test_hidden_tokens_are_left_out(self, layout):
        tok = hide_boundary_tokens(self._tok(layout))
        assert _render(tok, [7, 3, 4, 8, 5]) == "w7 w8"

    def test_every_access_gets_its_own_detokenizer(self, layout):
        tok = hide_boundary_tokens(self._tok(layout))
        first, second = tok.detokenizer, tok.detokenizer
        first.add_token(1)
        second.add_token(2)
        first.finalize()
        second.finalize()
        assert (first.text, second.text) == ("w1", "w2")

    def test_hiding_twice_does_not_stack(self, layout):
        tok = hide_boundary_tokens(hide_boundary_tokens(self._tok(layout)))
        assert _render(tok, [7, 3, 8]) == "w7 w8"
        from vllm_mlx.utils.tokenizer import _TokenSkippingDetokenizer

        assert not isinstance(tok.detokenizer._inner, _TokenSkippingDetokenizer)

    def test_a_plain_tokenizer_is_wrapped(self):
        tok = hide_boundary_tokens(_special_tokenizer(pad_id=3, bos_id=4, eos_id=5))
        assert _render(tok, [7, 3, 8]) == "w7 w8"

    def test_eos_ignoring_view_has_no_eos_and_hides_the_markers(self, layout):
        base = hide_boundary_tokens(self._tok(layout))
        view = eos_ignoring_tokenizer(base)
        assert view.eos_token_ids == set()
        assert _render(view, [7, 5, 3, 8]) == "w7 w8"
        # The tokenizer the view was made from is not touched.
        assert base.eos_token_ids == {5}
        assert _render(base, [7, 5, 8]) == "w7 w8"

    def test_eos_ignoring_view_renders_everything_for_skip_special_tokens_false(
        self, layout
    ):
        base = hide_boundary_tokens(self._tok(layout))
        view = eos_ignoring_tokenizer(base, skip_special_tokens=False)
        assert view.eos_token_ids == set()
        assert _render(view, [7, 5, 3, 4, 8]) == "w7 <eos> <pad> <bos> w8"

    def test_unfiltered_view_renders_everything_and_still_stops_on_eos(self, layout):
        base = hide_boundary_tokens(self._tok(layout))
        view = unfiltered_tokenizer(base)
        assert view.eos_token_ids == {5}
        assert _render(view, [7, 3, 4, 8]) == "w7 <pad> <bos> w8"
        assert _render(base, [7, 3, 4, 8]) == "w7 w8"

    def test_an_unknown_layout_keeps_the_model_loadable(self, layout, caplog):
        tok = self._tok(layout)
        for name in ("_detokenizer", "_detokenizer_class"):
            tok.__dict__.pop(name, None)
        with caplog.at_level(logging.WARNING, logger="vllm_mlx.utils.tokenizer"):
            assert hide_boundary_tokens(tok) is tok
        assert "unknown detokenizer layout" in caplog.text


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

    def test_skip_special_tokens_is_decided_per_request(
        self, model, greedy_tokens, hermes
    ):
        """One scheduler, two concurrent requests, opposite settings."""
        tok = _special_tokenizer(**_ids(hermes))
        out = _run(
            self._scheduler(model, tok),
            [_request("keep", skip_special_tokens=False), _request("drop")],
        )
        hidden = {hermes.pad_id, hermes.bos_id}
        assert out["drop"].text.split() == _words(greedy_tokens, hidden)
        every = ["<pad>" if t == hermes.pad_id else f"w{t}" for t in greedy_tokens]
        every = [
            "<bos>" if t == hermes.bos_id else w for t, w in zip(greedy_tokens, every)
        ]
        for text in (out["keep"].text, out["keep"].streamed):
            assert text.split() == every
        assert out["keep"].completion_tokens == out["drop"].completion_tokens == CAP

    def test_ignore_eos_renders_the_eos_only_with_skip_special_tokens_false(
        self, model, greedy_tokens, eos_generated
    ):
        tok = _special_tokenizer(**_ids(eos_generated))
        sched = self._scheduler(model, tok)
        out = _run(
            sched,
            [
                _request("hide", ignore_eos=True),
                _request("show", ignore_eos=True, skip_special_tokens=False),
            ],
        )
        assert out["hide"].completion_tokens == CAP
        hidden = {eos_generated.eos_id, eos_generated.pad_id}
        assert out["hide"].text.split() == _words(greedy_tokens, hidden)
        assert "<eos>" in out["show"].text.split()
        assert "<eos>" in out["show"].streamed.split()
        assert out["show"].text.split().count("<eos>") == greedy_tokens.count(
            eos_generated.eos_id
        )

    @pytest.mark.parametrize("skip", [True, False])
    def test_the_token_that_ends_a_request_is_never_text(
        self, model, greedy_tokens, skip
    ):
        """A request whose first token is the EOS has no text at all.

        The finish path used to decode the raw ids when no detokenizer existed,
        which printed the stop token (``<eos>``).
        """
        first = greedy_tokens[0]
        tok = _special_tokenizer(pad_id=1, bos_id=2, eos_id=first)
        out = _run(
            self._scheduler(model, tok), [_request("a", skip_special_tokens=skip)]
        )["a"]
        assert out.finish_reason == "stop"
        assert out.completion_tokens == 1
        assert out.text == ""
        assert out.streamed == ""


class TestMLLMScheduler:
    """``MLLMScheduler._process_batch_responses`` with the model forward faked."""

    def _scheduler(self, **params):
        from vllm_mlx.mllm_scheduler import MLLMRequest, MLLMScheduler

        sched = MLLMScheduler.__new__(MLLMScheduler)
        sched.processor = _wrapper(_special_tokenizer(pad_id=5, bos_id=6, eos_id=9))
        sched.stop_tokens = {9}
        sched.uid_to_request_id = {0: "r"}
        sched._detokenizer_pool = {}
        sched.batch_generator = None
        sched.total_completion_tokens = 0
        sched.num_requests_processed = 0
        sched.running = {
            "r": MLLMRequest(
                request_id="r", prompt="p", sampling_params=SamplingParams(**params)
            )
        }
        return sched

    def _run(self, sched, tokens, finish="length"):
        from vllm_mlx.mllm_batch_generator import MLLMBatchResponse

        responses = [
            MLLMBatchResponse(
                uid=0,
                request_id="r",
                token=t,
                logprobs=mx.zeros((4,)),
                finish_reason=finish if i == len(tokens) - 1 else None,
            )
            for i, t in enumerate(tokens)
        ]
        outputs, _ = sched._process_batch_responses(responses)
        return outputs

    def test_generated_pad_and_bos_are_not_in_the_text(self):
        outputs = self._run(self._scheduler(), [3, 5, 6, 4])

        assert outputs[-1].completion_tokens == 4
        assert outputs[-1].output_text.split() == ["w3", "w4"]
        assert "".join(o.new_text for o in outputs).split() == ["w3", "w4"]

    def test_skip_special_tokens_false_renders_them(self):
        outputs = self._run(self._scheduler(skip_special_tokens=False), [3, 5, 6, 4])

        assert outputs[-1].output_text.split() == ["w3", "<pad>", "<bos>", "w4"]
        assert "".join(o.new_text for o in outputs).split() == [
            "w3",
            "<pad>",
            "<bos>",
            "w4",
        ]

    def test_ignore_eos_hides_the_eos_unless_skip_special_tokens_is_false(self):
        hidden = self._run(self._scheduler(ignore_eos=True), [3, 9, 4])
        assert hidden[-1].output_text.split() == ["w3", "w4"]
        shown = self._run(
            self._scheduler(ignore_eos=True, skip_special_tokens=False), [3, 9, 4]
        )
        assert shown[-1].output_text.split() == ["w3", "<eos>", "w4"]

    @pytest.mark.parametrize("skip", [True, False])
    def test_a_request_ended_by_its_first_token_has_no_text(self, skip):
        outputs = self._run(
            self._scheduler(skip_special_tokens=skip), [9], finish="stop"
        )
        assert outputs[-1].finish_reason == "stop"
        assert outputs[-1].output_text == ""

    def test_add_request_carries_the_flag_into_sampling_params(self):
        from vllm_mlx.mllm_scheduler import MLLMScheduler

        sched = MLLMScheduler.__new__(MLLMScheduler)
        sched.processor = _wrapper(_special_tokenizer(pad_id=5, bos_id=6, eos_id=9))
        sched.requests = {}
        sched.waiting = MagicMock()

        request_id = sched.add_request("p", skip_special_tokens=False)
        assert sched.requests[request_id].sampling_params.skip_special_tokens is False
        request_id = sched.add_request("p")
        assert sched.requests[request_id].sampling_params.skip_special_tokens is True


class TestSimpleEngineTokenizer:
    """``load_model_with_fallback`` is how the Simple engine gets its tokenizer."""

    def _loaded(self, model, tokenizer, layout="current"):
        with (
            patch(
                "vllm_mlx.utils.tokenizer._prefer_fast_mistral_tokenizer",
                side_effect=lambda name, config: config,
            ),
            patch(
                "vllm_mlx.utils.tokenizer._load_model_with_fallback",
                return_value=(model, _wrapper(tokenizer, layout)),
            ),
            patch(
                "vllm_mlx.utils.tokenizer._install_custom_chat_template",
                side_effect=lambda name, tok: tok,
            ),
        ):
            return load_model_with_fallback("test-model")

    def _language_model(self, model, tokenizer, layout="current"):
        from vllm_mlx.models.llm import MLXLanguageModel

        loaded_model, loaded_tokenizer = self._loaded(model, tokenizer, layout)
        wrapper = MLXLanguageModel.__new__(MLXLanguageModel)
        wrapper.model = loaded_model
        wrapper.tokenizer = loaded_tokenizer
        wrapper._loaded = True
        wrapper._mtp = False
        return wrapper

    def _stream(self, lm, **kwargs):
        chunks = list(
            lm.stream_generate(
                prompt=mx.array(PROMPT), max_tokens=CAP, temperature=0.0, **kwargs
            )
        )
        return chunks, "".join(c.text for c in chunks)

    def test_streamed_text_hides_generated_pad_and_bos(
        self, model, greedy_tokens, hermes, layout
    ):
        lm = self._language_model(model, _special_tokenizer(**_ids(hermes)), layout)
        chunks, text = self._stream(lm)
        assert [c.token for c in chunks] == greedy_tokens
        assert "<pad>" not in text and "<bos>" not in text
        assert text.split() == _words(greedy_tokens, {hermes.pad_id, hermes.bos_id})

    def test_generate_text_hides_generated_pad_and_bos(
        self, model, greedy_tokens, hermes, layout
    ):
        lm = self._language_model(model, _special_tokenizer(**_ids(hermes)), layout)
        out = lm.generate(prompt=mx.array(PROMPT), max_tokens=CAP, temperature=0.0)
        assert "<pad>" not in out.text and "<bos>" not in out.text
        assert out.text.split() == _words(greedy_tokens, {hermes.pad_id, hermes.bos_id})

    def test_generate_reports_the_cap_and_the_generated_ids(
        self, model, greedy_tokens, hermes, layout
    ):
        """Hidden tokens must not make a run that hit the cap look stopped: the
        ids and the finish reason are the generator's, not a re-encoding of the
        (shorter) visible text."""
        lm = self._language_model(model, _special_tokenizer(**_ids(hermes)), layout)
        out = lm.generate(prompt=mx.array(PROMPT), max_tokens=CAP, temperature=0.0)
        assert out.tokens == greedy_tokens
        assert out.finish_reason == "length"

    def test_skip_special_tokens_false_renders_them(
        self, model, greedy_tokens, hermes, layout
    ):
        lm = self._language_model(model, _special_tokenizer(**_ids(hermes)), layout)
        every = [
            {hermes.pad_id: "<pad>", hermes.bos_id: "<bos>"}.get(t, f"w{t}")
            for t in greedy_tokens
        ]
        _, text = self._stream(lm, skip_special_tokens=False)
        assert text.split() == every
        out = lm.generate(
            prompt=mx.array(PROMPT),
            max_tokens=CAP,
            temperature=0.0,
            skip_special_tokens=False,
        )
        assert out.text.split() == every
        # The setting belongs to the request: the next one hides them again.
        _, text = self._stream(lm)
        assert text.split() == _words(greedy_tokens, {hermes.pad_id, hermes.bos_id})

    def test_ignore_eos_renders_the_eos_only_with_skip_special_tokens_false(
        self, model, greedy_tokens, eos_generated, layout
    ):
        lm = self._language_model(
            model, _special_tokenizer(**_ids(eos_generated)), layout
        )
        chunks, text = self._stream(lm, ignore_eos=True)
        assert len(chunks) == CAP
        assert text.split() == _words(
            greedy_tokens, {eos_generated.eos_id, eos_generated.pad_id}
        )
        _, text = self._stream(lm, ignore_eos=True, skip_special_tokens=False)
        assert "<eos>" in text.split()

    def test_the_eos_that_ends_a_request_is_not_text_even_with_skip_false(
        self, model, greedy_tokens, eos_generated
    ):
        lm = self._language_model(model, _special_tokenizer(**_ids(eos_generated)))
        chunks, text = self._stream(lm, skip_special_tokens=False)
        # Generation ends at the EOS (the second token); it is counted, not text.
        assert [c.token for c in chunks] == greedy_tokens[:2]
        assert text.split() == [f"w{greedy_tokens[0]}"]


class _SpecialEngines(_RealEngines):
    """``SimpleEngine`` routes whose tokenizer has PAD, BOS and EOS tokens, set up
    as ``load_model_with_fallback`` leaves a loaded tokenizer."""

    def __init__(self, model, pad_id, bos_id, eos_id):
        super().__init__(model)
        self.ids = dict(pad_id=pad_id, bos_id=bos_id, eos_id=eos_id)

    def _tokenizer(self, eos_id=None):
        wrapper = hide_boundary_tokens(_wrapper(_special_tokenizer(**self.ids)))
        wrapper._tokenizer.chat_template = CHAT_TEMPLATE
        return wrapper


@pytest.fixture(scope="module")
def special_engines(model):
    """The engines' prompt is ``CHAT``; PAD is a token the model generates for it."""
    plain = _RealEngines(model)
    import asyncio

    async def greedy():
        out = await plain.llm_engine().chat(
            messages=CHAT, max_tokens=CAP, temperature=0.0
        )
        return [int(word[1:]) for word in out.text.split()]

    generated = asyncio.run(greedy())
    prompt_ids = {1, 2, 5, 6, 7, 8}
    pad = next((t for t in generated if t not in prompt_ids), None)
    if pad is None:
        pytest.skip("the tiny model only repeats the prompt for CHAT")
    free = [i for i in range(VOCAB - 1, 0, -1) if i not in generated + [pad]]
    engines = _SpecialEngines(model, pad_id=pad, bos_id=free[0], eos_id=free[1])
    engines.generated, engines.pad = generated, pad
    return engines


async def _text(engine, route, **kwargs):
    if route == "stream_chat":
        last = None
        async for out in engine.stream_chat(
            messages=CHAT, max_tokens=CAP, temperature=0.0, **kwargs
        ):
            last = out
        return last
    if route == "chat":
        return await engine.chat(
            messages=CHAT, max_tokens=CAP, temperature=0.0, **kwargs
        )
    return await engine.generate(
        prompt="w1 w2 w5 w6 w7 w8", max_tokens=CAP, temperature=0.0, **kwargs
    )


@pytest.mark.anyio
class TestSimpleEngineRoutes:
    """What a client sees from the Simple engine, on real decode paths."""

    @pytest.mark.parametrize("route", ["stream_chat", "chat", "generate"])
    @pytest.mark.parametrize("build", ["llm_engine", "mllm_text_route_engine"])
    async def test_the_generated_pad_is_hidden_unless_asked_for(
        self, special_engines, build, route
    ):
        if build == "mllm_text_route_engine" and route == "generate":
            pytest.skip("a multimodal engine decodes completions through mlx-vlm")
        make = getattr(special_engines, build)
        pad = special_engines.pad
        expected = [f"w{t}" for t in special_engines.generated if t != pad]
        every = ["<pad>" if t == pad else f"w{t}" for t in special_engines.generated]

        hidden = await _text(make(), route)
        assert hidden.text.split() == expected
        shown = await _text(make(), route, skip_special_tokens=False)
        assert shown.text.split() == every
        if route != "chat":  # a non-stream chat counts the re-encoded text
            assert hidden.completion_tokens == shown.completion_tokens == CAP

    async def test_the_prefix_trie_path_follows_the_request_too(self, special_engines):
        pad = special_engines.pad
        expected = [f"w{t}" for t in special_engines.generated if t != pad]
        every = ["<pad>" if t == pad else f"w{t}" for t in special_engines.generated]

        hidden = await _text(special_engines.llm_engine(trie=True), "stream_chat")
        assert hidden.text.split() == expected
        shown = await _text(
            special_engines.llm_engine(trie=True),
            "stream_chat",
            skip_special_tokens=False,
        )
        assert shown.text.split() == every

    async def test_the_text_route_tokenizer_is_built_hiding_the_markers(self):
        """``_initialize_text_model`` wraps the tokenizer of the derived text model."""
        from vllm_mlx.engine.simple import SimpleEngine

        tokenizer = _special_tokenizer(pad_id=3, bos_id=4, eos_id=5)
        text_model = MagicMock()
        text_model.mtp = None
        mock_mllm = MagicMock()
        mock_mllm.model = MagicMock()
        mock_mllm.get_tokenizer.return_value = tokenizer

        with (
            patch("vllm_mlx.models.mllm.MLXMultimodalLM", return_value=mock_mllm),
            patch(
                "vllm_mlx.text_model_from_vlm.build_text_model",
                return_value=text_model,
            ),
        ):
            engine = SimpleEngine("gemma-4-test", force_mllm=True, mtp=False)
            engine._mllm_draft_model_path = "some/drafter"
            try:
                await engine.start()
                await engine._ensure_text_model_for_request(mllm_draft_requested=False)
                assert _render(engine._text_tokenizer, [7, 3, 4, 8]) == "w7 w8"
            finally:
                await engine.stop()


@pytest.mark.anyio
class TestSpecPrefillSeedToken:
    """The first token of a SpecPrefill generation is sampled outside mlx-lm."""

    async def _first_chunk(self, seed_id, **kwargs):
        from vllm_mlx.engine.simple import SimpleEngine

        def score(*args, cancel_check=None, **kw):
            return mx.array([0.5], dtype=mx.float32)

        def sparse(*args, cancel_check=None, **kw):
            return mx.zeros((1, 1, 8), dtype=mx.float32)

        with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False):
            engine = SimpleEngine("test-model")
        engine._loaded = True
        engine._draft_model = MagicMock()
        engine._model = MagicMock()
        engine._model.model = MagicMock()
        engine._model.tokenizer = hide_boundary_tokens(
            _special_tokenizer(pad_id=3, bos_id=4, eos_id=5)
        )
        engine._model.stream_generate.return_value = iter(())
        with (
            patch("mlx_lm.models.cache.make_prompt_cache", return_value=[]),
            patch(
                "mlx_lm.sample_utils.make_sampler",
                return_value=lambda logits: mx.array([seed_id], dtype=mx.int32),
            ),
            patch("vllm_mlx.specprefill.score_tokens", side_effect=score),
            patch(
                "vllm_mlx.specprefill.select_chunks",
                return_value=mx.array([0], dtype=mx.int32),
            ),
            patch("vllm_mlx.specprefill.sparse_prefill", side_effect=sparse),
            patch("vllm_mlx.specprefill.cleanup_rope"),
        ):
            return [
                chunk
                async for chunk in engine._stream_generate_specprefill(
                    prompt="hello",
                    tokens=[1, 2, 3, 4],
                    max_tokens=4,
                    temperature=0.0,
                    top_p=1.0,
                    **kwargs,
                )
            ]

    @pytest.mark.parametrize("skip", [True, False])
    async def test_a_seed_that_is_the_eos_ends_the_request_with_no_text(self, skip):
        chunks = await self._first_chunk(5, skip_special_tokens=skip)
        assert "".join(c.new_text for c in chunks) == ""
        assert chunks[-1].finish_reason == "stop"

    @pytest.mark.parametrize("skip, expected", [(True, ""), (False, "<pad>")])
    async def test_a_seed_that_is_a_pad_follows_the_request(self, skip, expected):
        chunks = await self._first_chunk(3, skip_special_tokens=skip)
        assert "".join(c.new_text for c in chunks) == expected

    async def test_ignore_eos_seed_is_text_only_when_asked_for(self):
        hidden = await self._first_chunk(5, ignore_eos=True)
        assert "".join(c.new_text for c in hidden) == ""
        shown = await self._first_chunk(5, ignore_eos=True, skip_special_tokens=False)
        assert "".join(c.new_text for c in shown) == "<eos>"


class TestSchedulerHidesTheProcessorsStopIds:
    def test_stop_ids_only_the_processor_declares_are_not_rendered(
        self, model, greedy_tokens, distinct
    ):
        """A multimodal processor wraps the tokenizer and can carry the EOS set."""
        eos = distinct[1]
        inner = _special_tokenizer(
            pad_id=distinct[2], bos_id=VOCAB - 1, eos_id=VOCAB - 2
        )

        class Processor:
            tokenizer = inner
            eos_token_ids = {eos}

            def __getattr__(self, name):
                raise AttributeError(name)

        sched = Scheduler(
            model=model,
            tokenizer=Processor(),
            config=SchedulerConfig(enable_prefix_cache=False),
        )
        assert eos in sched._hidden_token_ids()
