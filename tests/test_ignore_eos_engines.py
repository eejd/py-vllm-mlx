# SPDX-License-Identifier: Apache-2.0
"""``ignore_eos`` in every engine path: generation runs past EOS to the cap.

vLLM semantics: with ``ignore_eos`` the model's EOS tokens do not end the
request. The cap (``max_tokens``) or an explicit ``stop_token_ids`` entry does,
EOS tokens are generated and counted in ``completion_tokens``, and they are
not rendered into the text (vLLM hides special tokens by default; a natural
stop never shows them either).

The scheduler and tokenizer tests run a real tiny random-weight model through
the real mlx-lm ``BatchGenerator`` / ``stream_generate``, so they exercise the
actual stop logic, not a mock of it.
"""

import asyncio
import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")
llama = pytest.importorskip("mlx_lm.models.llama")
tokenizers = pytest.importorskip("tokenizers")

from transformers import PreTrainedTokenizerFast  # noqa: E402

from vllm_mlx.request import (  # noqa: E402
    Request,
    SamplingParams,
    UnsupportedRequestOption,
)
from vllm_mlx.scheduler import (  # noqa: E402
    Scheduler,
    SchedulerConfig,
    _stop_override_insert_kwargs,
)

VOCAB = 64
PROMPT = [5, 6, 7, 8]
CAP = 8


def _tiny_model():
    mx.random.seed(3)
    args = llama.ModelArgs(
        model_type="llama",
        hidden_size=32,
        num_hidden_layers=2,
        intermediate_size=64,
        num_attention_heads=2,
        rms_norm_eps=1e-5,
        vocab_size=VOCAB,
        num_key_value_heads=2,
    )
    model = llama.Model(args)
    mx.eval(model.parameters())
    return model


def _tokenizer(eos_id=None):
    """A real offline HF tokenizer whose ``decode`` prints every id as ``w<id>``.

    Special tokens are not skipped by ``decode``, so an EOS id that leaks into
    the text shows up as ``<eos>``.
    """
    vocab = {f"w{i}": i for i in range(VOCAB)}
    if eos_id is not None:
        vocab.pop(f"w{eos_id}")
        vocab["<eos>"] = eos_id
    vocab["<unk>"] = VOCAB
    raw = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token="<unk>"))
    raw.pre_tokenizer = tokenizers.pre_tokenizers.WhitespaceSplit()
    kwargs = {"unk_token": "<unk>"}
    if eos_id is not None:
        kwargs["eos_token"] = "<eos>"
    return PreTrainedTokenizerFast(tokenizer_object=raw, **kwargs)


def _wrapped(eos_id=None):
    from mlx_lm.tokenizer_utils import TokenizerWrapper

    return TokenizerWrapper(_tokenizer(eos_id))


@pytest.fixture(scope="module")
def model():
    return _tiny_model()


@pytest.fixture(scope="module")
def greedy_tokens(model):
    """The first CAP greedy tokens for PROMPT (no EOS configured)."""
    sched = Scheduler(
        model=model,
        tokenizer=_tokenizer(),
        config=SchedulerConfig(enable_prefix_cache=False),
    )
    out = _run(sched, [_request("probe", max_tokens=CAP)])["probe"]
    return out.tokens


def _first_occurrences(greedy_tokens):
    return [
        (i, tok) for i, tok in enumerate(greedy_tokens) if tok not in greedy_tokens[:i]
    ]


def _eos_for(greedy_tokens):
    """(index, id) of the first new token after the first, so EOS cuts a prefix."""
    candidates = _first_occurrences(greedy_tokens)[1:]
    if not candidates:
        pytest.skip("greedy sequence of the tiny model never changes token")
    return candidates[0]


def _request(request_id, max_tokens=CAP, **params):
    request = Request(
        request_id=request_id,
        prompt="p",
        prompt_token_ids=list(PROMPT),
        sampling_params=SamplingParams(
            max_tokens=max_tokens, temperature=0.0, **params
        ),
    )
    request.num_prompt_tokens = len(PROMPT)
    return request


def _run(sched, requests):
    for request in requests:
        sched.add_request(request)
    final = {}
    streamed = {r.request_id: "" for r in requests}
    tokens = {r.request_id: [] for r in requests}
    for _ in range(100):
        step = sched.step()
        for out in step.outputs:
            streamed[out.request_id] += out.new_text
            tokens[out.request_id].extend(out.new_token_ids)
            if out.finished:
                final[out.request_id] = SimpleNamespace(
                    finish_reason=out.finish_reason,
                    completion_tokens=out.completion_tokens,
                    text=out.output_text,
                    streamed=streamed[out.request_id],
                    tokens=tokens[out.request_id],
                )
        if not sched.has_requests():
            break
    return final


def _scheduler(model, eos_id):
    return Scheduler(
        model=model,
        tokenizer=_tokenizer(eos_id),
        config=SchedulerConfig(enable_prefix_cache=False),
    )


class TestBatchedScheduler:
    def test_default_request_stops_at_eos(self, model, greedy_tokens):
        index, eos = _eos_for(greedy_tokens)
        out = _run(_scheduler(model, eos), [_request("a")])["a"]

        assert out.finish_reason == "stop"
        assert out.tokens == greedy_tokens[: index + 1]

    def test_ignore_eos_runs_to_the_cap(self, model, greedy_tokens):
        _, eos = _eos_for(greedy_tokens)
        out = _run(_scheduler(model, eos), [_request("a", ignore_eos=True)])["a"]

        assert out.finish_reason == "length"
        # Exactly the cap, EOS tokens included, so a benchmark can verify N.
        assert out.completion_tokens == CAP
        assert out.tokens == greedy_tokens

    def test_eos_tokens_are_not_rendered(self, model, greedy_tokens):
        _, eos = _eos_for(greedy_tokens)
        out = _run(_scheduler(model, eos), [_request("a", ignore_eos=True)])["a"]

        assert "<eos>" not in out.text
        assert "<eos>" not in out.streamed
        visible = [t for t in greedy_tokens if t != eos]
        assert out.text.split() == [f"w{t}" for t in visible]
        # Streamed deltas add up to the final text.
        assert out.streamed.split() == out.text.split()

    def test_explicit_stop_token_ids_still_stop(self, model, greedy_tokens):
        eos_index, eos = _eos_for(greedy_tokens)
        later = [(i, t) for i, t in _first_occurrences(greedy_tokens) if i > eos_index]
        if not later:
            pytest.skip("no later distinct token to use as an explicit stop")
        stop_at, stop_id = later[0]
        out = _run(
            _scheduler(model, eos),
            [_request("a", ignore_eos=True, stop_token_ids=[stop_id])],
        )["a"]

        # Past EOS, but not past the client's own stop token.
        assert out.finish_reason == "stop"
        assert out.tokens == greedy_tokens[: stop_at + 1]

    def test_output_made_only_of_eos_has_empty_text(self, model, greedy_tokens):
        # One token, and it is EOS: the request finishes before any other token
        # reaches the detokenizer, so the final text must not fall back to
        # decoding the raw ids (which would print ``<eos>``).
        out = _run(
            _scheduler(model, greedy_tokens[0]),
            [_request("a", max_tokens=1, ignore_eos=True)],
        )["a"]

        assert out.finish_reason == "length"
        assert out.completion_tokens == 1
        assert out.text == ""

    def test_requests_in_one_batch_are_independent(self, model, greedy_tokens):
        index, eos = _eos_for(greedy_tokens)
        out = _run(
            _scheduler(model, eos),
            [_request("stops"), _request("ignores", ignore_eos=True)],
        )

        assert out["stops"].finish_reason == "stop"
        assert out["stops"].tokens == greedy_tokens[: index + 1]
        assert out["ignores"].finish_reason == "length"
        assert out["ignores"].tokens == greedy_tokens

    def test_first_token_eos_is_hidden_and_counted(self, model, greedy_tokens):
        out = _run(
            _scheduler(model, greedy_tokens[0]), [_request("a", ignore_eos=True)]
        )["a"]

        assert out.tokens == greedy_tokens
        assert "<eos>" not in out.text


class _Gen:
    """Fakes for the two spellings of the per-sequence stop override."""

    @staticmethod
    def new_api():
        class G:
            def insert(self, prompts, max_tokens=None, stop_sequences=None):
                pass

        return G()

    @staticmethod
    def old_api():
        class G:
            def insert(self, prompts, max_tokens=None, state_machines=None):
                pass

        return G()

    @staticmethod
    def neither():
        class G:
            def insert(self, prompts, max_tokens=None):
                pass

        return G()


class TestStopOverrideKwargs:
    def test_stop_sequences_spelling(self):
        from mlx_lm.generate import StopSequences  # noqa: F401  (mlx-lm >= 0.32)

        kwargs = _stop_override_insert_kwargs(_Gen.new_api(), [7])
        (stop,) = kwargs["stop_sequences"]
        matcher = stop.matcher()
        assert matcher.advance(7) is True
        assert _stop_override_insert_kwargs(_Gen.new_api(), []) is not None
        empty = _stop_override_insert_kwargs(_Gen.new_api(), [])["stop_sequences"][0]
        # No stop ids: nothing ever stops, EOS included.
        assert not any(empty.matcher().advance(t) for t in range(VOCAB))

    def test_state_machines_spelling(self):
        generate = importlib.import_module("mlx_lm.generate")
        if not hasattr(generate, "SequenceStateMachine"):
            pytest.skip("mlx-lm 0.31.x only")
        kwargs = _stop_override_insert_kwargs(_Gen.old_api(), [7])
        (machine,) = kwargs["state_machines"]
        state = machine.make_state()
        _, match, _ = machine.match(state, 7)
        assert match is not None
        empty = _stop_override_insert_kwargs(_Gen.old_api(), [])["state_machines"][0]
        state = empty.make_state()
        assert all(empty.match(state, t)[1] is None for t in range(VOCAB))

    def test_unknown_batch_generator_fails_loudly(self):
        with pytest.raises(RuntimeError, match="per-sequence stop"):
            _stop_override_insert_kwargs(_Gen.neither(), [])


class TestAdmission:
    """A mlx-lm without the per-sequence stop API refuses ignore_eos up front.

    Finding out while scheduling would abort the whole running batch and leave
    the triggering request without an owner.
    """

    def _scheduler(self, model, monkeypatch, generator):
        monkeypatch.setattr("vllm_mlx.scheduler.BatchGenerator", generator)
        return _scheduler(model, eos_id=None)

    def test_ignore_eos_is_refused_before_anything_is_queued(self, model, monkeypatch):
        sched = self._scheduler(model, monkeypatch, type(_Gen.neither()))
        request = _request("a", ignore_eos=True)

        with pytest.raises(UnsupportedRequestOption, match="ignore_eos"):
            sched.add_request(request)

        assert "a" not in sched.requests
        assert not sched.has_requests()

    def test_other_requests_are_unaffected(self, model, monkeypatch):
        sched = self._scheduler(model, monkeypatch, type(_Gen.neither()))

        sched.add_request(_request("a"))

        assert sched.has_requests()

    def test_a_supporting_release_admits_ignore_eos(self, model, monkeypatch):
        for generator in (type(_Gen.new_api()), type(_Gen.old_api())):
            sched = self._scheduler(model, monkeypatch, generator)
            sched.add_request(_request("a", ignore_eos=True))
            assert sched.has_requests()

    @pytest.mark.anyio
    async def test_the_batched_engine_refuses_at_the_door(self, monkeypatch):
        from vllm_mlx.engine.batched import BatchedEngine

        with patch("vllm_mlx.engine.batched.is_mllm_model", return_value=False):
            engine = BatchedEngine("test-model")
        monkeypatch.setattr("vllm_mlx.scheduler.BatchGenerator", type(_Gen.neither()))

        with pytest.raises(UnsupportedRequestOption, match="ignore_eos"):
            await engine.validate_request_options(ignore_eos=True)
        await engine.validate_request_options()  # no flag, nothing to refuse

    def test_the_llm_mtp_draft_hook_never_meets_an_ignore_eos_request(self):
        """``_mtp_next`` ends a request on any EOS draft, ignoring ignore_eos.

        It is only installed on a BatchGenerator with a ``_step`` hook, and the
        installed release's has none, so the hook is dormant. Should a release
        bring ``_step`` back it must not also advertise a per-sequence stop
        override, or ``ignore_eos`` would reach that unaware code.
        """
        from vllm_mlx import scheduler

        generator = scheduler.BatchGenerator
        has_step_hook = hasattr(generator, "_step")
        has_stop_override = scheduler._stop_override_param(generator) is not None
        assert not (has_step_hook and has_stop_override)


class TestTokenizerView:
    def test_mlx_lm_stream_generate_runs_past_eos(self, model, greedy_tokens):
        from mlx_lm import stream_generate

        from vllm_mlx.utils.tokenizer import eos_ignoring_tokenizer

        index, eos = _eos_for(greedy_tokens)
        tokenizer = _wrapped(eos)
        prompt = mx.array(PROMPT)

        stopped = list(stream_generate(model, tokenizer, prompt=prompt, max_tokens=CAP))
        assert stopped[-1].finish_reason == "stop"
        assert len(stopped) == index + 1

        view = eos_ignoring_tokenizer(tokenizer)
        ran = list(stream_generate(model, view, prompt=prompt, max_tokens=CAP))
        assert len(ran) == CAP
        assert ran[-1].finish_reason == "length"
        text = "".join(r.text for r in ran)
        assert "<eos>" not in text
        assert text.split() == [f"w{t}" for t in greedy_tokens if t != eos]

        # The shared tokenizer still stops on EOS afterwards.
        assert tokenizer.eos_token_ids == {eos}
        again = list(stream_generate(model, tokenizer, prompt=prompt, max_tokens=CAP))
        assert len(again) == index + 1

    def test_language_model_wrapper_honours_ignore_eos(self, model, greedy_tokens):
        from vllm_mlx.models.llm import MLXLanguageModel

        index, eos = _eos_for(greedy_tokens)
        wrapper = MLXLanguageModel.__new__(MLXLanguageModel)
        wrapper.model = model
        wrapper.tokenizer = _wrapped(eos)
        wrapper._loaded = True
        wrapper._mtp = False

        def run(**kwargs):
            return list(
                wrapper.stream_generate(
                    prompt=mx.array(PROMPT),
                    max_tokens=CAP,
                    temperature=0.0,
                    **kwargs,
                )
            )

        assert len(run()) == index + 1

        ran = run(ignore_eos=True)
        assert len(ran) == CAP
        assert ran[-1].finish_reason == "length"
        assert "<eos>" not in "".join(c.text for c in ran)


class TestMLLMBatchGenerator:
    """Real ``MLLMBatchGenerator.next()`` with only the model forward faked."""

    def _generator(self):
        from tests.test_batch_reshuffle_metadata import TestGeneratorPathChurn

        gen = TestGeneratorPathChurn()._make_generator()
        gen.stop_tokens = {1}  # the fake decode step always returns token 1
        return gen

    def _request(self, uid, max_tokens, **kwargs):
        from vllm_mlx.mllm_batch_generator import MLLMBatchRequest

        return MLLMBatchRequest(
            uid=uid,
            request_id=f"req-{uid}",
            prompt="x",
            max_tokens=max_tokens,
            is_text_only=True,
            **kwargs,
        )

    def _drain(self, gen, requests, chunked=False):
        from vllm_mlx.mllm_batch_generator import install_chunked_prefill_mllm

        gen.unprocessed_requests.extend(requests)
        finish = {}
        counts = {r.uid: 0 for r in requests}
        for step in range(20):
            for r in gen.next():
                counts[r.uid] += 1
                if r.finish_reason is not None:
                    finish[r.uid] = r.finish_reason
            if step == 0 and chunked:
                # Admission went through the stock path; decode now runs the
                # chunked-prefill generation step.
                install_chunked_prefill_mllm(gen, budget=16)
            if len(finish) == len(requests):
                break
        return finish, counts

    def test_stop_predicate_covers_every_decode_path(self):
        """The stock step, chunked-prefill step and MTP draft emission all ask it."""
        from vllm_mlx.mllm_batch_generator import _is_stop_token

        stops = {1}
        plain = self._request(0, 5)
        ignoring = self._request(1, 5, ignore_eos=True)

        assert _is_stop_token(stops, 1, plain)
        assert not _is_stop_token(stops, 1, ignoring)
        assert not _is_stop_token(stops, 2, plain)
        # A draft whose request already left the batch keeps the stock rule.
        assert _is_stop_token(stops, 1, None)

    @pytest.mark.parametrize("chunked", [False, True], ids=["default", "chunked"])
    def test_ignore_eos_request_survives_eos_while_neighbour_stops(self, chunked):
        gen = self._generator()
        finish, counts = self._drain(
            gen,
            [self._request(0, 5), self._request(1, 5, ignore_eos=True)],
            chunked=chunked,
        )

        assert finish == {0: "stop", 1: "length"}
        assert counts[1] == 5
        assert counts[0] < 5


class TestMLLMScheduler:
    def _scheduler(self, ignore_eos):
        from vllm_mlx.mllm_scheduler import MLLMRequest, MLLMScheduler

        sched = MLLMScheduler.__new__(MLLMScheduler)
        sched.processor = _wrapped(eos_id=9)
        sched.stop_tokens = {9}
        sched.uid_to_request_id = {0: "r"}
        sched._detokenizer_pool = {}
        sched.batch_generator = None
        sched.total_completion_tokens = 0
        sched.num_requests_processed = 0
        request = MLLMRequest(
            request_id="r",
            prompt="p",
            sampling_params=SamplingParams(ignore_eos=ignore_eos),
        )
        sched.running = {"r": request}
        return sched

    def _responses(self, tokens):
        from vllm_mlx.mllm_batch_generator import MLLMBatchResponse

        return [
            MLLMBatchResponse(
                uid=0,
                request_id="r",
                token=t,
                logprobs=mx.zeros((4,)),
                finish_reason="length" if i == len(tokens) - 1 else None,
            )
            for i, t in enumerate(tokens)
        ]

    def test_eos_is_counted_but_not_rendered_under_ignore_eos(self):
        sched = self._scheduler(ignore_eos=True)
        outputs, _ = sched._process_batch_responses(self._responses([3, 9, 4]))

        assert outputs[-1].completion_tokens == 3
        assert outputs[-1].output_text.split() == ["w3", "w4"]
        assert [o.new_text for o in outputs][1] == ""

    def test_add_request_carries_the_flag_into_sampling_params(self):
        from vllm_mlx.mllm_scheduler import MLLMScheduler

        sched = MLLMScheduler.__new__(MLLMScheduler)
        sched.processor = _wrapped()
        sched.requests = {}
        sched.waiting = MagicMock()

        request_id = sched.add_request("p", ignore_eos=True)
        assert sched.requests[request_id].sampling_params.ignore_eos is True
        request_id = sched.add_request("p")
        assert sched.requests[request_id].sampling_params.ignore_eos is False


def _request_output():
    return MagicMock(
        output_text="x",
        output_token_ids=[1],
        prompt_tokens=1,
        completion_tokens=1,
        finish_reason="length",
    )


class TestBatchedEngineForwarding:
    def _engine(self, mllm=False):
        from vllm_mlx.engine.batched import BatchedEngine

        with patch("vllm_mlx.engine.batched.is_mllm_model", return_value=mllm):
            engine = BatchedEngine("test-model")
        engine._loaded = True
        engine._is_mllm = mllm
        return engine

    @pytest.mark.anyio
    async def test_generate_passes_ignore_eos_to_sampling_params(self):
        engine = self._engine()
        output = _request_output()
        engine._engine = MagicMock()
        engine._engine.generate = AsyncMock(return_value=output)

        await engine.generate(prompt="p", max_tokens=4, ignore_eos=True)
        await engine.generate(prompt="p", max_tokens=4)

        first, second = (
            c.kwargs["sampling_params"] for c in engine._engine.generate.call_args_list
        )
        assert first.ignore_eos is True
        assert second.ignore_eos is False

    @pytest.mark.anyio
    async def test_mllm_generate_passes_ignore_eos_to_the_scheduler(self):
        engine = self._engine(mllm=True)
        output = _request_output()
        engine._mllm_scheduler = MagicMock()
        engine._mllm_scheduler.generate = AsyncMock(return_value=output)

        await engine.generate(prompt="p", max_tokens=4, ignore_eos=True)

        assert engine._mllm_scheduler.generate.call_args.kwargs["ignore_eos"] is True


class TestSimpleEngineValidation:
    def _engine(self, is_mllm=False):
        from vllm_mlx.engine.simple import SimpleEngine

        with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=is_mllm):
            engine = SimpleEngine("test-model")
        engine._loaded = True
        engine._is_mllm = is_mllm
        return engine

    @pytest.mark.anyio
    async def test_text_models_and_flagless_requests_are_accepted(self):
        text = self._engine(is_mllm=False)
        await text.validate_request_options(ignore_eos=True, has_media=True)
        await text.validate_request_options(ignore_eos=True, completion=True)

        mllm = self._engine(is_mllm=True)
        await mllm.validate_request_options(has_media=True, completion=True)

    @pytest.mark.anyio
    async def test_mllm_refuses_exactly_what_stream_chat_routes_to_mlx_vlm(self):
        engine = self._engine(is_mllm=True)
        engine._text_model = object()
        engine._text_model_initialization_attempted = True

        # Text-only chat with the text route available: accepted.
        await engine.validate_request_options(ignore_eos=True)
        for kwargs in (
            {"has_media": True},  # media decodes through mlx-vlm
            {"completion": True},  # completions always do
        ):
            with pytest.raises(UnsupportedRequestOption, match="ignore_eos"):
                await engine.validate_request_options(ignore_eos=True, **kwargs)

        # No text model: text-only chat falls back to mlx-vlm too.
        engine._text_model = None
        with pytest.raises(UnsupportedRequestOption, match="ignore_eos"):
            await engine.validate_request_options(ignore_eos=True)

        # A drafter requested per request, or by default, takes the mlx-vlm route.
        engine._text_model = object()
        engine._mllm_draft_model_path = "drafter"
        with pytest.raises(UnsupportedRequestOption, match="ignore_eos"):
            await engine.validate_request_options(ignore_eos=True, mllm_draft=True)
        await engine.validate_request_options(ignore_eos=True, mllm_draft=False)
        engine._default_mllm_draft = True
        with pytest.raises(UnsupportedRequestOption, match="ignore_eos"):
            await engine.validate_request_options(ignore_eos=True)
        await engine.validate_request_options(ignore_eos=True, mllm_draft=False)

    @pytest.mark.anyio
    async def test_the_route_itself_still_refuses_as_a_backstop(self):
        engine = self._engine(is_mllm=True)
        engine._model = MagicMock()
        engine._text_model = None

        with pytest.raises(UnsupportedRequestOption, match="ignore_eos"):
            async for _ in engine.stream_generate(
                prompt="p", max_tokens=4, ignore_eos=True
            ):
                pass
        engine._model.stream_generate.assert_not_called()

        with pytest.raises(UnsupportedRequestOption, match="ignore_eos"):
            await engine.chat(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": "file:///x.png"},
                            }
                        ],
                    }
                ],
                max_tokens=4,
                ignore_eos=True,
            )
        engine._model.chat.assert_not_called()


CHAT_TEMPLATE = "{% for m in messages %}{{ m['content'] }} {% endfor %}"
CHAT = [
    {"role": "system", "content": "w1 w2"},
    {"role": "user", "content": "w5 w6 w7 w8"},
]


class _RealEngines:
    """SimpleEngine routes driven end to end by the real tiny model.

    The chat template renders the message contents, so ``w<id>`` words in the
    prompt are token ids and the output text ``w<id> ...`` reads back as ids.
    """

    def __init__(self, model):
        self.model = model

    def _tokenizer(self, eos_id=None):
        wrapper = _wrapped(eos_id)
        wrapper._tokenizer.chat_template = CHAT_TEMPLATE
        return wrapper

    def llm_engine(self, eos_id=None, trie=False):
        from vllm_mlx.engine.simple import SimpleEngine
        from vllm_mlx.models.llm import MLXLanguageModel

        with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False):
            engine = SimpleEngine("tiny")
        wrapper = MLXLanguageModel.__new__(MLXLanguageModel)
        wrapper.model_name = "tiny"
        wrapper.model = self.model
        wrapper.tokenizer = self._tokenizer(eos_id)
        wrapper._loaded = True
        wrapper._mtp = False
        engine._model = wrapper
        engine._loaded = True
        engine._is_mllm = False
        engine._prefix_trie_cache_enabled = trie
        engine._supports_system_kv_cache = trie
        return engine

    def mllm_text_route_engine(self, eos_id=None):
        """A multimodal engine whose text-only chat takes the mlx-lm text route."""
        from vllm_mlx.engine.simple import SimpleEngine

        with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=True):
            engine = SimpleEngine("tiny-mllm")
        engine._loaded = True
        engine._is_mllm = True
        engine._model = MagicMock()
        engine._text_model = self.model
        engine._text_tokenizer = self._tokenizer(eos_id)
        engine._text_model_initialization_attempted = True
        engine._supports_system_kv_cache = False
        return engine


@pytest.fixture(scope="module")
def engines(model):
    return _RealEngines(model)


def _flag(ignore_eos):
    return {"ignore_eos": True} if ignore_eos else {}


async def _stream_chat(engine, ignore_eos=False):
    last = None
    async for out in engine.stream_chat(
        messages=CHAT, max_tokens=CAP, temperature=0.0, **_flag(ignore_eos)
    ):
        last = out
    return last


async def _chat(engine, ignore_eos=False):
    return await engine.chat(
        messages=CHAT, max_tokens=CAP, temperature=0.0, **_flag(ignore_eos)
    )


async def _complete(engine, ignore_eos=False):
    return await engine.generate(
        prompt="w1 w2 w5 w6 w7 w8", max_tokens=CAP, temperature=0.0, **_flag(ignore_eos)
    )


def _ids(text):
    return [int(word[1:]) for word in text.split()]


class TestSimpleEngineRoutes:
    """What a client sees from the Simple engine, on real decode paths."""

    async def _check(self, make_engine, run):
        """EOS cuts the default request short and is ignored under ignore_eos."""
        greedy = _ids((await run(make_engine(None))).text)
        assert len(greedy) == CAP
        _, eos = _eos_for(greedy)

        stopped = await run(make_engine(eos))
        assert stopped.finish_reason == "stop"
        assert stopped.completion_tokens < CAP

        ran = await run(make_engine(eos), ignore_eos=True)
        assert ran.finish_reason == "length"
        # Exactly the cap, EOS tokens included, so a benchmark can verify N.
        assert ran.completion_tokens == CAP
        assert "<eos>" not in ran.text
        assert _ids(ran.text) == [t for t in greedy if t != eos]

    @pytest.mark.anyio
    async def test_llm_stream_chat(self, engines):
        await self._check(engines.llm_engine, _stream_chat)

    @pytest.mark.anyio
    async def test_llm_non_stream_chat(self, engines):
        await self._check(engines.llm_engine, _chat)

    @pytest.mark.anyio
    async def test_llm_completions(self, engines):
        await self._check(engines.llm_engine, _complete)

    @pytest.mark.anyio
    async def test_llm_prefix_trie_cache_path(self, engines):
        """The cache-aware stream path builds its own mlx-lm call."""
        view = importlib.import_module(
            "vllm_mlx.utils.tokenizer"
        ).eos_ignoring_tokenizer
        with patch("vllm_mlx.engine.simple.eos_ignoring_tokenizer", wraps=view) as spy:
            await self._check(
                lambda eos: engines.llm_engine(eos, trie=True), _stream_chat
            )
        # Only that path asks the engine module for the view directly.
        assert spy.called

    @pytest.mark.anyio
    async def test_multimodal_engine_text_route(self, engines):
        await self._check(engines.mllm_text_route_engine, _stream_chat)
        await self._check(engines.mllm_text_route_engine, _chat)
