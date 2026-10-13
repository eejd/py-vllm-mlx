# SPDX-License-Identifier: Apache-2.0
"""MLXModelRunner's two-phase step and structured-output (grammar bitmask) support.

vLLM computes the grammar bitmask only after ``execute_model`` returned
(``EngineCore.step``) and hands it to ``sample_tokens``; a runner that samples
inside ``execute_model`` can never apply it, and the engine then terminates the
request with ``xgrammar ... Failed to advance FSM``. These tests run a tiny
random-weight llama on MLX through the runner.
"""

from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")
pytest.importorskip("vllm")
pytest.importorskip("xgrammar")

from mlx_lm.models import llama  # noqa: E402
from vllm import SamplingParams  # noqa: E402
from vllm.sampling_params import StructuredOutputsParams  # noqa: E402
from vllm.v1.core.sched.output import GrammarOutput  # noqa: E402
from vllm_mlx_plugin.model_runner import (  # noqa: E402
    MLXModelRunner,
    StructuredOutputError,
)

VOCAB = 100
EOS = 0


def _runner() -> MLXModelRunner:
    mx.random.seed(0)
    model = llama.Model(
        llama.ModelArgs(
            model_type="llama",
            hidden_size=32,
            num_hidden_layers=2,
            intermediate_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            rms_norm_eps=1e-5,
            vocab_size=VOCAB,
        )
    )
    mx.eval(model.parameters())
    runner = MLXModelRunner(
        SimpleNamespace(
            model_config=SimpleNamespace(model="tiny"),
            cache_config=SimpleNamespace(block_size=16),
            scheduler_config=None,
        ),
        enable_optimizations=False,
    )
    runner.model = model
    runner.tokenizer = SimpleNamespace(eos_token_id=EOS)
    runner._loaded = True
    return runner


def _plain():
    return SamplingParams(temperature=0.0)


def _structured():
    return SamplingParams(
        temperature=0.0, structured_outputs=StructuredOutputsParams(json_object=True)
    )


class _Driver:
    """Feeds the runner what vLLM's scheduler would for each step.

    Tracks, per request, the prompt length and the tokens computed and sampled
    so far (``num_computed_tokens`` before the step, ``num_output_tokens``), and
    how many tokens the step schedules: the whole remaining prompt (or a chunk
    of it), or one decode token.
    """

    def __init__(self):
        self.runner = _runner()
        self.prompt_len: dict[str, int] = {}
        self.computed: dict[str, int] = {}
        self.outputs: dict[str, int] = {}

    def step(
        self,
        new=(),
        decode=(),
        resumed=None,
        grammar=None,
        chunk=None,
        finished=(),
        preempted=(),
        prefix=None,
    ) -> dict[str, list[int]]:
        """``new``: (req_id, prompt, params) tuples; ``decode``: running req ids;
        ``resumed``: {req_id: all token ids} of preempted requests rescheduled;
        ``chunk``: {req_id: n} caps the prefill tokens scheduled this step;
        ``prefix``: {req_id: n} new requests that vLLM's prefix cache already
        holds the first n tokens of."""
        resumed = resumed or {}
        chunk = chunk or {}
        prefix = prefix or {}
        num_scheduled: dict[str, int] = {}
        new_reqs = []
        for req_id, prompt, params in new:
            self.prompt_len[req_id] = len(prompt)
            self.computed[req_id] = prefix.get(req_id, 0)
            self.outputs[req_id] = 0
            left = len(prompt) - self.computed[req_id]
            num_scheduled[req_id] = min(chunk.get(req_id, left), left)
            new_reqs.append(
                SimpleNamespace(
                    req_id=req_id,
                    prompt_token_ids=prompt,
                    sampling_params=params,
                    num_computed_tokens=self.computed[req_id],
                )
            )
        cached_ids = list(decode) + list(resumed)
        computed, outputs = [], []
        for req_id in cached_ids:
            if req_id in resumed:
                # The scheduler re-prefills prompt plus outputs from scratch.
                self.computed[req_id] = 0
                total = len(resumed[req_id])
                num_scheduled[req_id] = min(chunk.get(req_id, total), total)
            else:
                prefill_left = self.prompt_len[req_id] - self.computed[req_id]
                num_scheduled[req_id] = (
                    min(chunk.get(req_id, prefill_left), prefill_left)
                    if prefill_left > 0
                    else 1
                )
            computed.append(self.computed[req_id])
            outputs.append(self.outputs[req_id])
        scheduler_output = SimpleNamespace(
            scheduled_new_reqs=new_reqs,
            scheduled_cached_reqs=SimpleNamespace(
                req_ids=cached_ids,
                resumed_req_ids=set(resumed),
                all_token_ids=resumed,
                num_computed_tokens=computed,
                num_output_tokens=outputs,
            ),
            num_scheduled_tokens=num_scheduled,
            finished_req_ids=set(finished),
            preempted_req_ids=set(preempted),
        )

        assert self.runner.execute_model(scheduler_output) is None
        out = self.runner.sample_tokens(grammar)
        sampled = dict(zip(out.req_ids, out.sampled_token_ids))
        for req_id, tokens in sampled.items():
            self.computed[req_id] += num_scheduled[req_id]
            self.outputs[req_id] += len(tokens)
        return sampled

    def tokens(self, step_result: dict[str, list[int]]) -> dict[str, int]:
        return {rid: toks[0] for rid, toks in step_result.items()}


def _allowed(*token_ids: int) -> np.ndarray:
    """One packed int32 bitmask row allowing only these tokens.

    Bit i of word j is token 32*j+i, as in ``xgr.allocate_token_bitmask``."""
    allowed = np.zeros(((VOCAB + 31) // 32) * 32, dtype=bool)
    allowed[list(token_ids)] = True
    return np.packbits(allowed, bitorder="little").view(np.int32)


def _grammar(**rows: np.ndarray) -> GrammarOutput:
    return GrammarOutput(list(rows), np.stack(list(rows.values())))


def _unconstrained_run(prompt, n):
    """The tokens the model itself picks for ``prompt`` (greedy)."""
    d = _Driver()
    tokens = [d.tokens(d.step(new=[("a", prompt, _plain())]))["a"]]
    for _ in range(n - 1):
        tokens.append(d.tokens(d.step(decode=["a"]))["a"])
    return tokens


PROMPT = [5, 6, 7, 8]


def test_execute_model_samples_nothing_sample_tokens_does():
    runner = _runner()
    out = runner.execute_model(
        SimpleNamespace(
            scheduled_new_reqs=[
                SimpleNamespace(
                    req_id="a",
                    prompt_token_ids=PROMPT,
                    sampling_params=_plain(),
                    num_computed_tokens=0,
                )
            ],
            scheduled_cached_reqs=SimpleNamespace(
                req_ids=[],
                resumed_req_ids=set(),
                all_token_ids={},
                num_computed_tokens=[],
                num_output_tokens=[],
            ),
            num_scheduled_tokens={"a": len(PROMPT)},
            finished_req_ids=set(),
        )
    )
    assert out is None
    sampled = runner.sample_tokens(None)
    assert sampled.req_ids == ["a"] and len(sampled.sampled_token_ids[0]) == 1
    # Nothing left over: a second call has no step to sample.
    assert runner.sample_tokens(None) is None


def test_cleanup_only_step_returns_an_empty_output_directly():
    d = _Driver()
    d.step(new=[("a", PROMPT, _plain())])
    out = d.runner.execute_model(
        SimpleNamespace(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=SimpleNamespace(
                req_ids=[],
                resumed_req_ids=set(),
                all_token_ids={},
                num_computed_tokens=[],
                num_output_tokens=[],
            ),
            num_scheduled_tokens={},
            finished_req_ids={"a"},
        )
    )
    assert out.req_ids == [] and out.sampled_token_ids == []
    assert "a" not in d.runner._requests and "a" not in d.runner._prompt_lens


def test_grammar_bitmask_constrains_the_sampled_token():
    free = _unconstrained_run(PROMPT, 4)
    # Allow exactly one token the model would not pick on its own, each step.
    target = next(t for t in range(11, VOCAB) if t not in free)
    d = _Driver()
    got = [
        d.tokens(
            d.step(
                new=[("s", PROMPT, _structured())],
                grammar=_grammar(s=_allowed(target)),
            )
        )["s"]
    ]
    for _ in range(3):
        got.append(
            d.tokens(d.step(decode=["s"], grammar=_grammar(s=_allowed(target))))["s"]
        )
    assert got == [target] * 4
    assert free != got


def test_constraint_moves_with_the_grammar_state_each_step():
    """The row of step n+1 is applied to token n+1, not fixed at request start."""
    d = _Driver()
    sequence = [21, 22, 23]
    got = [
        d.tokens(
            d.step(
                new=[("s", PROMPT, _structured())],
                grammar=_grammar(s=_allowed(sequence[0])),
            )
        )["s"]
    ]
    for want in sequence[1:]:
        got.append(
            d.tokens(d.step(decode=["s"], grammar=_grammar(s=_allowed(want))))["s"]
        )
    assert got == sequence


def test_each_bitmask_row_reaches_its_own_request_and_others_are_untouched():
    free = _unconstrained_run([9, 10, 11], 1)[0]
    d = _Driver()
    out = d.tokens(
        d.step(
            new=[
                ("s1", PROMPT, _structured()),
                ("plain", [9, 10, 11], _plain()),
                ("s2", PROMPT, _structured()),
            ],
            # Grammar rows are in structured_output_request_ids order, which
            # need not match the batch order.
            grammar=_grammar(s2=_allowed(31), s1=_allowed(30)),
        )
    )
    assert out == {"s1": 30, "plain": free, "s2": 31}


def test_structured_request_without_a_bitmask_row_samples_freely():
    """vLLM leaves a structured request out of the grammar output while it has
    no constraint to apply."""
    free = _unconstrained_run(PROMPT, 1)[0]
    d = _Driver()
    assert d.tokens(d.step(new=[("s", PROMPT, _structured())]))["s"] == free


def test_masked_stepper_matches_mlx_lm_when_nothing_is_masked():
    """The one-token-at-a-time stepper is the same computation as generate_step."""
    free = _unconstrained_run(PROMPT, 6)
    d = _Driver()
    got = [d.tokens(d.step(new=[("s", PROMPT, _structured())]))["s"]]
    for _ in range(5):
        got.append(d.tokens(d.step(decode=["s"]))["s"])
    assert got == free


@pytest.mark.parametrize("params", [_plain, _structured], ids=["plain", "structured"])
def test_chunked_prefill_samples_only_on_the_step_that_completes_the_prompt(params):
    """A partial prompt chunk reports no token: the async scheduler counts one
    pending output per request that completes its prompt, and an extra token
    makes ``num_output_placeholders`` negative (an assert in vLLM's
    async_scheduler that kills the engine core)."""
    prompt = list(range(5, 15))
    free = _unconstrained_run(prompt, 3)
    d = _Driver()
    assert d.step(new=[("a", prompt, params())], chunk={"a": 4}) == {"a": []}
    assert d.step(decode=["a"], chunk={"a": 3}) == {"a": []}
    # The remaining 3 prompt tokens complete it: now there is a first token.
    got = [d.tokens(d.step(decode=["a"]))["a"]]
    got += [d.tokens(d.step(decode=["a"]))["a"] for _ in range(2)]
    assert got == free


def test_chunked_prefill_of_one_request_does_not_hold_back_another():
    free = _unconstrained_run([9, 10, 11], 1)[0]
    d = _Driver()
    out = d.step(
        new=[("long", list(range(5, 15)), _plain()), ("short", [9, 10, 11], _plain())],
        chunk={"long": 4},
    )
    assert out == {"long": [], "short": [free]}


def test_resumed_structured_request_is_rebuilt_and_still_constrained():
    d = _Driver()
    d.step(new=[("s", PROMPT, _structured())], grammar=_grammar(s=_allowed(40)))
    history = PROMPT + [40]
    got = d.step(
        resumed={"s": history},
        grammar=_grammar(s=_allowed(41)),
    )
    assert got == {"s": [41]}


def test_chunked_reprefill_of_a_resumed_request_counts_its_outputs_too():
    """A preempted request is re-prefilled over prompt plus the outputs it already
    has: a chunk that covers the prompt but not those outputs completes nothing."""
    free = _unconstrained_run(PROMPT, 3)
    d = _Driver()
    d.step(new=[("a", PROMPT, _plain())])
    d.step(decode=["a"])
    history = PROMPT + free[:2]
    assert d.step(resumed={"a": history}, chunk={"a": len(PROMPT)}) == {"a": []}
    # Rescheduled in full, it completes the history and yields the next token.
    assert d.tokens(d.step(resumed={"a": history}))["a"] == free[2]


def test_bitmask_for_a_request_started_without_structured_outputs_is_an_error():
    d = _Driver()
    with pytest.raises(StructuredOutputError):
        # Terminating it with EOS would return text that ignores the grammar.
        d.step(new=[("a", PROMPT, _plain())], grammar=_grammar(a=_allowed(3)))


def test_misshapen_bitmask_is_an_error():
    d = _Driver()
    bad = GrammarOutput(["s"], np.stack([_allowed(3), _allowed(4)]))
    with pytest.raises(StructuredOutputError):
        d.step(new=[("s", PROMPT, _structured())], grammar=bad)


LONG_PROMPT = list(range(5, 17))  # 12 tokens


def _greedy_tokens(params_factory, prompt, n):
    d = _Driver()
    out = [d.tokens(d.step(new=[("a", prompt, params_factory())]))["a"]]
    for _ in range(n - 1):
        out.append(d.tokens(d.step(decode=["a"]))["a"])
    return out


# -- sampling parameters ----------------------------------------------------


@pytest.mark.parametrize("top_k", [VOCAB, VOCAB + 1, 10**6])
def test_top_k_at_or_above_the_vocab_is_disabled_and_does_not_poison_sampling(
    top_k, caplog
):
    """vLLM treats top_k >= vocab as disabled.  mlx-lm raises inside an
    mx.compile that owns the global RNG state, which breaks every later
    temperature > 0 sample in the process (and the error would be turned into
    an EOS, i.e. an empty answer)."""
    d = _Driver()
    with caplog.at_level("ERROR"):
        for req_id, k in (("big", top_k), ("after", 5)):
            params = SamplingParams(temperature=1.0, top_k=k)
            d.step(new=[(req_id, PROMPT, params)])
            for _ in range(3):
                d.step(decode=[req_id])
    assert not [r for r in caplog.records if r.levelname == "ERROR"], caplog.text


def test_top_k_above_the_vocab_samples_like_no_top_k():
    mx.random.seed(7)
    free = _greedy_tokens(lambda: SamplingParams(temperature=1.0, top_k=0), PROMPT, 6)
    mx.random.seed(7)
    big = _greedy_tokens(
        lambda: SamplingParams(temperature=1.0, top_k=VOCAB * 3), PROMPT, 6
    )
    assert big == free


def test_temperature_above_zero_samples():
    greedy = _unconstrained_run(PROMPT, 24)
    mx.random.seed(3)
    sampled = _greedy_tokens(lambda: SamplingParams(temperature=1.5), PROMPT, 24)
    assert sampled != greedy


@pytest.mark.parametrize(
    "narrow",
    [dict(top_k=1), dict(top_p=0.001), dict(min_p=0.999)],
    ids=["top_k", "top_p", "min_p"],
)
@pytest.mark.parametrize("structured", [False, True], ids=["plain", "structured"])
def test_top_k_top_p_min_p_narrow_the_distribution_to_the_greedy_token(
    narrow, structured
):
    """Each filter, set to keep only the top token, makes temperature 1 greedy."""
    greedy = _unconstrained_run(PROMPT, 8)
    mx.random.seed(11)
    so = StructuredOutputsParams(json_object=True) if structured else None
    got = _greedy_tokens(
        lambda: SamplingParams(temperature=1.0, structured_outputs=so, **narrow),
        PROMPT,
        8,
    )
    assert got == greedy


@pytest.mark.parametrize("structured", [False, True], ids=["plain", "structured"])
def test_logit_bias_steers_the_sampled_token(structured):
    so = StructuredOutputsParams(json_object=True) if structured else None
    got = _greedy_tokens(
        lambda: SamplingParams(
            temperature=0.0, logit_bias={55: 100.0}, structured_outputs=so
        ),
        PROMPT,
        4,
    )
    assert got == [55] * 4


# -- structured-output failures ---------------------------------------------


def _structured_step(bitmask):
    d = _Driver()
    out = GrammarOutput(["s"], bitmask)
    return d, lambda: d.step(new=[("s", PROMPT, _structured())], grammar=out)


@pytest.mark.parametrize(
    "bitmask",
    [
        _allowed(3).astype(np.int64)[None],  # wrong dtype
        _allowed(3)[None, :3],  # 96 token bits for a 100-token vocab
    ],
    ids=["int64", "narrower-than-vocab"],
)
def test_an_unusable_bitmask_raises_instead_of_ending_the_request(bitmask):
    _, run = _structured_step(bitmask)
    with pytest.raises(StructuredOutputError):
        run()


def test_a_grammar_that_allows_no_token_leaves_the_decision_to_the_scheduler():
    """Not EOS (a clean stop) and not an arbitrary NaN token: the unconstrained
    sample goes back, and the scheduler's grammar rejects it and ends the
    request with an error, without taking the engine down."""
    free = _unconstrained_run(PROMPT, 1)[0]
    d = _Driver()
    zero = np.zeros((1, (VOCAB + 31) // 32), dtype=np.int32)
    got = d.tokens(
        d.step(
            new=[("s", PROMPT, _structured())],
            grammar=GrammarOutput(["s"], zero),
        )
    )
    assert got == {"s": free}


# -- prefill ---------------------------------------------------------------


@pytest.mark.parametrize("structured", [False, True], ids=["plain", "structured"])
def test_a_partial_chunk_is_prefilled_when_scheduled(structured):
    """The step budget bounds the work: a chunk is fed to the cache in its own
    step, not all at once when the prompt completes."""
    so = StructuredOutputsParams(json_object=True) if structured else None
    d = _Driver()
    params = SamplingParams(temperature=0.0, structured_outputs=so)
    assert d.step(new=[("a", LONG_PROMPT, params)], chunk={"a": 5}) == {"a": []}
    assert d.runner._requests["a"].fed == 5
    assert d.step(decode=["a"], chunk={"a": 4}) == {"a": []}
    assert d.runner._requests["a"].fed == 9
    first = d.tokens(d.step(decode=["a"]))["a"]
    assert first == _unconstrained_run(LONG_PROMPT, 1)[0]


@pytest.mark.parametrize("structured", [False, True], ids=["plain", "structured"])
def test_prompt_longer_than_one_prefill_step_gives_the_same_tokens(
    structured, monkeypatch
):
    from vllm_mlx_plugin.model_runner import _RequestState

    free = _unconstrained_run(LONG_PROMPT, 4)
    monkeypatch.setattr(_RequestState, "PREFILL_STEP_SIZE", 3)
    so = StructuredOutputsParams(json_object=True) if structured else None
    got = _greedy_tokens(
        lambda: SamplingParams(temperature=0.0, structured_outputs=so),
        LONG_PROMPT,
        4,
    )
    assert got == free


def test_prefill_is_fed_in_prefill_step_sized_forward_passes(monkeypatch):
    from vllm_mlx_plugin.model_runner import MLXModelRunner, _RequestState

    monkeypatch.setattr(_RequestState, "PREFILL_STEP_SIZE", 3)
    calls = []
    new_state = MLXModelRunner._new_state

    def spied(self, tokens, params):
        state = new_state(self, tokens, params)
        model = state._model
        state._model = lambda x, **kw: calls.append(x.shape[1]) or model(x, **kw)
        return state

    monkeypatch.setattr(MLXModelRunner, "_new_state", spied)
    d = _Driver()
    d.step(new=[("a", LONG_PROMPT, _plain())], chunk={"a": 10})
    assert calls == [3, 3, 3, 1]


@pytest.mark.parametrize("structured", [False, True], ids=["plain", "structured"])
def test_a_prefix_cache_hit_is_still_fed_from_the_first_token(structured):
    """vLLM's num_computed_tokens speaks for its own KV blocks, not MLX's."""
    so = StructuredOutputsParams(json_object=True) if structured else None
    free = _unconstrained_run(LONG_PROMPT, 2)
    d = _Driver()
    params = SamplingParams(temperature=0.0, structured_outputs=so)
    first = d.tokens(d.step(new=[("a", LONG_PROMPT, params)], prefix={"a": 8}))
    assert first == {"a": free[0]}
    assert d.tokens(d.step(decode=["a"])) == {"a": free[1]}


def test_a_prefix_cache_hit_that_still_has_chunks_to_go_samples_nothing_yet():
    d = _Driver()
    out = d.step(
        new=[("a", LONG_PROMPT, _plain())], prefix={"a": 4}, chunk={"a": 3}
    )
    assert out == {"a": []}
    assert d.runner._requests["a"].fed == 7


# -- preemption, cleanup ----------------------------------------------------


def test_a_preempted_request_gives_up_its_cache_and_resumes_from_the_history():
    free = _unconstrained_run(PROMPT, 4)
    d = _Driver()
    d.step(new=[("a", PROMPT, _plain())])
    d.step(decode=["a"])
    # The scheduler preempts it in a step that runs something else.
    d.step(new=[("b", PROMPT, _plain())], preempted=["a"])
    assert "a" not in d.runner._requests
    assert "a" in d.runner._params and "a" in d.runner._prompt_lens
    history = PROMPT + free[:2]
    assert d.tokens(d.step(resumed={"a": history}))["a"] == free[2]
    assert d.tokens(d.step(decode=["a"]))["a"] == free[3]


def test_a_resumed_request_is_rebuilt_not_continued_from_the_stale_cache():
    """If the old state were kept, resuming with a *different* history would
    go on from it; the rebuild starts from what the scheduler sent."""
    d = _Driver()
    d.step(new=[("a", PROMPT, _plain())])
    other = [30, 31, 32, 33, 34]
    got = d.tokens(d.step(resumed={"a": other}))["a"]
    assert got == _unconstrained_run(other, 1)[0]


def test_finishing_a_request_forgets_everything_about_it():
    d = _Driver()
    d.step(new=[("a", PROMPT, _plain())])
    d.step(finished=["a"], new=[("b", PROMPT, _plain())])
    r = d.runner
    assert "a" not in r._requests and "a" not in r._params
    assert "a" not in r._prompt_lens


# -- failure handling -------------------------------------------------------


def test_a_request_that_cannot_generate_is_ended_with_eos():
    """Without a sampled stop token the scheduler would keep it running."""
    runner = _runner()
    assert runner._step_request("ghost") == [EOS]


def test_an_mlx_error_ends_that_request_with_eos():
    d = _Driver()
    d.step(new=[("a", PROMPT, _plain())])

    def boom(_row):
        raise RuntimeError("metal failure")

    d.runner._requests["a"].next_token = boom
    assert d.runner._step_request("a") == [EOS]
    assert "a" not in d.runner._requests
