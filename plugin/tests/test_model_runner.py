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
    ) -> dict[str, list[int]]:
        """``new``: (req_id, prompt, params) tuples; ``decode``: running req ids;
        ``resumed``: {req_id: all token ids} of preempted requests rescheduled;
        ``chunk``: {req_id: n} caps the prefill tokens scheduled this step."""
        resumed = resumed or {}
        chunk = chunk or {}
        num_scheduled: dict[str, int] = {}
        new_reqs = []
        for req_id, prompt, params in new:
            self.prompt_len[req_id] = len(prompt)
            self.computed[req_id] = 0
            self.outputs[req_id] = 0
            num_scheduled[req_id] = min(chunk.get(req_id, len(prompt)), len(prompt))
            new_reqs.append(
                SimpleNamespace(
                    req_id=req_id,
                    prompt_token_ids=prompt,
                    sampling_params=params,
                    num_computed_tokens=0,
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
    assert "a" not in d.runner._gens and "a" not in d.runner._prompt_lens


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
