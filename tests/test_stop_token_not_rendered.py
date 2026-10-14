# SPDX-License-Identifier: Apache-2.0
"""The token that ends a request is a terminator, never content.

vLLM counts the stop/EOS token in ``completion_tokens`` but never renders it
(``skip_special_tokens`` hides EOS, and a stop token's text is only kept under
``include_stop_str_in_output``). A request whose *very first* token is the stop
token used to leak it: the schedulers build the final text from a streaming
detokenizer that is only created when a content token arrives, and otherwise
fell back to decoding every generated id, stop token included. Gemma-3 does
this on a tool-result turn: its first token is ``<end_of_turn>`` and the
non-streaming response came back with ``content == "<end_of_turn>"``.

The tests run a real tiny random-weight model through the real batch generator
(LLM scheduler) and the real MLLM scheduler response loop, with a real offline
tokenizer whose ``decode`` prints special tokens, so a leak is visible.
"""

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from tests.test_ignore_eos_engines import (  # noqa: E402
    _eos_for,
    _request,
    _run,
    _scheduler,
    _wrapped,
    greedy_tokens,  # noqa: F401  (fixture)
    model,  # noqa: F401  (fixture)
)
from vllm_mlx.request import SamplingParams  # noqa: E402


class TestBatchedScheduler:
    def test_stop_token_is_not_rendered_after_content(self, model, greedy_tokens):
        index, eos = _eos_for(greedy_tokens)
        out = _run(_scheduler(model, eos), [_request("a")])["a"]

        assert out.finish_reason == "stop"
        assert "<eos>" not in out.text
        assert out.text.split() == [f"w{t}" for t in greedy_tokens[:index]]
        assert out.streamed.split() == out.text.split()

    def test_first_token_stop_has_empty_text(self, model, greedy_tokens):
        out = _run(_scheduler(model, greedy_tokens[0]), [_request("a")])["a"]

        assert out.finish_reason == "stop"
        assert out.text == ""
        assert out.streamed == ""
        # vLLM keeps the stop token in the generated ids: it is counted.
        assert out.completion_tokens == 1
        assert out.tokens == [greedy_tokens[0]]

    def test_first_token_explicit_stop_token_has_empty_text(self, model, greedy_tokens):
        # ``w<id>`` is an ordinary word for this tokenizer, so only the
        # stop-finishing rule (not the special-token skipping) can hide it.
        out = _run(
            _scheduler(model, None),
            [_request("a", stop_token_ids=[greedy_tokens[0]])],
        )["a"]

        assert out.finish_reason == "stop"
        assert out.text == ""
        assert out.streamed == ""

    def test_neighbours_keep_their_own_text(self, model, greedy_tokens):
        out = _run(
            _scheduler(model, greedy_tokens[0]),
            [_request("stops"), _request("ignores", ignore_eos=True)],
        )

        assert out["stops"].text == ""
        assert out["ignores"].finish_reason == "length"
        assert out["ignores"].tokens == greedy_tokens


class TestMLLMScheduler:
    def _scheduler(self, ignore_eos=False):
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

    def _responses(self, tokens, final_reason):
        from vllm_mlx.mllm_batch_generator import MLLMBatchResponse

        return [
            MLLMBatchResponse(
                uid=0,
                request_id="r",
                token=t,
                logprobs=mx.zeros((4,)),
                finish_reason=final_reason if i == len(tokens) - 1 else None,
            )
            for i, t in enumerate(tokens)
        ]

    def _outputs(self, tokens, final_reason="stop", **kwargs):
        sched = self._scheduler(**kwargs)
        outputs, _ = sched._process_batch_responses(
            self._responses(tokens, final_reason)
        )
        return sched, outputs

    def test_first_token_stop_is_not_content(self):
        _, outputs = self._outputs([9])

        assert outputs[-1].finished and outputs[-1].finish_reason == "stop"
        assert outputs[-1].output_text == ""
        assert [o.new_text for o in outputs] == [""]
        # Counted like vLLM / the Simple engine count the EOS token.
        assert outputs[-1].completion_tokens == 1
        assert outputs[-1].output_token_ids == [9]

    def test_later_stop_is_not_content(self):
        sched, outputs = self._outputs([3, 4, 9])

        assert outputs[-1].output_text.split() == ["w3", "w4"]
        assert "".join(o.new_text for o in outputs).split() == ["w3", "w4"]
        assert outputs[-1].completion_tokens == 3
        assert sched._detokenizer_pool == {}

    def test_stop_token_counted_like_the_other_engines(self):
        sched, _ = self._outputs([3, 9])

        assert sched.total_completion_tokens == 2

    def test_length_finish_keeps_all_content(self):
        _, outputs = self._outputs([3, 4], final_reason="length")

        assert outputs[-1].output_text.split() == ["w3", "w4"]
