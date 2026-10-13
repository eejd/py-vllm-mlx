# SPDX-License-Identifier: Apache-2.0
"""``MLXLanguageModel.generate`` reports the tokens the model generated.

The Simple engine's non-streaming LLM chat reads ``completion_tokens`` and the
finish reason from it. It used to re-encode the output text: the encode adds the
tokenizer's BOS, never contains the stop token (so a reply ended by EOS was one
short) and cannot tell a stop from the cap. Llama-3.2-1B live: ``max_tokens=3``
reported ``completion_tokens`` 4 while the streaming route said 3.

vLLM's ``len(token_ids)``: every generated token, the one that ended the request
included. These tests run the real mlx-lm generator on the tiny random-weight
model of ``test_ignore_eos_engines`` (whose tokenizer prints ids as ``w<id>``).
"""

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from tests.test_ignore_eos_engines import (  # noqa: E402
    CAP,
    CHAT,
    PROMPT,
    _chat,
    _eos_for,
    _ids,
    _RealEngines,
    _stream_chat,
    VOCAB,
    _wrapped,
    greedy_tokens,  # noqa: F401  (fixture)
    model,  # noqa: F401  (fixture)
)
from vllm_mlx.models.llm import MLXLanguageModel  # noqa: E402

BOS = VOCAB - 1


def _llm(model, eos_id=None):
    wrapper = MLXLanguageModel.__new__(MLXLanguageModel)
    wrapper.model_name = "tiny"
    wrapper.model = model
    wrapper.tokenizer = _wrapped(eos_id)
    wrapper._loaded = True
    wrapper._mtp = False
    return wrapper


def _generate(wrapper, max_tokens=CAP, **kwargs):
    return wrapper.generate(
        prompt=PROMPT, max_tokens=max_tokens, temperature=0.0, **kwargs
    )


class TestGenerate:
    def test_cap_reports_exactly_the_cap(self, model, greedy_tokens):
        out = _generate(_llm(model))

        assert out.tokens == greedy_tokens
        assert out.finish_reason == "length"

    def test_stop_token_is_counted_and_not_rendered(self, model, greedy_tokens):
        index, eos = _eos_for(greedy_tokens)

        out = _generate(_llm(model, eos))

        assert out.finish_reason == "stop"
        assert out.tokens == greedy_tokens[: index + 1]
        assert out.text.split() == [f"w{t}" for t in greedy_tokens[:index]]

    def test_a_reply_that_is_only_the_stop_token(self, model, greedy_tokens):
        out = _generate(_llm(model, greedy_tokens[0]))

        assert out.finish_reason == "stop"
        assert out.tokens == [greedy_tokens[0]]
        assert out.text == ""

    def test_ignore_eos_runs_to_the_cap_with_every_token_counted(
        self, model, greedy_tokens
    ):
        _, eos = _eos_for(greedy_tokens)

        out = _generate(_llm(model, eos), ignore_eos=True)

        assert out.finish_reason == "length"
        assert out.tokens == greedy_tokens
        assert "<eos>" not in out.text

    def test_stopping_on_the_last_allowed_token_is_a_stop(self, model, greedy_tokens):
        # The EOS is the max_tokens-th token: the generator stops on it, so the
        # reply ended on its own and is not a cut.
        index, eos = _eos_for(greedy_tokens)

        out = _generate(_llm(model, eos), max_tokens=index + 1)

        assert out.finish_reason == "stop"
        assert out.tokens == greedy_tokens[: index + 1]


class TestSimpleEngineNonStreamingChat:
    """What a client reads from ``/v1/chat/completions`` without ``stream``."""

    @pytest.mark.anyio
    async def test_counts_match_the_streaming_route(self, model):
        engines = _RealEngines(model)
        greedy = _ids((await _chat(engines.llm_engine(None))).text)
        assert len(greedy) == CAP
        index, eos = _eos_for(greedy)

        for eos_id, expected, reason in [
            (None, CAP, "length"),
            (eos, index + 1, "stop"),
        ]:
            plain = await _chat(engines.llm_engine(eos_id))
            streamed = await _stream_chat(engines.llm_engine(eos_id))

            assert plain.completion_tokens == expected
            assert plain.finish_reason == reason
            assert plain.completion_tokens == streamed.completion_tokens
            assert plain.finish_reason == streamed.finish_reason

    @pytest.mark.anyio
    async def test_prompt_tokens_are_untouched(self, model):
        engines = _RealEngines(model)

        plain = await _chat(engines.llm_engine(None))

        assert plain.prompt_tokens == sum(len(m["content"].split()) for m in CHAT)


def _bos_tokenizer():
    """Tokenizer that, like Llama/Gemma, prepends BOS to every encode by default."""
    import tokenizers
    from mlx_lm.tokenizer_utils import TokenizerWrapper
    from transformers import PreTrainedTokenizerFast

    vocab = {f"w{i}": i for i in range(BOS)}
    vocab["<bos>"] = BOS
    raw = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token="w0"))
    raw.pre_tokenizer = tokenizers.pre_tokenizers.WhitespaceSplit()
    raw.post_processor = tokenizers.processors.TemplateProcessing(
        single="<bos> $A", special_tokens=[("<bos>", BOS)]
    )
    return TokenizerWrapper(
        PreTrainedTokenizerFast(
            tokenizer_object=raw, bos_token="<bos>", unk_token="w0"
        )
    )


class TestStreamingPromptTokens:
    """``StreamingOutput.prompt_tokens`` is the size of the prompt mlx-lm decodes."""

    def _prompt_tokens(self, model, prompt):
        wrapper = _llm(model)
        wrapper.tokenizer = _bos_tokenizer()
        out = list(wrapper.stream_generate(prompt=prompt, max_tokens=2, temperature=0.0))
        return out[0].prompt_tokens

    def _mlx_lm_prompt_size(self, model, prompt):
        from mlx_lm import stream_generate

        tokenizer = _bos_tokenizer()
        return next(
            iter(stream_generate(model, tokenizer, prompt=prompt, max_tokens=2))
        ).prompt_tokens

    @pytest.mark.parametrize(
        "prompt",
        ["<bos> w5 w6 w7", "w5 w6 w7"],
        ids=["rendered-chat-prompt-with-bos", "raw-completion-prompt"],
    )
    def test_matches_what_mlx_lm_feeds_the_model(self, model, prompt):
        assert self._prompt_tokens(model, prompt) == self._mlx_lm_prompt_size(
            model, prompt
        )

    def test_a_prompt_that_starts_with_bos_does_not_get_a_second(self, model):
        assert self._prompt_tokens(model, "<bos> w5 w6 w7") == 4

    def test_a_raw_prompt_keeps_its_bos(self, model):
        assert self._prompt_tokens(model, "w5 w6 w7") == 4
