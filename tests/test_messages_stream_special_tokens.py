# SPDX-License-Identifier: Apache-2.0
"""``/v1/messages`` streaming keeps special tokens out of the text for every non-harmony
parser.

The harmony parsers need their control tokens, so the stream feeds them the raw delta; for
everything else a delta made only of special tokens (``<|im_end|>``, ``<|eot_id|>``) is
skipped and the parsers read the filtered text, as before. An eager tool parser handed such
a delta would return it as content.
"""

import re

import pytest

import vllm_mlx.server as srv
from tests.test_reasoning_tool_call_contract import _run
from vllm_mlx.reasoning import get_parser
from vllm_mlx.tool_parsers import ToolParserManager

RAW = "Hello.<|im_end|> more<|eot_id|>Yo"
SHOWN = "Hello. moreYo"
EAGER = ["auto", "minicpm", "lfm2", "phi4_mini_json", "deepseek_v4"]
OTHERS = ["hermes", "qwen", "llama", "mistral", "gemma4", "kimi", "xlam", "harmony"]


def _chunks(raw, chunking):
    if chunking == "tokens":
        return re.findall(r"<\|[a-z_]+\|>|\s+|\w+|.", raw)
    # a special token and the text next to it in one delta
    return re.findall(r"\S*\s*", raw)[:-1]


def _configure(monkeypatch, tool_parser, reasoning_parser):
    import tests.test_reasoning_tool_call_contract as contract

    monkeypatch.setattr(contract, "_chunks", _chunks)
    monkeypatch.setattr(srv, "_max_request_tokens", 32768)
    monkeypatch.setattr(srv, "_enable_auto_tool_choice", True)
    monkeypatch.setattr(srv, "_tool_call_parser", tool_parser)
    monkeypatch.setattr(
        srv, "_tool_parser_instance", ToolParserManager.get_tool_parser(tool_parser)()
    )
    monkeypatch.setattr(
        srv,
        "_reasoning_parser",
        get_parser(reasoning_parser)() if reasoning_parser else None,
    )
    monkeypatch.setattr(srv, "_reasoning_parser_name", reasoning_parser)


@pytest.mark.parametrize("chunking", ["tokens", "mixed"])
@pytest.mark.parametrize("reasoning_parser", [None, "gemma4", "glm4", "gpt_oss"])
@pytest.mark.parametrize("tool_parser", EAGER + OTHERS)
@pytest.mark.anyio
async def test_special_tokens_never_reach_the_text(
    request, monkeypatch, tool_parser, reasoning_parser, chunking
):
    if tool_parser in EAGER and reasoning_parser is None and chunking == "mixed":
        # unchanged from origin/main: an eager parser is handed the raw delta and returns
        # it, special token and all, when the token shares its delta with text
        request.applymarker(pytest.mark.xfail(strict=True, raises=AssertionError))
    _configure(monkeypatch, tool_parser, reasoning_parser)

    seen = await _run(
        "anthropic",
        monkeypatch,
        RAW,
        chunking,
        stream=True,
        thinking_open_in_prompt=False,
    )

    assert seen.content == SHOWN
    assert "<|" not in seen.content + seen.reasoning


class _Recorder:
    """A parser that records what it is fed."""

    fed: list[str] = []


def _recording_parsers(monkeypatch, *, reads_control_tokens):
    from vllm_mlx.reasoning.base import DeltaMessage, ReasoningParser
    from vllm_mlx.tool_parsers.abstract_tool_parser import ToolParser

    reasoning_fed: list[str] = []
    tool_fed: list[str] = []

    class RecordingReasoning(ReasoningParser):
        READS_CONTROL_TOKENS = reads_control_tokens

        def extract_reasoning(self, model_output):
            return None, model_output

        def extract_reasoning_streaming(self, previous_text, current_text, delta_text):
            reasoning_fed.append(delta_text)
            return DeltaMessage(content=delta_text)

    class RecordingTool(ToolParser):
        REQUIRES_EAGER_STREAMING = True

        def extract_tool_calls(self, model_output, request=None):
            raise NotImplementedError

        def extract_tool_calls_streaming(
            self, previous_text, current_text, delta_text, *args, **kwargs
        ):
            tool_fed.append(delta_text)
            return {"content": delta_text}

    monkeypatch.setattr(srv, "_enable_auto_tool_choice", True)
    monkeypatch.setattr(srv, "_tool_call_parser", "recording")
    monkeypatch.setattr(srv, "_tool_parser_instance", RecordingTool())
    monkeypatch.setattr(srv, "_reasoning_parser", RecordingReasoning())
    monkeypatch.setattr(srv, "_reasoning_parser_name", None)
    monkeypatch.setattr(srv, "_max_request_tokens", 32768)
    return reasoning_fed, tool_fed


@pytest.mark.anyio
async def test_a_parser_that_does_not_ask_for_control_tokens_reads_the_filtered_text(
    monkeypatch,
):
    """The special-token filter stays in front of every parser that does not declare
    READS_CONTROL_TOKENS: they are fed the filtered delta, and a delta that is only
    special tokens is not fed at all."""
    import tests.test_reasoning_tool_call_contract as contract

    monkeypatch.setattr(contract, "_chunks", _chunks)
    reasoning_fed, tool_fed = _recording_parsers(
        monkeypatch, reads_control_tokens=False
    )

    seen = await _run(
        "anthropic",
        monkeypatch,
        RAW,
        "tokens",
        stream=True,
        thinking_open_in_prompt=False,
    )

    assert seen.content == SHOWN
    assert "".join(reasoning_fed) == SHOWN
    assert "".join(tool_fed) == SHOWN
    assert not [d for d in reasoning_fed + tool_fed if "<|" in d or not d]


@pytest.mark.anyio
async def test_a_reasoning_parser_that_asks_for_control_tokens_is_fed_them(monkeypatch):
    import tests.test_reasoning_tool_call_contract as contract

    monkeypatch.setattr(contract, "_chunks", _chunks)
    reasoning_fed, _ = _recording_parsers(monkeypatch, reads_control_tokens=True)

    seen = await _run(
        "anthropic",
        monkeypatch,
        RAW,
        "tokens",
        stream=True,
        thinking_open_in_prompt=False,
    )

    assert "".join(reasoning_fed) == RAW
    assert seen.content == SHOWN  # what is shown is filtered all the same
