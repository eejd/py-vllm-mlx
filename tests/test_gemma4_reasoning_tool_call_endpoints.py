# SPDX-License-Identifier: Apache-2.0
"""Gemma 4 reasoning + tool parsers through the real handlers, whatever the chunking.

A delta that carries ``<|channel>`` with text after it, or both channel markers, used to
be dropped and could leave the reasoning parser in the thinking phase for the rest of
the stream, so a tool call after the thought streamed as reasoning (py-vllm-mlx#72).
Same fake-engine pattern as ``test_reasoning_tool_call_contract.py``.
"""

import re

import pytest

import vllm_mlx.server as srv
from tests.test_reasoning_tool_call_contract import (
    NON_STREAMING_SITES,
    SITES,
    TEXT,
    _run,
)
from vllm_mlx.reasoning import get_parser
from vllm_mlx.tool_parsers import ToolParserManager

CALL = '<|tool_call>call:echo_args{text:<|"|>%s<|"|>}<tool_call|>' % TEXT
THOUGHT = "Let me call it."

RAW = {
    "thought-then-call": f"<|channel>thought\n{THOUGHT}<channel|>{CALL}",
    "thought-newline-then-call": f"<|channel>thought\n{THOUGHT}\n<channel|>\n{CALL}",
    "response-marker-then-call": f"<|channel>thought\n{THOUGHT}\n<|channel>response\n{CALL}",
}

_TOKEN_RE = re.compile(
    r"<\|channel>|<channel\|>|<\|tool_call>|<tool_call\|>|<\|\"\|>|\s+|\w+|."
)


def _chunks(raw: str, chunking: str) -> list[str]:
    tokens = _TOKEN_RE.findall(raw)
    if chunking == "whole":
        return [raw]
    if chunking == "tokens":
        return tokens
    size = int(chunking.removeprefix("x"))
    return ["".join(tokens[i : i + size]) for i in range(0, len(tokens), size)]


# Special tokens (<|channel>, <|tool_call>, ...) are single tokens in a real tokenizer and
# so never split across deltas; a delta can hold several tokens, though.
CHUNKINGS = ["whole", "tokens", "x2", "x3", "x5"]


@pytest.fixture(autouse=True)
def gemma4_parsers(monkeypatch):
    monkeypatch.setattr(srv, "_enable_auto_tool_choice", True)
    monkeypatch.setattr(srv, "_tool_call_parser", "gemma4")
    monkeypatch.setattr(
        srv, "_tool_parser_instance", ToolParserManager.get_tool_parser("gemma4")()
    )
    monkeypatch.setattr(srv, "_reasoning_parser", get_parser("gemma4")())
    monkeypatch.setattr(srv, "_reasoning_parser_name", "gemma4")


def _assert_one_call_after_the_thought(seen):
    assert seen.calls == [("echo_args", {"text": TEXT})]
    assert seen.reasoning.strip() == THOUGHT
    for leaked in ("<|channel>", "<channel|>", "<|tool_call>", "call:echo_args"):
        assert leaked not in seen.content
        assert leaked not in seen.reasoning
    assert seen.content.strip() == ""


@pytest.fixture(autouse=True)
def _patch_engine_chunks(monkeypatch):
    """The shared fake engine splits with the Qwen token regex; use this module's."""
    import tests.test_reasoning_tool_call_contract as contract

    monkeypatch.setattr(contract, "_chunks", _chunks)


@pytest.mark.parametrize("chunking", CHUNKINGS)
@pytest.mark.parametrize("raw_name", list(RAW))
@pytest.mark.parametrize("site", SITES)
@pytest.mark.anyio
async def test_streaming_call_after_the_thought_is_a_call(
    monkeypatch, site, raw_name, chunking
):
    seen = await _run(
        site,
        monkeypatch,
        RAW[raw_name],
        chunking,
        stream=True,
        thinking_open_in_prompt=False,
    )

    _assert_one_call_after_the_thought(seen)


@pytest.mark.parametrize("raw_name", list(RAW))
@pytest.mark.parametrize("site", NON_STREAMING_SITES)
@pytest.mark.anyio
async def test_non_streaming_call_after_the_thought_is_a_call(
    monkeypatch, site, raw_name
):
    seen = await _run(
        site,
        monkeypatch,
        RAW[raw_name],
        "whole",
        stream=False,
        thinking_open_in_prompt=False,
    )

    _assert_one_call_after_the_thought(seen)
