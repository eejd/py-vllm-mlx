# SPDX-License-Identifier: Apache-2.0
"""Streaming tool calls must be emitted exactly once per index.

A client assembles a streamed tool call by concatenating ``function.name`` and
``function.arguments`` over every ``delta.tool_calls`` entry that shares an
``index`` (the OpenAI streaming contract). A parser that re-sends an already
emitted call on a later delta therefore corrupts it: ``get_weatherget_weather``
and ``{"city": "Paris"}{"city": "Paris"}``.
"""

import json

import pytest

from vllm_mlx.tool_parsers.gemma4_tool_parser import Gemma4ToolParser
from vllm_mlx.tool_parsers.hermes_tool_parser import HermesToolParser
from vllm_mlx.tool_parsers.lfm2_tool_parser import Lfm2ToolParser
from vllm_mlx.tool_parsers.minicpm_tool_parser import MiniCPMToolParser
from vllm_mlx.tool_parsers.qwen_tool_parser import QwenToolParser

_QWEN_CALL = '<tool_call>\n{{"name": "get_weather", "arguments": {{"city": "{c}"}}}}\n</tool_call>'
_GEMMA_CALL = '<|tool_call>call:get_weather{{city:<|"|>{c}<|"|>}}<tool_call|>'
_LFM2_CALL = "<|tool_call_start|>[get_weather(city='{c}')]<|tool_call_end|>"
_MINICPM_CALL = '<function name="get_weather"><param name="city">{c}</param></function>'

# (parser class, one-call template)
PARSERS = {
    "qwen": (QwenToolParser, _QWEN_CALL),
    "hermes": (HermesToolParser, _QWEN_CALL),
    "gemma4": (Gemma4ToolParser, _GEMMA_CALL),
    "lfm2": (Lfm2ToolParser, _LFM2_CALL),
    "minicpm": (MiniCPMToolParser, _MINICPM_CALL),
}

CHUNK_SIZES = [1, 3, 7, 10_000]


def _chunks(text: str, size: int) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)]


def _stream(parser, text: str, size: int) -> dict[int, dict]:
    """Drive the parser delta by delta; assemble calls as a client would."""
    parser.reset()
    calls: dict[int, dict] = {}
    emissions: dict[int, int] = {}
    accumulated = ""
    for delta in _chunks(text, size):
        previous, accumulated = accumulated, accumulated + delta
        result = parser.extract_tool_calls_streaming(
            previous_text=previous,
            current_text=accumulated,
            delta_text=delta,
            request={"tools": []},
        )
        for tc in (result or {}).get("tool_calls") or []:
            i = tc["index"]
            emissions[i] = emissions.get(i, 0) + 1
            slot = calls.setdefault(i, {"name": "", "arguments": ""})
            slot["name"] += tc["function"].get("name") or ""
            slot["arguments"] += tc["function"].get("arguments") or ""
    for i, slot in calls.items():
        slot["emissions"] = emissions[i]
    return calls


def _assert_once(calls: dict[int, dict], cities: list[str]) -> None:
    assert sorted(calls) == list(range(len(cities))), calls
    for i, city in enumerate(cities):
        assert calls[i]["name"] == "get_weather", calls[i]
        assert json.loads(calls[i]["arguments"]) == {"city": city}, calls[i]
        assert calls[i]["emissions"] == 1, calls[i]


@pytest.mark.parametrize("size", CHUNK_SIZES)
@pytest.mark.parametrize("name", sorted(PARSERS))
class TestStreamedToolCallsEmitOnce:
    def _parser(self, name):
        return PARSERS[name][0]()

    def _call(self, name, city):
        return PARSERS[name][1].format(c=city)

    def test_single_call(self, name, size):
        text = self._call(name, "Paris")
        _assert_once(_stream(self._parser(name), text, size), ["Paris"])

    def test_single_call_with_trailing_text(self, name, size):
        text = self._call(name, "Paris") + "\nDone."
        _assert_once(_stream(self._parser(name), text, size), ["Paris"])

    def test_two_calls(self, name, size):
        text = self._call(name, "Paris") + "\n" + self._call(name, "Tokyo")
        _assert_once(_stream(self._parser(name), text, size), ["Paris", "Tokyo"])

    def test_leading_text_then_two_calls(self, name, size):
        text = (
            "Checking both.\n"
            + self._call(name, "Paris")
            + "\n"
            + self._call(name, "Tokyo")
        )
        _assert_once(_stream(self._parser(name), text, size), ["Paris", "Tokyo"])


@pytest.mark.parametrize("size", CHUNK_SIZES)
def test_lfm2_parallel_calls_in_one_block_emit_once(size):
    """LFM2 renders parallel calls inside a single marker pair."""
    text = (
        "<|tool_call_start|>"
        "[get_weather(city='Paris'), get_weather(city='Tokyo')]"
        "<|tool_call_end|>"
    )
    _assert_once(_stream(Lfm2ToolParser(), text, size), ["Paris", "Tokyo"])


@pytest.mark.parametrize("size", CHUNK_SIZES)
def test_lfm2_second_block_continues_the_index(size):
    text = (
        "<|tool_call_start|>[get_weather(city='Paris'), get_weather(city='Tokyo')]"
        "<|tool_call_end|>"
        "<|tool_call_start|>[get_weather(city='Rome')]<|tool_call_end|>"
    )
    _assert_once(_stream(Lfm2ToolParser(), text, size), ["Paris", "Tokyo", "Rome"])
