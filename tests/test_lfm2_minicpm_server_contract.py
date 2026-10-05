# SPDX-License-Identifier: Apache-2.0
"""How the LFM2 and MiniCPM parsers behave under the server's streaming contract.

The server calls ``finalize_streaming`` at the end of a response only after a delta that
produced nothing, unless the parser sets ``REQUIRES_EAGER_STREAMING``. These parsers set
it, because the last delta can both complete one call and leave another unclosed.
"""

import time

import pytest

from vllm_mlx import server
from vllm_mlx.tool_parsers.lfm2_tool_parser import Lfm2ToolParser
from vllm_mlx.tool_parsers.minicpm_tool_parser import MiniCPMToolParser

LFM_TWO = "<|tool_call_start|>[a(x=1)]<|tool_call_end|><|tool_call_start|>[b(y=2)]"
MINI_TWO = (
    '<function name="a"><param name="x">1</param></function>'
    '<function name="b"><param name="y">2</param>'
)

CASES = {
    "lfm2": (Lfm2ToolParser, LFM_TWO, "<|tool_call_start|>[f(x=1)]<|tool_call_end|>"),
    "minicpm": (
        MiniCPMToolParser,
        MINI_TWO,
        '<function name="f"><param name="x">1</param></function>',
    ),
}


def _run_like_the_server(parser, text, size):
    """Stream ``text`` and finish the way the server does; return (calls, content)."""
    parser.reset()
    acc, calls, content, last = "", {}, "", None
    deltas = [text[i : i + size] for i in range(0, len(text), size)] or [""]
    for n, delta in enumerate(deltas):
        previous, acc = acc, acc + delta
        last = parser.extract_tool_calls_streaming(previous, acc, delta)
        if n == len(deltas) - 1 and (
            last is None or server._requires_eager_tool_streaming(parser)
        ):
            last = server._finalize_streaming_tool_result(parser, acc, last)
        for tc in (last or {}).get("tool_calls", []):
            assert tc["index"] not in calls, f"index {tc['index']} emitted twice"
            calls[tc["index"]] = tc["function"]["name"]
        content += (last or {}).get("content") or ""
    return calls, content


@pytest.mark.parametrize("name", sorted(CASES))
class TestEagerFinalize:
    def test_parser_asks_for_eager_streaming(self, name):
        assert server._requires_eager_tool_streaming(CASES[name][0]())

    @pytest.mark.parametrize("size", [1, 7, 64, 10_000])
    def test_unclosed_block_after_a_closed_one_is_not_lost(self, name, size):
        cls, two, _one = CASES[name]
        calls, _content = _run_like_the_server(cls(), two, size)
        assert calls == {0: "a", 1: "b"}

    @pytest.mark.parametrize("size", [1, 7, 10_000])
    def test_a_complete_call_is_not_emitted_twice_by_finalize(self, name, size):
        cls, _two, one = CASES[name]
        calls, _content = _run_like_the_server(cls(), one, size)
        assert calls == {0: "f"}

    def test_plain_text_is_unchanged_by_eager_routing(self, name):
        cls = CASES[name][0]
        calls, content = _run_like_the_server(cls(), "Hello there, world.", 3)
        assert calls == {}
        assert content == "Hello there, world."

    def test_plain_text_costs_constant_time_per_delta(self, name):
        cls = CASES[name][0]
        parser = cls()
        parser.reset()
        text = "word " * 40_000  # 200k characters
        acc = ""
        start = time.perf_counter()
        for i in range(0, len(text), 3):
            delta = text[i : i + 3]
            previous, acc = acc, acc + delta
            parser.extract_tool_calls_streaming(previous, acc, delta)
        # A per-delta scan of the accumulated text would take minutes here.
        assert time.perf_counter() - start < 5


class TestMiniCPMProseThatLooksLikeATag:
    """``<function`` inside prose is not an element."""

    @pytest.mark.parametrize("size", [1, 3, 8, 64, 10_000])
    @pytest.mark.parametrize(
        "prose",
        [
            "Use <functions> to call tools.",
            "The <functional> style works.",
            "A <function> tag has no name.",
            "x <function name=",  # generation ends inside what could be a tag
        ],
    )
    def test_prose_is_not_swallowed(self, prose, size):
        calls, content = _run_like_the_server(MiniCPMToolParser(), prose, size)
        assert calls == {}
        assert content == prose

    @pytest.mark.parametrize("size", [1, 5, 10_000])
    def test_a_real_call_after_lookalike_prose_is_still_found(self, size):
        text = (
            "Use <functions> to call.\n"
            '<function name="get_weather"><param name="city">Paris</param></function>'
        )
        calls, content = _run_like_the_server(MiniCPMToolParser(), text, size)
        assert calls == {0: "get_weather"}
        assert content.split() == ["Use", "<functions>", "to", "call."]


@pytest.mark.parametrize("name", sorted(CASES))
class TestLeadingWhitespaceIsChunkIndependent:
    """Leading whitespace is dropped however it is split from the first word."""

    @pytest.mark.parametrize("size", [1, 2, 3, 5, 13, 64, 10_000])
    def test_plain_prose(self, name, size):
        _calls, content = _run_like_the_server(
            CASES[name][0](), "   three spaces then word. And more.", size
        )
        assert content == "three spaces then word. And more."

    @pytest.mark.parametrize("size", [1, 7, 60, 10_000])
    def test_prose_right_after_a_call(self, name, size):
        cls, _two, one = CASES[name]
        _calls, content = _run_like_the_server(cls(), one + " PostWord more", size)
        assert content == "PostWord more"


class TestMiniCPMTagShapes:
    @pytest.mark.parametrize("size", [1, 3, 10_000])
    def test_extra_whitespace_inside_the_opening_tag(self, size):
        text = '<function   name="f"><param name="x">1</param></function>'
        calls, content = _run_like_the_server(MiniCPMToolParser(), text, size)
        assert calls == {0: "f"}
        assert content == ""

    @pytest.mark.parametrize("prefix", ["Use <functions> to ", "I <func", "a < b "])
    def test_a_lookalike_early_does_not_disable_the_constant_time_path(self, prefix):
        parser = MiniCPMToolParser()
        parser.reset()
        text = prefix + "word " * 40_000
        acc = ""
        start = time.perf_counter()
        for i in range(0, len(text), 3):
            delta = text[i : i + 3]
            previous, acc = acc, acc + delta
            parser.extract_tool_calls_streaming(previous, acc, delta)
        assert time.perf_counter() - start < 3
        assert parser._open_seen is False
