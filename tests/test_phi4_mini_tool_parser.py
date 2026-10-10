# SPDX-License-Identifier: Apache-2.0
"""Tests for the Microsoft Phi-4-mini ``functools[...]`` tool call parser."""

import json
import random
import re

import pytest

from vllm_mlx.tool_parsers import ToolParserManager
from vllm_mlx.tool_parsers.phi4_mini_tool_parser import Phi4MiniToolParser


def _args(call: dict) -> dict:
    return json.loads(call["arguments"])


def _call(name: str, args: dict, key: str = "arguments") -> dict:
    return {"name": name, key: args}


def _text(*calls: dict) -> str:
    return "functools" + json.dumps(list(calls))


def _vllm_regex_extract(model_output: str):
    """vLLM's ``phi4_mini_json`` extraction, kept to show what it gets wrong."""
    match = re.search(r"functools\[(.*?)\]", model_output, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads("[" + match.group(1) + "]")
    except json.JSONDecodeError:
        return None


def _stream(chunks: list[str]) -> tuple[str, dict[int, dict]]:
    """Drive the parser the way the server does; return (content, calls by index)."""
    parser = Phi4MiniToolParser()
    previous = ""
    content = ""
    calls: dict[int, dict] = {}
    emitted: list[int] = []

    def take(result):
        nonlocal content
        if not result:
            return
        content += result.get("content", "")
        for tc in result.get("tool_calls", []):
            emitted.append(tc["index"])
            calls[tc["index"]] = {
                "id": tc["id"],
                "name": tc["function"]["name"],
                "arguments": tc["function"]["arguments"],
            }

    for chunk in chunks:
        current = previous + chunk
        take(parser.extract_tool_calls_streaming(previous, current, chunk))
        previous = current
    take(parser.finalize_streaming(previous))
    assert len(emitted) == len(set(emitted)), f"a call was emitted twice: {emitted}"
    return content, calls


def _chars(text: str) -> list[str]:
    return list(text)


class TestRegistration:
    def test_registered_names(self):
        for name in ("phi4_mini_json", "phi4_mini", "phi4"):
            assert ToolParserManager.get_tool_parser(name) is Phi4MiniToolParser

    def test_declares_streaming_marker_and_native_format(self):
        assert Phi4MiniToolParser.STREAMING_MARKERS == ("functools",)
        assert Phi4MiniToolParser.REQUIRES_EAGER_STREAMING
        assert Phi4MiniToolParser.supports_native_format()
        # An ``_END`` attribute would make the server drop text after a call.
        assert not hasattr(Phi4MiniToolParser, "_END")


class TestExtract:
    def setup_method(self):
        self.parser = Phi4MiniToolParser()

    def test_single_call(self):
        result = self.parser.extract_tool_calls(
            'functools[{"name": "get_weather", "arguments": {"city": "Paris"}}]'
        )
        assert result.tools_called is True
        assert [c["name"] for c in result.tool_calls] == ["get_weather"]
        assert _args(result.tool_calls[0]) == {"city": "Paris"}
        assert result.tool_calls[0]["id"]
        assert result.content is None

    def test_parallel_calls(self):
        result = self.parser.extract_tool_calls(
            _text(
                _call("get_weather", {"city": "Paris"}),
                _call("get_weather", {"city": "Tokyo"}),
            )
        )
        assert [c["name"] for c in result.tool_calls] == ["get_weather"] * 2
        assert [_args(c) for c in result.tool_calls] == [
            {"city": "Paris"},
            {"city": "Tokyo"},
        ]
        assert len({c["id"] for c in result.tool_calls}) == 2

    def test_parameters_key_is_accepted(self):
        result = self.parser.extract_tool_calls(
            _text(_call("f", {"x": 1}, key="parameters"))
        )
        assert _args(result.tool_calls[0]) == {"x": 1}

    def test_arguments_wins_over_parameters(self):
        text = 'functools[{"name": "f", "arguments": {"a": 1}, "parameters": {"b": 2}}]'
        result = self.parser.extract_tool_calls(text)
        assert _args(result.tool_calls[0]) == {"a": 1}

    def test_call_without_arguments_has_empty_object(self):
        result = self.parser.extract_tool_calls('functools[{"name": "ping"}]')
        assert result.tool_calls[0]["arguments"] == "{}"

    def test_arguments_double_encoded_as_string(self):
        result = self.parser.extract_tool_calls(
            'functools[{"name": "f", "arguments": "{\\"a\\": 1}"}]'
        )
        assert _args(result.tool_calls[0]) == {"a": 1}

    def test_no_call_is_plain_content(self):
        result = self.parser.extract_tool_calls("The capital of Poland is Warsaw.")
        assert result.tools_called is False
        assert result.tool_calls == []
        assert result.content == "The capital of Poland is Warsaw."

    def test_functools_not_followed_by_bracket_is_prose(self):
        text = "Use functools.partial, or functools (the module)."
        result = self.parser.extract_tool_calls(text)
        assert result.tools_called is False
        assert result.content == text

    def test_text_before_functools_is_content(self):
        result = self.parser.extract_tool_calls(
            "Let me check the weather.\n" + _text(_call("get_weather", {"city": "X"}))
        )
        assert result.tools_called is True
        assert result.content == "Let me check the weather."
        assert _args(result.tool_calls[0]) == {"city": "X"}

    def test_text_after_the_list_is_content(self):
        result = self.parser.extract_tool_calls(
            _text(_call("f", {})) + " Anything else?"
        )
        assert result.tools_called is True
        assert result.content == "Anything else?"

    def test_two_lists_in_one_output(self):
        result = self.parser.extract_tool_calls(
            _text(_call("a", {"x": 1})) + "\n" + _text(_call("b", {"y": 2}))
        )
        assert [c["name"] for c in result.tool_calls] == ["a", "b"]

    def test_closing_bracket_inside_a_string_argument(self):
        value = "arr[0] ] and [ nested [ ] ]"
        text = _text(_call("note", {"text": value}))
        # The vLLM regex stops at the first ``]`` and cannot even parse that.
        assert _vllm_regex_extract(text) is None
        result = self.parser.extract_tool_calls(text)
        assert result.tools_called is True
        assert _args(result.tool_calls[0]) == {"text": value}

    def test_list_argument_is_not_cut_at_its_closing_bracket(self):
        text = _text(_call("f", {"ids": [1, 2, [3]], "after": "kept"}))
        # Naive regex: the match ends at the first ``]``, losing ``after``.
        assert _vllm_regex_extract(text) is None
        result = self.parser.extract_tool_calls(text)
        assert _args(result.tool_calls[0]) == {"ids": [1, 2, [3]], "after": "kept"}

    def test_escaped_quote_before_bracket_in_string(self):
        value = 'say "hi]" \\ then ]'
        result = self.parser.extract_tool_calls(_text(_call("say", {"t": value})))
        assert _args(result.tool_calls[0]) == {"t": value}

    def test_calls_after_a_bracketed_argument_are_kept(self):
        result = self.parser.extract_tool_calls(
            _text(_call("a", {"v": ["]"]}), _call("b", {"v": 2}))
        )
        assert [c["name"] for c in result.tool_calls] == ["a", "b"]

    def test_non_ascii_is_not_escaped(self):
        result = self.parser.extract_tool_calls(
            _text(_call("f", {"city": "Zürich 東京"}))
        )
        assert result.tool_calls[0]["arguments"] == '{"city": "Zürich 東京"}'

    @pytest.mark.parametrize(
        "payload",
        [
            'functools[{"name": "f", "arguments": {"a": 1}]',  # object never closed
            'functools[{"name": "f", "arguments": {"a": }}]',  # bad value
            "functools[{'name': 'f', 'arguments': {}}]",  # single quotes
            'functools[{"name": "f", "arguments": {"a": 1},}]',  # trailing comma
            "functools[]",  # empty list
            'functools[{"name": "f", "arguments": [1, 2]}]',  # arguments not an object
            'functools[{"name": "f", "arguments": "not json"}]',
            'functools[{"name": "f", "arguments": null}]',
            'functools["f"]',  # call is not an object
            'functools[{"arguments": {"a": 1}}]',  # no name
            'functools[{"name": 7, "arguments": {}}]',  # name not a string
            'functools[{"name": "", "arguments": {}}]',
        ],
    )
    def test_malformed_list_never_yields_a_call(self, payload):
        result = self.parser.extract_tool_calls(payload)
        assert result.tools_called is False
        assert result.tool_calls == []
        # Nothing is dropped silently: the model's text stays visible.
        assert result.content == payload

    def test_one_bad_call_rejects_the_whole_list(self):
        text = _text(_call("good", {"a": 1}), {"name": "bad name", "arguments": {}})
        result = self.parser.extract_tool_calls(text)
        assert result.tools_called is False
        assert result.content == text

    @pytest.mark.parametrize(
        "name",
        ["get weather", "get_weather()", "a;b", "../etc", "<tool>", "1abc", "x" * 200],
    )
    def test_tool_name_must_be_an_identifier(self, name):
        result = self.parser.extract_tool_calls(_text(_call(name, {})))
        assert result.tools_called is False

    @pytest.mark.parametrize(
        "name", ["get_weather", "_f", "get-weather", "ns.fn", "A1"]
    )
    def test_identifier_names_are_accepted(self, name):
        result = self.parser.extract_tool_calls(_text(_call(name, {})))
        assert [c["name"] for c in result.tool_calls] == [name]

    def test_truncated_generation_is_content_not_a_call(self):
        text = 'functools[{"name": "f", "arguments": {"city": "Par'
        result = self.parser.extract_tool_calls(text)
        assert result.tools_called is False
        assert result.content == text

    def test_malformed_list_does_not_hide_a_good_one(self):
        good = _text(_call("ok", {"a": 1}))
        result = self.parser.extract_tool_calls("functools[nope] " + good)
        assert [c["name"] for c in result.tool_calls] == ["ok"]
        assert result.content == "functools[nope]"


STREAM_CASES = {
    "single": _text(_call("get_weather", {"city": "Paris"})),
    "parallel": _text(_call("a", {"x": 1}), _call("b", {"y": [1, 2]})),
    "prose_before": "Checking now. " + _text(_call("a", {"x": "]"})),
    "prose_after": _text(_call("a", {})) + " Done.",
    "two_lists": _text(_call("a", {})) + "\n" + _text(_call("b", {"k": "v"})),
    "plain": "Just an answer with no tools, mentioning functools in passing.",
    "functools_prose": "Use functools[0] carefully, and functools.partial.",
    "malformed": "functools[nope] then " + _text(_call("ok", {"a": 1})),
    "partial_marker_tail": "ends like a marker functools",
    "partial_marker_tail_2": "ends like a marker functo",
}


class TestStreaming:
    def setup_method(self):
        self.parser = Phi4MiniToolParser()

    @staticmethod
    def _chunkings(text: str):
        yield "whole", [text]
        yield "chars", _chars(text)
        rng = random.Random(7)
        for i in range(25):
            chunks, pos = [], 0
            while pos < len(text):
                step = rng.randint(1, 9)
                chunks.append(text[pos : pos + step])
                pos += step
            yield f"random{i}", chunks

    @pytest.mark.parametrize("case", sorted(STREAM_CASES))
    def test_streaming_matches_non_streaming_for_every_chunking(self, case):
        text = STREAM_CASES[case]
        expected = Phi4MiniToolParser().extract_tool_calls(text)
        for label, chunks in self._chunkings(text):
            content, calls = _stream(chunks)
            assert [calls[i]["name"] for i in sorted(calls)] == [
                c["name"] for c in expected.tool_calls
            ], label
            assert sorted(calls) == list(range(len(expected.tool_calls))), label
            assert [calls[i]["arguments"] for i in sorted(calls)] == [
                c["arguments"] for c in expected.tool_calls
            ], label
            assert content.strip() == (expected.content or ""), label

    def test_call_is_emitted_only_when_its_list_closes(self):
        text = _text(_call("get_weather", {"city": "Paris"}))
        parser = Phi4MiniToolParser()
        previous = ""
        results = []
        for ch in text:
            current = previous + ch
            results.append(parser.extract_tool_calls_streaming(previous, current, ch))
            previous = current
        assert all(r is None for r in results[:-1])
        assert results[-1]["tool_calls"][0]["index"] == 0
        assert results[-1]["tool_calls"][0]["function"]["name"] == "get_weather"

    def test_parallel_calls_keep_absolute_indices_across_lists(self):
        text = _text(_call("a", {}), _call("b", {})) + "\n" + _text(_call("c", {}))
        _, calls = _stream(_chars(text))
        assert [calls[i]["name"] for i in sorted(calls)] == ["a", "b", "c"]

    def test_marker_split_across_deltas_never_leaks_as_prose(self):
        content, calls = _stream(["Okay. func", "tools", '[{"name": "f"', "}]"])
        assert content == "Okay. "
        assert [c["name"] for c in calls.values()] == ["f"]

    def test_prose_is_streamed_before_the_call(self):
        parser = Phi4MiniToolParser()
        first = parser.extract_tool_calls_streaming("", "Hello there", "Hello there")
        assert first == {"content": "Hello there"}

    def test_unclosed_list_is_flushed_as_content_at_the_end(self):
        text = 'Hmm functools[{"name": "f", "arguments": {"a": '
        content, calls = _stream(_chars(text))
        assert calls == {}
        assert content == text

    def test_trailing_marker_prefix_is_flushed_at_the_end(self):
        content, calls = _stream(_chars("Costs 5 func"))
        assert calls == {}
        assert content == "Costs 5 func"

    def test_reset_clears_stream_state(self):
        text = _text(_call("a", {}))
        parser = Phi4MiniToolParser()
        previous = ""
        for ch in text:
            parser.extract_tool_calls_streaming(previous, previous + ch, ch)
            previous += ch
        parser.reset()
        previous = ""
        seen = []
        for ch in text:
            r = parser.extract_tool_calls_streaming(previous, previous + ch, ch)
            previous += ch
            if r:
                seen.append(r)
        assert len(seen) == 1
        assert seen[0]["tool_calls"][0]["index"] == 0
