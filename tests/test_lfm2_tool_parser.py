# SPDX-License-Identifier: Apache-2.0
"""Tests for the Liquid LFM2 / LFM2.5 tool call parser."""

import json

from vllm_mlx.tool_parsers import ToolParserManager
from vllm_mlx.tool_parsers.lfm2_tool_parser import (
    TOOL_CALL_END,
    TOOL_CALL_START,
    Lfm2ToolParser,
)


def _block(payload: str) -> str:
    return f"{TOOL_CALL_START}{payload}{TOOL_CALL_END}"


def _args(call: dict) -> dict:
    return json.loads(call["arguments"])


class TestLfm2Registration:
    def test_registered_names(self):
        for name in ("lfm2", "lfm2.5"):
            assert ToolParserManager.get_tool_parser(name) is Lfm2ToolParser

    def test_declares_streaming_marker_and_native_format(self):
        assert Lfm2ToolParser.STREAMING_MARKERS == (TOOL_CALL_START,)
        assert Lfm2ToolParser.supports_native_format()
        # An ``_END`` attribute would make the server drop text after a call.
        assert not hasattr(Lfm2ToolParser, "_END")


class TestLfm2Extract:
    def setup_method(self):
        self.parser = Lfm2ToolParser()

    def test_single_call(self):
        result = self.parser.extract_tool_calls(_block("[get_weather(city='Paris')]"))
        assert result.tools_called is True
        assert [c["name"] for c in result.tool_calls] == ["get_weather"]
        assert _args(result.tool_calls[0]) == {"city": "Paris"}
        assert result.content is None

    def test_parallel_calls_in_one_block(self):
        result = self.parser.extract_tool_calls(
            _block("[get_weather(city='Paris'), get_weather(city='Tokyo')]")
        )
        assert [_args(c) for c in result.tool_calls] == [
            {"city": "Paris"},
            {"city": "Tokyo"},
        ]

    def test_calls_in_separate_blocks(self):
        text = _block("[a(x=1)]") + "\n" + _block("[b(y=2)]")
        result = self.parser.extract_tool_calls(text)
        assert [c["name"] for c in result.tool_calls] == ["a", "b"]

    def test_text_around_the_block_is_content(self):
        result = self.parser.extract_tool_calls(
            "Checking. " + _block("[f(x=1)]") + " Done."
        )
        assert result.tools_called is True
        assert result.content == "Checking.  Done."

    def test_escaped_quote_and_newline(self):
        result = self.parser.extract_tool_calls(
            _block("[say(text='it\\'s a\\nnew line')]")
        )
        assert _args(result.tool_calls[0]) == {"text": "it's a\nnew line"}

    def test_double_quoted_string(self):
        result = self.parser.extract_tool_calls(_block('[say(text="hi")]'))
        assert _args(result.tool_calls[0]) == {"text": "hi"}

    def test_scalars_and_python_literals(self):
        result = self.parser.extract_tool_calls(
            _block("[f(n=3, r=0.5, neg=-2, on=True, off=False, none=None)]")
        )
        assert _args(result.tool_calls[0]) == {
            "n": 3,
            "r": 0.5,
            "neg": -2,
            "on": True,
            "off": False,
            "none": None,
        }

    def test_json_literals_inside_nested_values(self):
        # The template renders mappings/lists through ``tojson``.
        result = self.parser.extract_tool_calls(
            _block('[f(opts={"a": [1, null, true], "b": false}, tags=["x", "y"])]')
        )
        assert _args(result.tool_calls[0]) == {
            "opts": {"a": [1, None, True], "b": False},
            "tags": ["x", "y"],
        }

    def test_no_arguments(self):
        result = self.parser.extract_tool_calls(_block("[now()]"))
        assert result.tool_calls[0]["name"] == "now"
        assert _args(result.tool_calls[0]) == {}

    def test_dotted_function_name(self):
        result = self.parser.extract_tool_calls(_block("[math.add(a=1, b=2)]"))
        assert result.tool_calls[0]["name"] == "math.add"

    def test_unicode_is_not_escaped(self):
        result = self.parser.extract_tool_calls(_block("[f(city='Zürich')]"))
        assert "Zürich" in result.tool_calls[0]["arguments"]

    def test_bare_call_without_list_brackets(self):
        result = self.parser.extract_tool_calls(_block("get_weather(city='Paris')"))
        assert result.tools_called is True
        assert _args(result.tool_calls[0]) == {"city": "Paris"}

    def test_think_block_is_stripped(self):
        result = self.parser.extract_tool_calls(
            "<think>plan the call</think>" + _block("[f(x=1)]")
        )
        assert result.tools_called is True
        assert result.content is None

    def test_missing_end_marker_still_parses_complete_output(self):
        result = self.parser.extract_tool_calls(TOOL_CALL_START + "[f(x=1)]")
        assert result.tools_called is True

    def test_plain_text_is_not_a_call(self):
        result = self.parser.extract_tool_calls("Hello there.")
        assert result.tools_called is False
        assert result.tool_calls == []
        assert result.content == "Hello there."

    def test_list_in_prose_without_markers_is_not_a_call(self):
        result = self.parser.extract_tool_calls("[f(x=1)] is how you call it")
        assert result.tools_called is False


class TestLfm2Malformed:
    def setup_method(self):
        self.parser = Lfm2ToolParser()

    def _assert_visible_not_called(self, payload: str):
        result = self.parser.extract_tool_calls(_block(payload))
        assert result.tools_called is False
        assert payload.strip() in (result.content or "")

    def test_positional_arguments_rejected(self):
        self._assert_visible_not_called("[f(1)]")

    def test_star_star_kwargs_rejected(self):
        self._assert_visible_not_called("[f(**{'a': 1})]")

    def test_non_literal_value_rejected(self):
        self._assert_visible_not_called("[f(x=open('/etc/passwd'))]")

    def test_bare_identifier_value_rejected(self):
        self._assert_visible_not_called("[f(x=Paris)]")

    def test_syntax_error_rejected(self):
        self._assert_visible_not_called("[f(x=1")

    def test_one_bad_call_rejects_the_block(self):
        self._assert_visible_not_called("[f(x=1), g(2)]")

    def test_empty_block(self):
        result = self.parser.extract_tool_calls(_block(""))
        assert result.tools_called is False

    def test_a_bad_block_does_not_hide_a_good_one(self):
        text = _block("[f(1)]") + _block("[g(x=1)]")
        result = self.parser.extract_tool_calls(text)
        assert [c["name"] for c in result.tool_calls] == ["g"]


class TestLfm2Streaming:
    def setup_method(self):
        self.parser = Lfm2ToolParser()
        self.parser.reset()

    def _feed(self, previous: str, delta: str):
        return self.parser.extract_tool_calls_streaming(
            previous_text=previous,
            current_text=previous + delta,
            delta_text=delta,
            request={"tools": []},
        )

    def test_plain_text_passes_through(self):
        assert self._feed("", "Hello") == {"content": "Hello"}

    def test_buffers_inside_an_open_block(self):
        assert self._feed("Sure. ", TOOL_CALL_START + "[get_weather(ci") is None
        assert self._feed(TOOL_CALL_START + "[get_weather(ci", "ty='Pa") is None

    def test_prose_sharing_a_delta_with_the_start_marker_is_kept(self):
        assert self._feed("", "Sure. " + TOOL_CALL_START + "[f(") == {
            "content": "Sure. "
        }

    def test_emits_on_end_marker(self):
        previous = TOOL_CALL_START + "[get_weather(city='Paris')]"
        result = self._feed(previous, TOOL_CALL_END)
        calls = result["tool_calls"]
        assert [c["index"] for c in calls] == [0]
        assert calls[0]["function"]["name"] == "get_weather"
        assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Paris"}

    def test_parallel_calls_in_one_block_get_consecutive_indexes(self):
        previous = TOOL_CALL_START + "[a(x=1), b(y=2)]"
        result = self._feed(previous, TOOL_CALL_END)
        assert [c["index"] for c in result["tool_calls"]] == [0, 1]

    def test_second_block_continues_the_index(self):
        first = _block("[a(x=1), b(y=2)]")
        second_open = first + TOOL_CALL_START + "[c(z=3)]"
        result = self._feed(second_open, TOOL_CALL_END)
        assert [(c["index"], c["function"]["name"]) for c in result["tool_calls"]] == [
            (2, "c")
        ]

    def test_whitespace_between_blocks_is_dropped(self):
        assert self._feed(_block("[a(x=1)]"), "\n") is None

    def test_prose_after_the_last_block_passes_through(self):
        assert self._feed(_block("[a(x=1)]"), "Done.") == {"content": "Done."}

    def test_end_marker_and_trailing_text_in_one_delta(self):
        previous = TOOL_CALL_START + "[a(x=1)]"
        result = self._feed(previous, TOOL_CALL_END + "Done.")
        assert [c["index"] for c in result["tool_calls"]] == [0]
        assert result["content"] == "Done."

    def test_malformed_closed_block_surfaces_as_content(self):
        previous = TOOL_CALL_START + "[f(1)]"
        assert self._feed(previous, TOOL_CALL_END) == {"content": "[f(1)]"}

    def test_end_marker_split_across_deltas_emits_once(self):
        text = _block("[a(x=1)]")
        emitted = 0
        acc = ""
        for ch in text:
            result = self._feed(acc, ch)
            acc += ch
            emitted += len((result or {}).get("tool_calls", []))
        assert emitted == 1


class TestLfm2Finalize:
    def setup_method(self):
        self.parser = Lfm2ToolParser()

    def test_unclosed_complete_block_is_emitted(self):
        result = self.parser.finalize_streaming(TOOL_CALL_START + "[f(x=1)]")
        assert [c["index"] for c in result["tool_calls"]] == [0]

    def test_unclosed_block_continues_the_index(self):
        text = _block("[a(x=1)]") + TOOL_CALL_START + "[b(y=2)]"
        result = self.parser.finalize_streaming(text)
        assert [(c["index"], c["function"]["name"]) for c in result["tool_calls"]] == [
            (1, "b")
        ]

    def test_truncated_block_is_not_emitted(self):
        assert self.parser.finalize_streaming(TOOL_CALL_START + "[f(x=") is None

    def test_closed_blocks_are_left_alone(self):
        assert self.parser.finalize_streaming(_block("[a(x=1)]")) is None

    def test_plain_text_is_left_alone(self):
        assert self.parser.finalize_streaming("hello") is None
