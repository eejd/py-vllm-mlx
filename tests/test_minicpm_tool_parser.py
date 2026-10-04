# SPDX-License-Identifier: Apache-2.0
"""Tests for the MiniCPM5 tool call parser."""

import json

from vllm_mlx.tool_parsers import ToolParserManager
from vllm_mlx.tool_parsers.minicpm_tool_parser import MiniCPMToolParser


def _call(name: str, **params: str) -> str:
    inner = "".join(f'<param name="{k}">{v}</param>' for k, v in params.items())
    return f'<function name="{name}">{inner}</function>'


def _args(call: dict) -> dict:
    return json.loads(call["arguments"])


def _request(name: str, **types: str) -> dict:
    return {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "parameters": {
                        "type": "object",
                        "properties": {k: {"type": t} for k, t in types.items()},
                    },
                },
            }
        ]
    }


class TestMiniCPMRegistration:
    def test_registered_names(self):
        for name in ("minicpm", "minicpm5"):
            assert ToolParserManager.get_tool_parser(name) is MiniCPMToolParser

    def test_declares_streaming_marker_and_native_format(self):
        assert MiniCPMToolParser.STREAMING_MARKERS == ("<function",)
        assert MiniCPMToolParser.supports_native_format()
        assert not hasattr(MiniCPMToolParser, "_END")


class TestMiniCPMExtract:
    def setup_method(self):
        self.parser = MiniCPMToolParser()

    def test_single_call(self):
        result = self.parser.extract_tool_calls(_call("get_weather", city="Paris"))
        assert result.tools_called is True
        assert result.tool_calls[0]["name"] == "get_weather"
        assert _args(result.tool_calls[0]) == {"city": "Paris"}
        assert result.content is None

    def test_parallel_calls(self):
        text = (
            _call("get_weather", city="Paris")
            + "\n"
            + _call("get_weather", city="Tokyo")
        )
        result = self.parser.extract_tool_calls(text)
        assert [_args(c) for c in result.tool_calls] == [
            {"city": "Paris"},
            {"city": "Tokyo"},
        ]

    def test_no_params(self):
        result = self.parser.extract_tool_calls('<function name="now"></function>')
        assert _args(result.tool_calls[0]) == {}

    def test_text_around_calls_is_content(self):
        text = "Checking. " + _call("f", x="1") + " Done."
        result = self.parser.extract_tool_calls(text)
        assert result.content == "Checking.  Done."

    def test_cdata_value_is_kept_verbatim(self):
        text = (
            '<function name="f"><param name="code">'
            "<![CDATA[if a < b && c:\n  return 1 ]]>"
            "</param></function>"
        )
        result = self.parser.extract_tool_calls(text)
        assert _args(result.tool_calls[0]) == {"code": "if a < b && c:\n  return 1 "}

    def test_multiline_plain_value(self):
        text = '<function name="f"><param name="t">line 1\nline 2</param></function>'
        result = self.parser.extract_tool_calls(text)
        assert _args(result.tool_calls[0]) == {"t": "line 1\nline 2"}

    def test_unicode_is_not_escaped(self):
        result = self.parser.extract_tool_calls(_call("f", city="Zürich"))
        assert "Zürich" in result.tool_calls[0]["arguments"]

    def test_think_block_is_stripped(self):
        text = "<think>\n\n</think>\n\n" + _call("f", x="1")
        result = self.parser.extract_tool_calls(text)
        assert result.tools_called is True
        assert result.content is None

    def test_stray_legacy_tool_call_tokens_are_dropped(self):
        text = "<tool_call>" + _call("f", x="1") + "</tool_call>"
        result = self.parser.extract_tool_calls(text)
        assert result.tools_called is True
        assert result.content is None

    def test_plain_text_is_not_a_call(self):
        result = self.parser.extract_tool_calls("Hello there.")
        assert result.tools_called is False
        assert result.content == "Hello there."

    def test_unclosed_element_is_not_a_complete_call(self):
        result = self.parser.extract_tool_calls('<function name="f"><param name="x">1')
        assert result.tools_called is False


class TestMiniCPMTyping:
    def setup_method(self):
        self.parser = MiniCPMToolParser()

    def test_schema_types_values(self):
        req = _request(
            "f",
            s="string",
            n="integer",
            r="number",
            b="boolean",
            a="array",
            o="object",
        )
        text = _call("f", s="123", n="42", r="0.5", b="true", a="[1, 2]", o='{"k": 1}')
        result = self.parser.extract_tool_calls(text, req)
        assert _args(result.tool_calls[0]) == {
            "s": "123",
            "n": 42,
            "r": 0.5,
            "b": True,
            "a": [1, 2],
            "o": {"k": 1},
        }

    def test_string_schema_keeps_json_looking_text(self):
        req = _request("f", s="string")
        result = self.parser.extract_tool_calls(_call("f", s="true"), req)
        assert _args(result.tool_calls[0]) == {"s": "true"}

    def test_python_repr_values_from_template_replay(self):
        req = _request("f", b="boolean", o="object", a="array", n="null")
        text = _call(
            "f", b="True", o="{'a': [1, None, True]}", a="['x', 'y']", n="None"
        )
        result = self.parser.extract_tool_calls(text, req)
        assert _args(result.tool_calls[0]) == {
            "b": True,
            "o": {"a": [1, None, True]},
            "a": ["x", "y"],
            "n": None,
        }

    def test_without_a_schema_values_are_parsed_then_fall_back_to_text(self):
        text = _call("f", n="3", w="Paris", p="123 Main St")
        result = self.parser.extract_tool_calls(text)
        assert _args(result.tool_calls[0]) == {"n": 3, "w": "Paris", "p": "123 Main St"}

    def test_union_schema_with_string_and_number_is_parsed(self):
        req = {
            "tools": [
                {
                    "function": {
                        "name": "f",
                        "parameters": {
                            "properties": {
                                "v": {"anyOf": [{"type": "string"}, {"type": "number"}]}
                            }
                        },
                    }
                }
            ]
        }
        result = self.parser.extract_tool_calls(_call("f", v="5"), req)
        assert _args(result.tool_calls[0]) == {"v": 5}


class TestMiniCPMStreaming:
    def setup_method(self):
        self.parser = MiniCPMToolParser()
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

    def test_buffers_inside_an_open_element(self):
        assert self._feed("Sure. ", '<function name="f"><param name="x">') is None
        assert self._feed('<function name="f"><param name="x">', "1</param>") is None

    def test_prose_sharing_a_delta_with_the_opening_tag_is_kept(self):
        assert self._feed("", 'Sure. <function name="f">') == {"content": "Sure. "}

    def test_emits_when_the_closing_tag_arrives(self):
        previous = '<function name="f"><param name="x">1</param>'
        result = self._feed(previous, "</function>")
        calls = result["tool_calls"]
        assert [c["index"] for c in calls] == [0]
        assert calls[0]["function"]["name"] == "f"
        assert json.loads(calls[0]["function"]["arguments"]) == {"x": 1}

    def test_second_element_continues_the_index(self):
        previous = _call("a", x="1") + '<function name="b"><param name="y">2</param>'
        result = self._feed(previous, "</function>")
        assert [(c["index"], c["function"]["name"]) for c in result["tool_calls"]] == [
            (1, "b")
        ]

    def test_whitespace_between_elements_is_dropped(self):
        assert self._feed(_call("a", x="1"), "\n") is None

    def test_prose_after_the_last_element_passes_through(self):
        assert self._feed(_call("a", x="1"), "Done.") == {"content": "Done."}

    def test_closing_tag_and_trailing_text_in_one_delta(self):
        previous = '<function name="a"><param name="x">1</param>'
        result = self._feed(previous, "</function>Done.")
        assert [c["index"] for c in result["tool_calls"]] == [0]
        assert result["content"] == "Done."

    def test_closing_tag_split_across_deltas_emits_once(self):
        text = _call("a", x="1")
        emitted = 0
        acc = ""
        for ch in text:
            result = self._feed(acc, ch)
            acc += ch
            emitted += len((result or {}).get("tool_calls", []))
        assert emitted == 1


class TestMiniCPMFinalize:
    def setup_method(self):
        self.parser = MiniCPMToolParser()

    def test_unclosed_element_with_closed_params_is_emitted(self):
        text = '<function name="f"><param name="x">1</param>'
        result = self.parser.finalize_streaming(text)
        assert [c["index"] for c in result["tool_calls"]] == [0]

    def test_unclosed_element_continues_the_index(self):
        text = _call("a", x="1") + '<function name="b"><param name="y">2</param>'
        result = self.parser.finalize_streaming(text)
        assert [(c["index"], c["function"]["name"]) for c in result["tool_calls"]] == [
            (1, "b")
        ]

    def test_truncated_param_is_not_emitted(self):
        text = '<function name="f"><param name="x">1'
        assert self.parser.finalize_streaming(text) is None

    def test_closed_elements_are_left_alone(self):
        assert self.parser.finalize_streaming(_call("a", x="1")) is None

    def test_plain_text_is_left_alone(self):
        assert self.parser.finalize_streaming("hello") is None


class TestMiniCPMCdataBoundaries:
    """A CDATA value may contain the tags that would otherwise end an element."""

    def setup_method(self):
        self.parser = MiniCPMToolParser()

    def test_cdata_value_containing_the_function_close_tag(self):
        text = (
            '<function name="run"><param name="code">'
            '<![CDATA[x = "</function>"]]></param></function>'
        )
        result = self.parser.extract_tool_calls(text)
        assert _args(result.tool_calls[0]) == {"code": 'x = "</function>"'}
        assert result.content is None

    def test_cdata_value_containing_the_param_close_tag(self):
        text = (
            '<function name="run"><param name="code">'
            '<![CDATA[x = "</param>"]]></param></function>'
        )
        result = self.parser.extract_tool_calls(text)
        assert _args(result.tool_calls[0]) == {"code": 'x = "</param>"'}

    def test_two_cdata_params_in_one_element(self):
        text = (
            '<function name="f"><param name="a"><![CDATA[1 < 2]]></param>'
            '<param name="b"><![CDATA[</function>]]></param></function>'
        )
        result = self.parser.extract_tool_calls(text)
        assert _args(result.tool_calls[0]) == {"a": "1 < 2", "b": "</function>"}

    def test_unterminated_cdata_does_not_hang_or_call(self):
        text = '<function name="f"><param name="a">' + "<![CDATA[x " * 200
        result = self.parser.extract_tool_calls(text)
        assert result.tools_called is False

    def test_streaming_a_cdata_value_with_a_close_tag_emits_one_call(self):
        text = (
            '<function name="run"><param name="code">'
            '<![CDATA[x = "</function>"]]></param></function>'
        )
        for size in (1, 5, 10_000):
            self.parser.reset()
            acc, calls = "", []
            for i in range(0, len(text), size):
                delta = text[i : i + size]
                previous, acc = acc, acc + delta
                out = self.parser.extract_tool_calls_streaming(previous, acc, delta)
                calls += (out or {}).get("tool_calls", [])
            assert [c["index"] for c in calls] == [0], size
            assert json.loads(calls[0]["function"]["arguments"]) == {
                "code": 'x = "</function>"'
            }


class TestMiniCPMStrayClosingTags:
    """A closing tag with no element must not shift the index of later calls."""

    def setup_method(self):
        self.parser = MiniCPMToolParser()

    def _stream(self, text, size):
        self.parser.reset()
        acc, calls = "", []
        for i in range(0, len(text), size):
            delta = text[i : i + size]
            previous, acc = acc, acc + delta
            out = self.parser.extract_tool_calls_streaming(previous, acc, delta)
            calls += (out or {}).get("tool_calls", [])
        return calls

    def test_stray_close_before_a_call(self):
        text = "</function>stray" + _call("get_weather", city="Paris")
        for size in (1, 3, 7, 50, 10_000):
            calls = self._stream(text, size)
            assert [c["index"] for c in calls] == [0], size
            assert calls[0]["function"]["name"] == "get_weather"

    def test_two_stray_closes_before_a_call(self):
        text = "</function></function>" + _call("f", x="1") + _call("g", y="2")
        for size in (1, 4, 10_000):
            calls = self._stream(text, size)
            assert [(c["index"], c["function"]["name"]) for c in calls] == [
                (0, "f"),
                (1, "g"),
            ], size

    def test_stray_close_is_not_content_in_the_non_streaming_result(self):
        result = self.parser.extract_tool_calls("</function>" + _call("f", x="1"))
        assert result.tools_called is True
        assert result.content is None


class TestMiniCPMFinalizeAmbiguity:
    def setup_method(self):
        self.parser = MiniCPMToolParser()

    def test_two_open_elements_are_not_merged_into_one_call(self):
        text = (
            '<function name="f"><param name="x">1</param>'
            '<function name="g"><param name="y">2</param>'
        )
        assert self.parser.finalize_streaming(text) is None

    def test_unclosed_cdata_is_not_emitted(self):
        text = '<function name="f"><param name="x"><![CDATA[abc'
        assert self.parser.finalize_streaming(text) is None


class TestMiniCPMProseAfterACall:
    def test_word_separators_after_a_call_survive_token_sized_deltas(self):
        parser = MiniCPMToolParser()
        parser.reset()
        text = _call("f", x="1") + " Mid " + _call("g", y="2") + " End."
        for size in (1, 2, 3):
            parser.reset()
            acc, content, calls = "", "", []
            for i in range(0, len(text), size):
                delta = text[i : i + size]
                previous, acc = acc, acc + delta
                out = parser.extract_tool_calls_streaming(previous, acc, delta)
                content += (out or {}).get("content", "")
                calls += (out or {}).get("tool_calls", [])
            assert [c["index"] for c in calls] == [0, 1], size
            assert content.split() == ["Mid", "End."], (size, content)
