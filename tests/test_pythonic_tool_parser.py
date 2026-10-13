# SPDX-License-Identifier: Apache-2.0
"""The ``pythonic`` tool parser: ``[fn(k=v), ...]`` answers (Llama 3.2/4, Gemma 3).

The accepted/rejected sets are those of vLLM's ``pythonic`` parser and its shared
helpers (tests/tool_parsers/test_pythonic_tool_parser.py and test_utils.py in vLLM
v0.31); the places where this parser deliberately differs are pinned in
``TestDifferencesFromVllm``.
"""

import json
import random
import time

import pytest

from vllm_mlx.tool_parsers import ToolParserManager
from vllm_mlx.tool_parsers.pythonic_tool_parser import (
    PythonicToolParser,
    _INCOMPLETE,
    _INVALID,
    _ListScanner,
)

# vLLM's fixtures (tests/tool_parsers/test_pythonic_tool_parser.py)
SIMPLE_OUTPUT = "get_weather(city='San Francisco', metric='celsius')"
SIMPLE_ARGS = {"city": "San Francisco", "metric": "celsius"}
MORE_TYPES_OUTPUT = (
    "register_user(name='John Doe', age=37, "
    "address={'city': 'San Francisco', 'state': 'CA'}, role=None, "
    "passed_test=True, aliases=['John', 'Johnny'])"
)
MORE_TYPES_ARGS = {
    "name": "John Doe",
    "age": 37,
    "address": {"city": "San Francisco", "state": "CA"},
    "role": None,
    "passed_test": True,
    "aliases": ["John", "Johnny"],
}
ESCAPED_OUTPUT = r"get_weather(city='Martha\'s Vineyard', metric='\"cool units\"')"
ESCAPED_ARGS = {"city": "Martha's Vineyard", "metric": '"cool units"'}

CHUNK_SIZES = [1, 2, 3, 7, 64, 100_000]


def _parser() -> PythonicToolParser:
    parser = ToolParserManager.get_tool_parser("pythonic")()
    parser.reset()
    return parser


def _non_streaming(text):
    result = _parser().extract_tool_calls(text)
    calls = [(c["name"], json.loads(c["arguments"])) for c in result.tool_calls]
    assert result.tools_called == bool(calls)
    return calls, result.content


def _stream(text, size, parser=None):
    """Stream ``text`` in ``size``-character deltas the way the server does.

    Returns ``(calls, content, emissions)``: calls by absolute index, the content
    handed out, and how many times each index was emitted.
    """
    parser = parser or _parser()
    acc, calls, emissions, content = "", {}, {}, []
    deltas = [text[i : i + size] for i in range(0, len(text), size)] or [""]
    for n, delta in enumerate(deltas):
        previous, acc = acc, acc + delta
        outs = [parser.extract_tool_calls_streaming(previous, acc, delta)]
        if n == len(deltas) - 1:
            outs.append(parser.finalize_streaming(acc))
        for out in outs:
            for tc in (out or {}).get("tool_calls", []):
                emissions[tc["index"]] = emissions.get(tc["index"], 0) + 1
                assert tc["type"] == "function" and tc["id"]
                calls[tc["index"]] = (
                    tc["function"]["name"],
                    json.loads(tc["function"]["arguments"]),
                )
            content.append((out or {}).get("content") or "")
    return calls, "".join(content), emissions


def _both(text, size):
    """Assert streaming agrees with non-streaming on ``text``; return the result."""
    expected_calls, expected_content = _non_streaming(text)
    calls, content, emissions = _stream(text, size)
    assert [calls[i] for i in sorted(calls)] == expected_calls, text
    assert sorted(calls) == list(range(len(expected_calls))), text
    assert all(n == 1 for n in emissions.values()), (text, emissions)
    assert content == (expected_content or ""), text
    return expected_calls, expected_content


class TestRegistration:
    def test_names_resolve_to_one_class(self):
        for name in ("pythonic", "gemma3_pythonic"):
            assert ToolParserManager.get_tool_parser(name) is PythonicToolParser

    def test_server_contract_flags(self):
        # No marker exists, so the server must hand every delta over and finalize at
        # the end of every response; the shipped template renders tool turns itself.
        assert PythonicToolParser.REQUIRES_EAGER_STREAMING is True
        assert PythonicToolParser.supports_native_format() is True


@pytest.mark.parametrize("size", CHUNK_SIZES)
class TestVllmFixtures:
    @pytest.mark.parametrize("text", ["How can I help you today?", "", "Hello [world]"])
    def test_no_tool_call_is_content_unchanged(self, text, size):
        calls, content = _both(text, size)
        assert calls == [] and content == text

    @pytest.mark.parametrize(
        "output, args",
        [
            (SIMPLE_OUTPUT, SIMPLE_ARGS),
            (MORE_TYPES_OUTPUT, MORE_TYPES_ARGS),
            ("get_weather()", {}),
            ("do_something_cool(additional_data={})", {"additional_data": {}}),
            ("do_something_cool(steps=[])", {"steps": []}),
            (ESCAPED_OUTPUT, ESCAPED_ARGS),
        ],
        ids=[
            "simple",
            "more_types",
            "parameterless",
            "empty_dict",
            "empty_list",
            "esc",
        ],
    )
    def test_single_call(self, output, args, size):
        calls, content = _both(f"[{output}]", size)
        assert calls[0][1] == args
        assert content is None

    def test_parallel_calls(self, size):
        calls, content = _both(f"[{SIMPLE_OUTPUT}, {MORE_TYPES_OUTPUT}]", size)
        assert [c[1] for c in calls] == [SIMPLE_ARGS, MORE_TYPES_ARGS]
        assert content is None

    def test_arguments_are_json_text_in_declaration_order(self, size):
        result = _parser().extract_tool_calls(f"[{MORE_TYPES_OUTPUT}]")
        assert result.tool_calls[0]["arguments"] == (
            '{"name": "John Doe", "age": 37, "address": {"city": "San Francisco", '
            '"state": "CA"}, "role": null, "passed_test": true, '
            '"aliases": ["John", "Johnny"]}'
        )


class TestValues:
    """vLLM's ``get_parameter_value`` and the JSON spellings it accepts."""

    @pytest.mark.parametrize(
        "source, expected",
        [
            ("-1", -1),
            ("+1", 1),
            ("-1.5", -1.5),
            ("- -2", 2),
            ("[-1, -2.5]", [-1, -2.5]),
            ("{'lon': -122.4}", {"lon": -122.4}),
            ("true", True),
            ("false", False),
            ("null", None),
            (
                "{'a': [1, null, true], 'b': {'c': false}}",
                {"a": [1, None, True], "b": {"c": False}},
            ),
            ("(800, 600)", [800, 600]),
            ("{'b', 'a'}", ["b", "a"]),  # source order, not sorted
            ("f'hello'", "hello"),
            ("f''", ""),
            ("1_000", 1000),
            ("0x10", 16),
            ("1e3", 1000.0),
            ("{2: 'a', None: 'b', True: 'c'}", {"2": "a", "null": "b", "true": "c"}),
            ("'tab\\there'", "tab\there"),
            ("'über'", "über"),
            ('"it\'s"', "it's"),
            ("'''multi\nline'''", "multi\nline"),
            ("'a' 'b'", "ab"),
        ],
    )
    @pytest.mark.parametrize("size", [1, 100_000])
    def test_accepted(self, source, expected, size):
        calls, content = _both(f"[f(x={source})]", size)
        assert calls == [("f", {"x": expected})]
        assert content is None

    @pytest.mark.parametrize(
        "source",
        [
            "b'bytes'",
            "1+2j",
            "...",
            "some_name",
            "1 + 2",
            "not True",
            "~1",
            "-'a'",
            "-True",
            "f'{x}'",
            "f'a{1}'",
            "{'a': 1, **other}",
            "{[1]: 2}",
            "{b'k': 1}",
            "[x for x in y]",
            "lambda: 1",
            "g()",
            "a.b",
            "1e999",
            "-1e999",
            "[1, 1e999]",
            "{'k': float('nan')}",
        ],
    )
    @pytest.mark.parametrize("size", [1, 100_000])
    def test_rejected_value_leaves_the_whole_response_as_content(self, source, size):
        text = f"[ok(a=1), f(x={source})]"
        calls, content = _both(text, size)
        assert calls == [] and content == text

    def test_non_finite_numbers_never_reach_arguments(self):
        # vLLM would put ``Infinity`` (not JSON) into ``arguments``
        assert _non_streaming("[f(x=1e999)]") == ([], "[f(x=1e999)]")

    def test_duplicate_dict_keys_keep_the_last(self):
        assert _non_streaming("[f(x={'a': 1, 'a': 2})]")[0] == [("f", {"x": {"a": 2}})]

    def test_repeated_keyword_keeps_the_last(self):
        # Python's parser accepts it (only the compiler objects), as in vLLM
        assert _both("[f(k=1, k=2)]", 1)[0] == [("f", {"k": 2})]

    def test_non_ascii_is_not_escaped(self):
        result = _parser().extract_tool_calls("[f(city='Zürich')]")
        assert result.tool_calls[0]["arguments"] == '{"city": "Zürich"}'

    def test_large_integers_survive(self):
        big = 2**70
        assert _non_streaming(f"[f(n={big})]")[0] == [("f", {"n": big})]


class TestCallShape:
    @pytest.mark.parametrize("size", [1, 100_000])
    @pytest.mark.parametrize(
        "name", ["get_weather", "a.b.c", "_private", "módulo.función", "A1_b2", "x . y"]
    )
    def test_names_and_dotted_names(self, name, size):
        calls, _ = _both(f"[{name}(k=1)]", size)
        assert calls == [(name.replace(" ", ""), {"k": 1})]

    @pytest.mark.parametrize("size", [1, 100_000])
    @pytest.mark.parametrize(
        "text",
        [
            "[f(1)]",  # positional
            "[f('a', k=1)]",
            "[f(k=1, 2)]",  # a positional after a keyword is a SyntaxError anyway
            "[f(*args)]",
            "[f(**kwargs)]",
            "[f(k=1, **kwargs)]",
            "[f(from=1)]",  # reserved word as keyword
            "[]",
            "[ ]",
            "[1, 2, 3]",
            "['a']",
            "[f]",
            "[f(), 5]",
            "[f(), g]",
            "[f()(k=1)]",
            "[f(k=1)(k=2)]",
            "[a[0](k=1)]",
            "[(lambda: 1)()]",
            "[1(k=1)]",
            "[f() g()]",  # no comma
            "[f(),, g()]",
            "[,f()]",
            "[f(k=1]",
            "[f(k=1))]",
            "[f(k=[1)]",
            "[f(k=1), ",
            "[if(k=1)]",
            "[f(k='unterminated\n')]",
            "[f(k=1)\x00]",
            "[f(k=1)\u00a0, g()]",  # NBSP is not Python whitespace
        ],
    )
    def test_not_a_call_list_is_content(self, text, size):
        calls, content = _both(text, size)
        assert calls == [] and content == text

    @pytest.mark.parametrize("size", [1, 5, 100_000])
    @pytest.mark.parametrize(
        "text",
        [
            "[f(a = 1)]",
            "[ f(a=1) ]",
            "[\n  f(a=1),\n  g(b=2)\n]",
            "[f(a=\n1)]",
            "[f(a=1),]",  # trailing comma is legal Python
            "[f(a=1) , g(b=2)]",
            "[f (a=1)]",
            "[f(a=1)]\n",
            "[f(a=1)]   ",
            "[f(a=1)]\r\n",
            "[f(a=1)\f]",
            "[f(\ta=1,\tb=[\n1,\n2,\n])]",
        ],
    )
    def test_python_formatting_is_accepted(self, text, size):
        calls, content = _both(text, size)
        assert [c[0] for c in calls] and content is None

    @pytest.mark.parametrize("size", [1, 100_000])
    @pytest.mark.parametrize(
        "text",
        [
            " [f()]",
            "\n[f()]",
            "Sure! [f()]",
            "```\n[f()]\n```",
            "(f(),)",
            "f()",
            "{f()}",
        ],
    )
    def test_only_a_leading_bracket_starts_a_call_list(self, text, size):
        calls, content = _both(text, size)
        assert calls == [] and content == text

    def test_every_call_gets_its_own_id(self):
        result = _parser().extract_tool_calls("[a(x=1), a(x=1)]")
        ids = [c["id"] for c in result.tool_calls]
        assert len(set(ids)) == 2 and all(i.startswith("call_") for i in ids)


class TestBracketsInStrings:
    @pytest.mark.parametrize("size", [1, 2, 100_000])
    @pytest.mark.parametrize("quote", ["single", "double"])
    @pytest.mark.parametrize(
        "value",
        [
            "a)]b",
            "[(",
            '"',
            "it's",
            "''",
            "\\",  # a backslash before the closing quote must be escaped, not eaten
            "a\\\\",
            "}{",
            "x'''y",
            'x"""y',
            "\\n\\t",
        ],
    )
    def test_brackets_and_quotes_inside_strings_are_text(self, value, quote, size):
        # json.dumps writes a double-quoted literal Python reads back identically;
        # for the single-quoted form escape what must be escaped by hand.
        if quote == "double":
            literal = json.dumps(value)
        else:
            literal = "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"
        calls, content = _both(f"[f(x={literal}, y=[{literal}]), g(z={literal})]", size)
        assert calls == [
            ("f", {"x": value, "y": [value]}),
            ("g", {"z": value}),
        ]
        assert content is None

    @pytest.mark.parametrize("size", [1, 2, 3, 100_000])
    def test_triple_quoted_strings(self, size):
        text = "[f(x='''a'b\"c''', y=\"\"\"d''e\"\"\", z='', w=\"\")]"
        calls, _ = _both(text, size)
        assert calls == [("f", {"x": "a'b\"c", "y": "d''e", "z": "", "w": ""})]

    @pytest.mark.parametrize("size", [1, 2, 3])
    def test_empty_string_followed_by_a_comma_is_not_a_triple_quote(self, size):
        calls, _ = _both("[f(a='', b='')]", size)
        assert calls == [("f", {"a": "", "b": ""})]


class TestTextAfterTheList:
    """Only the leading list is the call; what follows is content."""

    @pytest.mark.parametrize("size", [1, 4, 100_000])
    def test_prose_after_the_list_is_content(self, size):
        calls, content = _both("[f(a=1)]\n\nLet me know.", size)
        assert calls == [("f", {"a": 1})]
        assert content == "Let me know."

    @pytest.mark.parametrize("size", [1, 4, 100_000])
    def test_whitespace_after_the_list_is_nothing(self, size):
        calls, content = _both("[f(a=1)] \n\t ", size)
        assert calls == [("f", {"a": 1})] and content is None

    @pytest.mark.parametrize("size", [1, 3, 100_000])
    def test_a_second_list_is_content_not_more_calls(self, size):
        calls, content = _both("[f(a=1)][g(b=2)]", size)
        assert calls == [("f", {"a": 1})]
        assert content == "[g(b=2)]"


class TestStreaming:
    @pytest.mark.parametrize("size", [1, 5, 100_000])
    def test_prose_streams_through_unchanged(self, size):
        text = "I could not find a function for that.\n  Anything else?"
        calls, content, _ = _stream(text, size)
        assert calls == {} and content == text

    def test_a_list_is_withheld_until_it_closes_then_emitted_once(self):
        parser = _parser()
        text = "[get_weather(city='Paris'), get_time(tz='UTC')]"
        acc, outs = "", []
        for ch in text:
            previous, acc = acc, acc + ch
            outs.append(parser.extract_tool_calls_streaming(previous, acc, ch))
        assert all(o is None for o in outs[:-1])
        assert [tc["index"] for tc in outs[-1]["tool_calls"]] == [0, 1]
        assert [tc["function"]["name"] for tc in outs[-1]["tool_calls"]] == [
            "get_weather",
            "get_time",
        ]
        assert parser.finalize_streaming(acc) is None

    @pytest.mark.parametrize("size", [1, 3, 100_000])
    def test_numbers_split_at_the_decimal_point_are_whole(self, size):
        calls, _ = _both("[f(lat=37.7749, lon=-122.4194, n=[1.5, 2.25])]", size)
        assert calls == [("f", {"lat": 37.7749, "lon": -122.4194, "n": [1.5, 2.25]})]

    @pytest.mark.parametrize("size", [1, 4, 100_000])
    def test_a_bracket_text_that_is_not_a_call_list_is_not_swallowed(self, size):
        text = "[1] is the first item, [2] the second."
        calls, content, _ = _stream(text, size)
        assert calls == {} and content == text

    @pytest.mark.parametrize("size", [1, 4, 100_000])
    def test_prose_release_is_immediate_once_it_cannot_be_a_call(self, size):
        parser = _parser()
        acc, got = "", []
        text = "[Note: this is prose and keeps going for a while"
        for i in range(0, len(text), size):
            delta = text[i : i + size]
            previous, acc = acc, acc + delta
            out = parser.extract_tool_calls_streaming(previous, acc, delta)
            got.append((out or {}).get("content") or "")
        # released at the ``:``, not at the end of the response
        assert "".join(got) == text
        assert got[-1] != "" or size == 1

    @pytest.mark.parametrize(
        "text",
        [
            "[",
            "[f",
            "[f(",
            "[f(x=",
            "[f(x='abc",
            "[f(x=1)",
            "[f(x=1),",
            "[f(x=1), g(y=2",
            "[f(x=1), g(",
            "[f(x='''a",
            "[f(x=''",
        ],
    )
    @pytest.mark.parametrize("size", [1, 100_000])
    def test_a_list_the_model_never_closed_is_released_as_content_at_the_end(
        self, text, size
    ):
        calls, content, _ = _stream(text, size)
        assert calls == {}
        assert content == text
        assert _non_streaming(text) == ([], text)

    def test_finalize_is_idempotent_and_silent_after_a_complete_response(self):
        parser = _parser()
        text = "[f(x=1)]"
        parser.extract_tool_calls_streaming("", text, text)
        assert parser.finalize_streaming(text) is None
        assert parser.finalize_streaming(text) is None

    def test_finalize_resolves_a_list_the_last_delta_completed(self):
        # the server may finalize a response whose last delta was never routed here
        parser = _parser()
        out = parser.finalize_streaming("[f(x=1)] done")
        assert [tc["function"]["name"] for tc in out["tool_calls"]] == ["f"]
        assert out["content"] == "done"

    def test_reset_starts_a_new_response(self):
        parser = _parser()
        _stream("[f(x=1)]", 3, parser)
        parser.reset()
        calls, content, _ = _stream("plain", 2, parser)
        assert calls == {} and content == "plain"
        parser.reset()
        calls, _, _ = _stream("[g(y=2)]", 2, parser)
        assert calls == {0: ("g", {"y": 2})}

    def test_split_utf8_boundaries_are_not_special(self):
        calls, content = _both("[f(x='日本語の文字列')]", 1)
        assert calls == [("f", {"x": "日本語の文字列"})]


class TestDifferencesFromVllm:
    """What vLLM's parser gets wrong or leaves undecided, and this one settles."""

    def test_positional_argument_is_rejected_in_streaming_too(self):
        # vLLM streams ``[f(1)]`` as a call to ``f`` with ``{}``
        assert _stream("[f(1)]", 1)[:2] == ({}, "[f(1)]")

    def test_dotted_names_are_calls_in_both_modes(self):
        # vLLM's regex rejects them in non-streaming and its streaming path accepts them
        assert _both("[a.b.c(x=1)]", 1)[0] == [("a.b.c", {"x": 1})]

    def test_non_streaming_does_not_drop_text_after_the_list(self):
        # vLLM returns the call and drops ``Hello`` (a second statement is ignored)
        assert _non_streaming("[f(a=1)]\n\nHello")[1] == "Hello"

    def test_regex_backtracking_input_costs_nothing(self):
        # vLLM's regex needs a timeout for this (VLLM_TOOL_PARSE_REGEX_TIMEOUT_SECONDS)
        text = "[A(A=" + "\t)A(A=,\t" * 400
        started = time.perf_counter()
        assert _non_streaming(text) == ([], text)
        assert _stream(text, 7)[1] == text
        assert time.perf_counter() - started < 2


class TestScanner:
    def test_statuses(self):
        scanner = _ListScanner()
        assert scanner.advance("") == _INCOMPLETE
        assert scanner.advance("[f(") == _INCOMPLETE
        assert scanner.advance("[f(a=1)") == _INCOMPLETE
        assert scanner.advance("[f(a=1)]") == "complete"
        assert scanner.end == len("[f(a=1)]")
        assert scanner.calls == [("f", '{"a": 1}')]
        assert scanner.advance("[f(a=1)] tail") == "complete"  # sticky
        assert _ListScanner().advance("x") == _INVALID

    def test_ambiguous_quotes_at_the_end_wait_for_the_next_character(self):
        scanner = _ListScanner()
        for prefix in ("[f(x='", "[f(x=''", "[f(x='''"):
            assert scanner.advance(prefix) == _INCOMPLETE
        # ``''`` + more is an empty string, ``'''`` + more opens a long string
        assert _ListScanner().advance("[f(x='', y=1)]") == "complete"
        assert _ListScanner().advance("[f(x='''a''', y=1)]") == "complete"
        assert _ListScanner().advance("[f(x=''')]") == _INCOMPLETE

    @pytest.mark.filterwarnings("ignore::SyntaxWarning")
    def test_matches_python_on_mutated_lists(self):
        """The scanner finds the end of the list where ``ast`` does, on every mutation."""
        import ast

        rng = random.Random(7)
        seeds = [
            "[f(a=1, b='x)]', c=[1, {'k': (2, 3)}]), g.h(d=None)]",
            "[f(a='''t\"w'o''', b=\"\")]",
            "[f(a={'a': [1, 2], 'b': {3}}), g()]",
        ]
        glyphs = "()[]{}'\",=: \n\\#.x1"
        checked = 0
        for _ in range(3000):
            text = list(rng.choice(seeds))
            for _ in range(rng.randint(0, 3)):
                text[rng.randrange(len(text))] = rng.choice(glyphs)
            text = "".join(text)
            expected_end = None
            for end in range(1, len(text) + 1):
                if text[end - 1] != "]":
                    continue
                try:
                    node = ast.parse(text[:end], mode="eval").body
                except (SyntaxError, ValueError):
                    continue
                if (
                    isinstance(node, ast.List)
                    and node.elts
                    and all(
                        isinstance(e, ast.Call)
                        and not e.args
                        and not any(k.arg is None for k in e.keywords)
                        for e in node.elts
                    )
                ):
                    expected_end = end
                break
            scanner = _ListScanner()
            status = scanner.advance(text)
            if expected_end is None:
                assert status != "complete", text
            else:
                # arguments the converter rejects (names, calls) are not calls here
                if status == "complete":
                    assert scanner.end == expected_end, text
            checked += 1
        assert checked == 3000


def _random_output(rng):
    words = [
        "Paris",
        "it's",
        'say "hi"',
        "a ] b",
        "(x)",
        "Zürich",
        "a\\b",
        "line\nbreak",
    ]

    def literal():
        kind = rng.choice(["str", "int", "float", "bool", "none", "list", "dict"])
        if kind == "str":
            return repr(rng.choice(words))
        if kind == "int":
            return str(rng.randint(-50, 5000))
        if kind == "float":
            return repr(round(rng.uniform(-200, 200), rng.randint(1, 6)))
        if kind == "bool":
            return rng.choice(["True", "False", "true", "false"])
        if kind == "none":
            return rng.choice(["None", "null"])
        if kind == "list":
            return (
                "["
                + ", ".join(repr(rng.choice(words)) for _ in range(rng.randint(0, 3)))
                + "]"
            )
        return (
            "{"
            + ", ".join(
                f"{rng.choice(words)!r}: {rng.randint(0, 9)}"
                for _ in range(rng.randint(0, 2))
            )
            + "}"
        )

    calls = [
        f"{rng.choice(['f', 'get_weather', 'a.b'])}("
        + ", ".join(f"k{j}={literal()}" for j in range(rng.randint(0, 3)))
        + ")"
        for _ in range(rng.randint(1, 3))
    ]
    out = "[" + rng.choice([", ", ",\n  "]).join(calls) + "]"
    out += rng.choice(["", "", "\n", " Done.", "\n\nThanks!"])
    return out


@pytest.mark.parametrize("size", [1, 2, 3, 5, 13, 64, 10_000])
def test_streaming_equals_non_streaming_on_random_outputs(size):
    rng = random.Random(size)
    for _ in range(60):
        text = _random_output(rng)
        if rng.random() < 0.4:  # break it somewhere: still the same on both sides
            cut = rng.randrange(len(text))
            text = rng.choice([text[:cut], text[:cut] + text[cut + 1 :]])
        _both(text, size)


class TestCost:
    def test_prose_costs_constant_time_per_delta(self):
        parser = _parser()
        text = "word " * 40_000
        acc = ""
        started = time.perf_counter()
        for i in range(0, len(text), 3):
            delta = text[i : i + 3]
            previous, acc = acc, acc + delta
            parser.extract_tool_calls_streaming(previous, acc, delta)
        assert time.perf_counter() - started < 5

    def test_a_huge_argument_streams_in_linear_time(self):
        body = ("x'] y\\' " * 20_000)[:150_000]
        text = "[write_file(path='a', content=" + repr(body) + ")]"
        started = time.perf_counter()
        calls, content, _ = _stream(text, 3)
        assert calls[0][1]["content"] == body
        assert content == ""
        # re-reading the open call on every delta would take minutes
        assert time.perf_counter() - started < 10
