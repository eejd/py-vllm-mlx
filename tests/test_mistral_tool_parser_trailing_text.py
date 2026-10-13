# SPDX-License-Identifier: Apache-2.0
"""Mistral tool parser: text the model writes after a call's arguments.

A Mistral v11+ call (``[TOOL_CALLS]name[ARGS]{...}``) has no terminator of its own, so a
model may follow the closing brace with ordinary text. Seen live from
mlx-community/Ministral-3-3B-Instruct-2512-4bit (temperature 0, a system prompt plus the
tool-echo ``code_value`` request)::

    [TOOL_CALLS]echo_args[ARGS]{"text": "def f(x):\\n    return x + 1", "count": 0}

    The Python code has been recorded as two lines with proper indentation.

That text is not part of the arguments. Upstream vLLM keeps the call and drops the text
(vllm-project/vllm#48975). Before, this server

- dropped the whole call in a complete response (the arguments "were not JSON") and
  returned the raw ``[TOOL_CALLS]`` markup as content with ``finish_reason: stop``, and
- streamed the text as part of ``function.arguments`` (not JSON) with
  ``finish_reason: tool_calls``,

so the same generation produced different answers with and without ``stream``.
"""

import json
import random
import re

import pytest

from vllm_mlx.tool_parsers.mistral_tool_parser import MistralToolParser

CALL = '[TOOL_CALLS]echo_args[ARGS]{"text": "def f(x):\\n    return x + 1", "count": 0}'
PROSE = "\n\nThe Python code has been recorded as two lines with proper indentation."

_ATOMS = re.compile(r"\[TOOL_CALLS\]|\[ARGS\]|.", re.DOTALL)


def _deltas(text: str, size: int | None, seed: int = 0) -> list[str]:
    """Split ``text`` into deltas of ``size`` atoms (random 1..size when ``seed``
    varies); ``[TOOL_CALLS]`` and ``[ARGS]`` stay whole, they are single tokens.
    ``None`` is one delta for the whole text."""
    atoms = _ATOMS.findall(text)
    if size is None:
        return ["".join(atoms)]
    rng = random.Random(seed)
    out, i = [], 0
    while i < len(atoms):
        step = rng.randint(1, size)
        out.append("".join(atoms[i : i + step]))
        i += step
    return out


def _stream(deltas: list[str]) -> tuple[list[dict], str]:
    """Replay ``deltas``; return the calls (name, arguments, id count) and content."""
    parser = MistralToolParser()
    calls: dict[int, dict] = {}
    content = ""
    current = ""
    for delta in deltas:
        previous, current = current, current + delta
        result = parser.extract_tool_calls_streaming(previous, current, delta)
        if not result:
            continue
        content += result.get("content", "")
        for tc in result.get("tool_calls", []):
            call = calls.setdefault(
                tc["index"], {"name": "", "arguments": "", "ids": []}
            )
            call["name"] += tc["function"].get("name", "")
            call["arguments"] += tc["function"].get("arguments", "")
            if "id" in tc:
                call["ids"].append(tc["id"])
    return [calls[i] for i in sorted(calls)], content


def _complete(text: str):
    return MistralToolParser().extract_tool_calls(text)


class TestCompleteResponse:
    def test_call_followed_by_prose_is_kept(self):
        result = _complete(CALL + PROSE)

        assert result.tools_called
        assert [c["name"] for c in result.tool_calls] == ["echo_args"]
        assert json.loads(result.tool_calls[0]["arguments"]) == {
            "text": "def f(x):\n    return x + 1",
            "count": 0,
        }
        assert result.content is None

    def test_prose_before_the_call_is_still_content(self):
        result = _complete("Recording it now. " + CALL + PROSE)

        assert result.tools_called
        assert result.content == "Recording it now."

    def test_each_call_is_cut_at_the_end_of_its_arguments(self):
        text = (
            '[TOOL_CALLS]get_weather[ARGS]{"city": "Paris"} Done with Paris.\n'
            '[TOOL_CALLS]get_time[ARGS]{"tz": "CET"}\n\nAll done.'
        )

        result = _complete(text)

        assert [c["name"] for c in result.tool_calls] == ["get_weather", "get_time"]
        assert [json.loads(c["arguments"]) for c in result.tool_calls] == [
            {"city": "Paris"},
            {"tz": "CET"},
        ]

    def test_quotes_in_prose_do_not_hide_the_next_call(self):
        """Text after a call's arguments is not JSON: an odd number of double quotes
        in it must not leave a string open that swallows the next marker."""
        text = (
            '[TOOL_CALLS]get_weather[ARGS]{"city": "Paris"} He said "wait.\n'
            '[TOOL_CALLS]get_time[ARGS]{"tz": "CET"}'
        )

        result = _complete(text)

        assert [c["name"] for c in result.tool_calls] == ["get_weather", "get_time"]

    def test_braces_in_prose_are_not_arguments(self):
        result = _complete(CALL + ' Use {"x": 1} next, or a } or {.')

        assert [json.loads(c["arguments"])["count"] for c in result.tool_calls] == [0]

    def test_legacy_name_brace_format_drops_trailing_text(self):
        result = _complete('[TOOL_CALLS]get_weather{"city": "Paris"} Estimating...')

        assert result.tools_called
        assert result.tool_calls[0]["arguments"] == '{"city": "Paris"}'

    def test_legacy_array_format_drops_trailing_text(self):
        text = (
            '[TOOL_CALLS] [{"name": "get_weather", "arguments": {"city": "Paris"}}]'
            " Here is the weather [1]."
        )

        result = _complete(text)

        assert result.tools_called
        assert [c["name"] for c in result.tool_calls] == ["get_weather"]
        assert json.loads(result.tool_calls[0]["arguments"]) == {"city": "Paris"}

    def test_arguments_that_never_close_are_still_rejected(self):
        """Only text *after* a complete value is dropped: a cut-off value is a broken
        call, not a call plus text."""
        assert not _complete('[TOOL_CALLS]echo_args[ARGS]{"text": "abc').tools_called


# Generations whose calls must come out the same with and without streaming.
PARITY_OUTPUTS = {
    "call": CALL,
    "call_then_prose": CALL + PROSE,
    "two_calls_newline": CALL + "\n" + CALL.replace("count\": 0", 'count": 1'),
    "prose_between_calls": (
        '[TOOL_CALLS]a[ARGS]{"x": 1} He said "hold on.\n[TOOL_CALLS]b[ARGS]{"y": 2}'
    ),
    "braces_and_quotes_in_strings": (
        '[TOOL_CALLS]f[ARGS]{"s": "a } b ] c \\" d", "n": {"k": ["v", {"z": null}]}}'
    ),
    "marker_inside_string": '[TOOL_CALLS]f[ARGS]{"s": "[TOOL_CALLS]rm"} trailing',
    "content_then_call_then_prose": "Sure. " + CALL + " Bye.",
    "legacy_brace": '[TOOL_CALLS]get_weather{"city": "Paris"} Estimating...',
    # an escaped quote, then a brace that is still inside the string
    "escaped_quote_then_brace": '[TOOL_CALLS]f[ARGS]{"s": "q\\" } x"} tail',
    # the value is a string: it ends at its closing quote, braces after it are prose
    "top_level_string": '[TOOL_CALLS]f[ARGS]"abc" then {x} and "y',
    "leading_whitespace": '[TOOL_CALLS]f[ARGS] \n\t{"a": 1} tail',
    "bare_scalar": "[TOOL_CALLS]f[ARGS]5 bye",
    # prose with an odd number of quotes after a bare scalar must not hide the marker
    "bare_scalar_then_quoted_prose_then_call": (
        '[TOOL_CALLS]a[ARGS]5 He said "wait.\n[TOOL_CALLS]b[ARGS]{"y": 2}'
    ),
    "bare_scalar_then_call": '[TOOL_CALLS]a[ARGS]true [TOOL_CALLS]b[ARGS]{"y": 2}',
    # a name extract_tool_calls refuses is refused in a stream too
    "name_that_is_not_an_identifier": (
        '[TOOL_CALLS]get weather[ARGS]{"s": 1}\n[TOOL_CALLS]ok[ARGS]{"a": 1}'
    ),
    "legacy_call_without_a_name": '[TOOL_CALLS]{"s": 1}[TOOL_CALLS]ok{"a": 1}',
    "call_without_a_name": '[TOOL_CALLS][ARGS]{"s": 1}[TOOL_CALLS]ok[ARGS]{"a": 1}',
    # a space between the name and [ARGS] does not end the call
    "space_before_args": '[TOOL_CALLS]f [ARGS]{"s": "[TOOL_CALLS]rm"} tail',
    # quote state is per call: the second call's string hides its own marker
    "marker_inside_string_of_second_call": (
        '[TOOL_CALLS]a[ARGS]{"x": 1}\n[TOOL_CALLS]b[ARGS]{"s": "[TOOL_CALLS]rm"} tail'
    ),
}


@pytest.mark.parametrize("name", sorted(PARITY_OUTPUTS))
@pytest.mark.parametrize("size", [None, 1, 2, 3, 7, 40])
def test_streamed_calls_equal_complete_calls(name, size):
    """However the generation is cut into deltas, the streamed calls are the calls of
    the complete response: same names, same arguments, valid JSON, one id each."""
    text = PARITY_OUTPUTS[name]
    expected = _complete(text)
    assert expected.tools_called

    for seed in range(8 if size else 1):
        streamed, content = _stream(_deltas(text, size, seed))

        assert [c["name"] for c in streamed] == [
            c["name"] for c in expected.tool_calls
        ]
        assert [c["arguments"] for c in streamed] == [
            c["arguments"] for c in expected.tool_calls
        ]
        for call in streamed:
            json.loads(call["arguments"])
            assert len(call["ids"]) == 1
        assert content.strip() == (expected.content or "")


def test_two_calls_in_the_delta_that_opens_the_first():
    """The first call's boundary and a second call arriving in one delta: the second
    call needs its own index, not a place in the first call's arguments."""
    streamed, _ = _stream(
        ['[TOOL_CALLS]a[ARGS]{"x": 1}[TOOL_CALLS]b[ARGS]{"y": 2}']
    )

    assert [(c["name"], c["arguments"]) for c in streamed] == [
        ("a", '{"x": 1}'),
        ("b", '{"y": 2}'),
    ]


def test_text_after_the_arguments_is_not_streamed_into_them():
    streamed, content = _stream(["[TOOL_CALLS]f[ARGS]", '{"a": 1}', "\n\n", "Done."])

    assert [c["arguments"] for c in streamed] == ['{"a": 1}']
    assert content == ""


def test_arguments_cut_off_by_the_token_limit_stream_as_they_are():
    streamed, _ = _stream(["[TOOL_CALLS]f[ARGS]", '{"a": "unterminat'])

    assert [c["arguments"] for c in streamed] == ['{"a": "unterminat']


def test_malformed_arguments_are_rejected_in_a_complete_response_not_in_a_stream():
    """The one place the two modes still differ, on purpose. A complete response can
    refuse a call whose arguments are not JSON (this repo's choice; upstream vLLM keeps
    the raw text as the arguments); a stream has already sent the arguments by the
    time it can tell, so it carries them as generated."""
    text = '[TOOL_CALLS]f[ARGS]{"a": 1,}'

    assert not _complete(text).tools_called
    streamed, _ = _stream(_deltas(text, 3))
    assert [c["arguments"] for c in streamed] == ['{"a": 1,}']


def test_a_marker_after_an_unfinished_name_starts_a_new_call():
    """A call whose name never reached its boundary is abandoned, not glued to the
    next call's name ('a[TOOL_CALLS]b')."""
    streamed, _ = _stream(["[TOOL_CALLS]", "a", '[TOOL_CALLS]b[ARGS]{"x": 1}'])

    assert [(c["name"], c["arguments"]) for c in streamed] == [("b", '{"x": 1}')]


def test_text_before_a_marker_in_an_unfinished_name_is_not_content():
    streamed, content = _stream(
        ["[TOOL_CALLS]", "a", 'b[TOOL_CALLS]c[ARGS]{"x": 1}']
    )

    assert [(c["name"], c["arguments"]) for c in streamed] == [("c", '{"x": 1}')]
    assert content == ""


def test_the_first_arguments_text_is_scanned_for_strings():
    """Arguments that arrive with the boundary: a marker inside their open string is
    data, not a forged call."""
    streamed, _ = _stream(
        ["[TOOL_CALLS]", "f", '[ARGS]{"q', '": "[TOOL_CALLS]rm[ARGS]{}"}']
    )

    assert [(c["name"], c["arguments"]) for c in streamed] == [
        ("f", '{"q": "[TOOL_CALLS]rm[ARGS]{}"}')
    ]


def test_a_call_that_is_not_sent_leaves_no_gap_in_the_indices():
    parser = MistralToolParser()
    text = '[TOOL_CALLS]bad name[ARGS]{}[TOOL_CALLS]a[ARGS]{"x": 1}[TOOL_CALLS]b[ARGS]{}'
    indices = []
    current = ""
    for delta in _deltas(text, 3):
        previous, current = current, current + delta
        result = parser.extract_tool_calls_streaming(previous, current, delta)
        indices += [tc["index"] for tc in (result or {}).get("tool_calls", [])]

    assert sorted(set(indices)) == [0, 1]


# --- random generations ------------------------------------------------------------

_PIECES = ["a", "Paris", "[TOOL_CALLS]", "[ARGS]", "}", "{", "]", "[", '"', "\\", "\n"]
_PIECES += ["é世", "😀", " ", "x y", "[TOOL_CALLS]rm", '\\"']
_PROSE = ["", "", " Done.", "\n\nThe result is recorded.", ' He said "wait.']
_PROSE += [' {"x": 1} next', " } or {", "\n", " [1] ", "  ", ' "quoted" ', ' end"']
_NAMES = ["f", "get_weather", "echo_args", "a_b-c.d"]


def _random_value(rng, depth=0):
    kind = rng.random()
    if depth < 2 and kind < 0.15:
        keys = ["k", "text", 'q"x']
        return {
            rng.choice(keys): _random_value(rng, depth + 1)
            for _ in range(rng.randint(0, 3))
        }
    if depth < 2 and kind < 0.3:
        return [_random_value(rng, depth + 1) for _ in range(rng.randint(0, 3))]
    if kind < 0.4:
        return rng.choice([0, 1, -5, 3.5, True, False, None])
    return "".join(rng.choice(_PIECES) for _ in range(rng.randint(0, 5)))


def _random_generation(rng):
    text = rng.choice(["", "", "Sure. ", "Let me check.\n"])
    calls = rng.randint(1, 4)
    for number in range(calls):
        args = {
            rng.choice(["city", "text", "n"]): _random_value(rng)
            for _ in range(rng.randint(0, 3))
        }
        value = json.dumps(args, ensure_ascii=rng.random() < 0.5)
        value = rng.choice(["", "", "", " ", "\n"]) + value
        text += "[TOOL_CALLS]" + rng.choice(_NAMES) + "[ARGS]" + value
        last = number == calls - 1
        text += rng.choice(_PROSE) if last else rng.choice(["", "\n", " "] + _PROSE)
    return text


def test_random_generations_stream_as_they_parse():
    """4000 generations of 1-4 calls (arguments full of markers, braces, quotes and
    escapes; prose before, between and after), each cut into deltas at random with
    ``[TOOL_CALLS]`` and ``[ARGS]`` whole: the streamed calls are the parsed calls."""
    rng = random.Random(20261013)
    for _ in range(4000):
        text = _random_generation(rng)
        expected = _complete(text)
        streamed, content = _stream(_deltas(text, rng.choice([1, 2, 5, 12]), rng.random()))

        assert [(c["name"], c["arguments"]) for c in streamed] == [
            (c["name"], c["arguments"]) for c in expected.tool_calls
        ], text
        assert all(len(c["ids"]) == 1 for c in streamed), text
        assert content.strip() == (expected.content or ""), text
