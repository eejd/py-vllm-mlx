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
