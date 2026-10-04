# SPDX-License-Identifier: Apache-2.0
"""Streaming must agree with non-streaming, however the output is cut into deltas.

Random (seeded) model outputs in each parser's real format are streamed at several chunk
sizes and compared with ``extract_tool_calls`` on the whole text: the same calls, in the
same order, each emitted exactly once under a contiguous absolute index.
"""

import json
import random

import pytest

from vllm_mlx.tool_parsers.lfm2_tool_parser import Lfm2ToolParser
from vllm_mlx.tool_parsers.minicpm_tool_parser import MiniCPMToolParser

_PROSE = ["", "Sure. ", "Let me check.\n", "Done."]
_WORDS = ["Paris", "it's", 'say "hi"', "a b", "Zürich", "x</function>y", "1 < 2"]


def _lfm_value(rng):
    kind = rng.choice(["str", "int", "bool", "nested"])
    if kind == "int":
        return str(rng.randint(-5, 500))
    if kind == "bool":
        return rng.choice(["True", "False", "None"])
    if kind == "nested":
        return '{"a": [1, null, true], "b": "x"}'
    word = rng.choice([w for w in _WORDS if "<" not in w])
    return "'" + word.replace("\\", "\\\\").replace("'", "\\'") + "'"


def _lfm_output(rng):
    out = rng.choice(_PROSE)
    for _ in range(rng.randint(1, 3)):
        calls = [
            f"f{rng.randint(0, 3)}("
            + ", ".join(f"k{j}={_lfm_value(rng)}" for j in range(rng.randint(0, 3)))
            + ")"
            for _ in range(rng.randint(1, 3))
        ]
        out += f"<|tool_call_start|>[{', '.join(calls)}]<|tool_call_end|>"
        out += rng.choice(_PROSE)
    return out


def _minicpm_output(rng):
    out = rng.choice(_PROSE)
    for _ in range(rng.randint(1, 4)):
        params = ""
        for j in range(rng.randint(0, 3)):
            word = rng.choice(_WORDS)
            value = f"<![CDATA[{word}]]>" if ("<" in word or "&" in word) else word
            if "\n" in word or '"' in word:
                value = f"<![CDATA[{word}]]>"
            params += f'<param name="k{j}">{value}</param>'
        out += f'<function name="f{rng.randint(0, 3)}">{params}</function>'
        out += rng.choice(["", "\n", " "])
    return out + rng.choice(_PROSE)


CASES = {
    "lfm2": (Lfm2ToolParser, _lfm_output),
    "minicpm": (MiniCPMToolParser, _minicpm_output),
}


def _stream_calls(parser, text, size):
    parser.reset()
    calls, seen, acc = {}, {}, ""
    for i in range(0, len(text), size):
        delta = text[i : i + size]
        previous, acc = acc, acc + delta
        out = parser.extract_tool_calls_streaming(previous, acc, delta)
        for tc in (out or {}).get("tool_calls", []):
            slot = calls.setdefault(tc["index"], {"name": "", "arguments": ""})
            slot["name"] += tc["function"]["name"] or ""
            slot["arguments"] += tc["function"]["arguments"] or ""
            seen[tc["index"]] = seen.get(tc["index"], 0) + 1
    return calls, seen


@pytest.mark.parametrize("size", [1, 2, 5, 13, 64, 10_000])
@pytest.mark.parametrize("name", sorted(CASES))
def test_streamed_calls_equal_the_non_streaming_calls(name, size):
    parser_cls, make = CASES[name]
    rng = random.Random(f"{name}-{size}")
    for _ in range(40):
        text = make(rng)
        expected = parser_cls().extract_tool_calls(text).tool_calls
        calls, seen = _stream_calls(parser_cls(), text, size)

        assert sorted(calls) == list(range(len(expected))), (text, calls)
        assert all(n == 1 for n in seen.values()), (text, seen)
        assert [
            (calls[i]["name"], json.loads(calls[i]["arguments"])) for i in sorted(calls)
        ] == [(c["name"], json.loads(c["arguments"])) for c in expected], text
