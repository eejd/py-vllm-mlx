# SPDX-License-Identifier: Apache-2.0
"""Several Mistral calls in one streamed delta are several calls, not one.

A delta that carries more than one ``[TOOL_CALLS]`` (several tokens at once) used to fold
every call after the first into the first call's arguments (``{...}\\n[TOOL_CALLS]f[ARGS]
{...}``): one call with arguments that are not JSON.
"""

import json

import pytest

from vllm_mlx.tool_parsers import ToolParserManager

A = '[TOOL_CALLS]first[ARGS]{"x": 1}'
B = '[TOOL_CALLS]second[ARGS]{"y": "[TOOL_CALLS]kept"}'
C = '[TOOL_CALLS]third[ARGS]{"z": [1, 2]}'


def _stream(chunks):
    parser = ToolParserManager.get_tool_parser("mistral")()
    parser.reset()
    seen, calls = "", {}
    for delta in chunks:
        previous, seen = seen, seen + delta
        result = parser.extract_tool_calls_streaming(previous, seen, delta)
        for tc in (result or {}).get("tool_calls") or []:
            slot = calls.setdefault(tc["index"], {"id": None, "name": "", "args": ""})
            slot["id"] = slot["id"] or tc.get("id")
            slot["name"] += tc["function"].get("name") or ""
            slot["args"] += tc["function"].get("arguments") or ""
    return calls


@pytest.mark.parametrize(
    "chunks",
    [
        [A + B + C],
        [A, B + C],
        [A + B, C],
        [A[:12], A[12:] + B[:12], B[12:] + C],
        [A + "\n" + B + "\n" + C],
    ],
    ids=["one-delta", "first-then-two", "two-then-one", "split", "newlines"],
)
def test_every_call_of_a_delta_gets_its_own_index(chunks):
    calls = _stream(chunks)

    assert list(calls) == [0, 1, 2]
    assert [c["name"] for c in calls.values()] == ["first", "second", "third"]
    assert [json.loads(c["args"]) for c in calls.values()] == [
        {"x": 1},
        {"y": "[TOOL_CALLS]kept"},  # a marker inside a JSON string is data
        {"z": [1, 2]},
    ]
    assert all(c["id"] for c in calls.values())
    assert len({c["id"] for c in calls.values()}) == 3
