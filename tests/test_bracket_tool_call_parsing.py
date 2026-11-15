"""Bracket-style ``[Calling tool: ...]`` recovery in the generic parser."""

import json

from vllm_mlx.api.tool_calling import parse_tool_calls
from vllm_mlx.tool_parsers.gemma4_tool_parser import Gemma4ToolParser

RAW_NEWLINE_CALL = (
    '[Calling tool: exec_command({"cmd":"echo \\"def add(a, b):\n'
    '    return a + b\\" > calc.py"})]'
)


def test_raw_newline_inside_json_string_is_recovered():
    cleaned, calls = parse_tool_calls(RAW_NEWLINE_CALL)

    assert [c.function.name for c in calls] == ["exec_command"]
    args = json.loads(calls[0].function.arguments)
    assert args["cmd"] == 'echo "def add(a, b):\n    return a + b" > calc.py'
    assert cleaned == ""


def test_invalid_json_still_yields_no_call():
    _, calls = parse_tool_calls('[Calling tool: f({"a": })]')
    assert not calls


def test_gemma4_parser_alone_does_not_parse_bracket_text():
    assert Gemma4ToolParser().extract_tool_calls(RAW_NEWLINE_CALL).tools_called is False
