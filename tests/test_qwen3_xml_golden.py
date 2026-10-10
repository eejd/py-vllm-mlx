"""Qwen3XMLToolParser against the shared golden corpus (py-quantbench parser_golden.jsonl)."""

import hashlib
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("transformers")
sys.path.insert(0, str(Path(__file__).parent))

from parser_golden_support import DATA, check_case, load_cases  # noqa: E402

from vllm_mlx.tool_parsers.qwen3_xml_tool_parser import (  # noqa: E402
    Qwen3XMLToolParser,
    repair_json_arguments,
)

PINNED_SHA256 = "cdeb531c941bd95caf2d64f995dc35c6a26e2a58041c17d601487ea79b099735"
CASES = load_cases()


def test_corpus_is_the_pinned_one():
    assert hashlib.sha256(Path(DATA).read_bytes()).hexdigest() == PINNED_SHA256


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
def test_non_streaming_extraction(case):
    result = Qwen3XMLToolParser(None).extract_tool_calls(
        case["text"], {"tools": case["tools"]}
    )
    calls = [
        {"name": c["name"], "arguments": c["arguments"]} for c in result.tool_calls
    ]
    assert check_case(case, calls, result.content) == []


def test_truncated_call_becomes_content_not_a_call():
    case = next(c for c in CASES if c["id"] == "truncated_mid_value")
    result = Qwen3XMLToolParser(None).extract_tool_calls(
        case["text"], {"tools": case["tools"]}
    )
    assert not result.tools_called and "Tok" in result.content


def test_parse_error_is_logged_once_per_stream(caplog):
    text = (
        "<tool_call>\n<function=get_weather>\n<parameter=city>\nTokyo\n<parameter=days>\n3\n"
        "</parameter>\n</function>\n</tool_call>"
    )
    parser = Qwen3XMLToolParser(None)
    with caplog.at_level("WARNING"):
        parser.extract_tool_calls(text)
    logged = [r for r in caplog.records if "parsing XML elements" in r.getMessage()]
    # The unpatched parser logged this dropped closer three times (measured on 0.5.0-local4).
    assert len(logged) == 1


@pytest.mark.parametrize(
    "raw,expected",
    [
        ('{"a": 1}', '{"a": 1}'),
        ('{"a": 1', '{"a": 1}'),
        ('{"a": {"b": 1}', '{"a": {"b": 1}}'),
        ('{"a": "tru', None),
        ('{"a": [1, 2', None),
        ("[1]", None),
        ("", None),
    ],
)
def test_repair_json_arguments(raw, expected):
    assert repair_json_arguments(raw) == expected


def test_server_coerce_closes_missing_braces_and_never_completes_values():
    from vllm_mlx.server import _coerce_tool_arguments

    assert json.loads(
        _coerce_tool_arguments('{"city": "Tokyo"', "get_weather", None)
    ) == {"city": "Tokyo"}
    assert _coerce_tool_arguments('{"city": "To', "get_weather", None) == '{"city": "To'
    assert (
        _coerce_tool_arguments('{"city": "Tokyo"}', "get_weather", None)
        == '{"city": "Tokyo"}'
    )


def _extract(text):
    tools = CASES[0]["tools"]
    return Qwen3XMLToolParser(None).extract_tool_calls(text, {"tools": tools})


def test_only_the_truncated_last_call_is_dropped():
    text = (
        "<tool_call>\n<function=get_weather>\n<parameter=city>\nTokyo\n</parameter>\n"
        "</function>\n</tool_call>\n<tool_call>\n<function=read_file>\n<parameter=path>\n/a.p"
    )
    result = _extract(text)
    assert [c["name"] for c in result.tool_calls] == ["get_weather"]
    assert json.loads(result.tool_calls[0]["arguments"]) == {"city": "Tokyo"}
    assert "read_file" in result.content and "/a.p" in result.content


def test_second_parameter_cut_off_drops_the_call():
    text = (
        "<tool_call>\n<function=get_weather>\n<parameter=city>\nTokyo\n</parameter>\n"
        "<parameter=days>\n3"
    )
    result = _extract(text)
    assert not result.tools_called and "days" in result.content


def test_prose_mentioning_function_tag_falls_back_to_valid_only():
    text = (
        "Use <function=nothing> as the marker.\n<tool_call>\n<function=get_weather>\n"
        "<parameter=city>\nTokyo\n</parameter>\n</function>\n</tool_call>"
    )
    result = _extract(text)
    assert [json.loads(c["arguments"]) for c in result.tool_calls] == [
        {"city": "Tokyo"}
    ]
