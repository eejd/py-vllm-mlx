# SPDX-License-Identifier: Apache-2.0
"""A final ``<tool_call>`` block the model never closed must still be a tool call.

Small Qwen3 models sometimes end a multi-call response with the last call's JSON and
``<|im_end|>``, skipping ``</tool_call>`` (eejd/py-vllm-mlx#21). The qwen and hermes parsers
used to require the closing tag and dropped that call, so the response under-called.
"""

import json

import pytest

from vllm_mlx import server
from vllm_mlx.tool_parsers.hermes_tool_parser import HermesToolParser
from vllm_mlx.tool_parsers.qwen_tool_parser import QwenToolParser

PARSERS = [QwenToolParser, HermesToolParser]

FASTA = '{"name": "fetch", "arguments": {"id": "X1", "format": "fasta"}}'
GENBANK = '{"name": "fetch", "arguments": {"id": "X1", "format": "genbank"}}'
UPSTREAM = '{"name": "fetch", "arguments": {"id": "X1", "upstream": 500}}'


def _closed(body: str) -> str:
    return f"<tool_call>\n{body}\n</tool_call>"


# The raw output from parallel_106 in #21: two closed blocks, then an unterminated one.
PARALLEL_106 = "\n".join([_closed(FASTA), _closed(GENBANK), f"<tool_call>\n{UPSTREAM}"])


@pytest.fixture(params=PARSERS, ids=lambda c: c.__name__)
def parser(request):
    return request.param()


class TestNonStreaming:
    def test_unterminated_last_block_is_kept(self, parser):
        result = parser.extract_tool_calls(PARALLEL_106)
        assert result.tools_called
        assert [json.loads(c["arguments"]) for c in result.tool_calls] == [
            {"id": "X1", "format": "fasta"},
            {"id": "X1", "format": "genbank"},
            {"id": "X1", "upstream": 500},
        ]
        assert not result.content

    def test_single_unterminated_block(self, parser):
        result = parser.extract_tool_calls(f"<tool_call>\n{FASTA}")
        assert result.tools_called and len(result.tool_calls) == 1

    def test_unterminated_block_with_invalid_json_is_dropped(self, parser):
        text = (
            _closed(FASTA) + '\n<tool_call>\n{"name": "fetch", "arguments": {"id": "X1"'
        )
        result = parser.extract_tool_calls(text)
        assert [json.loads(c["arguments"]) for c in result.tool_calls] == [
            {"id": "X1", "format": "fasta"}
        ]

    def test_truncated_midway_is_not_a_call(self, parser):
        result = parser.extract_tool_calls('<tool_call>\n{"name": "fetch", "argum')
        assert not result.tool_calls

    def test_hermes_prose_after_json_is_kept_by_the_lenient_fallback(self):
        # Not the new \\Z alternative: hermes' pre-existing TOOL_CALL_LENIENT_PATTERN
        # accepts a complete {"name", "arguments"} object whatever follows it.
        result = HermesToolParser().extract_tool_calls(
            f"<tool_call>\n{FASTA}\nand some prose"
        )
        assert len(result.tool_calls) == 1

    def test_two_unclosed_blocks_yield_no_calls(self, parser):
        # Limitation: only the LAST block may be unterminated. With no closing tag at all,
        # the first block's lazy match runs to the final "}" and is not valid JSON.
        result = parser.extract_tool_calls(
            f"<tool_call>\n{FASTA}\n<tool_call>\n{GENBANK}"
        )
        assert not result.tool_calls

    def test_closed_blocks_unchanged(self, parser):
        text = "Sure.\n" + _closed(FASTA) + "\n" + _closed(GENBANK)
        result = parser.extract_tool_calls(text)
        assert len(result.tool_calls) == 2
        assert result.content == "Sure."

    def test_qwen_strips_unparsed_unclosed_markup(self):
        text = (
            _closed(FASTA) + '\n<tool_call>\n{"name": "fetch", "arguments": {"id": "X1"'
        )
        assert "<tool_call>" not in (
            QwenToolParser().extract_tool_calls(text).content or ""
        )

    def test_qwen_block_must_run_to_the_end_of_the_output(self):
        result = QwenToolParser().extract_tool_calls(
            f"<tool_call>\n{FASTA}\nand some prose"
        )
        assert not result.tool_calls


def _stream(parser, text: str, size: int) -> tuple[dict[int, dict], dict | None]:
    """Drive the parser as the server does: per delta, then the end-of-stream hook."""
    parser.reset()
    calls: dict[int, dict] = {}
    emissions: dict[int, int] = {}
    acc = ""
    last = None
    for i in range(0, len(text), size):
        delta = text[i : i + size]
        prev, acc = acc, acc + delta
        last = parser.extract_tool_calls_streaming(
            previous_text=prev,
            current_text=acc,
            delta_text=delta,
            request={"tools": []},
        )
        _collect(last, calls, emissions)
    # Same decision as the server's end-of-stream hook.
    final = None
    if server._should_finalize_tool_stream(parser, last):
        # merged with None so only what the hook itself adds is collected
        final = server._finalize_streaming_tool_result(parser, acc, None)
        _collect(final, calls, emissions)
    assert all(n == 1 for n in emissions.values()), emissions
    return calls, final


def _collect(result, calls, emissions):
    for tc in (result or {}).get("tool_calls") or []:
        i = tc["index"]
        emissions[i] = emissions.get(i, 0) + 1
        slot = calls.setdefault(i, {"name": "", "arguments": ""})
        slot["name"] += tc["function"].get("name") or ""
        slot["arguments"] += tc["function"].get("arguments") or ""


class TestStreaming:
    @pytest.mark.parametrize("size", [1, 5, 17, 10_000])
    def test_unterminated_last_block_emitted_once(self, parser, size):
        calls, _ = _stream(parser, PARALLEL_106, size)
        assert sorted(calls) == [0, 1, 2]
        assert [json.loads(calls[i]["arguments"])["id"] for i in sorted(calls)] == [
            "X1"
        ] * 3
        assert json.loads(calls[2]["arguments"]) == {"id": "X1", "upstream": 500}

    @pytest.mark.parametrize("size", [1, 5, 10_000])
    def test_single_unterminated_block(self, parser, size):
        calls, final = _stream(parser, f"<tool_call>\n{FASTA}", size)
        assert sorted(calls) == [0]
        assert final is not None  # nothing was emitted before the end of the stream

    @pytest.mark.parametrize("size", [1, 7, 10_000])
    def test_closed_blocks_need_no_finalize(self, parser, size):
        calls, final = _stream(parser, _closed(FASTA) + "\n" + _closed(GENBANK), size)
        assert sorted(calls) == [0, 1]
        assert final is None

    def test_invalid_json_tail_emits_nothing_extra(self, parser):
        text = _closed(FASTA) + '\n<tool_call>\n{"name": "fetch", "arguments": {"id": '
        calls, _ = _stream(parser, text, 4)
        assert sorted(calls) == [0]
