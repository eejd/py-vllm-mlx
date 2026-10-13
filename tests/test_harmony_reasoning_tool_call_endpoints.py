# SPDX-License-Identifier: Apache-2.0
"""Harmony (gpt-oss) reasoning parsers with the harmony tool parser, end to end.

``--reasoning-parser gpt_oss`` / ``harmony`` together with ``--tool-call-parser
harmony`` / ``gpt-oss``, through the real handlers with a fake engine, on every endpoint,
streaming and not, for several chunkings (py-vllm-mlx#73). The contract is the one of
``test_reasoning_tool_call_contract.py``:

* exactly one tool call, ``finish_reason: tool_calls``;
* the analysis text only in ``reasoning``, no control token anywhere in ``content``;
* a call that is only mentioned inside the analysis channel is not a call.
"""

import json
import re

import pytest
from fastapi.testclient import TestClient

import vllm_mlx.server as srv
from tests.test_reasoning_tool_call_contract import (
    ENDPOINT_BODIES,
    NON_STREAMING_SITES,
    SITES,
    TEXT,
    _engine,
    _run,
    _sse_payloads,
)
from vllm_mlx.reasoning import get_parser
from vllm_mlx.tool_parsers import ToolParserManager

ANALYSIS = "The user wants it. I should call echo_args."
FINAL = "Done, the call went out."
# the model plans the call in the analysis channel, in words
ANALYSIS_MENTIONING_A_CALL = (
    "I could call to=functions.echo_args with " + json.dumps({"text": TEXT}) + " but no."
)
ARGS = json.dumps({"text": TEXT})

_COMMENTARY_CALL = (
    "<|channel|>commentary to=functions.echo_args <|constrain|>json"
    f"<|message|>{ARGS}<|call|>"
)
# <|call|> and <|return|> are stop tokens: a generation ends on one without decoding it
_COMMENTARY_CALL_AT_EOS = _COMMENTARY_CALL.removesuffix("<|call|>")


def _analysis(text: str, end: str = "<|end|>") -> str:
    return f"<|channel|>analysis<|message|>{text}{end}"


def _final(text: str, end: str = "<|return|>") -> str:
    return f"<|start|>assistant<|channel|>final<|message|>{text}{end}"


RAW = {
    # analysis, the call, a final message
    "analysis-call-final": (
        _analysis(ANALYSIS) + "<|start|>assistant" + _COMMENTARY_CALL + _final(FINAL),
        ANALYSIS,
        FINAL,
        1,
    ),
    # analysis straight into the call, no final channel
    "analysis-into-call": (
        _analysis(ANALYSIS) + "<|start|>assistant" + _COMMENTARY_CALL,
        ANALYSIS,
        "",
        1,
    ),
    # the same, ended the way a real generation ends: no closing token
    "analysis-into-call-at-eos": (
        _analysis(ANALYSIS) + "<|start|>assistant" + _COMMENTARY_CALL_AT_EOS,
        ANALYSIS,
        "",
        1,
    ),
    # a preamble message to the user before the call
    "preamble-then-call": (
        _analysis(ANALYSIS)
        + "<|start|>assistant<|channel|>commentary<|message|>Looking it up.<|end|>"
        + "<|start|>assistant"
        + _COMMENTARY_CALL_AT_EOS,
        ANALYSIS,
        "",
        1,
    ),
    # the call without an analysis at all
    "call-only": (_COMMENTARY_CALL, "", "", 1),
    # a call that is only mentioned in the analysis, and a plain final answer
    "call-mentioned-in-analysis": (
        _analysis(ANALYSIS_MENTIONING_A_CALL) + _final(FINAL),
        ANALYSIS_MENTIONING_A_CALL,
        FINAL,
        0,
    ),
    # a plain answer that ends at EOS, without <|return|>
    "final-at-eos": (_analysis(ANALYSIS) + _final(FINAL, end=""), ANALYSIS, FINAL, 0),
    # only the analysis, which mentions a call
    "analysis-only": (
        _analysis(ANALYSIS_MENTIONING_A_CALL),
        ANALYSIS_MENTIONING_A_CALL,
        "",
        0,
    ),
    # a generation cut off inside the analysis
    "cut-off-in-analysis": (
        _analysis(ANALYSIS, end=""),
        ANALYSIS,
        "",
        0,
    ),
    # a generation cut off inside the call's arguments: not a call
    "cut-off-in-call": (
        _analysis(ANALYSIS)
        + "<|start|>assistant"
        + _COMMENTARY_CALL_AT_EOS[: -len('done"}')],
        ANALYSIS,
        "",
        0,
    ),
}

_TOKEN_RE = re.compile(r"<\|[a-z]+\|>|\s+|[^\s<]+|<")


def _chunks(raw: str, chunking: str) -> list[str]:
    tokens = _TOKEN_RE.findall(raw)
    if chunking == "whole":
        return [raw]
    if chunking == "tokens":
        return tokens
    size = int(chunking.removeprefix("x"))
    return ["".join(tokens[i : i + size]) for i in range(0, len(tokens), size)]


CHUNKINGS = ["whole", "tokens", "x2", "x3"]
REASONING_PARSERS = ["gpt_oss", "harmony"]
TOOL_PARSERS = ["harmony", "gpt-oss"]


@pytest.fixture(params=REASONING_PARSERS)
def reasoning_name(request):
    return request.param


@pytest.fixture(params=TOOL_PARSERS)
def tool_name(request):
    return request.param


@pytest.fixture
def harmony_parsers(monkeypatch, reasoning_name, tool_name):
    import tests.test_reasoning_tool_call_contract as contract

    monkeypatch.setattr(contract, "_chunks", _chunks)
    # other tests leave a small --max-request-tokens behind; /v1/messages checks it
    monkeypatch.setattr(srv, "_max_request_tokens", 32768)
    monkeypatch.setattr(srv, "_enable_auto_tool_choice", True)
    monkeypatch.setattr(srv, "_tool_call_parser", tool_name)
    monkeypatch.setattr(
        srv, "_tool_parser_instance", ToolParserManager.get_tool_parser(tool_name)()
    )
    monkeypatch.setattr(srv, "_reasoning_parser", get_parser(reasoning_name)())
    monkeypatch.setattr(srv, "_reasoning_parser_name", reasoning_name)


def _assert_contract(seen, analysis, final, calls):
    assert seen.calls == [("echo_args", {"text": TEXT})] * calls
    assert seen.reasoning.strip() == analysis
    assert seen.content.strip() == final
    assert re.findall(r"<\|[a-z]+\|>", seen.content + seen.reasoning) == []


@pytest.mark.parametrize("chunking", CHUNKINGS)
@pytest.mark.parametrize("name", list(RAW))
@pytest.mark.parametrize("site", SITES)
@pytest.mark.anyio
async def test_streaming(monkeypatch, harmony_parsers, site, name, chunking):
    raw, analysis, final, calls = RAW[name]

    seen = await _run(
        site, monkeypatch, raw, chunking, stream=True, thinking_open_in_prompt=False
    )

    _assert_contract(seen, analysis, final, calls)


@pytest.mark.parametrize("name", list(RAW))
@pytest.mark.parametrize("site", NON_STREAMING_SITES)
@pytest.mark.anyio
async def test_non_streaming(monkeypatch, harmony_parsers, site, name):
    raw, analysis, final, calls = RAW[name]

    seen = await _run(
        site, monkeypatch, raw, "whole", stream=False, thinking_open_in_prompt=False
    )

    _assert_contract(seen, analysis, final, calls)


@pytest.mark.parametrize("chunking", CHUNKINGS)
@pytest.mark.parametrize("name", list(RAW))
@pytest.mark.anyio
async def test_streaming_equals_non_streaming(
    monkeypatch, harmony_parsers, name, chunking
):
    raw, *_ = RAW[name]

    whole = await _run(
        "chat", monkeypatch, raw, "whole", stream=False, thinking_open_in_prompt=False
    )
    streamed = await _run(
        "chat", monkeypatch, raw, chunking, stream=True, thinking_open_in_prompt=False
    )

    assert streamed.calls == whole.calls
    assert streamed.reasoning.strip() == whole.reasoning.strip()
    assert streamed.content.strip() == whole.content.strip()


# --- the HTTP endpoints: finish reason and the calls a client sees -------------------


def _post(monkeypatch, path, raw, chunking, stream):
    monkeypatch.setattr(srv, "_engine", _engine(raw, chunking, thinking_open_in_prompt=False))
    monkeypatch.setattr(srv, "_model_name", "test-model")
    body = ENDPOINT_BODIES[path]()
    if stream:
        body["stream"] = True
    return TestClient(srv.app).post(path, json=body)


@pytest.mark.parametrize("name", list(RAW))
@pytest.mark.parametrize("path", list(ENDPOINT_BODIES))
def test_endpoints_return_the_call_and_the_finish_reason(
    monkeypatch, harmony_parsers, path, name
):
    raw, _analysis_text, _final, calls = RAW[name]

    response = _post(monkeypatch, path, raw, "whole", stream=False)

    assert response.status_code == 200, response.text
    body = response.json()
    if path == "/v1/chat/completions":
        choice = body["choices"][0]
        assert len(choice["message"].get("tool_calls") or []) == calls
        assert choice["finish_reason"] == ("tool_calls" if calls else "stop")
    elif path == "/v1/responses":
        assert len([o for o in body["output"] if o["type"] == "function_call"]) == calls
    else:
        assert len([b for b in body["content"] if b["type"] == "tool_use"]) == calls
        assert body["stop_reason"] == ("tool_use" if calls else "end_turn")


@pytest.mark.parametrize("chunking", ["tokens", "x3"])
@pytest.mark.parametrize("name", list(RAW))
def test_chat_stream_finish_reason(monkeypatch, harmony_parsers, name, chunking):
    raw, _analysis_text, _final, calls = RAW[name]

    response = _post(monkeypatch, "/v1/chat/completions", raw, chunking, stream=True)

    assert response.status_code == 200
    choices = [
        p["choices"][0]
        for p in _sse_payloads(response.text.splitlines(True))
        if p.get("choices")
    ]
    indexes = {tc["index"] for c in choices for tc in c["delta"].get("tool_calls") or []}
    assert len(indexes) == calls
    assert [c["finish_reason"] for c in choices if c["finish_reason"]] == [
        "tool_calls" if calls else "stop"
    ]


# --- the harmony tool parser on its own (no --reasoning-parser) ----------------------


@pytest.fixture
def no_reasoning_parser(monkeypatch, tool_name):
    import tests.test_reasoning_tool_call_contract as contract

    monkeypatch.setattr(contract, "_chunks", _chunks)
    # other tests leave a small --max-request-tokens behind; /v1/messages checks it
    monkeypatch.setattr(srv, "_max_request_tokens", 32768)
    monkeypatch.setattr(srv, "_enable_auto_tool_choice", True)
    monkeypatch.setattr(srv, "_tool_call_parser", tool_name)
    monkeypatch.setattr(
        srv, "_tool_parser_instance", ToolParserManager.get_tool_parser(tool_name)()
    )
    monkeypatch.setattr(srv, "_reasoning_parser", None)
    monkeypatch.setattr(srv, "_reasoning_parser_name", None)


@pytest.mark.parametrize("chunking", CHUNKINGS)
@pytest.mark.parametrize("name", list(RAW))
@pytest.mark.parametrize("site", SITES)
@pytest.mark.anyio
async def test_tool_parser_alone_streaming(
    monkeypatch, no_reasoning_parser, site, name, chunking
):
    """Without a reasoning parser the analysis is dropped, calls and the final
    message come through; a delta that is only control tokens still reaches the parser."""
    raw, _analysis_text, final, calls = RAW[name]

    seen = await _run(
        site, monkeypatch, raw, chunking, stream=True, thinking_open_in_prompt=False
    )

    assert seen.calls == [("echo_args", {"text": TEXT})] * calls
    assert seen.content.strip() == final
    assert re.findall(r"<\|[a-z]+\|>", seen.content) == []
