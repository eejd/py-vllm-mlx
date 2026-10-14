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
    "I could call to=functions.echo_args with "
    + json.dumps({"text": TEXT})
    + " but no."
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
        "Looking it up.",  # a commentary message to no one is content, as in vLLM
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
    # a message addressed to a function is a call whatever its channel (vLLM's reading)
    "call-in-the-analysis-channel": (
        "<|channel|>analysis to=functions.echo_args <|constrain|>json"
        f"<|message|>{ARGS}<|call|>",
        "",
        "",
        1,
    ),
    # a commentary message addressed to no one is content (a preamble to the user)
    "preamble-only": (
        _analysis(ANALYSIS)
        + "<|start|>assistant<|channel|>commentary<|message|>Looking it up.<|end|>",
        ANALYSIS,
        "Looking it up.",
        0,
    ),
    "cut-off-in-a-preamble": (
        _analysis(ANALYSIS)
        + "<|start|>assistant<|channel|>commentary<|message|>Looking",
        ANALYSIS,
        "Looking",
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
    monkeypatch.setattr(
        srv, "_engine", _engine(raw, chunking, thinking_open_in_prompt=False)
    )
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
    indexes = {
        tc["index"] for c in choices for tc in c["delta"].get("tool_calls") or []
    }
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
    message come through; a delta that is only control tokens still reaches the parser.
    """
    raw, _analysis_text, final, calls = RAW[name]

    seen = await _run(
        site, monkeypatch, raw, chunking, stream=True, thinking_open_in_prompt=False
    )

    assert seen.calls == [("echo_args", {"text": TEXT})] * calls
    assert seen.content.strip() == final
    assert re.findall(r"<\|[a-z]+\|>", seen.content) == []


@pytest.mark.parametrize("chunking", ["tokens", "x3"])
@pytest.mark.parametrize("name", list(RAW))
def test_chat_stream_sends_no_empty_chunk(monkeypatch, harmony_parsers, name, chunking):
    """Control tokens the parsers consume must not become empty deltas on the wire."""
    raw, *_ = RAW[name]

    response = _post(monkeypatch, "/v1/chat/completions", raw, chunking, stream=True)

    for payload in _sse_payloads(response.text.splitlines(True)):
        for choice in payload.get("choices") or []:
            delta = choice["delta"]
            assert (
                delta.get("content")
                or delta.get("reasoning_content")
                or delta.get("reasoning")
                or delta.get("tool_calls")
                or delta.get("role")
                or choice.get("finish_reason")
            ), payload


@pytest.mark.parametrize("chunking", CHUNKINGS)
@pytest.mark.parametrize("tool_parser", ["hermes", "qwen"])
@pytest.mark.parametrize(
    "name",
    ["analysis-final-eos" if False else "final-at-eos", "call-mentioned-in-analysis"],
)
@pytest.mark.anyio
async def test_messages_stream_feeds_a_harmony_reasoning_parser_the_control_tokens(
    monkeypatch, harmony_parsers, name, tool_parser, chunking
):
    """The reasoning parser alone needs them, whatever tool parser is configured."""
    monkeypatch.setattr(srv, "_tool_call_parser", tool_parser)
    monkeypatch.setattr(
        srv, "_tool_parser_instance", ToolParserManager.get_tool_parser(tool_parser)()
    )
    raw, analysis, final, calls = RAW[name]

    seen = await _run(
        "anthropic",
        monkeypatch,
        raw,
        chunking,
        stream=True,
        thinking_open_in_prompt=False,
    )

    assert (seen.reasoning.strip(), seen.content.strip()) == (analysis, final)


def _read_endpoint(path, response, stream):
    """(content, reasoning, number of calls, finish reason) as a client assembles them."""
    content = reasoning = ""
    calls = 0
    finish = None
    if not stream:
        body = response.json()
        if path == "/v1/chat/completions":
            message = body["choices"][0]["message"]
            return (
                message.get("content") or "",
                message.get("reasoning") or message.get("reasoning_content") or "",
                len(message.get("tool_calls") or []),
                body["choices"][0]["finish_reason"],
            )
        if path == "/v1/messages":
            for block in body["content"]:
                content += block.get("text", "") if block["type"] == "text" else ""
                reasoning += block["thinking"] if block["type"] == "thinking" else ""
                calls += block["type"] == "tool_use"
            return content, reasoning, calls, "tool_calls" if calls else "stop"
        for item in body["output"]:
            calls += item["type"] == "function_call"
            if item["type"] == "reasoning":
                reasoning += "".join(
                    part.get("text", "")
                    for part in (item.get("summary") or [])
                    + (item.get("content") or [])
                )
            if item["type"] == "message":
                content += "".join(part.get("text", "") for part in item["content"])
        return content, reasoning, calls, "tool_calls" if calls else "stop"
    events = [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: ") and line[6:].strip() != "[DONE]"
    ]
    if path == "/v1/chat/completions":
        indexes = set()
        for event in events:
            for choice in event.get("choices") or []:
                delta = choice["delta"]
                content += delta.get("content") or ""
                reasoning += (
                    delta.get("reasoning") or delta.get("reasoning_content") or ""
                )
                indexes |= {t["index"] for t in delta.get("tool_calls") or []}
                finish = choice.get("finish_reason") or finish
        return content, reasoning, len(indexes), finish
    if path == "/v1/messages":
        for event in events:
            if event["type"] == "content_block_delta":
                delta = event["delta"]
                content += (
                    delta.get("text", "") if delta["type"] == "text_delta" else ""
                )
                reasoning += (
                    delta["thinking"] if delta["type"] == "thinking_delta" else ""
                )
            calls += (
                event["type"] == "content_block_start"
                and event["content_block"]["type"] == "tool_use"
            )
            if event["type"] == "message_delta":
                finish = (
                    "tool_calls"
                    if event["delta"]["stop_reason"] == "tool_use"
                    else "stop"
                )
        return content, reasoning, calls, finish
    for event in events:
        kind = event.get("type", "")
        if kind == "response.output_text.delta":
            content += event["delta"]
        if "reasoning" in kind and "delta" in kind:
            reasoning += event.get("delta", "")
        calls += (
            kind == "response.output_item.done"
            and event["item"]["type"] == "function_call"
        )
    return content, reasoning, calls, "tool_calls" if calls else "stop"


THINKING_OFF_CHUNKINGS = ["whole", "tokens", "x3", "x4", "x5"]


@pytest.mark.parametrize("chunking", THINKING_OFF_CHUNKINGS)
@pytest.mark.parametrize("with_tools", [True, False], ids=["tools", "no-tools"])
@pytest.mark.parametrize("path", list(ENDPOINT_BODIES))
@pytest.mark.parametrize("name", list(RAW))
def test_thinking_disabled_never_shows_the_analysis_or_a_control_token(
    monkeypatch, harmony_parsers, name, path, with_tools, chunking
):
    """enable_thinking=false: the model may still open an analysis channel. On every
    endpoint, streaming or not, in any chunking, with or without tools: the analysis is
    dropped (never reasoning, never content), no control token and no ``assistant``
    role word reaches the client, and a declared tool's call is returned exactly once.
    (The streaming sites used to stop feeding the harmony tool parser on a delta whose
    only visible part was reasoning: calls lost, and ``assistant`` shown as content.)"""
    raw, _analysis_text, final, calls = RAW[name]
    stream = chunking != "whole"
    monkeypatch.setattr(
        srv, "_engine", _engine(raw, chunking, thinking_open_in_prompt=False)
    )
    monkeypatch.setattr(srv, "_model_name", "test-model")
    body = ENDPOINT_BODIES[path]()
    body.update(stream=stream, chat_template_kwargs={"enable_thinking": False})
    if not with_tools:
        body.pop("tools")

    response = TestClient(srv.app).post(path, json=body)

    assert response.status_code == 200, response.text
    content, reasoning, n_calls, finish = _read_endpoint(path, response, stream)
    expected = calls if with_tools else 0
    assert content.strip() == final
    assert not reasoning.strip()
    assert re.findall(r"<\|[a-z]+\|>|to=functions", content) == []
    assert n_calls == expected
    assert finish == ("tool_calls" if expected else "stop")


@pytest.mark.parametrize(
    "name", ["call-only", "analysis-into-call-at-eos", "call-in-the-analysis-channel"]
)
def test_a_call_with_no_tools_declared_is_never_content(
    monkeypatch, harmony_parsers, name
):
    """No tools in the request (or tool_choice none): the model's call is not a call and
    not text either; the control tokens never reach the client."""
    raw, analysis, _final, _calls = RAW[name]
    monkeypatch.setattr(
        srv, "_engine", _engine(raw, "whole", thinking_open_in_prompt=False)
    )
    monkeypatch.setattr(srv, "_model_name", "test-model")
    body = {
        k: v
        for k, v in ENDPOINT_BODIES["/v1/chat/completions"]().items()
        if k != "tools"
    }

    response = TestClient(srv.app).post("/v1/chat/completions", json=body)

    assert response.status_code == 200, response.text
    message = response.json()["choices"][0]["message"]
    assert not message.get("tool_calls")
    assert "<|" not in (message.get("content") or "")
    assert "to=functions" not in (message.get("content") or "")


@pytest.mark.parametrize("with_tools", [False, True])
def test_a_call_only_output_leaves_no_control_tokens_in_the_cleaned_content(
    harmony_parsers, with_tools
):
    """The shared extraction (responses, messages, chat): nothing to separate."""
    raw, *_ = RAW["call-only"]
    request = _request_with_or_without_tools(with_tools)

    reasoning, content, calls = srv._extract_reasoning_and_tool_calls(
        raw, request, engine=_engine(raw, "whole"), chat_kwargs={}
    )

    assert not reasoning
    assert len(calls or []) == (1 if with_tools else 0)
    assert "<|" not in (content or "") and "to=functions" not in (content or "")


def _request_with_or_without_tools(with_tools):
    from tests.test_reasoning_tool_call_contract import _request

    request = _request(False)
    return request if with_tools else request.model_copy(update={"tools": None})


@pytest.mark.anyio
async def test_responses_stream_final_calls_are_parsed_with_thinking_disabled(
    monkeypatch,
):
    """With thinking disabled the end-of-stream extraction of /v1/responses reads the
    output as an answer: a call in an output that carries no reasoning marker at all is a
    call, even when the template would have opened the thinking (the engine's prompt
    ends in ``<think>``), exactly as for the other endpoints."""
    from tests.test_reasoning_tool_call_contract import (
        _json_call,
        _request,
        _sse_payloads,
    )
    from vllm_mlx.api.responses_models import ResponsesRequest

    monkeypatch.setattr(srv, "_max_request_tokens", 32768)
    monkeypatch.setattr(srv, "_enable_auto_tool_choice", True)
    monkeypatch.setattr(srv, "_tool_call_parser", "hermes")
    monkeypatch.setattr(
        srv, "_tool_parser_instance", ToolParserManager.get_tool_parser("hermes")()
    )
    monkeypatch.setattr(srv, "_reasoning_parser", get_parser("qwen3")())
    monkeypatch.setattr(srv, "_reasoning_parser_name", "qwen3")
    raw = "Calling it.\n" + _json_call()
    engine = _engine(raw, "whole", thinking_open_in_prompt=True)
    request = _request(True)
    off = {"chat_template_kwargs": {"enable_thinking": False}}
    monkeypatch.setattr(
        srv,
        "_prepare_streaming_responses_request",
        lambda req, engine=None: (engine_obj, request, request.messages, off),
    )
    engine_obj = engine

    chunks = [
        c
        async for c in srv._stream_responses_request(
            ResponsesRequest(model="test-model", input="call it", stream=True)
        )
    ]

    completed = next(
        e["response"]
        for e in _sse_payloads(chunks)
        if e["type"] == "response.completed"
    )
    assert [i["name"] for i in completed["output"] if i["type"] == "function_call"] == [
        "echo_args"
    ]
