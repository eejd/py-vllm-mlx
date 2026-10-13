# SPDX-License-Identifier: Apache-2.0
"""A ``<|python_tag|>`` call cut off at the end of generation is content, not nothing (#67).

The llama parser holds a tagged call that is a valid JSON prefix back while the model is
still generating it. Once generation ended (EOS or length cap) the call can never complete,
so every streaming site must release it as content, the control token stripped, exactly as
non-streaming does, and never as a tool call. Each site carries its own copy of the
end-of-stream decision, so all of them are driven for real here.
"""

import json
import logging

import pytest

import vllm_mlx.server as srv
from vllm_mlx.api.models import ChatCompletionRequest
from vllm_mlx.api.responses_models import ResponsesRequest
from vllm_mlx.engine.base import GenerationOutput
from vllm_mlx.tool_parsers import LlamaToolParser

TAG = "<|python_tag|>"
# Llama-3.2-1B-Instruct-4bit, py-quantbench code-value 2026-10-13: one closing brace short
CUT_OFF = (
    '{"type": "function", "function": {"name": "echo_args", '
    '"parameters": {"text": "hello", "n": 3}}'
)
COMPLETE = '{"name": "echo_args", "parameters": {"text": "hello"}}'
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "echo_args",
            "parameters": {"type": "object", "properties": {}},
        },
    }
]
SITES = ["chat", "anthropic", "responses"]


@pytest.fixture(autouse=True)
def _llama_server(monkeypatch):
    monkeypatch.setattr(srv, "_enable_auto_tool_choice", True)
    monkeypatch.setattr(srv, "_tool_call_parser", "llama")
    monkeypatch.setattr(srv, "_tool_parser_instance", LlamaToolParser())
    monkeypatch.setattr(srv, "_reasoning_parser", None)
    monkeypatch.setattr(srv, "_reasoning_parser_name", None)


def _engine(deltas, finish_reason, *, finish_on_last_delta):
    """Engine streaming ``deltas``; the finished output is either an extra empty one or
    the last token's own delta (what a length cap looks like)."""

    async def stream_chat(**kwargs):
        text = ""
        for i, delta in enumerate(deltas):
            text += delta
            last = i == len(deltas) - 1
            done = last and finish_on_last_delta
            yield GenerationOutput(
                text=text,
                new_text=delta,
                finished=done,
                finish_reason=finish_reason if done else None,
                prompt_tokens=7,
                completion_tokens=i + 1,
            )
        if not finish_on_last_delta:
            yield GenerationOutput(
                text=text,
                new_text="",
                finished=True,
                finish_reason=finish_reason,
                prompt_tokens=7,
                completion_tokens=len(deltas),
            )

    return type(
        "FakeEngine",
        (),
        {
            "model_name": "test-model",
            "preserve_native_tool_format": False,
            "stream_chat": staticmethod(stream_chat),
        },
    )()


def _request():
    return ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "call it"}],
        stream=True,
        tools=TOOLS,
    )


def _payloads(chunks):
    out = []
    for raw in chunks:
        for line in raw.splitlines():
            if line.startswith("data: "):
                body = line.removeprefix("data: ").strip()
                if body and body != "[DONE]":
                    out.append(json.loads(body))
    return out


async def _run(site, monkeypatch, engine):
    """Drive one streaming site; return (content deltas, tool call names, stop reason)."""
    request = _request()
    if site == "chat":
        chunks = [
            c
            async for c in srv.stream_chat_completion(engine, request.messages, request)
        ]
        choices = [p["choices"][0] for p in _payloads(chunks) if p.get("choices")]
        texts = [c["delta"]["content"] for c in choices if c["delta"].get("content")]
        calls = [
            tc["function"]["name"]
            for c in choices
            for tc in c["delta"].get("tool_calls") or []
        ]
        return texts, calls, [c["finish_reason"] for c in choices if c["finish_reason"]]

    if site == "anthropic":
        prepared = type(
            "Prepared",
            (),
            {"messages": request.messages, "tools": None, "chat_kwargs": {}},
        )()
        anthropic_request = type(
            "AnthropicRequest", (), {"model": "test-model", "max_tokens": 64}
        )()
        chunks = [
            c
            async for c in srv._stream_anthropic_messages(
                engine, request, anthropic_request, prepared
            )
        ]
        events = []
        for raw in chunks:
            for line in raw.splitlines():
                if line.startswith("data: "):
                    events.append(json.loads(line.removeprefix("data: ")))
        texts = [
            e["delta"]["text"]
            for e in events
            if e["type"] == "content_block_delta" and e["delta"]["type"] == "text_delta"
        ]
        calls = [
            e["content_block"]["name"]
            for e in events
            if e["type"] == "content_block_start"
            and e["content_block"]["type"] == "tool_use"
        ]
        return (
            texts,
            calls,
            [e["delta"]["stop_reason"] for e in events if e["type"] == "message_delta"],
        )

    monkeypatch.setattr(
        srv,
        "_prepare_streaming_responses_request",
        lambda req: (engine, request, request.messages, {}),
    )
    chunks = [
        c
        async for c in srv._stream_responses_request(
            ResponsesRequest(model="test-model", input="call it", stream=True)
        )
    ]
    events = _payloads(chunks)
    texts = [e["delta"] for e in events if e["type"] == "response.output_text.delta"]
    completed = next(e["response"] for e in events if e["type"] == "response.completed")
    calls = [i["name"] for i in completed["output"] if i["type"] == "function_call"]
    message_text = "".join(
        part["text"]
        for item in completed["output"]
        if item["type"] == "message"
        for part in item["content"]
    )
    # the completed response carries the same text the deltas did
    assert message_text == "".join(texts)
    return texts, calls, [completed["status"]]


CHUNKINGS = {
    "whole": [TAG + CUT_OFF],
    "tag-then-body": [TAG, CUT_OFF[:25], CUT_OFF[25:]],
    "per-char": [TAG] + list(CUT_OFF),
}


@pytest.mark.parametrize("site", SITES)
@pytest.mark.parametrize("finish_reason", ["stop", "length"])
@pytest.mark.parametrize("finish_on_last_delta", [False, True])
@pytest.mark.parametrize("chunking", list(CHUNKINGS))
@pytest.mark.anyio
async def test_a_cut_off_tagged_call_is_released_as_content_when_generation_ends(
    monkeypatch, caplog, site, finish_reason, finish_on_last_delta, chunking
):
    engine = _engine(
        CHUNKINGS[chunking], finish_reason, finish_on_last_delta=finish_on_last_delta
    )

    with caplog.at_level(logging.WARNING, logger="vllm_mlx.tool_parsers"):
        texts, calls, stops = await _run(site, monkeypatch, engine)

    assert "".join(texts) == CUT_OFF
    assert TAG not in "".join(texts)
    assert calls == []
    # still pending until the last output: the content arrives in one piece, at the end
    assert texts == [CUT_OFF]
    assert "tool_calls" not in stops
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1
    assert "echo_args" in warnings[0].getMessage()


@pytest.mark.parametrize("site", SITES)
@pytest.mark.anyio
async def test_text_before_the_cut_off_call_streams_at_once_and_is_not_repeated(
    monkeypatch, site
):
    engine = _engine(
        ["Let me check. ", TAG, CUT_OFF[:30], CUT_OFF[30:]],
        "stop",
        finish_on_last_delta=False,
    )

    texts, calls, _ = await _run(site, monkeypatch, engine)

    assert texts == ["Let me check. ", CUT_OFF]
    assert calls == []


@pytest.mark.parametrize("site", SITES)
@pytest.mark.parametrize("finish_on_last_delta", [False, True])
@pytest.mark.anyio
async def test_a_complete_call_is_unchanged_and_finalising_adds_no_content(
    monkeypatch, caplog, site, finish_on_last_delta
):
    engine = _engine(
        [TAG, COMPLETE[:20], COMPLETE[20:]],
        "stop",
        finish_on_last_delta=finish_on_last_delta,
    )

    with caplog.at_level(logging.WARNING, logger="vllm_mlx.tool_parsers"):
        texts, calls, _ = await _run(site, monkeypatch, engine)

    assert texts == []
    assert calls == ["echo_args"]
    assert not [r for r in caplog.records if r.levelname == "WARNING"]


def test_non_streaming_server_path_returns_the_cut_off_call_as_content(caplog):
    with caplog.at_level(logging.WARNING, logger="vllm_mlx.tool_parsers"):
        content, calls = srv._parse_tool_calls_with_parser(TAG + CUT_OFF, _request())

    assert calls is None
    assert content == CUT_OFF


@pytest.mark.parametrize("site", SITES)
@pytest.mark.anyio
async def test_the_finishing_delta_may_both_emit_text_and_open_the_cut_off_call(
    monkeypatch, site
):
    # the parser answers that delta with content (not None), so the end-of-stream hook
    # must run after a result as well, or the pending call would be hidden
    engine = _engine(
        ["Let me check. " + TAG + CUT_OFF], "length", finish_on_last_delta=True
    )

    texts, calls, _ = await _run(site, monkeypatch, engine)

    assert "".join(texts) == "Let me check. " + CUT_OFF
    assert calls == []
