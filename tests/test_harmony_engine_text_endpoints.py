# SPDX-License-Identifier: Apache-2.0
"""gpt-oss through the REAL engine text path and the non-streaming endpoints
(py-quantbench#244).

``GenerationOutput.text`` of the batched and simple engines is
``clean_output_text(raw)``: it deletes the Harmony channel tokens and flattens the
channels. The harmony reasoning and tool parsers read those tokens, so a
non-streaming request with the default ``skip_special_tokens`` got the analysis text
and the call arguments glued into ``content`` and no call (a streamed request read
the raw deltas and worked). The other harmony tests hand the parsers raw text
through a fake engine, so none of them saw it.

Here the engine is a real ``BatchedEngine`` (its ``generate``/``chat`` cleaning is
the code under test) over a stand-in for its scheduler that returns *captured*
gpt-oss-20b output: the decoded text of ``mlx-community/gpt-oss-20b-MXFP4-Q8``
generations from py-quantbench's 2026-10-18 qualification run
(``docs/evidence/model-qualification-2026-10-18/gpt-oss-20b-mxfp4-q8/
result.json``, ``raw`` of the greedy cases). A generation ends on a stop token
(``<|return|>``/``<|call|>``) that is not decoded, so the captures end without one."""

import json
import re
import types

import pytest
from fastapi.testclient import TestClient

import vllm_mlx.server as srv
from vllm_mlx.engine.batched import BatchedEngine
from vllm_mlx.reasoning import get_parser
from vllm_mlx.tool_parsers import ToolParserManager

# tool_call weather-00 (tool: get_weather)
CALL = (
    "<|channel|>analysis<|message|>We need to call get_weather function.<|end|>"
    "<|start|>assistant<|channel|>commentary to=functions.get_weather "
    '<|constrain|>json<|message|>{"city":"Bogota","unit":"celsius"}'
)
CALL_REASONING = "We need to call get_weather function."
CALL_ARGS = {"city": "Bogota", "unit": "celsius"}

# tool_arg list-01 (tool: echo_args)
ARG_CALL = (
    "<|channel|>analysis<|message|>We need to call the function echo_args with text "
    '\'list\' and items ["age", "computer", "issue"]. No other arguments.<|end|>'
    "<|start|>assistant<|channel|>commentary to=functions.echo_args <|constrain|>json"
    '<|message|>{"text":"list","items":["age","computer","issue"]}'
)

# reasoning legs-00 (no tool)
ANSWER = (
    "<|channel|>analysis<|message|>We need to compute total legs. Cows have 4 legs each. "
    "27 cows => 27*4 = 108 legs. Hens have 2 legs each. 4 hens => 4*2 = 8 legs. "
    "Total = 108 + 8 = 116 legs. Provide brief reasoning.<|end|>"
    "<|start|>assistant<|channel|>final<|message|>Cows have 4 legs and hens have 2 legs."
    "\n\n- 27 cows \u00d7 4 legs = 108 legs  \n- 4 hens \u00d7 2 legs = 8 legs  \n\n"
    "Total legs = 108 + 8 = **116**."
)
ANSWER_REASONING = (
    "We need to compute total legs. Cows have 4 legs each. 27 cows => 27*4 = 108 legs. "
    "Hens have 2 legs each. 4 hens => 4*2 = 8 legs. Total = 108 + 8 = 116 legs. "
    "Provide brief reasoning."
)
ANSWER_CONTENT = (
    "Cows have 4 legs and hens have 2 legs.\n\n- 27 cows \u00d7 4 legs = 108 legs  \n"
    "- 4 hens \u00d7 2 legs = 8 legs  \n\nTotal legs = 108 + 8 = **116**."
)

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}, "unit": {"type": "string"}},
            "required": ["city"],
        },
    },
}
ECHO_TOOL = {
    "type": "function",
    "function": {
        "name": "echo_args",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "items": {"type": "array", "items": {"type": "string"}},
            },
        },
    },
}


class _Scheduler:
    """What ``BatchedEngine._engine`` is for ``generate``: records the request, returns
    the captured text (``output_text`` is what the scheduler decodes)."""

    def __init__(self, raw):
        self.raw = raw
        self.sampling_params = None

    async def generate(self, prompt, sampling_params, add_special_tokens=None):
        self.sampling_params = sampling_params
        return types.SimpleNamespace(
            output_text=self.raw,
            output_token_ids=[],
            prompt_tokens=7,
            completion_tokens=3,
            finish_reason="stop",
            mtp_drafts=0,
            mtp_accepted=0,
            cached_tokens=0,
            error=None,
            finished=True,
        )


def _engine(raw):
    engine = object.__new__(BatchedEngine)
    engine._loaded = True
    engine._is_mllm = False
    engine._mllm_scheduler = None
    engine._engine = _Scheduler(raw)
    engine._model_name = "test-model"
    engine.preserve_native_tool_format = False
    engine._apply_chat_template = lambda messages, *a, **k: "<prompt>"

    async def validate_chat_template(messages, **kwargs):
        return None

    engine.validate_chat_template = validate_chat_template
    return engine


def _use_parsers(monkeypatch, *, tool, reasoning):
    monkeypatch.setattr(srv, "_max_request_tokens", 32768)
    monkeypatch.setattr(srv, "_enable_auto_tool_choice", tool is not None)
    monkeypatch.setattr(srv, "_tool_call_parser", tool)
    monkeypatch.setattr(
        srv,
        "_tool_parser_instance",
        ToolParserManager.get_tool_parser(tool)() if tool else None,
    )
    monkeypatch.setattr(
        srv, "_reasoning_parser", get_parser(reasoning)() if reasoning else None
    )
    monkeypatch.setattr(srv, "_reasoning_parser_name", reasoning)
    monkeypatch.setattr(srv, "_model_name", "test-model")


def _post(monkeypatch, path, raw, tool_def=None, **extra):
    engine = _engine(raw)
    monkeypatch.setattr(srv, "_engine", engine)
    if path == "/v1/chat/completions":
        body = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "go"}],
            "temperature": 0,
        }
        if tool_def:
            body["tools"] = [tool_def]
    elif path == "/v1/responses":
        body = {"model": "test-model", "input": "go"}
        if tool_def:
            body["tools"] = [{"type": "function", **tool_def["function"]}]
    else:
        body = {
            "model": "test-model",
            "max_tokens": 256,
            "messages": [{"role": "user", "content": "go"}],
        }
        if tool_def:
            body["tools"] = [
                {
                    "name": tool_def["function"]["name"],
                    "input_schema": tool_def["function"]["parameters"],
                }
            ]
    response = TestClient(srv.app).post(path, json=body | extra)
    assert response.status_code == 200, response.text
    return engine, response.json()


def _seen(path, payload):
    """(reasoning, content, calls[(name, args)]) as a client of ``path`` sees them."""
    if path == "/v1/chat/completions":
        msg = payload["choices"][0]["message"]
        calls = [
            (c["function"]["name"], json.loads(c["function"]["arguments"]))
            for c in msg.get("tool_calls") or []
        ]
        return (
            msg.get("reasoning") or msg.get("reasoning_content") or "",
            msg.get("content") or "",
            calls,
        )
    if path == "/v1/responses":
        out = payload["output"]
        reasoning = "".join(
            part["text"]
            for o in out
            if o["type"] == "reasoning"
            for part in o.get("content") or o.get("summary") or []
        )
        content = "".join(
            part["text"]
            for o in out
            if o["type"] == "message"
            for part in o["content"]
            if part.get("text")
        )
        calls = [
            (o["name"], json.loads(o["arguments"]))
            for o in out
            if o["type"] == "function_call"
        ]
        return reasoning, content, calls
    blocks = payload["content"]
    return (
        "".join(b["thinking"] for b in blocks if b["type"] == "thinking"),
        "".join(b["text"] for b in blocks if b["type"] == "text"),
        [(b["name"], b["input"]) for b in blocks if b["type"] == "tool_use"],
    )


PATHS = ["/v1/chat/completions", "/v1/responses", "/v1/messages"]
PARSERS = [
    pytest.param("harmony", "gpt_oss", id="harmony+gpt_oss"),
    pytest.param("gpt-oss", "harmony", id="gpt-oss+harmony"),
]


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("tool,reasoning", PARSERS)
def test_a_call_is_returned_with_default_request_parameters(
    monkeypatch, path, tool, reasoning
):
    _use_parsers(monkeypatch, tool=tool, reasoning=reasoning)

    engine, payload = _post(monkeypatch, path, CALL, WEATHER_TOOL)

    r, content, calls = _seen(path, payload)
    assert calls == [("get_weather", CALL_ARGS)]
    assert r.strip() == CALL_REASONING
    assert content.strip() == ""
    # the engine was asked for the text as generated: the parsers need its channels
    assert engine._engine.sampling_params.skip_special_tokens is False
    if path == "/v1/chat/completions":
        assert payload["choices"][0]["finish_reason"] == "tool_calls"


@pytest.mark.parametrize("path", PATHS)
def test_a_call_with_structured_arguments(monkeypatch, path):
    _use_parsers(monkeypatch, tool="harmony", reasoning="gpt_oss")

    _, payload = _post(monkeypatch, path, ARG_CALL, ECHO_TOOL)

    _, _, calls = _seen(path, payload)
    assert calls == [
        ("echo_args", {"text": "list", "items": ["age", "computer", "issue"]})
    ]


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("tools", [False, True])
def test_the_reasoning_is_split_from_the_answer(monkeypatch, path, tools):
    _use_parsers(monkeypatch, tool="harmony" if tools else None, reasoning="gpt_oss")

    _, payload = _post(monkeypatch, path, ANSWER, WEATHER_TOOL if tools else None)

    r, content, calls = _seen(path, payload)
    assert calls == []
    assert r.strip() == ANSWER_REASONING
    assert content.strip() == ANSWER_CONTENT
    assert re.findall(r"<\|[a-z]+\|>", r + content) == []


def test_the_tool_parser_alone_returns_the_call(monkeypatch):
    _use_parsers(monkeypatch, tool="harmony", reasoning=None)

    engine, payload = _post(monkeypatch, "/v1/chat/completions", CALL, WEATHER_TOOL)

    _, content, calls = _seen("/v1/chat/completions", payload)
    assert calls == [("get_weather", CALL_ARGS)]
    assert re.findall(r"<\|[a-z]+\|>", content) == []
    assert engine._engine.sampling_params.skip_special_tokens is False


def test_a_tool_parser_alone_leaves_a_request_without_tools_cleaned_by_the_engine(
    monkeypatch,
):
    """With no tools nothing reads the channels: the engine keeps its final-channel
    fallback."""
    _use_parsers(monkeypatch, tool="harmony", reasoning=None)

    engine, payload = _post(monkeypatch, "/v1/chat/completions", ANSWER)

    _, content, calls = _seen("/v1/chat/completions", payload)
    assert calls == []
    assert content.strip() == ANSWER_CONTENT
    assert engine._engine.sampling_params.skip_special_tokens is True


@pytest.mark.parametrize("path", PATHS)
def test_streaming_is_left_alone(monkeypatch, path):
    """A stream reads the raw deltas; it must not start asking for rendered boundary
    tokens."""
    from vllm_mlx.engine.base import GenerationOutput

    _use_parsers(monkeypatch, tool="harmony", reasoning="gpt_oss")
    engine = _engine(CALL)
    seen = {}

    async def stream_chat(**kwargs):
        seen.update(kwargs)
        yield GenerationOutput(
            text=CALL, new_text=CALL, finished=True, finish_reason="stop"
        )

    engine.stream_chat = stream_chat
    monkeypatch.setattr(srv, "_engine", engine)
    body = {
        "/v1/chat/completions": {
            "model": "test-model",
            "messages": [{"role": "user", "content": "go"}],
            "tools": [WEATHER_TOOL],
        },
        "/v1/responses": {
            "model": "test-model",
            "input": "go",
            "tools": [{"type": "function", **WEATHER_TOOL["function"]}],
        },
        "/v1/messages": {
            "model": "test-model",
            "max_tokens": 256,
            "messages": [{"role": "user", "content": "go"}],
            "tools": [
                {
                    "name": "get_weather",
                    "input_schema": WEATHER_TOOL["function"]["parameters"],
                }
            ],
        },
    }[path]
    response = TestClient(srv.app).post(path, json=body | {"stream": True})

    assert response.status_code == 200, response.text
    assert seen and "skip_special_tokens" not in seen


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("tool,reasoning", PARSERS)
def test_tool_choice_none_returns_no_call_and_no_control_token(
    monkeypatch, path, tool, reasoning
):
    """tools + ``tool_choice: none``: the model was not offered the tools, a call is
    not honoured, and the raw channel text the parsers now read never reaches the
    client (the final cleaning of every endpoint still runs)."""
    _use_parsers(monkeypatch, tool=tool, reasoning=reasoning)
    choice = {"type": "none"} if path == "/v1/messages" else "none"

    _, payload = _post(monkeypatch, path, CALL, WEATHER_TOOL, tool_choice=choice)

    r, content, calls = _seen(path, payload)
    assert calls == []
    assert r.strip() == CALL_REASONING  # the reasoning parser still separates it
    assert re.findall(r"<\|[a-z]+\|>", r + content) == []


def test_a_response_format_failure_does_not_echo_the_channel_text(monkeypatch):
    """The ``response_format`` fallback (``cleaned_text or output.text``) can now be
    handed the raw channel text (a harmony output with no final channel); the 422 it
    ends in must not carry it.

    The JSON logits processor is switched off (the stand-in engine has no tokenizer
    to build one from), so the request reaches the validation of the generated text,
    which is what this pins."""
    _use_parsers(monkeypatch, tool=None, reasoning="gpt_oss")
    monkeypatch.setattr(
        srv,
        "_prepare_json_logits_processor",
        lambda engine, messages, *a, **k: (messages, None),
    )
    validated = []
    real = srv._apply_response_format_or_raise

    def spy(text, response_format, **kwargs):
        validated.append(text)
        return real(text, response_format, **kwargs)

    monkeypatch.setattr(srv, "_apply_response_format_or_raise", spy)
    # a generation cut off inside the analysis: no final channel, no content
    monkeypatch.setattr(srv, "_engine", _engine(CALL.split("<|end|>")[0]))

    response = TestClient(srv.app).post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "go"}],
            "response_format": {"type": "json_object"},
        },
    )

    # the fallback got the raw channel text
    assert validated and "<|channel|>" in validated[0]
    assert response.status_code == 422, response.text
    assert response.json()["detail"]["error"] == "invalid_response_format_output"
    for echoed in ("<|", "get_weather", "analysis"):
        assert echoed not in response.text


def test_a_client_that_asks_for_skip_special_tokens_false_still_gets_the_call(
    monkeypatch,
):
    _use_parsers(monkeypatch, tool="harmony", reasoning="gpt_oss")

    _, payload = _post(
        monkeypatch,
        "/v1/chat/completions",
        CALL,
        WEATHER_TOOL,
        skip_special_tokens=False,
    )

    assert _seen("/v1/chat/completions", payload)[2] == [("get_weather", CALL_ARGS)]


def test_without_a_harmony_parser_the_default_is_untouched(monkeypatch):
    """No parser reads control tokens: skip_special_tokens stays the engine default."""
    _use_parsers(monkeypatch, tool="hermes", reasoning="qwen3")

    engine, _ = _post(monkeypatch, "/v1/chat/completions", "hello", WEATHER_TOOL)

    assert engine._engine.sampling_params.skip_special_tokens is True
