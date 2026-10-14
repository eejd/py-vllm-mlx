# SPDX-License-Identifier: Apache-2.0
"""The pythonic parser under the server's real streaming sites.

``pythonic`` has no marker: a response is a tool call only if its first character is
``[``. The server must therefore hand it every delta (``REQUIRES_EAGER_STREAMING``) and
finalize at the end of every response, so a list the model never closed is released as
content and a list the last delta closed is not lost. Chat completions, Anthropic
messages and the Responses API each carry their own copy of that logic and are all
driven here.
"""

import json
from types import SimpleNamespace

import pytest

import vllm_mlx.server as srv
from vllm_mlx.api.models import ChatCompletionRequest
from vllm_mlx.api.responses_models import ResponsesRequest
from vllm_mlx.engine.base import GenerationOutput
from vllm_mlx.tool_parsers import ToolParserManager

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
            },
        },
    }
]
SITES = ["chat", "anthropic", "responses"]
CALL = "[get_weather(city='Paris', days=3)]"
TWO = "[get_weather(city='Paris'), get_weather(city='Tokyo', days=2)]"


class _PassThroughReasoning:
    """A reasoning parser that finds no reasoning: every site takes its reasoning branch."""

    def extract_reasoning_streaming(self, previous, current, delta):
        return SimpleNamespace(reasoning=None, content=delta)

    def extract_reasoning(self, model_output):
        return None, model_output

    def reset_state(self, *args, **kwargs):
        pass


@pytest.fixture(autouse=True, params=[False, True], ids=["plain", "reasoning-branch"])
def _pythonic_server(monkeypatch, request):
    parser_cls = ToolParserManager.get_tool_parser("pythonic")
    monkeypatch.setattr(srv, "_enable_auto_tool_choice", True)
    monkeypatch.setattr(srv, "_tool_call_parser", "pythonic")
    monkeypatch.setattr(srv, "_tool_parser_instance", parser_cls())
    monkeypatch.setattr(
        srv, "_reasoning_parser", _PassThroughReasoning() if request.param else None
    )
    monkeypatch.setattr(srv, "_reasoning_parser_name", None)


def _engine(deltas, finish_reason="stop", *, finish_on_last_delta):
    async def stream_chat(**kwargs):
        text = ""
        for i, delta in enumerate(deltas):
            text += delta
            done = i == len(deltas) - 1 and finish_on_last_delta
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
        messages=[{"role": "user", "content": "weather?"}],
        stream=True,
        tools=TOOLS,
    )


def _sse(chunks):
    out = []
    for raw in chunks:
        for line in raw.splitlines():
            if line.startswith("data: "):
                body = line.removeprefix("data: ").strip()
                if body and body != "[DONE]":
                    out.append(json.loads(body))
    return out


async def _run(site, monkeypatch, engine):
    """Drive one site; return ``(content deltas, [(name, arguments)], stop reasons)``."""
    request = _request()
    if site == "chat":
        chunks = [
            c
            async for c in srv.stream_chat_completion(engine, request.messages, request)
        ]
        choices = [p["choices"][0] for p in _sse(chunks) if p.get("choices")]
        texts = [c["delta"]["content"] for c in choices if c["delta"].get("content")]
        slots: dict[int, dict] = {}
        for c in choices:
            for tc in c["delta"].get("tool_calls") or []:
                slot = slots.setdefault(tc["index"], {"name": "", "arguments": ""})
                slot["name"] += tc["function"].get("name") or ""
                slot["arguments"] += tc["function"].get("arguments") or ""
        calls = [(s["name"], s["arguments"]) for _, s in sorted(slots.items())]
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
        events = _sse(chunks)
        texts = [
            e["delta"]["text"]
            for e in events
            if e["type"] == "content_block_delta" and e["delta"]["type"] == "text_delta"
        ]
        calls, partial = [], {}
        for e in events:
            if e["type"] == "content_block_start" and (
                e["content_block"]["type"] == "tool_use"
            ):
                partial[e["index"]] = [e["content_block"]["name"], ""]
                calls.append(partial[e["index"]])
            elif e["type"] == "content_block_delta" and (
                e["delta"]["type"] == "input_json_delta"
            ):
                partial[e["index"]][1] += e["delta"]["partial_json"]
        return (
            texts,
            [tuple(c) for c in calls],
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
            ResponsesRequest(model="test-model", input="weather?", stream=True)
        )
    ]
    events = _sse(chunks)
    texts = [e["delta"] for e in events if e["type"] == "response.output_text.delta"]
    completed = next(e["response"] for e in events if e["type"] == "response.completed")
    calls = [
        (i["name"], i["arguments"])
        for i in completed["output"]
        if i["type"] == "function_call"
    ]
    message_text = "".join(
        part["text"]
        for item in completed["output"]
        if item["type"] == "message"
        for part in item["content"]
    )
    assert message_text == "".join(texts)  # the final response carries what streamed
    return texts, calls, [completed["status"]]


def _chunked(text, size):
    return [text[i : i + size] for i in range(0, len(text), size)]


SIZES = [1, 4, 10_000]


@pytest.mark.parametrize("site", SITES)
@pytest.mark.parametrize("finish_on_last_delta", [False, True])
@pytest.mark.parametrize("size", SIZES)
@pytest.mark.anyio
async def test_a_call_list_becomes_tool_calls_and_nothing_else(
    monkeypatch, site, finish_on_last_delta, size
):
    engine = _engine(_chunked(TWO, size), finish_on_last_delta=finish_on_last_delta)

    texts, calls, stops = await _run(site, monkeypatch, engine)

    assert texts == []
    assert [(n, json.loads(a)) for n, a in calls] == [
        ("get_weather", {"city": "Paris"}),
        ("get_weather", {"city": "Tokyo", "days": 2}),
    ]
    assert stops and stops[-1] in ("tool_calls", "tool_use", "completed")


@pytest.mark.parametrize("site", SITES)
@pytest.mark.parametrize("finish_on_last_delta", [False, True])
@pytest.mark.parametrize("size", SIZES)
@pytest.mark.anyio
async def test_plain_text_streams_as_content(
    monkeypatch, site, finish_on_last_delta, size
):
    text = "It is sunny in Paris today."
    engine = _engine(_chunked(text, size), finish_on_last_delta=finish_on_last_delta)

    texts, calls, stops = await _run(site, monkeypatch, engine)

    assert "".join(texts) == text
    assert calls == []
    assert "tool_calls" not in stops and "tool_use" not in stops


@pytest.mark.parametrize("site", SITES)
@pytest.mark.parametrize("finish_on_last_delta", [False, True])
@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize(
    "text",
    [
        "[1] is the first item.",
        "[Note: I cannot check the weather.]",
        "[get_weather(1)]",  # positional
        "[get_weather(city=city_name)]",  # a variable, not a literal
    ],
)
@pytest.mark.anyio
async def test_bracket_text_that_is_not_a_call_list_is_content(
    monkeypatch, site, finish_on_last_delta, size, text
):
    engine = _engine(_chunked(text, size), finish_on_last_delta=finish_on_last_delta)

    texts, calls, stops = await _run(site, monkeypatch, engine)

    assert "".join(texts) == text
    assert calls == []
    assert "tool_calls" not in stops and "tool_use" not in stops


@pytest.mark.parametrize("site", SITES)
@pytest.mark.parametrize("finish_reason", ["stop", "length"])
@pytest.mark.parametrize("finish_on_last_delta", [False, True])
@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize(
    "text",
    [
        "[get_weather(city='Par",  # cut mid-argument
        "[get_weather(city='Paris')",  # the closing bracket never came
        "[get_weather(city='Paris'), get_wea",  # a complete call, then a cut one
    ],
)
@pytest.mark.anyio
async def test_a_list_cut_off_before_its_bracket_is_released_as_content(
    monkeypatch, site, finish_reason, finish_on_last_delta, size, text
):
    engine = _engine(
        _chunked(text, size), finish_reason, finish_on_last_delta=finish_on_last_delta
    )

    texts, calls, stops = await _run(site, monkeypatch, engine)

    # what non-streaming returns for the same text
    assert "".join(texts) == text
    assert calls == []
    assert "tool_calls" not in stops and "tool_use" not in stops


@pytest.mark.parametrize("site", SITES)
@pytest.mark.anyio
async def test_the_finishing_delta_may_close_the_list(monkeypatch, site):
    engine = _engine([CALL[:-1], CALL[-1]], finish_on_last_delta=True)

    texts, calls, _ = await _run(site, monkeypatch, engine)

    assert texts == []
    assert [(n, json.loads(a)) for n, a in calls] == [
        ("get_weather", {"city": "Paris", "days": 3})
    ]


@pytest.mark.parametrize("site", SITES)
@pytest.mark.parametrize("finish_on_last_delta", [False, True])
@pytest.mark.parametrize("size", [5, len(CALL) + 3, 10_000])
@pytest.mark.anyio
async def test_text_after_the_list_is_content_next_to_the_calls(
    monkeypatch, site, finish_on_last_delta, size
):
    # size len(CALL)+3 and 10_000 put the tail in the delta that closes the list; the
    # server then sends that text as its own chunk ahead of the calls (the order of
    # chunks is the server's; the aggregate equals the non-streaming content)
    engine = _engine(
        _chunked(CALL + "\n\nChecking now.", size),
        finish_on_last_delta=finish_on_last_delta,
    )

    texts, calls, _ = await _run(site, monkeypatch, engine)

    assert "".join(texts) == "Checking now."
    assert [n for n, _ in calls] == ["get_weather"]


def test_non_streaming_server_path_agrees_with_streaming():
    content, calls = srv._parse_tool_calls_with_parser(CALL, _request())
    assert not content
    assert [c.function.name for c in calls] == ["get_weather"]
    assert json.loads(calls[0].function.arguments) == {"city": "Paris", "days": 3}

    content, calls = srv._parse_tool_calls_with_parser("[1] is first", _request())
    assert calls is None
    assert content == "[1] is first"
