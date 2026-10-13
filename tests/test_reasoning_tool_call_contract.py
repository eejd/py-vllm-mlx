# SPDX-License-Identifier: Apache-2.0
"""With a reasoning parser the tool parser sees the content part only, never the reasoning.

Qwen3.8-27B (py-quantbench, 2026-10-13) drafts its call inside the thinking
("Let me make the call: <tool_call>...") and then, after ``</think>``, makes the real
one. Run with ``--reasoning-parser qwen3`` the server returned two identical calls, and
the reasoning text had a hole where the draft was; run without a reasoning parser it
returned one. The reasoning parser used to lift ``<tool_call>`` blocks out of the
thinking into the content ("promotion"), so the tool parser saw the draft and the real
call. The contract, for every endpoint, streaming and not:

* the reasoning is separated first; calls are parsed from the content part only;
* call-shaped markup inside the reasoning stays reasoning, untouched;
* a call that exists only inside the reasoning is not a call (the same answer as with no
  reasoning parser, where the tool parsers strip the thinking before parsing);
* a generation cut off inside the thinking has no content, so it has no calls.

Every site carries its own copy of the reasoning/tool hand-off, so all of them are driven
for real here, with the real parsers and a fake engine.
"""

import json
import logging
import re

import pytest
from fastapi.testclient import TestClient

import vllm_mlx.server as srv
from vllm_mlx.api.models import ChatCompletionRequest
from vllm_mlx.api.responses_models import ResponsesRequest
from vllm_mlx.engine.base import GenerationOutput
from vllm_mlx.reasoning import get_parser
from vllm_mlx.tool_parsers import ToolParserManager

TEXT = "if a < b and c > d then <ok> & done"
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "echo_args",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
            },
        },
    }
]


def _xml_call(text: str = TEXT) -> str:
    return (
        "<tool_call>\n<function=echo_args>\n<parameter=text>\n"
        f"{text}\n</parameter>\n</function>\n</tool_call>"
    )


def _json_call(text: str = TEXT) -> str:
    return '<tool_call>\n{"name": "echo_args", "arguments": %s}\n</tool_call>' % (
        json.dumps({"text": text})
    )


# tool parser launch flags -> (enable_auto_tool_choice, --tool-call-parser, call format)
TOOL_CONFIGS = {
    "qwen3_coder": (True, "qwen3_coder", _xml_call),
    "hermes": (True, "hermes", _json_call),
    # no --tool-call-parser: the generic parser, which does not strip thinking itself
    "generic": (False, None, _xml_call),
}
SITES = ["chat", "anthropic", "responses"]
NON_STREAMING_SITES = ["chat", "responses"]


@pytest.fixture(params=list(TOOL_CONFIGS))
def tool_config(request, monkeypatch):
    auto, parser_name, make_call = TOOL_CONFIGS[request.param]
    monkeypatch.setattr(srv, "_enable_auto_tool_choice", auto)
    monkeypatch.setattr(srv, "_tool_call_parser", parser_name)
    monkeypatch.setattr(
        srv,
        "_tool_parser_instance",
        ToolParserManager.get_tool_parser(parser_name)() if parser_name else None,
    )
    monkeypatch.setattr(srv, "_reasoning_parser", get_parser("qwen3")())
    monkeypatch.setattr(srv, "_reasoning_parser_name", "qwen3")
    return make_call


def _scenarios(make_call):
    """Raw decoded model output. The template ends the prompt in ``<think>``, so the
    output starts inside the thinking."""
    call = make_call()
    draft = "Let me make the call:\n\n" + call + "\n\nThis should be fine."
    return {
        # the reported case: a draft in the thinking and the real call after it
        "draft-then-call": (f"The user wants it.\n\n{draft}\n</think>\n\n{call}", 1),
        # the same with an explicit opener
        "explicit-opener": (
            f"<think>\nThe user wants it.\n\n{draft}\n</think>\n\n{call}",
            1,
        ),
        # no draft: the baseline every config must keep
        "call-only": (f"The user wants it.\n</think>\n\n{call}", 1),
        # a call that exists only inside a closed thinking block, plain answer after it
        "draft-only": (f"The user wants it.\n\n{draft}\n</think>\n\nDone.", 0),
        # generation cut off inside the thinking, after the draft
        "cut-off-in-thinking": (f"The user wants it.\n\n{draft}", 0),
        # same, the draft itself cut off
        "cut-off-in-draft": ("The user wants it.\n\nLet me:\n\n" + call[:30], 0),
    }


SCENARIOS = list(_scenarios(_xml_call))
CHUNKINGS = ["whole", "tokens", "lines"]
# Special tokens (<think>, <tool_call>, ...) are single tokens in a real tokenizer, so
# they arrive whole. (A marker split across deltas is a separate matter: with no
# reasoning parser at all, the streaming fast path emits "<tool" ahead of "_call>".)
_TOKEN_RE = re.compile(
    r"</?think>|</?tool_call>|</?function(?:=[^>\s]*)?>|</?parameter(?:=[^>\s]*)?>|\s+|[^\s<]+|<"
)


def _chunks(raw: str, chunking: str) -> list[str]:
    if chunking == "whole":
        return [raw]
    if chunking == "tokens":
        return _TOKEN_RE.findall(raw)
    return raw.splitlines(keepends=True)


def _engine(
    raw: str,
    chunking: str = "whole",
    finish_reason: str = "stop",
    *,
    thinking_open_in_prompt: bool = True,
):
    deltas = _chunks(raw, chunking)
    # what the Qwen3.5/3.8 template renders: an open <think>, or a closed empty one
    # when thinking is switched off
    prompt = "<|im_start|>assistant\n" + (
        "<think>\n" if thinking_open_in_prompt else "<think>\n\n</think>\n\n"
    )

    def apply_chat_template(messages, **kwargs):
        return prompt

    async def stream_chat(**kwargs):
        text = ""
        for i, delta in enumerate(deltas):
            text += delta
            last = i == len(deltas) - 1
            yield GenerationOutput(
                text=text,
                new_text=delta,
                finished=last,
                finish_reason=finish_reason if last else None,
                prompt_tokens=7,
                completion_tokens=i + 1,
            )

    async def chat(**kwargs):
        return GenerationOutput(
            text=raw,
            new_text=raw,
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
            "is_mllm": False,
            "preserve_native_tool_format": False,
            "stream_chat": staticmethod(stream_chat),
            "chat": staticmethod(chat),
            "_apply_chat_template": staticmethod(apply_chat_template),
        },
    )()


def _request(stream: bool):
    return ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "call it"}],
        stream=stream,
        tools=TOOLS,
    )


def _sse_payloads(chunks):
    out = []
    for raw in chunks:
        for line in raw.splitlines():
            if line.startswith("data: "):
                body = line.removeprefix("data: ").strip()
                if body and body != "[DONE]":
                    out.append(json.loads(body))
    return out


class Seen:
    def __init__(self, reasoning, content, calls):
        self.reasoning = reasoning
        self.content = content
        self.calls = calls  # list of (name, parsed arguments)


def _call(name, arguments):
    return (name, json.loads(arguments) if isinstance(arguments, str) else arguments)


async def _run(
    site,
    monkeypatch,
    raw,
    chunking,
    *,
    stream,
    finish_reason="stop",
    thinking_open_in_prompt=True,
):
    engine = _engine(
        raw,
        chunking,
        finish_reason,
        thinking_open_in_prompt=thinking_open_in_prompt,
    )
    request = _request(stream)

    if site == "chat" and not stream:
        reasoning, content, calls = srv._extract_reasoning_and_tool_calls(
            raw, request, engine=engine, chat_kwargs={}
        )
        return Seen(
            reasoning or "",
            content or "",
            [_call(c.function.name, c.function.arguments) for c in calls or []],
        )

    if site == "chat":
        chunks = [
            c
            async for c in srv.stream_chat_completion(engine, request.messages, request)
        ]
        deltas = [
            p["choices"][0]["delta"] for p in _sse_payloads(chunks) if p.get("choices")
        ]
        by_index: dict[int, dict] = {}
        for delta in deltas:
            for tc in delta.get("tool_calls") or []:
                slot = by_index.setdefault(tc["index"], {"name": "", "args": ""})
                slot["name"] += tc["function"].get("name") or ""
                slot["args"] += tc["function"].get("arguments") or ""
        return Seen(
            "".join(
                d.get("reasoning") or d.get("reasoning_content") or "" for d in deltas
            ),
            "".join(d.get("content") or "" for d in deltas),
            [_call(s["name"], s["args"]) for s in by_index.values()],
        )

    if site == "anthropic" and not stream:
        pytest.skip(
            "the Anthropic non-streaming path is _extract_reasoning_and_tool_calls"
        )

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
        events = _sse_payloads(chunks)
        thinking = "".join(
            e["delta"]["thinking"]
            for e in events
            if e["type"] == "content_block_delta"
            and e["delta"]["type"] == "thinking_delta"
        )
        text = "".join(
            e["delta"]["text"]
            for e in events
            if e["type"] == "content_block_delta" and e["delta"]["type"] == "text_delta"
        )
        starts = {
            e["index"]: e["content_block"]
            for e in events
            if e["type"] == "content_block_start"
            and e["content_block"]["type"] == "tool_use"
        }
        args = {}
        for e in events:
            if e["type"] == "content_block_delta" and e["delta"]["type"] == (
                "input_json_delta"
            ):
                args[e["index"]] = args.get(e["index"], "") + e["delta"]["partial_json"]
        return Seen(
            thinking,
            text,
            [_call(b["name"], args.get(i, "{}")) for i, b in starts.items()],
        )

    # responses
    responses_request = ResponsesRequest(
        model="test-model", input="call it", stream=stream
    )
    if stream:
        monkeypatch.setattr(
            srv,
            "_prepare_streaming_responses_request",
            lambda req, engine=None: (engine_obj, request, request.messages, {}),
        )
        engine_obj = engine
        chunks = [c async for c in srv._stream_responses_request(responses_request)]
        completed = next(
            e["response"]
            for e in _sse_payloads(chunks)
            if e["type"] == "response.completed"
        )
    else:
        monkeypatch.setattr(
            srv,
            "_prepare_responses_request",
            lambda req, engine=None: (engine, request, request.messages, {}),
        )

        async def _no_wait(awaitable, *args, **kwargs):
            return await awaitable

        monkeypatch.setattr(srv, "_wait_with_disconnect", _no_wait)
        response, _ = await srv._run_responses_request(responses_request, None, engine)
        completed = response.model_dump()
    reasoning = "".join(
        part["text"]
        for item in completed["output"]
        if item["type"] == "reasoning"
        for part in item["content"]
    )
    content = "".join(
        part["text"]
        for item in completed["output"]
        if item["type"] == "message"
        for part in item["content"]
    )
    calls = [
        _call(item["name"], item["arguments"])
        for item in completed["output"]
        if item["type"] == "function_call"
    ]
    return Seen(reasoning, content, calls)


def _assert_contract(seen: Seen, raw: str, expected_calls: int, tool_config):
    assert seen.calls == [("echo_args", {"text": TEXT})] * expected_calls
    # the content never carries call markup or any of the thinking
    assert "<tool_call>" not in seen.content
    assert "<function=" not in seen.content
    assert "The user wants it." not in seen.content
    assert "</think>" not in seen.content and "</think>" not in seen.reasoning
    # call-shaped markup in the thinking stays in the thinking, byte for byte
    thinking = raw.split("</think>")[0].removeprefix("<think>\n")
    draft = thinking.split("Let me", 1)[1] if "Let me" in thinking else None
    if draft is not None:
        assert "<tool_call>" in seen.reasoning
        assert seen.reasoning.strip() == thinking.strip()


@pytest.mark.parametrize("scenario", SCENARIOS)
@pytest.mark.parametrize("site", NON_STREAMING_SITES)
@pytest.mark.anyio
async def test_non_streaming_parses_calls_from_the_content_only(
    monkeypatch, tool_config, site, scenario
):
    raw, expected_calls = _scenarios(tool_config)[scenario]

    seen = await _run(site, monkeypatch, raw, "whole", stream=False)

    _assert_contract(seen, raw, expected_calls, tool_config)


@pytest.mark.parametrize("chunking", CHUNKINGS)
@pytest.mark.parametrize("scenario", SCENARIOS)
@pytest.mark.parametrize("site", SITES)
@pytest.mark.anyio
async def test_streaming_parses_calls_from_the_content_only(
    monkeypatch, tool_config, site, scenario, chunking
):
    raw, expected_calls = _scenarios(tool_config)[scenario]

    seen = await _run(site, monkeypatch, raw, chunking, stream=True)

    _assert_contract(seen, raw, expected_calls, tool_config)


@pytest.mark.parametrize("scenario", SCENARIOS)
@pytest.mark.anyio
async def test_streaming_and_non_streaming_agree_on_chat(
    monkeypatch, tool_config, scenario
):
    raw, _ = _scenarios(tool_config)[scenario]

    whole = await _run("chat", monkeypatch, raw, "whole", stream=False)
    streamed = await _run("chat", monkeypatch, raw, "tokens", stream=True)

    assert streamed.calls == whole.calls
    assert streamed.reasoning.strip() == whole.reasoning.strip()
    assert streamed.content.strip() == whole.content.strip()


def test_draft_only_with_a_plain_answer_keeps_the_answer(monkeypatch, tool_config):
    raw, _ = _scenarios(tool_config)["draft-only"]

    reasoning, content, calls = srv._extract_reasoning_and_tool_calls(
        raw, _request(False), engine=_engine(raw), chat_kwargs={}
    )

    assert calls is None
    assert content == "Done."
    assert "<tool_call>" in reasoning


@pytest.mark.parametrize("site", NON_STREAMING_SITES)
@pytest.mark.anyio
async def test_a_plain_reply_after_a_template_closed_thinking_is_still_content(
    monkeypatch, tool_config, site
):
    # The template closed the thinking itself, so an output with no tag at all is the
    # answer, not a thinking cut off: its call is a call.
    raw = "Calling it.\n\n" + tool_config()

    seen = await _run(
        site, monkeypatch, raw, "whole", stream=False, thinking_open_in_prompt=False
    )

    assert seen.calls == [("echo_args", {"text": TEXT})]
    assert seen.reasoning == ""
    assert seen.content.strip() == "Calling it."


# --- the reasoning parsers themselves: nothing leaves the thinking --------------------

THINK_TAG_PARSERS = ["qwen3", "deepseek_r1", "glm4", "step3p5", "poolside_v1"]
# glm4 reads a tag split across deltas as content and step3p5 rewrites the whitespace
# around newlines: both are their own business, so compare modulo whitespace and feed
# the tags whole.
THINKING = "Plan.\n\nLet me call it:\n\n" + _xml_call() + "\n\nOk.\n\n" + _json_call()


def _squash(text: str | None) -> str:
    return "".join((text or "").split())


def _stream_parser(parser, chunks):
    parser.reset_state()
    reasoning, content, seen = "", "", ""
    for chunk in chunks:
        previous, seen = seen, seen + chunk
        delta = parser.extract_reasoning_streaming(previous, seen, chunk)
        if delta:
            reasoning += delta.reasoning or ""
            content += delta.content or ""
    final = parser.finalize_stream()
    if final:
        reasoning += final.reasoning or ""
        content += final.content or ""
    return reasoning, content


@pytest.mark.parametrize("chunking", ["whole", "tokens", "lines"])
@pytest.mark.parametrize("name", THINK_TAG_PARSERS)
def test_call_markup_in_the_thinking_is_reasoning_never_content(name, chunking):
    raw = f"<think>\n{THINKING}\n</think>\n\nThe answer."

    reasoning, content = get_parser(name)().extract_reasoning(raw)
    streamed_reasoning, streamed_content = _stream_parser(
        get_parser(name)(), _chunks(raw, chunking)
    )

    assert _squash(reasoning) == _squash(THINKING)
    assert content == "The answer."
    assert _squash(streamed_reasoning) == _squash(THINKING)
    assert streamed_content.strip() == "The answer."


@pytest.mark.parametrize("name", THINK_TAG_PARSERS)
def test_a_call_in_a_thinking_cut_off_before_it_closes_leaves_no_content(name):
    raw = f"<think>\n{THINKING}"

    reasoning, content = get_parser(name)().extract_reasoning(raw)
    streamed_reasoning, streamed_content = _stream_parser(
        get_parser(name)(), _chunks(raw, "tokens")
    )

    assert _squash(reasoning) == _squash(THINKING)
    assert content is None
    assert _squash(streamed_reasoning) == _squash(THINKING)
    assert streamed_content == ""


# --- the HTTP endpoints, as a client sees them ----------------------------------------
# The helper-level tests above hand the sites their inputs; these go through the real
# handlers, which is where the engine and the chat kwargs reach the reasoning split.

ENDPOINT_BODIES = {
    "/v1/chat/completions": lambda: {
        "model": "test-model",
        "messages": [{"role": "user", "content": "call it"}],
        "tools": TOOLS,
    },
    "/v1/responses": lambda: {
        "model": "test-model",
        "input": "call it",
        "tools": [{"type": "function", "name": "echo_args", **TOOLS[0]["function"]}],
    },
    "/v1/messages": lambda: {
        "model": "test-model",
        "max_tokens": 256,
        "messages": [{"role": "user", "content": "call it"}],
        "tools": [
            {
                "name": "echo_args",
                "input_schema": TOOLS[0]["function"]["parameters"],
            }
        ],
    },
}


def _endpoint_calls(path, payload):
    if path == "/v1/chat/completions":
        return payload["choices"][0]["message"].get("tool_calls") or []
    if path == "/v1/responses":
        return [o for o in payload["output"] if o["type"] == "function_call"]
    return [b for b in payload["content"] if b["type"] == "tool_use"]


@pytest.mark.parametrize("scenario", ["draft-then-call", "cut-off-in-thinking"])
@pytest.mark.parametrize("path", list(ENDPOINT_BODIES))
def test_endpoints_return_one_call_for_a_draft_then_call_and_none_for_a_cut_off(
    monkeypatch, tool_config, path, scenario
):
    raw, expected_calls = _scenarios(tool_config)[scenario]
    monkeypatch.setattr(srv, "_engine", _engine(raw, "tokens"))
    monkeypatch.setattr(srv, "_model_name", "test-model")

    response = TestClient(srv.app).post(path, json=ENDPOINT_BODIES[path]())

    assert response.status_code == 200
    assert len(_endpoint_calls(path, response.json())) == expected_calls


@pytest.mark.parametrize("scenario", ["draft-then-call", "cut-off-in-thinking"])
def test_the_chat_endpoint_streams_one_call_for_a_draft_then_call_and_none_for_a_cut_off(
    monkeypatch, tool_config, scenario
):
    raw, expected_calls = _scenarios(tool_config)[scenario]
    monkeypatch.setattr(srv, "_engine", _engine(raw, "tokens"))
    monkeypatch.setattr(srv, "_model_name", "test-model")

    response = TestClient(srv.app).post(
        "/v1/chat/completions",
        json={**ENDPOINT_BODIES["/v1/chat/completions"](), "stream": True},
    )

    deltas = [
        p["choices"][0]["delta"]
        for p in _sse_payloads(response.text.splitlines(True))
        if p.get("choices")
    ]
    indexes = {tc["index"] for d in deltas for tc in d.get("tool_calls") or []}
    assert len(indexes) == expected_calls


# --- the operator is told when a call was written only inside the thinking ------------

WARNING_TEXT = "Tool-call markup found in the reasoning"


def _call_in_thinking_warnings(caplog):
    return [
        r.getMessage()
        for r in caplog.records
        if r.levelname == "WARNING" and WARNING_TEXT in r.getMessage()
    ]


@pytest.mark.parametrize(
    ("scenario", "warned"),
    [
        # the model's only call is in the thinking, or it was cut off in it: no call
        # goes out, and the log is the only trace of why
        ("draft-only", 1),
        ("cut-off-in-thinking", 1),
        ("cut-off-in-draft", 1),
        # a real call follows, or there is no call-shaped markup at all: nothing to say
        ("draft-then-call", 0),
        ("call-only", 0),
    ],
)
@pytest.mark.parametrize("stream", [False, True], ids=["non-stream", "stream"])
@pytest.mark.parametrize("site", SITES)
@pytest.mark.anyio
async def test_a_call_that_exists_only_in_the_thinking_is_logged_once_without_its_text(
    monkeypatch, caplog, tool_config, site, stream, scenario, warned
):
    if site == "anthropic" and not stream:
        pytest.skip("covered through /v1/messages below")
    raw, _ = _scenarios(tool_config)[scenario]

    with caplog.at_level(logging.WARNING, logger="vllm_mlx.server"):
        await _run(site, monkeypatch, raw, "tokens", stream=stream)

    messages = _call_in_thinking_warnings(caplog)
    assert len(messages) == warned
    for message in messages:
        # sizes only: the reasoning and its arguments are not written to the log
        assert "if a < b" not in message and "echo_args" not in message


@pytest.mark.parametrize("path", list(ENDPOINT_BODIES))
def test_endpoints_log_a_call_that_exists_only_in_the_thinking_once(
    monkeypatch, caplog, tool_config, path
):
    raw, _ = _scenarios(tool_config)["draft-only"]
    monkeypatch.setattr(srv, "_engine", _engine(raw, "tokens"))
    monkeypatch.setattr(srv, "_model_name", "test-model")

    with caplog.at_level(logging.WARNING, logger="vllm_mlx.server"):
        response = TestClient(srv.app).post(path, json=ENDPOINT_BODIES[path]())

    assert response.status_code == 200
    assert len(_call_in_thinking_warnings(caplog)) == 1


@pytest.mark.anyio
async def test_no_warning_without_tools_or_for_plain_thinking(
    monkeypatch, caplog, tool_config
):
    plain = "Just thinking.\n</think>\n\nDone."
    markup_without_tools = _scenarios(tool_config)["draft-only"][0]
    request = _request(False)

    with caplog.at_level(logging.WARNING, logger="vllm_mlx.server"):
        srv._extract_reasoning_and_tool_calls(
            plain, request, engine=_engine(plain), chat_kwargs={}
        )
        no_tools = request.model_copy(update={"tools": None})
        srv._extract_reasoning_and_tool_calls(
            markup_without_tools, no_tools, engine=_engine(plain), chat_kwargs={}
        )
        refused = request.model_copy(update={"tool_choice": "none"})
        srv._extract_reasoning_and_tool_calls(
            markup_without_tools, refused, engine=_engine(plain), chat_kwargs={}
        )

    assert _call_in_thinking_warnings(caplog) == []


@pytest.mark.parametrize("marker", ["<tool_call>", "<function=echo_args>"])
def test_the_streaming_watch_sees_a_marker_split_across_deltas_and_warns_once(
    caplog, marker
):
    watch = srv._ReasoningMarkupWatch(
        ToolParserManager.get_tool_parser("qwen3_coder")()
    )

    with caplog.at_level(logging.WARNING, logger="vllm_mlx.server"):
        for char in "I will call it: " + marker + " now":
            watch.feed(char)
        watch.warn_if_no_call(False)

    assert watch.seen
    assert len(_call_in_thinking_warnings(caplog)) == 1


def test_the_streaming_watch_is_silent_with_a_call_or_without_markup(caplog):
    quiet = srv._ReasoningMarkupWatch(None)
    plain = srv._ReasoningMarkupWatch(ToolParserManager.get_tool_parser("hermes")())
    called = srv._ReasoningMarkupWatch(ToolParserManager.get_tool_parser("hermes")())

    with caplog.at_level(logging.WARNING, logger="vllm_mlx.server"):
        quiet.feed(_xml_call())  # no tool parser: tools were not declared
        quiet.warn_if_no_call(False)
        plain.feed("a < b and c > d, so call_it(x) is fine")
        plain.warn_if_no_call(False)
        called.feed(_xml_call())
        called.warn_if_no_call(True)

    assert _call_in_thinking_warnings(caplog) == []
