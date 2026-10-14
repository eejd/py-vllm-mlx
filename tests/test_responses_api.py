# SPDX-License-Identifier: Apache-2.0
"""Tests for the OpenAI-compatible Responses API."""

import json
import platform
import sys
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin" or platform.machine() != "arm64",
    reason="Requires Apple Silicon",
)


@pytest.fixture()
def client():
    from vllm_mlx.server import app

    return TestClient(app)


@pytest.fixture(autouse=True)
def server_state():
    import vllm_mlx.server as srv

    original_engine = srv._engine
    original_model_name = srv._model_name
    original_store = srv._responses_store
    original_store_max_size = srv._RESPONSES_STORE_MAX_SIZE
    original_api_key = srv._api_key
    original_default_chat_template_kwargs = getattr(
        srv, "_default_chat_template_kwargs", None
    )

    srv._engine = None
    srv._model_name = "test-model"
    srv._responses_store = OrderedDict()
    srv._RESPONSES_STORE_MAX_SIZE = 1000
    srv._api_key = None
    srv._default_chat_template_kwargs = None

    try:
        yield
    finally:
        srv._engine = original_engine
        srv._model_name = original_model_name
        srv._responses_store = original_store
        srv._RESPONSES_STORE_MAX_SIZE = original_store_max_size
        srv._api_key = original_api_key
        srv._default_chat_template_kwargs = original_default_chat_template_kwargs


def _mock_engine(*outputs):
    engine = MagicMock()
    engine.model_name = "test-model"
    engine.is_mllm = False
    engine.use_harmony_rendering = False
    engine.preserve_native_tool_format = False
    engine.chat = AsyncMock(side_effect=list(outputs))
    stream_calls = []

    async def _stream_chat(**kwargs):
        stream_calls.append(kwargs)
        for output in getattr(engine, "_stream_outputs", []):
            yield output

    engine._stream_calls = stream_calls
    engine._stream_outputs = []
    engine.stream_chat = _stream_chat
    return engine


def _output(
    text: str,
    prompt_tokens: int = 7,
    completion_tokens: int = 3,
    cached_tokens: int | None = None,
):
    out = SimpleNamespace(
        text=text,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        finish_reason="stop",
    )
    if cached_tokens is not None:
        out.cached_tokens = cached_tokens
    return out


def _stream_output(
    new_text: str,
    prompt_tokens: int = 7,
    completion_tokens: int = 1,
    finish_reason: str | None = None,
    cached_tokens: int | None = None,
):
    out = SimpleNamespace(
        new_text=new_text,
        text=new_text,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        finish_reason=finish_reason,
        finished=finish_reason is not None,
    )
    if cached_tokens is not None:
        out.cached_tokens = cached_tokens
    return out


def _parse_sse_events(body: str) -> list[tuple[str, dict]]:
    events = []
    for chunk in body.strip().split("\n\n"):
        if not chunk.strip():
            continue
        event_type = None
        payload = None
        for line in chunk.splitlines():
            if line.startswith("event: "):
                event_type = line.removeprefix("event: ").strip()
            elif line.startswith("data: "):
                payload = json.loads(line.removeprefix("data: ").strip())
        if event_type is not None and payload is not None:
            events.append((event_type, payload))
    return events


class TestResponsesEndpoint:
    def test_basic_response(self, client):
        import vllm_mlx.server as srv

        srv._engine = _mock_engine(_output("Hello there"))

        resp = client.post(
            "/v1/responses",
            json={"model": "test-model", "input": "Say hello"},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["object"] == "response"
        assert body["output_text"] == "Hello there"
        assert body["output"][0]["type"] == "message"
        assert body["output"][0]["content"][0]["type"] == "output_text"
        assert body["usage"]["input_tokens"] == 7
        assert body["usage"]["output_tokens"] == 3

    def test_responses_applies_server_default_chat_template_kwargs(self, client):
        import vllm_mlx.server as srv

        engine = _mock_engine(_output("Hello there"))
        srv._engine = engine
        srv._default_chat_template_kwargs = {"enable_thinking": False}

        resp = client.post(
            "/v1/responses",
            json={"model": "test-model", "input": "Say hello"},
        )

        assert resp.status_code == 200
        assert engine.chat.call_args.kwargs["chat_template_kwargs"] == {
            "enable_thinking": False
        }

    def test_streaming_responses_validates_remote_media_once(self, client, monkeypatch):
        import vllm_mlx.server as srv

        engine = _mock_engine()
        engine._stream_outputs = [_stream_output("Hello there", finish_reason="stop")]
        srv._engine = engine

        validate_calls = []

        def fake_validate(messages):
            validate_calls.append(messages)

        def fake_responses_to_chat(_request):
            return srv.ChatCompletionRequest(
                model="test-model",
                messages=[
                    srv.Message(
                        role="user",
                        content=[
                            {"type": "text", "text": "describe this"},
                            {
                                "type": "image_url",
                                "image_url": {"url": "https://example.com/image.png"},
                            },
                        ],
                    )
                ],
                max_tokens=8,
                stream=True,
            )

        monkeypatch.setattr(srv, "_validate_remote_media_urls", fake_validate)
        monkeypatch.setattr(
            srv, "_responses_request_to_chat_request", fake_responses_to_chat
        )

        resp = client.post(
            "/v1/responses",
            json={"model": "test-model", "input": "describe this", "stream": True},
        )

        assert resp.status_code == 200
        assert len(validate_calls) == 1

    def test_responses_request_kwargs_override_server_defaults(self, client):
        import vllm_mlx.server as srv

        engine = _mock_engine(_output("Hello there"))
        srv._engine = engine
        srv._default_chat_template_kwargs = {
            "enable_thinking": False,
            "server_default_only": "yes",
        }

        resp = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "input": "Say hello",
                "chat_template_kwargs": {
                    "enable_thinking": True,
                    "request_only": 1,
                },
            },
        )

        assert resp.status_code == 200
        assert engine.chat.call_args.kwargs["chat_template_kwargs"] == {
            "enable_thinking": True,
            "server_default_only": "yes",
            "request_only": 1,
        }

    def test_previous_response_id_reuses_prior_context(self, client):
        import vllm_mlx.server as srv

        engine = _mock_engine(_output("First answer"), _output("Second answer"))
        srv._engine = engine

        first = client.post(
            "/v1/responses",
            json={"model": "test-model", "input": "First prompt"},
        )
        first_id = first.json()["id"]

        second = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "previous_response_id": first_id,
                "input": "Follow-up prompt",
            },
        )

        assert second.status_code == 200
        second_messages = engine.chat.call_args_list[1].kwargs["messages"]
        assert second_messages[0]["role"] == "user"
        assert second_messages[0]["content"] == "First prompt"
        assert second_messages[1]["role"] == "assistant"
        assert second_messages[1]["content"] == "First answer"
        assert second_messages[2]["role"] == "user"
        assert second_messages[2]["content"] == "Follow-up prompt"

    def test_previous_response_id_chains_across_multiple_follow_ups(self, client):
        import vllm_mlx.server as srv

        engine = _mock_engine(
            _output("First answer"),
            _output("Second answer"),
            _output("Third answer"),
        )
        srv._engine = engine

        first = client.post(
            "/v1/responses",
            json={"model": "test-model", "input": "First prompt"},
        )
        first_id = first.json()["id"]

        second = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "previous_response_id": first_id,
                "input": "Second prompt",
            },
        )
        second_id = second.json()["id"]

        third = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "previous_response_id": second_id,
                "input": "Third prompt",
            },
        )

        assert third.status_code == 200
        third_messages = engine.chat.call_args_list[2].kwargs["messages"]
        assert third_messages[0]["role"] == "user"
        assert third_messages[0]["content"] == "First prompt"
        assert third_messages[1]["role"] == "assistant"
        assert third_messages[1]["content"] == "First answer"
        assert third_messages[2]["role"] == "user"
        assert third_messages[2]["content"] == "Second prompt"
        assert third_messages[3]["role"] == "assistant"
        assert third_messages[3]["content"] == "Second answer"
        assert third_messages[4]["role"] == "user"
        assert third_messages[4]["content"] == "Third prompt"

    def test_previous_response_id_does_not_carry_prior_instructions(self, client):
        import vllm_mlx.server as srv

        engine = _mock_engine(_output("First answer"), _output("Second answer"))
        srv._engine = engine

        first = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "instructions": "First system instruction",
                "input": "First prompt",
            },
        )
        first_id = first.json()["id"]

        second = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "instructions": "Second system instruction",
                "previous_response_id": first_id,
                "input": "Follow-up prompt",
            },
        )

        assert second.status_code == 200
        second_messages = engine.chat.call_args_list[1].kwargs["messages"]
        assert second_messages[0]["role"] == "system"
        assert second_messages[0]["content"] == "Second system instruction"
        assert "First system instruction" not in second_messages[0]["content"]
        assert second_messages[1]["role"] == "user"
        assert second_messages[1]["content"] == "First prompt"
        assert second_messages[2]["role"] == "assistant"
        assert second_messages[3]["role"] == "user"

    def test_previous_response_id_preserves_prior_system_message_items(self, client):
        import vllm_mlx.server as srv

        engine = _mock_engine(_output("First answer"), _output("Second answer"))
        srv._engine = engine

        first = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "input": [
                    {"type": "message", "role": "system", "content": "Persist me"},
                    {"type": "message", "role": "user", "content": "First prompt"},
                ],
            },
        )
        first_id = first.json()["id"]

        second = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "previous_response_id": first_id,
                "input": "Follow-up prompt",
            },
        )

        assert second.status_code == 200
        second_messages = engine.chat.call_args_list[1].kwargs["messages"]
        assert second_messages[0]["role"] == "system"
        assert second_messages[0]["content"] == "Persist me"
        assert second_messages[1]["role"] == "user"
        assert second_messages[1]["content"] == "First prompt"
        assert second_messages[2]["role"] == "assistant"
        assert second_messages[3]["role"] == "user"

    def test_previous_response_id_missing_returns_404(self, client):
        import vllm_mlx.server as srv

        srv._engine = _mock_engine(_output("unused"))

        resp = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "previous_response_id": "resp_missing",
                "input": "Follow-up prompt",
            },
        )

        assert resp.status_code == 404
        assert "resp_missing" in resp.json()["detail"]

    def test_developer_role_is_normalized_to_system(self, client):
        import vllm_mlx.server as srv

        engine = _mock_engine(_output("Ready"))
        srv._engine = engine

        resp = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "input": [
                    {"type": "message", "role": "user", "content": "Hi"},
                    {"type": "message", "role": "developer", "content": "Be terse"},
                ],
            },
        )

        assert resp.status_code == 200
        messages = engine.chat.call_args.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "Be terse"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Hi"

    def test_system_prompt_canonicalization_strips_billing_header(self, client):
        import vllm_mlx.server as srv

        engine = _mock_engine(_output("Ready"))
        srv._engine = engine

        resp = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "instructions": (
                    "Be terse.\n"
                    "x-anthropic-billing-header: account=abc; cch=rotating-hash\n"
                    "Answer directly."
                ),
                "input": "Say hello",
            },
        )

        assert resp.status_code == 200
        messages = engine.chat.call_args.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "Be terse.\nAnswer directly."

    def test_instructions_and_developer_message_are_merged(self, client):
        import vllm_mlx.server as srv

        engine = _mock_engine(_output("Ready"))
        srv._engine = engine

        resp = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "instructions": "System instructions",
                "input": [
                    {
                        "type": "message",
                        "role": "developer",
                        "content": "Developer note",
                    },
                    {"type": "message", "role": "user", "content": "Hi"},
                ],
            },
        )

        assert resp.status_code == 200
        messages = engine.chat.call_args.kwargs["messages"]
        assert len([m for m in messages if m["role"] == "system"]) == 1
        assert messages[0]["role"] == "system"
        assert "System instructions" in messages[0]["content"]
        assert "Developer note" in messages[0]["content"]
        assert messages[1]["role"] == "user"

    def test_function_call_output_input_is_mapped_cleanly(self, client):
        import vllm_mlx.server as srv

        engine = _mock_engine(_output("Done"))
        srv._engine = engine

        resp = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "input": [
                    {"type": "message", "role": "user", "content": "Run it"},
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "shell",
                        "arguments": '{"cmd":"pwd"}',
                    },
                    {
                        "type": "function_call_output",
                        "call_id": "call_1",
                        "output": "/tmp/work",
                    },
                ],
            },
        )

        assert resp.status_code == 200
        messages = engine.chat.call_args.kwargs["messages"]
        assert messages[1]["role"] == "assistant"
        assert "[Calling tool: shell(" in messages[1]["content"]
        assert messages[2]["role"] == "user"
        assert "[Tool Result (call_1)]" in messages[2]["content"]
        assert "/tmp/work" in messages[2]["content"]

    def test_unsupported_tools_and_items_do_not_fail(self, client):
        import vllm_mlx.server as srv

        engine = _mock_engine(_output("Fallback answer"))
        srv._engine = engine

        resp = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "input": [
                    {"type": "message", "role": "user", "content": "Answer directly"},
                    {
                        "type": "web_search_call",
                        "status": "completed",
                        "action": {"type": "search", "query": "ignored"},
                    },
                ],
                "tools": [
                    {"type": "web_search_preview"},
                    {"type": "file_search", "vector_store_ids": ["vs_123"]},
                    {
                        "type": "function",
                        "name": "shell",
                        "parameters": {"type": "object", "properties": {}},
                    },
                ],
            },
        )

        assert resp.status_code == 200
        messages = engine.chat.call_args.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert "not available on this backend" in messages[0]["content"]
        assert messages[1]["role"] == "user"
        assert engine.chat.call_args.kwargs["tools"][0]["type"] == "function"

    def test_function_call_response_item(self, client):
        import vllm_mlx.server as srv

        srv._engine = _mock_engine(
            _output('<tool_call>{"name":"shell","arguments":{"cmd":"pwd"}}</tool_call>')
        )

        resp = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "input": "Use a tool",
                "tools": [
                    {
                        "type": "function",
                        "name": "shell",
                        "parameters": {"type": "object", "properties": {}},
                    }
                ],
            },
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["output"][0]["type"] == "function_call"
        assert body["output"][0]["name"] == "shell"
        assert body["output_text"] == ""

    def test_store_false_skips_persistence(self, client):
        import vllm_mlx.server as srv

        srv._engine = _mock_engine(_output("Ephemeral answer"))

        first = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "input": "Do not store this",
                "store": False,
            },
        )

        assert first.status_code == 200
        assert first.json()["store"] is False
        assert first.json()["id"] not in srv._responses_store

        second = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "previous_response_id": first.json()["id"],
                "input": "Follow-up prompt",
            },
        )

        assert second.status_code == 404

    def test_responses_store_is_lru_bounded(self, client):
        import vllm_mlx.server as srv

        srv._RESPONSES_STORE_MAX_SIZE = 2
        srv._engine = _mock_engine(
            _output("First answer"),
            _output("Second answer"),
            _output("Third answer"),
        )

        first = client.post(
            "/v1/responses",
            json={"model": "test-model", "input": "First prompt"},
        )
        second = client.post(
            "/v1/responses",
            json={"model": "test-model", "input": "Second prompt"},
        )
        third = client.post(
            "/v1/responses",
            json={"model": "test-model", "input": "Third prompt"},
        )

        assert first.status_code == 200
        assert second.status_code == 200
        assert third.status_code == 200
        assert list(srv._responses_store) == [second.json()["id"], third.json()["id"]]
        assert first.json()["id"] not in srv._responses_store

    def test_streaming_response_returns_sse_events(self, client):
        import vllm_mlx.server as srv

        engine = _mock_engine(_output("unused"))
        engine.chat = AsyncMock(
            side_effect=AssertionError("stream path should not call chat")
        )
        engine._stream_outputs = [
            _stream_output("Hello ", completion_tokens=1),
            _stream_output("stream", completion_tokens=2, finish_reason="stop"),
        ]
        srv._engine = engine

        with client.stream(
            "POST",
            "/v1/responses",
            json={"model": "test-model", "input": "Hello", "stream": True},
        ) as resp:
            body = "".join(resp.iter_text())

        assert resp.status_code == 200
        assert "event: response.created" in body
        assert "event: response.output_text.delta" in body
        assert "Hello stream" in body
        assert "event: response.completed" in body
        assert len(engine._stream_calls) == 1
        engine.chat.assert_not_awaited()

    def test_streaming_reasoning_flushes_partial_marker(self, client, monkeypatch):
        import vllm_mlx.server as srv

        engine = _mock_engine(_output("unused"))
        engine.chat = AsyncMock(
            side_effect=AssertionError("stream path should not call chat")
        )
        engine._stream_outputs = [
            _stream_output("thinking</thi", completion_tokens=2, finish_reason="stop")
        ]
        srv._engine = engine
        monkeypatch.setattr(srv, "_reasoning_parser_name", "qwen3")
        monkeypatch.setattr(srv, "_reasoning_parser", None)

        with client.stream(
            "POST",
            "/v1/responses",
            json={"model": "test-model", "input": "Think", "stream": True},
        ) as resp:
            body = "".join(resp.iter_text())

        assert resp.status_code == 200
        events = _parse_sse_events(body)
        reasoning = "".join(
            payload["delta"]
            for event_type, payload in events
            if event_type == "response.reasoning_text.delta"
        )
        event_types = [event_type for event_type, _ in events]

        assert reasoning == "thinking</thi"
        assert event_types.index("response.reasoning_text.delta") < event_types.index(
            "response.completed"
        )

    def test_streaming_response_sequence_metadata_is_monotonic(self, client):
        import vllm_mlx.server as srv

        engine = _mock_engine(_output("unused"))
        engine.chat = AsyncMock(
            side_effect=AssertionError("stream path should not call chat")
        )
        engine._stream_outputs = [
            _stream_output("Hello ", completion_tokens=1),
            _stream_output("stream", completion_tokens=2, finish_reason="stop"),
        ]
        srv._engine = engine

        with client.stream(
            "POST",
            "/v1/responses",
            json={"model": "test-model", "input": "Hello", "stream": True},
        ) as resp:
            body = "".join(resp.iter_text())

        assert resp.status_code == 200
        events = _parse_sse_events(body)
        assert [event_type for event_type, _ in events[:2]] == [
            "response.created",
            "response.in_progress",
        ]
        sequence_numbers = [payload["sequence_number"] for _, payload in events]
        assert sequence_numbers == sorted(sequence_numbers)
        created_payload = events[0][1]
        completed_payload = next(
            payload
            for event_type, payload in events
            if event_type == "response.completed"
        )
        assert created_payload["response"]["id"] == completed_payload["response"]["id"]
        assert completed_payload["response"]["output_text"] == "Hello stream"

    def test_streaming_response_bracket_tool_call_does_not_leak_text(
        self, client, monkeypatch
    ):
        import vllm_mlx.server as srv

        engine = _mock_engine(_output("unused"))
        engine.chat = AsyncMock(
            side_effect=AssertionError("stream path should not call chat")
        )
        engine._stream_outputs = [
            _stream_output('[Calling tool: add({"a": 1, "b": 2})'),
            _stream_output("]", completion_tokens=2, finish_reason="stop"),
        ]
        srv._engine = engine
        monkeypatch.setattr(srv, "_enable_auto_tool_choice", True)
        monkeypatch.setattr(srv, "_tool_call_parser", "qwen3")
        monkeypatch.setattr(srv, "_tool_parser_instance", None)
        monkeypatch.setattr(srv, "_reasoning_parser", None)

        with client.stream(
            "POST",
            "/v1/responses",
            json={
                "model": "test-model",
                "input": "Add two numbers",
                "stream": True,
                "tools": [
                    {
                        "type": "function",
                        "name": "add",
                        "description": "Add two numbers",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "a": {"type": "integer"},
                                "b": {"type": "integer"},
                            },
                            "required": ["a", "b"],
                        },
                    }
                ],
            },
        ) as resp:
            body = "".join(resp.iter_text())

        assert resp.status_code == 200
        events = _parse_sse_events(body)
        output_text_deltas = [
            payload["delta"]
            for event_type, payload in events
            if event_type == "response.output_text.delta"
        ]
        function_call_deltas = [
            payload
            for event_type, payload in events
            if event_type == "response.function_call_arguments.delta"
        ]

        assert not any("[Calling tool:" in delta for delta in output_text_deltas)
        assert len(function_call_deltas) == 1
        assert function_call_deltas[0]["delta"] == '{"a": 1, "b": 2}'

    def test_deepseek_v4_reasoning_routes_dsml_through_tool_parser(
        self, client, monkeypatch
    ):
        import vllm_mlx.server as srv

        d = "｜DSML｜"
        engine = _mock_engine(_output("unused"))
        engine._stream_outputs = [
            _stream_output("thinking</think>Checking.\n\n"),
            _stream_output("<"),
            _stream_output(d),
            _stream_output("tool_calls>\n"),
            _stream_output(f'<{d}invoke name="get_weather">\n'),
            _stream_output(
                f'<{d}parameter name="city" string="true">Prague' f"</{d}parameter>\n"
            ),
            _stream_output(f"</{d}invoke>\n"),
            _stream_output(f"</{d}tool_calls>", finish_reason="stop"),
        ]
        srv._engine = engine
        monkeypatch.setattr(srv, "_enable_auto_tool_choice", True)
        monkeypatch.setattr(srv, "_tool_call_parser", "deepseek_v4")
        monkeypatch.setattr(srv, "_tool_parser_instance", None)
        monkeypatch.setattr(srv, "_reasoning_parser_name", "deepseek_v4")
        monkeypatch.setattr(srv, "_reasoning_parser", None)

        with client.stream(
            "POST",
            "/v1/responses",
            json={
                "model": "test-model",
                "input": "Check the weather",
                "stream": True,
                "tools": [
                    {
                        "type": "function",
                        "name": "get_weather",
                        "parameters": {"type": "object"},
                    }
                ],
            },
        ) as resp:
            body = "".join(resp.iter_text())

        events = _parse_sse_events(body)
        text_deltas = [
            payload["delta"]
            for event_type, payload in events
            if event_type == "response.output_text.delta"
        ]
        function_items = [
            payload["item"]
            for event_type, payload in events
            if event_type == "response.output_item.added"
            and payload["item"]["type"] == "function_call"
        ]
        completed = next(
            payload["response"]
            for event_type, payload in events
            if event_type == "response.completed"
        )

        assert resp.status_code == 200
        assert "".join(text_deltas).strip() == "Checking."
        assert d not in "".join(text_deltas)
        assert d not in completed["output_text"]
        assert len(function_items) == 1
        assert function_items[0]["name"] == "get_weather"

    def test_streaming_response_without_tools_keeps_llama_shaped_json(
        self, client, monkeypatch
    ):
        """Configured parsing must not suppress JSON when no tools are declared."""
        import vllm_mlx.server as srv

        text = '{"name":"Alice","parameters":{"age":42}}'
        engine = _mock_engine(_output("unused"))
        engine._stream_outputs = [_stream_output(text, finish_reason="stop")]
        srv._engine = engine
        monkeypatch.setattr(srv, "_enable_auto_tool_choice", True)
        monkeypatch.setattr(srv, "_tool_call_parser", "llama")
        monkeypatch.setattr(srv, "_tool_parser_instance", None)
        monkeypatch.setattr(srv, "_reasoning_parser", None)

        with client.stream(
            "POST",
            "/v1/responses",
            json={"model": "test-model", "input": "Describe Alice", "stream": True},
        ) as resp:
            body = "".join(resp.iter_text())

        events = _parse_sse_events(body)
        text_deltas = [
            payload["delta"]
            for event_type, payload in events
            if event_type == "response.output_text.delta"
        ]

        assert resp.status_code == 200
        assert text_deltas == [text]
        assert not any(
            event_type.startswith("response.function_call") for event_type, _ in events
        )

    def test_json_object_response_format_is_rejected(self, client):
        import vllm_mlx.server as srv

        srv._engine = _mock_engine(_output("Hello"))

        resp = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "input": "Hello",
                "text": {"format": {"type": "json_object"}},
            },
        )

        assert resp.status_code == 400
        assert "json_object" in resp.json()["detail"]

    def test_reasoning_configuration_is_ignored(self, client):
        import vllm_mlx.server as srv

        engine = _mock_engine(_output("Hello"))
        srv._engine = engine

        resp = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "input": "Hello",
                "reasoning": {"effort": "xhigh"},
            },
        )

        assert resp.status_code == 200
        assert engine.chat.await_count == 1

    def test_reasoning_input_item_is_accepted(self, client):
        import vllm_mlx.server as srv

        engine = _mock_engine(_output("Hello"))
        srv._engine = engine

        resp = client.post(
            "/v1/responses",
            json={
                "model": "test-model",
                "input": [
                    {"type": "message", "role": "user", "content": "Hello"},
                    {
                        "type": "reasoning",
                        "content": [{"type": "reasoning_text", "text": "x"}],
                    },
                ],
            },
        )

        assert resp.status_code == 200
        messages = engine.chat.call_args.kwargs["messages"]
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "x"

    def test_length_finish_reason_marks_response_incomplete(self, client):
        import vllm_mlx.server as srv

        output = _output("Cut off", completion_tokens=5)
        output.finish_reason = "length"
        srv._engine = _mock_engine(output)

        resp = client.post(
            "/v1/responses",
            json={"model": "test-model", "input": "Hello", "max_output_tokens": 5},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "incomplete"
        assert body["incomplete_details"] == {"reason": "max_output_tokens"}


class TestResponsesCachedTokens:
    """usage.input_tokens_details.cached_tokens is the engine's value, or absent."""

    def test_reported_cached_tokens_reach_the_response(self, client):
        import vllm_mlx.server as srv

        srv._engine = _mock_engine(_output("Hi", prompt_tokens=40, cached_tokens=39))
        body = client.post(
            "/v1/responses", json={"model": "test-model", "input": "hello"}
        ).json()
        assert body["usage"]["input_tokens_details"] == {"cached_tokens": 39}

    def test_a_reported_zero_is_kept(self, client):
        import vllm_mlx.server as srv

        srv._engine = _mock_engine(_output("Hi", cached_tokens=0))
        body = client.post(
            "/v1/responses", json={"model": "test-model", "input": "hello"}
        ).json()
        assert body["usage"]["input_tokens_details"] == {"cached_tokens": 0}

    def test_unreported_cached_tokens_are_omitted_not_zero(self, client):
        import vllm_mlx.server as srv

        srv._engine = _mock_engine(_output("Hi"))
        body = client.post(
            "/v1/responses", json={"model": "test-model", "input": "hello"}
        ).json()
        assert "input_tokens_details" not in body["usage"]
        assert body["usage"]["output_tokens_details"] == {"reasoning_tokens": 0}

    def test_streaming_completed_event_carries_the_last_reported_value(self, client):
        import vllm_mlx.server as srv

        engine = _mock_engine()
        engine._stream_outputs = [
            _stream_output("Hel", prompt_tokens=40, cached_tokens=39),
            _stream_output("lo", prompt_tokens=40, finish_reason="stop"),
        ]
        srv._engine = engine
        resp = client.post(
            "/v1/responses",
            json={"model": "test-model", "input": "hello", "stream": True},
        )
        events = _parse_sse_events(resp.text)
        completed = [p for t, p in events if t == "response.completed"]
        assert completed[0]["response"]["usage"]["input_tokens_details"] == {
            "cached_tokens": 39
        }

    def test_streaming_without_a_report_omits_the_details(self, client):
        import vllm_mlx.server as srv

        engine = _mock_engine()
        engine._stream_outputs = [_stream_output("Hi", finish_reason="stop")]
        srv._engine = engine
        resp = client.post(
            "/v1/responses",
            json={"model": "test-model", "input": "hello", "stream": True},
        )
        events = _parse_sse_events(resp.text)
        completed = [p for t, p in events if t == "response.completed"]
        assert "input_tokens_details" not in completed[0]["response"]["usage"]

    def test_stored_response_keeps_the_details(self, client):
        import vllm_mlx.server as srv

        srv._engine = _mock_engine(_output("Hi", cached_tokens=5))
        body = client.post(
            "/v1/responses", json={"model": "test-model", "input": "hello"}
        ).json()
        stored = srv._responses_store[body["id"]]["response"]
        assert stored.usage.input_tokens_details.cached_tokens == 5


class TestResponsesRegistryRouting:
    """Registry mode: /v1/responses serves request.model through the manager (#54)."""

    @pytest.fixture()
    def registry(self, monkeypatch):
        import vllm_mlx.server as srv
        from vllm_mlx.model_registry import ModelLease

        engines = {
            "alpha": _mock_engine(_output("from alpha", cached_tokens=11)),
            "beta": _mock_engine(_output("from beta")),
        }
        for name, eng in engines.items():
            eng._stream_outputs = [
                _stream_output(f"stream {name}", finish_reason="stop", cached_tokens=3)
            ]
        manager = SimpleNamespace(
            active_requests=0,
            acquired=[],
            registered_model_names=list(engines),
            has_model=lambda name: name in engines,
        )

        async def acquire(model_name):
            if model_name == "broken":
                raise RuntimeError("model failed to load")
            manager.active_requests += 1
            manager.acquired.append(model_name)

            async def release():
                manager.active_requests -= 1

            return ModelLease(manager, model_name, engines[model_name], release)

        manager.acquire = acquire
        manager.engines = engines
        monkeypatch.setattr(srv, "_model_manager", manager)
        monkeypatch.setattr(srv, "_model_name", None)
        monkeypatch.setattr(srv, "_active_request_contexts", {})
        srv._engine = None  # nothing global to fall back on
        return manager

    def test_each_request_is_served_by_its_own_model(self, client, registry):
        for name, text in (("alpha", "from alpha"), ("beta", "from beta")):
            resp = client.post("/v1/responses", json={"model": name, "input": "hi"})
            assert resp.status_code == 200, resp.text
            body = resp.json()
            assert body["output"][0]["content"][0]["text"] == text
        assert registry.acquired == ["alpha", "beta"]
        assert registry.engines["alpha"].chat.await_count == 1
        assert registry.engines["beta"].chat.await_count == 1

    def test_cached_tokens_come_from_the_requested_models_engine(
        self, client, registry
    ):
        body = client.post(
            "/v1/responses", json={"model": "alpha", "input": "hi"}
        ).json()
        assert body["usage"]["input_tokens_details"] == {"cached_tokens": 11}
        body = client.post(
            "/v1/responses", json={"model": "beta", "input": "hi"}
        ).json()
        assert "input_tokens_details" not in body["usage"]

    def test_lease_is_released_after_a_response(self, client, registry):
        import vllm_mlx.server as srv

        client.post("/v1/responses", json={"model": "alpha", "input": "hi"})
        assert registry.active_requests == 0
        assert srv._active_request_contexts == {}

    def test_streaming_is_served_by_the_requested_model_and_released(
        self, client, registry
    ):
        import vllm_mlx.server as srv

        resp = client.post(
            "/v1/responses", json={"model": "beta", "input": "hi", "stream": True}
        )
        assert resp.status_code == 200, resp.text
        events = _parse_sse_events(resp.text)
        completed = [p for t, p in events if t == "response.completed"]
        assert "stream beta" in json.dumps(completed[0])
        assert completed[0]["response"]["usage"]["input_tokens_details"] == {
            "cached_tokens": 3
        }
        assert registry.engines["beta"]._stream_calls
        assert not registry.engines["alpha"]._stream_calls
        assert registry.active_requests == 0
        assert srv._active_request_contexts == {}

    def test_engine_failure_releases_the_lease(self, client, registry):
        registry.engines["alpha"].chat = AsyncMock(side_effect=RuntimeError("boom"))
        with pytest.raises(RuntimeError):
            client.post("/v1/responses", json={"model": "alpha", "input": "hi"})
        assert registry.active_requests == 0

    def test_unknown_model_is_rejected_without_a_lease(self, client, registry):
        resp = client.post("/v1/responses", json={"model": "nope", "input": "hi"})
        assert resp.status_code == 404
        assert registry.acquired == []
        assert registry.active_requests == 0

    def test_load_failure_is_503_and_leaks_nothing(self, client, registry):
        registry.has_model = lambda name: True
        resp = client.post("/v1/responses", json={"model": "broken", "input": "hi"})
        assert resp.status_code == 503
        assert "failed to load" in resp.text
        assert registry.active_requests == 0

    def test_blocked_media_url_releases_the_lease(self, client, registry, monkeypatch):
        import vllm_mlx.server as srv
        from vllm_mlx.models.mllm import UnsafeRemoteURLError

        def blocked(_messages):
            raise UnsafeRemoteURLError("private address")

        monkeypatch.setattr(srv, "_validate_remote_media_urls", blocked)
        for stream in (False, True):
            resp = client.post(
                "/v1/responses",
                json={"model": "alpha", "input": "hi", "stream": stream},
            )
            assert resp.status_code == 400, resp.text
            assert registry.acquired.count("alpha") == (2 if stream else 1)
            assert registry.active_requests == 0


_PNG = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4"
    "nGP4z8DwHwAFAAH/q842iQAAAABJRU5ErkJggg=="
)
_PNG_2 = _PNG.replace("iVBORw0KGgo", "iVBORw0KGgp")


def _user_input(*parts):
    return [{"type": "message", "role": "user", "content": list(parts)}]


def _text(text):
    return {"type": "input_text", "text": text}


def _image(url=_PNG, **extra):
    return {"type": "input_image", "image_url": url, **extra}


class TestResponsesInputImages:
    """``input_image`` parts reach the engine exactly as chat-route images do."""

    def _post(self, client, engine, input_items, **extra):
        import vllm_mlx.server as srv

        srv._engine = engine
        return client.post(
            "/v1/responses",
            json={"model": "test-model", "input": input_items, **extra},
        )

    def _chat_call(self, client, engine, content):
        import vllm_mlx.server as srv

        srv._engine = engine
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": content}],
            },
        )
        assert resp.status_code == 200, resp.text
        return engine.chat.call_args.kwargs

    def test_llm_engine_gets_text_and_the_extracted_image(self, client):
        engine = _mock_engine(_output("a pixel"))
        resp = self._post(client, engine, _user_input(_text("What is this?"), _image()))

        assert resp.status_code == 200, resp.text
        kwargs = engine.chat.call_args.kwargs
        assert kwargs["messages"] == [{"role": "user", "content": "What is this?"}]
        assert kwargs["images"] == [_PNG]

    def test_llm_engine_matches_the_chat_route(self, client):
        responses_engine = _mock_engine(_output("a"))
        self._post(client, responses_engine, _user_input(_text("Describe"), _image()))
        chat_engine = _mock_engine(_output("a"))
        chat = self._chat_call(
            client,
            chat_engine,
            [
                {"type": "text", "text": "Describe"},
                {"type": "image_url", "image_url": {"url": _PNG}},
            ],
        )

        got = responses_engine.chat.call_args.kwargs
        assert got["messages"] == chat["messages"]
        assert got["images"] == chat["images"]

    def test_mllm_engine_keeps_the_image_in_the_message_like_the_chat_route(
        self, client
    ):
        # An MLLM engine reads media out of the messages; its chat() has no use
        # for an ``images`` argument, so flattening the message would drop it.
        responses_engine = _mock_engine(_output("a"))
        responses_engine.is_mllm = True
        self._post(
            client,
            responses_engine,
            _user_input(_text("Describe"), _image(detail="high")),
        )
        chat_engine = _mock_engine(_output("a"))
        chat_engine.is_mllm = True
        chat = self._chat_call(
            client,
            chat_engine,
            [
                {"type": "text", "text": "Describe"},
                {"type": "image_url", "image_url": {"url": _PNG, "detail": "high"}},
            ],
        )

        got = responses_engine.chat.call_args.kwargs
        assert got["messages"] == chat["messages"]
        assert got["messages"][0]["content"][1]["image_url"] == {
            "url": _PNG,
            "detail": "high",
        }
        assert "images" not in got

    def test_image_only_message_is_not_an_empty_conversation(self, client):
        engine = _mock_engine(_output("a"))
        resp = self._post(client, engine, _user_input(_image()))

        assert resp.status_code == 200, resp.text
        assert engine.chat.call_args.kwargs["images"] == [_PNG]

    def test_image_detail_defaults_to_auto(self, client):
        engine = _mock_engine(_output("a"))
        engine.is_mllm = True
        self._post(client, engine, _user_input(_image()))

        part = engine.chat.call_args.kwargs["messages"][0]["content"][0]
        assert part["image_url"]["detail"] == "auto"

    def test_parts_keep_their_request_order(self, client):
        engine = _mock_engine(_output("a"))
        engine.is_mllm = True
        self._post(
            client,
            engine,
            _user_input(_text("one"), _image(_PNG), _text("two"), _image(_PNG_2)),
        )

        content = engine.chat.call_args.kwargs["messages"][0]["content"]
        assert [part.get("text") or part["image_url"]["url"] for part in content] == [
            "one",
            _PNG,
            "two",
            _PNG_2,
        ]

    def test_llm_engine_gets_every_image_in_order(self, client):
        engine = _mock_engine(_output("a"))
        self._post(
            client, engine, _user_input(_image(_PNG), _text("vs"), _image(_PNG_2))
        )

        assert engine.chat.call_args.kwargs["images"] == [_PNG, _PNG_2]

    def test_streaming_request_carries_the_image(self, client):
        engine = _mock_engine()
        engine._stream_outputs = [_stream_output("ok", finish_reason="stop")]
        resp = self._post(
            client, engine, _user_input(_text("Describe"), _image()), stream=True
        )

        assert resp.status_code == 200, resp.text
        call = engine._stream_calls[0]
        assert call["messages"] == [{"role": "user", "content": "Describe"}]
        assert call["images"] == [_PNG]

    def test_message_item_without_a_type_is_a_message(self, client):
        # EasyInputMessage: ``type`` is optional when there is a role.
        engine = _mock_engine(_output("a"))
        resp = self._post(
            client,
            engine,
            [{"role": "user", "content": [_text("Describe"), _image()]}],
        )

        assert resp.status_code == 200, resp.text
        kwargs = engine.chat.call_args.kwargs
        assert kwargs["messages"] == [{"role": "user", "content": "Describe"}]
        assert kwargs["images"] == [_PNG]

    def test_image_is_replayed_through_previous_response_id(self, client):
        engine = _mock_engine(_output("a pixel"), _output("still a pixel"))
        first = self._post(client, engine, _user_input(_text("What?"), _image()))
        resp = self._post(
            client,
            engine,
            "And now?",
            previous_response_id=first.json()["id"],
        )

        assert resp.status_code == 200, resp.text
        kwargs = engine.chat.call_args_list[1].kwargs
        assert [m["role"] for m in kwargs["messages"]] == [
            "user",
            "assistant",
            "user",
        ]
        assert kwargs["images"] == [_PNG]

    def test_text_part_types_of_history_are_accepted(self, client):
        engine = _mock_engine(_output("a"))
        resp = self._post(
            client,
            engine,
            [
                {"type": "message", "role": "user", "content": [_text("Hi")]},
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Hello"}],
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "refusal", "refusal": "No."}],
                },
                {"type": "message", "role": "user", "content": [_text("Again")]},
            ],
        )

        assert resp.status_code == 200, resp.text
        contents = [m["content"] for m in engine.chat.call_args.kwargs["messages"]]
        assert contents == ["Hi", "Hello\n\nNo.", "Again"]

    @pytest.mark.parametrize("stream", [False, True])
    @pytest.mark.parametrize(
        "part, expected",
        [
            (
                {"type": "input_file", "file_data": "abc", "filename": "a.pdf"},
                "input_file",
            ),
            (
                {"type": "input_audio", "input_audio": {"data": "x", "format": "wav"}},
                "input_audio",
            ),
            ({"type": "made_up"}, "made_up"),
        ],
    )
    def test_unsupported_part_is_a_400_naming_it(self, client, part, expected, stream):
        engine = _mock_engine(_output("never"))
        resp = self._post(client, engine, _user_input(_text("hi"), part), stream=stream)

        assert resp.status_code == 400, resp.text
        assert expected in resp.json()["detail"]
        engine.chat.assert_not_called()
        assert engine._stream_calls == []

    @pytest.mark.parametrize(
        "part, expected",
        [
            ({"type": "input_image", "file_id": "file-1"}, "file_id"),
            ({"type": "input_image"}, "image_url"),
            ({"type": "input_image", "image_url": ""}, "image_url"),
            ({"type": "input_image", "image_url": {"url": _PNG}}, "image_url"),
            (_image(detail="ultra"), "detail"),
            ({"image_url": _PNG}, "no `type`"),
        ],
    )
    def test_malformed_part_is_a_400(self, client, part, expected):
        engine = _mock_engine(_output("never"))
        resp = self._post(client, engine, _user_input(part))

        assert resp.status_code == 400, resp.text
        assert expected in resp.json()["detail"]
        engine.chat.assert_not_called()

    @pytest.mark.parametrize("role", ["system", "developer"])
    def test_image_in_a_system_message_is_a_400(self, client, role):
        engine = _mock_engine(_output("never"))
        resp = self._post(
            client,
            engine,
            [
                {"type": "message", "role": role, "content": [_image()]},
                {"type": "message", "role": "user", "content": "hi"},
            ],
        )

        assert resp.status_code == 400, resp.text
        assert "system or developer" in resp.json()["detail"]
        engine.chat.assert_not_called()

    @pytest.mark.parametrize("stream", [False, True])
    @pytest.mark.parametrize(
        "items, seen",
        [
            ([], None),
            ([{"type": "item_reference", "id": "msg_1"}], "item_reference"),
            (
                [
                    {"type": "item_reference", "id": "msg_1"},
                    {"type": "computer_call_output", "call_id": "c"},
                ],
                "computer_call_output, item_reference",
            ),
        ],
    )
    def test_empty_conversation_is_a_400_not_a_500(self, client, items, seen, stream):
        engine = _mock_engine(_output("never"))
        resp = self._post(client, engine, items, stream=stream, instructions="Be nice")

        assert resp.status_code == 400, resp.text
        detail = resp.json()["detail"]
        assert "no conversation message" in detail
        if seen:
            assert seen in detail
        engine.chat.assert_not_called()
        assert engine._stream_calls == []

    def test_history_alone_is_not_an_empty_conversation(self, client):
        engine = _mock_engine(_output("one"), _output("two"))
        first = self._post(client, engine, "Hello")
        resp = self._post(client, engine, [], previous_response_id=first.json()["id"])

        assert resp.status_code == 200, resp.text
        roles = [m["role"] for m in engine.chat.call_args_list[1].kwargs["messages"]]
        assert roles == ["user", "assistant"]

    @pytest.mark.parametrize("stream", [False, True])
    def test_private_image_url_is_refused_like_on_the_chat_route(self, client, stream):
        engine = _mock_engine(_output("never"))
        resp = self._post(
            client,
            engine,
            _user_input(_image("http://127.0.0.1:9/secret.png")),
            stream=stream,
        )

        assert resp.status_code == 400, resp.text
        engine.chat.assert_not_called()
        assert engine._stream_calls == []
