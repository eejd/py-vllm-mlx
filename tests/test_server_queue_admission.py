# SPDX-License-Identifier: Apache-2.0
"""HTTP mapping for serialized-engine admission failures: 503 + Retry-After."""

import pytest
from fastapi.testclient import TestClient

import vllm_mlx.server as server
from vllm_mlx.engine.base import EngineQueueFull, EngineQueueTimeout


class BusyEngine:
    """Engine whose every generation route fails admission immediately."""

    is_mllm = False
    preserve_native_tool_format = False
    model_name = "served-model"
    tokenizer = None

    def __init__(self, exc):
        self._exc = exc

    async def chat(self, *args, **kwargs):
        raise self._exc

    async def generate(self, *args, **kwargs):
        raise self._exc

    async def stream_chat(self, *args, **kwargs):
        raise self._exc
        yield  # pragma: no cover

    async def stream_generate(self, *args, **kwargs):
        raise self._exc
        yield  # pragma: no cover

    def get_stats(self):
        return {}


@pytest.fixture
def busy_client(monkeypatch):
    def install(exc):
        engine = BusyEngine(exc)

        async def fake_acquire(*args, **kwargs):
            return engine

        async def fake_release(*args, **kwargs):
            return None

        monkeypatch.setattr(server, "_engine", engine)
        monkeypatch.setattr(server, "_model_name", "served-model")
        monkeypatch.setattr(server, "_validate_model_name", lambda _m: None)
        monkeypatch.setattr(server, "_acquire_default_engine_for_request", fake_acquire)
        monkeypatch.setattr(server, "_release_default_engine", fake_release)
        monkeypatch.setattr(server, "_api_key", None)
        monkeypatch.setattr(server, "_max_context_tokens", None)
        return TestClient(server.app, raise_server_exceptions=False)

    return install


_CHAT = {"model": "served-model", "messages": [{"role": "user", "content": "hi"}]}
_RESPONSES = {"model": "served-model", "input": "hi"}
_MESSAGES = {
    "model": "served-model",
    "max_tokens": 8,
    "messages": [{"role": "user", "content": "hi"}],
}


@pytest.mark.parametrize(
    "exc,code",
    [
        (EngineQueueFull("queue full", retry_after_s=1), "text_generation_queue_full"),
        (EngineQueueTimeout("timed out", retry_after_s=7), "text_generation_queue_timeout"),
    ],
)
def test_non_stream_chat_busy_is_503_with_retry_after(busy_client, exc, code):
    client = busy_client(exc)
    response = client.post("/v1/chat/completions", json=_CHAT)
    assert response.status_code == 503
    assert response.headers["retry-after"] == str(exc.retry_after_s)
    assert response.json()["detail"]["error"] == code


@pytest.mark.parametrize(
    "path,body",
    [("/v1/responses", _RESPONSES), ("/v1/messages", _MESSAGES)],
)
def test_non_stream_responses_and_messages_busy_is_503_not_500(busy_client, path, body):
    client = busy_client(EngineQueueFull("queue full", retry_after_s=2))
    response = client.post(path, json=body)
    assert response.status_code == 503
    assert response.headers["retry-after"] == "2"
    assert response.json()["detail"]["error"] == "text_generation_queue_full"


@pytest.mark.parametrize(
    "path,body",
    [
        ("/v1/chat/completions", {**_CHAT, "stream": True}),
        ("/v1/completions", {"model": "served-model", "prompt": "hi", "stream": True}),
        ("/v1/responses", {**_RESPONSES, "stream": True}),
        ("/v1/messages", {**_MESSAGES, "stream": True}),
    ],
)
def test_stream_busy_before_first_byte_is_503(busy_client, path, body):
    client = busy_client(EngineQueueTimeout("timed out", retry_after_s=3))
    response = client.post(path, json=body)
    assert response.status_code == 503
    assert response.headers["retry-after"] == "3"
    assert "text/event-stream" not in response.headers.get("content-type", "")
