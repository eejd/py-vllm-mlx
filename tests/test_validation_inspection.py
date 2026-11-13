# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for admission request decoding behavior."""
import asyncio

from types import SimpleNamespace

import pytest

from vllm_mlx.api.models import ChatCompletionRequest
from vllm_mlx.server import _prepare_chat_completion_invocation


@pytest.mark.parametrize("ignore_eos", [False, None])
def test_ignore_eos_false_or_unknown_does_not_enable_forcing(ignore_eos):
    request = ChatCompletionRequest.model_validate(
        {
            "model": "fixture",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 32,
            "ignore_eos": ignore_eos,
        }
    )
    engine = SimpleNamespace(is_mllm=False, preserve_native_tool_format=False)
    prepared = _prepare_chat_completion_invocation(engine, request, 32)
    assert prepared.chat_kwargs["max_tokens"] == 32
    assert prepared.chat_kwargs.get("ignore_eos") is not True
    assert not prepared.chat_kwargs.get("min_tokens")


def test_inspection_authentication_and_unknown_engine(monkeypatch):
    from fastapi.testclient import TestClient
    import vllm_mlx.server as server

    monkeypatch.setattr(server, "_api_key", "fixture-key")
    monkeypatch.setattr(server, "_engine", None)
    client = TestClient(server.app)
    assert client.post("/v1/inspection", json={}).status_code in (401, 403)
    response = client.post(
        "/v1/inspection", json={},
        headers={"Authorization": "Bearer fixture-key"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["kind"] == "vllm-mlx-inspection"
    assert body["instance"]["pid"] > 0
    assert body["generation"]["active_generation"] is None
    assert body["generation"]["unknown_reason"]


def test_inspection_distinguishes_tool_and_reasoning_parsers(monkeypatch):
    from fastapi.testclient import TestClient
    import vllm_mlx.server as server

    class ToolParser:
        def __init__(self, tokenizer=None):
            self.tokenizer = tokenizer

    class ReasoningParser:
        pass

    monkeypatch.setattr(server, "_api_key", None)
    monkeypatch.setattr(server, "_engine", None)
    monkeypatch.setattr(server, "_enable_auto_tool_choice", True)
    monkeypatch.setattr(server, "_tool_call_parser", "tool-format")
    monkeypatch.setattr(server, "_tool_parser_instance", ToolParser())
    monkeypatch.setattr(server, "_reasoning_parser", ReasoningParser())
    monkeypatch.setattr(server, "_reasoning_parser_name", "reasoning-format")
    body = TestClient(server.app).post("/v1/inspection", json={}).json()
    assert body["tool_parser"]["id"] == "tool-format"
    assert body["reasoning_parser"]["id"] == "reasoning-format"
    assert body["tool_parser"]["class_source"] == __file__
    assert body["reasoning_parser"]["class_source"] == __file__


@pytest.mark.parametrize("is_mllm", [False, True])
def test_inspection_reports_applied_reasoning_and_fallback(monkeypatch, is_mllm):
    from fastapi.testclient import TestClient
    from vllm_mlx.engine import SimpleEngine
    import vllm_mlx.server as server

    class Tokenizer:
        chat_template = "fixture template"
        bos_token = None
        reject_thinking = False

        def apply_chat_template(self, messages, **kwargs):
            if self.reject_thinking and "enable_thinking" in kwargs:
                raise TypeError("unsupported")
            return "fixture prompt"

        def encode(self, prompt, add_special_tokens=True):
            return [1, 2]

    tokenizer = Tokenizer()
    engine = SimpleEngine("fixture", force_mllm=is_mllm)
    engine._loaded = True
    engine._model = SimpleNamespace(tokenizer=tokenizer)
    if is_mllm:
        engine._text_model = object()
        engine._text_tokenizer = tokenizer
    monkeypatch.setattr(server, "_engine", engine)
    monkeypatch.setattr(server, "_model_name", "fixture")
    monkeypatch.setattr(server, "_api_key", None)
    monkeypatch.setattr(server, "_max_context_tokens", 4096)
    monkeypatch.setattr(server, "_default_chat_template_kwargs", None)
    monkeypatch.setenv("VLLM_MLX_ENABLE_THINKING", "true")
    client = TestClient(server.app)
    before = server._generation_accepted

    def template(payload):
        response = client.post("/v1/inspection", json=payload)
        assert response.status_code == 200
        return response.json()["template"]

    value = template({})
    assert value["kwargs"] == {}
    assert value["effective_kwargs"]["enable_thinking"] is True
    assert value["reasoning"] == {"enable_thinking": True, "source": "engine-default", "reason": None}
    value = template({"enable_thinking": False})
    assert value["reasoning"]["enable_thinking"] is False
    assert value["reasoning"]["source"] == "request"
    monkeypatch.setattr(server, "_default_chat_template_kwargs", {"enable_thinking": False})
    value = template({})
    assert value["reasoning"]["source"] == "server-default"
    assert value["reasoning"]["enable_thinking"] is False
    value = template({"chat_template_kwargs": {"enable_thinking": True}})
    assert value["reasoning"]["source"] == "request"
    assert value["reasoning"]["enable_thinking"] is True
    value = template({"chat_template_kwargs": {"enable_thinking": "not-a-boolean"}})
    assert value["reasoning"] == {"enable_thinking": None, "source": None,
                                   "reason": "effective enable_thinking is not a boolean"}
    value = template({"tools_present": True})
    assert value["effective_kwargs"] is None
    assert value["reasoning"]["enable_thinking"] is None
    tokenizer.reject_thinking = True
    value = template({})
    assert value["reasoning"] == {"enable_thinking": None, "source": None,
                                   "reason": "effective enable_thinking was not applied by the renderer"}
    assert server._generation_accepted == before


def test_bounded_context_refuses_unavailable_renderer(monkeypatch):
    from fastapi import HTTPException
    import vllm_mlx.server as server

    monkeypatch.setattr(server, "_max_context_tokens", 4096)
    with pytest.raises(HTTPException, match="context rendering unavailable"):
        asyncio.run(server._enforce_context(SimpleNamespace(), [], {"max_tokens": 1024}))


def test_unavailable_mllm_text_route_refuses_without_guessing(monkeypatch):
    from fastapi import HTTPException
    from fastapi.testclient import TestClient
    from vllm_mlx.engine import SimpleEngine
    import vllm_mlx.server as server

    engine = SimpleEngine("fixture", force_mllm=True)
    engine._loaded = True
    engine._model = SimpleNamespace()
    engine._text_model = None
    engine._text_tokenizer = None
    monkeypatch.setattr(server, "_engine", engine)
    monkeypatch.setattr(server, "_api_key", None)
    monkeypatch.setattr(server, "_max_context_tokens", 4096)
    body = TestClient(server.app).post("/v1/inspection", json={}).json()
    assert body["model"]["render_path"] is None
    assert body["template"]["sha256"] is None
    with pytest.raises(HTTPException):
        asyncio.run(server._enforce_context(engine, [{"role": "user", "content": "x"}], {"max_tokens": 10}))


def test_inspection_reports_effective_mllm_text_cache_after_probe(monkeypatch):
    from fastapi.testclient import TestClient
    from vllm_mlx.engine import SimpleEngine
    import vllm_mlx.server as server

    engine = SimpleEngine("fixture", force_mllm=True, enable_cache=True)
    engine._loaded = True
    engine._model = SimpleNamespace()
    engine._text_model = object()
    engine._text_tokenizer = SimpleNamespace(chat_template="fixture")
    engine._prefix_trie_cache_enabled = False
    engine._supports_system_kv_cache = False
    monkeypatch.setattr(server, "_engine", engine)
    monkeypatch.setattr(server, "_api_key", None)
    body = TestClient(server.app).post("/v1/inspection", json={}).json()
    cache = body["cache"]
    assert body["model"]["render_path"] == "mllm-text"
    assert cache["requested"]["media_cache_enabled"] is True
    assert cache["requested"]["system_kv_capacity"] > 0
    assert cache["enabled"] is False
    assert cache["prefix_trie_enabled"] is False
    assert cache["prefix_trie_capacity"] == 0
    assert cache["system_kv_capacity"] == 0
    assert cache["system_kv_supported"] is False
    assert cache["effective"]["enabled"] is False
    assert cache["persistence"] is False

    # The inspection follows the actual probe result, not the constructor flag.
    engine._supports_system_kv_cache = True
    cache = TestClient(server.app).post("/v1/inspection", json={}).json()["cache"]
    assert cache["enabled"] is True
    assert cache["system_kv_capacity"] == engine._system_kv_capacity


def test_unknown_render_path_never_claims_caches_off():
    from vllm_mlx.engine import SimpleEngine
    from vllm_mlx.server import _inspection_cache_policy

    cache = _inspection_cache_policy(SimpleEngine("fixture", force_mllm=True), text_route=False)
    assert cache["enabled"] is None
    assert cache["system_kv_capacity"] is None
    assert cache["unknown_reason"]


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        ("/v1/chat/completions", {"model": "fixture", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 3}),
        ("/v1/responses", {"model": "fixture", "input": "hello", "max_output_tokens": 3}),
        ("/v1/messages", {"model": "fixture", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 3}),
    ],
)
@pytest.mark.parametrize("is_mllm", [False, True])
def test_context_guard_blocks_generation_on_all_routes(monkeypatch, path, payload, is_mllm):
    from fastapi.testclient import TestClient
    from vllm_mlx.engine import SimpleEngine
    from vllm_mlx.engine import GenerationOutput
    import vllm_mlx.server as server

    class Tokenizer:
        chat_template = "{{ messages }}"
        bos_token = None

        def apply_chat_template(self, messages, **kwargs):
            return "rendered prompt"

        def encode(self, prompt, add_special_tokens=True):
            return list(range(10))

    class FixtureEngine(SimpleEngine):
        def __init__(self):
            super().__init__("fixture")
            self._is_mllm = is_mllm
            self._loaded = True
            self._model = SimpleNamespace(tokenizer=Tokenizer())
            if is_mllm:
                self._text_model = object()
                self._text_tokenizer = Tokenizer()
            self.calls = 0

        async def chat(self, **kwargs):
            async with self._acquire_generation_slot("fixture"):
                self.calls += 1
                return GenerationOutput(text="ok", prompt_tokens=10, completion_tokens=1)

    engine = FixtureEngine()
    monkeypatch.setattr(server, "_engine", engine)
    monkeypatch.setattr(server, "_model_name", "fixture")
    monkeypatch.setattr(server, "_api_key", None)
    monkeypatch.setattr(server, "_max_context_tokens", 12)
    monkeypatch.setattr(server, "_generation_accounting_unknown", False)
    client = TestClient(server.app)
    inspection = client.post("/v1/inspection", json={
        "messages": [{"role": "user", "content": "hello"}], "max_tokens": 3,
    })
    assert inspection.status_code == 200
    body = inspection.json()
    assert body["template"]["source"] == ("text_tokenizer" if is_mllm else "tokenizer")
    assert body["template"]["sha256"]
    assert body["rendering"]["rendered_prompt_tokens"] == 10
    assert body["rendering"]["total_tokens"] == 13
    response = client.post(path, json=payload)
    assert response.status_code == 400
    assert engine.calls == 0
    response = client.post(path, json={**payload, "stream": True})
    assert response.status_code == 400
    assert engine.calls == 0
    route = {"/v1/chat/completions": "chat", "/v1/responses": "responses", "/v1/messages": "messages"}[path]
    before = body["generation"]["routes"][route]
    monkeypatch.setattr(server, "_max_context_tokens", 13)
    response = client.post(path, json=payload)
    assert response.status_code == 200
    assert engine.calls == 1
    after = client.post("/v1/inspection", json={}).json()
    assert after["generation"]["routes"][route] == before + 1


@pytest.mark.asyncio
async def test_stream_accounting_stays_active_until_generator_closes(monkeypatch):
    import asyncio
    from vllm_mlx.engine import SimpleEngine
    import vllm_mlx.server as server

    closing = asyncio.Event()
    release = asyncio.Event()

    class Engine(SimpleEngine):
        def __init__(self):
            super().__init__("fixture")

        async def stream_chat(self, **kwargs):
            async with self._acquire_generation_slot("fixture"):
                try:
                    yield "token"
                finally:
                    closing.set()
                    await release.wait()

    monkeypatch.setattr(server, "_max_context_tokens", None)
    before = server._generation_active
    monkeypatch.setattr(server, "_generation_accounting_unknown", False)
    accepted = server._generation_accepted
    cancelled = server._generation_outcomes["chat"]["cancelled"]
    stream = server._observed_stream_chat(Engine(), "chat", [], {})
    assert await anext(stream) == "token"
    assert server._generation_accepted == accepted + 1
    task = asyncio.create_task(stream.aclose())
    await closing.wait()
    assert server._generation_active == before + 1
    release.set()
    await task
    assert server._generation_active == before
    assert server._generation_outcomes["chat"]["cancelled"] == cancelled + 1


@pytest.mark.asyncio
async def test_engine_admission_counts_busy_failure_and_cancellation(monkeypatch):
    import asyncio
    from vllm_mlx.engine import SimpleEngine
    from vllm_mlx.engine.base import EngineBusy
    import vllm_mlx.server as server

    class Engine(SimpleEngine):
        def __init__(self):
            super().__init__("fixture")
            self.mode = "failure"
            self.entered = asyncio.Event()

        async def chat(self, **kwargs):
            async with self._acquire_generation_slot("fixture"):
                self.entered.set()
                if self.mode == "failure":
                    raise RuntimeError("fixture failure")
                await asyncio.Event().wait()

    monkeypatch.setattr(server, "_max_context_tokens", None)
    monkeypatch.setattr(server, "_generation_accounting_unknown", False)
    engine = Engine()
    accepted = server._generation_accepted
    outcomes = server._generation_outcomes["chat"].copy()
    with pytest.raises(RuntimeError, match="fixture failure"):
        await server._observed_chat(engine, "chat", [], {})
    assert server._generation_accepted == accepted + 1
    assert server._generation_outcomes["chat"]["failed"] == outcomes["failed"] + 1

    engine.mode = "wait"
    engine.entered.clear()
    task = asyncio.create_task(server._observed_chat(engine, "chat", [], {}))
    await engine.entered.wait()
    with pytest.raises(EngineBusy):
        await server._observed_chat(engine, "chat", [], {})
    assert server._generation_outcomes["chat"]["rejected"] == outcomes["rejected"] + 1
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert server._generation_accepted == accepted + 2
    assert server._generation_outcomes["chat"]["cancelled"] == outcomes["cancelled"] + 1
    assert not server._generation_accounting_unknown
    engine._generation_lock_admission = "wait"
    rejected = server._generation_outcomes["chat"]["rejected"]
    cancelled = server._generation_outcomes["chat"]["cancelled"]
    async with engine._generation_lock:
        waiting = asyncio.create_task(server._observed_chat(engine, "chat", [], {}))
        while engine._generation_waiters == 0:
            await asyncio.sleep(0)
        monkeypatch.setattr(server, "_engine", engine)
        inspection = await server.inspect_serving(server.InspectionRequest())
        assert inspection["generation"]["active_generation"] == 1
        assert inspection["generation"]["queued_generation"] == 1
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
    assert server._generation_outcomes["chat"]["cancelled"] == cancelled + 1
    assert server._generation_outcomes["chat"]["rejected"] == rejected


@pytest.mark.asyncio
async def test_untracked_internal_generation_makes_counters_unknown(monkeypatch):
    from vllm_mlx.engine import SimpleEngine, GenerationOutput
    import vllm_mlx.server as server

    class Engine(SimpleEngine):
        def __init__(self):
            super().__init__("fixture")

        async def _stream_chat_impl(self, **kwargs):
            yield GenerationOutput(text="warmup")

    engine = Engine()
    monkeypatch.setattr(server, "_engine", engine)
    monkeypatch.setattr(server, "_generation_accounting_unknown", False)
    stream = engine.stream_chat(messages=[])
    await anext(stream)
    await stream.aclose()
    result = await server.inspect_serving(server.InspectionRequest())
    assert result["generation"]["accepted_generation_requests"] is None
    assert result["generation"]["unknown_reason"] == "internal generation not accounted"


def test_real_server_subprocess_inspection_and_counter_delta(tmp_path):
    import json
    import os
    from pathlib import Path
    import socket
    import subprocess
    import sys
    import textwrap
    import time
    import urllib.error
    import urllib.request

    script = tmp_path / "fixture_server.py"
    script.write_text(textwrap.dedent("""
        import sys
        from types import SimpleNamespace
        import uvicorn
        import vllm_mlx.server as server
        from vllm_mlx.engine import SimpleEngine, GenerationOutput

        class Tokenizer:
            chat_template = "{{ messages }}"
            bos_token = None
            def apply_chat_template(self, messages, **kwargs):
                return "fixture prompt"
            def encode(self, prompt, add_special_tokens=True):
                return [1, 2, 3]

        class FixtureEngine(SimpleEngine):
            def __init__(self):
                super().__init__("fixture")
                self._is_mllm = False
                self._loaded = True
                self._model = SimpleNamespace(tokenizer=Tokenizer())
            async def chat(self, **kwargs):
                async with self._acquire_generation_slot("fixture"):
                    return GenerationOutput(text="ok", prompt_tokens=3, completion_tokens=1)

        server._engine = FixtureEngine()
        server._model_name = "fixture"
        server._api_key = "fixture-key"
        server._max_context_tokens = 10
        uvicorn.run(server.app, host="127.0.0.1", port=int(sys.argv[1]),
                    lifespan="off", log_level="error")
    """))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    proc = subprocess.Popen(
        [sys.executable, str(script), str(port)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])},
    )

    def post(path, data, key=True):
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = "Bearer fixture-key"
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}{path}",
            data=json.dumps(data).encode(), headers=headers, method="POST",
        )
        with urllib.request.urlopen(request, timeout=2) as response:
            return json.load(response)

    try:
        deadline = time.monotonic() + 15
        while True:
            try:
                before = post("/v1/inspection", {})
                break
            except urllib.error.URLError:
                if proc.poll() is not None or time.monotonic() >= deadline:
                    raise
                time.sleep(0.05)
        with pytest.raises(urllib.error.HTTPError) as unauthorized:
            post("/v1/inspection", {}, key=False)
        assert unauthorized.value.code in (401, 403)
        result = post("/v1/chat/completions", {
            "model": "fixture", "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 3,
        })
        assert result["choices"][0]["message"]["content"] == "ok"
        after = post("/v1/inspection", {})
        assert after["instance"]["pid"] == proc.pid
        assert after["generation"]["accepted_generation_requests"] == before["generation"]["accepted_generation_requests"] + 1
        assert after["generation"]["routes"]["chat"] == before["generation"]["routes"]["chat"] + 1
        assert after["generation"]["active_generation"] == 0
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_mllm_text_route_initializes_before_rendering(monkeypatch):
    # Draft-mode MLLM skips eager text init in start(); the guard must init
    # lazily before rendering instead of refusing a loadable route forever.
    from vllm_mlx.engine import SimpleEngine
    import vllm_mlx.server as server

    engine = SimpleEngine("fixture", force_mllm=True)
    engine._loaded = True
    engine._model = SimpleNamespace()
    engine._text_model = None
    engine._text_tokenizer = None
    calls = []

    async def ensure(*, mllm_draft_requested):
        calls.append(mllm_draft_requested)
        engine._text_model = SimpleNamespace()
        engine._text_tokenizer = SimpleNamespace(
            bos_token=None, encode=lambda prompt, add_special_tokens=True: [0] * 5)

    monkeypatch.setattr(engine, "_ensure_text_model_for_request", ensure)
    monkeypatch.setattr(engine, "_should_route_text_through_text_model",
                        lambda *, mllm_draft_requested: True)
    monkeypatch.setattr(engine, "_render_text_model_prompt",
                        lambda messages, tools, kwargs, enable_thinking: ("prompt", None, None))
    monkeypatch.setattr(server, "_max_context_tokens", 4096)
    asyncio.run(server._enforce_context(engine, [{"role": "user", "content": "x"}], {"max_tokens": 10}))
    assert calls == [False]
