# SPDX-License-Identifier: Apache-2.0
"""Tests for Prometheus server metrics.

Runs against a real FastAPI TestClient built from vllm_mlx.server, which
pulls in the full server dependency chain (uvicorn, fastapi, prometheus-client,
mlx.core). Kept Apple-Silicon-only rather than mlx-stubbed: unlike
test_mllm_steps_executed_stat.py's mlx.core-only need, this file's import
chain also needs uvicorn/prometheus-client, neither of which the Linux
test-matrix job installs (see PR #749's review -- an earlier version of
this file ran here via tests/_mlx_stub.py and errored at fixture setup
with ModuleNotFoundError: No module named 'uvicorn'). The one assertion
that specifically needed Linux coverage (get_stats()["steps_executed"]
reaching the vllm_mlx_engine_steps_executed gauge) now has a dependency-light
equivalent in test_mllm_steps_executed_stat.py's
TestMetricsEngineStepsExecutedGauge, which calls MetricsCollector directly
and needs neither uvicorn nor a real prometheus_client registry. This file
still runs in the Apple job for full HTTP-layer integration coverage.
"""

import asyncio
import platform
import sys
from types import SimpleNamespace

import pytest

# Skip all tests if not on Apple Silicon
pytestmark = pytest.mark.skipif(
    sys.platform != "darwin" or platform.machine() != "arm64",
    reason="Requires Apple Silicon",
)


class FakeEngine:
    """Small fake engine for metrics endpoint tests."""

    model_name = "metrics-model"
    is_mllm = False
    preserve_native_tool_format = False
    tokenizer = None

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def generate(self, **kwargs):
        from vllm_mlx.engine.base import GenerationOutput

        return GenerationOutput(
            text="Hello",
            tokens=[1, 2],
            prompt_tokens=4,
            completion_tokens=2,
            finish_reason="stop",
        )

    async def stream_generate(self, **kwargs):
        from vllm_mlx.engine.base import GenerationOutput

        yield GenerationOutput(
            text="Hel",
            new_text="Hel",
            prompt_tokens=4,
            completion_tokens=1,
            finished=False,
        )
        yield GenerationOutput(
            text="Hello",
            new_text="lo",
            prompt_tokens=4,
            completion_tokens=2,
            finish_reason="stop",
            finished=True,
        )

    async def chat(self, **kwargs):
        from vllm_mlx.engine.base import GenerationOutput

        return GenerationOutput(
            text="Hello from chat",
            tokens=[1, 2, 3],
            prompt_tokens=5,
            completion_tokens=3,
            finish_reason="stop",
        )

    async def stream_chat(self, **kwargs):
        from vllm_mlx.engine.base import GenerationOutput

        yield GenerationOutput(
            text="Hel",
            new_text="Hel",
            prompt_tokens=5,
            completion_tokens=1,
            finished=False,
        )
        yield GenerationOutput(
            text="Hello",
            new_text="lo",
            prompt_tokens=5,
            completion_tokens=2,
            finish_reason="stop",
            finished=True,
        )

    def get_stats(self):
        return {
            "engine_type": "simple",
            "is_mllm": False,
            "num_waiting": 2,
            "num_running": 1,
            "steps_executed": 7,
            "uptime_seconds": 42.5,
            "metal_active_memory_gb": 1.25,
            "metal_peak_memory_gb": 2.5,
            "metal_cache_memory_gb": 0.5,
            "memory_aware_cache": {
                "entry_count": 3,
                "hits": 4,
                "misses": 1,
                "evictions": 0,
                "hit_rate": 0.8,
                "memory_utilization": 0.25,
                "tokens_saved": 128,
                "current_memory_mb": 64,
                "max_memory_mb": 256,
            },
        }


@pytest.fixture()
def metrics_client(monkeypatch):
    """Create a TestClient with a fresh metrics collector."""
    from fastapi.testclient import TestClient

    import vllm_mlx.server as server
    from vllm_mlx.metrics import MetricsCollector

    collector = MetricsCollector()
    monkeypatch.setattr(server, "_metrics", collector)
    monkeypatch.setattr(server, "_engine", None)
    monkeypatch.setattr(server, "_model_name", "metrics-model")
    monkeypatch.setattr(server, "_api_key", None)
    monkeypatch.setattr(server, "_mcp_manager", None)
    monkeypatch.setattr(server, "_reasoning_parser", None)
    monkeypatch.setattr(server, "_tool_parser_instance", None)
    monkeypatch.setattr(server, "_tool_call_parser", None)
    monkeypatch.setattr(server, "_enable_auto_tool_choice", False)
    monkeypatch.setattr(server, "_default_timeout", 30.0)
    monkeypatch.setattr(server, "_default_max_tokens", 128)
    monkeypatch.setattr(
        server,
        "_rate_limiter",
        server.RateLimiter(requests_per_minute=60, enabled=False),
    )

    with TestClient(server.app) as client:
        yield client, server, collector


class TestMetricsEndpoint:
    """Tests for Prometheus /metrics exposure and accounting."""

    def test_metrics_endpoint_disabled_returns_404(self, metrics_client):
        client, _server, collector = metrics_client

        collector.configure(enabled=False)
        response = client.get("/metrics")

        assert response.status_code == 404

    def test_metrics_endpoint_reports_dead_gauges_without_engine_or_manager(
        self, metrics_client
    ):
        client, server, collector = metrics_client

        collector.configure(enabled=True)
        assert server._engine is None
        assert server._model_manager is None

        response = client.get("/metrics")

        assert response.status_code == 200
        assert "vllm_mlx_model_loaded 0.0" in response.text
        assert "vllm_mlx_scheduler_waiting_requests 0.0" in response.text

    def test_metrics_endpoint_scrapes_registry_mode_engine(
        self, metrics_client, monkeypatch
    ):
        client, server, collector = metrics_client

        collector.configure(enabled=True)

        class FakeModelManager:
            def __init__(self, engine):
                self._engine = engine

            def get_metrics_engine(self):
                return self._engine

            async def shutdown(self):
                return None

        monkeypatch.setattr(server, "_model_manager", FakeModelManager(FakeEngine()))

        response = client.get("/metrics")

        assert response.status_code == 200
        assert "vllm_mlx_model_loaded 1.0" in response.text
        assert "vllm_mlx_scheduler_waiting_requests 2.0" in response.text
        assert (
            'vllm_mlx_cache_type{cache_type="memory_aware_cache"} 1.0' in response.text
        )

    def test_metrics_endpoint_registry_mode_dead_gauges_when_idle(
        self, metrics_client, monkeypatch
    ):
        client, server, collector = metrics_client

        collector.configure(enabled=True)

        class FakeModelManager:
            def get_metrics_engine(self):
                return None

            async def shutdown(self):
                return None

        monkeypatch.setattr(server, "_model_manager", FakeModelManager())

        response = client.get("/metrics")

        assert response.status_code == 200
        assert "vllm_mlx_model_loaded 0.0" in response.text

    def test_metrics_endpoint_scrapes_runtime_stats(self, metrics_client, monkeypatch):
        client, server, collector = metrics_client

        collector.configure(enabled=True)
        monkeypatch.setattr(server, "_engine", FakeEngine())

        response = client.get("/metrics")

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/plain; version=")
        assert "charset=utf-8" in response.headers["content-type"]
        assert "vllm_mlx_model_loaded 1.0" in response.text
        assert 'vllm_mlx_engine_type{engine_type="simple"} 1.0' in response.text
        assert "vllm_mlx_scheduler_waiting_requests 2.0" in response.text
        assert (
            'vllm_mlx_cache_type{cache_type="memory_aware_cache"} 1.0' in response.text
        )
        # get_stats()["steps_executed"] must reach the gauge -- the
        # duck-typed read side of #746 (the producer side, MLLMScheduler /
        # BatchedEngine, is covered by test_mllm_steps_executed_stat.py).
        assert "vllm_mlx_engine_steps_executed 7.0" in response.text

    def test_metrics_collapse_unmatched_paths(self, metrics_client, monkeypatch):
        client, server, collector = metrics_client

        collector.configure(enabled=True)
        monkeypatch.setattr(server, "_engine", FakeEngine())

        miss = client.get("/definitely-not-a-real-route")
        scrape = client.get("/metrics")

        assert miss.status_code == 404
        assert (
            'vllm_mlx_http_requests_total{method="GET",path="__unmatched__",status_code="404"} 1.0'
            in scrape.text
        )

    def test_completion_request_updates_metrics(self, metrics_client, monkeypatch):
        client, server, collector = metrics_client

        collector.configure(enabled=True)
        monkeypatch.setattr(server, "_engine", FakeEngine())

        response = client.post(
            "/v1/completions",
            json={
                "model": "metrics-model",
                "prompt": "Hello",
                "max_tokens": 8,
            },
        )
        scrape = client.get("/metrics")

        assert response.status_code == 200
        assert (
            'vllm_mlx_inference_requests_total{endpoint="completions",result="success",stream="false"} 1.0'
            in scrape.text
        )
        assert (
            'vllm_mlx_prompt_tokens_total{endpoint="completions",stream="false"} 4.0'
            in scrape.text
        )
        assert (
            'vllm_mlx_completion_tokens_total{endpoint="completions",stream="false"} 2.0'
            in scrape.text
        )
        assert (
            'vllm_mlx_http_requests_total{method="POST",path="/v1/completions",status_code="200"} 1.0'
            in scrape.text
        )

    def test_streaming_chat_records_ttft_and_stream_metrics(
        self, metrics_client, monkeypatch
    ):
        client, server, collector = metrics_client

        collector.configure(enabled=True)
        monkeypatch.setattr(server, "_engine", FakeEngine())

        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "metrics-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
                "max_tokens": 8,
            },
        ) as response:
            body = "".join(response.iter_text())

        scrape = client.get("/metrics")

        assert response.status_code == 200
        assert "data: [DONE]" in body
        assert (
            'vllm_mlx_inference_requests_total{endpoint="chat_completions",result="success",stream="true"} 1.0'
            in scrape.text
        )
        assert (
            'vllm_mlx_inference_ttft_seconds_count{endpoint="chat_completions",stream="true"} 1.0'
            in scrape.text
        )
        assert (
            'vllm_mlx_prompt_tokens_total{endpoint="chat_completions",stream="true"} 5.0'
            in scrape.text
        )
        assert (
            'vllm_mlx_completion_tokens_total{endpoint="chat_completions",stream="true"} 2.0'
            in scrape.text
        )


class TestMetricsMiddlewareStreamingTiming:
    """Regression coverage for `_metrics_middleware`'s in-flight gauge timing.

    `call_next()` (Starlette's BaseHTTPMiddleware) resolves as soon as the
    ASGI response has *started* -- for a StreamingResponse this is long
    before the body finishes sending. These tests exercise
    `_metrics_middleware` directly, with a fake `call_next` returning a
    controllable fake streaming response, instead of driving a real
    FastAPI/TestClient/uvicorn round trip -- this makes the exact moment the
    gauge changes deterministic and independent of any real inference or
    real async I/O, per `TestDisconnectGuard`'s established pattern for this
    file.
    """

    PATH = "/v1/chat/completions"

    def _make_collector_and_request(self, monkeypatch):
        import vllm_mlx.server as server
        from vllm_mlx.metrics import MetricsCollector

        collector = MetricsCollector()
        collector.configure(enabled=True)
        monkeypatch.setattr(server, "_metrics", collector)
        monkeypatch.setattr(
            server, "_metrics_path_for_request", lambda request: self.PATH
        )
        request = SimpleNamespace(method="POST")
        return server, collector, request

    @staticmethod
    def _inflight_value(collector, path):
        payload, _ = collector.render_metrics(engine=None, mcp_manager=None)
        needle = f'vllm_mlx_http_requests_in_flight{{method="POST",path="{path}"}}'
        for line in payload.decode().splitlines():
            if line.startswith(needle):
                return float(line.split()[-1])
        return None

    @staticmethod
    def _total_count(collector, path, status_code):
        payload, _ = collector.render_metrics(engine=None, mcp_manager=None)
        needle = (
            f'vllm_mlx_http_requests_total{{method="POST",path="{path}",'
            f'status_code="{status_code}"}}'
        )
        for line in payload.decode().splitlines():
            if line.startswith(needle):
                return float(line.split()[-1])
        return 0.0

    @pytest.mark.anyio
    async def test_streaming_body_holds_gauge_until_exhausted(self, monkeypatch):
        server, collector, request = self._make_collector_and_request(monkeypatch)
        resume = asyncio.Event()

        async def slow_body():
            yield b"data: role-preamble\n\n"
            await resume.wait()
            yield b"data: [DONE]\n\n"

        async def call_next(_request):
            return SimpleNamespace(body_iterator=slow_body(), status_code=200)

        # Prometheus only emits a line for a labeled gauge once it has been
        # touched at least once -- untouched, it's absent, not 0.0.
        assert self._inflight_value(collector, self.PATH) is None

        response = await server._metrics_middleware(request, call_next)
        # call_next() has already resolved here -- this is exactly the
        # moment the OLD code decremented the gauge, before any real
        # generation has happened.
        assert self._inflight_value(collector, self.PATH) == 1.0

        body_iter = response.body_iterator.__aiter__()
        first_chunk = await body_iter.__anext__()
        assert first_chunk == b"data: role-preamble\n\n"
        assert self._inflight_value(collector, self.PATH) == 1.0, (
            "must still be in flight after only the content-free preamble "
            "chunk has been sent -- this is the exact case the old gauge "
            "got wrong"
        )

        resume.set()
        remaining = [chunk async for chunk in body_iter]

        assert remaining == [b"data: [DONE]\n\n"]
        assert (
            self._inflight_value(collector, self.PATH) == 0.0
        ), "must be decremented once the body is actually exhausted"

    @pytest.mark.anyio
    async def test_error_mid_stream_decrements_exactly_once(self, monkeypatch):
        server, collector, request = self._make_collector_and_request(monkeypatch)

        async def erroring_body():
            yield b"data: role-preamble\n\n"
            raise RuntimeError("boom")

        async def call_next(_request):
            return SimpleNamespace(body_iterator=erroring_body(), status_code=200)

        response = await server._metrics_middleware(request, call_next)
        assert self._inflight_value(collector, self.PATH) == 1.0

        received = []
        with pytest.raises(RuntimeError, match="boom"):
            async for chunk in response.body_iterator:
                received.append(chunk)

        assert received == [b"data: role-preamble\n\n"]
        assert (
            self._inflight_value(collector, self.PATH) == 0.0
        ), "a mid-stream error must not leak the gauge stuck at 1"

        # The generator is already closed by the propagated exception,
        # closing it again (e.g. GC, or a caller's own cleanup) must not
        # double-decrement past zero.
        await response.body_iterator.aclose()
        assert self._inflight_value(collector, self.PATH) == 0.0

    @pytest.mark.anyio
    async def test_call_next_raising_before_any_response_finishes_once(
        self, monkeypatch
    ):
        server, collector, request = self._make_collector_and_request(monkeypatch)

        async def call_next(_request):
            raise RuntimeError("engine acquisition failed")

        with pytest.raises(RuntimeError, match="engine acquisition failed"):
            await server._metrics_middleware(request, call_next)

        assert self._inflight_value(collector, self.PATH) == 0.0
        assert self._total_count(collector, self.PATH, 500) == 1.0

    @pytest.mark.anyio
    async def test_non_streaming_single_chunk_body_still_settles_immediately(
        self, monkeypatch
    ):
        """Non-streaming responses are unaffected: their entire body is
        already produced before `call_next()` returns, so the returned
        body_iterator yields once and is done -- net timing is unchanged
        from before this fix."""
        server, collector, request = self._make_collector_and_request(monkeypatch)

        async def single_chunk_body():
            yield b'{"id": "cmpl-1", "object": "chat.completion"}'

        async def call_next(_request):
            return SimpleNamespace(body_iterator=single_chunk_body(), status_code=200)

        response = await server._metrics_middleware(request, call_next)
        assert self._inflight_value(collector, self.PATH) == 1.0

        received = [chunk async for chunk in response.body_iterator]

        assert received == [b'{"id": "cmpl-1", "object": "chat.completion"}']
        assert self._inflight_value(collector, self.PATH) == 0.0

    @pytest.mark.anyio
    async def test_send_failure_mid_stream_settles_at_request_exit(self, monkeypatch):
        """Reproduces the PR #782 review finding (Thump604): on the real
        Starlette `_StreamingResponse.__call__` path, `send()` raising
        mid-stream (e.g. a genuine client disconnect) abandons
        `body_iterator` without ever closing it -- `async for`/`await` do
        not call `aclose()` on early termination via a propagated
        exception. `body_iterator`-only wrapping settles metrics only
        whenever that abandoned generator happens to be garbage-collected,
        not at request exit. This must settle synchronously instead, the
        moment the response's ASGI call raises."""
        server, collector, request = self._make_collector_and_request(monkeypatch)

        class FakeASGIStreamingResponse:
            """Minimal stand-in for Starlette's `_StreamingResponse`:
            `__call__` mirrors its body-iteration loop exactly (no
            try/finally of its own around it), so a `send()` failure
            propagates straight out, leaving `body_iterator` suspended
            mid-yield -- exactly like the real object this stands in for."""

            def __init__(self, body_iterator, status_code=200):
                self.body_iterator = body_iterator
                self.status_code = status_code

            async def __call__(self, scope, receive, send):
                async for chunk in self.body_iterator:
                    await send({"type": "http.response.body", "body": chunk})

        async def slow_body():
            yield b"first chunk"
            yield b"second chunk"  # never reached -- send() dies on the first

        async def call_next(_request):
            return FakeASGIStreamingResponse(slow_body(), status_code=200)

        response = await server._metrics_middleware(request, call_next)
        assert self._inflight_value(collector, self.PATH) == 1.0

        async def dying_send(_message):
            raise OSError("Broken pipe")

        with pytest.raises(OSError, match="Broken pipe"):
            await response(None, None, dying_send)

        assert self._inflight_value(collector, self.PATH) == 0.0, (
            "a send() failure mid-stream (a real client disconnect) must "
            "settle metrics at request exit, not only whenever the "
            "abandoned generator happens to be garbage-collected"
        )


def _scrape(collector, **kwargs):
    payload, _ = collector.render_metrics(engine=kwargs.pop("engine", None), mcp_manager=None, **kwargs)
    return payload.decode()


def _state(**over):
    state = {
        "counters": {
            "hits": 3, "misses": 1, "evictions": 2, "tokens_saved": 120,
            "discarded_hits": 1, "entry_count": 4, "current_memory_mb": 2.0,
            "max_memory_mb": 64.0,
        },
        "memory_limit": {"bytes": 64 << 20, "source": "explicit"},
    }
    state.update(over)
    return state


class TestPrefixCacheSeries:
    """Per-model prefix-cache series built from the cache_state blocks (#42)."""

    @pytest.fixture()
    def collector(self):
        from vllm_mlx.metrics import MetricsCollector

        c = MetricsCollector()
        c.configure(enabled=True)
        return c

    def test_counters_limit_and_persistence_are_exported_per_model(self, collector):
        text = _scrape(
            collector,
            cache_states={"m1": _state()},
            persistence={
                "applies": True,
                "dirs": {"/d": {"loaded": 3, "entries_on_disk": 5}},
            },
        )
        assert 'vllm_mlx_prefix_cache_hits{model="m1"} 3.0' in text
        assert 'vllm_mlx_prefix_cache_misses{model="m1"} 1.0' in text
        assert 'vllm_mlx_prefix_cache_discarded_hits{model="m1"} 1.0' in text
        assert 'vllm_mlx_prefix_cache_evictions{model="m1"} 2.0' in text
        assert 'vllm_mlx_prefix_cache_tokens_saved{model="m1"} 120.0' in text
        assert 'vllm_mlx_prefix_cache_entries{model="m1"} 4.0' in text
        assert 'vllm_mlx_prefix_cache_memory_bytes{model="m1"} 2.097152e+06' in text
        assert (
            'vllm_mlx_prefix_cache_memory_limit_bytes{model="m1",source="explicit"}'
            " 6.7108864e+07" in text
        )
        assert "vllm_mlx_prefix_cache_persistence_applies 1.0" in text
        assert "vllm_mlx_prefix_cache_persisted_entries_loaded 3.0" in text
        assert "vllm_mlx_prefix_cache_persisted_entries_on_disk 5.0" in text

    def test_two_models_get_two_label_sets_and_an_unloaded_one_drops_out(self, collector):
        both = {"a": _state(), "b": _state(counters={**_state()["counters"], "hits": 9})}
        text = _scrape(collector, cache_states=both)
        assert 'vllm_mlx_prefix_cache_hits{model="a"} 3.0' in text
        assert 'vllm_mlx_prefix_cache_hits{model="b"} 9.0' in text
        later = _scrape(collector, cache_states={"a": _state()})
        assert 'model="b"' not in later
        assert 'vllm_mlx_prefix_cache_hits{model="a"} 3.0' in later

    def test_an_engine_that_reports_nothing_gets_no_series_not_zeros(self, collector):
        unreported = {"counters": {"value": "unreported", "reason": "x"},
                      "memory_limit": {"value": "unreported", "reason": "x"}}
        text = _scrape(collector, cache_states={"m": unreported})
        assert 'vllm_mlx_prefix_cache_hits{model="m"}' not in text
        assert "vllm_mlx_prefix_cache_memory_limit_bytes{" not in text

    def test_missing_cache_state_exports_no_per_model_series(self, collector):
        text = _scrape(collector)
        assert "vllm_mlx_prefix_cache_hits{" not in text
        assert "vllm_mlx_prefix_cache_persistence_applies 0.0" in text

    def test_the_unlabeled_gauges_say_whether_the_engine_reports(self, collector):
        class Quiet:
            def get_stats(self):
                return {}

        assert "vllm_mlx_cache_stats_reported 0.0" in _scrape(collector, engine=Quiet())

        class WithCache(Quiet):
            def get_stats(self):
                return {"memory_aware_cache": {"hits": 2, "discarded_hits": 1}}

        text = _scrape(collector, engine=WithCache())
        assert "vllm_mlx_cache_stats_reported 1.0" in text
        assert "vllm_mlx_cache_discarded_hits 1.0" in text

    def test_the_simple_engines_trie_counters_reach_the_cache_gauges(self, collector):
        class Simple:
            def get_stats(self):
                return {"engine_type": "simple"}

            def get_cache_stats(self):
                return {"hits": 4, "misses": 1, "tokens_saved": 900, "entry_count": 2,
                        "current_memory_mb": 1.0}

        text = _scrape(collector, engine=Simple())
        assert 'vllm_mlx_cache_type{cache_type="prefix_trie_cache"} 1.0' in text
        assert "vllm_mlx_cache_hits 4.0" in text
        assert "vllm_mlx_cache_tokens_saved 900.0" in text
        assert "vllm_mlx_cache_stats_reported 1.0" in text

    def test_endpoint_exports_the_single_model_series(self, metrics_client):
        client, server, collector = metrics_client
        collector.configure(enabled=True)

        class Eng(FakeEngine):
            def get_cache_stats(self):
                return {"hits": 7, "misses": 2, "tokens_saved": 50, "entry_count": 1,
                        "current_memory_mb": 1.0,
                        "memory_limit": {"bytes": 1 << 20, "source": "fallback_8gb"}}

        server._engine = Eng()
        text = client.get("/metrics").text
        assert 'vllm_mlx_prefix_cache_hits{model="metrics-model"} 7.0' in text
        assert 'source="fallback_8gb"' in text

    def test_endpoint_survives_a_cache_state_failure(self, metrics_client):
        client, server, collector = metrics_client
        collector.configure(enabled=True)

        class Broken(FakeEngine):
            def get_cache_stats(self):
                raise RuntimeError("boom")

        server._engine = Broken()
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_endpoint_exports_every_loaded_registry_model(self, metrics_client, monkeypatch):
        client, server, collector = metrics_client
        collector.configure(enabled=True)

        def eng(hits):
            class E(FakeEngine):
                def get_cache_stats(self):
                    return {"hits": hits, "misses": 0, "tokens_saved": 1, "entry_count": 1,
                            "current_memory_mb": 1.0}

            return E()

        class Manager:
            def get_metrics_engine(self):
                return eng(1)

            def loaded_engines(self):
                return [("alpha", eng(1)), ("beta", eng(5))]

            async def shutdown(self):
                return None

        monkeypatch.setattr(server, "_model_manager", Manager())
        text = client.get("/metrics").text
        assert 'vllm_mlx_prefix_cache_hits{model="alpha"} 1.0' in text
        assert 'vllm_mlx_prefix_cache_hits{model="beta"} 5.0' in text

    def test_a_model_name_with_quotes_and_newlines_is_escaped(self, collector):
        text = _scrape(collector, cache_states={'we"ird\nname': _state()})
        assert 'model="we\\"ird\\nname"' in text

    def test_an_engine_without_top_level_hits_does_not_feed_the_trie_fallback(self, collector):
        class Other:
            def get_stats(self):
                return {}

            def get_cache_stats(self):
                return {"mllm_cache": {"entries": 1}}

        text = _scrape(collector, engine=Other())
        assert 'cache_type="prefix_trie_cache"} 0.0' in text
        assert "vllm_mlx_cache_stats_reported 0.0" in text

    def test_a_scrape_asks_the_engine_for_cache_stats_once(self, metrics_client):
        client, server, collector = metrics_client
        collector.configure(enabled=True)
        calls = []

        class Eng(FakeEngine):
            def get_stats(self):
                return {}

            def get_cache_stats(self):
                calls.append(1)
                return {"hits": 1, "misses": 0, "tokens_saved": 1, "entry_count": 1,
                        "current_memory_mb": 1.0}

        server._engine = Eng()
        client.get("/metrics")
        assert len(calls) == 1

    def test_persistence_does_not_apply_when_no_engine_is_loaded(self, metrics_client):
        client, server, collector = metrics_client
        collector.configure(enabled=True)
        assert server._engine is None
        assert "vllm_mlx_prefix_cache_persistence_applies 0.0" in client.get("/metrics").text
