# SPDX-License-Identifier: Apache-2.0
"""Cache observability (py-vllm-mlx#37): per-request cached tokens, the cache_state snapshot,
inert-option reporting, the memory-limit derivation and the quantization-aware fingerprint.

Rule under test: a value the engine does not report is omitted or ``"unreported"``, never a
default that looks like data (an unreported cached-token count is not 0).
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm_mlx import cache_state
from vllm_mlx.api.models import PromptTokensDetails, Usage
from vllm_mlx.cli import create_parser
from vllm_mlx.engine.base import GenerationOutput
from vllm_mlx.output_collector import RequestOutputCollector
from vllm_mlx.request import Request, RequestOutput, SamplingParams

# --- Usage.prompt_tokens_details --------------------------------------------------------


def test_usage_omits_details_when_unreported():
    u = Usage(prompt_tokens=5, completion_tokens=2, total_tokens=7)
    assert "prompt_tokens_details" not in u.model_dump()
    assert "prompt_tokens_details" not in json.loads(u.model_dump_json())


def test_usage_reports_zero_cached_tokens_as_zero_not_omitted():
    u = Usage(prompt_tokens=5, prompt_tokens_details=PromptTokensDetails(cached_tokens=0))
    assert u.model_dump()["prompt_tokens_details"] == {"cached_tokens": 0}


def test_get_usage_distinguishes_unreported_zero_and_positive():
    from vllm_mlx.server import get_usage

    unreported = get_usage(GenerationOutput(text="x", prompt_tokens=9, completion_tokens=1))
    zero = get_usage(GenerationOutput(text="x", prompt_tokens=9, completion_tokens=1, cached_tokens=0))
    some = get_usage(GenerationOutput(text="x", prompt_tokens=9, completion_tokens=1, cached_tokens=6))
    assert unreported.prompt_tokens_details is None
    assert zero.prompt_tokens_details.cached_tokens == 0
    assert some.prompt_tokens_details.cached_tokens == 6
    assert some.total_tokens == 10


# --- the value travels from the scheduler to the API ----------------------------------------


def test_defaults_mean_unreported():
    assert RequestOutput(request_id="r").cached_tokens is None
    assert GenerationOutput(text="").cached_tokens is None


def test_collector_merge_keeps_the_latest_reported_value():
    c = RequestOutputCollector(aggregate=True)
    first = RequestOutput(request_id="r", new_text="a", cached_tokens=4)
    later = RequestOutput(request_id="r", new_text="b")  # producer ahead of consumer
    assert c._merge_outputs(first, later).cached_tokens == 4
    assert c._merge_outputs(first, RequestOutput(request_id="r", cached_tokens=5)).cached_tokens == 5


class _Resp:
    def __init__(self, uid, token, finish_reason=None):
        self.uid, self.token, self.finish_reason = uid, token, finish_reason
        self.logprobs, self.cache_out = None, None


def _scheduler():
    from vllm_mlx.scheduler import Scheduler, SchedulerConfig

    tok = MagicMock()
    tok.encode = lambda x: list(range(len(x.split())))
    tok.eos_token_id = 0
    return Scheduler(MagicMock(), tok, SchedulerConfig(enable_prefix_cache=False))


@pytest.mark.parametrize("cached", [0, 3])
def test_scheduler_copies_the_requests_cached_tokens_onto_its_output(cached):
    s = _scheduler()
    req = Request(request_id="a", prompt="one two three four", sampling_params=SamplingParams())
    req.output_token_ids = [1]  # skip the prompt-only cache store
    req.cached_tokens = cached
    s.running = {"a": req}
    s.uid_to_request_id = {1: "a"}
    s.batch_generator = SimpleNamespace()
    outputs, _ = s._process_batch_responses([_Resp(1, 5, "stop")])
    assert outputs[0].cached_tokens == cached


# --- the endpoints ----------------------------------------------------------------------------


def _patch_server(monkeypatch, engine):
    import vllm_mlx.server as server

    async def acquire(raw_request, **kw):
        return engine

    async def release(**kw):
        return None

    monkeypatch.setattr(server, "_validate_model_name", lambda _m: None)
    monkeypatch.setattr(server, "_acquire_default_engine_for_request", acquire)
    monkeypatch.setattr(server, "_release_default_engine", release)
    monkeypatch.setattr(server, "_model_name", "served-model")
    monkeypatch.setattr(server, "_default_max_tokens", 64)
    monkeypatch.setattr(server, "_default_timeout", 30.0)
    monkeypatch.setattr(server, "_enable_auto_tool_choice", False)
    monkeypatch.setattr(server, "_tool_call_parser", None)
    monkeypatch.setattr(server, "_tool_parser_instance", None)
    monkeypatch.setattr(server, "_reasoning_parser_name", None)
    monkeypatch.setattr(server, "_reasoning_parser", None)
    return server


class _Engine:
    model_name = "fake"
    is_mllm = False
    preserve_native_tool_format = False

    def __init__(self, cached):
        self.cached = cached

    def _out(self, **kw):
        return GenerationOutput(
            text="ok", prompt_tokens=10, completion_tokens=2, finish_reason="stop",
            cached_tokens=self.cached, **kw,
        )

    async def chat(self, messages, **kwargs):
        return self._out()

    async def generate(self, prompt, **kwargs):
        return self._out()

    async def stream_chat(self, messages, **kwargs):
        yield self._out(new_text="ok", finished=True)

    async def stream_generate(self, prompt, **kwargs):
        yield self._out(new_text="ok", finished=True)


@pytest.mark.anyio
@pytest.mark.parametrize(("cached", "expect"), [(None, None), (0, 0), (7, 7)])
async def test_chat_completion_usage_reports_cached_tokens(monkeypatch, cached, expect):
    server = _patch_server(monkeypatch, _Engine(cached))
    req = server.ChatCompletionRequest(
        model="served-model", messages=[server.Message(role="user", content="hi")], max_tokens=4
    )
    resp = await server.create_chat_completion(req, raw_request=None)
    usage = json.loads(resp.model_dump_json())["usage"]
    if expect is None:
        assert "prompt_tokens_details" not in usage
    else:
        assert usage["prompt_tokens_details"] == {"cached_tokens": expect}
    assert usage["prompt_tokens"] == 10


@pytest.mark.anyio
@pytest.mark.parametrize(("cached", "expect"), [(None, None), (0, 0), (7, 7)])
async def test_streaming_chat_usage_chunk_reports_cached_tokens(monkeypatch, cached, expect):
    server = _patch_server(monkeypatch, _Engine(cached))
    req = server.ChatCompletionRequest(
        model="served-model",
        messages=[server.Message(role="user", content="hi")],
        stream=True,
        stream_options={"include_usage": True},
    )
    chunks = [c async for c in server.stream_chat_completion(_Engine(cached), req.messages, req)]
    payloads = [
        json.loads(c.removeprefix("data: ").strip()) for c in chunks if c != "data: [DONE]\n\n"
    ]
    usages = [p["usage"] for p in payloads if p.get("usage")]
    assert usages, "no usage chunk"
    for u in usages:
        if expect is None:
            assert "prompt_tokens_details" not in u
        else:
            assert u["prompt_tokens_details"] == {"cached_tokens": expect}


@pytest.mark.anyio
async def test_completions_usage_sums_cached_tokens_across_prompts_and_omits_if_any_unreported(
    monkeypatch,
):
    class Mixed(_Engine):
        def __init__(self, values):
            super().__init__(None)
            self.values = list(values)

        async def generate(self, prompt, **kwargs):
            self.cached = self.values.pop(0)
            return self._out()

    server = _patch_server(monkeypatch, None)

    async def run(values):
        eng = Mixed(values)
        monkeypatch.setattr(server, "_acquire_default_engine_for_request",
                            lambda raw, **kw: _coro(eng))
        req = server.CompletionRequest(model="served-model", prompt=["a b", "c d"], max_tokens=2)
        r = await server.create_completion(req, raw_request=None)
        return json.loads(r.model_dump_json())["usage"]

    both = await run([3, 4])
    assert both["prompt_tokens_details"] == {"cached_tokens": 7}
    one_missing = await run([3, None])
    assert "prompt_tokens_details" not in one_missing


async def _coro(value):
    return value


# --- inert options and the launch record ---------------------------------------------------


def _args(*argv):
    return create_parser().parse_args(["serve", "model", *argv])


def test_table_defaults_match_the_real_parser():
    ns = _args()
    for flag, (attr, default) in cache_state.CONTINUOUS_BATCHING_ONLY.items():
        assert getattr(ns, attr) == default, flag


def test_simple_engine_reports_the_cache_options_it_ignores():
    ns = _args("--kv-cache-quantization", "--cache-memory-mb", "512", "--use-paged-cache")
    assert cache_state.inert_options(ns) == [
        "--cache-memory-mb", "--kv-cache-quantization", "--use-paged-cache",
    ]
    opts = cache_state.launch_options(ns)
    assert opts["continuous_batching"] is False
    assert opts["kv_cache_quantization"]["requested"] is True
    assert opts["kv_cache_quantization"]["effective"] is False


def test_nothing_is_inert_with_defaults_or_with_continuous_batching():
    assert cache_state.inert_options(_args()) == []
    cb = _args("--continuous-batching", "--kv-cache-quantization", "--cache-memory-mb", "512")
    assert cache_state.inert_options(cb) == []
    assert cache_state.launch_options(cb)["kv_cache_quantization"]["effective"] is True


def test_registry_mode_ignores_persistence_options_and_the_cli_says_so():
    quiet = _args("--models-config", "m.yaml")
    assert cache_state.registry_ignores_persistence(quiet) is False  # defaults: nothing to warn
    for extra in (["--prefix-cache-dir", "/x"], ["--prefix-cache-persist", "none"],
                  ["--prefix-cache-reset", "start"]):
        ns = create_parser().parse_args(["serve", "--models-config", "m.yaml", *extra])
        assert cache_state.registry_ignores_persistence(ns) is True, extra
    # single-model serving honors them: no warning
    assert cache_state.registry_ignores_persistence(_args("--prefix-cache-persist", "none")) is False


# --- the cache_state block ------------------------------------------------------------------


def test_build_marks_what_the_engine_cannot_report_as_unreported_with_a_reason():
    state = cache_state.build(
        engine=SimpleNamespace(), launch=None, engine_cache={"system_kv_cache": {}},
        persistence={"policy": {}, "dirs": {}}, registry_mode=False,
    )
    assert state["schema_version"] == cache_state.SCHEMA_VERSION
    assert state["counters"]["value"] == cache_state.UNREPORTED and state["counters"]["reason"]
    assert state["memory_limit"]["value"] == cache_state.UNREPORTED
    assert state["launch_options"]["value"] == cache_state.UNREPORTED
    assert state["inert_options"] == cache_state.UNREPORTED
    assert state["engine"]["continuous_batching"] == cache_state.UNREPORTED


def test_build_reports_counters_and_the_memory_limit_derivation():
    stats = {
        "hits": 3, "misses": 1, "evictions": 0, "tokens_saved": 120, "entry_count": 2,
        "current_memory_mb": 1.5, "max_memory_mb": 64.0, "hit_rate": 0.75,
        "memory_limit": {"bytes": 64 << 20, "source": "explicit", "max_memory_mb": 64},
    }
    state = cache_state.build(
        engine=SimpleNamespace(), launch=cache_state.launch_options(_args("--continuous-batching")),
        engine_cache=stats, persistence={"policy": {}, "dirs": {}}, registry_mode=True,
    )
    assert state["counters"] == {
        "hits": 3, "misses": 1, "evictions": 0, "tokens_saved": 120, "entry_count": 2,
        "current_memory_mb": 1.5, "max_memory_mb": 64.0,
    }
    assert state["memory_limit"]["source"] == "explicit"
    assert state["engine"] == {"class": "SimpleNamespace", "continuous_batching": True,
                               "registry_mode": True}
    assert set(state["versions"]) == {"vllm_mlx", "mlx", "mlx_lm"}


def test_build_with_no_engine_is_unreported_not_empty():
    state = cache_state.build(
        engine=None, launch=None, engine_cache=None, persistence={}, registry_mode=False
    )
    assert state["engine"]["class"] is None
    assert state["counters"]["reason"] == "no engine is loaded"


@pytest.mark.parametrize("with_mlx_vlm", [False, True])
def test_cache_stats_endpoint_carries_cache_state(monkeypatch, with_mlx_vlm):
    import sys
    import types

    import vllm_mlx.server as server
    from fastapi.testclient import TestClient

    if with_mlx_vlm:  # the endpoint has a separate return path when mlx_vlm is importable
        utils = types.ModuleType("mlx_vlm.utils")
        utils.get_multimodal_kv_cache_stats = lambda: {}
        utils.get_pil_cache_stats = lambda: {}
        utils.get_pixel_values_cache_stats = lambda: {}
        monkeypatch.setitem(sys.modules, "mlx_vlm", types.ModuleType("mlx_vlm"))
        monkeypatch.setitem(sys.modules, "mlx_vlm.utils", utils)

    class Eng:
        def get_cache_stats(self):
            return {"hits": 1, "misses": 0, "evictions": 0, "tokens_saved": 5, "entry_count": 1,
                    "memory_limit": {"bytes": 1, "source": "explicit"}}

    monkeypatch.setattr(server, "_engine", Eng())
    monkeypatch.setattr(server, "_api_key", None)
    server.set_cache_launch_options(cache_state.launch_options(_args("--continuous-batching")))
    try:
        body = TestClient(server.app).get("/v1/cache/stats").json()
    finally:
        server.set_cache_launch_options(None)
    assert body["cache_state"]["counters"]["hits"] == 1
    assert body["cache_state"]["persistence"] == body["persistence"]
    assert body["cache_state"]["launch_options"]["continuous_batching"] is True
    assert ("multimodal_kv_cache" in body) is with_mlx_vlm


# --- memory limit derivation ------------------------------------------------------------------


def test_memory_limit_details_explicit_percent_and_fallback(monkeypatch):
    import vllm_mlx.memory_cache as mc

    explicit = mc.MemoryCacheConfig(max_memory_mb=100).memory_limit_details()
    assert explicit["source"] == "explicit" and explicit["bytes"] == 100 * mc._BYTES_PER_MB

    monkeypatch.setattr(mc, "_get_available_memory", lambda: 10 * 1024 * mc._BYTES_PER_MB)
    pct = mc.MemoryCacheConfig(max_memory_percent=0.25).memory_limit_details()
    assert pct["source"] == "percent_of_available"
    assert pct["available_bytes"] == 10 * 1024 * mc._BYTES_PER_MB and pct["percent"] == 0.25
    assert pct["bytes"] == int(10 * 1024 * mc._BYTES_PER_MB * 0.25)

    monkeypatch.setattr(mc, "_get_available_memory", lambda: 0)
    assert mc.MemoryCacheConfig().memory_limit_details()["source"] == "fallback_8gb"


def test_cache_get_stats_includes_how_the_limit_was_derived():
    from vllm_mlx.memory_cache import MemoryAwarePrefixCache, MemoryCacheConfig

    cache = MemoryAwarePrefixCache(MagicMock(), MemoryCacheConfig(max_memory_mb=64))
    stats = cache.get_stats()
    assert stats["memory_limit"]["source"] == "explicit"
    assert stats["memory_limit"]["bytes"] == 64 * (1 << 20)


# --- the fingerprint covers weight quantization --------------------------------------------


def _quant_model(bits=4, group_size=64, quantized=True):
    import mlx.nn as nn

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = (
                nn.QuantizedLinear(64, 64, bias=False, group_size=group_size, bits=bits)
                if quantized
                else nn.Linear(64, 64, bias=False)
            )
            self.args = SimpleNamespace(
                num_hidden_layers=2, hidden_size=64, vocab_size=100, model_type="t"
            )

    return M()


def test_fingerprint_differs_across_bits_group_size_and_unquantized():
    from vllm_mlx.memory_cache import _compute_model_fingerprint as fp

    base = fp(_quant_model(4, 64))
    assert fp(_quant_model(4, 64)) == base  # deterministic
    assert fp(_quant_model(6, 64)) != base
    assert fp(_quant_model(8, 64)) != fp(_quant_model(6, 64))
    assert fp(_quant_model(4, 32)) != base
    assert fp(_quant_model(quantized=False)) != base


def test_quantization_signature_reports_none_and_counts():
    from vllm_mlx.memory_cache import _quantization_signature as sig

    assert sig(_quant_model(quantized=False)) == "none"
    assert sig(_quant_model(4, 64)) == "QuantizedLinear:4:64:affinex1"


def test_quantization_signature_unavailable_is_visible_not_silent(caplog):
    from vllm_mlx.memory_cache import _quantization_signature as sig

    with caplog.at_level("WARNING"):
        assert sig(object()) == "unavailable"
    assert "cannot read the model's quantization" in caplog.text


def _kv_entry():
    import mlx.core as mx
    from mlx_lm.models.cache import KVCache

    layer = KVCache()
    layer.keys = mx.ones((1, 1, 8, 4), dtype=mx.float32)
    layer.values = layer.keys
    layer.offset = 8
    return [layer]


def _cache_for(model):
    from vllm_mlx.memory_cache import MemoryAwarePrefixCache, MemoryCacheConfig

    return MemoryAwarePrefixCache(
        model, MemoryCacheConfig(max_memory_mb=64, max_entries=10, min_prefix_tokens=1)
    )


def test_a_persisted_cache_is_refused_by_a_differently_quantized_model(tmp_path):
    src = _cache_for(_quant_model(4, 64))
    assert src.store(list(range(8)), _kv_entry())
    assert src.save_to_disk(str(tmp_path))

    assert _cache_for(_quant_model(4, 64)).load_from_disk(str(tmp_path)) == 1  # same checkpoint
    assert _cache_for(_quant_model(6, 64)).load_from_disk(str(tmp_path)) == 0  # other bits
    assert _cache_for(_quant_model(quantized=False)).load_from_disk(str(tmp_path)) == 0


# --- BatchedEngine copies the scheduler's value onto GenerationOutput -----------------------


def _batched(fake_core):
    from vllm_mlx.engine.batched import BatchedEngine

    eng = BatchedEngine.__new__(BatchedEngine)
    eng._loaded = True
    eng._is_mllm = False
    eng._mllm_scheduler = None
    eng._engine = fake_core
    return eng


class _FakeCore:
    def __init__(self, cached):
        self.cached = cached

    def _out(self, **kw):
        return RequestOutput(
            request_id="r", output_text="hi", new_text="hi", output_token_ids=[1],
            finished=True, finish_reason="stop", prompt_tokens=12, completion_tokens=1,
            cached_tokens=self.cached, **kw,
        )

    async def generate(self, prompt, sampling_params):
        return self._out()

    async def add_request(self, prompt, sampling_params, prefix_boundary=0):
        return "r"

    async def stream_outputs(self, request_id):
        yield self._out()


@pytest.mark.anyio
@pytest.mark.parametrize("cached", [None, 0, 9])
async def test_batched_engine_generate_and_stream_carry_cached_tokens(cached):
    eng = _batched(_FakeCore(cached))
    assert (await eng.generate("p")).cached_tokens == cached
    streamed = [o async for o in eng.stream_generate("p")]
    assert [o.cached_tokens for o in streamed] == [cached]


# --- registry mode: per-model state ----------------------------------------------------------


def test_registry_mode_reports_each_loaded_model_and_that_persistence_does_not_apply(
    monkeypatch, tmp_path
):
    import vllm_mlx.server as server
    from fastapi.testclient import TestClient
    from vllm_mlx.prefix_cache_persistence import PersistencePolicy

    class Eng:
        def __init__(self, hits):
            self.hits = hits

        def get_cache_stats(self):
            return {"hits": self.hits, "misses": 0, "evictions": 0, "tokens_saved": 0,
                    "entry_count": 0, "memory_limit": {"bytes": 1, "source": "explicit"}}

    class Manager:
        def loaded_engines(self):
            return [("a", Eng(1)), ("b", Eng(2))]

    server.set_prefix_cache_policy(PersistencePolicy(base_dir=str(tmp_path)))
    monkeypatch.setattr(server, "_engine", None)
    monkeypatch.setattr(server, "_model_manager", Manager())
    monkeypatch.setattr(server, "_api_key", None)
    try:
        body = TestClient(server.app).get("/v1/cache/stats").json()
    finally:
        server.set_prefix_cache_policy(PersistencePolicy())
    assert body["cache_state"]["counters"]["reason"].startswith("registry mode")
    assert body["cache_state"]["engine"]["registry_mode"] is True
    assert set(body["models"]) == {"a", "b"}
    assert body["models"]["a"]["cache_state"]["counters"]["hits"] == 1
    assert body["models"]["b"]["cache_state"]["counters"]["hits"] == 2
    for p in (body["persistence"], body["models"]["a"]["cache_state"]["persistence"]):
        assert p["applies"] is False and "registry mode" in p["not_applied_reason"]


def test_persistence_applies_outside_registry_mode(monkeypatch):
    import vllm_mlx.server as server

    monkeypatch.setattr(server, "_model_manager", None)
    snap = server._persistence_snapshot()
    assert snap["applies"] is True and "not_applied_reason" not in snap


def test_a_single_model_server_has_no_models_key(monkeypatch):
    import vllm_mlx.server as server
    from fastapi.testclient import TestClient

    monkeypatch.setattr(server, "_engine", None)
    monkeypatch.setattr(server, "_model_manager", None)
    monkeypatch.setattr(server, "_api_key", None)
    assert "models" not in TestClient(server.app).get("/v1/cache/stats").json()


def test_model_manager_lists_only_loaded_engines():
    from vllm_mlx.model_registry import ModelManager

    mgr = ModelManager.__new__(ModelManager)
    mgr._loaded = {
        "x": SimpleNamespace(engine="ex"),
        "y": SimpleNamespace(engine="ey"),
    }
    assert mgr.loaded_engines() == [("x", "ex"), ("y", "ey")]
    mgr._loaded = {}
    assert mgr.loaded_engines() == []
