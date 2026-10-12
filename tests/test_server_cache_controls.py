# SPDX-License-Identifier: Apache-2.0
"""Tests for cache control endpoints."""

import sys
import types

import pytest

from fastapi.testclient import TestClient


def test_cache_stats_includes_engine_cache(monkeypatch):
    import vllm_mlx.server as server

    fake_utils = types.ModuleType("mlx_vlm.utils")
    fake_utils.get_multimodal_kv_cache_stats = lambda: {"entries": 1}
    fake_utils.get_pixel_values_cache_stats = lambda: {"entries": 2}
    fake_utils.get_pil_cache_stats = lambda: {"entries": 3}

    class DummyEngine:
        def get_cache_stats(self):
            return {"prefix_cache": {"hits": 7, "misses": 2}}

    original_engine = server._engine
    original_api_key = server._api_key
    original_module = sys.modules.get("mlx_vlm.utils")
    try:
        server._engine = DummyEngine()
        server._api_key = None
        sys.modules["mlx_vlm.utils"] = fake_utils
        client = TestClient(server.app)

        response = client.get("/v1/cache/stats")
        assert response.status_code == 200
        assert response.json()["engine_cache"] == {
            "prefix_cache": {"hits": 7, "misses": 2}
        }
    finally:
        server._engine = original_engine
        server._api_key = original_api_key
        if original_module is not None:
            sys.modules["mlx_vlm.utils"] = original_module
        else:
            sys.modules.pop("mlx_vlm.utils", None)


def test_clear_cache_clears_engine_managed_runtime_caches(monkeypatch):
    import vllm_mlx.server as server

    calls = {"multimodal": 0, "pixel": 0, "engine": 0}
    fake_utils = types.ModuleType("mlx_vlm.utils")

    def clear_multimodal():
        calls["multimodal"] += 1

    def clear_pixel():
        calls["pixel"] += 1

    fake_utils.clear_multimodal_kv_cache = clear_multimodal
    fake_utils.clear_pixel_values_cache = clear_pixel

    class DummyEngine:
        def clear_runtime_caches(self):
            calls["engine"] += 1
            return {"prefix_cache": True}

    original_engine = server._engine
    original_api_key = server._api_key
    original_module = sys.modules.get("mlx_vlm.utils")
    try:
        server._engine = DummyEngine()
        server._api_key = None
        sys.modules["mlx_vlm.utils"] = fake_utils
        client = TestClient(server.app)

        response = client.delete("/v1/cache")
        assert response.status_code == 200
        assert response.json()["engine_cache"] == {"prefix_cache": True}
        assert calls == {"multimodal": 1, "pixel": 1, "engine": 1}
    finally:
        server._engine = original_engine
        server._api_key = original_api_key
        if original_module is not None:
            sys.modules["mlx_vlm.utils"] = original_module
        else:
            sys.modules.pop("mlx_vlm.utils", None)


STATS_HOOKS = {
    "get_multimodal_kv_cache_stats": {"entries": 1},
    "get_pixel_values_cache_stats": {"entries": 2},
    "get_pil_cache_stats": {"entries": 3},
}
STATS_KEYS = ("multimodal_kv_cache", "pixel_values_cache", "pil_image_cache")


class _Engine:
    def __init__(self):
        self.cleared = 0

    def get_cache_stats(self):
        return {"prefix_cache": {"hits": 7, "misses": 2}}

    def clear_runtime_caches(self):
        self.cleared += 1
        return {"prefix_cache": True}


def _client(monkeypatch, utils):
    """A test client whose ``mlx_vlm.utils`` is ``utils`` (None: not importable)."""
    import vllm_mlx.server as server

    monkeypatch.setattr(server, "_engine", _Engine())
    monkeypatch.setattr(server, "_api_key", None)
    monkeypatch.setitem(sys.modules, "mlx_vlm.utils", utils)
    return TestClient(server.app), server._engine


def _utils(**funcs):
    mod = types.ModuleType("mlx_vlm.utils")
    for name, value in funcs.items():
        setattr(mod, name, value)
    return mod


def test_cache_stats_without_mlx_vlm_helpers_keeps_engine_stats_and_has_no_error(
    monkeypatch,
):
    # stock mlx-vlm (through 0.7.6) has none of the helpers: that is the normal case, not a failure
    for utils in (None, _utils()):
        client, _ = _client(monkeypatch, utils)
        body = client.get("/v1/cache/stats").json()
        assert "error" not in body
        assert body["engine_cache"] == {"prefix_cache": {"hits": 7, "misses": 2}}
        assert "cache_state" in body and "persistence" in body
        assert not any(k in body for k in STATS_KEYS)
        assert body["multimodal_cache_missing"] == sorted(STATS_HOOKS)


def test_cache_stats_reports_the_helpers_that_exist_and_names_the_missing_ones(
    monkeypatch,
):
    client, _ = _client(
        monkeypatch,
        _utils(get_pixel_values_cache_stats=lambda: {"entries": 2}),
    )
    body = client.get("/v1/cache/stats").json()
    assert body["pixel_values_cache"] == {"entries": 2}
    assert "multimodal_kv_cache" not in body and "pil_image_cache" not in body
    assert body["multimodal_cache_missing"] == [
        "get_multimodal_kv_cache_stats",
        "get_pil_cache_stats",
    ]


def test_cache_stats_with_every_helper_has_no_missing_list(monkeypatch):
    client, _ = _client(
        monkeypatch, _utils(**{n: (lambda v=v: v) for n, v in STATS_HOOKS.items()})
    )
    body = client.get("/v1/cache/stats").json()
    assert [body[k] for k in STATS_KEYS] == list(STATS_HOOKS.values())
    assert "multimodal_cache_missing" not in body and "error" not in body


def test_a_failing_helper_is_not_swallowed(monkeypatch):
    # only a missing helper is optional: one that exists and raises is a real fault
    def boom():
        raise RuntimeError("stats broke")

    client, _ = _client(monkeypatch, _utils(get_pil_cache_stats=boom))
    with pytest.raises(RuntimeError, match="stats broke"):
        client.get("/v1/cache/stats")


def test_clear_cache_without_mlx_vlm_helpers_still_clears_the_engine(monkeypatch):
    for utils in (None, _utils()):
        client, engine = _client(monkeypatch, utils)
        before = engine.cleared
        body = client.delete("/v1/cache").json()
        assert engine.cleared == before + 1
        assert body["status"] == "cleared" and "error" not in body
        assert body["engine_cache"] == {"prefix_cache": True}
        assert body["caches"] == []
        assert body["multimodal_cache_missing"] == [
            "clear_multimodal_kv_cache",
            "clear_pixel_values_cache",
        ]


def test_clear_cache_calls_only_the_helpers_that_exist(monkeypatch):
    calls = []
    client, _ = _client(
        monkeypatch, _utils(clear_pixel_values_cache=lambda: calls.append("pixel"))
    )
    body = client.delete("/v1/cache").json()
    assert calls == ["pixel"]
    assert body["caches"] == ["pixel_values", "pil_image"]
    assert body["multimodal_cache_missing"] == ["clear_multimodal_kv_cache"]
