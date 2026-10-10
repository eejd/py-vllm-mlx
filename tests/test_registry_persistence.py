# SPDX-License-Identifier: Apache-2.0
"""Registry mode (--models-config) loads and saves persisted prefix caches (py-vllm-mlx#41)."""

import asyncio
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

import vllm_mlx.server as server
from vllm_mlx.prefix_cache_persistence import PersistencePolicy


class CachingEngine:
    """A continuous-batching-like engine: it has the persisted-cache hooks."""

    def __init__(self, entries=0):
        self.entries = entries
        self.loaded_from: list[str] = []
        self.saved_to: list[str] = []

    def load_cache_from_disk(self, cache_dir):
        self.loaded_from.append(cache_dir)
        return self.entries

    def save_cache_to_disk(self, cache_dir):
        self.saved_to.append(cache_dir)
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        return True


class SimpleLikeEngine:
    """No persisted-cache hooks (the Simple engine)."""


def _config(tmp_path, name):
    source = tmp_path / "models" / name
    source.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        resolved_source=str(source),
        entry=SimpleNamespace(name=name, source=str(source)),
    )


@pytest.fixture(autouse=True)
def _policy(tmp_path):
    server.set_prefix_cache_policy(PersistencePolicy(base_dir=str(tmp_path / "cache")))
    yield
    server.set_prefix_cache_policy(PersistencePolicy())


def test_each_model_loads_from_and_saves_to_its_own_directory(tmp_path):
    async def _run():
        a, b = CachingEngine(entries=2), CachingEngine(entries=0)
        ca, cb = _config(tmp_path, "alpha"), _config(tmp_path, "beta")
        await server._registry_restore_engine_state(ca, a)
        await server._registry_restore_engine_state(cb, b)
        await server._registry_persist_engine_state(ca, a)
        await server._registry_persist_engine_state(cb, b)
        assert a.loaded_from == a.saved_to and len(a.loaded_from) == 1
        assert b.loaded_from == b.saved_to and len(b.loaded_from) == 1
        assert a.loaded_from != b.loaded_from
        assert str(tmp_path / "cache") in a.loaded_from[0]
        # the same key as single-model serving: the real model path
        assert a.loaded_from[0] == server._get_cache_dir(ca.entry.source)
        state = server._persistence_snapshot()["dirs"]
        assert {d["loaded"] for d in state.values()} == {2, 0}

    asyncio.run(_run())


def test_the_persist_mode_is_honored_per_model(tmp_path):
    async def _run():
        server.set_prefix_cache_policy(
            PersistencePolicy(base_dir=str(tmp_path / "cache"), persist="load-only")
        )
        engine, cfg = CachingEngine(entries=1), _config(tmp_path, "alpha")
        await server._registry_restore_engine_state(cfg, engine)
        await server._registry_persist_engine_state(cfg, engine)
        assert len(engine.loaded_from) == 1
        assert engine.saved_to == []  # load-only never writes

    asyncio.run(_run())


def test_reset_at_start_applies_on_every_cold_load(tmp_path):
    async def _run():
        server.set_prefix_cache_policy(
            PersistencePolicy(
                base_dir=str(tmp_path / "cache"), persist="save-only", reset="start"
            )
        )
        engine, cfg = CachingEngine(), _config(tmp_path, "alpha")
        cache_dir = Path(server._get_cache_dir(cfg.entry.source))
        for _ in range(2):  # a load, then a reload after an unload
            cache_dir.mkdir(parents=True, exist_ok=True)
            (cache_dir / "index.json").write_text("{}")
            await server._registry_restore_engine_state(cfg, engine)
            assert not (cache_dir / "index.json").exists()

    asyncio.run(_run())


def test_a_simple_engine_is_skipped_and_the_user_is_told_when_options_were_given(
    tmp_path, caplog
):
    async def _run():
        server.set_prefix_cache_policy(
            PersistencePolicy(base_dir=str(tmp_path / "cache"), persist="load-only")
        )
        cfg = _config(tmp_path, "alpha")
        with caplog.at_level(logging.WARNING, logger="vllm_mlx.server"):
            await server._registry_restore_engine_state(cfg, SimpleLikeEngine())
            await server._registry_persist_engine_state(cfg, SimpleLikeEngine())
        assert "do not apply to model alpha" in caplog.text
        assert server._persistence_snapshot()["dirs"] == {}

    asyncio.run(_run())


def test_a_simple_engine_is_skipped_quietly_with_the_default_policy(tmp_path, caplog):
    async def _run():
        server.set_prefix_cache_policy(PersistencePolicy())
        with caplog.at_level(logging.WARNING, logger="vllm_mlx.server"):
            await server._registry_restore_engine_state(
                _config(tmp_path, "alpha"), SimpleLikeEngine()
            )
        assert "do not apply" not in caplog.text

    asyncio.run(_run())


def test_the_manager_is_built_with_the_persistence_hooks(tmp_path, monkeypatch):
    cfg = tmp_path / "models.yaml"
    (tmp_path / "m").mkdir()
    cfg.write_text(
        "manager:\n  memory_budget_gb: 8\n"
        f"models:\n  - name: m\n    path: {tmp_path / 'm'}\n    estimated_memory_gb: 1\n"
    )

    captured = {}
    real = server.ModelManager

    def spy(*args, **kwargs):
        captured.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(server, "ModelManager", spy)
    monkeypatch.setattr(server, "_model_manager", None)
    from tests.test_model_registry import _defaults

    server.load_model_registry(str(cfg), defaults=_defaults())
    assert captured["on_engine_loaded"] is server._registry_restore_engine_state
    assert captured["on_engine_unloading"] is server._registry_persist_engine_state


def test_a_repo_id_source_keeps_one_directory_across_snapshots_and_modes(tmp_path):
    async def _run():
        engine = CachingEngine()
        # two revisions of one repo id resolve to different snapshot paths
        for rev in ("aaa", "bbb"):
            cfg = SimpleNamespace(
                resolved_source=f"/hf/models--org--m/snapshots/{rev}",
                entry=SimpleNamespace(name="m", source="org/m"),
            )
            await server._registry_restore_engine_state(cfg, engine)
        assert engine.loaded_from[0] == engine.loaded_from[1]
        # and it is the directory single-model serving derives from the same argument
        assert engine.loaded_from[0] == server._get_cache_dir("org/m")

    asyncio.run(_run())


def test_two_entries_with_one_source_are_warned_about(tmp_path, monkeypatch, caplog):
    (tmp_path / "m").mkdir()
    cfg = tmp_path / "models.yaml"
    cfg.write_text(
        "manager:\n  memory_budget_gb: 8\nmodels:\n"
        f"  - name: a\n    path: {tmp_path / 'm'}\n    estimated_memory_gb: 1\n"
        f"  - name: b\n    path: {tmp_path / 'm'}\n    estimated_memory_gb: 1\n"
    )
    from tests.test_model_registry import _defaults

    monkeypatch.setattr(server, "_model_manager", None)
    with caplog.at_level(logging.WARNING, logger="vllm_mlx.server"):
        server.load_model_registry(str(cfg), defaults=_defaults())
    assert "share the source" in caplog.text
