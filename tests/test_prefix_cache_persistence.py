# SPDX-License-Identifier: Apache-2.0
"""Persisted prefix cache controls: directory, persist mode, reset on start/stop.

No model and no MLX arrays: a stub engine writes and reads the same file layout as
``MemoryAwarePrefixCache.save_to_disk`` (``index.json``, ``entry_<i>.safetensors``,
``entry_<i>_tokens.bin``), driven through the real server load/save hooks.
"""

import asyncio
import hashlib
import json
import os
import sys
import types
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import vllm_mlx.server as server
from vllm_mlx.cli import create_parser
from vllm_mlx.prefix_cache_persistence import (
    ENV_DIR,
    PERSIST_MODES,
    RESET_MODES,
    PersistenceError,
    PersistencePolicy,
    default_base_dir,
    describe_dir,
    reset_cache_dir,
)

# --- the valid persist x reset matrix (documented in the module docstring) ---------------

VALID = {
    ("auto", "never"), ("auto", "start"),
    ("none", "never"), ("none", "start"), ("none", "stop"), ("none", "both"),
    ("load-only", "never"),
    ("save-only", "never"), ("save-only", "start"),
}


@pytest.mark.parametrize("persist", PERSIST_MODES)
@pytest.mark.parametrize("reset", RESET_MODES)
def test_every_persist_reset_cell_is_accepted_or_refused_as_documented(persist, reset):
    if (persist, reset) in VALID:
        assert PersistencePolicy(persist=persist, reset=reset).persist == persist
    else:
        with pytest.raises(PersistenceError):
            PersistencePolicy(persist=persist, reset=reset)


def test_defaults_are_the_historical_behavior(monkeypatch):
    monkeypatch.delenv(ENV_DIR, raising=False)
    p = PersistencePolicy.from_options()
    assert (p.persist, p.reset, p.loads, p.saves) == ("auto", "never", True, True)
    assert p.resolve_dir("org/model") == os.path.join(default_base_dir(), "org--model")
    assert default_base_dir() == os.path.join(
        os.path.expanduser("~"), ".cache", "vllm-mlx", "prefix_cache"
    )


def test_directory_option_beats_env_and_env_beats_default(tmp_path):
    env = {ENV_DIR: str(tmp_path / "from-env")}
    assert PersistencePolicy.from_options(environ=env).base_dir == str(tmp_path / "from-env")
    opt = PersistencePolicy.from_options(str(tmp_path / "opt"), environ=env)
    assert opt.base_dir == str(tmp_path / "opt")
    assert opt.resolve_dir("a\\b") == str(tmp_path / "opt" / "a--b")
    assert PersistencePolicy.from_options(environ={}).base_dir is None


@pytest.mark.parametrize("bad", ["relative/dir", "./x", "x"])
def test_relative_base_directory_is_refused(bad):
    with pytest.raises(PersistenceError, match="absolute"):
        PersistencePolicy(base_dir=bad)


def test_unknown_modes_are_refused():
    with pytest.raises(PersistenceError):
        PersistencePolicy(persist="sometimes")
    with pytest.raises(PersistenceError):
        PersistencePolicy(reset="eventually")


# --- reset_cache_dir: narrow, and refuses what it must -----------------------------------


def _populate(d: Path, entries: int = 2) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / "index.json").write_text(json.dumps({"entries": [{"index": i} for i in range(entries)]}))
    for i in range(entries):
        (d / f"entry_{i}.safetensors").write_bytes(b"k" * 100)
        (d / f"entry_{i}_tokens.bin").write_bytes(b"t" * 8)


def test_describe_dir_counts_entries_and_bytes(tmp_path):
    d = tmp_path / "m"
    _populate(d, 3)
    (d / "notes.txt").write_text("not ours")
    entries, nbytes = describe_dir(d)
    assert entries == 3
    index_size = (d / "index.json").stat().st_size
    assert nbytes == 3 * 108 + index_size  # the unrelated file is not counted
    assert describe_dir(tmp_path / "missing") == (0, 0)


def test_reset_deletes_only_the_persistence_files_and_removes_an_empty_dir(tmp_path):
    d = tmp_path / "m"
    _populate(d, 2)
    r = reset_cache_dir(d)
    assert r.deleted_files == 5 and r.deleted_bytes > 0 and r.left_alone == []
    assert not d.exists()


def test_reset_leaves_foreign_files_in_place(tmp_path):
    d = tmp_path / "m"
    _populate(d, 1)
    (d / "keep.txt").write_text("keep me")
    (d / "entry_x.safetensors").write_text("not an entry name")
    r = reset_cache_dir(d)
    assert r.deleted_files == 3
    assert sorted(r.left_alone) == ["entry_x.safetensors", "keep.txt"]
    assert sorted(p.name for p in d.iterdir()) == ["entry_x.safetensors", "keep.txt"]


def test_a_directory_without_persistence_files_is_not_touched(tmp_path):
    d = tmp_path / "other"
    d.mkdir()
    (d / "a.txt").write_text("a")
    r = reset_cache_dir(d)
    assert r.deleted_files == 0 and r.left_alone == ["a.txt"]
    assert (d / "a.txt").exists()


def test_reset_of_a_missing_directory_is_a_no_op(tmp_path):
    assert reset_cache_dir(tmp_path / "nope").deleted_files == 0


def test_reset_refuses_a_symlinked_directory_and_does_not_follow_it(tmp_path):
    real = tmp_path / "real"
    _populate(real, 1)
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(PersistenceError, match="symlink"):
        reset_cache_dir(link)
    assert (real / "index.json").exists()


def test_reset_does_not_follow_a_symlink_named_like_an_entry(tmp_path):
    d = tmp_path / "m"
    _populate(d, 1)
    victim = tmp_path / "victim.bin"
    victim.write_bytes(b"precious")
    (d / "entry_9.safetensors").symlink_to(victim)
    r = reset_cache_dir(d)
    assert victim.read_bytes() == b"precious"
    assert "entry_9.safetensors" in r.left_alone


@pytest.mark.parametrize(
    "target",
    ["/", os.path.expanduser("~"), os.path.dirname(os.path.expanduser("~"))],
)
def test_reset_refuses_root_home_and_a_parent_of_home(target):
    with pytest.raises(PersistenceError, match="not a cache directory"):
        reset_cache_dir(target)


def test_reset_refuses_a_relative_path_and_a_regular_file(tmp_path):
    with pytest.raises(PersistenceError, match="relative"):
        reset_cache_dir("some/relative")
    f = tmp_path / "file"
    f.write_text("x")
    with pytest.raises(PersistenceError, match="not a directory"):
        reset_cache_dir(f)


# --- the real server hooks, with a stub engine -------------------------------------------


class StubEngine:
    """Writes and reads the layout of MemoryAwarePrefixCache.save_to_disk."""

    def __init__(self, entries: int = 2):
        self.entries = entries
        self.loads: list[str] = []
        self.saves: list[str] = []

    def save_cache_to_disk(self, cache_dir: str) -> bool:
        self.saves.append(cache_dir)
        _populate(Path(cache_dir), self.entries)
        return True

    def load_cache_from_disk(self, cache_dir: str) -> int:
        self.loads.append(cache_dir)
        idx = Path(cache_dir) / "index.json"
        return len(json.loads(idx.read_text())["entries"]) if idx.exists() else 0


@pytest.fixture
def policy_for(tmp_path):
    original_policy = server._prefix_cache_policy
    original_model_path = server._model_path

    def make(persist="auto", reset="never"):
        server.set_prefix_cache_policy(
            PersistencePolicy(base_dir=str(tmp_path / "base"), persist=persist, reset=reset)
        )
        server._model_path = "org/model"
        return tmp_path / "base" / "org--model"

    yield make
    server.set_prefix_cache_policy(original_policy)
    server._model_path = original_model_path


def _digest(d: Path) -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.iterdir())}


def _run(coro):
    return asyncio.run(coro)


def test_auto_saves_at_stop_and_loads_at_start(policy_for):
    d = policy_for("auto", "never")
    eng = StubEngine(2)
    _run(server._save_prefix_cache_to_disk(eng))
    assert eng.saves == [str(d)] and (d / "index.json").exists()
    eng2 = StubEngine()
    _run(server._load_prefix_cache_from_disk(eng2))
    assert eng2.loads == [str(d)]
    assert server._prefix_cache_state[str(d)]["loaded"] == 2
    assert server._prefix_cache_state[str(d)]["entries_on_disk"] == 2


def test_none_never_reads_or_writes(policy_for):
    d = policy_for("none", "never")
    _populate(d, 2)
    before = _digest(d)
    eng = StubEngine()
    _run(server._load_prefix_cache_from_disk(eng))
    _run(server._save_prefix_cache_to_disk(eng))
    assert eng.loads == [] and eng.saves == []
    assert _digest(d) == before
    assert server._prefix_cache_state[str(d)]["loaded"] == 0


def test_load_only_starts_warm_and_never_modifies_the_preserved_cache(policy_for):
    d = policy_for("load-only", "never")
    _populate(d, 3)
    before = _digest(d)
    eng = StubEngine(entries=1)  # would overwrite with 1 entry if it saved
    _run(server._load_prefix_cache_from_disk(eng))
    _run(server._save_prefix_cache_to_disk(eng))
    assert server._prefix_cache_state[str(d)]["loaded"] == 3
    assert eng.saves == [] and _digest(d) == before


def test_save_only_starts_cold_even_with_a_cache_on_disk_and_saves_at_stop(policy_for):
    d = policy_for("save-only", "never")
    _populate(d, 3)
    eng = StubEngine(entries=1)
    _run(server._load_prefix_cache_from_disk(eng))
    assert eng.loads == [] and server._prefix_cache_state[str(d)]["loaded"] == 0
    _run(server._save_prefix_cache_to_disk(eng))
    assert eng.saves == [str(d)]


def test_reset_at_start_gives_a_cold_start_with_auto_and_then_saves(policy_for):
    d = policy_for("auto", "start")
    _populate(d, 3)
    eng = StubEngine(entries=1)
    _run(server._load_prefix_cache_from_disk(eng))
    st = server._prefix_cache_state[str(d)]
    assert st["reset_at_start"]["deleted_files"] == 7 and st["loaded"] == 0
    assert st["entries_on_disk"] == 0
    _run(server._save_prefix_cache_to_disk(eng))
    assert describe_dir(d)[0] == 1


def test_none_with_reset_both_leaves_nothing_behind(policy_for):
    d = policy_for("none", "both")
    _populate(d, 2)
    eng = StubEngine()
    _run(server._load_prefix_cache_from_disk(eng))
    assert not d.exists()
    _populate(d, 2)  # something appears while the server ran
    _run(server._save_prefix_cache_to_disk(eng))
    assert not d.exists() and eng.saves == [] and eng.loads == []


def test_registry_models_get_their_own_directories(policy_for):
    d = policy_for("auto", "never")
    server._model_path = None  # registry mode has no single model path
    a, b = StubEngine(1), StubEngine(2)
    _run(server._save_prefix_cache_to_disk(a, model_key="org/a"))
    _run(server._save_prefix_cache_to_disk(b, model_key="org/b"))
    base = d.parent
    assert describe_dir(base / "org--a")[0] == 1 and describe_dir(base / "org--b")[0] == 2
    assert not (base / "default").exists()


def test_registry_hooks_pass_the_models_name(policy_for):
    d = policy_for("auto", "never")
    server._model_path = None
    spec = types.SimpleNamespace(model_name="org/regmodel")
    eng = StubEngine(2)
    _run(server._persist_engine_state(spec, eng))
    assert (d.parent / "org--regmodel" / "index.json").exists()
    eng2 = StubEngine()
    _run(server._restore_engine_state(spec, eng2))
    assert eng2.loads == [str(d.parent / "org--regmodel")]


def test_an_unsafe_reset_at_start_fails_instead_of_silently_loading(policy_for, tmp_path):
    d = policy_for("auto", "start")
    real = tmp_path / "elsewhere"
    _populate(real, 2)
    d.parent.mkdir(parents=True, exist_ok=True)
    d.symlink_to(real, target_is_directory=True)
    eng = StubEngine()
    with pytest.raises(PersistenceError, match="symlink"):
        _run(server._load_prefix_cache_from_disk(eng))
    assert eng.loads == [] and (real / "index.json").exists()


def test_cache_stats_reports_the_policy_and_what_startup_did(policy_for):
    d = policy_for("load-only", "never")
    _populate(d, 2)
    eng = StubEngine()
    _run(server._load_prefix_cache_from_disk(eng))

    fake_utils = types.ModuleType("mlx_vlm.utils")
    fake_utils.get_multimodal_kv_cache_stats = lambda: {}
    fake_utils.get_pixel_values_cache_stats = lambda: {}
    fake_utils.get_pil_cache_stats = lambda: {}
    original_engine, original_key = server._engine, server._api_key
    original_module = sys.modules.get("mlx_vlm.utils")
    try:
        server._engine = None
        server._api_key = None
        sys.modules["mlx_vlm.utils"] = fake_utils
        body = TestClient(server.app).get("/v1/cache/stats").json()
    finally:
        server._engine, server._api_key = original_engine, original_key
        if original_module is not None:
            sys.modules["mlx_vlm.utils"] = original_module
        else:
            sys.modules.pop("mlx_vlm.utils", None)
    p = body["persistence"]
    assert p["policy"]["persist"] == "load-only" and p["policy"]["reset"] == "never"
    assert p["policy"]["base_dir"] == str(d.parent)
    assert p["dirs"][str(d)]["loaded"] == 2


# --- command line ------------------------------------------------------------------------


def test_serve_accepts_the_options_and_defaults_are_historical():
    parser = create_parser()
    a = parser.parse_args(["serve", "m"])
    assert (a.prefix_cache_dir, a.prefix_cache_persist, a.prefix_cache_reset) == (
        None, "auto", "never",
    )
    b = parser.parse_args([
        "serve", "m", "--prefix-cache-dir", "/x", "--prefix-cache-persist", "load-only",
        "--prefix-cache-reset", "never",
    ])
    assert (b.prefix_cache_dir, b.prefix_cache_persist) == ("/x", "load-only")
    with pytest.raises(SystemExit):
        parser.parse_args(["serve", "m", "--prefix-cache-persist", "sometimes"])


@pytest.mark.parametrize(
    "persist,reset", [("load-only", "start"), ("auto", "stop"), ("save-only", "both")]
)
def test_serve_refuses_a_contradictory_combination_before_loading_anything(
    persist, reset, capsys, monkeypatch
):
    from vllm_mlx import cli

    def boom(*a, **k):
        raise AssertionError("a model load was attempted")

    monkeypatch.setattr(server, "load_model", boom)
    monkeypatch.setattr(server, "load_model_registry", boom)
    args = create_parser().parse_args(
        ["serve", "m", "--prefix-cache-persist", persist, "--prefix-cache-reset", reset]
    )
    with pytest.raises(SystemExit) as exc:
        cli.serve_command(args)
    assert exc.value.code == 1
    assert "--prefix-cache" in capsys.readouterr().out


def test_the_worktree_package_is_the_one_under_test():
    assert Path(server.__file__).parent.parent == Path(__file__).resolve().parent.parent
