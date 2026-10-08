# SPDX-License-Identifier: Apache-2.0
"""Control over the persisted prefix cache: where it lives, whether it is read
or written, and when it is deleted.

The memory-aware prefix cache is saved to disk at shutdown and reloaded at
startup (``MemoryAwarePrefixCache.save_to_disk`` / ``load_from_disk``). By
default that is invisible state: a fixed path under ``$HOME`` and an
unconditional load and save, so the starting cache of a run is whatever the
previous run left behind. This module makes it explicit:

* ``--prefix-cache-dir`` / ``VLLM_MLX_PREFIX_CACHE_DIR``: the base directory
  (default ``$HOME/.cache/vllm-mlx/prefix_cache``). Each model keeps its own
  subdirectory under it, named as before, so existing caches keep working.
* ``--prefix-cache-persist``: ``auto`` (load at start, save at stop: the
  historical behavior), ``none`` (never read or write the persisted cache),
  ``load-only`` (start from the preserved cache and never modify it) or
  ``save-only`` (start cold, write at stop).
* ``--prefix-cache-reset``: delete this model's persisted entries ``start``
  (before the load), ``stop`` (after the save), ``both`` or ``never``.

Valid combinations (an unmarked cell is refused at startup)::

    persist \\ reset   never   start   stop   both
    auto              ok      ok      -      -      (stop would delete what was just saved)
    none              ok      ok      ok     ok
    load-only         ok      -       -      -      (a preserved cache is never modified)
    save-only         ok      ok      -      -

Deleting is deliberately narrow: only the files the persistence code writes
(``index.json``, ``entry_<i>.safetensors``, ``entry_<i>_tokens.bin``) inside
the resolved per-model directory, never through a symlink, never in ``/`` or
the home directory itself, and a directory with none of those files is left
alone. Anything else in the directory is reported and left in place.
"""

from __future__ import annotations

import os
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

ENV_DIR = "VLLM_MLX_PREFIX_CACHE_DIR"
PERSIST_MODES = ("auto", "none", "load-only", "save-only")
RESET_MODES = ("never", "start", "stop", "both")

INDEX_NAME = "index.json"
_ENTRY_RE = re.compile(r"^entry_(\d+)(\.safetensors|_tokens\.bin)$")


class PersistenceError(ValueError):
    """An invalid policy, or a reset that would not be safe to perform."""


def default_base_dir() -> str:
    return os.path.join(os.path.expanduser("~"), ".cache", "vllm-mlx", "prefix_cache")


def safe_model_name(model_name: object) -> str:
    """Directory name for a model (unchanged from the historical scheme)."""
    return str(model_name).replace("/", "--").replace("\\", "--")


def _is_recognized(name: str) -> bool:
    return name == INDEX_NAME or _ENTRY_RE.match(name) is not None


@dataclass(frozen=True)
class PersistencePolicy:
    base_dir: str | None = None
    persist: str = "auto"
    reset: str = "never"

    def __post_init__(self) -> None:
        if self.persist not in PERSIST_MODES:
            raise PersistenceError(
                f"--prefix-cache-persist must be one of {PERSIST_MODES}, "
                f"got {self.persist!r}"
            )
        if self.reset not in RESET_MODES:
            raise PersistenceError(
                f"--prefix-cache-reset must be one of {RESET_MODES}, got {self.reset!r}"
            )
        if self.base_dir is not None:
            base = os.path.expanduser(self.base_dir)
            if not os.path.isabs(base):
                raise PersistenceError(
                    f"--prefix-cache-dir must be an absolute path, got {self.base_dir!r}"
                )
            object.__setattr__(self, "base_dir", os.path.normpath(base))
        if self.persist == "load-only" and self.reset != "never":
            raise PersistenceError(
                "--prefix-cache-persist load-only never modifies the preserved cache, "
                f"so --prefix-cache-reset must be 'never' (got {self.reset!r})"
            )
        if self.persist in ("auto", "save-only") and self.reset in ("stop", "both"):
            raise PersistenceError(
                f"--prefix-cache-persist {self.persist} saves at stop, so "
                f"--prefix-cache-reset {self.reset} would delete what was just saved; "
                "use --prefix-cache-persist none to clean up without saving"
            )

    @classmethod
    def from_options(
        cls,
        base_dir: str | None = None,
        persist: str | None = None,
        reset: str | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> PersistencePolicy:
        env = os.environ if environ is None else environ
        chosen = base_dir or env.get(ENV_DIR) or None
        return cls(
            base_dir=chosen.strip() if chosen else None,
            persist=persist or "auto",
            reset=reset or "never",
        )

    @property
    def loads(self) -> bool:
        return self.persist in ("auto", "load-only")

    @property
    def saves(self) -> bool:
        return self.persist in ("auto", "save-only")

    @property
    def resets_at_start(self) -> bool:
        return self.reset in ("start", "both")

    @property
    def resets_at_stop(self) -> bool:
        return self.reset in ("stop", "both")

    def resolve_dir(self, model_name: object) -> str:
        return os.path.join(self.base_dir or default_base_dir(), safe_model_name(model_name))

    def as_dict(self) -> dict[str, object]:
        return {
            "base_dir": self.base_dir or default_base_dir(),
            "base_dir_source": "option/env" if self.base_dir else "default",
            "persist": self.persist,
            "reset": self.reset,
        }


@dataclass
class ResetResult:
    deleted_files: int = 0
    deleted_bytes: int = 0
    left_alone: list[str] = field(default_factory=list)


def describe_dir(path: str | os.PathLike[str]) -> tuple[int, int]:
    """(number of persisted entries, bytes of persisted files) in ``path``."""
    entries = 0
    total = 0
    try:
        with os.scandir(path) as it:
            for de in it:
                if not _is_recognized(de.name) or not de.is_file(follow_symlinks=False):
                    continue
                total += de.stat(follow_symlinks=False).st_size
                m = _ENTRY_RE.match(de.name)
                if m and m.group(2) == ".safetensors":
                    entries += 1
    except (FileNotFoundError, NotADirectoryError):
        return 0, 0
    return entries, total


def reset_cache_dir(path: str | os.PathLike[str]) -> ResetResult:
    """Delete the persisted entries in one model's cache directory.

    Refuses (``PersistenceError``) an unsafe target; returns what was deleted
    and what was left alone. A missing directory, or one with none of the
    persistence files, is left untouched.
    """
    p = Path(os.path.expanduser(os.fspath(path)))
    if not p.is_absolute():
        raise PersistenceError(f"refusing to reset a relative path: {os.fspath(path)!r}")
    home = Path(os.path.expanduser("~"))
    if p == Path(p.anchor) or p == home or p in home.parents:
        raise PersistenceError(f"refusing to reset {p}: not a cache directory")
    try:
        st = os.lstat(p)
    except FileNotFoundError:
        return ResetResult()
    if stat.S_ISLNK(st.st_mode):
        raise PersistenceError(f"refusing to reset {p}: it is a symlink")
    if not stat.S_ISDIR(st.st_mode):
        raise PersistenceError(f"refusing to reset {p}: not a directory")

    result = ResetResult()
    with os.scandir(p) as it:
        children = sorted(it, key=lambda e: e.name)
    recognized = [c for c in children if _is_recognized(c.name)]
    if not recognized:
        result.left_alone = [c.name for c in children]
        return result
    for c in children:
        if not _is_recognized(c.name):
            result.left_alone.append(c.name)
            continue
        if not c.is_file(follow_symlinks=False):
            result.left_alone.append(c.name)  # a symlink or directory by that name
            continue
        size = c.stat(follow_symlinks=False).st_size
        os.unlink(c.path)
        result.deleted_files += 1
        result.deleted_bytes += size
    try:
        os.rmdir(p)  # only succeeds when nothing else is in it
    except OSError:
        pass
    return result
