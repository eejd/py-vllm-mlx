# SPDX-License-Identifier: Apache-2.0
"""
SSD KV cache tiering for vllm-mlx.

This module provides a cold-tier disk cache that sits behind
MemoryAwarePrefixCache. Evicted entries spill to NVMe instead of being
discarded, and cold-tier fetches reload from disk asynchronously with
RAM budget reservation before the read completes.

Key design:
- SQLite for atomic metadata index (no mutable JSON)
- Async writer thread for non-blocking spills
- Per-layer serializer interface for hybrid cache types
- Atomic temp-file + rename writes for crash consistency
- Metrics exposed from day one
"""

from __future__ import annotations

import array as _array
import hashlib
import json
import logging
import os
import queue
import re
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from .cache_identity import CacheIdentity
from .mlx_cache_compat import recurrent_arrays

try:  # advisory locks mark a namespace as in use; absent on Windows
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None

logger = logging.getLogger(__name__)

_BYTES_PER_MB = 1024 * 1024
_BYTES_PER_GB = 1024 * 1024 * 1024
_PREFIX_FILTER_TOKENS = 16
_TOKEN_BYTES = _array.array("i").itemsize
_LEGACY_ENTRY_RE = re.compile(r"^[0-9a-f]{64}(\.tmp)?$")
_DIGEST_DIR_RE = re.compile(r"^[0-9a-f]{32}$")
# A namespace being deleted is renamed to this first (see _reclaim_stale_namespaces).
_TOMBSTONE_RE = re.compile(r"^[0-9a-f]{32}\.reclaim-[0-9]+-[0-9]+$")


@dataclass(frozen=True)
class SSDCacheConfig:
    """Configuration for SSD cache tier.

    Attributes:
        cache_dir: Base directory for SSD cache files. The tier keeps its files in
            ``cache_dir/<identity digest>/``, so servers, models and
            configurations that share one base directory never read each
            other's entries.
        identity: What the entries are valid for (:class:`CacheIdentity` of the
            prefix cache the tier backs: model weights, config, mlx/mlx-lm
            versions, KV quantization, max-kv-size). Required.
        max_size_gb: Maximum total size of SSD cache in GB (per identity).
        max_entries: Maximum number of entries in SSD cache.
        file_permissions: Unix permission bits for cache data files.
        dir_permissions: Unix permission bits for cache directories.
        spill_queue_size: Max pending spill operations before dropping.
        retention_seconds: Optional max age for cache entries (None = no expiry).
        stale_namespace_days: Sibling identity directories (other models,
            revisions, library versions) that nothing has used for this long
            (``identity.json``, ``index.db`` and its WAL all older) and that no
            running tier holds (an advisory lock) are deleted when a tier
            starts (None = never). On a filesystem without advisory locks no
            directory can be shown to be idle, so none is reclaimed.
    """

    cache_dir: str | None = None
    identity: CacheIdentity | None = None
    max_size_gb: float = 10.0
    max_entries: int = 10000
    file_permissions: int = 0o600
    dir_permissions: int = 0o700
    spill_queue_size: int = 64
    retention_seconds: int | None = None
    stale_namespace_days: float | None = 30.0

    def __post_init__(self) -> None:
        if self.max_size_gb <= 0:
            raise ValueError(f"max_size_gb must be > 0, got {self.max_size_gb}")
        if self.max_entries < 1:
            raise ValueError(f"max_entries must be >= 1, got {self.max_entries}")
        if self.spill_queue_size < 1:
            raise ValueError(
                f"spill_queue_size must be >= 1, got {self.spill_queue_size}"
            )

    @property
    def max_size_bytes(self) -> int:
        """Maximum cache size in bytes."""
        return int(self.max_size_gb * _BYTES_PER_GB)


@dataclass
class SSDCacheStats:
    """Statistics for SSD cache tier — exposed from day one.

    Attributes:
        spill_count: Number of entries spilled to SSD.
        spill_bytes: Total bytes written to SSD.
        ssd_hits: Number of successful SSD cache lookups.
        ssd_misses: Number of SSD cache lookup misses.
        reload_latency_sum: Sum of reload latencies in seconds.
        reload_bytes: Total bytes read from SSD.
        promotion_failures: Number of failed promotions (no RAM budget, read error).
        promotions_rejected_oversized: Promotions refused because the entry is
            larger than the whole in-memory cache budget (it could never fit).
    """

    spill_count: int = 0
    spill_bytes: int = 0
    ssd_hits: int = 0
    ssd_misses: int = 0
    reload_latency_sum: float = 0.0
    reload_bytes: int = 0
    promotion_failures: int = 0
    promotions_rejected_oversized: int = 0

    def to_dict(self) -> dict:
        total_lookups = self.ssd_hits + self.ssd_misses
        hit_rate = self.ssd_hits / total_lookups if total_lookups > 0 else 0.0
        avg_latency_ms = (
            (self.reload_latency_sum / self.ssd_hits * 1000)
            if self.ssd_hits > 0
            else 0.0
        )
        return {
            "spill_count": self.spill_count,
            "spill_bytes": self.spill_bytes,
            "ssd_hits": self.ssd_hits,
            "ssd_misses": self.ssd_misses,
            "ssd_hit_rate": round(hit_rate, 4),
            "reload_latency_sum_s": round(self.reload_latency_sum, 4),
            "avg_reload_latency_ms": round(avg_latency_ms, 2),
            "reload_bytes": self.reload_bytes,
            "promotion_failures": self.promotion_failures,
            "promotions_rejected_oversized": self.promotions_rejected_oversized,
        }


def _tokens_to_blob(tokens: tuple[int, ...]) -> bytes:
    """Serialize token tuple to a compact binary blob for SQLite storage.

    Uses the full token sequence as a binary blob for prefix matching.
    """
    arr = _array.array("i", tokens)
    return arr.tobytes()


def _blob_to_tokens(blob: bytes) -> tuple[int, ...]:
    """Deserialize binary blob back to token tuple."""
    arr = _array.array("i")
    arr.frombytes(blob)
    return tuple(arr)


def _tokens_hash(tokens: tuple[int, ...]) -> str:
    """Compute SHA-256 hex digest of a token sequence for use as primary key."""
    return hashlib.sha256(_tokens_to_blob(tokens)).hexdigest()


def _prefix_hash(tokens: tuple[int, ...]) -> str:
    """Hash the bounded token prefix used to prefilter prefix lookups."""
    return _tokens_hash(tokens[:_PREFIX_FILTER_TOKENS])


class SSDIndex:
    """SQLite-backed index for SSD cache entries.

    Uses SQLite for atomic metadata operations instead of mutable JSON.
    The token sequence is stored as a binary blob for prefix-searchable
    representation. The primary key is a SHA-256 hash of the token sequence.

    Thread safety: All operations are serialized through a threading.Lock.
    The SQLite connection uses WAL mode for concurrent read/write safety.
    """

    # 2: CacheList (GLM-5.2/5.3 DSA) layers gained a dedicated serializer.
    #    Before it, they were mis-dispatched to ArraysCacheSerializer; with
    #    fp16 that silently succeeded, writing an entry tagged
    #    layer_type="ArraysCache" with the per-sub offsets lost. Those entries
    #    still match on token hash after an upgrade, so they must be dropped.
    _SCHEMA_VERSION = 2

    def __init__(self, cache_dir: str) -> None:
        self._cache_dir = cache_dir
        self._db_lock = threading.Lock()
        # Set to the previous version when an incompatible index was purged,
        # so the tier knows to clear the matching data directories.
        self.migrated_from: int | None = None
        db_path = os.path.join(cache_dir, "index.db")
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        schema_sql = """
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS entries (
                token_hash   TEXT PRIMARY KEY,
                tokens_blob  BLOB NOT NULL,
                prefix_hash  TEXT,
                num_tokens   INTEGER NOT NULL,
                file_path    TEXT NOT NULL,
                memory_bytes INTEGER NOT NULL,
                created_at   REAL NOT NULL,
                accessed_at  REAL NOT NULL
            );

            """
        self._conn.executescript(schema_sql)
        self._ensure_column("entries", "prefix_hash", "TEXT")
        # 1 = every layer of the entry is a plain KV layer, so the entry can be
        # cut back to a shorter shared prefix; NULL (entries written by an
        # older build) and 0 = it cannot, or is not known to.
        self._ensure_column("entries", "trimmable", "INTEGER")
        self._conn.executescript("""
            CREATE INDEX IF NOT EXISTS idx_entries_accessed
                ON entries(accessed_at);

            CREATE INDEX IF NOT EXISTS idx_entries_num_tokens
                ON entries(num_tokens);

            CREATE INDEX IF NOT EXISTS idx_entries_prefix_hash_num_tokens
                ON entries(prefix_hash, num_tokens);
            """)
        # Insert schema version if not present, or drop an index written by an
        # older, incompatible build. Entries are keyed on the token hash alone
        # and the cache dir is not build-namespaced, so a stale entry would
        # otherwise be served to a reader that cannot interpret it.
        cur = self._conn.execute("SELECT version FROM schema_version")
        row = cur.fetchone()
        if row is None:
            self._conn.execute(
                "INSERT INTO schema_version (version) VALUES (?)",
                (self._SCHEMA_VERSION,),
            )
        elif row[0] != self._SCHEMA_VERSION:
            self.migrated_from = row[0]
            self._conn.execute("DELETE FROM entries")
            self._conn.execute(
                "UPDATE schema_version SET version = ?", (self._SCHEMA_VERSION,)
            )
            logger.info(
                "[ssd_cache] index schema %s -> %s: dropped all entries",
                row[0],
                self._SCHEMA_VERSION,
            )
        self._backfill_prefix_hashes()
        self._conn.commit()

    def _ensure_column(self, table: str, column: str, definition: str) -> None:
        cur = self._conn.execute(f"PRAGMA table_info({table})")
        if column not in {row["name"] for row in cur.fetchall()}:
            self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def _backfill_prefix_hashes(self) -> None:
        cur = self._conn.execute(
            "SELECT token_hash, tokens_blob FROM entries WHERE prefix_hash IS NULL"
        )
        rows = cur.fetchall()
        for row in rows:
            tokens = _blob_to_tokens(row["tokens_blob"])
            self._conn.execute(
                "UPDATE entries SET prefix_hash = ? WHERE token_hash = ?",
                (_prefix_hash(tokens), row["token_hash"]),
            )

    def insert_entry(
        self,
        tokens_key: tuple[int, ...],
        file_path: str,
        memory_bytes: int,
        num_tokens: int,
        trimmable: bool = False,
    ) -> None:
        """Insert or replace a cache entry in the index."""
        now = time.time()
        token_hash = _tokens_hash(tokens_key)
        prefix_hash = _prefix_hash(tokens_key)
        tokens_blob = _tokens_to_blob(tokens_key)
        with self._db_lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO entries
                    (token_hash, tokens_blob, prefix_hash, num_tokens, file_path,
                     memory_bytes, created_at, accessed_at, trimmable)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    token_hash,
                    tokens_blob,
                    prefix_hash,
                    num_tokens,
                    file_path,
                    memory_bytes,
                    now,
                    now,
                    1 if trimmable else 0,
                ),
            )
            self._conn.commit()

    def lookup_exact(self, tokens_key: tuple[int, ...]) -> dict | None:
        """Look up an exact token sequence. Returns dict or None."""
        token_hash = _tokens_hash(tokens_key)
        with self._db_lock:
            cur = self._conn.execute(
                "SELECT file_path, memory_bytes, num_tokens FROM entries WHERE token_hash = ?",
                (token_hash,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return {
            "file_path": row["file_path"],
            "memory_bytes": row["memory_bytes"],
            "num_tokens": row["num_tokens"],
        }

    def lookup_prefix(self, query_tokens: tuple[int, ...]) -> list[dict]:
        """Find entries whose token sequence is a prefix of query_tokens.

        Uses a bounded token-prefix hash to avoid scanning all entries, then
        compares the full stored token blob against the corresponding prefix of
        query_tokens.

        Returns list of dicts sorted by num_tokens descending (longest prefix first).
        """
        query_len = len(query_tokens)
        query_blob = _tokens_to_blob(query_tokens)
        prefix_hashes = {
            _tokens_hash(query_tokens[:n])
            for n in range(1, min(query_len, _PREFIX_FILTER_TOKENS) + 1)
        }
        if not prefix_hashes:
            return []

        with self._db_lock:
            placeholders = ",".join("?" for _ in prefix_hashes)
            cur = self._conn.execute(
                "SELECT token_hash, tokens_blob, num_tokens, file_path, memory_bytes "
                f"FROM entries WHERE num_tokens <= ? AND prefix_hash IN ({placeholders}) "
                "ORDER BY num_tokens DESC",
                (query_len, *prefix_hashes),
            )
            rows = cur.fetchall()

        results = []
        for row in rows:
            stored_blob = row["tokens_blob"]
            n = row["num_tokens"]
            prefix_blob = query_blob[: n * 4]
            if stored_blob == prefix_blob:
                results.append(
                    {
                        "token_hash": row["token_hash"],
                        "file_path": row["file_path"],
                        "memory_bytes": row["memory_bytes"],
                        "num_tokens": n,
                    }
                )
        return results

    def lookup_lcp(self, query_tokens: tuple[int, ...], min_tokens: int) -> dict | None:
        """Find the entry sharing the longest token prefix with the query.

        Serves what ``lookup_prefix`` cannot: a stored key that is *longer* than
        the query (an exact repeat of a prompt whose entry also holds the reply
        it generated) or that diverges from it after a shared head. Only
        entries written as trimmable (plain KV layers) qualify, since the
        caller cuts the entry back to the shared prefix.

        The shared prefix must be at least ``min_tokens`` long (and at least
        ``_PREFIX_FILTER_TOKENS``, the head the index hashes). Returns the
        entry's metadata plus ``lcp`` (the shared length) and ``tokens`` (its
        full key); the longest shared prefix wins, the shortest entry on a tie.
        """
        floor = max(int(min_tokens), _PREFIX_FILTER_TOKENS)
        if len(query_tokens) < floor:
            return None
        query_blob = _tokens_to_blob(query_tokens)
        with self._db_lock:
            cur = self._conn.execute(
                "SELECT token_hash, tokens_blob, num_tokens, file_path, memory_bytes "
                "FROM entries WHERE trimmable = 1 AND num_tokens > ? "
                "AND prefix_hash = ? AND substr(tokens_blob, 1, ?) = ?",
                (
                    floor,
                    _prefix_hash(query_tokens),
                    floor * _TOKEN_BYTES,
                    query_blob[: floor * _TOKEN_BYTES],
                ),
            )
            rows = cur.fetchall()

        query = np.frombuffer(query_blob, dtype=np.intc)
        best: dict | None = None
        for row in rows:
            stored = np.frombuffer(row["tokens_blob"], dtype=np.intc)
            n = min(len(stored), len(query))
            differs = np.flatnonzero(stored[:n] != query[:n])
            lcp = int(differs[0]) if len(differs) else n
            # A stored key that is a prefix of the query needs no trimming and
            # belongs to lookup_prefix; one that stops short of the floor is
            # too little to reuse.
            if lcp < floor or lcp >= row["num_tokens"]:
                continue
            if (
                best is None
                or lcp > best["lcp"]
                or (lcp == best["lcp"] and row["num_tokens"] < best["num_tokens"])
            ):
                best = {
                    "token_hash": row["token_hash"],
                    "file_path": row["file_path"],
                    "memory_bytes": row["memory_bytes"],
                    "num_tokens": row["num_tokens"],
                    "lcp": lcp,
                    "tokens": _blob_to_tokens(row["tokens_blob"]),
                }
        return best

    def delete_entry(self, tokens_key: tuple[int, ...]) -> None:
        """Delete an entry by token sequence."""
        token_hash = _tokens_hash(tokens_key)
        with self._db_lock:
            self._conn.execute(
                "DELETE FROM entries WHERE token_hash = ?", (token_hash,)
            )
            self._conn.commit()

    def delete_all(self) -> None:
        """Drop every entry (the data directories are the caller's to remove)."""
        with self._db_lock:
            self._conn.execute("DELETE FROM entries")
            self._conn.commit()

    def get_lru(self, limit: int = 10) -> list[dict]:
        """Get the least recently used entries, ordered oldest first."""
        with self._db_lock:
            cur = self._conn.execute(
                "SELECT token_hash, tokens_blob, num_tokens, file_path, memory_bytes "
                "FROM entries ORDER BY accessed_at ASC LIMIT ?",
                (limit,),
            )
            rows = cur.fetchall()
        results = []
        for row in rows:
            results.append(
                {
                    "token_hash": row["token_hash"],
                    "tokens_blob": row["tokens_blob"],
                    "file_path": row["file_path"],
                    "memory_bytes": row["memory_bytes"],
                    "num_tokens": row["num_tokens"],
                }
            )
        return results

    def get_total_bytes(self) -> int:
        """Get total memory_bytes across all entries."""
        with self._db_lock:
            cur = self._conn.execute(
                "SELECT COALESCE(SUM(memory_bytes), 0) FROM entries"
            )
            return cur.fetchone()[0]

    def get_entry_count(self) -> int:
        """Get number of entries in the index."""
        with self._db_lock:
            cur = self._conn.execute("SELECT COUNT(*) FROM entries")
            return cur.fetchone()[0]

    def touch(self, tokens_key: tuple[int, ...]) -> None:
        """Update accessed_at timestamp for an entry (marks as recently used)."""
        token_hash = _tokens_hash(tokens_key)
        with self._db_lock:
            self._conn.execute(
                "UPDATE entries SET accessed_at = ? WHERE token_hash = ?",
                (time.time(), token_hash),
            )
            self._conn.commit()

    def all_entries(self) -> list[dict]:
        """Return all entries (for startup reconciliation)."""
        with self._db_lock:
            cur = self._conn.execute(
                "SELECT token_hash, tokens_blob, num_tokens, file_path, memory_bytes "
                "FROM entries ORDER BY accessed_at DESC"
            )
            rows = cur.fetchall()
        results = []
        for row in rows:
            results.append(
                {
                    "token_hash": row["token_hash"],
                    "tokens_blob": row["tokens_blob"],
                    "file_path": row["file_path"],
                    "memory_bytes": row["memory_bytes"],
                    "num_tokens": row["num_tokens"],
                }
            )
        return results

    def close(self) -> None:
        """Close the SQLite connection."""
        with self._db_lock:
            self._conn.close()


# Support matrix: maps cache type names to their serializer status
SERIALIZER_SUPPORT_MATRIX = {
    "KVCache": "supported",
    "RotatingKVCache": "supported",  # Serialized as KVCache (keys/values/offset)
    "ArraysCache": "supported",
    "MambaCache": "supported",  # Legacy name for ArraysCache
    "CacheList": "supported",  # DSA (GLM-5.2/DeepSeek-V3.2): list of KVCache subs
    "_QuantizedCacheWrapper": "supported_via_dequant_on_spill",
    "QuantizedKVCache": "supported_via_dequant_on_spill",
}


class LayerSerializer(ABC):
    """Interface for per-layer cache serialization.

    Spill is split across two threads: ``snapshot_layer`` runs on the
    producer (request handler) thread so the mx→numpy materialization
    happens where the per-request Stream(gpu, N) is registered;
    ``serialize_layer`` then runs on the SSD writer thread with numpy only.
    """

    @abstractmethod
    def snapshot_layer(self, layer: Any) -> dict[str, Any]:
        """Producer-thread CPU snapshot of an MLX-backed cache layer."""
        ...

    @abstractmethod
    def serialize_layer(
        self, snapshot: dict[str, Any], layer_idx: int, file_path: str
    ) -> dict[str, Any]:
        """Writer-thread: persist a snapshot to safetensors at file_path.

        Returns metadata dict with at least 'layer_type'.
        """
        ...

    @abstractmethod
    def deserialize_layer(self, file_path: str, metadata: dict[str, Any]) -> dict:
        """Read a layer back from disk. Returns layer-state dict."""
        ...


def _mx_to_numpy_safe(arr: Any) -> tuple[np.ndarray, str | None]:
    """mx.array → np.ndarray, upcasting numpy-unsupported dtypes (bf16) to fp32.

    Returns (numpy_array, original_dtype_name_or_None). The name is only set
    when an upcast happened, so the SSD-promote path can cast back.
    """
    try:
        return np.array(arr), None
    except (RuntimeError, ValueError) as exc:
        # numpy ↔ mlx bf16 buffer-protocol mismatch. The exception type is not
        # stable across mlx versions — older builds raise RuntimeError, mlx
        # 0.31.x raises ValueError ("'bfloat16' is not a valid PEP 3118 buffer
        # format string") — so match on the message, not the class. Re-raise
        # anything else — don't swallow unrelated errors.
        if "buffer format string" not in str(exc):
            raise
        import mlx.core as mx

        original_dtype = str(arr.dtype).rsplit(".", 1)[-1]
        upcast = arr.astype(mx.float32)
        mx.eval(upcast)  # astype is lazy; force materialization here
        return np.array(upcast), original_dtype


class KVCacheSerializer(LayerSerializer):
    """Serializer for KVCache and RotatingKVCache layers.

    Handles layers with .keys, .values, .offset attributes.
    RotatingKVCache also has .max_size, .keep, .step, ._idx.
    """

    # Extra attributes carried by RotatingKVCache (but not vanilla KVCache).
    # Stored in metadata so deserialize can faithfully reconstruct either type.
    _ROTATING_ATTRS = ("max_size", "keep", "step", "_idx")

    def snapshot_layer(self, layer: Any) -> dict[str, Any]:
        # A KVCache leaves keys/values None until its first update_and_fetch.
        # np.array(None) is a 0-d *object* array whose .size is 1, so it slips
        # past serialize_layer's empty-array branch and only fails inside
        # save_file ("dtype object is not covered") -- on the writer thread,
        # where _writer_loop swallows the error and orphans the .tmp dir. Fail
        # here instead: the same entry is dropped, but loudly and locally.
        # NOTE: a size-0 array is NOT None -- the dense-mode DSA indexer is
        # empty but materialized, and must still round-trip.
        if (
            getattr(layer, "keys", None) is None
            or getattr(layer, "values", None) is None
        ):
            raise ValueError(
                f"{type(layer).__name__} keys/values are not materialized "
                "(None); layer cannot be spilled to SSD"
            )
        keys_np, keys_orig_dtype = _mx_to_numpy_safe(layer.keys)
        values_np, values_orig_dtype = _mx_to_numpy_safe(layer.values)

        # Quantized-spill cast-back sentinel: the enqueue_spill dequant path
        # may have cast bf16 → fp16 before snapshot to dodge numpy's PEP 3118
        # buffer-protocol mismatch. The cast loses the original-dtype signal
        # _mx_to_numpy_safe would otherwise capture (since fp16 IS numpy-
        # supported, _mx_to_numpy_safe returns dtype=None). Honor an explicit
        # sentinel on the layer when present so the reload path can restore
        # bf16 instead of leaving the model with fp16 KV.
        keys_orig_dtype = (
            getattr(layer, "_ssd_keys_original_dtype", None) or keys_orig_dtype
        )
        values_orig_dtype = (
            getattr(layer, "_ssd_values_original_dtype", None) or values_orig_dtype
        )

        snapshot: dict[str, Any] = {
            "keys_np": keys_np,
            "values_np": values_np,
            "offset": layer.offset,
        }
        if keys_orig_dtype is not None:
            snapshot["keys_original_dtype"] = keys_orig_dtype
        if values_orig_dtype is not None:
            snapshot["values_original_dtype"] = values_orig_dtype

        # RotatingKVCache extras (plain Python scalars, no MLX).
        for attr in self._ROTATING_ATTRS:
            if hasattr(layer, attr):
                snapshot[attr] = getattr(layer, attr)
        return snapshot

    def serialize_layer(
        self, snapshot: dict[str, Any], layer_idx: int, file_path: str
    ) -> dict[str, Any]:
        from safetensors.numpy import save_file

        tensors = {
            f"layer_{layer_idx}_keys": snapshot["keys_np"],
            f"layer_{layer_idx}_values": snapshot["values_np"],
        }
        save_file(tensors, file_path)

        metadata = {
            "layer_type": "KVCache",
            "layer_idx": layer_idx,
            "offset": snapshot["offset"],
        }
        for k in ("keys_original_dtype", "values_original_dtype"):
            if k in snapshot:
                metadata[k] = snapshot[k]
        for attr in self._ROTATING_ATTRS:
            if attr in snapshot:
                metadata[attr] = snapshot[attr]

        return metadata

    def deserialize_layer(self, file_path: str, metadata: dict[str, Any]) -> dict:
        from safetensors.numpy import load_file

        layer_idx = metadata["layer_idx"]
        tensors = load_file(file_path)

        result = {
            "keys": tensors[f"layer_{layer_idx}_keys"],
            "values": tensors[f"layer_{layer_idx}_values"],
            "offset": metadata["offset"],
        }
        # Dtype hints surfaced so _reconstruct_ssd_layers can cast back.
        for k in ("keys_original_dtype", "values_original_dtype"):
            if k in metadata:
                result[k] = metadata[k]
        for attr in self._ROTATING_ATTRS:
            if attr in metadata:
                result[attr] = metadata[attr]
        return result


class ArraysCacheSerializer(LayerSerializer):
    """Serializer for ArraysCache (Mamba/linear attention) layers.

    Handles layers with .state attribute containing a list of arrays.
    """

    def snapshot_layer(self, layer: Any) -> dict[str, Any]:
        state_np: list[np.ndarray] = []
        original_dtypes: list[str | None] = []
        # ``layer.cache`` is the array list on every mlx-lm contract; in 0.32
        # ``.state`` is ``(cache, left_padding, lengths)`` and would not iterate
        # as arrays.
        for arr in recurrent_arrays(layer):
            np_arr, orig = _mx_to_numpy_safe(arr)
            state_np.append(np_arr)
            original_dtypes.append(orig)

        snapshot: dict[str, Any] = {"state_np": state_np}
        # Skip the dtype list in the common (fp16/fp32) case.
        if any(d is not None for d in original_dtypes):
            snapshot["state_original_dtypes"] = original_dtypes
        return snapshot

    def serialize_layer(
        self, snapshot: dict[str, Any], layer_idx: int, file_path: str
    ) -> dict[str, Any]:
        # Writer-thread side: pure numpy + disk.
        from safetensors.numpy import save_file

        state_np = snapshot["state_np"]
        tensors = {
            f"layer_{layer_idx}_state_{i}": arr for i, arr in enumerate(state_np)
        }
        save_file(tensors, file_path)

        metadata = {
            "layer_type": "ArraysCache",
            "layer_idx": layer_idx,
            "num_arrays": len(state_np),
        }
        if "state_original_dtypes" in snapshot:
            metadata["state_original_dtypes"] = snapshot["state_original_dtypes"]
        return metadata

    def deserialize_layer(self, file_path: str, metadata: dict[str, Any]) -> dict:
        from safetensors.numpy import load_file

        layer_idx = metadata["layer_idx"]
        num_arrays = metadata["num_arrays"]
        tensors = load_file(file_path)

        state = []
        for i in range(num_arrays):
            state.append(tensors[f"layer_{layer_idx}_state_{i}"])
        result = {"state": state}
        if "state_original_dtypes" in metadata:
            result["state_original_dtypes"] = metadata["state_original_dtypes"]
        return result


class CacheListSerializer(LayerSerializer):
    """Serializer for DSA ``CacheList`` layers (GLM-5.2 / DeepSeek-V3.2).

    A ``CacheList`` wraps N sub-caches — for DSA each is a ``KVCache``:
    ``(latent, indexer)`` for full layers, ``(latent,)`` for shared layers.
    Below the sparse ``index_topk`` threshold the indexer sub-cache is empty
    (size-0 arrays).

    ``CacheList`` also exposes a list ``.state``, so the duck-typed dispatcher
    used to mis-route it to ``ArraysCacheSerializer``. Depending on the mlx_lm
    build that either crashed (``.state`` yields per-sub ``(keys, values)``
    tuples that ``_mx_to_numpy_safe`` cannot handle) or silently flattened the
    sub-caches and lost the CacheList structure. This serializer instead walks
    ``layer.caches`` directly — independent of ``.state`` semantics — snapshots
    each sub ``KVCache`` (reusing ``KVCacheSerializer``), flattens their arrays
    into one safetensors file under ``sub_{j}_*`` keys, and records the per-sub
    layout (offset, dtype hints, empty-array shapes) in metadata.
    """

    def snapshot_layer(self, layer: Any) -> dict[str, Any]:
        sub_ser = KVCacheSerializer()
        sub_snapshots = []
        for sub in layer.caches:
            if not (
                hasattr(sub, "keys")
                and hasattr(sub, "values")
                and hasattr(sub, "offset")
            ):
                raise ValueError(
                    f"CacheList sub-cache {type(sub).__name__} is not "
                    "KVCache-like; unsupported for SSD spill"
                )
            sub_snapshots.append(sub_ser.snapshot_layer(sub))
        return {"sub_snapshots": sub_snapshots}

    def serialize_layer(
        self, snapshot: dict[str, Any], layer_idx: int, file_path: str
    ) -> dict[str, Any]:
        from safetensors.numpy import save_file

        sub_snapshots = snapshot["sub_snapshots"]
        tensors: dict[str, np.ndarray] = {}
        sub_meta: list[dict[str, Any]] = []
        for j, sub in enumerate(sub_snapshots):
            keys_np = sub["keys_np"]
            values_np = sub["values_np"]
            m: dict[str, Any] = {"offset": sub["offset"]}
            # Size-0 arrays (empty DSA indexer) can't go through safetensors;
            # record shape+dtype and recreate on load, like the disk-persist path.
            if getattr(keys_np, "size", 1) == 0:
                m["keys_empty"] = [list(keys_np.shape), str(keys_np.dtype)]
            else:
                tensors[f"sub_{j}_keys"] = keys_np
            if getattr(values_np, "size", 1) == 0:
                m["values_empty"] = [list(values_np.shape), str(values_np.dtype)]
            else:
                tensors[f"sub_{j}_values"] = values_np
            for k in ("keys_original_dtype", "values_original_dtype"):
                if k in sub:
                    m[k] = sub[k]
            for attr in KVCacheSerializer._ROTATING_ATTRS:
                if attr in sub:
                    m[attr] = sub[attr]
            sub_meta.append(m)

        if not tensors:
            # safetensors rejects an empty tensor dict; write a sentinel. (The
            # latent sub-cache is always populated, so this is a guard only.)
            tensors["__placeholder__"] = np.zeros((1,), dtype=np.uint8)
        save_file(tensors, file_path)

        return {
            "layer_type": "CacheList",
            "layer_idx": layer_idx,
            "sub_count": len(sub_snapshots),
            "sub_meta": sub_meta,
        }

    def deserialize_layer(self, file_path: str, metadata: dict[str, Any]) -> dict:
        from safetensors.numpy import load_file

        sub_count = metadata["sub_count"]
        sub_meta = metadata["sub_meta"]
        tensors = load_file(file_path)

        subs = []
        for j in range(sub_count):
            m = sub_meta[j]
            kkey, vkey = f"sub_{j}_keys", f"sub_{j}_values"
            if kkey in tensors:
                keys = tensors[kkey]
            else:
                shape, dt = m["keys_empty"]
                keys = np.zeros(shape, dtype=np.dtype(dt))
            if vkey in tensors:
                values = tensors[vkey]
            else:
                shape, dt = m["values_empty"]
                values = np.zeros(shape, dtype=np.dtype(dt))

            sub_ld: dict[str, Any] = {
                "keys": keys,
                "values": values,
                "offset": m["offset"],
            }
            for k in ("keys_original_dtype", "values_original_dtype"):
                if k in m:
                    sub_ld[k] = m[k]
            for attr in KVCacheSerializer._ROTATING_ATTRS:
                if attr in m:
                    sub_ld[attr] = m[attr]
            subs.append(sub_ld)
        return {"cachelist_subs": subs}


def get_serializer_for_layer(layer: Any) -> LayerSerializer:
    """Return the appropriate serializer for a cache layer.

    Dispatches based on duck-typing:
    - If layer has .caches (DSA CacheList) -> CacheListSerializer
    - If layer has .keys and .values and .offset -> KVCacheSerializer
    - If layer has .state and it's a list -> ArraysCacheSerializer

    The CacheList check comes first because a CacheList ALSO has a list
    ``.state`` and would otherwise be mis-routed to ArraysCacheSerializer.

    Raises ValueError for unsupported layer types.
    """
    if isinstance(getattr(layer, "caches", None), (list, tuple)):
        return CacheListSerializer()
    if hasattr(layer, "keys") and hasattr(layer, "values") and hasattr(layer, "offset"):
        return KVCacheSerializer()
    if recurrent_arrays(layer) is not None:
        return ArraysCacheSerializer()
    raise ValueError(
        f"Unsupported cache layer type: {type(layer).__name__}. "
        f"Supported: {list(SERIALIZER_SUPPORT_MATRIX.keys())}"
    )


def _kv_dict_error(layer: dict, n: int, nested: bool = False) -> str | None:
    """Why a deserialized KV layer is not ``n`` tokens of KV, or None if it is."""
    offset = layer["offset"]
    if not isinstance(offset, int) or isinstance(offset, bool):
        return f"offset {offset!r} is not an integer"
    if offset != n and not (nested and offset == 0):
        return f"offset {offset} != {n} tokens"
    if offset == 0:
        return None
    keys = layer["keys"]
    if keys.ndim < 3:
        return f"keys of shape {tuple(keys.shape)} have no sequence axis"
    max_size = layer.get("max_size")
    # A rotating layer holds at most its window, whatever its offset.
    needed = (
        min(offset, max_size) if isinstance(max_size, int) and max_size > 0 else offset
    )
    if keys.shape[-2] < needed:
        return f"keys hold {keys.shape[-2]} positions for offset {offset}"
    return None


def _layer_data_error(layer: dict, n: int) -> str | None:
    """Why a deserialized layer does not belong to an ``n``-token entry."""
    if "cachelist_subs" in layer:
        for sub in layer["cachelist_subs"]:
            error = _kv_dict_error(sub, n, nested=True)
            if error is not None:
                return error
        return None
    if "keys" in layer:
        return _kv_dict_error(layer, n)
    return None  # recurrent state carries no position


def _manifest_error(
    manifest: dict, entry_dir: str, tokens: tuple[int, ...]
) -> str | None:
    """Why an entry directory is not the entry for ``tokens``, or None if it is."""
    if manifest.get("num_tokens") != len(tokens):
        return (
            f"manifest records {manifest.get('num_tokens')} tokens, "
            f"key has {len(tokens)}"
        )
    layers = manifest.get("layers")
    if not isinstance(layers, list) or manifest.get("num_layers") != len(layers):
        return "manifest layer list does not match its layer count"
    try:
        with open(os.path.join(entry_dir, "tokens.bin"), "rb") as f:
            stored = f.read()
    except OSError as e:
        return f"tokens.bin unreadable: {e}"
    if stored != _tokens_to_blob(tokens):
        return "tokens.bin differs from the key"
    return None


class SSDCacheTier:
    """Cold-tier disk cache for KV cache entries.

    Manages a SQLite-indexed on-disk cache directory. Evicted RAM entries
    are spilled here via an async writer thread. Cold-tier fetches reload
    from disk asynchronously with RAM budget reservation.

    Directory layout::

        cache_dir/
          index.db           # SQLite metadata index
          data/              # safetensors files per entry
            {hash}/          # one directory per entry
              layer_0.safetensors
              layer_1.safetensors
              manifest.json  # per-entry layer metadata
    """

    _WRITER_JOIN_TIMEOUT_S = 5.0

    def __init__(self, config: SSDCacheConfig) -> None:
        self._config = config
        self._closed = True
        self._writer_thread: threading.Thread | None = None
        self._namespace_lock_fd: int | None = None

        if config.cache_dir is None:
            raise ValueError("SSDCacheConfig.cache_dir must be set")
        if config.identity is None:
            raise ValueError(
                "SSDCacheConfig.identity must be set: entries are only valid for the "
                "model and configuration that computed them"
            )

        self._base_dir = config.cache_dir
        self._identity = config.identity
        self._cache_dir = os.path.join(self._base_dir, config.identity.digest)
        self._data_dir = os.path.join(self._cache_dir, "data")

        # Create directory structure
        os.makedirs(self._base_dir, mode=config.dir_permissions, exist_ok=True)
        self._discard_untagged_layout()
        self._claim_namespace()

        try:
            self._write_identity_file()
            self._reclaim_stale_namespaces()

            # Open SQLite index
            self._index = SSDIndex(self._cache_dir)
            if getattr(self._index, "migrated_from", None) is not None:
                self._purge_data_dir()

            # Stats
            self._stats = SSDCacheStats()
            self._lock = threading.Lock()

            # Lifecycle state is independent from the stats lock: close() may
            # wait for a writer that still needs the stats lock to finish its
            # current entry.
            self._lifecycle_lock = threading.Lock()
            self._accepting_spills = True
            self._writer_shutdown_requested = False
            # Held by the writer while it writes one entry and by clear(), so a
            # flush never overlaps a write. ``_epoch`` counts flushes: a spill
            # queued or promoted before one is dropped, never published after.
            self._write_lock = threading.Lock()
            self._epoch = 0

            # Spill queue and writer thread
            self._spill_queue: queue.Queue = queue.Queue(
                maxsize=config.spill_queue_size
            )
            self._closed = False
        except Exception:
            index = getattr(self, "_index", None)
            if index is not None:
                try:
                    index.close()
                except Exception:
                    logger.exception(
                        "ssd_cache: failed to close index during init cleanup"
                    )
            self._release_namespace()
            raise

    _NAMESPACE_LOCK = ".lock"

    def _claim_namespace(self) -> None:
        """Create this tier's directory and mark it as in use by this process.

        The mark is a shared advisory lock (``flock``) on ``<namespace>/.lock``,
        held until close() (or the process's death, which the kernel handles).
        Reclaiming a stale namespace needs the exclusive lock, so a directory
        a running server holds is never reclaimed however old its files look
        (a server can sit idle, or serve only from RAM, for longer than the
        retention). A reclaim moves the directory away (to a name no tier
        uses) while it holds the lock: a claim that raced it finds the path
        gone or a different ``.lock`` and starts over on a fresh directory.

        On a filesystem that cannot lock (network or FUSE mounts: ``ENOLCK``,
        ``ENOTSUP``) the tier runs without the mark, as builds before it did;
        nothing can then be proved idle, so such a directory is not reclaimed.
        """
        config = self._config
        for _ in range(5):
            os.makedirs(self._cache_dir, mode=config.dir_permissions, exist_ok=True)
            os.makedirs(self._data_dir, mode=config.dir_permissions, exist_ok=True)
            if fcntl is None:
                return
            lock_path = os.path.join(self._cache_dir, self._NAMESPACE_LOCK)
            try:
                fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, config.file_permissions)
            except FileNotFoundError:
                continue  # a reclaim moved the directory away under us
            except OSError:
                logger.warning(
                    "[ssd_cache] cannot create %s; running without the in-use mark",
                    lock_path,
                    exc_info=True,
                )
                return
            try:
                fcntl.flock(fd, fcntl.LOCK_SH)
                if os.fstat(fd).st_ino == os.stat(lock_path).st_ino:
                    self._namespace_lock_fd = fd
                    return
            except FileNotFoundError:
                pass  # moved away by the reclaim that held the lock
            except OSError as exc:
                os.close(fd)
                logger.warning(
                    "[ssd_cache] advisory locks are unavailable on %s (%s); running "
                    "without the in-use mark, and stale namespaces cannot be "
                    "reclaimed from this directory",
                    self._base_dir,
                    exc,
                )
                return
            except BaseException:
                os.close(fd)
                raise
            os.close(fd)
        raise RuntimeError(f"could not claim the SSD cache directory {self._cache_dir}")

    def _release_namespace(self) -> None:
        fd, self._namespace_lock_fd = self._namespace_lock_fd, None
        if fd is not None:
            os.close(fd)  # closing drops the flock

    @staticmethod
    def _is_legacy_index(path: str) -> bool:
        """Whether ``path`` is the index of the builds that kept it in the base dir.

        ``--ssd-cache-dir`` is user-supplied, and any SQLite file named
        ``index.db`` starts with the same header: only a database holding the
        ``entries`` table of those builds (token_hash, tokens_blob, file_path)
        is ours to delete. The probe must not change what it inspects: SQLite
        creates ``-wal`` / ``-shm`` files next to a WAL-mode database it opens,
        even read-only. A database with no WAL is opened ``immutable``; one that
        has a WAL (a crash left it: the table may live only there) is probed as
        a copy in a temporary directory.
        """
        import pathlib
        import shutil
        import sqlite3
        import tempfile

        wal = path + "-wal"
        try:
            with open(path, "rb") as f:
                if f.read(16) != b"SQLite format 3\x00":
                    return False
            has_wal = os.path.getsize(wal) > 0
        except FileNotFoundError:
            has_wal = False
        except OSError:
            return False

        def has_legacy_table(db_path: str, query: str) -> bool:
            try:
                conn = sqlite3.connect(pathlib.Path(db_path).as_uri() + query, uri=True)
            except sqlite3.Error:
                return False
            try:
                columns = {r[1] for r in conn.execute("PRAGMA table_info(entries)")}
            except sqlite3.Error:
                return False
            finally:
                conn.close()
            return {"token_hash", "tokens_blob", "file_path"} <= columns

        if not has_wal:
            return has_legacy_table(path, "?mode=ro&immutable=1")
        try:
            with tempfile.TemporaryDirectory(prefix="ssd-probe-") as tmp:
                shutil.copyfile(path, os.path.join(tmp, "index.db"))
                shutil.copyfile(wal, os.path.join(tmp, "index.db-wal"))
                return has_legacy_table(os.path.join(tmp, "index.db"), "?mode=ro")
        except OSError:
            return False

    def _discard_untagged_layout(self) -> None:
        """Remove entries a build that keyed on tokens alone left in the base dir.

        Those builds kept ``index.db`` and ``data/<sha256>/`` directly in the
        base directory with nothing recording which model computed them, so none
        of it can be attributed to the current identity. ``--ssd-cache-dir`` is
        user-supplied and ``data`` / ``quarantine`` are generic names, so only
        what is positively one of those entries goes: a SQLite ``index.db``
        holding their ``entries`` table (with its WAL files), and ``<64 hex>`` (or ``.tmp``) directories that
        hold a ``manifest.json``. Anything else, including the ``data`` and
        ``quarantine`` directories themselves while they hold other files, is
        left alone.
        """
        import shutil

        removed: list[str] = []
        index_db = os.path.join(self._base_dir, "index.db")
        if self._is_legacy_index(index_db):
            for suffix in ("", "-wal", "-shm"):
                path = index_db + suffix
                try:
                    os.unlink(path)
                    removed.append(os.path.basename(path))
                except FileNotFoundError:
                    pass
                except OSError:
                    logger.warning(
                        "[ssd_cache] failed to remove %s", path, exc_info=True
                    )
        for container in ("data", "quarantine"):
            root = os.path.join(self._base_dir, container)
            if os.path.islink(root) or not os.path.isdir(root):
                continue
            for name in os.listdir(root):
                path = os.path.join(root, name)
                if (
                    _LEGACY_ENTRY_RE.match(name)
                    and os.path.isdir(path)
                    and not os.path.islink(path)
                    and os.path.isfile(os.path.join(path, "manifest.json"))
                ):
                    try:
                        shutil.rmtree(path)
                        removed.append(f"{container}/{name}")
                    except OSError:
                        logger.warning(
                            "[ssd_cache] failed to remove %s", path, exc_info=True
                        )
            try:
                os.rmdir(root)  # only if nothing else is in it
            except OSError:
                pass
        if removed:
            logger.warning(
                "[ssd_cache] removed %d entries of an older layout from %s: they "
                "recorded no model identity",
                len(removed),
                self._base_dir,
            )

    @staticmethod
    def _namespace_last_used(path: str, use_dir: bool = True) -> float | None:
        """When a namespace directory was last used (None: cannot be told).

        The newest of ``identity.json`` (written at each tier start),
        ``index.db`` and its WAL (written by every spill, hit and eviction); a
        directory with none of them (an interrupted start) is as old as the
        directory itself.
        """
        newest: float | None = None
        for name in ("identity.json", "index.db", "index.db-wal"):
            try:
                mtime = os.stat(os.path.join(path, name)).st_mtime
            except OSError:
                continue
            newest = mtime if newest is None else max(newest, mtime)
        if newest is None and use_dir:
            try:
                newest = os.stat(path).st_mtime
            except OSError:
                return None
        return newest

    def _reclaim_stale_namespaces(self) -> None:
        """Delete sibling identity directories nobody has used for a while.

        Every mlx / mlx-lm upgrade, model revision or config change names a new
        directory, and the size budget is per directory, so without this each
        of them strands a whole cache. A directory goes only when it is named
        like an identity digest, nothing in it (identity marker, index, WAL) has
        been written for ``stale_namespace_days``, no running tier holds it
        (``_claim_namespace``) and it is not this tier's own.

        Under the exclusive lock the directory is first renamed to a tombstone
        (``<digest>.reclaim-<pid>-<ns>``, which no tier uses), then the lock is
        dropped and the tombstone deleted: deleting in place would remove the
        ``.lock`` file first, and a tier starting meanwhile would create a new
        one, lock it and settle into a directory being deleted. Tombstones an
        interrupted reclaim left are deleted on the next start.
        """
        import shutil
        import time as _time

        days = self._config.stale_namespace_days
        if days is None:
            return
        cutoff = _time.time() - days * 86400
        names = os.listdir(self._base_dir)
        for name in names:
            if _TOMBSTONE_RE.match(name):
                shutil.rmtree(os.path.join(self._base_dir, name), ignore_errors=True)
        for name in names:
            path = os.path.join(self._base_dir, name)
            if (
                name == self._identity.digest
                or not _DIGEST_DIR_RE.match(name)
                or os.path.islink(path)
                or not os.path.isdir(path)
            ):
                continue
            last_used = self._namespace_last_used(path)
            if last_used is None or last_used >= cutoff:
                continue
            tombstone = self._tombstone_if_idle(path, name, cutoff, last_used)
            if tombstone is None:
                continue
            try:
                shutil.rmtree(tombstone)
            except OSError:
                logger.warning(
                    "[ssd_cache] could not delete reclaimed namespace %s",
                    tombstone,
                    exc_info=True,
                )
                continue
            logger.info("[ssd_cache] reclaimed %s: not used for %s days", name, days)

    def _tombstone_if_idle(
        self, path: str, name: str, cutoff: float, last_used: float
    ) -> str | None:
        """Rename ``path`` to a tombstone if no tier holds it and it is still stale.

        Returns the tombstone's path, or None (nothing changed) when a tier holds
        the directory, locks are unavailable, or it was used again meanwhile.
        """
        if fcntl is None:
            return None
        try:
            lock_fd = os.open(
                os.path.join(path, self._NAMESPACE_LOCK),
                os.O_CREAT | os.O_RDWR,
                self._config.file_permissions,
            )
        except OSError:
            return None  # gone already, or not ours to open
        try:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                logger.debug(
                    "[ssd_cache] %s is held by a running tier; not reclaimed", name
                )
                return None
            except OSError as exc:
                logger.debug(
                    "[ssd_cache] cannot lock %s (%s); not reclaimed", name, exc
                )
                return None
            # A tier may have started, written and closed between the check and
            # the lock: judge again now that nothing can change it. (Taking the
            # lock created ``.lock``, which touches the directory itself: a
            # directory with none of the marker files keeps its earlier age.)
            fresh = self._namespace_last_used(path, use_dir=False)
            if (fresh if fresh is not None else last_used) >= cutoff:
                return None
            tombstone = os.path.join(
                self._base_dir, f"{name}.reclaim-{os.getpid()}-{time.time_ns()}"
            )
            try:
                os.rename(path, tombstone)
            except OSError:
                logger.warning(
                    "[ssd_cache] could not reclaim stale namespace %s",
                    path,
                    exc_info=True,
                )
                return None
            return tombstone
        finally:
            os.close(lock_fd)  # a claimer waiting on the old inode starts over

    def _write_identity_file(self) -> None:
        """Record what this directory's entries are valid for (for operators)."""
        path = os.path.join(self._cache_dir, "identity.json")
        tmp = path + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(
                    {
                        "digest": self._identity.digest,
                        "model": self._identity.model,
                        "kv": self._identity.kv,
                        "detail": self._identity.detail,
                    },
                    f,
                    indent=2,
                    default=str,
                )
            os.replace(tmp, path)
        except OSError:
            logger.warning("[ssd_cache] could not write %s", path, exc_info=True)

    @property
    def namespace_dir(self) -> str:
        """The directory this tier's files live in (base dir + identity digest)."""
        return self._cache_dir

    def _purge_data_dir(self) -> None:
        """Delete every spilled entry after an incompatible schema bump.

        The index rows are already gone, so reconcile() would eventually sweep
        these as orphans -- but reconcile is not guaranteed to run before the
        first spill, and leaving gigabytes of unreadable files on disk counting
        against the size budget is worse than a short startup cost.
        """
        import shutil

        if not os.path.isdir(self._data_dir):
            return
        for entry_name in os.listdir(self._data_dir):
            entry_path = os.path.join(self._data_dir, entry_name)
            if not os.path.isdir(entry_path):
                continue
            try:
                shutil.rmtree(entry_path)
            except OSError:
                logger.warning(
                    "[ssd_cache] failed to remove stale entry directory %s",
                    entry_name,
                    exc_info=True,
                )

    @staticmethod
    def _entry_hash(tokens: tuple[int, ...]) -> str:
        """Compute deterministic hash for a token sequence."""
        return _tokens_hash(tokens)

    def get_stats(self) -> dict:
        """Return current SSD cache statistics."""
        return self._stats.to_dict()

    def start_writer(self) -> None:
        """Start the background spill writer thread."""
        with self._lifecycle_lock:
            if self._closed or self._writer_shutdown_requested:
                raise RuntimeError("cannot start a closed SSD cache tier")
            if self._writer_thread is not None:
                return
            self._writer_thread = threading.Thread(
                target=self._writer_loop, daemon=True, name="ssd-cache-writer"
            )
            self._writer_thread.start()
        logger.info("[ssd_cache] writer thread started")

    def _writer_loop(self) -> None:
        """Drain spill queue and persist entries. Numpy-only — no MLX here."""
        while True:
            item = self._spill_queue.get()
            if item is None:  # Poison pill for shutdown
                break

            epoch, tokens_key, layer_snapshots, memory_bytes = item
            try:
                with self._write_lock:
                    if epoch != self._epoch:
                        continue  # queued before a flush; the flush dropped it
                    self._write_entry(tokens_key, layer_snapshots, memory_bytes)
            except Exception:
                logger.exception(
                    f"[ssd_cache] failed to write entry " f"({len(tokens_key)} tokens)"
                )

    def enqueue_spill(
        self,
        tokens: tuple[int, ...],
        cache: list[Any],
        memory_bytes: int,
    ) -> bool:
        """Enqueue a cache entry for async spill to SSD.

        Must be called on the producer thread (the request handler that
        owns the layer's Stream(gpu, N)) — the snapshot below materializes
        MLX → numpy here so the writer thread never has to.

        Entries are write-once: the KV of a token key is fixed by the cache
        identity (the tier's directory), so a key that is already indexed is
        not written again (it is marked recently used). Without this an entry
        promoted into RAM and evicted again, by a workload cycling through
        more conversations than the memory cache holds, would be re-snapshotted
        and rewritten in full on every round trip.

        Returns True if the entry is on the tier or enqueued, False if it was
        dropped (queue full, tier closing, snapshot failure).
        """
        # The epoch the entry was handed over in: a flush (clear()) that lands
        # while it is being snapshotted or queued must drop it.
        epoch = self._epoch
        with self._lifecycle_lock:
            if not self._accepting_spills:
                return False
            # Under the lock: close() closes the index under it.
            if self._index.lookup_exact(tokens) is not None:
                self._index.touch(tokens)
                return True

        # Dequantize on the CALLER's thread, which owns the MLX GPU stream.
        # mx.dequantize is a GPU compute op; running it on the writer thread
        # aborts the process ("no Stream(gpu,N) in current thread"). Materialize
        # with mx.eval so the writer thread only does a host-side copy. The layer
        # serializers handle plain KVCache/ArraysCache only; any layer whose
        # .keys/.values are a tuple/list of arrays (packed, scales, biases)
        # — either our `_QuantizedCacheWrapper` or mlx-lm's native
        # `QuantizedKVCache` produced when --kv-cache-quantization is on — must
        # be reduced to a single dense array per attribute before queueing.
        import mlx.core as mx
        from .memory_cache import _QuantizedCacheWrapper, _dequantize_cache

        def _is_quantized_layer(layer):
            if isinstance(layer, _QuantizedCacheWrapper):
                return True
            keys = getattr(layer, "keys", None)
            return isinstance(keys, (tuple, list))

        if any(_is_quantized_layer(layer) for layer in cache):
            try:
                from mlx_lm.models.cache import QuantizedKVCache
            except ImportError:
                QuantizedKVCache = ()  # never matches isinstance below

            converted: list = []
            for layer in cache:
                if isinstance(layer, _QuantizedCacheWrapper):
                    # Existing path: wrapper → orig_type with dequantized keys.
                    converted.extend(_dequantize_cache([layer]))
                elif isinstance(layer, QuantizedKVCache) or (
                    hasattr(layer, "keys") and isinstance(layer.keys, (tuple, list))
                ):
                    # Native mlx-lm QuantizedKVCache: keys/values are
                    # (packed, scales, biases) tuples. Dequantize in place
                    # into a fresh KVCache-shaped duck-type for the serializer.
                    from mlx_lm.models.cache import KVCache as _KVCache

                    bits = getattr(layer, "bits", 8)
                    group_size = getattr(layer, "group_size", 64)
                    kv = _KVCache.__new__(_KVCache)
                    kv.keys = mx.dequantize(
                        *layer.keys, group_size=group_size, bits=bits
                    )
                    kv.values = mx.dequantize(
                        *layer.values, group_size=group_size, bits=bits
                    )
                    kv.offset = getattr(layer, "offset", kv.keys.shape[-2])
                    if (
                        kv.keys is not None
                        and hasattr(kv.keys, "shape")
                        and len(kv.keys.shape) >= 3
                        and kv.offset < kv.keys.shape[-2]
                    ):
                        kv.keys = kv.keys[..., : kv.offset, :]
                        kv.values = kv.values[..., : kv.offset, :]
                    converted.append(kv)
                else:
                    converted.append(layer)
            cache = converted
            # Numpy's PEP 3118 buffer protocol doesn't understand bfloat16 —
            # it sees the bf16 array as format "B" (uint8) but the buffer items
            # are 2 bytes, so np.array() raises a RuntimeError mismatch. Cast
            # to float16 here on the CALLER's stream: KV values sit well within
            # the fp16 ±65504 range, fp16 has MORE mantissa bits than bf16, and
            # the byte size is identical. Stash the pre-cast dtype as a sentinel
            # so KVCacheSerializer.snapshot_layer can record it and the reload
            # path (scheduler._reconstruct_ssd_layers) can cast back to bf16 —
            # otherwise the model receives fp16 KV where it computed bf16.
            # Then force eval so the writer thread sees materialized host buffers.
            for layer in cache:
                k = getattr(layer, "keys", None)
                v = getattr(layer, "values", None)
                if k is None or v is None or isinstance(k, (tuple, list)):
                    continue
                if str(getattr(k, "dtype", "")).endswith("bfloat16"):
                    layer._ssd_keys_original_dtype = "bfloat16"
                    layer._ssd_values_original_dtype = "bfloat16"
                    layer.keys = k.astype(mx.float16)
                    layer.values = v.astype(mx.float16)
                mx.eval(layer.keys, layer.values)
            logger.info(
                "[ssd_cache] dequantized %d layers before spill (%d tokens)",
                len(cache),
                len(tokens),
            )

        try:
            layer_snapshots: list[tuple[LayerSerializer, dict[str, Any]]] = []
            for layer in cache:
                serializer = get_serializer_for_layer(layer)
                snapshot = serializer.snapshot_layer(layer)
                layer_snapshots.append((serializer, snapshot))
        except Exception:
            logger.exception(
                "[ssd_cache] failed to snapshot layers for spill "
                f"({len(tokens)} tokens) — entry dropped"
            )
            return False

        with self._lifecycle_lock:
            if not self._accepting_spills:
                return False
            try:
                self._spill_queue.put_nowait(
                    (epoch, tokens, layer_snapshots, memory_bytes)
                )
                return True
            except queue.Full:
                logger.warning(
                    f"[ssd_cache] spill queue full, dropping entry "
                    f"({len(tokens)} tokens, {memory_bytes} bytes)"
                )
                return False

    def _write_entry(
        self,
        tokens_key: tuple[int, ...],
        layer_snapshots: list[tuple[LayerSerializer, dict[str, Any]]],
        memory_bytes: int,
    ) -> None:
        """Atomically persist one entry (writer thread; numpy-only input)."""
        import shutil

        entry_hash = self._entry_hash(tokens_key)
        entry_dir = os.path.join(self._data_dir, entry_hash)
        tmp_dir = entry_dir + ".tmp"

        # Clean up any leftover tmp dir from a previous crash
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)

        os.makedirs(tmp_dir, mode=self._config.dir_permissions, exist_ok=True)

        layer_manifests = []
        total_file_bytes = 0

        for i, (serializer, snapshot) in enumerate(layer_snapshots):
            layer_path = os.path.join(tmp_dir, f"layer_{i}.safetensors")
            metadata = serializer.serialize_layer(snapshot, i, layer_path)
            layer_manifests.append(metadata)

            # Set file permissions
            os.chmod(layer_path, self._config.file_permissions)
            total_file_bytes += os.path.getsize(layer_path)

        # Write manifest
        manifest = {
            "num_layers": len(layer_snapshots),
            "layers": layer_manifests,
            "memory_bytes": memory_bytes,
            "num_tokens": len(tokens_key),
        }
        manifest_path = os.path.join(tmp_dir, "manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)
        os.chmod(manifest_path, self._config.file_permissions)

        # Save tokens binary
        tokens_path = os.path.join(tmp_dir, "tokens.bin")
        arr = _array.array("i", tokens_key)
        with open(tokens_path, "wb") as f:
            arr.tofile(f)
        os.chmod(tokens_path, self._config.file_permissions)

        # Atomic rename: tmp_dir -> entry_dir
        if os.path.exists(entry_dir):
            shutil.rmtree(entry_dir)
        os.rename(tmp_dir, entry_dir)

        # Update index
        relative_path = entry_hash
        self._index.insert_entry(
            tokens_key=tokens_key,
            file_path=relative_path,
            memory_bytes=memory_bytes,
            num_tokens=len(tokens_key),
            trimmable=all(
                isinstance(serializer, KVCacheSerializer) and "max_size" not in snapshot
                for serializer, snapshot in layer_snapshots
            ),
        )

        # Update stats
        with self._lock:
            self._stats.spill_count += 1
            self._stats.spill_bytes += total_file_bytes

        logger.debug(
            f"[ssd_cache] spilled entry: {len(tokens_key)} tokens, "
            f"{total_file_bytes} bytes on disk"
        )

        # Enforce capacity after write
        self._enforce_capacity()

    def lookup_ssd(self, tokens: tuple[int, ...]) -> dict | None:
        """Synchronous check whether tokens exist in SSD tier.

        This is fast (SQLite lookup only, no disk I/O for data).
        Called from synchronous fetch() to report an SSD candidate.

        Returns:
            Dict with entry metadata if found, None otherwise.
        """
        result = self._index.lookup_exact(tokens)
        if result is not None:
            return result
        return None

    def lookup_ssd_prefix(self, tokens: tuple[int, ...]) -> dict | None:
        """Find the longest prefix match in the SSD tier.

        Returns the longest-prefix entry metadata or None.
        """
        results = self._index.lookup_prefix(tokens)
        if results:
            return results[0]  # Already sorted by num_tokens DESC
        return None

    def lookup_ssd_lcp(self, tokens: tuple[int, ...], min_tokens: int) -> dict | None:
        """Find the entry sharing the longest prefix with ``tokens``.

        For a stored key longer than the query or diverging from it (see
        ``SSDIndex.lookup_lcp``). The entry's data covers its whole key; the
        caller reads it and cuts it back to the shared ``lcp`` tokens.
        """
        return self._index.lookup_lcp(tokens, min_tokens)

    def remove(self, tokens: tuple[int, ...]) -> bool:
        """Remove an entry from both the SSD index and its data directory."""
        import shutil

        meta = self._index.lookup_exact(tokens)
        if meta is None:
            return False

        # Remove the index first so a data-file cleanup failure cannot expose
        # the rejected entry to another promotion. Reconciliation removes any
        # orphaned directory left behind.
        self._index.delete_entry(tokens)
        entry_dir = os.path.join(self._data_dir, meta["file_path"])
        try:
            if os.path.exists(entry_dir):
                shutil.rmtree(entry_dir)
        except OSError:
            logger.warning(
                "[ssd_cache] failed to remove entry directory %s",
                meta["file_path"],
                exc_info=True,
            )
        return True

    def clear(self) -> int:
        """Drop every entry of this tier (a cache flush), returning how many.

        Spills still queued, one being written and promotions in flight are all
        dropped: nothing written before the flush can be promoted after it.
        Waits for the entry the writer is on, if any. A closed tier has nothing
        left to drop.
        """
        import shutil

        if self._closed:
            return 0

        with self._write_lock:
            self._epoch += 1
            while True:
                try:
                    item = self._spill_queue.get_nowait()
                except queue.Empty:
                    break
                if item is None:
                    # close() is waiting on its shutdown sentinel; it stays queued.
                    self._spill_queue.put_nowait(None)
                    break
            dropped = self._index.get_entry_count()
            self._index.delete_all()
            for name in os.listdir(self._data_dir):
                path = os.path.join(self._data_dir, name)
                try:
                    if os.path.isdir(path) and not os.path.islink(path):
                        shutil.rmtree(path)
                    else:
                        os.unlink(path)
                except OSError:
                    logger.warning(
                        "[ssd_cache] failed to remove %s during clear",
                        path,
                        exc_info=True,
                    )
        logger.info("[ssd_cache] cleared %d entries", dropped)
        return dropped

    async def async_promote(
        self,
        tokens: tuple[int, ...],
        reserve_budget_fn,
        release_budget_fn,
    ) -> list | None:
        """Promote an entry from SSD to RAM asynchronously.

        CRITICAL: Reserves RAM budget BEFORE the disk read, to avoid
        thrash when multiple promotions race.

        Args:
            tokens: Token sequence to promote.
            reserve_budget_fn: Callable(nbytes) -> bool. Must return True
                if budget is available and reserved, False otherwise.
            release_budget_fn: Callable(nbytes) -> None. Called to release
                budget on failure.

        Returns:
            List of deserialized cache layers, or None if promotion failed.
        """
        import asyncio

        epoch = self._epoch
        # Step 1: Look up metadata (fast, SQLite)
        meta = self._index.lookup_exact(tokens)
        if meta is None:
            with self._lock:
                self._stats.ssd_misses += 1
            return None

        memory_bytes = meta["memory_bytes"]

        # Step 2: Reserve RAM budget BEFORE disk read
        if not reserve_budget_fn(memory_bytes):
            with self._lock:
                self._stats.promotion_failures += 1
            logger.warning(
                f"[ssd_cache] promotion denied: cannot reserve "
                f"{memory_bytes} bytes RAM budget"
            )
            return None

        # Step 3: Read from disk (in thread pool to avoid blocking event loop)
        # Use shield-and-await-on-cancel per CLAUDE.md Golden Rule #4:
        # budget must be released even if the calling task is cancelled.
        t0 = time.time()
        worker = asyncio.ensure_future(
            asyncio.to_thread(self._read_entry, tokens, meta["file_path"])
        )
        try:
            cache_layers = await asyncio.shield(worker)
        except asyncio.CancelledError:
            # Caller cancelled — still need to wait for the disk read
            # to finish, then release the budget
            try:
                await worker
            except Exception:
                pass
            release_budget_fn(memory_bytes)
            raise
        except Exception:
            # Release budget on read failure
            release_budget_fn(memory_bytes)
            with self._lock:
                self._stats.promotion_failures += 1
            logger.exception(
                f"[ssd_cache] failed to read entry from disk "
                f"({meta['num_tokens']} tokens)"
            )
            return None

        if epoch != self._epoch:
            # A flush landed while the entry was being read.
            release_budget_fn(memory_bytes)
            return None

        if cache_layers is None:
            # Corrupted entry — release budget, quarantine entry
            release_budget_fn(memory_bytes)
            with self._lock:
                self._stats.promotion_failures += 1
            return None

        dt = time.time() - t0
        total_read_bytes = sum(
            os.path.getsize(
                os.path.join(
                    self._data_dir, meta["file_path"], f"layer_{i}.safetensors"
                )
            )
            for i in range(len(cache_layers))
            if os.path.exists(
                os.path.join(
                    self._data_dir, meta["file_path"], f"layer_{i}.safetensors"
                )
            )
        )

        with self._lock:
            self._stats.ssd_hits += 1
            self._stats.reload_latency_sum += dt
            self._stats.reload_bytes += total_read_bytes

        # Update access time in index
        self._index.touch(tokens)

        logger.info(
            f"[ssd_cache] promoted entry: {meta['num_tokens']} tokens, "
            f"{total_read_bytes} bytes, {dt*1000:.1f}ms"
        )

        return cache_layers

    def _read_entry(self, tokens: tuple[int, ...], relative_path: str) -> list | None:
        """Read a cache entry from disk. Called from thread pool.

        Returns list of deserialized layer dicts, or None on corruption.
        """
        entry_dir = os.path.join(self._data_dir, relative_path)
        manifest_path = os.path.join(entry_dir, "manifest.json")

        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"[ssd_cache] corrupt manifest for {relative_path}: {e}")
            self._quarantine_entry(tokens, relative_path)
            return None
        # An entry is served as the KV of ``tokens`` and the request then
        # prefills only what follows them, so one that holds anything else is a
        # wrong answer, not a miss: check it against the key before using it.
        error = _manifest_error(manifest, entry_dir, tokens)
        if error is not None:
            logger.warning(f"[ssd_cache] inconsistent entry {relative_path}: {error}")
            self._quarantine_entry(tokens, relative_path)
            return None

        cache_layers = []
        for layer_meta in manifest["layers"]:
            layer_idx = layer_meta["layer_idx"]
            layer_path = os.path.join(entry_dir, f"layer_{layer_idx}.safetensors")
            layer_type = layer_meta["layer_type"]

            try:
                if layer_type in ("KVCache", "RotatingKVCache"):
                    serializer = KVCacheSerializer()
                elif layer_type in ("ArraysCache", "MambaCache"):
                    serializer = ArraysCacheSerializer()
                elif layer_type == "CacheList":
                    serializer = CacheListSerializer()
                else:
                    logger.warning(
                        f"[ssd_cache] unknown layer type {layer_type}, skipping"
                    )
                    self._quarantine_entry(tokens, relative_path)
                    return None

                layer_data = serializer.deserialize_layer(layer_path, layer_meta)
                error = _layer_data_error(layer_data, len(tokens))
                if error is not None:
                    logger.warning(
                        f"[ssd_cache] inconsistent layer {layer_idx} in "
                        f"{relative_path}: {error}"
                    )
                    self._quarantine_entry(tokens, relative_path)
                    return None
                cache_layers.append(layer_data)
            except Exception as e:
                logger.warning(
                    f"[ssd_cache] corrupt layer {layer_idx} in {relative_path}: {e}"
                )
                self._quarantine_entry(tokens, relative_path)
                return None

        return cache_layers

    def _quarantine_entry(self, tokens: tuple[int, ...], relative_path: str) -> None:
        """Move a corrupt entry to quarantine and remove from index."""
        entry_dir = os.path.join(self._data_dir, relative_path)
        quarantine_dir = os.path.join(self._cache_dir, "quarantine", relative_path)

        try:
            if os.path.exists(entry_dir):
                os.makedirs(
                    os.path.dirname(quarantine_dir),
                    mode=self._config.dir_permissions,
                    exist_ok=True,
                )
                os.rename(entry_dir, quarantine_dir)
                logger.warning(
                    f"[ssd_cache] quarantined corrupt entry: {relative_path}"
                )
        except OSError as e:
            logger.warning(f"[ssd_cache] failed to quarantine {relative_path}: {e}")

        self._index.delete_entry(tokens)

    def _enforce_capacity(self) -> None:
        """Evict oldest SSD entries until within capacity limits.

        Called after each spill write. Removes entries by LRU order
        until both entry count and total bytes are within bounds.
        """
        import shutil

        while True:
            entry_count = self._index.get_entry_count()
            total_bytes = self._index.get_total_bytes()

            needs_evict = (
                entry_count > self._config.max_entries
                or total_bytes > self._config.max_size_bytes
            )
            if not needs_evict:
                break

            lru = self._index.get_lru(limit=1)
            if not lru:
                break

            victim = lru[0]
            victim_tokens = _blob_to_tokens(victim["tokens_blob"])
            victim_dir = os.path.join(self._data_dir, victim["file_path"])

            # Delete data files
            if os.path.exists(victim_dir):
                shutil.rmtree(victim_dir)

            # Delete from index
            self._index.delete_entry(victim_tokens)

            logger.debug(
                f"[ssd_cache] disk LRU evicted: {victim['num_tokens']} tokens, "
                f"{victim['memory_bytes']} bytes"
            )

    def reconcile(self) -> int:
        """Reconcile index with files on disk.

        Removes index entries whose data files are missing.
        Removes data directories not in the index.

        Returns number of entries cleaned up.
        """
        import shutil

        cleaned = 0

        # Phase 1: Remove index entries with missing data dirs
        all_entries = self._index.all_entries()
        for entry in all_entries:
            entry_dir = os.path.join(self._data_dir, entry["file_path"])
            manifest_path = os.path.join(entry_dir, "manifest.json")
            if not os.path.isdir(entry_dir) or not os.path.exists(manifest_path):
                tokens = _blob_to_tokens(entry["tokens_blob"])
                self._index.delete_entry(tokens)
                cleaned += 1
                logger.info(
                    f"[ssd_cache] reconcile: removed orphaned index entry "
                    f"({entry['num_tokens']} tokens, path={entry['file_path']})"
                )

        # Phase 2: Remove data directories not in the index
        if os.path.isdir(self._data_dir):
            indexed_hashes = {e["file_path"] for e in self._index.all_entries()}
            for entry_name in os.listdir(self._data_dir):
                entry_path = os.path.join(self._data_dir, entry_name)
                if (
                    os.path.isdir(entry_path)
                    and entry_name not in indexed_hashes
                    and not entry_name.endswith(".tmp")
                ):
                    shutil.rmtree(entry_path)
                    cleaned += 1
                    logger.info(
                        f"[ssd_cache] reconcile: removed orphaned data dir "
                        f"{entry_name}"
                    )

        if cleaned > 0:
            logger.info(f"[ssd_cache] reconciliation cleaned {cleaned} entries")

        return cleaned

    def close(self) -> None:
        """Drain pending spills, stop the writer, and release resources.

        If the writer does not terminate within the bounded join, retain both
        the thread reference and the SQLite index so a live writer can finish
        safely. A later close() call can retry the join.
        """
        with self._lifecycle_lock:
            if self._closed:
                return

            self._accepting_spills = False
            if self._writer_thread is not None:
                shutdown_deadline = time.monotonic() + self._WRITER_JOIN_TIMEOUT_S
                if not self._writer_shutdown_requested:
                    # FIFO ordering makes the poison pill a drain barrier: all
                    # spills accepted before shutdown are written first.
                    try:
                        self._spill_queue.put(None, timeout=self._WRITER_JOIN_TIMEOUT_S)
                    except queue.Full as exc:
                        raise TimeoutError(
                            "SSD cache shutdown sentinel could not be queued "
                            "before timeout"
                        ) from exc
                    self._writer_shutdown_requested = True

                join_timeout = max(0.0, shutdown_deadline - time.monotonic())
                self._writer_thread.join(timeout=join_timeout)
                if self._writer_thread.is_alive():
                    raise TimeoutError(
                        "SSD cache writer thread did not stop before timeout"
                    )
                self._writer_thread = None

            self._index.close()
            self._closed = True
            self._release_namespace()
        logger.info("[ssd_cache] SSDCacheTier closed")

    async def aclose(self) -> None:
        """Close without blocking the caller's asyncio event loop."""
        import asyncio

        await asyncio.to_thread(self.close)
