# SPDX-License-Identifier: Apache-2.0
"""Tests for SSD KV cache tiering."""

import pytest

from vllm_mlx.cache_identity import CacheIdentity
from vllm_mlx.ssd_cache import SSDCacheConfig, SSDCacheStats

IDENT = CacheIdentity(model="model-a", window="window-a", kv="kv-a")


class TestSSDCacheConfig:
    """Tests for SSDCacheConfig."""

    def test_default_config(self):
        config = SSDCacheConfig()
        assert config.cache_dir is None
        assert config.max_size_gb == 10.0
        assert config.max_entries == 10000
        assert config.file_permissions == 0o600
        assert config.dir_permissions == 0o700
        assert config.spill_queue_size == 64
        assert config.retention_seconds is None

    def test_custom_config(self):
        config = SSDCacheConfig(
            identity=IDENT,
            cache_dir="/tmp/test-ssd-cache",
            max_size_gb=5.0,
            max_entries=500,
        )
        assert config.cache_dir == "/tmp/test-ssd-cache"
        assert config.max_size_gb == 5.0
        assert config.max_entries == 500

    def test_max_size_bytes(self):
        config = SSDCacheConfig(max_size_gb=2.0)
        assert config.max_size_bytes == 2 * 1024 * 1024 * 1024

    def test_invalid_max_size(self):
        with pytest.raises(ValueError, match="max_size_gb"):
            SSDCacheConfig(max_size_gb=0.0)

    def test_invalid_max_entries(self):
        with pytest.raises(ValueError, match="max_entries"):
            SSDCacheConfig(max_entries=0)

    def test_invalid_spill_queue_size(self):
        with pytest.raises(ValueError, match="spill_queue_size"):
            SSDCacheConfig(spill_queue_size=0)


class TestSSDCacheStats:
    """Tests for SSDCacheStats."""

    def test_initial_stats(self):
        stats = SSDCacheStats()
        assert stats.spill_count == 0
        assert stats.spill_bytes == 0
        assert stats.ssd_hits == 0
        assert stats.ssd_misses == 0
        assert stats.reload_latency_sum == 0.0
        assert stats.reload_bytes == 0
        assert stats.promotion_failures == 0

    def test_to_dict(self):
        stats = SSDCacheStats(
            spill_count=10,
            spill_bytes=1024 * 1024,
            ssd_hits=5,
            ssd_misses=3,
            reload_latency_sum=0.5,
            reload_bytes=512 * 1024,
            promotion_failures=1,
        )
        d = stats.to_dict()
        assert d["spill_count"] == 10
        assert d["spill_bytes"] == 1024 * 1024
        assert d["ssd_hits"] == 5
        assert d["ssd_misses"] == 3
        assert d["reload_bytes"] == 512 * 1024
        assert d["promotion_failures"] == 1
        assert d["ssd_hit_rate"] == pytest.approx(5 / 8)
        assert d["avg_reload_latency_ms"] == pytest.approx(100.0)

    def test_hit_rate_no_lookups(self):
        stats = SSDCacheStats()
        d = stats.to_dict()
        assert d["ssd_hit_rate"] == 0.0
        assert d["avg_reload_latency_ms"] == 0.0


import json
import os
import threading

from vllm_mlx import ssd_cache as ssd_cache_module
from vllm_mlx.ssd_cache import SSDIndex


class TestSSDIndex:
    """Tests for SQLite-backed SSD cache index."""

    @pytest.fixture
    def db_dir(self, tmp_path):
        return str(tmp_path / "ssd_index_test")

    @pytest.fixture
    def index(self, db_dir):
        os.makedirs(db_dir, exist_ok=True)
        idx = SSDIndex(db_dir)
        yield idx
        idx.close()

    def test_create_opens_db(self, index, db_dir):
        assert os.path.exists(os.path.join(db_dir, "index.db"))

    def test_insert_and_lookup_exact(self, index):
        tokens = (1, 2, 3, 4, 5)
        index.insert_entry(
            tokens_key=tokens,
            file_path="entry_abc123.safetensors",
            memory_bytes=4096,
            num_tokens=5,
        )
        result = index.lookup_exact(tokens)
        assert result is not None
        assert result["file_path"] == "entry_abc123.safetensors"
        assert result["memory_bytes"] == 4096
        assert result["num_tokens"] == 5

    def test_lookup_exact_miss(self, index):
        result = index.lookup_exact((99, 98, 97))
        assert result is None

    def test_lookup_prefix(self, index):
        # Insert a few entries
        index.insert_entry((1, 2, 3), "a.safetensors", 1000, 3)
        index.insert_entry((1, 2, 3, 4, 5), "b.safetensors", 2000, 5)
        index.insert_entry((1, 2, 3, 4, 5, 6, 7), "c.safetensors", 3000, 7)
        index.insert_entry((9, 8, 7), "d.safetensors", 1000, 3)

        # Lookup prefix matches for (1,2,3,4,5,6,7,8)
        results = index.lookup_prefix((1, 2, 3, 4, 5, 6, 7, 8))
        # Should return entries whose tokens are a prefix of the query
        file_paths = [r["file_path"] for r in results]
        assert "a.safetensors" in file_paths
        assert "b.safetensors" in file_paths
        assert "c.safetensors" in file_paths
        assert "d.safetensors" not in file_paths

    def test_lookup_prefix_uses_bounded_prefix_hash_candidates(self, index):
        index.insert_entry((1, 2, 3, 4), "match.safetensors", 1000, 4)
        for i in range(100):
            index.insert_entry((9000 + i, 2, 3, 4), f"miss-{i}.safetensors", 1000, 4)

        results = index.lookup_prefix((1, 2, 3, 4, 5))

        assert [r["file_path"] for r in results] == ["match.safetensors"]
        with index._db_lock:
            cur = index._conn.execute(
                "SELECT COUNT(*) FROM entries WHERE prefix_hash = ?",
                (ssd_cache_module._prefix_hash((1, 2, 3, 4)),),
            )
            assert cur.fetchone()[0] == 1

    def test_lookup_lcp_serves_a_stored_key_longer_than_the_query(self, index):
        stored = tuple(range(100, 200))
        index.insert_entry(stored, "long", 1000, len(stored), trimmable=True)

        hit = index.lookup_lcp(stored[:60], min_tokens=16)

        assert hit["file_path"] == "long"
        assert hit["lcp"] == 60
        assert hit["tokens"] == stored

    def test_lookup_lcp_serves_a_diverging_key_up_to_the_shared_head(self, index):
        stored = tuple(range(100, 200))
        index.insert_entry(stored, "long", 1000, len(stored), trimmable=True)

        hit = index.lookup_lcp(stored[:40] + (7, 8, 9, 10), min_tokens=16)

        assert hit["lcp"] == 40

    def test_lookup_lcp_prefers_the_longest_shared_prefix_then_the_shorter_entry(
        self, index
    ):
        base = tuple(range(100, 220))
        index.insert_entry(base[:90] + (1, 2), "diverges-at-90", 1, 92, trimmable=True)
        index.insert_entry(base, "long", 1, 120, trimmable=True)
        index.insert_entry(base[:110], "shorter", 1, 110, trimmable=True)

        assert index.lookup_lcp(base[:100], 16)["file_path"] == "shorter"
        assert index.lookup_lcp(base[:90] + (5,) * 5, 16)["lcp"] == 90

    def test_lookup_lcp_never_returns_what_cannot_be_cut_back(self, index):
        stored = tuple(range(100, 200))
        index.insert_entry(stored, "rotating", 1000, len(stored), trimmable=False)
        index.insert_entry(stored[:50] + (3,) * 60, "legacy", 1, 110)  # no layout known
        with index._db_lock:
            index._conn.execute(
                "UPDATE entries SET trimmable = NULL WHERE file_path = 'legacy'"
            )
            index._conn.commit()

        assert index.lookup_lcp(stored[:40], min_tokens=16) is None

    def test_lookup_lcp_leaves_prefix_keys_and_short_matches_alone(self, index):
        stored = tuple(range(100, 200))
        index.insert_entry(stored, "long", 1000, len(stored), trimmable=True)

        # the stored key is a prefix of the query: lookup_prefix's, nothing to cut
        assert index.lookup_lcp(stored + (1, 2, 3), 16) is None
        # the same key: lookup_exact's
        assert index.lookup_lcp(stored, 16) is None
        # shares 30 tokens: below the caller's minimum, and a head of another key
        assert index.lookup_lcp(stored[:30] + (5,) * 40, min_tokens=32) is None
        assert index.lookup_lcp((9,) + stored[1:60], min_tokens=16) is None
        # a query shorter than the minimum has no shared prefix long enough
        assert index.lookup_lcp(stored[:20], min_tokens=32) is None

    def test_legacy_index_backfills_prefix_hashes(self, db_dir):
        import sqlite3
        import time

        os.makedirs(db_dir, exist_ok=True)
        conn = sqlite3.connect(os.path.join(db_dir, "index.db"))
        conn.executescript(f"""
            CREATE TABLE schema_version (version INTEGER NOT NULL);
            INSERT INTO schema_version (version)
                VALUES ({SSDIndex._SCHEMA_VERSION});
            CREATE TABLE entries (
                token_hash   TEXT PRIMARY KEY,
                tokens_blob  BLOB NOT NULL,
                num_tokens   INTEGER NOT NULL,
                file_path    TEXT NOT NULL,
                memory_bytes INTEGER NOT NULL,
                created_at   REAL NOT NULL,
                accessed_at  REAL NOT NULL
            );
            """)
        tokens = (1, 2, 3)
        now = time.time()
        conn.execute(
            "INSERT INTO entries VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                ssd_cache_module._tokens_hash(tokens),
                ssd_cache_module._tokens_to_blob(tokens),
                len(tokens),
                "legacy.safetensors",
                1000,
                now,
                now,
            ),
        )
        conn.commit()
        conn.close()

        idx = SSDIndex(db_dir)
        try:
            assert (
                idx.lookup_prefix((1, 2, 3, 4))[0]["file_path"] == "legacy.safetensors"
            )
            with idx._db_lock:
                cur = idx._conn.execute(
                    "SELECT prefix_hash FROM entries WHERE file_path = ?",
                    ("legacy.safetensors",),
                )
                assert cur.fetchone()[0] == ssd_cache_module._prefix_hash(tokens)
        finally:
            idx.close()

    def test_delete_entry(self, index):
        tokens = (10, 20, 30)
        index.insert_entry(tokens, "x.safetensors", 500, 3)
        assert index.lookup_exact(tokens) is not None

        index.delete_entry(tokens)
        assert index.lookup_exact(tokens) is None

    def test_get_lru(self, index):
        import time

        index.insert_entry((1,), "a.safetensors", 100, 1)
        time.sleep(0.01)  # Ensure different timestamps
        index.insert_entry((2,), "b.safetensors", 200, 1)
        time.sleep(0.01)
        index.insert_entry((3,), "c.safetensors", 300, 1)

        # Get oldest 2 entries
        lru = index.get_lru(limit=2)
        assert len(lru) == 2
        assert lru[0]["file_path"] == "a.safetensors"
        assert lru[1]["file_path"] == "b.safetensors"

    def test_get_total_bytes(self, index):
        index.insert_entry((1,), "a.safetensors", 1000, 1)
        index.insert_entry((2,), "b.safetensors", 2000, 1)
        assert index.get_total_bytes() == 3000

    def test_get_total_bytes_empty(self, index):
        assert index.get_total_bytes() == 0

    def test_get_entry_count(self, index):
        assert index.get_entry_count() == 0
        index.insert_entry((1,), "a.safetensors", 100, 1)
        index.insert_entry((2,), "b.safetensors", 200, 1)
        assert index.get_entry_count() == 2

    def test_insert_duplicate_replaces(self, index):
        tokens = (1, 2, 3)
        index.insert_entry(tokens, "old.safetensors", 100, 3)
        index.insert_entry(tokens, "new.safetensors", 200, 3)
        result = index.lookup_exact(tokens)
        assert result["file_path"] == "new.safetensors"
        assert result["memory_bytes"] == 200
        assert index.get_entry_count() == 1

    def test_touch_updates_access_time(self, index):
        import time

        index.insert_entry((1,), "a.safetensors", 100, 1)
        time.sleep(0.01)
        index.insert_entry((2,), "b.safetensors", 100, 1)

        # Touch the older entry
        index.touch((1,))

        # Now (2,) should be the LRU entry
        lru = index.get_lru(limit=1)
        assert lru[0]["file_path"] == "b.safetensors"


import numpy as np

from vllm_mlx.ssd_cache import (
    KVCacheSerializer,
    ArraysCacheSerializer,
    get_serializer_for_layer,
    SERIALIZER_SUPPORT_MATRIX,
)


class MockMLXArray:
    """Minimal mock for MLX array with shape, dtype, and numpy conversion."""

    def __init__(self, data):
        self._data = np.array(data)
        self.shape = self._data.shape
        self.dtype = type("dtype", (), {"size": self._data.dtype.itemsize})()

    def __array__(self):
        return self._data


class MockKVCacheLayer:
    """Mock KVCache layer with keys, values, offset."""

    def __init__(self, keys, values, offset):
        self.keys = keys
        self.values = values
        self.offset = offset


class MockArraysCacheLayer:
    """Mock ArraysCache layer with state list."""

    def __init__(self, state):
        self.state = state


class TestLayerSerializer:
    """Tests for per-layer serializer interface."""

    def test_support_matrix_has_required_types(self):
        assert "KVCache" in SERIALIZER_SUPPORT_MATRIX
        assert "RotatingKVCache" in SERIALIZER_SUPPORT_MATRIX
        assert "ArraysCache" in SERIALIZER_SUPPORT_MATRIX

    def test_kv_cache_serializer_round_trip(self, tmp_path):
        keys = MockMLXArray(np.random.randn(1, 8, 32, 64).astype(np.float16))
        values = MockMLXArray(np.random.randn(1, 8, 32, 64).astype(np.float16))
        layer = MockKVCacheLayer(keys=keys, values=values, offset=32)

        serializer = KVCacheSerializer()
        file_path = str(tmp_path / "layer_0.safetensors")
        # serialize_layer now expects a snapshot (CPU-only dict) produced by
        # snapshot_layer, not the raw MLX-backed layer. This mirrors the
        # producer/writer thread split: snapshot runs on the request thread,
        # serialize runs on the SSD writer thread with only the snapshot.
        snapshot = serializer.snapshot_layer(layer)
        metadata = serializer.serialize_layer(snapshot, 0, file_path)

        assert os.path.exists(file_path)
        assert metadata["layer_type"] == "KVCache"
        assert metadata["offset"] == 32

        restored = serializer.deserialize_layer(file_path, metadata)
        assert restored["offset"] == 32
        np.testing.assert_array_almost_equal(
            np.array(restored["keys"]),
            np.array(keys),
        )
        np.testing.assert_array_almost_equal(
            np.array(restored["values"]),
            np.array(values),
        )

    def test_arrays_cache_serializer_round_trip(self, tmp_path):
        arr0 = MockMLXArray(np.random.randn(1, 64, 128).astype(np.float16))
        arr1 = MockMLXArray(np.random.randn(1, 64, 128).astype(np.float16))
        layer = MockArraysCacheLayer(state=[arr0, arr1])

        serializer = ArraysCacheSerializer()
        file_path = str(tmp_path / "layer_0.safetensors")
        # See KVCacheSerializer round-trip test: snapshot is the producer-
        # side step, serialize is the writer-side step.
        snapshot = serializer.snapshot_layer(layer)
        metadata = serializer.serialize_layer(snapshot, 0, file_path)

        assert os.path.exists(file_path)
        assert metadata["layer_type"] == "ArraysCache"
        assert metadata["num_arrays"] == 2

        restored = serializer.deserialize_layer(file_path, metadata)
        assert len(restored["state"]) == 2
        np.testing.assert_array_almost_equal(
            np.array(restored["state"][0]),
            np.array(arr0),
        )

    def test_get_serializer_for_kvcache(self):
        layer = MockKVCacheLayer(keys=None, values=None, offset=0)
        s = get_serializer_for_layer(layer)
        assert isinstance(s, KVCacheSerializer)

    def test_get_serializer_for_arrays_cache(self):
        layer = MockArraysCacheLayer(state=[])
        s = get_serializer_for_layer(layer)
        assert isinstance(s, ArraysCacheSerializer)

    def test_get_serializer_unknown_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            get_serializer_for_layer("not a cache layer")

    def test_support_matrix_includes_quantized_kv_cache(self):
        """SUPPORT_MATRIX must surface both quantized cache types so the
        diagnostics table reflects what enqueue_spill's dequant path handles."""
        assert (
            SERIALIZER_SUPPORT_MATRIX["QuantizedKVCache"]
            == "supported_via_dequant_on_spill"
        )
        assert (
            SERIALIZER_SUPPORT_MATRIX["_QuantizedCacheWrapper"]
            == "supported_via_dequant_on_spill"
        )

    def test_mx_to_numpy_safe_upcasts_a_real_bfloat16_array(self):
        """Real bf16 mx.array, not a simulated exception.

        The other bf16 tests raise the buffer-protocol error by hand, so they
        pass regardless of which exception class mlx actually uses. That hid a
        live bug: the handler caught only RuntimeError while mlx 0.31.x raises
        ValueError ("'bfloat16' is not a valid PEP 3118 buffer format string"),
        so every bf16 KV spill failed on current mlx. Drive the real array.
        """
        mx = pytest.importorskip("mlx.core")
        from vllm_mlx.ssd_cache import _mx_to_numpy_safe

        arr = mx.zeros((2, 3), dtype=mx.bfloat16)
        np_arr, original_dtype = _mx_to_numpy_safe(arr)

        assert original_dtype == "bfloat16"
        assert np_arr.dtype == np.float32
        assert np_arr.shape == (2, 3)

    def test_snapshot_layer_honors_original_dtype_sentinel(self):
        """When enqueue_spill's dequant path stashed a pre-cast dtype on the
        layer, snapshot_layer must surface it in the snapshot — without this,
        the reload path leaves the model holding fp16 KV where it computed
        bf16. Regression for the dtype-round-trip gap @waybarrios flagged on
        PR #605 (the fp16-cast layers carried no dtype hint because
        _mx_to_numpy_safe only sets a hint on its own upcast path)."""
        keys = MockMLXArray(np.random.randn(1, 8, 4, 16).astype(np.float16))
        values = MockMLXArray(np.random.randn(1, 8, 4, 16).astype(np.float16))
        layer = MockKVCacheLayer(keys=keys, values=values, offset=4)
        layer._ssd_keys_original_dtype = "bfloat16"
        layer._ssd_values_original_dtype = "bfloat16"

        snapshot = KVCacheSerializer().snapshot_layer(layer)

        assert snapshot["keys_original_dtype"] == "bfloat16"
        assert snapshot["values_original_dtype"] == "bfloat16"

    def test_snapshot_layer_sentinel_overrides_autodetect(self):
        """When both the sentinel and the _mx_to_numpy_safe autodetect produce
        a dtype, the explicit sentinel wins. This pins down the precedence so
        a future cast-chain (e.g. bf16 → fp16 → bf16 round-trip pre-snapshot)
        can't silently lose the original dtype."""

        class _BF16RaisingArray(MockMLXArray):
            def __init__(self, data):
                super().__init__(data)
                # Simulate the bf16 buffer-protocol RuntimeError so
                # _mx_to_numpy_safe reports "bfloat16" via autodetect; the
                # sentinel below should still win.
                self.dtype = type(
                    "dtype",
                    (),
                    {"size": 2, "__str__": lambda self: "mlx.core.bfloat16"},
                )()

            def __array__(self):
                raise RuntimeError("buffer format string mismatch")

            def astype(self, _dt):
                return MockMLXArray(self._data)

        keys = _BF16RaisingArray(np.random.randn(1, 4, 2, 8).astype(np.float32))
        values = _BF16RaisingArray(np.random.randn(1, 4, 2, 8).astype(np.float32))
        layer = MockKVCacheLayer(keys=keys, values=values, offset=2)
        layer._ssd_keys_original_dtype = "bfloat16"
        layer._ssd_values_original_dtype = "bfloat16"

        snapshot = KVCacheSerializer().snapshot_layer(layer)

        # Sentinel must win over the upcast-recovered hint either way; both
        # paths point to bfloat16 here, but the assertion guards that the
        # sentinel branch is the one actually consulted.
        assert snapshot["keys_original_dtype"] == "bfloat16"
        assert snapshot["values_original_dtype"] == "bfloat16"

    def test_snapshot_layer_no_sentinel_falls_back_to_autodetect(self):
        """Without a sentinel, snapshot_layer must keep the prior behavior:
        plain fp16/fp32 layers record no dtype hint (None), bf16 layers get
        the autodetected upcast dtype via _mx_to_numpy_safe. Guards against
        the sentinel logic eating the plain path."""
        keys = MockMLXArray(np.random.randn(1, 4, 4, 8).astype(np.float16))
        values = MockMLXArray(np.random.randn(1, 4, 4, 8).astype(np.float16))
        layer = MockKVCacheLayer(keys=keys, values=values, offset=4)

        snapshot = KVCacheSerializer().snapshot_layer(layer)

        assert "keys_original_dtype" not in snapshot
        assert "values_original_dtype" not in snapshot


from vllm_mlx.ssd_cache import SSDCacheTier


class TestSSDCacheTierCore:
    """Tests for SSDCacheTier initialization and directory setup."""

    def test_init_creates_directory(self, tmp_path):
        cache_dir = str(tmp_path / "ssd_tier_test")
        config = SSDCacheConfig(identity=IDENT, cache_dir=cache_dir)
        tier = SSDCacheTier(config)
        try:
            ns = os.path.join(cache_dir, IDENT.digest)
            assert tier.namespace_dir == ns
            assert os.path.isdir(cache_dir)
            assert os.path.exists(os.path.join(ns, "data"))
            assert os.path.exists(os.path.join(ns, "index.db"))
        finally:
            tier.close()

    def test_dir_permissions(self, tmp_path):
        cache_dir = str(tmp_path / "ssd_tier_perms")
        config = SSDCacheConfig(
            identity=IDENT, cache_dir=cache_dir, dir_permissions=0o700
        )
        tier = SSDCacheTier(config)
        try:
            stat = os.stat(os.path.join(tier.namespace_dir, "data"))
            # Check owner permissions (at least rwx for owner)
            assert stat.st_mode & 0o700 == 0o700
        finally:
            tier.close()

    def test_entry_hash_deterministic(self):
        tokens = (1, 2, 3, 4, 5)
        h1 = SSDCacheTier._entry_hash(tokens)
        h2 = SSDCacheTier._entry_hash(tokens)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex digest

    def test_entry_hash_different_for_different_tokens(self):
        h1 = SSDCacheTier._entry_hash((1, 2, 3))
        h2 = SSDCacheTier._entry_hash((1, 2, 4))
        assert h1 != h2

    def test_stats_initial(self, tmp_path):
        config = SSDCacheConfig(identity=IDENT, cache_dir=str(tmp_path / "stats_test"))
        tier = SSDCacheTier(config)
        try:
            stats = tier.get_stats()
            assert stats["spill_count"] == 0
            assert stats["ssd_hits"] == 0
        finally:
            tier.close()

    def test_remove_deletes_the_index_and_entry_directory(self, tmp_path):
        tokens = (1, 2, 3)
        config = SSDCacheConfig(identity=IDENT, cache_dir=str(tmp_path / "remove_test"))
        tier = SSDCacheTier(config)
        entry_hash = tier._entry_hash(tokens)
        entry_dir = os.path.join(tier._data_dir, entry_hash)
        os.makedirs(entry_dir)
        tier._index.insert_entry(tokens, entry_hash, memory_bytes=128, num_tokens=3)

        try:
            assert tier.remove(tokens) is True
            assert tier._index.lookup_exact(tokens) is None
            assert not os.path.exists(entry_dir)
            assert tier.remove(tokens) is False
        finally:
            tier.close()

    def test_close_idempotent(self, tmp_path):
        config = SSDCacheConfig(identity=IDENT, cache_dir=str(tmp_path / "close_test"))
        tier = SSDCacheTier(config)
        tier.close()
        tier.close()  # Should not raise

    def test_init_closes_index_when_later_setup_fails(self, tmp_path, monkeypatch):
        fake_index = type(
            "FakeIndex",
            (),
            {
                "closed": False,
                "close": lambda self: setattr(self, "closed", True),
            },
        )()

        monkeypatch.setattr(ssd_cache_module, "SSDIndex", lambda cache_dir: fake_index)

        def fail_queue(*args, **kwargs):
            raise RuntimeError("queue setup failed")

        monkeypatch.setattr(ssd_cache_module.queue, "Queue", fail_queue)

        config = SSDCacheConfig(identity=IDENT, cache_dir=str(tmp_path / "init_fail"))
        with pytest.raises(RuntimeError, match="queue setup failed"):
            SSDCacheTier(config)

        assert fake_index.closed

    def test_init_preserves_setup_failure_when_index_close_fails(
        self, tmp_path, monkeypatch
    ):
        class FakeIndex:
            def close(self):
                raise RuntimeError("index close failed")

        monkeypatch.setattr(ssd_cache_module, "SSDIndex", lambda cache_dir: FakeIndex())

        def fail_queue(*args, **kwargs):
            raise RuntimeError("queue setup failed")

        monkeypatch.setattr(ssd_cache_module.queue, "Queue", fail_queue)

        config = SSDCacheConfig(
            identity=IDENT, cache_dir=str(tmp_path / "init_close_fail")
        )
        with pytest.raises(RuntimeError, match="queue setup failed"):
            SSDCacheTier(config)


import time


class TestAsyncSpillWriter:
    """Tests for the async spill writer thread."""

    @pytest.fixture
    def tier_with_writer(self, tmp_path):
        config = SSDCacheConfig(identity=IDENT, cache_dir=str(tmp_path / "writer_test"))
        tier = SSDCacheTier(config)
        tier.start_writer()
        yield tier
        tier.close()

    def test_spill_writes_entry_to_disk(self, tier_with_writer):
        tokens = (1, 2, 3, 4, 5)
        keys = MockMLXArray(np.random.randn(1, 8, 16, 64).astype(np.float16))
        values = MockMLXArray(np.random.randn(1, 8, 16, 64).astype(np.float16))
        layer = MockKVCacheLayer(keys=keys, values=values, offset=16)
        cache = [layer]

        tier_with_writer.enqueue_spill(tokens, cache, memory_bytes=2048)

        # Wait for async write to complete (with timeout)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if tier_with_writer._stats.spill_count > 0:
                break
            time.sleep(0.05)

        assert tier_with_writer._stats.spill_count == 1
        assert tier_with_writer._stats.spill_bytes > 0

        # Verify entry in index
        result = tier_with_writer._index.lookup_exact(tokens)
        assert result is not None
        assert result["num_tokens"] == 5

    def test_spill_atomic_write(self, tier_with_writer):
        """Verify no partial files exist after spill."""
        tokens = (10, 20, 30)
        keys = MockMLXArray(np.random.randn(1, 4, 8, 32).astype(np.float16))
        values = MockMLXArray(np.random.randn(1, 4, 8, 32).astype(np.float16))
        layer = MockKVCacheLayer(keys=keys, values=values, offset=8)

        tier_with_writer.enqueue_spill(tokens, [layer], memory_bytes=1024)

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if tier_with_writer._stats.spill_count > 0:
                break
            time.sleep(0.05)

        # Check that no .tmp files remain
        data_dir = tier_with_writer._data_dir
        for root, dirs, files in os.walk(data_dir):
            for f in files:
                assert not f.endswith(".tmp"), f"Temp file left behind: {f}"

    def test_spill_queue_full_drops(self, tmp_path):
        """When queue is full, new spills are dropped (not blocking)."""
        config = SSDCacheConfig(
            identity=IDENT,
            cache_dir=str(tmp_path / "queue_full_test"),
            spill_queue_size=2,
        )
        tier = SSDCacheTier(config)
        # Don't start writer — queue will fill up
        keys = MockMLXArray(np.zeros((1, 1, 1, 1), dtype=np.float16))
        values = MockMLXArray(np.zeros((1, 1, 1, 1), dtype=np.float16))
        layer = MockKVCacheLayer(keys=keys, values=values, offset=1)

        # Enqueue more than capacity
        for i in range(5):
            tier.enqueue_spill((i,), [layer], memory_bytes=100)

        # Queue should be at capacity, not have raised
        assert tier._spill_queue.qsize() <= 2
        tier.close()


from unittest.mock import MagicMock
from vllm_mlx.memory_cache import MemoryAwarePrefixCache, MemoryCacheConfig


class MockKVCacheForSpill:
    """Mock KVCache with keys, values, offset for memory estimation."""

    def __init__(self, size_bytes: int):
        # Use real numpy arrays so serialization works
        self.keys = MockMLXArray(np.zeros((1, 4, 8, 16), dtype=np.float16))
        self.values = MockMLXArray(np.zeros((1, 4, 8, 16), dtype=np.float16))
        self.offset = 8


class TestSpillPath:
    """Tests for eviction -> SSD spill integration."""

    def test_evict_lru_calls_ssd_spill(self, tmp_path):
        model = MagicMock()
        # min_prefix_tokens=1: test uses 10-token sequences (prod default 128
        # would silently reject every store()).
        config = MemoryCacheConfig(max_memory_mb=1, max_entries=3, min_prefix_tokens=1)
        cache = MemoryAwarePrefixCache(model, config)

        ssd_config = SSDCacheConfig(
            identity=IDENT, cache_dir=str(tmp_path / "spill_test")
        )
        ssd_tier = SSDCacheTier(ssd_config)
        ssd_tier.start_writer()
        cache.set_ssd_tier(ssd_tier)

        # Fill cache to capacity (3 entries)
        for i in range(3):
            kv = [MockKVCacheForSpill(1000)]
            cache.store(list(range(i * 10, (i + 1) * 10)), kv)

        assert len(cache) == 3

        # Store one more to trigger eviction
        kv = [MockKVCacheForSpill(1000)]
        cache.store(list(range(100, 110)), kv)

        # Wait for spill
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if ssd_tier._stats.spill_count > 0:
                break
            time.sleep(0.05)

        assert ssd_tier._stats.spill_count >= 1
        ssd_tier.close()

    def test_evict_without_ssd_tier_still_works(self):
        """Eviction without SSD tier should work as before (discard)."""
        model = MagicMock()
        config = MemoryCacheConfig(max_memory_mb=1, max_entries=2, min_prefix_tokens=1)
        cache = MemoryAwarePrefixCache(model, config)

        for i in range(5):
            kv = [MockKVCacheForSpill(1000)]
            cache.store(list(range(i * 10, (i + 1) * 10)), kv)

        # Should not raise, just discard
        assert len(cache) <= 2

    def test_snapshot_runs_on_caller_thread(self, tmp_path):
        """Regression: enqueue_spill must materialize layers on the caller
        thread (real mx.array hard-aborts the process if accessed off the
        Stream(gpu, N) it was created on)."""
        import threading as _threading

        creator_thread_id = _threading.get_ident()

        class ThreadBoundArray:
            # Python-level proxy for the MLX cross-thread Stream abort.
            def __init__(self, data, owner_tid):
                self._data = data
                self._owner_tid = owner_tid

            def __array__(self):
                if _threading.get_ident() != self._owner_tid:
                    raise RuntimeError("accessed from wrong thread")
                return self._data

        class ThreadBoundKV:
            def __init__(self, tid):
                self.keys = ThreadBoundArray(
                    np.zeros((1, 1, 1, 1), dtype=np.float16), tid
                )
                self.values = ThreadBoundArray(
                    np.zeros((1, 1, 1, 1), dtype=np.float16), tid
                )
                self.offset = 1

        ssd_config = SSDCacheConfig(
            identity=IDENT, cache_dir=str(tmp_path / "snapshot_test")
        )
        ssd_tier = SSDCacheTier(ssd_config)
        ssd_tier.start_writer()

        try:
            tokens = tuple(range(0, 200))
            enqueued = ssd_tier.enqueue_spill(
                tokens,
                [ThreadBoundKV(creator_thread_id)],
                memory_bytes=64,
            )
            assert enqueued is True

            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                if ssd_tier._stats.spill_count >= 1:
                    break
                time.sleep(0.05)
            assert ssd_tier._stats.spill_count == 1
        finally:
            ssd_tier.close()


import asyncio


class TestAsyncFetchPath:
    """Tests for async SSD fetch with RAM budget reservation."""

    @pytest.fixture
    def populated_tier(self, tmp_path):
        """Create an SSD tier with one pre-written entry."""
        config = SSDCacheConfig(identity=IDENT, cache_dir=str(tmp_path / "fetch_test"))
        tier = SSDCacheTier(config)
        tier.start_writer()

        tokens = (1, 2, 3, 4, 5)
        keys = MockMLXArray(np.random.randn(1, 8, 5, 64).astype(np.float16))
        values = MockMLXArray(np.random.randn(1, 8, 5, 64).astype(np.float16))
        layer = MockKVCacheLayer(keys=keys, values=values, offset=5)

        tier.enqueue_spill(tokens, [layer], memory_bytes=2048)

        # Wait for write
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if tier._stats.spill_count > 0:
                break
            time.sleep(0.05)

        yield tier, tokens, keys, values
        tier.close()

    def test_lookup_ssd_returns_candidate(self, populated_tier):
        tier, tokens, _, _ = populated_tier
        candidate = tier.lookup_ssd(tokens)
        assert candidate is not None
        assert candidate["num_tokens"] == 5
        assert candidate["memory_bytes"] == 2048

    def test_lookup_ssd_miss(self, populated_tier):
        tier, _, _, _ = populated_tier
        candidate = tier.lookup_ssd((99, 98, 97))
        assert candidate is None

    def test_async_promote_reserves_budget_then_reads(self, populated_tier):
        """RAM budget is reserved BEFORE the SSD read completes."""
        tier, tokens, keys, values = populated_tier

        budget_reserved = []
        budget_released = []

        def reserve_fn(nbytes):
            budget_reserved.append(nbytes)
            return True  # Budget available

        def release_fn(nbytes):
            budget_released.append(nbytes)

        result = asyncio.run(tier.async_promote(tokens, reserve_fn, release_fn))

        # Budget was reserved
        assert len(budget_reserved) == 1
        assert budget_reserved[0] == 2048

        # Result contains the cache layers
        assert result is not None
        assert len(result) == 1  # One layer

        # Stats updated
        assert tier._stats.ssd_hits == 1
        assert tier._stats.reload_bytes > 0

    def test_async_promote_budget_denied(self, populated_tier):
        """When budget reservation fails, promote returns None."""
        tier, tokens, _, _ = populated_tier

        def reserve_fn(nbytes):
            return False  # Budget denied

        def release_fn(nbytes):
            pass

        result = asyncio.run(tier.async_promote(tokens, reserve_fn, release_fn))

        assert result is None
        assert tier._stats.promotion_failures == 1

    def test_async_promote_read_failure_releases_budget(self, populated_tier):
        """If disk read fails after reservation, budget is released."""
        tier, tokens, _, _ = populated_tier

        budget_reserved = []
        budget_released = []

        def reserve_fn(nbytes):
            budget_reserved.append(nbytes)
            return True

        def release_fn(nbytes):
            budget_released.append(nbytes)

        # Corrupt the entry on disk
        entry_hash = tier._entry_hash(tokens)
        entry_dir = os.path.join(tier._data_dir, entry_hash)
        manifest_path = os.path.join(entry_dir, "manifest.json")
        with open(manifest_path, "w") as f:
            f.write("corrupted!")

        result = asyncio.run(tier.async_promote(tokens, reserve_fn, release_fn))

        assert result is None
        # Budget was reserved then released
        assert len(budget_reserved) == 1
        assert len(budget_released) == 1
        assert budget_released[0] == budget_reserved[0]


class TestCapacityManagement:
    """Tests for SSD disk capacity management."""

    @pytest.fixture
    def small_tier(self, tmp_path):
        """SSD tier with very small capacity for testing eviction."""
        config = SSDCacheConfig(
            identity=IDENT,
            cache_dir=str(tmp_path / "capacity_test"),
            max_size_gb=0.0001,  # ~100KB
            max_entries=3,
        )
        tier = SSDCacheTier(config)
        tier.start_writer()
        yield tier
        tier.close()

    def _write_and_wait(self, tier, tokens, size=1024):
        keys = MockMLXArray(np.random.randn(1, 4, 8, 32).astype(np.float16))
        values = MockMLXArray(np.random.randn(1, 4, 8, 32).astype(np.float16))
        layer = MockKVCacheLayer(keys=keys, values=values, offset=8)
        tier.enqueue_spill(tokens, [layer], memory_bytes=size)

        deadline = time.monotonic() + 5.0
        initial_count = tier._stats.spill_count
        while time.monotonic() < deadline:
            if tier._stats.spill_count > initial_count:
                break
            time.sleep(0.05)

    def test_disk_lru_eviction(self, small_tier):
        """When disk capacity is exceeded, oldest entries are evicted."""
        # Fill with 3 entries (at max_entries limit)
        for i in range(3):
            self._write_and_wait(small_tier, tuple(range(i * 10, (i + 1) * 10)))

        assert small_tier._index.get_entry_count() == 3

        # Write one more — should trigger disk LRU eviction
        self._write_and_wait(small_tier, (100, 101, 102))

        # Should still be at or below max_entries
        assert small_tier._index.get_entry_count() <= 3

    def test_startup_reconciliation(self, tmp_path):
        """On startup, index is reconciled with files on disk."""
        cache_dir = str(tmp_path / "reconcile_test")
        config = SSDCacheConfig(identity=IDENT, cache_dir=cache_dir)
        tier = SSDCacheTier(config)
        tier.start_writer()

        # Write an entry
        tokens = (1, 2, 3)
        keys = MockMLXArray(np.random.randn(1, 4, 3, 32).astype(np.float16))
        values = MockMLXArray(np.random.randn(1, 4, 3, 32).astype(np.float16))
        layer = MockKVCacheLayer(keys=keys, values=values, offset=3)
        tier.enqueue_spill(tokens, [layer], memory_bytes=1024)

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if tier._stats.spill_count > 0:
                break
            time.sleep(0.05)

        tier.close()

        # Delete the data file but leave the index entry
        entry_hash = SSDCacheTier._entry_hash(tokens)
        entry_dir = os.path.join(tier.namespace_dir, "data", entry_hash)
        import shutil

        shutil.rmtree(entry_dir)

        # Re-open — reconciliation should detect the missing file
        tier2 = SSDCacheTier(config)
        tier2.reconcile()

        # The orphaned index entry should have been removed
        assert tier2._index.lookup_exact(tokens) is None
        tier2.close()


class TestCLIIntegration:
    """Tests for CLI argument parsing of SSD cache flags."""

    def test_scheduler_config_has_ssd_fields(self):
        from vllm_mlx.scheduler import SchedulerConfig

        config = SchedulerConfig()
        assert config.ssd_cache_dir is None
        assert config.ssd_cache_max_gb == 10.0

    def test_scheduler_config_custom_ssd(self):
        from vllm_mlx.scheduler import SchedulerConfig

        config = SchedulerConfig(
            ssd_cache_dir="/tmp/test-ssd",
            ssd_cache_max_gb=5.0,
        )
        assert config.ssd_cache_dir == "/tmp/test-ssd"
        assert config.ssd_cache_max_gb == 5.0


class TestMemoryCacheSSDCheck:
    """Tests for MemoryAwarePrefixCache.check_ssd() method."""

    def test_check_ssd_returns_candidate_on_miss(self, tmp_path):
        model = MagicMock()
        config = MemoryCacheConfig(max_memory_mb=1, max_entries=2, min_prefix_tokens=1)
        cache = MemoryAwarePrefixCache(model, config)

        ssd_config = SSDCacheConfig(
            identity=IDENT, cache_dir=str(tmp_path / "check_ssd_test")
        )
        ssd_tier = SSDCacheTier(ssd_config)
        ssd_tier.start_writer()
        cache.set_ssd_tier(ssd_tier)

        # Write an entry to SSD directly
        tokens = (1, 2, 3, 4, 5)
        keys = MockMLXArray(np.random.randn(1, 4, 8, 32).astype(np.float16))
        values = MockMLXArray(np.random.randn(1, 4, 8, 32).astype(np.float16))
        layer = MockKVCacheLayer(keys=keys, values=values, offset=8)
        ssd_tier.enqueue_spill(tokens, [layer], memory_bytes=1024)

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if ssd_tier._stats.spill_count > 0:
                break
            time.sleep(0.05)

        # RAM fetch should miss
        result, remaining = cache.fetch(list(tokens))
        assert result is None

        # But SSD check should find it
        candidate = cache.check_ssd(list(tokens))
        assert candidate is not None
        assert candidate["num_tokens"] == 5

        ssd_tier.close()

    def test_check_ssd_returns_none_without_tier(self):
        model = MagicMock()
        config = MemoryCacheConfig(max_memory_mb=1, min_prefix_tokens=1)
        cache = MemoryAwarePrefixCache(model, config)

        candidate = cache.check_ssd([1, 2, 3])
        assert candidate is None

    def test_check_ssd_returns_none_on_ram_hit(self, tmp_path):
        """When RAM has a hit, check_ssd should return None (not needed)."""
        model = MagicMock()
        config = MemoryCacheConfig(max_memory_mb=1, min_prefix_tokens=1)
        cache = MemoryAwarePrefixCache(model, config)

        kv = [MockKVCacheForSpill(1000)]
        cache.store([1, 2, 3], kv)

        # RAM hit exists
        result, _ = cache.fetch([1, 2, 3])
        assert result is not None

        # check_ssd should indicate no SSD needed
        candidate = cache.check_ssd([1, 2, 3])
        assert candidate is None


class TestIntegrationSpillAndFetch:
    """End-to-end tests: spill from RAM -> SSD -> promote back."""

    @pytest.fixture
    def cache_with_ssd(self, tmp_path):
        """RAM cache + SSD tier, small limits to force eviction."""
        model = MagicMock()
        config = MemoryCacheConfig(max_memory_mb=1, max_entries=2, min_prefix_tokens=1)
        ram_cache = MemoryAwarePrefixCache(model, config)

        ssd_config = SSDCacheConfig(
            identity=IDENT,
            cache_dir=str(tmp_path / "integration_test"),
            max_size_gb=1.0,
        )
        ssd_tier = SSDCacheTier(ssd_config)
        ssd_tier.start_writer()
        ram_cache.set_ssd_tier(ssd_tier)

        yield ram_cache, ssd_tier
        ssd_tier.close()

    def _make_kv_layer(self, offset=10):
        keys = MockMLXArray(np.random.randn(1, 8, offset, 64).astype(np.float16))
        values = MockMLXArray(np.random.randn(1, 8, offset, 64).astype(np.float16))
        return MockKVCacheLayer(keys=keys, values=values, offset=offset)

    def test_full_round_trip(self, cache_with_ssd):
        """Store -> evict to SSD -> fetch miss in RAM -> find in SSD."""
        ram_cache, ssd_tier = cache_with_ssd

        # Store 2 entries (fills RAM cache)
        tokens_a = list(range(0, 10))
        tokens_b = list(range(10, 20))
        ram_cache.store(tokens_a, [self._make_kv_layer()])
        ram_cache.store(tokens_b, [self._make_kv_layer()])
        assert len(ram_cache) == 2

        # Store a third entry — should evict tokens_a to SSD
        tokens_c = list(range(20, 30))
        ram_cache.store(tokens_c, [self._make_kv_layer()])

        # Wait for SSD spill
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if ssd_tier._stats.spill_count > 0:
                break
            time.sleep(0.05)

        assert ssd_tier._stats.spill_count >= 1

        # RAM miss for tokens_a
        result, remaining = ram_cache.fetch(tokens_a)
        assert result is None

        # SSD should have tokens_a
        candidate = ram_cache.check_ssd(tokens_a)
        assert candidate is not None

        # Promote from SSD
        async def do_promote():
            return await ssd_tier.async_promote(
                tuple(tokens_a),
                lambda n: True,  # Budget always available
                lambda n: None,
            )

        promoted = asyncio.run(do_promote())
        assert promoted is not None
        assert len(promoted) == 1  # One layer
        assert ssd_tier._stats.ssd_hits == 1

    def test_hybrid_cache_round_trip(self, tmp_path):
        """ArraysCache layers survive SSD round-trip."""
        config = SSDCacheConfig(identity=IDENT, cache_dir=str(tmp_path / "hybrid_test"))
        tier = SSDCacheTier(config)
        tier.start_writer()

        tokens = (1, 2, 3)
        arr0 = MockMLXArray(np.random.randn(1, 64, 128).astype(np.float16))
        arr1 = MockMLXArray(np.random.randn(1, 64, 128).astype(np.float16))
        layer = MockArraysCacheLayer(state=[arr0, arr1])

        tier.enqueue_spill(tokens, [layer], memory_bytes=2048)

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if tier._stats.spill_count > 0:
                break
            time.sleep(0.05)

        async def do_promote():
            return await tier.async_promote(
                tokens,
                lambda n: True,
                lambda n: None,
            )

        promoted = asyncio.run(do_promote())
        assert promoted is not None
        assert len(promoted) == 1
        assert "state" in promoted[0]
        assert len(promoted[0]["state"]) == 2

        # Verify data integrity
        np.testing.assert_array_almost_equal(
            np.array(promoted[0]["state"][0]),
            np.array(arr0),
        )

        tier.close()

    def test_scheduler_reconstructs_real_arrays_cache(self):
        cache_mod = pytest.importorskip("mlx_lm.models.cache")
        mx = pytest.importorskip("mlx.core")
        from vllm_mlx.scheduler import Scheduler

        scheduler = object.__new__(Scheduler)
        reconstructed = Scheduler._reconstruct_ssd_layers(
            scheduler,
            [{"state": [np.array([1, 2], dtype=np.float32)]}],
        )

        assert reconstructed is not None
        assert isinstance(reconstructed[0], cache_mod.ArraysCache)
        assert reconstructed[0].cache[0].tolist() == mx.array([1, 2]).tolist()

    def test_capacity_eviction_end_to_end(self, tmp_path):
        """Entries beyond max_entries are evicted from SSD."""
        config = SSDCacheConfig(
            identity=IDENT,
            cache_dir=str(tmp_path / "cap_e2e"),
            max_entries=2,
        )
        tier = SSDCacheTier(config)
        tier.start_writer()

        for i in range(4):
            tokens = tuple(range(i * 10, (i + 1) * 10))
            layer = MockKVCacheLayer(
                keys=MockMLXArray(np.zeros((1, 1, 1, 1), dtype=np.float16)),
                values=MockMLXArray(np.zeros((1, 1, 1, 1), dtype=np.float16)),
                offset=1,
            )
            tier.enqueue_spill(tokens, [layer], memory_bytes=100)

            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                if tier._stats.spill_count > i:
                    break
                time.sleep(0.05)

        # Only max_entries (2) should remain
        assert tier._index.get_entry_count() <= 2

        # The latest entries should still be there
        assert tier._index.lookup_exact(tuple(range(30, 40))) is not None

        tier.close()

    def test_metrics_accuracy(self, cache_with_ssd):
        """Verify all SSD metrics are updated correctly."""
        ram_cache, ssd_tier = cache_with_ssd

        # Fill RAM and force one spill
        ram_cache.store(list(range(10)), [self._make_kv_layer()])
        ram_cache.store(list(range(10, 20)), [self._make_kv_layer()])
        ram_cache.store(list(range(20, 30)), [self._make_kv_layer()])

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if ssd_tier._stats.spill_count > 0:
                break
            time.sleep(0.05)

        stats = ssd_tier.get_stats()
        assert stats["spill_count"] >= 1
        assert stats["spill_bytes"] > 0

        # Promote the spilled entry
        evicted_tokens = tuple(range(10))  # First stored, first evicted
        candidate = ssd_tier.lookup_ssd(evicted_tokens)
        if candidate is not None:

            async def promote():
                return await ssd_tier.async_promote(
                    evicted_tokens, lambda n: True, lambda n: None
                )

            asyncio.run(promote())
            stats = ssd_tier.get_stats()
            assert stats["ssd_hits"] >= 1
            assert stats["reload_bytes"] > 0
            assert stats["avg_reload_latency_ms"] > 0


class TestCacheListSpillRoundTrip:
    """SSD spill must round-trip GLM-5.2 / DeepSeek-V3.2 DSA caches.

    A DSA layer is ``CacheList(KVCache latent, KVCache indexer)`` (full layers)
    or ``CacheList(KVCache latent)`` (shared layers). The indexer sub-cache is
    empty below the sparse ``index_topk`` threshold. ``CacheList`` exposes a
    list ``.state``, so the duck-typed dispatcher used to mis-route it to
    ``ArraysCacheSerializer``; iterating ``.state`` then yields per-sub
    ``(keys, values)`` tuples and ``_mx_to_numpy_safe`` choked
    (``'tuple' object has no attribute 'dtype'``), dropping every entry. A
    dedicated ``CacheListSerializer`` round-trips them.
    """

    def _make_dsa_cache(self):
        import mlx.core as mx
        from mlx_lm.models.cache import KVCache, CacheList

        # Full layer: populated bf16 latent + empty (dense-mode) indexer.
        lat = KVCache()
        lat.update_and_fetch(
            mx.random.normal((1, 4, 6, 16)).astype(mx.bfloat16),
            mx.random.normal((1, 4, 6, 16)).astype(mx.bfloat16),
        )
        idx = KVCache()
        idx.keys = mx.zeros((1, 4, 0, 16), dtype=mx.float32)
        idx.values = mx.zeros((1, 4, 0, 16), dtype=mx.float32)
        idx.offset = 0
        full = CacheList(lat, idx)

        # Shared layer: single latent sub-cache.
        lat2 = KVCache()
        lat2.update_and_fetch(
            mx.random.normal((1, 4, 6, 16)).astype(mx.bfloat16),
            mx.random.normal((1, 4, 6, 16)).astype(mx.bfloat16),
        )
        shared = CacheList(lat2)
        return [full, shared], lat, lat2

    def test_dispatch_picks_cachelist_serializer(self):
        from vllm_mlx.ssd_cache import (
            get_serializer_for_layer,
            CacheListSerializer,
            ArraysCacheSerializer,
        )

        cache, _, _ = self._make_dsa_cache()
        s = get_serializer_for_layer(cache[0])
        assert isinstance(s, CacheListSerializer)
        assert not isinstance(s, ArraysCacheSerializer)

    def test_cachelist_in_support_matrix(self):
        from vllm_mlx.ssd_cache import SERIALIZER_SUPPORT_MATRIX

        assert "CacheList" in SERIALIZER_SUPPORT_MATRIX

    def test_snapshot_serialize_deserialize_round_trip(self, tmp_path):
        from vllm_mlx.ssd_cache import get_serializer_for_layer

        cache, lat, lat2 = self._make_dsa_cache()
        ser = get_serializer_for_layer(cache[0])
        path = str(tmp_path / "layer_0.safetensors")

        snap = ser.snapshot_layer(cache[0])
        meta = ser.serialize_layer(snap, 0, path)
        assert meta["layer_type"] == "CacheList"
        assert meta["sub_count"] == 2
        assert os.path.exists(path)

        ld = ser.deserialize_layer(path, meta)
        assert "cachelist_subs" in ld
        subs = ld["cachelist_subs"]
        assert len(subs) == 2
        # Latent: offset + data preserved.
        assert subs[0]["offset"] == 6
        # Empty indexer preserved.
        assert subs[1]["offset"] == 0
        assert np.array(subs[1]["keys"]).size == 0

    def test_full_roundtrip_reconstructs_cachelist(self, tmp_path):
        import types
        import mlx.core as mx
        from vllm_mlx.ssd_cache import get_serializer_for_layer
        from vllm_mlx.scheduler import Scheduler

        cache, lat, lat2 = self._make_dsa_cache()

        # snapshot -> serialize -> deserialize each layer (what _read_entry does)
        dicts = []
        for i, layer in enumerate(cache):
            ser = get_serializer_for_layer(layer)
            path = str(tmp_path / f"layer_{i}.safetensors")
            snap = ser.snapshot_layer(layer)
            meta = ser.serialize_layer(snap, i, path)
            dicts.append(ser.deserialize_layer(path, meta))

        # reconstruct (scheduler) — method only touches module globals, so a
        # dummy self is sufficient.
        result = Scheduler._reconstruct_ssd_layers(types.SimpleNamespace(), dicts)
        assert result is not None
        assert [type(c).__name__ for c in result] == ["CacheList", "CacheList"]
        assert [len(c.caches) for c in result] == [2, 1]

        # Latent: dtype, offset, and data round-trip (bf16 -> fp32 -> bf16).
        r_lat = result[0].caches[0]
        assert str(r_lat.keys.dtype).endswith("bfloat16")
        assert r_lat.offset == 6
        assert bool(
            mx.allclose(
                r_lat.keys[..., :6, :].astype(mx.float32),
                lat.keys[..., :6, :].astype(mx.float32),
            )
        )
        # Empty indexer preserved.
        assert result[0].caches[1].offset == 0
        assert result[0].caches[1].keys.size == 0
        # Shared-layer latent round-trips too.
        assert result[1].caches[0].offset == 6
        assert bool(
            mx.allclose(
                result[1].caches[0].keys[..., :6, :].astype(mx.float32),
                lat2.keys[..., :6, :].astype(mx.float32),
            )
        )

    def test_plain_kvcache_still_dispatches_to_kvcache(self):
        """Regression: non-DSA plain KVCache must NOT hit the CacheList path."""
        import mlx.core as mx
        from mlx_lm.models.cache import KVCache
        from vllm_mlx.ssd_cache import get_serializer_for_layer, KVCacheSerializer

        layer = KVCache()
        layer.update_and_fetch(
            mx.random.normal((1, 4, 5, 16)), mx.random.normal((1, 4, 5, 16))
        )
        assert isinstance(get_serializer_for_layer(layer), KVCacheSerializer)


class TestSchemaVersionInvalidation:
    """A schema bump must drop entries written by an incompatible build.

    Motivating case: before CacheList got a dedicated serializer, a GLM-5.2/5.3
    DSA layer was mis-dispatched to ArraysCacheSerializer. With bf16 that raised
    and the entry was dropped, but with fp16 ``np.array((keys, values))`` stacks
    cleanly, so the spill was *written* -- tagged ``layer_type: "ArraysCache"``
    and missing the per-sub offsets. ``_entry_hash`` is the token hash alone and
    the cache dir is not build-namespaced, so such an entry still matches after
    the upgrade and reconstructs into an ArraysCache the DSA model cannot use.
    """

    @staticmethod
    def _seed(cache_dir, version):
        import sqlite3
        import time

        from vllm_mlx import ssd_cache as ssd_cache_module

        data_dir = os.path.join(cache_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        tokens = (1, 2, 3)
        entry_dir = os.path.join(data_dir, ssd_cache_module._tokens_hash(tokens))
        os.makedirs(entry_dir, exist_ok=True)
        with open(os.path.join(entry_dir, "manifest.json"), "w") as fh:
            fh.write('{"layers": [{"layer_type": "ArraysCache"}]}')

        conn = sqlite3.connect(os.path.join(cache_dir, "index.db"))
        conn.executescript("""
            CREATE TABLE schema_version (version INTEGER NOT NULL);
            CREATE TABLE entries (
                token_hash   TEXT PRIMARY KEY,
                tokens_blob  BLOB NOT NULL,
                prefix_hash  TEXT,
                num_tokens   INTEGER NOT NULL,
                file_path    TEXT NOT NULL,
                memory_bytes INTEGER NOT NULL,
                created_at   REAL NOT NULL,
                accessed_at  REAL NOT NULL
            );
            """)
        conn.execute("INSERT INTO schema_version (version) VALUES (?)", (version,))
        now = time.time()
        conn.execute(
            "INSERT INTO entries VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                ssd_cache_module._tokens_hash(tokens),
                ssd_cache_module._tokens_to_blob(tokens),
                ssd_cache_module._prefix_hash(tokens),
                len(tokens),
                ssd_cache_module._tokens_hash(tokens),
                1000,
                now,
                now,
            ),
        )
        conn.commit()
        conn.close()
        return tokens, entry_dir

    def test_older_schema_version_drops_every_entry(self, tmp_path):
        cache_dir = str(tmp_path / "cache")
        tokens, _ = self._seed(cache_dir, SSDIndex._SCHEMA_VERSION - 1)

        idx = SSDIndex(cache_dir)
        try:
            assert idx.migrated_from == SSDIndex._SCHEMA_VERSION - 1
            assert idx.get_entry_count() == 0
            assert idx.lookup_exact(tokens) is None
            with idx._db_lock:
                cur = idx._conn.execute("SELECT version FROM schema_version")
                assert cur.fetchone()[0] == SSDIndex._SCHEMA_VERSION
        finally:
            idx.close()

    def test_current_schema_version_keeps_entries(self, tmp_path):
        cache_dir = str(tmp_path / "cache")
        tokens, _ = self._seed(cache_dir, SSDIndex._SCHEMA_VERSION)

        idx = SSDIndex(cache_dir)
        try:
            assert idx.migrated_from is None
            assert idx.lookup_exact(tokens) is not None
        finally:
            idx.close()

    def test_tier_removes_stale_data_dirs_on_migration(self, tmp_path):
        from vllm_mlx.ssd_cache import SSDCacheTier

        cache_dir = str(tmp_path / "cache")
        namespace = os.path.join(cache_dir, IDENT.digest)
        _, entry_dir = self._seed(namespace, SSDIndex._SCHEMA_VERSION - 1)
        assert os.path.isdir(entry_dir)

        tier = SSDCacheTier(SSDCacheConfig(identity=IDENT, cache_dir=cache_dir))
        try:
            assert not os.path.exists(
                entry_dir
            ), "stale spill files must not survive a schema bump"
            assert os.path.isdir(os.path.join(namespace, "data"))
        finally:
            tier.close()

    def test_fresh_cache_dir_is_not_treated_as_a_migration(self, tmp_path):
        cache_dir = str(tmp_path / "cache")
        os.makedirs(cache_dir, exist_ok=True)
        idx = SSDIndex(cache_dir)
        try:
            assert idx.migrated_from is None
        finally:
            idx.close()


class TestUnsupportedSubCaches:
    """A layer that cannot be spilled must fail loudly, not write junk.

    ``snapshot_layer`` runs on the *writer* thread, where ``_writer_loop``
    catches and logs. A guard that raises there costs one dropped entry and a
    log line. Getting past the guard is worse: ``np.array(None)`` is a 0-d
    object array whose ``.size`` is 1, so it slips past the ``keys_empty``
    branch and only fails later in ``save_file`` ("dtype object is not
    covered") -- same dropped entry, but with an orphaned .tmp directory and a
    much less obvious message.
    """

    def _cache_list(self, *subs):
        from mlx_lm.models.cache import CacheList

        return CacheList(*subs)

    def _populated_kv(self):
        import mlx.core as mx
        from mlx_lm.models.cache import KVCache

        kv = KVCache()
        kv.update_and_fetch(
            mx.random.normal((1, 2, 4, 8)), mx.random.normal((1, 2, 4, 8))
        )
        return kv

    def test_unpopulated_sub_cache_raises(self):
        """KVCache leaves keys/values None until the first update_and_fetch."""
        from mlx_lm.models.cache import KVCache
        from vllm_mlx.ssd_cache import CacheListSerializer

        layer = self._cache_list(self._populated_kv(), KVCache())
        with pytest.raises(ValueError, match="not materialized"):
            CacheListSerializer().snapshot_layer(layer)

    def test_unpopulated_plain_kvcache_raises(self):
        """Same guard protects a non-CacheList layer -- one invariant, one place."""
        from mlx_lm.models.cache import KVCache
        from vllm_mlx.ssd_cache import KVCacheSerializer

        with pytest.raises(ValueError, match="not materialized"):
            KVCacheSerializer().snapshot_layer(KVCache())

    def test_empty_but_materialized_sub_cache_is_still_supported(self):
        """Regression guard: the DSA indexer is size-0, NOT None.

        The None check must not start rejecting the empty dense-mode indexer,
        which is a normal, spillable state.
        """
        import mlx.core as mx
        from mlx_lm.models.cache import KVCache
        from vllm_mlx.ssd_cache import CacheListSerializer

        idx = KVCache()
        idx.keys = mx.zeros((1, 2, 0, 8), dtype=mx.float32)
        idx.values = mx.zeros((1, 2, 0, 8), dtype=mx.float32)
        idx.offset = 0

        snap = CacheListSerializer().snapshot_layer(
            self._cache_list(self._populated_kv(), idx)
        )
        assert len(snap["sub_snapshots"]) == 2
        assert snap["sub_snapshots"][1]["keys_np"].size == 0

    def test_non_kvcache_sub_is_rejected_by_type(self):
        """A sub-cache without the KVCache shape has no serializer at all."""
        from vllm_mlx.ssd_cache import CacheListSerializer

        class Alien:
            pass

        layer = self._cache_list(self._populated_kv(), Alien())
        with pytest.raises(ValueError, match="not.*KVCache-like|unsupported"):
            CacheListSerializer().snapshot_layer(layer)

    def test_rejected_layer_writes_no_file(self, tmp_path):
        """The guard fires before serialize_layer, so nothing lands on disk."""
        from mlx_lm.models.cache import KVCache
        from vllm_mlx.ssd_cache import CacheListSerializer

        ser = CacheListSerializer()
        path = str(tmp_path / "layer_0.safetensors")
        with pytest.raises(ValueError):
            snap = ser.snapshot_layer(self._cache_list(KVCache()))
            ser.serialize_layer(snap, 0, path)
        assert not os.path.exists(path)


# --- identity namespacing, flush and read-time validation ---------------------------------


def _kv_layer(n, marker=1.0):
    """A KV layer whose contents say which tokens it was computed for."""
    keys = MockMLXArray(np.full((1, 2, n, 8), marker, dtype=np.float16))
    values = MockMLXArray(np.full((1, 2, n, 8), marker + 0.5, dtype=np.float16))
    return MockKVCacheLayer(keys=keys, values=values, offset=n)


def _spill(tier, tokens, marker=1.0):
    before = tier._stats.spill_count
    assert tier.enqueue_spill(tuple(tokens), [_kv_layer(len(tokens), marker)], 1024)
    deadline = time.monotonic() + 5.0
    while tier._stats.spill_count == before and time.monotonic() < deadline:
        time.sleep(0.02)
    assert tier._stats.spill_count == before + 1


def _tier(base, identity=IDENT, **kw):
    tier = SSDCacheTier(SSDCacheConfig(cache_dir=str(base), identity=identity, **kw))
    tier.start_writer()
    return tier


class TestIdentityNamespacing:
    """One --ssd-cache-dir shared by different models or configurations."""

    def test_another_identity_never_sees_the_entries(self, tmp_path):
        other = CacheIdentity(model="model-b", window="window-a", kv="kv-a")
        tokens = tuple(range(1, 17))
        a = _tier(tmp_path)
        _spill(a, tokens, marker=3.0)
        a.close()

        b = _tier(tmp_path, identity=other)
        try:
            assert b.reconcile() == 0
            assert b.lookup_ssd(tokens) is None
            assert b.lookup_ssd_prefix(tokens + (99,)) is None
            assert b.namespace_dir != a.namespace_dir
        finally:
            b.close()

        # ... and reconcile() on the other tier did not sweep a's files
        a2 = _tier(tmp_path)
        try:
            assert a2.lookup_ssd(tokens) is not None
            assert a2._read_entry(tokens, a2.lookup_ssd(tokens)["file_path"])
        finally:
            a2.close()

    @pytest.mark.parametrize(
        "other",
        [
            CacheIdentity(model="model-b", window="window-a", kv="kv-a"),
            CacheIdentity(model="model-a", window="window-b", kv="kv-a"),
            CacheIdentity(model="model-a", window="window-a", kv="kv-b"),
        ],
        ids=["model", "max-kv-size", "kv-quantization"],
    )
    def test_each_identity_component_selects_the_namespace(self, tmp_path, other):
        tokens = tuple(range(1, 17))
        a = _tier(tmp_path)
        _spill(a, tokens)
        a.close()
        b = _tier(tmp_path, identity=other)
        try:
            assert b.lookup_ssd(tokens) is None
        finally:
            b.close()

    @staticmethod
    def _legacy_layout(base):
        import sqlite3

        entry = base / "data" / ("ab" * 32)
        entry.mkdir(parents=True)
        (entry / "manifest.json").write_text("{}")
        (base / "quarantine" / ("cd" * 32)).mkdir(parents=True)
        (base / "quarantine" / ("cd" * 32) / "manifest.json").write_text("{}")
        conn = sqlite3.connect(base / "index.db")
        conn.execute(
            "CREATE TABLE entries (token_hash TEXT PRIMARY KEY, tokens_blob BLOB, "
            "num_tokens INTEGER, file_path TEXT, memory_bytes INTEGER)"
        )
        conn.commit()
        conn.close()

    def test_entries_of_the_untagged_layout_are_discarded(self, tmp_path):
        """A build that keyed on tokens alone left index.db and data/ in the base
        dir, with nothing saying which model wrote them."""
        self._legacy_layout(tmp_path)
        tier = _tier(tmp_path)
        try:
            assert not (tmp_path / "index.db").exists()
            assert not (tmp_path / "data").exists()
            assert not (tmp_path / "quarantine").exists()
            assert tier.get_stats()["spill_count"] == 0
        finally:
            tier.close()

    def test_unrelated_files_in_the_base_dir_are_never_removed(self, tmp_path):
        """--ssd-cache-dir is user-supplied; data/ and quarantine/ are generic names."""
        mine = tmp_path / "data" / "mine"
        mine.mkdir(parents=True)
        (mine / "x.csv").write_text("1,2,3")
        (mine / "manifest.json").write_text("{}")  # a manifest alone is not enough
        (tmp_path / "data" / "notes.txt").write_text("keep")
        # a 64-hex directory without a manifest is not one of ours either
        lookalike = tmp_path / "data" / ("ef" * 32)
        lookalike.mkdir()
        (lookalike / "payload.bin").write_text("keep")
        (tmp_path / "quarantine").mkdir()
        (tmp_path / "quarantine" / "mine.txt").write_text("keep")
        (tmp_path / "index.db").write_text("not a sqlite file")
        tier = _tier(tmp_path)
        try:
            assert (mine / "x.csv").read_text() == "1,2,3"
            assert (tmp_path / "data" / "notes.txt").exists()
            assert (lookalike / "payload.bin").exists()
            assert (tmp_path / "quarantine" / "mine.txt").exists()
            assert (tmp_path / "index.db").read_text() == "not a sqlite file"
        finally:
            tier.close()

    def test_an_unrelated_sqlite_index_db_in_the_base_dir_is_left_alone(self, tmp_path):
        """Any SQLite file has the same header: it must hold the legacy table."""
        import sqlite3

        conn = sqlite3.connect(tmp_path / "index.db")
        conn.execute("CREATE TABLE documents (id INTEGER PRIMARY KEY, body TEXT)")
        conn.execute("INSERT INTO documents (body) VALUES ('mine')")
        conn.commit()
        conn.close()
        sibling = sqlite3.connect(tmp_path / "other.db")  # not named index.db at all
        sibling.execute(
            "CREATE TABLE entries (token_hash TEXT, tokens_blob BLOB, file_path TEXT)"
        )
        sibling.commit()
        sibling.close()
        tier = _tier(tmp_path)
        try:
            check = sqlite3.connect(tmp_path / "index.db")
            assert check.execute("SELECT body FROM documents").fetchall() == [("mine",)]
            check.close()
            assert (tmp_path / "other.db").exists()
        finally:
            tier.close()

    @staticmethod
    def _listing(base):
        return sorted(os.listdir(base))

    def test_probing_a_foreign_wal_index_db_adds_no_files_beside_it(self, tmp_path):
        import sqlite3

        conn = sqlite3.connect(tmp_path / "index.db")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE documents (id INTEGER PRIMARY KEY, body TEXT)")
        conn.commit()
        conn.close()
        assert self._listing(tmp_path) == ["index.db"]
        before = self._listing(tmp_path)
        assert SSDCacheTier._is_legacy_index(str(tmp_path / "index.db")) is False
        assert self._listing(tmp_path) == before

    def test_a_legacy_index_whose_table_is_only_in_the_wal_is_still_recognised(
        self, tmp_path
    ):
        """A crash leaves the WAL; the schema may exist nowhere else."""
        import sqlite3

        conn = sqlite3.connect(tmp_path / "index.db")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA wal_autocheckpoint=0")
        conn.execute(
            "CREATE TABLE entries (token_hash TEXT PRIMARY KEY, tokens_blob BLOB, "
            "num_tokens INTEGER, file_path TEXT, memory_bytes INTEGER)"
        )
        conn.commit()  # the connection stays open: the table lives in the WAL
        try:
            assert os.path.getsize(tmp_path / "index.db-wal") > 0
            assert SSDCacheTier._is_legacy_index(str(tmp_path / "index.db")) is True
        finally:
            conn.close()

    def test_a_foreign_wal_index_db_with_a_live_wal_is_left_as_it_was(self, tmp_path):
        import sqlite3

        conn = sqlite3.connect(tmp_path / "index.db")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA wal_autocheckpoint=0")
        conn.execute("CREATE TABLE documents (id INTEGER PRIMARY KEY, body TEXT)")
        conn.commit()
        before = self._listing(tmp_path)
        sizes = {n: os.path.getsize(tmp_path / n) for n in before}
        try:
            tier = _tier(tmp_path)
            tier.close()
            assert (tmp_path / "index.db").exists()
            for name in before:
                assert os.path.getsize(tmp_path / name) == sizes[name], name
            assert {"index.db", "index.db-wal"} <= set(self._listing(tmp_path))
        finally:
            conn.close()

    def test_an_index_db_with_an_entries_table_of_another_shape_is_left_alone(
        self, tmp_path
    ):
        import sqlite3

        conn = sqlite3.connect(tmp_path / "index.db")
        conn.execute("CREATE TABLE entries (token_hash TEXT, note TEXT)")
        conn.commit()
        conn.close()
        tier = _tier(tmp_path)
        try:
            assert (tmp_path / "index.db").exists()
        finally:
            tier.close()

    def test_the_legacy_entries_go_and_the_user_files_beside_them_stay(self, tmp_path):
        self._legacy_layout(tmp_path)
        (tmp_path / "data" / "mine.txt").write_text("keep")
        tier = _tier(tmp_path)
        try:
            assert not (tmp_path / "data" / ("ab" * 32)).exists()
            assert (tmp_path / "data" / "mine.txt").exists()
            assert not (tmp_path / "index.db").exists()
        finally:
            tier.close()

    @staticmethod
    def _stale_namespace(base, age_days, name="1" * 32, with_marker=True):
        d = base / name
        (d / "data" / ("ab" * 32)).mkdir(parents=True)
        if with_marker:
            marker = d / "identity.json"
            marker.write_text("{}")
            t = time.time() - age_days * 86400
            os.utime(marker, (t, t))
        return d

    def test_namespaces_of_old_identities_are_reclaimed_after_the_retention(
        self, tmp_path
    ):
        old = self._stale_namespace(tmp_path, 45, "1" * 32)
        recent = self._stale_namespace(tmp_path, 3, "2" * 32)
        no_marker = self._stale_namespace(tmp_path, 0, "3" * 32, with_marker=False)
        other = tmp_path / "my-own-dir"
        other.mkdir()
        (other / "identity.json").write_text("{}")
        os.utime(other / "identity.json", (1, 1))
        tier = _tier(tmp_path, stale_namespace_days=30)
        try:
            assert not old.exists()
            assert recent.exists() and no_marker.exists() and other.exists()
            assert os.path.isdir(tier.namespace_dir)
        finally:
            tier.close()

    def test_a_tier_never_reclaims_its_own_namespace_and_refreshes_its_marker(
        self, tmp_path
    ):
        tier = _tier(tmp_path)
        tier.close()
        marker = os.path.join(tier.namespace_dir, "identity.json")
        os.utime(marker, (1, 1))  # looks ancient
        again = _tier(tmp_path, stale_namespace_days=30)
        try:
            assert os.path.isdir(again.namespace_dir)
            assert os.stat(marker).st_mtime > time.time() - 60
        finally:
            again.close()

    def test_a_namespace_a_running_tier_holds_is_never_reclaimed(self, tmp_path):
        """The marker is written once, at start: a server that has run (or idled)
        longer than the retention must keep its directory when another starts."""
        other = CacheIdentity(model="model-b", window="window-a", kv="kv-a")
        a = _tier(tmp_path, stale_namespace_days=30)
        try:
            _spill(a, tuple(range(1, 17)))
            long_ago = time.time() - 31 * 86400
            for name in ("identity.json", "index.db", "index.db-wal", "index.db-shm"):
                path = os.path.join(a.namespace_dir, name)
                if os.path.exists(path):
                    os.utime(path, (long_ago, long_ago))

            b = _tier(tmp_path, identity=other, stale_namespace_days=30)
            try:
                assert os.path.isdir(a.namespace_dir), "the live namespace was deleted"
                assert a.lookup_ssd(tuple(range(1, 17))) is not None
                _spill(a, tuple(range(100, 116)))  # and its writer still works
            finally:
                b.close()
        finally:
            a.close()

    def test_a_namespace_is_reclaimed_once_its_tier_has_closed(self, tmp_path):
        other = CacheIdentity(model="model-b", window="window-a", kv="kv-a")
        a = _tier(tmp_path)
        a.close()
        long_ago = time.time() - 31 * 86400
        for name in ("identity.json", "index.db", "index.db-wal", "index.db-shm"):
            path = os.path.join(a.namespace_dir, name)
            if os.path.exists(path):
                os.utime(path, (long_ago, long_ago))

        b = _tier(tmp_path, identity=other, stale_namespace_days=30)
        try:
            assert not os.path.exists(a.namespace_dir)
        finally:
            b.close()

    def test_use_of_the_index_keeps_a_namespace_young(self, tmp_path):
        """Only identity.json is old: the index was written recently (spills, hits)."""
        ns = self._stale_namespace(tmp_path, 45, "1" * 32)
        (ns / "index.db").write_text("x")
        tier = _tier(tmp_path, stale_namespace_days=30)
        try:
            assert ns.exists()
            long_ago = time.time() - 45 * 86400
            os.utime(ns / "index.db", (long_ago, long_ago))
        finally:
            tier.close()
        again = _tier(tmp_path, stale_namespace_days=30)
        try:
            assert not ns.exists()
        finally:
            again.close()

    def test_a_digest_directory_without_a_marker_is_handled_quietly(
        self, tmp_path, caplog
    ):
        ns = tmp_path / ("4" * 32)
        ns.mkdir()
        long_ago = time.time() - 90 * 86400
        os.utime(ns, (long_ago, long_ago))
        with caplog.at_level("DEBUG", logger="vllm_mlx.ssd_cache"):
            tier = _tier(tmp_path, stale_namespace_days=30)
        try:
            assert not ns.exists(), "an old directory with no marker is as old as it is"
            assert "Traceback" not in caplog.text
            assert not [r for r in caplog.records if r.levelname == "WARNING"]
        finally:
            tier.close()

    def test_two_tiers_of_one_identity_share_a_namespace_and_both_keep_it(
        self, tmp_path
    ):
        other = CacheIdentity(model="model-b", window="window-a", kv="kv-a")
        first, second = _tier(tmp_path), _tier(tmp_path)
        try:
            assert first.namespace_dir == second.namespace_dir
            first.close()
            long_ago = time.time() - 31 * 86400
            os.utime(
                os.path.join(second.namespace_dir, "identity.json"), (long_ago,) * 2
            )
            third = _tier(tmp_path, identity=other, stale_namespace_days=30)
            try:
                assert os.path.isdir(second.namespace_dir), "second still holds it"
            finally:
                third.close()
        finally:
            second.close()

    def test_a_claim_waits_for_a_reclaim_and_starts_over(self, tmp_path):
        """A tier starting on a directory another tier is reclaiming ends with a
        usable directory, not a lock file in a deleted one."""
        import fcntl
        import shutil
        import threading

        ns = self._stale_namespace(tmp_path, 45, IDENT.digest)
        lock = os.open(str(ns / ".lock"), os.O_CREAT | os.O_RDWR)
        fcntl.flock(lock, fcntl.LOCK_EX)  # the reclaiming tier
        result = {}

        def start():
            result["tier"] = _tier(tmp_path)

        thread = threading.Thread(target=start)
        thread.start()
        time.sleep(0.3)
        assert thread.is_alive(), "the claim must wait for the reclaim"
        shutil.rmtree(ns)
        os.close(lock)
        thread.join(timeout=10)
        tier = result["tier"]
        try:
            assert os.path.isdir(tier._data_dir)
            _spill(tier, tuple(range(1, 17)))
            assert tier.lookup_ssd(tuple(range(1, 17))) is not None
        finally:
            tier.close()

    @staticmethod
    def _backdate(
        tier_dir,
        days=31,
        names=("identity.json", "index.db", "index.db-wal", "index.db-shm"),
    ):
        long_ago = time.time() - days * 86400
        for name in names:
            path = os.path.join(tier_dir, name)
            if os.path.exists(path):
                os.utime(path, (long_ago, long_ago))

    def test_a_tier_starting_while_its_stale_directory_is_being_deleted_keeps_its_own(
        self, tmp_path, monkeypatch
    ):
        """The reclaim holds the exclusive lock and then deletes the directory; the
        ``.lock`` file goes first, so a tier of that identity starting meanwhile
        used to create a new one, lock it and settle in a directory being deleted."""
        import shutil

        other = CacheIdentity(model="model-b", window="window-a", kv="kv-a")
        old = _tier(tmp_path)
        old.close()
        self._backdate(old.namespace_dir)
        real_rmtree = shutil.rmtree
        started = {}

        def rmtree_with_a_start_in_the_middle(path, *args, **kwargs):
            lock = os.path.join(path, ".lock")
            if os.path.exists(lock):
                os.unlink(lock)  # what rmtree does first among the files
            if "tier" not in started:
                started["tier"] = None
                started["tier"] = _tier(  # same identity as the stale one
                    tmp_path, stale_namespace_days=None
                )
            return real_rmtree(path, *args, **kwargs)

        monkeypatch.setattr(shutil, "rmtree", rmtree_with_a_start_in_the_middle)
        reclaiming = _tier(tmp_path, identity=other, stale_namespace_days=30)
        monkeypatch.undo()
        mid = started["tier"]
        try:
            assert mid.namespace_dir == old.namespace_dir
            assert os.path.isfile(os.path.join(mid.namespace_dir, "index.db"))
            _spill(mid, tuple(range(1, 17)))
            assert mid.lookup_ssd(tuple(range(1, 17))) is not None
            assert not [n for n in os.listdir(tmp_path) if ".reclaim-" in n]
        finally:
            mid.close()
            reclaiming.close()

    def test_a_claim_starts_over_when_the_directory_is_removed_under_it(
        self, tmp_path, monkeypatch
    ):
        import shutil

        real_open = os.open
        calls = []

        def open_after_a_reclaim(path, *args, **kwargs):
            if str(path).endswith(".lock") and not calls:
                calls.append(path)
                shutil.rmtree(
                    os.path.dirname(path)
                )  # moved away between mkdir and open
            return real_open(path, *args, **kwargs)

        monkeypatch.setattr(os, "open", open_after_a_reclaim)
        tier = _tier(tmp_path)
        monkeypatch.undo()
        try:
            assert calls and os.path.isdir(tier._data_dir)
            _spill(tier, tuple(range(1, 17)))
        finally:
            tier.close()

    def test_a_claim_that_waited_on_a_replaced_lock_file_holds_the_new_one(
        self, tmp_path
    ):
        import fcntl
        import threading

        ns = self._stale_namespace(tmp_path, 45, IDENT.digest)
        lock_path = ns / ".lock"
        old = os.open(str(lock_path), os.O_CREAT | os.O_RDWR)
        fcntl.flock(old, fcntl.LOCK_EX)
        result = {}
        thread = threading.Thread(target=lambda: result.update(tier=_tier(tmp_path)))
        thread.start()
        time.sleep(0.3)
        assert thread.is_alive()
        os.unlink(lock_path)  # the directory stays; its lock file is replaced
        lock_path.write_text("")
        os.close(old)
        thread.join(timeout=10)
        tier = result["tier"]
        try:
            assert (
                os.fstat(tier._namespace_lock_fd).st_ino == os.stat(lock_path).st_ino
            ), "the tier holds a lock nobody else can see"
        finally:
            tier.close()

    def test_advisory_locks_that_cannot_be_taken_do_not_stop_a_tier_and_nothing_is_reclaimed(
        self, tmp_path, monkeypatch, caplog
    ):
        """Network and FUSE mounts can refuse flock (ENOLCK / ENOTSUP)."""
        import errno
        import fcntl

        stale = self._stale_namespace(tmp_path, 90, "5" * 32)

        def refuse(fd, operation):
            raise OSError(errno.ENOLCK, "No locks available")

        monkeypatch.setattr(fcntl, "flock", refuse)
        with caplog.at_level("DEBUG", logger="vllm_mlx.ssd_cache"):
            tier = _tier(tmp_path, stale_namespace_days=30)
        try:
            assert tier._namespace_lock_fd is None
            assert stale.exists(), "idleness cannot be shown without locks"
            assert "advisory locks are unavailable" in caplog.text
            _spill(tier, tuple(range(1, 17)))
            assert tier.lookup_ssd(tuple(range(1, 17))) is not None
        finally:
            tier.close()

    def test_a_namespace_used_between_the_check_and_the_lock_is_kept(
        self, tmp_path, monkeypatch
    ):
        import fcntl

        stale = self._stale_namespace(tmp_path, 90, "6" * 32)
        real_flock = fcntl.flock

        def a_tier_started_and_closed_first(fd, operation):
            if operation & fcntl.LOCK_EX:
                os.utime(stale / "identity.json")  # now
            return real_flock(fd, operation)

        monkeypatch.setattr(fcntl, "flock", a_tier_started_and_closed_first)
        tier = _tier(tmp_path, stale_namespace_days=30)
        try:
            assert stale.exists()
        finally:
            tier.close()

    def test_only_the_wal_is_recent(self, tmp_path):
        """Spills and hits write the WAL first; the main file can stay old."""
        ns = self._stale_namespace(tmp_path, 90, "7" * 32)
        (ns / "index.db").write_text("x")
        (ns / "index.db-wal").write_text("x")
        long_ago = time.time() - 90 * 86400
        os.utime(ns / "index.db", (long_ago, long_ago))
        tier = _tier(tmp_path, stale_namespace_days=30)
        try:
            assert ns.exists()
        finally:
            tier.close()

    def test_a_failed_start_releases_the_namespace(self, tmp_path, monkeypatch):
        import fcntl

        def broken(cache_dir):
            raise RuntimeError("index cannot be opened")

        monkeypatch.setattr(ssd_cache_module, "SSDIndex", broken)
        with pytest.raises(RuntimeError, match="index cannot be opened"):
            SSDCacheTier(SSDCacheConfig(cache_dir=str(tmp_path), identity=IDENT))
        monkeypatch.undo()
        fd = os.open(os.path.join(tmp_path, IDENT.digest, ".lock"), os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)  # raises if still held
        finally:
            os.close(fd)

    def test_tombstones_of_an_interrupted_reclaim_are_deleted_at_the_next_start(
        self, tmp_path
    ):
        leftover = tmp_path / ("8" * 32 + ".reclaim-123-456")
        (leftover / "data" / ("ab" * 32)).mkdir(parents=True)
        mine = tmp_path / ("8" * 32 + ".reclaim-notmine")  # not our naming: left alone
        mine.mkdir()
        tier = _tier(tmp_path, stale_namespace_days=30)
        try:
            assert not leftover.exists()
            assert mine.exists()
        finally:
            tier.close()

    def test_a_relative_cache_dir_works(self, tmp_path, monkeypatch):
        """--ssd-cache-dir is passed as given: ``cache`` is relative to the cwd."""
        monkeypatch.chdir(tmp_path)
        tier = _tier("cache")
        try:
            assert os.path.isabs(tier.namespace_dir)
            _spill(tier, tuple(range(1, 17)))
            assert tier.lookup_ssd(tuple(range(1, 17))) is not None
            assert os.path.isfile(tmp_path / "cache" / IDENT.digest / ".lock")
            # the working directory moving later does not move the tier's files
            monkeypatch.chdir("/")
            _spill(tier, tuple(range(100, 116)))
            assert tier.lookup_ssd(tuple(range(100, 116))) is not None
        finally:
            tier.close()
        assert not os.path.exists("/" + IDENT.digest)

    @pytest.mark.parametrize("spelling", ["cache", "./cache", "sub/../cache"])
    def test_a_relative_cache_dir_discards_the_legacy_layout_and_reclaims(
        self, tmp_path, monkeypatch, spelling
    ):
        monkeypatch.chdir(tmp_path)
        base = tmp_path / "cache"
        base.mkdir()
        self._legacy_layout(base)
        stale = self._stale_namespace(base, 45, "9" * 32)
        tier = _tier(spelling, stale_namespace_days=30)
        try:
            assert not (base / "index.db").exists()
            assert not stale.exists()
        finally:
            tier.close()

    def test_the_legacy_probe_takes_a_relative_path_and_a_missing_file(
        self, tmp_path, monkeypatch
    ):
        import sqlite3

        monkeypatch.chdir(tmp_path)
        assert SSDCacheTier._is_legacy_index("index.db") is False  # absent
        assert SSDCacheTier._is_legacy_index(".") is False  # not a file
        conn = sqlite3.connect("index.db")
        conn.execute(
            "CREATE TABLE entries (token_hash TEXT PRIMARY KEY, tokens_blob BLOB, "
            "num_tokens INTEGER, file_path TEXT, memory_bytes INTEGER)"
        )
        conn.commit()
        conn.close()
        assert SSDCacheTier._is_legacy_index("index.db") is True

    def test_a_foreign_database_too_big_to_copy_is_left_alone(
        self, tmp_path, monkeypatch
    ):
        """A WAL database is probed as a copy: a big one is not worth copying and is
        treated as not ours."""
        import sqlite3

        conn = sqlite3.connect(tmp_path / "index.db")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA wal_autocheckpoint=0")
        conn.execute(
            "CREATE TABLE entries (token_hash TEXT PRIMARY KEY, tokens_blob BLOB, "
            "num_tokens INTEGER, file_path TEXT, memory_bytes INTEGER)"
        )
        conn.commit()
        try:
            path = str(tmp_path / "index.db")
            assert SSDCacheTier._is_legacy_index(path) is True
            monkeypatch.setattr(ssd_cache_module, "_LEGACY_PROBE_COPY_LIMIT", 1024)
            assert SSDCacheTier._is_legacy_index(path) is False
        finally:
            conn.close()

    def test_a_lock_file_that_cannot_be_created_degrades_like_a_refused_lock(
        self, tmp_path, monkeypatch, caplog
    ):
        real_open = os.open

        def refuse_the_lock_file(path, *args, **kwargs):
            if str(path).endswith(".lock"):
                raise PermissionError(13, "Permission denied", str(path))
            return real_open(path, *args, **kwargs)

        monkeypatch.setattr(os, "open", refuse_the_lock_file)
        with caplog.at_level("WARNING", logger="vllm_mlx.ssd_cache"):
            tier = _tier(tmp_path)
        try:
            assert tier._namespace_lock_fd is None
            assert "running without the in-use mark" in caplog.text
            _spill(tier, tuple(range(1, 17)))
        finally:
            tier.close()

    def test_a_tombstone_a_concurrent_sweep_deletes_is_not_a_failure(
        self, tmp_path, monkeypatch, caplog
    ):
        """Two servers starting together both sweep and delete tombstones."""
        import shutil

        other = CacheIdentity(model="model-b", window="window-a", kv="kv-a")
        old = _tier(tmp_path)
        old.close()
        self._backdate(old.namespace_dir)
        real_rmtree = shutil.rmtree

        def the_other_sweeper_got_there_first(path, *args, **kwargs):
            real_rmtree(path)
            if not kwargs.get("ignore_errors"):
                raise FileNotFoundError(2, "No such file or directory", str(path))

        monkeypatch.setattr(shutil, "rmtree", the_other_sweeper_got_there_first)
        with caplog.at_level("INFO", logger="vllm_mlx.ssd_cache"):
            tier = _tier(tmp_path, identity=other, stale_namespace_days=30)
        monkeypatch.undo()
        try:
            assert not [r for r in caplog.records if r.levelname == "WARNING"]
            assert "reclaimed" in caplog.text
            assert not os.path.exists(old.namespace_dir)
            assert not [n for n in os.listdir(tmp_path) if ".reclaim-" in n]
        finally:
            tier.close()

    def test_tombstones_are_swept_even_when_reclaiming_is_turned_off(self, tmp_path):
        leftover = tmp_path / ("a" * 32 + ".reclaim-1-2")
        (leftover / "data").mkdir(parents=True)
        stale = self._stale_namespace(tmp_path, 400, "b" * 32)
        tier = _tier(tmp_path, stale_namespace_days=None)
        try:
            assert not leftover.exists()
            assert stale.exists(), "turning reclaiming off keeps stale namespaces"
        finally:
            tier.close()

    def test_reclaiming_can_be_turned_off(self, tmp_path):
        old = self._stale_namespace(tmp_path, 400)
        tier = _tier(tmp_path, stale_namespace_days=None)
        try:
            assert old.exists()
        finally:
            tier.close()

    def test_tier_requires_an_identity(self, tmp_path):
        with pytest.raises(ValueError, match="identity"):
            SSDCacheTier(SSDCacheConfig(cache_dir=str(tmp_path)))


class TestClear:
    """A cache flush must flush the cold tier too."""

    def test_clear_removes_every_entry_from_index_and_disk(self, tmp_path):
        tier = _tier(tmp_path)
        try:
            _spill(tier, tuple(range(1, 17)))
            _spill(tier, tuple(range(100, 116)))
            assert tier.clear() == 2
            assert tier.lookup_ssd(tuple(range(1, 17))) is None
            assert tier.lookup_ssd_prefix(tuple(range(1, 20))) is None
            assert os.listdir(os.path.join(tier.namespace_dir, "data")) == []
            # and the tier keeps working afterwards
            _spill(tier, tuple(range(1, 17)))
            assert tier.lookup_ssd(tuple(range(1, 17))) is not None
        finally:
            tier.close()

    def test_spills_queued_before_a_clear_are_dropped(self, tmp_path):
        tier = SSDCacheTier(SSDCacheConfig(cache_dir=str(tmp_path), identity=IDENT))
        try:
            # writer not started: the spill is queued, not written
            assert tier.enqueue_spill(tuple(range(16)), [_kv_layer(16)], 1024)
            tier.clear()
            tier.start_writer()
            tier.enqueue_spill(tuple(range(100, 116)), [_kv_layer(16)], 1024)
            deadline = time.monotonic() + 5.0
            while tier._stats.spill_count < 1 and time.monotonic() < deadline:
                time.sleep(0.02)
            assert tier.lookup_ssd(tuple(range(16))) is None, "pre-flush spill survived"
            assert tier.lookup_ssd(tuple(range(100, 116))) is not None
        finally:
            tier.close()

    def test_a_spill_the_writer_already_took_off_the_queue_is_dropped_too(
        self, tmp_path
    ):
        """The flush lands after the writer dequeued the entry and before it wrote it."""
        tier = SSDCacheTier(SSDCacheConfig(cache_dir=str(tmp_path), identity=IDENT))
        flushed = threading.Event()
        real_get = tier._spill_queue.get

        def get_then_flush(*args, **kwargs):
            item = real_get(*args, **kwargs)
            if item is not None and not flushed.is_set():
                tier.clear()
                flushed.set()
            return item

        tier._spill_queue.get = get_then_flush
        try:
            tier.start_writer()
            assert tier.enqueue_spill(tuple(range(16)), [_kv_layer(16)], 1024)
            assert flushed.wait(timeout=5.0)
            time.sleep(0.2)  # the writer would have written it by now
            assert tier.get_stats()["spill_count"] == 0
            assert tier.lookup_ssd(tuple(range(16))) is None
        finally:
            tier.close()

    def test_a_write_in_flight_during_clear_is_not_published(self, tmp_path):
        tier = _tier(tmp_path)
        started, release = threading.Event(), threading.Event()
        real_write = tier._write_entry

        def slow_write(*args):
            started.set()
            assert release.wait(timeout=5.0)
            return real_write(*args)

        tier._write_entry = slow_write
        try:
            tier.enqueue_spill(tuple(range(16)), [_kv_layer(16)], 1024)
            assert started.wait(timeout=5.0)
            cleared = threading.Thread(target=tier.clear)
            cleared.start()
            time.sleep(0.1)  # clear() waits for the entry the writer is on
            assert cleared.is_alive()
            release.set()
            cleared.join(timeout=5.0)
            assert not cleared.is_alive()
            assert tier.lookup_ssd(tuple(range(16))) is None
            assert os.listdir(os.path.join(tier.namespace_dir, "data")) == []
        finally:
            release.set()
            tier.close()

    def test_a_promotion_in_flight_during_clear_returns_nothing(self, tmp_path):
        tokens = tuple(range(1, 17))
        tier = _tier(tmp_path)
        try:
            _spill(tier, tokens)
            released = []
            real_read = tier._read_entry

            def read_then_flush(*args):
                layers = real_read(*args)
                tier.clear()
                return layers

            tier._read_entry = read_then_flush
            result = asyncio.run(
                tier.async_promote(tokens, lambda n: True, released.append)
            )
            assert result is None
            assert released == [1024], "the reserved budget must be given back"
        finally:
            tier.close()

    def test_memory_cache_clear_flushes_the_attached_tier(self, tmp_path):
        from unittest.mock import MagicMock

        from vllm_mlx.memory_cache import MemoryAwarePrefixCache, MemoryCacheConfig

        cache = MemoryAwarePrefixCache(
            MagicMock(), MemoryCacheConfig(max_memory_mb=64, min_prefix_tokens=1)
        )
        tier = _tier(tmp_path)
        try:
            cache.set_ssd_tier(tier)
            _spill(tier, tuple(range(1, 17)))
            cache.clear()
            assert cache.check_ssd(list(range(1, 17))) is None
        finally:
            cache.set_ssd_tier(None)
            tier.close()

    def test_clear_on_a_closed_tier_is_a_noop(self, tmp_path):
        tier = _tier(tmp_path)
        tier.close()
        assert tier.clear() == 0


class TestEntryValidation:
    """An entry is served as the KV of its key; one that is not is quarantined."""

    def _promote(self, tier, tokens):
        meta = tier.lookup_ssd(tokens)
        return tier._read_entry(tokens, meta["file_path"])

    def test_a_consistent_entry_reads_back(self, tmp_path):
        tokens = tuple(range(1, 17))
        tier = _tier(tmp_path)
        try:
            _spill(tier, tokens, marker=7.0)
            layers = self._promote(tier, tokens)
            assert layers is not None and layers[0]["offset"] == 16
            assert float(layers[0]["keys"][0, 0, 0, 0]) == 7.0
        finally:
            tier.close()

    @pytest.mark.parametrize("defect", ["tokens", "count", "offset", "short_keys"])
    def test_an_entry_that_is_not_its_keys_kv_is_not_served(self, tmp_path, defect):
        tokens = tuple(range(1, 17))
        tier = _tier(tmp_path)
        try:
            _spill(tier, tokens)
            entry = os.path.join(
                tier.namespace_dir, "data", tier.lookup_ssd(tokens)["file_path"]
            )
            manifest_path = os.path.join(entry, "manifest.json")
            manifest = json.load(open(manifest_path))
            if defect == "tokens":
                # another prompt's entry, e.g. after an interrupted rewrite
                import array

                with open(os.path.join(entry, "tokens.bin"), "wb") as f:
                    array.array("i", range(500, 516)).tofile(f)
            elif defect == "count":
                manifest["num_tokens"] = 12
            elif defect == "offset":
                manifest["layers"][0]["offset"] = 12
            else:
                from safetensors.numpy import save_file

                save_file(
                    {
                        "layer_0_keys": np.zeros((1, 2, 4, 8), dtype=np.float16),
                        "layer_0_values": np.zeros((1, 2, 4, 8), dtype=np.float16),
                    },
                    os.path.join(entry, "layer_0.safetensors"),
                )
            json.dump(manifest, open(manifest_path, "w"))

            assert self._promote(tier, tokens) is None
            assert tier.lookup_ssd(tokens) is None, "the bad entry must be dropped"
            assert os.path.isdir(os.path.join(tier.namespace_dir, "quarantine"))
        finally:
            tier.close()

    def test_a_rotating_layer_may_hold_fewer_positions_than_its_offset(self, tmp_path):
        tokens = tuple(range(1, 41))
        tier = _tier(tmp_path)
        try:
            keys = MockMLXArray(np.ones((1, 2, 16, 8), dtype=np.float16))
            layer = MockKVCacheLayer(keys=keys, values=keys, offset=40)
            layer.max_size, layer.keep, layer.step, layer._idx = 16, 0, 256, 8
            assert tier.enqueue_spill(tokens, [layer], 1024)
            deadline = time.monotonic() + 5.0
            while tier._stats.spill_count < 1 and time.monotonic() < deadline:
                time.sleep(0.02)
            layers = self._promote(tier, tokens)
            assert layers is not None and layers[0]["offset"] == 40
        finally:
            tier.close()


class TestRotatingLayerRoundTrip:
    """A sliding-window layer comes back from the SSD tier as one, not as a
    plain KVCache that merely carries the window attributes."""

    @staticmethod
    def _rotating(tokens, max_size=8, keep=0):
        import mlx.core as mx
        from mlx_lm.models.cache import RotatingKVCache

        layer = RotatingKVCache(max_size=max_size, keep=keep)
        # one position at a time past the window: the buffer has wrapped
        for t in range(tokens):
            kv = mx.full((1, 2, 1, 4), float(t))
            layer.update_and_fetch(kv, kv + 0.5)
        mx.eval(layer.keys, layer.values)
        return layer

    def _round_trip(self, tmp_path, layer, tokens):
        from vllm_mlx.scheduler import Scheduler

        tier = _tier(tmp_path)
        try:
            assert tier.enqueue_spill(tokens, [layer], 1024)
            deadline = time.monotonic() + 5.0
            while tier._stats.spill_count < 1 and time.monotonic() < deadline:
                time.sleep(0.02)
            meta = tier.lookup_ssd(tokens)
            dicts = tier._read_entry(tokens, meta["file_path"])
        finally:
            tier.close()
        assert dicts is not None
        return object.__new__(Scheduler)._reconstruct_ssd_layers(dicts)[0]

    @pytest.mark.parametrize("keep", [0, 2])
    def test_restored_layer_behaves_like_the_original(self, tmp_path, keep):
        import mlx.core as mx
        from mlx_lm.models.cache import RotatingKVCache

        tokens = tuple(range(1, 22))  # 21 tokens through a window of 8
        original = self._rotating(len(tokens), keep=keep)
        restored = self._round_trip(tmp_path, original, tokens)

        assert type(restored) is RotatingKVCache
        assert (restored.max_size, restored.keep) == (original.max_size, keep)
        assert (restored.offset, restored._idx) == (original.offset, original._idx)

        # the next position lands in the same slot and attends to the same keys
        nxt = mx.full((1, 2, 1, 4), 99.0)
        k0, v0 = original.update_and_fetch(nxt, nxt)
        k1, v1 = restored.update_and_fetch(nxt, nxt)
        assert mx.array_equal(k0, k1).item() and mx.array_equal(v0, v1).item()
        assert restored.offset == original.offset
