# SPDX-License-Identifier: Apache-2.0
"""mlx_cache_compat across the cache contracts that coexist in one process."""

import mlx.core as mx
import pytest
from mlx_lm.models import cache as lm_cache

from vllm_mlx import mlx_cache_compat as compat

try:
    from mlx_vlm.models import cache as vlm_cache
except ImportError:  # pragma: no cover
    vlm_cache = None

TOKENS = 5


def _kv(cls, tokens=TOKENS):
    c = cls()
    k = mx.arange(tokens * 8, dtype=mx.float32).reshape(1, 1, tokens, 8)
    c.update_and_fetch(k, k + 1000)
    return c


class _LegacyMetaKV:
    """A 0.31-contract cache: meta_state property and two-argument from_state."""

    def __init__(self):
        self.keys = self.values = None
        self.offset = 0

    @property
    def state(self):
        return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]

    @state.setter
    def state(self, v):
        self.keys, self.values = v
        self.offset = self.keys.shape[2]

    @property
    def meta_state(self):
        return (str(self.offset),)

    @meta_state.setter
    def meta_state(self, v):
        self.offset = int(v[0])

    @classmethod
    def from_state(cls, state, meta_state):
        obj = cls()
        obj.keys, obj.values = state
        obj.offset = int(meta_state[0])
        return obj


class TestContractDetection:
    def test_lm_032_has_no_meta_state(self):
        assert not compat.uses_meta_state(lm_cache.KVCache)
        assert compat._from_state_arity(lm_cache.KVCache) == 1

    def test_legacy_fake_has_meta_state(self):
        assert compat.uses_meta_state(_LegacyMetaKV)
        assert compat._from_state_arity(_LegacyMetaKV) == 2

    @pytest.mark.skipif(vlm_cache is None, reason="mlx_vlm not installed")
    def test_vlm_cache_follows_legacy_contract(self):
        assert compat.uses_meta_state(vlm_cache.KVCache)
        assert compat._from_state_arity(vlm_cache.KVCache) == 2


class TestKVView:
    def test_padded_buffer_is_cut_to_the_valid_tokens(self):
        c = _kv(lm_cache.KVCache)
        assert c.state[0].shape[2] > TOKENS  # the 0.32 buffer is padded
        k, v = compat.kv_view(c)
        assert k.shape[2] == v.shape[2] == TOKENS

    def test_batch_kv_of_one(self):
        b = lm_cache.BatchKVCache([0])
        x = mx.ones((1, 1, TOKENS, 8))
        b.update_and_fetch(x, x)
        k, _ = compat.kv_view(b)
        assert k.shape[2] == TOKENS

    def test_batch_of_two_is_not_viewable(self):
        b = lm_cache.BatchKVCache([0, 0])
        x = mx.ones((2, 1, TOKENS, 8))
        b.update_and_fetch(x, x)
        assert compat.kv_view(b) is None

    def test_non_plain_layers_have_no_view(self):
        assert compat.kv_view(lm_cache.RotatingKVCache(max_size=16)) is None
        assert compat.kv_view(lm_cache.ArraysCache(2)) is None

    def test_empty_layer(self):
        assert compat.kv_view(lm_cache.KVCache()) is None


class TestSnapshotRestore:
    def test_plain_kv_roundtrip_matches_original_tokens(self):
        src = _kv(lm_cache.KVCache)
        state, meta = compat.snapshot_state(src)
        assert state[0].shape[2] == TOKENS
        assert meta == (str(TOKENS),)
        out = compat.restore_from_state(lm_cache.KVCache, state, meta)
        assert out.offset == TOKENS
        assert mx.array_equal(out.keys_and_values()[0], src.keys_and_values()[0])
        assert mx.array_equal(out.keys_and_values()[1], src.keys_and_values()[1])

    def test_batch_kv_restores_as_plain_kv(self):
        b = lm_cache.BatchKVCache([0])
        x = mx.ones((1, 1, TOKENS, 8))
        b.update_and_fetch(x, x)
        state, meta = compat.snapshot_state(b)
        out = compat.restore_from_state(type(b), state, meta)
        assert type(out) is lm_cache.KVCache
        assert out.offset == TOKENS

    def test_legacy_contract_layer_roundtrips(self):
        src = _LegacyMetaKV()
        src.state = (mx.ones((1, 1, TOKENS, 8)), mx.ones((1, 1, TOKENS, 8)))
        state, meta = compat.snapshot_state(src)
        out = compat.restore_from_state(_LegacyMetaKV, state, meta)
        assert out.offset == TOKENS

    @pytest.mark.skipif(vlm_cache is None, reason="mlx_vlm not installed")
    def test_vlm_kv_roundtrip(self):
        src = _kv(vlm_cache.KVCache)
        state, meta = compat.snapshot_state(src)
        assert state[0].shape[2] == TOKENS
        out = compat.restore_from_state(vlm_cache.KVCache, state, meta)
        assert out.offset == TOKENS

    def test_recurrent_layer_roundtrips_through_its_own_state(self):
        a = lm_cache.ArraysCache(2)
        a[0] = mx.ones((1, 3))
        a[1] = mx.zeros((1, 2))
        state, meta = compat.snapshot_state(a)
        out = compat.restore_from_state(lm_cache.ArraysCache, state, meta)
        assert mx.array_equal(out.cache[0], a.cache[0])
        assert mx.array_equal(out.cache[1], a.cache[1])

    def test_class_without_from_state_raises(self):
        class _Bare:
            pass

        with pytest.raises(TypeError):
            compat.restore_from_state(_Bare, ())


class TestRecurrentArrays:
    def test_cache_list_is_the_array_list_not_the_state_tuple(self):
        a = lm_cache.ArraysCache(2)
        a[0] = mx.ones((1, 3))
        arrays = compat.recurrent_arrays(a)
        assert arrays is a.cache
        assert len(a.state) == 3  # (cache, left_padding, lengths) on 0.32

    def test_kv_layer_has_none(self):
        assert compat.recurrent_arrays(_kv(lm_cache.KVCache)) is None

    def test_set_replaces_in_place(self):
        a = lm_cache.ArraysCache(2)
        compat.set_recurrent_arrays(a, [mx.ones((1, 1)), None])
        assert a.cache[1] is None and a.cache[0].shape == (1, 1)


class TestCopyState:
    def test_plain_kv(self):
        src, dst = _kv(lm_cache.KVCache), lm_cache.KVCache()
        compat.copy_state(dst, src)
        assert dst.offset == TOKENS

    def test_cache_list_copies_children_without_rebuilding_them(self):
        src = lm_cache.CacheList(_kv(lm_cache.KVCache), lm_cache.ArraysCache(1))
        dst = lm_cache.CacheList(lm_cache.KVCache(), lm_cache.ArraysCache(1))
        compat.copy_state(dst, src)
        assert dst.caches[0].offset == TOKENS

    def test_cache_list_with_a_model_defined_child(self):
        class _ModelOwnedCache(lm_cache.KVCache):
            """Not in mlx_lm.models.cache's namespace, so the CacheList state
            setter (which looks classes up by name) cannot rebuild it."""

        child = _kv(_ModelOwnedCache)
        src = lm_cache.CacheList(child)
        dst = lm_cache.CacheList(_ModelOwnedCache())
        compat.copy_state(dst, src)
        assert dst.caches[0].offset == TOKENS


def test_prompt_cache_format_matches_installed_mlx_lm():
    expected = "meta" if compat.uses_meta_state(lm_cache._BaseCache) else "state-v2"
    assert compat.prompt_cache_format() == expected


class TestPagedPrefixCacheRoundTrip:
    """The paged prefix cache stored nothing for mlx-lm 0.32 layers: extraction
    required ``meta_state``, which 0.32 caches no longer have, so every request
    silently missed."""

    def _scheduler_extract(self, layers):
        from vllm_mlx.scheduler import Scheduler

        return Scheduler._extract_cache_states(Scheduler.__new__(Scheduler), layers)

    def test_extraction_is_not_empty_and_cuts_the_padded_buffer(self):
        kv = _kv(lm_cache.KVCache)
        extracted = self._scheduler_extract([kv])
        assert len(extracted) == 1
        state = extracted[0]["state"]
        assert state[0].shape[2] == TOKENS  # not the 256-slot buffer

    def test_store_and_reconstruct_hybrid_model(self):
        from vllm_mlx.paged_cache import PagedCacheManager
        from vllm_mlx.prefix_cache import BlockAwarePrefixCache
        from vllm_mlx.scheduler import Scheduler

        tokens = list(range(8))
        kv = _kv(lm_cache.KVCache, tokens=8)
        rec = lm_cache.ArraysCache(2)
        rec[0] = mx.arange(6, dtype=mx.float32).reshape(1, 6)
        rec[1] = mx.arange(4, dtype=mx.float32).reshape(1, 4)

        extracted = self._scheduler_extract([kv, rec])
        assert len(extracted) == 2

        manager = PagedCacheManager(block_size=4, max_blocks=10)
        prefix = BlockAwarePrefixCache(model=None, paged_cache_manager=manager)
        table = prefix.store_cache("r1", tokens, extracted)
        rebuilt = prefix.reconstruct_cache(table)

        assert rebuilt is not None
        assert rebuilt[0].offset == 8
        assert mx.array_equal(rebuilt[0].keys_and_values()[0], kv.keys_and_values()[0])
        assert mx.array_equal(rebuilt[1].cache[0], rec.cache[0])
        assert mx.array_equal(rebuilt[1].cache[1], rec.cache[1])

        scheduler = Scheduler.__new__(Scheduler)
        again = scheduler._reconstruct_cache_from_states(extracted)
        assert again is not None and again[0].offset == 8
