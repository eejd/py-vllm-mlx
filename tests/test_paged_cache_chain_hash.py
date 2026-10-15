# SPDX-License-Identifier: Apache-2.0
"""A paged block is reused only behind the prefix it was computed after (#104).

``--use-paged-cache`` keyed each block by the hash of its own tokens, so a block
cached behind prefix ``Z`` was served behind prefix ``A``: the KV of a token
depends on every token before it, and the answer changed. Block identity is now
vLLM's chain hash (parent block hash + tokens + extra keys).

The KV used here encodes its own prefix: the value at position ``p`` is a hash
of ``tokens[:p + 1]``. A block that is reused behind another prefix therefore
reconstructs values that differ from the cold ones, which is what these tests
compare.
"""

import threading

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
cache_mod = pytest.importorskip("mlx_lm.models.cache")

from vllm_mlx.paged_cache import (  # noqa: E402
    PagedCacheManager,
    compute_block_hash,
    compute_chain_hashes,
)
from vllm_mlx.prefix_cache import BlockAwarePrefixCache  # noqa: E402
from vllm_mlx.request import Request, SamplingParams  # noqa: E402
from vllm_mlx.scheduler import Scheduler, SchedulerConfig  # noqa: E402

BS = 4


def _prefix_values(tokens):
    """value[p] identifies tokens[:p + 1] (the whole prefix, not the token)."""
    h = 0
    values = []
    for token in tokens:
        h = (h * 31 + token + 1) % 65521
        values.append(float(h))
    return values


def _kv_layer(tokens, n=None):
    """Extracted state of a plain KV layer that has seen ``tokens[:n]``."""
    n = len(tokens) if n is None else n
    vals = mx.array(_prefix_values(tokens)[:n], dtype=mx.float32)
    keys = mx.broadcast_to(vals.reshape(1, 1, n, 1), (1, 2, n, 2))
    return {
        "state": (keys, keys + 0.5),
        "meta_state": (str(n),),
        "class_name": "KVCache",
        "class_ref": cache_mod.KVCache,
    }


def _recurrent_layer(tokens, n=None):
    """Extracted state of a recurrent layer after ``tokens[:n]``."""
    n = len(tokens) if n is None else n
    state = mx.array([[_prefix_values(tokens[:n])[-1]]], dtype=mx.float32)
    return {
        "state": [state],
        "meta_state": "",
        "class_name": "ArraysCache",
        "class_ref": cache_mod.ArraysCache,
    }


def _cache(max_blocks=64):
    manager = PagedCacheManager(block_size=BS, max_blocks=max_blocks)
    return BlockAwarePrefixCache(model=None, paged_cache_manager=manager), manager


def _words(*blocks):
    """Concatenate blocks given as (value, repeat-per-block) tuples."""
    return [t for block in blocks for t in block]


A, B, C, Z, Z1 = ([v] * BS for v in (1, 3, 5, 2, 7))


def _serve(cache, rid, tokens, extent=None):
    """Store ``tokens`` as a finished request would (KV for ``extent`` tokens)."""
    return cache.store_cache(rid, tokens, [_kv_layer(tokens, extent)])


def _hit(cache, rid, tokens, **kw):
    """Fetch and rebuild: (cached tokens, rebuilt layers, remaining)."""
    table, remaining = cache.fetch_cache(rid, tokens, **kw)
    if table is None:
        return 0, None, remaining
    return table.num_tokens, cache.reconstruct_cache(table), remaining


def _assert_is_cold_prefix(rebuilt, tokens, n):
    """The rebuilt KV equals what a cold prefill of ``tokens[:n]`` writes."""
    expected = _kv_layer(tokens, n)["state"]
    assert rebuilt[0].offset == n
    assert mx.array_equal(rebuilt[0].keys, expected[0])
    assert mx.array_equal(rebuilt[0].values, expected[1])


class TestChainHash:
    def test_parent_is_part_of_a_block_identity(self):
        root = compute_block_hash(None, C)
        behind_z = compute_block_hash(compute_block_hash(None, Z), C)
        behind_a = compute_block_hash(compute_block_hash(None, A), C)
        assert len({root, behind_z, behind_a}) == 3

    def test_extra_keys_are_part_of_a_block_identity(self):
        assert compute_block_hash(None, A, ("img-1",)) != compute_block_hash(
            None, A, ("img-2",)
        )
        assert compute_block_hash(None, A, ("img-1",)) != compute_block_hash(None, A)

    def test_chain_hashes_cover_full_blocks_only(self):
        hashes = compute_chain_hashes(A + B + [9], BS)
        assert len(hashes) == 2
        assert hashes[1] == compute_block_hash(hashes[0], B)

    def test_manager_lookup_follows_the_chain(self):
        manager = PagedCacheManager(block_size=BS, max_blocks=16)
        for tokens in (Z + C, A + B):
            blocks = manager.get_new_blocks(2)
            manager.cache_full_blocks(blocks, tokens, 0, 2)

        shared, remaining = manager.find_shared_prefix(A + C)
        assert len(shared) == 1  # A's block; C was only cached behind Z
        assert remaining == C

        assert manager.find_cached_block(C) is None  # C is not a root block
        zc = compute_chain_hashes(Z + C, BS)
        assert manager.find_cached_block(C, zc[0]) is not None
        assert manager.find_cached_block(C, compute_chain_hashes(A, BS)[0]) is None


class TestStaleKV:
    """The consistency probe's cases, without a model."""

    def test_block_cached_behind_another_prefix_is_not_served(self):
        cache, _ = _cache()
        _serve(cache, "x1", Z + C + [9])  # Z+C
        _serve(cache, "x2", A + B + [9])  # A+B
        x3 = A + C + [4]

        cached, rebuilt, remaining = _hit(cache, "x3", x3)

        assert cached == BS  # A only: C was never computed behind A
        assert remaining == C + [4]
        _assert_is_cold_prefix(rebuilt, x3, BS)

    def test_shifted_block_is_not_served(self):
        cache, _ = _cache()
        _serve(cache, "x1", Z + Z1 + C + [9])  # C cached one block later
        cached, rebuilt, _ = _hit(cache, "x3", A + C + [4])
        assert cached == 0 and rebuilt is None

    def test_deleted_middle_block_is_not_served(self):
        cache, _ = _cache()
        _serve(cache, "x4", A + Z1 + C + [9])  # A+Z1+C primed, A+C asked
        x3 = A + C + [4]
        cached, rebuilt, remaining = _hit(cache, "x3", x3)
        assert cached == BS
        assert remaining == C + [4]
        _assert_is_cold_prefix(rebuilt, x3, BS)

    def test_the_prefix_that_was_computed_is_still_a_full_hit(self):
        cache, manager = _cache()
        x1 = A + B + C + [9]
        _serve(cache, "x1", x1)
        cache.release_cache("x1")

        later = A + B + C + [1, 2]
        cached, rebuilt, remaining = _hit(cache, "x2", later)

        assert cached == 3 * BS
        assert remaining == [1, 2]
        _assert_is_cold_prefix(rebuilt, later, 3 * BS)
        assert manager.stats.cache_hits >= 3

    def test_requests_sharing_a_prefix_share_its_blocks(self):
        cache, manager = _cache()
        _serve(cache, "x1", A + B + [9])
        _serve(cache, "x2", A + C + [9])
        # A once, B once, C once: the second request reused A's block
        assert manager.stats.allocated_blocks - 1 == 3  # minus the null block

    def test_extra_keys_separate_identical_tokens(self):
        cache, _ = _cache()
        tokens = A + B + [9]
        cache.store_cache("x1", tokens, [_kv_layer(tokens)], extra_keys=("img-1",))
        assert _hit(cache, "same", tokens, extra_keys=("img-1",))[0] == 2 * BS
        assert _hit(cache, "other", tokens, extra_keys=("img-2",))[0] == 0
        assert _hit(cache, "none", tokens)[0] == 0

    def test_blocks_beyond_the_kv_that_exists_are_not_cached(self):
        # prompt+output = 3 blocks exactly, but the cache holds one position
        # fewer (a token that was sampled and never fed back).
        cache, _ = _cache()
        tokens = A + B + C
        _serve(cache, "x1", tokens, extent=3 * BS - 1)
        cached, rebuilt, remaining = _hit(cache, "x2", tokens + [9])

        assert cached == 2 * BS  # the third block is one position short
        assert remaining == C + [9]
        _assert_is_cold_prefix(rebuilt, tokens, 2 * BS)

    def test_a_table_that_is_not_this_sequences_chain_is_not_extended(self):
        cache, manager = _cache()
        _serve(cache, "x1", Z + C + [9])
        # the same request id comes back with other tokens, table still held
        table = _serve(cache, "x1", A + C + [9])
        assert table.num_tokens == 2 * BS
        _, rebuilt, _ = _hit(cache, "x2", A + C + [1])
        _assert_is_cold_prefix(rebuilt, A + C + [1], 2 * BS)


class TestHybridLayers:
    """Recurrent state exists only where a request ended."""

    @staticmethod
    def _store(cache, rid, tokens, extent):
        cache.store_cache(
            rid, tokens, [_kv_layer(tokens, extent), _recurrent_layer(tokens, extent)]
        )

    def test_snapshot_from_an_earlier_block_is_never_paired_with_later_kv(self):
        cache, _ = _cache()
        s1 = A + B
        self._store(cache, "s1", s1, 2 * BS)  # snapshot after 8 tokens
        s4 = A + B + C + Z
        self._store(cache, "s4", s4, 4 * BS - 1)  # KV for 15; no snapshot at 12

        table, remaining = cache.fetch_cache("q", s4 + [9])
        rebuilt = cache.reconstruct_cache(table)

        assert table.num_tokens == 2 * BS  # not 3 * BS with the state of 2 * BS
        assert remaining == C + Z + [9]
        assert rebuilt[1].cache[0].item() == _prefix_values(s1)[-1]
        assert rebuilt[0].offset == 2 * BS

    def test_a_table_ending_without_a_snapshot_cannot_be_rebuilt(self):
        cache, _ = _cache()
        self._store(cache, "s1", A + B, 2 * BS)  # b1 holds a snapshot
        s4 = A + B + C + Z
        table = cache.store_cache(
            "s4", s4, [_kv_layer(s4, 4 * BS - 1), _recurrent_layer(s4, 4 * BS - 1)]
        )
        assert len(table.block_ids) == 3  # b2 was cached without a snapshot
        # the last block decides: b1's snapshot is the state after 8 tokens,
        # not after the 12 the table claims
        assert cache.reconstruct_cache(table) is None

    def test_prefix_without_a_snapshot_is_a_miss_not_a_guess(self):
        cache, _ = _cache()
        s2 = A + B + C
        self._store(cache, "s2", s2, 3 * BS)  # snapshot only at 12
        cached, rebuilt, _ = _hit(cache, "q", A + B + Z)
        assert cached == 0 and rebuilt is None

    def test_a_later_request_that_ends_on_a_block_completes_its_snapshot(self):
        cache, _ = _cache()
        s2 = A + B + C
        self._store(cache, "s2", s2, 3 * BS)
        assert _hit(cache, "q1", A + B + Z)[0] == 0
        s1 = A + B
        self._store(cache, "s1", s1, 2 * BS)  # same tokens behind the same prefix

        table, _ = cache.fetch_cache("q2", A + B + Z)
        rebuilt = cache.reconstruct_cache(table)
        assert table.num_tokens == 2 * BS
        assert rebuilt[1].cache[0].item() == _prefix_values(s1)[-1]


class _WindowLayer:
    """A layer that carries its own position, rebuilt from meta_state."""

    @classmethod
    def from_state(cls, state, meta_state):
        layer = cls()
        layer.keys, layer.values = state
        layer.offset = int(meta_state[0])
        return layer


class TestLayersThatCarryAPosition:
    def test_prefix_of_such_a_layer_is_not_rebuilt_at_the_wrong_offset(self):
        cache, _ = _cache()
        tokens = A + B + [9]
        layer = _kv_layer(tokens, 2 * BS)
        layer["meta_state"] = (str(2 * BS),)
        layer["class_ref"] = _WindowLayer
        cache.store_cache("x1", tokens, [layer])

        # the whole stored sequence is fine
        table, _ = cache.fetch_cache("full", tokens + [1])
        assert cache.reconstruct_cache(table)[0].offset == 2 * BS
        # a prefix would claim BS tokens with an offset of 2 * BS
        table, _ = cache.fetch_cache("prefix", A + Z + [1])
        assert table.num_tokens == BS
        assert cache.reconstruct_cache(table) is None


class TestBlockLifecycle:
    def test_freed_blocks_stay_cached_until_recycled(self):
        cache, manager = _cache(max_blocks=8)
        tokens = A + B + C + [9]
        _serve(cache, "x1", tokens)
        cache.release_cache("x1")
        assert manager.stats.allocated_blocks == 1  # only the null block

        cached, rebuilt, _ = _hit(cache, "x2", tokens + [1])
        assert cached == 3 * BS
        _assert_is_cold_prefix(rebuilt, tokens + [1], 3 * BS)

    def test_deepest_blocks_are_recycled_first(self):
        cache, manager = _cache(max_blocks=8)
        tokens = A + B + C + [9]
        _serve(cache, "x1", tokens)
        cache.release_cache("x1")
        # take blocks until exactly one cached block has to be recycled
        spare = manager.free_blocks - 3
        manager.get_new_blocks(spare + 1)

        # the chain's tail went; its head is still reachable
        assert _hit(cache, "x2", tokens + [1])[0] == 2 * BS

    def test_recycled_block_holds_nothing_of_its_previous_life(self):
        manager = PagedCacheManager(block_size=BS, max_blocks=4)
        block = manager.allocate_block()
        manager.register_block_hash(block, A)
        block.cache_data = ["tensors of another sequence"]
        manager.free_block(block.block_id)

        for _ in range(3):
            recycled = manager.allocate_block()
            if recycled.block_id == block.block_id:
                break
        assert recycled.block_hash is None and recycled.cache_data is None
        assert manager.find_cached_block(A) is None

    def test_a_pool_smaller_than_the_traffic_keeps_caching(self):
        cache, _ = _cache(max_blocks=5)  # 4 usable blocks, 2 per sequence
        sequences = [[v] * BS + [v + 1] * BS + [9] for v in range(10, 20)]
        for i, tokens in enumerate(sequences):
            _hit(cache, f"q{i}", tokens)
            _serve(cache, f"q{i}", tokens)
            cache.release_cache(f"q{i}")
        # the most recent sequence is cached, not the first ones that fit
        assert _hit(cache, "again", sequences[-1] + [1])[0] == 2 * BS

    def test_a_block_found_free_cannot_be_recycled_before_it_is_pinned(self):
        cache, manager = _cache(max_blocks=6)
        tokens = A + B + [9]
        _serve(cache, "x1", tokens)
        cache.release_cache("x1")  # both blocks are cached, free and evictable

        real_touch = manager.touch
        took_the_pool = threading.Event()
        racer = []

        def touch_after_another_thread_took_the_pool(blocks):
            def take_every_free_block():
                manager.get_new_blocks(manager.free_blocks)
                took_the_pool.set()

            racer.append(threading.Thread(target=take_every_free_block))
            racer[0].start()
            took_the_pool.wait(0.3)  # only possible if the pool lock was not held
            real_touch(blocks)

        manager.touch = touch_after_another_thread_took_the_pool
        table, remaining = cache.fetch_cache("x2", tokens + [1])
        racer[0].join()

        assert table is not None and table.num_tokens == 2 * BS
        _assert_is_cold_prefix(cache.reconstruct_cache(table), tokens + [1], 2 * BS)

    def test_release_of_a_fetched_only_table_drops_its_references(self):
        cache, manager = _cache()
        tokens = A + B + [9]
        _serve(cache, "x1", tokens)
        cache.release_cache("x1")
        table, _ = cache.fetch_cache("x2", tokens + [1])
        assert all(manager.allocated_blocks[b].ref_count == 1 for b in table.block_ids)
        cache.release_cache("x2")
        assert manager.stats.allocated_blocks == 1


# ---------------------------------------------------------------------------
# Through the scheduler, with a model whose KV encodes its prefix
# ---------------------------------------------------------------------------


class _Tokenizer:
    eos_token_id = 99
    clean_up_tokenization_spaces = False

    @staticmethod
    def encode(text, **_kwargs):
        return [ord(char) % 26 for char in text]

    @staticmethod
    def decode(tokens, **_kwargs):
        return "".join(chr(65 + int(token) % 26) for token in tokens)


class _PrefixSumModel(nn.Module):
    """Logits depend on the running sum of every earlier token.

    The key written at a position is the prefix sum, so a cached block's KV
    knows what came before it; the next token reads the last cached key.
    Tokens are positive and prompts here never end exactly on a cached block
    (an exact hit rewinds the cache, which keeps the trimmed keys in its
    buffer).
    """

    def __init__(self):
        super().__init__()
        self.layers = [object()]

    def make_cache(self):
        return [cache_mod.KVCache()]

    def __call__(self, tokens, cache=None):
        layer = cache[0]
        values = tokens.astype(mx.float32)
        if layer.keys is None:
            prior = mx.zeros((values.shape[0], 1))
        else:
            # prefix sums only grow, so the largest key is the last one (the
            # buffer's padding is zeros)
            prior = mx.max(layer.keys, axis=(1, 2, 3))[:, None]
        cumulative = prior + mx.cumsum(values, axis=1)
        kv = cumulative[:, None, :, None]
        layer.update_and_fetch(kv, kv)
        selected = cumulative.astype(mx.int32) % 8
        vocabulary = mx.arange(8)[None, None, :]
        return mx.where(vocabulary == selected[:, :, None], 10.0, -10.0)


def _scheduler(max_blocks=64):
    return Scheduler(
        _PrefixSumModel(),
        _Tokenizer(),
        SchedulerConfig(
            enable_prefix_cache=True,
            use_paged_cache=True,
            paged_cache_block_size=BS,
            max_cache_blocks=max_blocks,
            max_num_seqs=1,
            prefill_batch_size=1,
            completion_batch_size=1,
        ),
    )


def _generate(scheduler, request_id, prompt, max_tokens=3, **request_kwargs):
    request = Request(
        request_id=request_id,
        prompt=list(prompt),
        sampling_params=SamplingParams(max_tokens=max_tokens, temperature=0.0),
        **request_kwargs,
    )
    scheduler.add_request(request)
    emitted = []
    for _ in range(24):
        result = scheduler.step()
        for output in result.outputs:
            emitted.extend(output.new_token_ids)
        if request_id in result.finished_request_ids:
            return emitted, request
    raise AssertionError(f"request {request_id} did not finish")


class TestSchedulerStaleKV:
    def test_chain_swap_answers_like_a_cold_request(self):
        x3 = A + C + [1]
        cold, cold_req = _generate(_scheduler(), "cold", x3)
        assert cold_req.cached_tokens == 0

        warm = _scheduler()
        _generate(warm, "x1", Z + C + [1])
        _generate(warm, "x2", A + B + [1])
        tokens, request = _generate(warm, "x3", x3)

        assert request.cached_tokens == BS  # A only
        assert tokens == cold

    def test_deletion_answers_like_a_cold_request(self):
        x3 = A + C + [1]
        cold, _ = _generate(_scheduler(), "cold", x3)
        warm = _scheduler()
        _generate(warm, "x4", A + Z1 + C + [1])
        tokens, request = _generate(warm, "x3", x3)
        assert request.cached_tokens == BS
        assert tokens == cold

    def test_shared_prefix_is_reused_in_full(self):
        warm = _scheduler()
        _generate(warm, "x2", A + B + [1])
        tokens, request = _generate(warm, "x5", A + B + [2, 3])
        cold, _ = _generate(_scheduler(), "cold", A + B + [2, 3])
        assert request.cached_tokens == 2 * BS
        assert tokens == cold

    def test_finished_requests_do_not_pin_their_blocks(self):
        warm = _scheduler()
        _generate(warm, "x2", A + B + [1])
        _generate(warm, "x5", A + B + [2, 3])
        assert warm.paged_cache_manager.stats.allocated_blocks == 1
        assert warm.block_aware_cache._request_tables == {}

    def test_a_small_pool_keeps_serving_hits(self):
        warm = _scheduler(max_blocks=6)  # 5 usable blocks
        sequences = [[v] * BS + [v + 1] * BS + [1] for v in range(10, 18)]
        for i, tokens in enumerate(sequences):
            _generate(warm, f"q{i}", tokens)
        _, request = _generate(warm, "again", sequences[-1] + [2])
        assert request.cached_tokens == 2 * BS

    def test_media_requests_neither_read_nor_write_the_paged_cache(self):
        warm = _scheduler()
        tokens = A + B + [1]
        _generate(warm, "text", tokens)

        # same tokens, different image: must not be served the text KV
        _, media = _generate(warm, "media", tokens, is_multimodal=True)
        assert media.cached_tokens == 0

        # and what it computed is not stored for text requests
        other = _scheduler()
        _generate(other, "media", tokens, is_multimodal=True)
        _, request = _generate(other, "text", tokens)
        assert request.cached_tokens == 0
