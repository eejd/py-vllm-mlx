# SPDX-License-Identifier: Apache-2.0
"""A matching SSD entry is promoted into a full memory cache, and an exact repeat of
an evicted prompt matches (py-vllm-mlx#113).

Before, ``try_reserve_memory`` only checked ``current + n <= max`` and evicted
nothing, so a cache that had just spilled (it is full) denied every promotion
(``[ssd_promote] ... budget denied``): the tier worked only into an empty cache.
And a spilled key is the prompt plus what was generated, longer than the
repeated prompt, so nothing matched it.

The simulated KV encodes the tokens it was computed for (the value at position
``p`` is a checksum of ``tokens[:p + 1]``), so an entry promoted behind other
tokens, or cut at the wrong position, is visible in the tensors. The real-model
tests run a small random-init llama through the production scheduler: the warm
answer must be token-identical to the cold one at temperature 0.
"""

import random
import time
from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")
cache_mod = pytest.importorskip("mlx_lm.models.cache")
llama = pytest.importorskip("mlx_lm.models.llama")
tree_map = pytest.importorskip("mlx.utils").tree_map

from tests.test_cache_identity import prefix_kv, tiny_llama  # noqa: E402
from tests.test_exact_hit_reuse import _Model  # noqa: E402
from vllm_mlx.memory_cache import (  # noqa: E402
    MemoryAwarePrefixCache,
    MemoryCacheConfig,
)
from vllm_mlx.request import Request, SamplingParams  # noqa: E402
from vllm_mlx.scheduler import Scheduler, SchedulerConfig  # noqa: E402
from vllm_mlx.ssd_cache import SSDCacheConfig, SSDCacheTier  # noqa: E402

HEAD_DIM = 32  # 1 KiB of simulated KV per token across 2 layers
MIN_PREFIX = 32


def _kv(tokens):
    """prefix_kv, materialized contiguously: it broadcasts one value over the head
    dimension (stride 0), which a real model's KV never is and numpy copies wrongly."""
    layers = prefix_kv(tokens, heads=2, head_dim=HEAD_DIM)
    for layer in layers:
        layer.keys = layer.keys + mx.zeros(layer.keys.shape)
        layer.values = layer.values + mx.zeros(layer.values.shape)
    mx.eval([(layer.keys, layer.values) for layer in layers])
    return layers


def _conversation(index, n):
    """A prompt no other conversation shares a token with, then its reply."""
    base = 1000 * (index + 1)
    prompt = [base + i for i in range(n)]
    reply = [base + 900 + i for i in range(8)]
    return prompt, reply


def _wait_spilled(tier, count):
    deadline = time.monotonic() + 15
    while tier.get_stats()["spill_count"] < count and time.monotonic() < deadline:
        time.sleep(0.02)
    assert tier.get_stats()["spill_count"] >= count


def _drain(tier):
    """Wait until every queued spill is on disk (the writer takes the write lock)."""
    deadline = time.monotonic() + 15
    while not tier._spill_queue.empty() and time.monotonic() < deadline:
        time.sleep(0.01)
    time.sleep(0.05)
    with tier._write_lock:
        pass


class World:
    """A 1 MB memory cache backed by an SSD tier, behind a scheduler."""

    def __init__(self, tmp_path, entry_tokens):
        self.n = entry_tokens
        self.cache = MemoryAwarePrefixCache(
            tiny_llama(1),
            MemoryCacheConfig(
                max_memory_mb=1, max_entries=50, min_prefix_tokens=MIN_PREFIX
            ),
        )
        self.tier = SSDCacheTier(
            SSDCacheConfig(cache_dir=str(tmp_path), identity=self.cache.identity)
        )
        self.tier.start_writer()
        self.cache.set_ssd_tier(self.tier)
        self.sched = Scheduler(
            model=_Model(),
            tokenizer=SimpleNamespace(eos_token_id=0, eos_token_ids={0}),
            config=SchedulerConfig(enable_prefix_cache=False),
        )
        self.sched.memory_aware_cache = self.cache
        self.sched._ssd_tier = self.tier
        self.conversations = {}

    def finish(self, index):
        """Store conversation ``index`` the way a finished request does."""
        prompt, reply = _conversation(index, self.n)
        key = prompt + reply
        assert self.cache.store(key, _kv(key), evict_prefixes=False)
        self.conversations[index] = (prompt, reply)

    def fill(self, indices, spilled):
        for index in indices:
            self.finish(index)
        _wait_spilled(self.tier, spilled)

    def request(self, query, request_id="r"):
        request = Request(
            request_id=request_id,
            prompt="p",
            prompt_token_ids=list(query),
            sampling_params=SamplingParams(),
        )
        request.num_prompt_tokens = len(query)
        self.sched.add_request(request)
        return request

    def promote(self):
        self.sched._try_promote_ssd_pending()

    def close(self):
        self.tier.close()


@pytest.fixture
def world(tmp_path):
    made = []

    def build(entry_tokens=400):
        w = World(tmp_path / f"w{len(made)}", entry_tokens)
        made.append(w)
        return w

    yield build
    for w in made:
        w.close()


def _assert_cold_kv(request, query, covered):
    """The request holds exactly the cold KV of ``query[:covered]``."""
    expected = _kv(query[:covered])
    assert len(request.prompt_cache) == len(expected)
    for got, want in zip(request.prompt_cache, expected):
        assert got.offset == covered
        assert mx.array_equal(got.keys[..., :covered, :], want.keys)
        assert mx.array_equal(got.values[..., :covered, :], want.values)


def _within_budget(cache):
    used = sum(e.memory_bytes for e in cache._entries.values())
    assert used == cache._current_memory, "accounting drifted from the entries"
    assert cache._current_memory <= cache._max_memory


# --- promotion into a full cache --------------------------------------------------------------


def test_turn_two_of_an_evicted_conversation_is_promoted_into_a_full_cache(world):
    w = world()
    w.fill([0, 1, 2], spilled=1)  # 408 KiB each: the third one evicts conversation 0
    assert w.cache.memory_usage_mb > 0.75, "the cache must be full for this to prove it"
    assert (w.conversations[0][0] + w.conversations[0][1]) not in w.cache

    prompt, reply = w.conversations[0]
    query = prompt + reply + [7001 + i for i in range(40)]
    request = w.request(query)
    assert request.cache_hit_type == "ssd_pending"
    w.promote()

    assert request.cache_hit_type == "ssd_hit"
    assert request.cached_tokens == len(prompt) + len(reply)
    assert request.remaining_tokens == query[len(prompt) + len(reply) :]
    _assert_cold_kv(request, query, request.cached_tokens)
    # room was made by evicting the least recently used entry, which was spilled
    assert (prompt + reply) in w.cache
    p1, r1 = w.conversations[1]
    assert (p1 + r1) not in w.cache
    assert (w.conversations[2][0] + w.conversations[2][1]) in w.cache
    _wait_spilled(w.tier, 2)
    assert w.tier.lookup_ssd(tuple(p1 + r1)) is not None
    _within_budget(w.cache)


def test_the_least_recently_used_entry_is_the_one_evicted(world):
    w = world()
    w.fill([0, 1, 2], spilled=1)
    p1, r1 = w.conversations[1]
    assert w.cache.fetch(p1 + r1)[0] is not None  # conversation 1 is now the freshest
    prompt, reply = w.conversations[0]
    request = w.request(prompt + reply + [7001 + i for i in range(40)])
    w.promote()

    assert request.cache_hit_type == "ssd_hit"
    assert (p1 + r1) in w.cache
    p2, r2 = w.conversations[2]
    assert (p2 + r2) not in w.cache
    _within_budget(w.cache)


def test_an_exact_repeat_of_an_evicted_prompt_is_served_from_the_longer_entry(world):
    w = world()
    w.fill([0, 1, 2], spilled=1)
    prompt, reply = w.conversations[0]

    request = w.request(prompt)  # the same prompt again; the entry also holds the reply
    assert request.cache_hit_type == "ssd_pending"
    assert request._ssd_candidate["match_type"] == "lcp"
    w.promote()

    assert request.cache_hit_type == "ssd_hit"
    assert request.cached_tokens == len(prompt)
    assert request.remaining_tokens == []
    _assert_cold_kv(request, prompt, len(prompt))
    # the cache keeps the whole entry; only the request's copy was cut back
    stored = w.cache._entries[tuple(prompt + reply)].cache[0]
    assert stored.offset == len(prompt) + len(reply)
    _within_budget(w.cache)


def test_the_async_promotion_makes_room_and_cuts_back_like_the_synchronous_one(world):
    import asyncio

    w = world()
    w.fill([0, 1, 2], spilled=1)
    prompt, reply = w.conversations[0]

    request = w.request(prompt)
    assert request._ssd_candidate["match_type"] == "lcp"
    assert asyncio.run(w.sched.promote_from_ssd(request))

    assert request.cache_hit_type == "ssd_hit"
    assert request.cached_tokens == len(prompt)
    _assert_cold_kv(request, prompt, len(prompt))
    assert (prompt + reply) in w.cache
    _within_budget(w.cache)


def test_a_divergent_continuation_is_served_up_to_the_shared_prefix(world):
    w = world()
    w.fill([0, 1, 2], spilled=1)
    prompt, _ = w.conversations[0]
    query = prompt[:300] + [8001 + i for i in range(60)]

    request = w.request(query)
    w.promote()

    assert request.cache_hit_type == "ssd_hit"
    assert request.cached_tokens == 300
    assert request.remaining_tokens == query[300:]
    _assert_cold_kv(request, query, 300)
    _within_budget(w.cache)


def test_a_shared_prefix_below_the_minimum_is_not_served(world):
    w = world()
    w.fill([0, 1, 2], spilled=1)
    prompt, _ = w.conversations[0]
    query = prompt[: MIN_PREFIX - 1] + [8001 + i for i in range(200)]

    request = w.request(query)

    assert request.cache_hit_type != "ssd_pending"
    assert request.cached_tokens == 0


def test_the_longer_of_a_prefix_entry_and_a_shared_prefix_wins(world):
    w = world()
    prompt, reply = _conversation(0, 400)
    short, long_ = prompt[:100], prompt + reply
    for key in (short, long_):
        assert w.tier.enqueue_spill(tuple(key), _kv(key), 1024)
    _wait_spilled(w.tier, 2)

    request = w.request(prompt[:350])  # inside the long entry, past the short one
    assert request._ssd_candidate["match_type"] == "lcp"
    assert request._ssd_candidate["matched_tokens"] == 350

    request = w.request(short + [9001 + i for i in range(50)], "r2")
    assert request._ssd_candidate["match_type"] == "prefix"
    assert request._ssd_candidate["matched_tokens"] == 100


@pytest.mark.parametrize("layout", ["rotating", "unknown"])
def test_an_entry_that_cannot_be_cut_back_is_not_served_for_a_shared_prefix(
    world, layout
):
    w = world()
    prompt, reply = _conversation(0, 400)
    key = prompt + reply
    if layout == "rotating":
        layers = []
        for _ in range(2):
            layer = cache_mod.RotatingKVCache(max_size=512)
            layer.update_and_fetch(
                mx.zeros((1, 2, len(key), HEAD_DIM)),
                mx.zeros((1, 2, len(key), HEAD_DIM)),
            )
            layers.append(layer)
        assert w.tier.enqueue_spill(tuple(key), layers, 1024)
    else:  # written by a build that did not record the layout
        assert w.tier.enqueue_spill(tuple(key), _kv(key), 1024)
    _wait_spilled(w.tier, 1)
    if layout == "unknown":
        with w.tier._index._db_lock:
            w.tier._index._conn.execute("UPDATE entries SET trimmable = NULL")
            w.tier._index._conn.commit()

    assert w.tier.lookup_ssd_lcp(tuple(prompt), MIN_PREFIX) is None
    request = w.request(prompt)
    assert request.cache_hit_type != "ssd_pending"
    # a query that extends the key still matches it as a prefix
    assert w.request(key + [9001], "r2").cache_hit_type == "ssd_pending"


def test_a_promoted_entry_that_turns_out_not_to_be_cuttable_is_not_served(world):
    """The layout flag says plain KV, the data says sliding window: refuse, don't guess."""
    w = world()
    prompt, reply = _conversation(0, 400)
    key = prompt + reply
    layers = []
    for _ in range(2):
        layer = cache_mod.RotatingKVCache(max_size=512)
        layer.update_and_fetch(
            mx.zeros((1, 2, len(key), HEAD_DIM)), mx.zeros((1, 2, len(key), HEAD_DIM))
        )
        layers.append(layer)
    assert w.tier.enqueue_spill(tuple(key), layers, 1024)
    _wait_spilled(w.tier, 1)
    with w.tier._index._db_lock:
        w.tier._index._conn.execute("UPDATE entries SET trimmable = 1")
        w.tier._index._conn.commit()

    request = w.request(prompt)
    assert request.cache_hit_type == "ssd_pending"
    w.promote()

    assert request.cache_hit_type == "miss"
    assert request.cached_tokens == 0 and request.prompt_cache is None
    assert len(w.cache) == 0


# --- budget rules -----------------------------------------------------------------------------


def test_an_entry_larger_than_the_whole_budget_is_rejected_not_read(world, caplog):
    w = world()
    w.fill([0, 1], spilled=0)
    big_prompt, big_reply = _conversation(5, 1100)  # ~1.1 MB: can never fit in 1 MB
    big = big_prompt + big_reply
    assert w.tier.enqueue_spill(tuple(big), _kv(big), 1_200_000)
    _wait_spilled(w.tier, 1)
    reads = []
    real_read = w.tier._read_entry
    w.tier._read_entry = lambda *a: reads.append(a) or real_read(*a)
    before = {k: e.memory_bytes for k, e in w.cache._entries.items()}

    request = w.request(big + [1])
    assert request.cache_hit_type == "ssd_pending"
    with caplog.at_level("WARNING"):
        w.promote()

    assert request.cache_hit_type == "miss"
    assert request.cached_tokens == 0
    assert reads == []
    assert {k: e.memory_bytes for k, e in w.cache._entries.items()} == before
    assert w.tier.get_stats()["promotions_rejected_oversized"] == 1
    assert "exceeds the whole cache budget" in caplog.text
    _within_budget(w.cache)


def test_promotions_of_one_pass_do_not_evict_each_other(world):
    w = world(entry_tokens=600)  # 614 KiB: the cache holds one entry
    w.fill([0, 1, 2], spilled=2)
    (p0, r0), (p1, r1) = w.conversations[0], w.conversations[1]
    first = w.request(p0 + r0 + [7001 + i for i in range(40)], "first")
    second = w.request(p1 + r1 + [7001 + i for i in range(40)], "second")
    assert first.cache_hit_type == second.cache_hit_type == "ssd_pending"

    w.promote()

    assert first.cache_hit_type == "ssd_hit"
    assert second.cache_hit_type == "miss"
    assert (p0 + r0) in w.cache, "the second promotion evicted the first one's entry"
    assert w.tier.get_stats()["promotion_failures"] == 1
    _within_budget(w.cache)


def test_reserving_memory_evicts_lru_entries_and_never_exceeds_the_limit(world):
    w = world()
    w.fill([0, 1], spilled=0)
    keys = list(w.cache._entries)
    need = w.cache._max_memory - w.cache._current_memory + 1  # one byte short
    assert w.cache.try_reserve_memory(need)
    assert list(w.cache._entries) == keys[1:], "the LRU entry goes first"
    assert w.cache._current_memory <= w.cache._max_memory
    w.cache.release_reserved_memory(need)
    _within_budget(w.cache)


def test_reserving_more_than_the_budget_evicts_nothing(world):
    w = world()
    w.fill([0, 1], spilled=0)
    keys = list(w.cache._entries)
    assert not w.cache.try_reserve_memory(w.cache._max_memory + 1)
    assert list(w.cache._entries) == keys
    _within_budget(w.cache)


def test_protected_entries_are_not_evicted_for_a_reservation(world):
    w = world()
    w.fill([0, 1], spilled=0)
    keys = list(w.cache._entries)
    need = w.cache._max_memory - w.cache._current_memory + 1
    assert w.cache.try_reserve_memory(need, protect={keys[0]})
    assert list(w.cache._entries) == [keys[0]]
    w.cache.release_reserved_memory(need)
    assert not w.cache.try_reserve_memory(w.cache._max_memory, protect=keys)
    assert list(w.cache._entries) == [keys[0]]
    _within_budget(w.cache)


# --- review of PR #115 ------------------------------------------------------------------------


def test_requests_sharing_one_entry_in_a_pass_are_all_served(world):
    """The second request's entry is the one the first just promoted: it is resident,
    not 'denied because the cache is held for requests promoted in this pass'."""
    w = world(entry_tokens=600)  # the cache holds one such entry
    w.fill([0, 1], spilled=1)
    prompt, reply = w.conversations[0]
    turn_two = prompt + reply + [7001 + i for i in range(40)]

    first = w.request(turn_two, "first")
    second = w.request(turn_two, "second")
    repeat = w.request(prompt, "repeat")  # the same entry, cut back
    assert {r.cache_hit_type for r in (first, second, repeat)} == {"ssd_pending"}
    w.promote()

    for request in (first, second):
        assert request.cache_hit_type == "ssd_hit"
        assert request.cached_tokens == len(prompt) + len(reply)
        _assert_cold_kv(request, turn_two, request.cached_tokens)
    assert repeat.cache_hit_type == "ssd_hit"
    assert repeat.cached_tokens == len(prompt)
    _assert_cold_kv(repeat, prompt, len(prompt))
    assert w.cache._entries[tuple(prompt + reply)].cache[0].offset == len(
        prompt + reply
    )
    assert w.tier.get_stats()["promotion_failures"] == 0
    assert w.tier.get_stats()["ssd_hits"] == 1, "the entry was read from disk once"
    _within_budget(w.cache)


@pytest.mark.parametrize("why", ["oversized", "uncuttable"])
def test_an_lcp_entry_that_cannot_be_used_falls_back_to_the_prefix_entry(world, why):
    w = world()
    prompt, reply = _conversation(0, 400)
    short, long_ = prompt[:100], prompt + reply
    assert w.tier.enqueue_spill(tuple(short), _kv(short), 1024)
    if why == "oversized":
        assert w.tier.enqueue_spill(tuple(long_), _kv(long_), 1_200_000)
    else:
        layers = []
        for _ in range(2):
            layer = cache_mod.RotatingKVCache(max_size=512)
            layer.update_and_fetch(
                mx.zeros((1, 2, len(long_), HEAD_DIM)),
                mx.zeros((1, 2, len(long_), HEAD_DIM)),
            )
            layers.append(layer)
        assert w.tier.enqueue_spill(tuple(long_), layers, 1024)
    _wait_spilled(w.tier, 2)
    if why == "uncuttable":  # the index claims a layout the data does not have
        with w.tier._index._db_lock:
            w.tier._index._conn.execute("UPDATE entries SET trimmable = 1")
            w.tier._index._conn.commit()

    query = prompt[:350] + [9001 + i for i in range(30)]
    request = w.request(query)
    assert request._ssd_candidate["match_type"] == "lcp"
    assert request._ssd_candidate["fallback"]["match_type"] == "prefix"
    w.promote()

    assert request.cache_hit_type == "ssd_hit"
    assert request.cached_tokens == 100
    assert request.remaining_tokens == query[100:]
    _assert_cold_kv(request, query, 100)
    _within_budget(w.cache)


def test_the_async_promotion_falls_back_too(world):
    import asyncio

    w = world()
    prompt, reply = _conversation(0, 400)
    short, long_ = prompt[:100], prompt + reply
    assert w.tier.enqueue_spill(tuple(short), _kv(short), 1024)
    assert w.tier.enqueue_spill(tuple(long_), _kv(long_), 1_200_000)
    _wait_spilled(w.tier, 2)
    request = w.request(prompt[:350] + [9001])

    assert asyncio.run(w.sched.promote_from_ssd(request))

    assert request.cache_hit_type == "ssd_hit" and request.cached_tokens == 100


def test_the_async_promotion_serves_a_resident_entry_without_reading_it_again(world):
    import asyncio

    w = world(entry_tokens=600)
    w.fill([0, 1], spilled=1)
    prompt, reply = w.conversations[0]
    query = prompt + reply + [7001 + i for i in range(40)]
    first, second = w.request(query, "first"), w.request(query, "second")

    assert asyncio.run(w.sched.promote_from_ssd(first))
    assert asyncio.run(w.sched.promote_from_ssd(second))

    assert second.cache_hit_type == "ssd_hit"
    assert second.cached_tokens == len(prompt) + len(reply)
    _assert_cold_kv(second, query, second.cached_tokens)
    assert w.tier.get_stats()["ssd_hits"] == 1, "the entry was read from disk twice"
    _within_budget(w.cache)


def test_without_a_usable_fallback_the_request_is_a_miss(world):
    w = world()
    prompt, reply = _conversation(0, 400)
    long_ = prompt + reply
    assert w.tier.enqueue_spill(tuple(long_), _kv(long_), 1_200_000)
    _wait_spilled(w.tier, 1)
    request = w.request(prompt[:350] + [9001])
    assert "fallback" not in request._ssd_candidate
    w.promote()
    assert request.cache_hit_type == "miss" and request.cached_tokens == 0


def test_an_entry_already_on_the_tier_is_not_written_again(world):
    w = world()
    key = tuple(_conversation(0, 400)[0])
    assert w.tier.enqueue_spill(key, _kv(list(key)), 1024)
    _wait_spilled(w.tier, 1)
    written = w.tier.get_stats()["spill_bytes"]

    assert w.tier.enqueue_spill(key, _kv(list(key)), 1024)
    time.sleep(0.5)

    assert w.tier.get_stats()["spill_count"] == 1
    assert w.tier.get_stats()["spill_bytes"] == written


def test_cycling_through_more_conversations_than_fit_writes_each_entry_once(world):
    """The thrash case: three conversations round-robin through a cache that holds two.
    Every request promotes its entry back and evicts another; the evicted ones are on
    the tier already, so nothing is rewritten."""
    w = world()  # 408 KiB entries: two fit in 1 MB
    w.fill([0, 1, 2], spilled=1)
    served = []
    for round_ in range(3):
        for index in (0, 1, 2):
            prompt, reply = w.conversations[index]
            _drain(w.tier)  # what the previous request evicted is on the tier
            request = w.request(prompt, f"r{round_}-{index}")
            if request.cache_hit_type == "ssd_pending":
                w.promote()
            served.append(request.cached_tokens)
            w.sched.waiting.clear()
            w.cache.fetch(prompt + reply)  # what a running request leaves behind
    _drain(w.tier)

    assert all(n >= 400 - 1 for n in served), served
    stats = w.tier.get_stats()
    assert stats["spill_count"] == 3, "an entry was rewritten after it was promoted"
    _within_budget(w.cache)


def test_a_rejected_ssd_hit_evicts_the_entry_it_came_from(world):
    """``_evict_incompatible_entry`` removes what the request was served from: the
    entry's key, which for an lcp hit is longer than the tokens the request reused."""
    w = world()
    w.fill([0, 1, 2], spilled=1)
    prompt, reply = w.conversations[0]
    request = w.request(prompt)
    w.promote()
    assert request.cache_hit_type == "ssd_hit"
    assert request.cached_tokens == len(prompt)
    assert (prompt + reply) in w.cache

    w.sched._evict_incompatible_entry(request)

    assert (prompt + reply) not in w.cache
    assert w.tier.lookup_ssd(tuple(prompt + reply)) is None


# --- real model: cold and warm answer alike ---------------------------------------------------

VOCAB = 128
MAX_TOKENS = 6
PROMPT_TOKENS = 600  # 300 KiB of KV: three entries fit in 1 MB


class _Tokenizer:
    eos_token_id = 0
    eos_token_ids = {0}
    bos_token = None
    vocab_size = VOCAB

    def decode(self, ids, **kwargs):
        return " ".join(map(str, ids))

    def encode(self, text, **kwargs):
        return [1]


@pytest.fixture(scope="module")
def model():
    mx.random.seed(3)
    net = llama.Model(
        llama.ModelArgs(
            model_type="llama",
            hidden_size=64,
            num_hidden_layers=2,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            rms_norm_eps=1e-5,
            vocab_size=VOCAB,
        )
    )
    net.update(tree_map(lambda p: p * 6.0 if p.ndim > 1 else p, net.parameters()))
    mx.eval(net.parameters())
    return net


def _scheduler(model, ssd_dir=None, memory_mb=1):
    kwargs = {"ssd_cache_dir": str(ssd_dir)} if ssd_dir is not None else {}
    return Scheduler(
        model=model,
        tokenizer=_Tokenizer(),
        config=SchedulerConfig(
            enable_prefix_cache=True,
            use_memory_aware_cache=True,
            cache_memory_mb=memory_mb,
            max_num_seqs=1,
            prefill_batch_size=1,
            completion_batch_size=1,
            **kwargs,
        ),
    )


def _generate(scheduler, request_id, prompt):
    request = Request(
        request_id=request_id,
        prompt="p",
        prompt_token_ids=list(prompt),
        sampling_params=SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0),
    )
    request.num_prompt_tokens = len(prompt)
    scheduler.add_request(request)
    for _ in range(MAX_TOKENS + 80):
        scheduler.step()
        if not scheduler.has_requests():
            break
    assert request.output_token_ids, "the request produced nothing"
    return list(request.output_token_ids), request


def _prompt(seed, n=PROMPT_TOKENS):
    rng = random.Random(seed)
    return [rng.randrange(1, VOCAB) for _ in range(n)]


@pytest.fixture
def full_cache_with_evicted_conversation(model, tmp_path):
    """Conversation 0 was served first, then three others filled the cache."""
    sched = _scheduler(model, tmp_path)
    try:
        prompts = [_prompt(seed) for seed in range(4)]
        outputs = [_generate(sched, f"c{i}", p)[0] for i, p in enumerate(prompts)]
        cache = sched.memory_aware_cache
        assert prompts[0] + outputs[0] not in cache, "conversation 0 must be evicted"
        assert cache._current_memory > 0.7 * cache._max_memory
        _wait_spilled(sched._ssd_tier, 1)
        yield sched, prompts, outputs
    finally:
        sched.close_ssd_tier()


def test_turn_two_after_eviction_is_served_from_the_tier_like_a_cold_request(
    model, full_cache_with_evicted_conversation
):
    sched, prompts, outputs = full_cache_with_evicted_conversation
    turn_two = prompts[0] + outputs[0] + _prompt(99, 40)
    cold, cold_request = _generate(_scheduler(model, memory_mb=64), "cold", turn_two)
    assert cold_request.cached_tokens == 0

    warm, request = _generate(sched, "warm", turn_two)

    assert request.cache_hit_type == "ssd_hit"
    assert request.cached_tokens >= len(prompts[0]), "the tier must serve the turn"
    assert warm == cold
    _within_budget(sched.memory_aware_cache)


def test_an_exact_repeat_after_eviction_is_served_from_the_tier_like_a_cold_request(
    model, full_cache_with_evicted_conversation
):
    sched, prompts, _ = full_cache_with_evicted_conversation
    cold, _ = _generate(_scheduler(model, memory_mb=64), "cold", prompts[0])

    warm, request = _generate(sched, "warm", prompts[0])

    assert request.cache_hit_type == "ssd_hit"
    assert request.cached_tokens >= len(prompts[0]) - 1
    assert warm == cold
    _within_budget(sched.memory_aware_cache)


def test_a_divergent_prompt_after_eviction_is_served_up_to_the_shared_prefix(
    model, full_cache_with_evicted_conversation
):
    sched, prompts, _ = full_cache_with_evicted_conversation
    query = prompts[0][:400] + _prompt(77, 100)
    cold, _ = _generate(_scheduler(model, memory_mb=64), "cold", query)

    warm, request = _generate(sched, "warm", query)

    assert request.cache_hit_type == "ssd_hit"
    assert request.cached_tokens == 400
    assert warm == cold
    _within_budget(sched.memory_aware_cache)
