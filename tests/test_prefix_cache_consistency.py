# SPDX-License-Identifier: Apache-2.0
"""The other prefix caches never serve KV behind a prefix it was not computed after.

#104 was found in the paged cache, whose blocks were keyed without their parent
hash. The token-keyed caches (memory-aware, trie) store a whole sequence under
its full token tuple. These tests run the consistency probe's scenarios (repeat,
extension, branch, chain swap, shift, deletion) against them with KV that
encodes its own prefix: the value at position ``p`` is a hash of
``tokens[:p + 1]``, so any entry reused behind another prefix, at another
position, or trimmed wrongly, is visible in the returned cache.
"""

import pytest

mx = pytest.importorskip("mlx.core")
cache_mod = pytest.importorskip("mlx_lm.models.cache")

from vllm_mlx.memory_cache import (  # noqa: E402
    MemoryAwarePrefixCache,
    MemoryCacheConfig,
)
from vllm_mlx.prefix_cache import PrefixCacheManager  # noqa: E402

BS = 4
A, B, C, Z, Z1 = ([v] * BS for v in (1, 3, 5, 2, 7))
TAIL = [9]


def _prefix_values(tokens):
    h = 0
    values = []
    for token in tokens:
        h = (h * 31 + token + 1) % 65521
        values.append(float(h))
    return values


def _layer(tokens):
    n = len(tokens)
    vals = mx.array(_prefix_values(tokens), dtype=mx.float32)
    keys = mx.broadcast_to(vals.reshape(1, 1, n, 1), (1, 2, n, 2))
    layer = cache_mod.KVCache()
    layer.keys = mx.array(keys)
    layer.values = mx.array(keys + 0.5)
    layer.offset = n
    return layer


class _Memory:
    name = "memory-aware"

    def __init__(self):
        self.cache = MemoryAwarePrefixCache(
            object(),
            MemoryCacheConfig(max_memory_mb=64, max_entries=32, min_prefix_tokens=1),
        )

    def store(self, tokens):
        self.cache.store(list(tokens), [_layer(tokens)])

    def fetch(self, tokens):
        return self.cache.fetch(list(tokens))


class _Trie:
    name = "trie"

    def __init__(self):
        self.cache = PrefixCacheManager(object(), max_entries=32)

    def store(self, tokens):
        self.cache.store_cache(list(tokens), [_layer(tokens)])

    def fetch(self, tokens):
        return self.cache.fetch_cache(list(tokens))


@pytest.fixture(params=[_Memory, _Trie], ids=lambda c: c.name)
def variant(request):
    return request.param()


def _assert_what_a_cold_prefill_writes(variant, query):
    """Whatever the cache returns is the cold KV of ``query``'s own prefix."""
    cache, remaining = variant.fetch(query)
    if not cache:
        assert remaining == query
        return 0
    covered = len(query) - len(remaining)
    assert query[covered:] == remaining
    layer = cache[0]
    # the cache may hold a longer buffer; only [:offset] is the cached prefix
    assert layer.offset == covered
    expected = mx.array(_prefix_values(query)[:covered], dtype=mx.float32)
    got = layer.keys[0, 0, :covered, 0]
    assert mx.array_equal(got, expected)
    return covered


def test_repeat_extension_and_branch(variant):
    p = A + B + C + TAIL
    variant.store(p)
    assert _assert_what_a_cold_prefill_writes(variant, p) > 0  # repeat
    assert _assert_what_a_cold_prefill_writes(variant, p + Z) == len(p)
    assert _assert_what_a_cold_prefill_writes(variant, A + B + Z) in (0, 2 * BS)


def test_chain_swap(variant):
    variant.store(Z + C + TAIL)
    variant.store(A + B + TAIL)
    # A+C: A's tokens were cached behind nothing, C behind Z
    covered = _assert_what_a_cold_prefill_writes(variant, A + C + TAIL)
    assert covered <= BS  # at most the common prefix with A+B


def test_shift_and_deletion(variant):
    variant.store(Z + Z1 + C + TAIL)
    assert _assert_what_a_cold_prefill_writes(variant, A + C + TAIL) == 0
    variant.store(A + Z1 + C + TAIL)
    covered = _assert_what_a_cold_prefill_writes(variant, A + C + TAIL)
    assert covered <= BS


def test_supersequence_is_trimmed_to_the_query(variant):
    variant.store(A + B + C + TAIL)
    covered = _assert_what_a_cold_prefill_writes(variant, A + B)
    assert covered in (0, 2 * BS)
