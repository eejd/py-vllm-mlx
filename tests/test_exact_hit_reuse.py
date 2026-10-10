# SPDX-License-Identifier: Apache-2.0
"""Exact prefix-cache hits are reused, and the hit counters describe reuse.

An exact (or supersequence) match used to be counted as a hit, then dropped by
the scheduler, which prefilled the whole prompt (py-vllm-mlx#39). Now a
rewindable entry is cut back one position on a copy and only the last token is
fed; anything else is prefilled and the credit is taken back from the counters.
Persisted entries are also held quantized when KV quantization is on (#40).
"""

from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")
cache_mod = pytest.importorskip("mlx_lm.models.cache")

from vllm_mlx.memory_cache import (  # noqa: E402
    MemoryAwarePrefixCache,
    MemoryCacheConfig,
    _QuantizedCacheWrapper,
)
from vllm_mlx.request import Request  # noqa: E402
from vllm_mlx.scheduler import SamplingParams, Scheduler, SchedulerConfig  # noqa: E402

LAYERS = 3


class _Layer:
    pass


class _Model:
    def __init__(self) -> None:
        self.layers = [_Layer() for _ in range(LAYERS)]


def _kv(tokens: int, dim: int = 4):
    layer = cache_mod.KVCache()
    layer.update_and_fetch(
        mx.arange(tokens * dim, dtype=mx.float32).reshape(1, 1, tokens, dim),
        mx.arange(tokens * dim, dtype=mx.float32).reshape(1, 1, tokens, dim),
    )
    return layer


def _rotating(tokens: int):
    layer = cache_mod.RotatingKVCache(max_size=64)
    layer.update_and_fetch(mx.zeros((1, 1, tokens, 4)), mx.zeros((1, 1, tokens, 4)))
    return layer


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def insert(self, prompts, **kwargs):
        self.calls.append({"prompts": [list(p) for p in prompts], **kwargs})
        return [1]


def _setup(entry_tokens, layer_factory=_kv, max_kv_size=0):
    sched = Scheduler(
        model=_Model(),
        tokenizer=SimpleNamespace(eos_token_id=0, eos_token_ids={0}),
        config=SchedulerConfig(enable_prefix_cache=False, max_kv_size=max_kv_size),
    )
    cache = MemoryAwarePrefixCache(
        _Model(),
        MemoryCacheConfig(max_memory_mb=64, max_entries=10, min_prefix_tokens=1),
    )
    sched.memory_aware_cache = cache
    if entry_tokens:
        n = len(entry_tokens)
        assert cache.store(
            list(entry_tokens), [layer_factory(n) for _ in range(LAYERS)]
        )
    rec = _Recorder()
    sched.batch_generator = rec
    params = SamplingParams()
    sched._current_sampler_params = (params.temperature, params.top_p, params.min_p)
    return sched, cache, rec


def _run(sched, prompt, request_id="r1"):
    request = Request(
        request_id=request_id,
        prompt="p",
        prompt_token_ids=list(prompt),
        sampling_params=SamplingParams(),
    )
    request.num_prompt_tokens = len(prompt)
    sched.add_request(request)
    sched._schedule_waiting()
    return request


def _stats(cache):
    s = cache.get_stats()
    return s["hits"], s["misses"], s["tokens_saved"], s["discarded_hits"]


PROMPT = list(range(1, 9))


def test_exact_repeat_reuses_all_but_the_last_token():
    sched, cache, rec = _setup(PROMPT)
    request = _run(sched, PROMPT)

    assert request.cache_hit_type == "exact"
    assert request.cached_tokens == 7
    assert rec.calls[0]["prompts"] == [[8]]
    used = rec.calls[0]["caches"][0]
    assert [layer.offset for layer in used] == [7] * LAYERS
    assert _stats(cache) == (1, 0, 7, 0)


def test_the_stored_entry_is_not_rewound_by_the_reuse():
    sched, cache, _ = _setup(PROMPT)
    _run(sched, PROMPT)
    stored = cache._entries[tuple(PROMPT)].cache
    assert [layer.offset for layer in stored] == [8] * LAYERS
    # and a second repeat still reuses 7 tokens
    sched2_request = _run(sched, PROMPT, request_id="r2")
    assert sched2_request.cached_tokens == 7


def test_supersequence_match_is_reused_the_same_way():
    sched, cache, rec = _setup(PROMPT + [9, 10])
    request = _run(sched, PROMPT)

    assert request.cache_hit_type == "supersequence"
    assert request.cached_tokens == 7
    assert rec.calls[0]["prompts"] == [[8]]
    assert [layer.offset for layer in rec.calls[0]["caches"][0]] == [7] * LAYERS
    assert _stats(cache) == (1, 0, 7, 0)


def test_a_non_rewindable_entry_is_prefilled_and_the_credit_is_taken_back():
    sched, cache, rec = _setup(PROMPT, layer_factory=_rotating)
    request = _run(sched, PROMPT)

    assert request.cached_tokens == 0
    assert rec.calls[0]["prompts"] == [PROMPT]
    assert rec.calls[0].get("caches") in (None, [None])
    assert _stats(cache) == (0, 1, 0, 1)


def test_a_prefix_hit_that_is_used_in_full_keeps_its_credit():
    sched, cache, rec = _setup(PROMPT[:5])
    request = _run(sched, PROMPT)

    assert request.cache_hit_type == "prefix"
    assert request.cached_tokens == 5
    assert rec.calls[0]["prompts"] == [PROMPT[5:]]
    assert _stats(cache) == (1, 0, 5, 0)


def test_an_entry_rejected_for_the_kv_bound_is_not_counted_as_a_hit():
    sched, cache, rec = _setup(PROMPT[:5], max_kv_size=64)
    request = _run(sched, PROMPT)

    assert request.cached_tokens == 0
    assert rec.calls[0]["prompts"] == [PROMPT]
    assert _stats(cache)[0] == 0
    assert _stats(cache)[3] == 1


def test_settling_twice_does_not_subtract_twice():
    sched, cache, _ = _setup(PROMPT[:5], max_kv_size=64)
    request = _run(sched, PROMPT)
    sched._settle_cache_use(request, 0)
    sched._settle_cache_use(request, 0)
    assert _stats(cache) == (0, 1, 0, 1)


def test_settle_hit_partial_and_full_and_never_negative():
    cache = MemoryAwarePrefixCache(
        _Model(), MemoryCacheConfig(max_memory_mb=64, min_prefix_tokens=1)
    )
    cache.store(PROMPT, [_kv(8) for _ in range(LAYERS)])
    cache.fetch(PROMPT)
    assert _stats(cache) == (1, 0, 8, 0)
    cache.settle_hit(8, 8)  # nothing to correct
    assert _stats(cache) == (1, 0, 8, 0)
    cache.settle_hit(8, 7)  # one token not used
    assert _stats(cache) == (1, 0, 7, 0)
    cache.settle_hit(8, 0)  # whole hit unused
    assert _stats(cache) == (0, 1, 0, 1)
    cache.settle_hit(100, 0)  # more than was ever credited: floors at zero
    assert min(_stats(cache)) >= 0


def test_discarded_hits_is_reported_in_the_stats_and_cache_state():
    sched, cache, _ = _setup(PROMPT[:5], max_kv_size=64)
    assert cache.get_stats()["discarded_hits"] == 0
    _run(sched, PROMPT)
    assert cache.get_stats()["discarded_hits"] == 1
    from vllm_mlx import cache_state

    assert "discarded_hits" in cache_state._COUNTER_KEYS


def test_exact_reuse_with_kv_quantization_on():
    sched, cache, rec = _setup(None)
    cache = MemoryAwarePrefixCache(
        _Model(),
        MemoryCacheConfig(
            max_memory_mb=64,
            min_prefix_tokens=1,
            kv_quantize=True,
            kv_bits=8,
            kv_group_size=64,
            kv_min_quantize_tokens=4,
        ),
    )
    sched.memory_aware_cache = cache
    assert cache.store(PROMPT, [_kv(8, dim=64) for _ in range(LAYERS)])
    request = _run(sched, PROMPT)

    assert request.cached_tokens == 7
    assert rec.calls[0]["prompts"] == [[8]]
    assert all(
        not isinstance(layer, _QuantizedCacheWrapper)
        for layer in rec.calls[0]["caches"][0]
    )
    assert [layer.offset for layer in rec.calls[0]["caches"][0]] == [7] * LAYERS
    assert _stats(cache) == (1, 0, 7, 0)


def test_reschedule_after_an_error_takes_the_whole_hit_back_once():
    sched, cache, _ = _setup(PROMPT[:5])
    request = _run(sched, PROMPT)
    assert _stats(cache) == (1, 0, 5, 0)
    request.status = __import__(
        "vllm_mlx.request", fromlist=["RequestStatus"]
    ).RequestStatus.RUNNING
    sched.running[request.request_id] = request
    sched._reschedule_running_requests()
    assert _stats(cache) == (0, 1, 0, 1)
    sched._settle_cache_use(request, 0)
    assert _stats(cache) == (0, 1, 0, 1)


def test_a_recurrent_layer_is_not_rewindable():
    from vllm_mlx.scheduler import _is_exactly_rewindable

    assert not _is_exactly_rewindable(cache_mod.ArraysCache(size=2))


def test_a_container_layer_is_not_rewindable():
    from vllm_mlx.scheduler import _is_exactly_rewindable

    assert not _is_exactly_rewindable(cache_mod.CacheList(_kv(4), _kv(4)))


# --- #40: persisted entries are quantized on load like fresh ones ---------------


def _save(tmp_path, tokens=8):
    src = MemoryAwarePrefixCache(
        _Model(), MemoryCacheConfig(max_memory_mb=64, min_prefix_tokens=1)
    )
    assert src.store(list(range(tokens)), [_kv(tokens, dim=64) for _ in range(LAYERS)])
    assert src.save_to_disk(str(tmp_path))
    return src


def _loader(**kwargs):
    return MemoryAwarePrefixCache(
        _Model(),
        MemoryCacheConfig(max_memory_mb=64, min_prefix_tokens=1, **kwargs),
    )


def test_loaded_entries_are_quantized_when_kv_quantization_is_on(tmp_path):
    _save(tmp_path)
    plain = _loader()
    assert plain.load_from_disk(str(tmp_path)) == 1
    quant = _loader(
        kv_quantize=True, kv_bits=8, kv_group_size=64, kv_min_quantize_tokens=4
    )
    assert quant.load_from_disk(str(tmp_path)) == 1

    layers = next(iter(quant._entries.values())).cache
    assert all(isinstance(layer, _QuantizedCacheWrapper) for layer in layers)
    plain_layers = next(iter(plain._entries.values())).cache
    assert not any(isinstance(layer, _QuantizedCacheWrapper) for layer in plain_layers)
    assert quant._current_memory < plain._current_memory


def test_a_loaded_quantized_entry_still_serves_an_exact_fetch(tmp_path):
    _save(tmp_path)
    quant = _loader(
        kv_quantize=True, kv_bits=8, kv_group_size=64, kv_min_quantize_tokens=4
    )
    quant.load_from_disk(str(tmp_path))
    cache, rest = quant.fetch(list(range(8)))
    assert cache is not None and rest == []
    assert not any(isinstance(layer, _QuantizedCacheWrapper) for layer in cache)


def test_short_entries_stay_unquantized_on_load(tmp_path):
    _save(tmp_path)
    quant = _loader(
        kv_quantize=True, kv_bits=8, kv_group_size=64, kv_min_quantize_tokens=64
    )
    assert quant.load_from_disk(str(tmp_path)) == 1
    layers = next(iter(quant._entries.values())).cache
    assert not any(isinstance(layer, _QuantizedCacheWrapper) for layer in layers)


def test_quantized_size_decides_whether_an_entry_fits(tmp_path):
    _save(tmp_path, tokens=64)
    plain = _loader()
    plain.load_from_disk(str(tmp_path))
    fp_bytes = plain._current_memory
    # a limit between the quantized and the fp16 size: only quantized fits
    limit_mb = (fp_bytes * 0.8) / (1024 * 1024)
    tight = MemoryAwarePrefixCache(
        _Model(),
        MemoryCacheConfig(
            max_memory_mb=max(limit_mb, 0.01),
            min_prefix_tokens=1,
            kv_quantize=True,
            kv_bits=8,
            kv_group_size=64,
            kv_min_quantize_tokens=4,
        ),
    )
    assert tight.load_from_disk(str(tmp_path)) == 1
    # negative control: the same limit refuses the unquantized entry
    fp_tight = MemoryAwarePrefixCache(
        _Model(),
        MemoryCacheConfig(max_memory_mb=max(limit_mb, 0.01), min_prefix_tokens=1),
    )
    assert fp_tight.load_from_disk(str(tmp_path)) == 0


def test_an_entry_that_cannot_be_quantized_is_kept_unquantized(tmp_path):
    src = MemoryAwarePrefixCache(
        _Model(), MemoryCacheConfig(max_memory_mb=64, min_prefix_tokens=1)
    )
    assert src.store(list(range(8)), [_kv(8, dim=48) for _ in range(LAYERS)])
    assert src.save_to_disk(str(tmp_path))
    quant = _loader(
        kv_quantize=True, kv_bits=8, kv_group_size=64, kv_min_quantize_tokens=4
    )
    assert quant.load_from_disk(str(tmp_path)) == 1  # dim 48 % 64 != 0
    layers = next(iter(quant._entries.values())).cache
    assert not any(isinstance(layer, _QuantizedCacheWrapper) for layer in layers)


def test_quantized_store_save_load_round_trip_stays_quantized(tmp_path):
    src = _loader(
        kv_quantize=True, kv_bits=8, kv_group_size=64, kv_min_quantize_tokens=4
    )
    assert src.store(list(range(8)), [_kv(8, dim=64) for _ in range(LAYERS)])
    assert src.save_to_disk(str(tmp_path))
    dst = _loader(
        kv_quantize=True, kv_bits=8, kv_group_size=64, kv_min_quantize_tokens=4
    )
    assert dst.load_from_disk(str(tmp_path)) == 1
    layers = next(iter(dst._entries.values())).cache
    assert all(isinstance(layer, _QuantizedCacheWrapper) for layer in layers)
    assert dst._current_memory == src._current_memory


def test_rotating_windows_are_never_treated_as_rewindable_even_if_they_claim_to_be():
    from vllm_mlx.scheduler import _is_exactly_rewindable

    class _Claims(cache_mod.RotatingKVCache):
        def is_trimmable(self):
            return True

    assert _is_exactly_rewindable(_kv(4))
    assert not _is_exactly_rewindable(_Claims(max_size=16))
