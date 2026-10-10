# SPDX-License-Identifier: Apache-2.0
"""Every hit kind with an empty remainder is rewound or prefilled (py-vllm-mlx#47).

#46 rewound exact/supersequence memory-cache hits by one position before
feeding the last token. A promoted SSD entry, a paged hit that ends on a block
boundary and a legacy prefix-cache hit reach the same branch with an empty
remainder and a state that covers the whole key; they used to feed the last
token without rewinding, putting it in the KV cache twice.
"""

from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm.models.cache")

from tests.test_exact_hit_reuse import (  # noqa: E402
    LAYERS,
    PROMPT,
    _kv,
    _rotating,
    _run,
    _setup,
)
from vllm_mlx.request import Request  # noqa: E402
from vllm_mlx.scheduler import SamplingParams  # noqa: E402


def _ssd_scheduler(layer_factory=_kv, prompt_len=8, stored_len=8):
    sched, cache, rec = _setup(None)
    stats = SimpleNamespace(promotion_failures=0, ssd_hits=0)
    touched = []
    key = tuple(PROMPT[:stored_len])
    sched._ssd_tier = SimpleNamespace(
        _stats=stats,
        _index=SimpleNamespace(touch=touched.append),
        _read_entry=lambda tokens, path: [layer_factory(stored_len) for _ in range(LAYERS)],
    )
    sched._reconstruct_ssd_layers = lambda layers: layers
    request = Request(
        request_id="r1",
        prompt="p",
        prompt_token_ids=list(PROMPT[:prompt_len]),
        sampling_params=SamplingParams(),
    )
    request.num_prompt_tokens = prompt_len
    sched.add_request(request)
    # The memory cache missed; hand the scheduler an SSD candidate.
    request.cache_hit_type = "ssd_pending"
    request._ssd_candidate = {
        "memory_bytes": 1024,
        "matched_key": key,
        "matched_tokens": stored_len,
        "file_path": "unused",
        "match_type": "exact" if stored_len == prompt_len else "prefix",
    }
    return sched, cache, rec, request, stats


def test_exact_ssd_hit_is_rewound_and_feeds_one_token():
    sched, cache, rec, request, stats = _ssd_scheduler()
    sched._schedule_waiting()

    assert request.cache_hit_type == "ssd_hit"
    assert request.cached_tokens == 7
    assert rec.calls[0]["prompts"] == [[8]]
    assert [layer.offset for layer in rec.calls[0]["caches"][0]] == [7] * LAYERS
    assert stats.ssd_hits == 1
    # The entry promoted into RAM is the stored one and is not rewound.
    stored = cache._entries[tuple(PROMPT)].cache
    assert [layer.offset for layer in stored] == [8] * LAYERS


def test_exact_ssd_hit_on_a_rotating_window_is_prefilled():
    sched, _, rec, request, _ = _ssd_scheduler(layer_factory=_rotating)
    # Let the entry past the KV-bound check so the rewind guard decides.
    sched._restored_cache_matches_kv_bound = lambda cache: True
    sched._schedule_waiting()

    assert request.cached_tokens == 0
    assert rec.calls[0]["prompts"] == [PROMPT]
    assert rec.calls[0].get("caches") in (None, [None])


def test_prefix_ssd_hit_keeps_its_remainder():
    sched, _, rec, request, _ = _ssd_scheduler(stored_len=5)
    sched._schedule_waiting()

    assert request.cached_tokens == 5
    assert rec.calls[0]["prompts"] == [PROMPT[5:]]
    assert [layer.offset for layer in rec.calls[0]["caches"][0]] == [5] * LAYERS


class _PagedFake:
    def __init__(self, layer_factory=_kv, tokens=8):
        self.layer_factory = layer_factory
        self.tokens = tokens

    def fetch_cache(self, request_id, prompt):
        table = SimpleNamespace(num_tokens=self.tokens, block_ids=[1, 2])
        return table, list(prompt[self.tokens :])

    def reconstruct_cache(self, table):
        return [self.layer_factory(table.num_tokens) for _ in range(LAYERS)]


def test_paged_hit_ending_on_a_block_boundary_is_rewound():
    sched, _, rec = _setup(None)
    sched.memory_aware_cache = None
    sched.block_aware_cache = _PagedFake()
    request = _run(sched, PROMPT)

    assert request.cache_hit_type == "hit"
    assert request.cached_tokens == 7
    assert rec.calls[0]["prompts"] == [[8]]
    assert [layer.offset for layer in rec.calls[0]["caches"][0]] == [7] * LAYERS


def test_paged_partial_hit_is_untouched():
    sched, _, rec = _setup(None)
    sched.memory_aware_cache = None
    sched.block_aware_cache = _PagedFake(tokens=4)
    request = _run(sched, PROMPT)

    assert request.cached_tokens == 4
    assert rec.calls[0]["prompts"] == [PROMPT[4:]]


class _LegacyFake:
    def __init__(self, layer_factory=_kv):
        self.layer_factory = layer_factory

    def fetch_cache(self, prompt):
        return [self.layer_factory(len(prompt)) for _ in range(LAYERS)], []


def test_legacy_prefix_cache_exact_hit_is_rewound():
    sched, _, rec = _setup(None)
    sched.memory_aware_cache = None
    sched.prefix_cache = _LegacyFake()
    request = _run(sched, PROMPT)

    assert request.cached_tokens == 7
    assert rec.calls[0]["prompts"] == [[8]]
    assert [layer.offset for layer in rec.calls[0]["caches"][0]] == [7] * LAYERS


def test_legacy_prefix_cache_non_rewindable_exact_hit_is_prefilled():
    sched, _, rec = _setup(None)
    sched.memory_aware_cache = None
    sched.prefix_cache = _LegacyFake(layer_factory=_rotating)
    sched._restored_cache_matches_kv_bound = lambda cache: True
    request = _run(sched, PROMPT)

    assert request.cached_tokens == 0
    assert rec.calls[0]["prompts"] == [PROMPT]


def _rotating_layer_dict(tokens=8, max_size=4):
    return {
        "keys": mx.zeros((1, 1, tokens, 4)).tolist(),
        "values": mx.zeros((1, 1, tokens, 4)).tolist(),
        "offset": tokens,
        "max_size": max_size,
        "keep": 0,
        "step": 256,
        "_idx": tokens,
    }


def test_rotating_window_restored_from_ssd_is_not_rewindable():
    from vllm_mlx.scheduler import _is_exactly_rewindable

    sched, _, _ = _setup(None)
    layers = sched._reconstruct_ssd_layers([_rotating_layer_dict()])
    assert layers is not None
    assert not _is_exactly_rewindable(layers[0])


def test_exact_ssd_hit_on_a_restored_rotating_window_is_prefilled():
    sched, _, rec, request, _ = _ssd_scheduler()
    sched._ssd_tier._read_entry = lambda tokens, path: [
        _rotating_layer_dict() for _ in range(LAYERS)
    ]
    del sched._reconstruct_ssd_layers  # use the real reconstruction
    sched._restored_cache_matches_kv_bound = lambda cache: True
    sched._schedule_waiting()

    assert request.cached_tokens == 0
    assert rec.calls[0]["prompts"] == [PROMPT]


def test_native_quantized_kv_layer_is_not_rewindable():
    from vllm_mlx.scheduler import _is_exactly_rewindable

    plain = _kv(8, dim=64)
    assert _is_exactly_rewindable(plain)
    quantized = plain.to_quantized(group_size=32, bits=8)
    assert isinstance(quantized.keys, (tuple, list))
    assert not _is_exactly_rewindable(quantized)
