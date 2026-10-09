# SPDX-License-Identifier: Apache-2.0
"""The MLLM path reports how much of the prompt the prefix cache supplied.

py-vllm-mlx#43: the scheduler's per-request ``cached_tokens`` used to be a
stub. The batch generator now records the reuse at each prefix-cache site and
takes back from the cache counters whatever it credited but did not use, the
same contract as the text scheduler (#46).
"""

import threading
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

mx = pytest.importorskip("mlx.core")
cache_mod = pytest.importorskip("mlx_lm.models.cache")

from vllm_mlx.memory_cache import MemoryAwarePrefixCache, MemoryCacheConfig  # noqa: E402
from vllm_mlx.mllm_batch_generator import (  # noqa: E402
    MLLMBatchGenerator,
    MLLMBatchRequest,
    MLLMBatchStats,
)

LAYERS = 2
PROMPT = list(range(1, 9))


class _Model:
    def __init__(self):
        self.layers = [object() for _ in range(LAYERS)]


def _kv(tokens):
    layer = cache_mod.KVCache()
    layer.update_and_fetch(mx.zeros((1, 1, tokens, 2)), mx.zeros((1, 1, tokens, 2)))
    return layer


def _rotating(tokens):
    layer = cache_mod.RotatingKVCache(max_size=4)
    layer.update_and_fetch(mx.zeros((1, 1, tokens, 2)), mx.zeros((1, 1, tokens, 2)))
    return layer


class _Fresh:
    def merge(self, caches):
        return self


def _generator(monkeypatch, stored_tokens=None, layer=_kv):
    cache = MemoryAwarePrefixCache(
        _Model(),
        MemoryCacheConfig(max_memory_mb=64, max_entries=10, min_prefix_tokens=1),
    )
    if stored_tokens:
        assert cache.store(
            list(stored_tokens), [layer(len(stored_tokens)) for _ in range(LAYERS)]
        )

    monkeypatch.setattr(mx, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(
        "mlx_lm.models.cache.make_prompt_cache", lambda *_, **__: [_Fresh()]
    )
    monkeypatch.setattr(
        "mlx_lm.sample_utils.make_sampler",
        lambda **_: MagicMock(return_value=mx.array([1], dtype=mx.uint32)),
    )
    monkeypatch.setattr("mlx_lm.sample_utils.make_logits_processors", lambda **_: [])

    gen = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
    gen.max_kv_size = 0
    gen._stats = MLLMBatchStats()
    gen._pending_error_responses = []
    gen._aborted_request_ids = set()
    gen._prefill_progress = {}
    gen._cache_reuse = {}
    gen._prefix_checkpoint_lock = threading.Lock()
    gen._request_prefix_checkpoints = {}
    gen.prefix_cache = cache
    gen._think_suffix_len = 0
    gen.prefill_step_size = 512
    gen.language_model = lambda tokens, **kw: mx.zeros((1, tokens.shape[1], 4))
    gen._language_model_kwargs = lambda *a, **k: {}
    gen.model = MagicMock()
    gen.sampler = MagicMock()
    gen._preprocess_request = lambda req: None
    gen._run_chunked_text_prefill = MagicMock(
        return_value=mx.array([[[0.0, 1.0]]], dtype=mx.float32)
    )
    return gen, cache


def _request(text_only=True, request_id="r1", prompt=PROMPT):
    req = MLLMBatchRequest(uid=1, request_id=request_id, prompt="x")
    req.input_ids = mx.array([prompt])
    req.is_text_only = text_only
    return req


def _stats(cache):
    s = cache.get_stats()
    return s["hits"], s["misses"], s["tokens_saved"], s["discarded_hits"]


def test_exact_hit_reports_all_but_the_last_token_and_settles_the_counters(monkeypatch):
    gen, cache = _generator(monkeypatch, PROMPT)
    MLLMBatchGenerator._process_prompts(gen, [_request()])

    assert gen.pop_cached_tokens("r1") == 7
    # fetch credited 8; one token was replayed, so 7 stay credited
    assert _stats(cache) == (1, 0, 7, 0)


def test_prefix_hit_reports_the_matched_prefix(monkeypatch):
    gen, cache = _generator(monkeypatch, PROMPT[:5])
    MLLMBatchGenerator._process_prompts(gen, [_request()])

    assert gen.pop_cached_tokens("r1") == 5
    assert _stats(cache) == (1, 0, 5, 0)


def test_miss_reports_zero(monkeypatch):
    gen, cache = _generator(monkeypatch, None)
    MLLMBatchGenerator._process_prompts(gen, [_request()])

    assert gen.pop_cached_tokens("r1") == 0
    assert _stats(cache)[:3] == (0, 1, 0)


def test_unrewindable_exact_hit_falls_through_and_gives_the_hit_back(monkeypatch):
    gen, cache = _generator(monkeypatch, PROMPT, layer=_rotating)
    MLLMBatchGenerator._process_prompts(gen, [_request()])

    assert gen.pop_cached_tokens("r1") == 0
    assert _stats(cache) == (0, 1, 0, 1)
    gen._run_chunked_text_prefill.assert_called_once()


def test_request_that_cannot_use_the_cache_reports_nothing(monkeypatch):
    gen, cache = _generator(monkeypatch, PROMPT)
    # Media and other ineligible requests skip the prefix cache entirely.
    monkeypatch.setattr(
        "vllm_mlx.mllm_batch_generator.is_text_only_prefix_cache_request",
        lambda req: False,
    )
    MLLMBatchGenerator._process_prompts(gen, [_request()])

    assert gen.pop_cached_tokens("r1") is None
    assert _stats(cache) == (0, 0, 0, 0)


def test_the_value_is_read_once_and_the_table_is_bounded(monkeypatch):
    gen, _ = _generator(monkeypatch, None)
    req = _request()
    gen._record_cache_use(req, 0, 3)
    assert gen.pop_cached_tokens("r1") == 3
    assert gen.pop_cached_tokens("r1") is None
    for i in range(5000):
        gen._record_cache_use(_request(request_id=f"x{i}"), 0, 1)
    assert len(gen._cache_reuse) <= 4096


def test_abort_forgets_the_record(monkeypatch):
    gen, _ = _generator(monkeypatch, None)
    gen._record_cache_use(_request(), 0, 3)
    gen.abort_prefill("r1")
    assert gen.pop_cached_tokens("r1") is None


def test_the_used_count_never_exceeds_the_prompt(monkeypatch):
    gen, _ = _generator(monkeypatch, None)
    gen._record_cache_use(_request(), 0, 99)
    assert gen.pop_cached_tokens("r1") == 8


# --- scheduler and engine plumbing ---------------------------------------------------------


class _Resp:
    def __init__(self, uid, token, finish_reason=None):
        self.uid, self.token, self.finish_reason = uid, token, finish_reason
        self.logprobs = None
        self.prompt_cache = None
        self.from_draft = False
        self.mtp_attempted = False
        self.mtp_attempted_count = 0
        self.specprefill_outcome = None


def _mllm_scheduler(reused):
    from vllm_mlx.mllm_scheduler import MLLMRequest, MLLMScheduler

    s = MLLMScheduler.__new__(MLLMScheduler)
    tok = MagicMock()
    tok.decode = lambda ids: "x"
    s.processor = SimpleNamespace(tokenizer=tok)
    s.uid_to_request_id = {1: "a"}
    s._detokenizer_pool = {}
    s.num_requests_processed = 0
    s.total_completion_tokens = 0
    req = MLLMRequest(request_id="a", prompt="p")
    req.num_prompt_tokens = 8
    s.running = {"a": req}
    s.batch_generator = SimpleNamespace(pop_cached_tokens=lambda rid: reused)
    return s, req


@pytest.mark.parametrize("reused,expected", [(7, 7), (0, 0), (None, None), (99, 8)])
def test_scheduler_puts_the_generators_value_on_its_output(reused, expected):
    s, req = _mllm_scheduler(reused)
    outputs, _ = s._process_batch_responses([_Resp(1, 5, "stop")])
    assert outputs[0].cached_tokens == expected
    assert req.cached_tokens == expected


def test_scheduler_status_reports_the_value():
    s, req = _mllm_scheduler(5)
    s._process_batch_responses([_Resp(1, 5)])
    assert req.cached_tokens == 5
