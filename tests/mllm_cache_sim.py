"""Simulated language model + helpers for MLLM batch-generator cache identity tests.

The model writes into every KV layer, at absolute position ``p``, the chain
hash of ``tokens[:p+1]`` (``h_p = (h_{p-1} * 31 + token + 1) mod MOD``) and
predicts the next token from the hash of the last position. A cache entry is
therefore only valid behind *exactly* the token prefix it was computed from:
extra positions, missing positions or KV computed behind another prefix are
visible in the tensors and change the generated tokens.

``mtp_forward`` is a native-MTP head that proposes the true next token
(``draft_correct=True``, every verify step is accepted) or a wrong one
(``draft_correct=False``, every verify step is rejected).
"""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache, BatchKVCache, KVCache

MOD = 9973
VOCAB = 64
LAYERS = 2


def chain(prev: int, token: int) -> int:
    return (prev * 31 + token + 1) % MOD


def chain_hashes(tokens, start: int = 0) -> list[int]:
    """Expected KV value at every position of ``tokens``."""
    out = []
    h = start
    for t in tokens:
        h = chain(h, t)
        out.append(h)
    return out


def next_token(tokens) -> int:
    """Greedy continuation of ``tokens`` under the simulated model."""
    return chain_hashes(tokens)[-1] % VOCAB


def greedy_continuation(tokens, n: int) -> list[int]:
    seq = list(tokens)
    out = []
    for _ in range(n):
        t = next_token(seq)
        out.append(t)
        seq.append(t)
    return out


class SimLanguageModel:
    """Language model with hash-chain KV (see module docstring)."""

    def __init__(
        self,
        draft_correct: bool = True,
        with_mtp: bool = False,
        hybrid: bool = False,
    ):
        self.draft_correct = draft_correct
        self.hybrid = hybrid
        self.calls: list[list[list[int]]] = []
        # (shape of the tokens fed, rope_deltas kwarg as a list or None) per call
        self.rope_deltas_log: list[tuple] = []
        if with_mtp:
            self.mtp = object()
        else:
            self.mtp_forward = None  # type: ignore[assignment]

    # mlx_lm.make_prompt_cache builds the per-request cache from this.
    def make_cache(self):
        caches = [KVCache() for _ in range(LAYERS)]
        if self.hybrid:
            # A recurrent layer (like Qwen3.5's GatedDeltaNet) holding the hash
            # of the last position: it cannot be trimmed, only snapshotted.
            caches.append(ArraysCache(1))
        return caches

    def _kv_layers(self, cache):
        return [c for c in cache if not isinstance(c, ArraysCache)]

    @staticmethod
    def _last_hash(cache_layer, batch: int) -> mx.array:
        if isinstance(cache_layer, BatchKVCache):
            if cache_layer._idx == 0:
                return mx.zeros((batch,), dtype=mx.int32)
            return cache_layer.keys[:, 0, cache_layer._idx - 1, 0].astype(mx.int32)
        if cache_layer.offset == 0:
            return mx.zeros((batch,), dtype=mx.int32)
        return cache_layer.keys[:, 0, cache_layer.offset - 1, 0].astype(mx.int32)

    def __call__(self, tokens, cache=None, return_hidden=False, **kwargs):
        rope_deltas = kwargs.get("rope_deltas")
        self.rope_deltas_log.append(
            (tuple(tokens.shape), None if rope_deltas is None else rope_deltas.tolist())
        )
        if tokens.ndim == 1:
            tokens = tokens[None, :]
        batch, length = tokens.shape
        self.calls.append(tokens.tolist())
        if self.hybrid:
            state = cache[-1][0]
            prev = (
                mx.zeros((batch,), dtype=mx.int32)
                if state is None
                else state[:, 0].astype(mx.int32)
            )
        else:
            prev = self._last_hash(cache[0], batch)
        hashes = []
        for j in range(length):
            prev = (prev * 31 + tokens[:, j].astype(mx.int32) + 1) % MOD
            hashes.append(prev)
        h = mx.stack(hashes, axis=1)  # [B, L]
        for layer, c in enumerate(self._kv_layers(cache)):
            kv = (h + layer).astype(mx.float32)[:, None, :, None]
            c.update_and_fetch(kv, kv)
        if self.hybrid:
            cache[-1][0] = h[:, -1:].astype(mx.float32)
        logits = mx.where(
            mx.arange(VOCAB)[None, None, :] == (h % VOCAB)[:, :, None], 10.0, 0.0
        )
        if return_hidden:
            return logits, h[:, :, None].astype(mx.float32)
        return logits

    def mtp_forward(self, hidden, next_token_ids, mtp_cache=None):
        """Draft logits for the token after ``next_token_ids`` given ``hidden``."""
        h = hidden[:, -1, 0].astype(mx.int32)
        nxt = (h * 31 + next_token_ids[:, 0].astype(mx.int32) + 1) % MOD % VOCAB
        if not self.draft_correct:
            nxt = (nxt + 1) % VOCAB
        return mx.where(
            mx.arange(VOCAB)[None, None, :] == nxt[:, None, None], 10.0, 0.0
        )


def make_generator(
    *,
    mtp: bool | None = None,
    draft_correct: bool = True,
    prefix_cache: bool = True,
    chunked_prefill: int = 0,
    prefill_step_size: int = 1024,
    min_prefix_tokens: int = 1,
    hybrid: bool = False,
):
    """Real ``MLLMBatchGenerator`` over the simulated model."""
    from vllm_mlx.memory_cache import MemoryCacheConfig
    from vllm_mlx.mllm_batch_generator import (
        MLLMBatchGenerator,
        install_chunked_prefill_mllm,
        install_mtp_mllm,
    )

    lm = SimLanguageModel(
        draft_correct=draft_correct, with_mtp=bool(mtp), hybrid=hybrid
    )
    model = SimpleNamespace(language_model=lm)
    gen = MLLMBatchGenerator(
        model,
        SimpleNamespace(),
        max_tokens=64,
        prefill_batch_size=4,
        completion_batch_size=8,
        prefill_step_size=prefill_step_size,
        prefix_cache_config=(
            MemoryCacheConfig(max_memory_mb=64, min_prefix_tokens=min_prefix_tokens)
            if prefix_cache
            else None
        ),
        enable_vision_cache=False,
    )
    if chunked_prefill:
        install_chunked_prefill_mllm(gen, budget=chunked_prefill)
    if mtp:
        install_mtp_mllm(gen, lm)
    return gen, lm


def make_request(request_id: str, ids, max_tokens: int, **kwargs):
    from vllm_mlx.mllm_batch_generator import MLLMBatchRequest

    req = MLLMBatchRequest(
        uid=0,
        request_id=request_id,
        prompt="",
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
        **kwargs,
    )
    req.input_ids = mx.array([list(ids)])
    req.is_text_only = True
    return req


def run_to_completion(gen, requests, max_steps: int = 500):
    """Drive the generator until every request finished; tokens per request id."""
    gen.insert(requests)
    out: dict[str, list[int]] = {r.request_id: [] for r in requests}
    done: set[str] = set()
    for _ in range(max_steps):
        for resp in gen.next():
            if resp.request_id in done:
                continue
            out[resp.request_id].append(resp.token)
            if resp.finish_reason is not None:
                done.add(resp.request_id)
        if len(done) == len(requests):
            return out
    raise AssertionError(f"generator did not finish: {out}")


def stored_entries(gen):
    """``{key: cache layers}`` of every entry in the generator's prefix cache."""
    return {tuple(k): e.cache for k, e in gen.prefix_cache._entries.items()}


def layer_offsets(cache_layers):
    return [int(c.offset) for c in cache_layers]


def layer_values(cache_layers, layer: int = 0) -> list[float]:
    """KV values (position axis) actually held by one layer of a stored entry."""
    c = cache_layers[layer]
    return c.keys[0, 0, : int(c.offset), 0].tolist()
