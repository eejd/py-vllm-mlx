# SPDX-License-Identifier: Apache-2.0
"""Simple-engine system-prompt KV snapshot: reused KV must be the KV of the request.

The tensors here are simulated so that the value at position ``p`` is a hash of
``tokens[:p + 1]`` (a chain: each position folds its token into the previous
position's value). KV that was computed behind another prefix, or overwritten by
another request, therefore differs visibly from the KV a cold prefill produces.

The aliasing under test: ``KVCache.state`` returns the live ``keys``/``values``
objects and ``update_and_fetch`` assigns into them in place, so two caches holding
the same Python array object write into each other's buffer.
"""

from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import numpy as np
import pytest
from mlx_lm.models.cache import (
    ArraysCache,
    CacheList,
    KVCache,
    QuantizedKVCache,
)

from vllm_mlx.engine.base import GenerationOutput
from vllm_mlx.engine.simple import SimpleEngine

MOD = 65521


def _chain(prev: int, token: int) -> int:
    return (prev * 31 + token + 1) % MOD


def _chain_values(tokens, prev=0):
    values = []
    for token in tokens:
        prev = _chain(prev, token)
        values.append(prev)
    return values


# --------------------------------------------------------------------------- #
# Cache-class level: snapshot / restore isolation for every supported layer type
# --------------------------------------------------------------------------- #


class _KVAdapter:
    """``KVCache``: position p of keys/values is the chain hash of tokens[:p+1]."""

    def new(self):
        return KVCache()

    def feed(self, cache, tokens, start):
        values = _chain_values(tokens)[start:]
        kv = mx.array(values, dtype=mx.float32).reshape(1, 1, len(values), 1)
        kv = mx.broadcast_to(kv, (1, 2, len(values), 4)) + 0
        cache.update_and_fetch(kv, kv + 1)

    def observed(self, cache):
        offset = cache.offset
        return [
            np.array(cache.keys[..., :offset, :]),
            np.array(cache.values[..., :offset, :]),
            offset,
        ]

    def rewind(self, cache, to):
        cache.trim(cache.offset - to)


class _QuantizedAdapter(_KVAdapter):
    def new(self):
        return QuantizedKVCache(group_size=64, bits=8)

    def feed(self, cache, tokens, start):
        values = _chain_values(tokens)[start:]
        kv = mx.array(values, dtype=mx.float32).reshape(1, 1, len(values), 1)
        ramp = mx.arange(64, dtype=mx.float32).reshape(1, 1, 1, 64) / 64.0
        kv = kv * (1.0 + ramp) + ramp
        cache.update_and_fetch(kv, kv + 1)

    def observed(self, cache):
        offset = cache.offset
        return [np.array(a[..., :offset, :]) for a in (*cache.keys, *cache.values)] + [
            offset
        ]


class _ArraysAdapter:
    """Recurrent state: one array holding the chain hash of the whole prefix."""

    def new(self):
        return ArraysCache(size=1)

    def feed(self, cache, tokens, start):
        prev = 0
        for token in tokens:
            prev = _chain(prev, token)
            # what mlx-lm's recurrent layers do: replace the entry
            cache[0] = mx.array([float(prev)], dtype=mx.float32)

    def observed(self, cache):
        return [np.array(a) for a in cache.state]

    rewind = None


class _ListAdapter:
    """``CacheList`` (container cache): a KV child and a recurrent child."""

    def __init__(self):
        self.kv, self.arrays = _KVAdapter(), _ArraysAdapter()

    def new(self):
        return CacheList(KVCache(), ArraysCache(size=1))

    def feed(self, cache, tokens, start):
        self.kv.feed(cache.caches[0], tokens, start)
        self.arrays.feed(cache.caches[1], tokens, start)

    def observed(self, cache):
        return [
            self.kv.observed(cache.caches[0]),
            self.arrays.observed(cache.caches[1]),
        ]

    rewind = None


ADAPTERS = {
    "kv": _KVAdapter,
    "quantized_kv": _QuantizedAdapter,
    "arrays": _ArraysAdapter,
    "cache_list": _ListAdapter,
}


def _same(a, b):
    if isinstance(a, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray):
        return a.shape == b.shape and bool((a == b).all())
    return a == b


def _cold(adapter, tokens):
    cache = adapter.new()
    adapter.feed(cache, tokens, 0)
    return cache


SYSTEM = [3, 1, 4, 1, 5, 9, 2, 6]
CONV_A = SYSTEM + [10, 11, 12, 13]
CONV_B = SYSTEM + [20, 21, 22]


def _needs_state_only_contract(name):
    """Quantized and nested layers round-trip through ``.state`` alone only on the
    mlx-lm 0.32 contract (up to 0.31 ``meta_state`` carries the offset); the engine
    never snapshots them (``_cache_class_is_system_snapshot_safe``)."""
    from vllm_mlx import mlx_cache_compat as compat

    return name in ("quantized_kv", "cache_list") and compat.uses_meta_state(
        QuantizedKVCache
    )


@pytest.fixture(params=sorted(ADAPTERS))
def adapter(request):
    if _needs_state_only_contract(request.param):
        pytest.skip("layer needs meta_state on this mlx-lm; engine never snapshots it")
    return ADAPTERS[request.param]()


def test_decode_behind_a_snapshot_leaves_the_conversation_that_took_it_intact(adapter):
    """The request that took the snapshot keeps decoding on its own cache; another
    request restored from the snapshot must not write into that cache (which the
    engine files in the prefix trie)."""
    live = _cold(adapter, SYSTEM)
    snapshot = SimpleEngine._snapshot_prompt_cache([live])

    adapter.feed(live, CONV_A, len(SYSTEM))  # request A decodes on `live`
    other = adapter.new()
    SimpleEngine._restore_prompt_cache([other], snapshot)
    adapter.feed(other, CONV_B, len(SYSTEM))  # request B reuses the system KV

    assert _same(adapter.observed(live), adapter.observed(_cold(adapter, CONV_A)))
    assert _same(adapter.observed(other), adapter.observed(_cold(adapter, CONV_B)))


def test_every_restore_starts_from_the_system_prefix(adapter):
    live = _cold(adapter, SYSTEM)
    snapshot = SimpleEngine._snapshot_prompt_cache([live])
    adapter.feed(live, CONV_A, len(SYSTEM))

    first = adapter.new()
    SimpleEngine._restore_prompt_cache([first], snapshot)
    adapter.feed(first, CONV_A, len(SYSTEM))
    second = adapter.new()
    SimpleEngine._restore_prompt_cache([second], snapshot)
    third = adapter.new()
    SimpleEngine._restore_prompt_cache([third], snapshot)
    adapter.feed(third, CONV_B, len(SYSTEM))

    assert _same(adapter.observed(second), adapter.observed(_cold(adapter, SYSTEM)))
    assert _same(adapter.observed(first), adapter.observed(_cold(adapter, CONV_A)))
    assert _same(adapter.observed(third), adapter.observed(_cold(adapter, CONV_B)))


def test_rewinding_a_live_or_restored_cache_does_not_change_the_snapshot():
    """Speculative rollback / trie trims move a cache back into the system prefix
    and rewrite it; the saved snapshot is not theirs to change."""
    adapter = _KVAdapter()
    live = _cold(adapter, SYSTEM)
    snapshot = SimpleEngine._snapshot_prompt_cache([live])

    adapter.rewind(live, 2)
    adapter.feed(live, SYSTEM[:2] + [77, 78, 79, 80, 81, 82], 2)  # other tokens
    restored = adapter.new()
    SimpleEngine._restore_prompt_cache([restored], snapshot)
    assert _same(adapter.observed(restored), adapter.observed(_cold(adapter, SYSTEM)))

    adapter.rewind(restored, 3)
    adapter.feed(restored, SYSTEM[:3] + [90, 91, 92, 93, 94], 3)
    again = adapter.new()
    SimpleEngine._restore_prompt_cache([again], snapshot)
    assert _same(adapter.observed(again), adapter.observed(_cold(adapter, SYSTEM)))


# --------------------------------------------------------------------------- #
# Engine level: system snapshot + prefix trie, chain-hash model
# --------------------------------------------------------------------------- #

VOCAB = 128


def _next_token(prev: int) -> int:
    return 32 + prev % 95


def _reference_generation(prompt_tokens, max_tokens):
    """What a cold prefill of the whole prompt generates with the chain model."""
    prev = 0
    for token in prompt_tokens:
        prev = _chain(prev, token)
    out = []
    for _ in range(max_tokens):
        token = _next_token(prev)
        out.append(token)
        prev = _chain(prev, token)
    return "".join(chr(t) for t in out)


class _ChainModel:
    """Each position's KV is the chain hash of the prefix, read back from the cache."""

    def __call__(self, tokens, cache=None):
        layer = cache[0]
        prev = 0
        if layer.offset:
            prev = int(layer.keys[0, 0, layer.offset - 1, 0].item())
        values = []
        for token in tokens[0].tolist():
            prev = _chain(prev, int(token))
            values.append(prev)
        kv = mx.array(values, dtype=mx.float32).reshape(1, 1, len(values), 1)
        layer.update_and_fetch(kv, kv)
        logits = np.zeros((1, len(values), VOCAB), dtype=np.float32)
        logits[0, -1, _next_token(prev)] = 1.0
        return mx.array(logits)


def _fake_stream_generate(model, _tokenizer, prompt, max_tokens, prompt_cache, **_):
    """mlx-lm semantics: the cache has seen every yielded token (offset == key)."""
    logits = model(prompt[None], cache=prompt_cache)
    for i in range(max_tokens):
        token = int(mx.argmax(logits[0, -1]).item())
        logits = model(mx.array([[token]]), cache=prompt_cache)
        yield SimpleNamespace(
            text=chr(token),
            token=token,
            finish_reason="length" if i == max_tokens - 1 else None,
        )


class _CharTokenizer:
    bos_token = None
    eos_token_ids = []

    def apply_chat_template(self, messages, **_kwargs):
        rendered = ""
        for message in messages:
            rendered += f"<|{message['role']}|>{message.get('content', '')}\n"
        return rendered + "<|assistant|>"

    def encode(self, text, add_special_tokens=True):
        return ([1] if add_special_tokens else []) + [ord(ch) for ch in text]


class _WordTokenizer(_CharTokenizer):
    """Whitespace-delimited words are single tokens: text cut mid-word re-tokenizes."""

    def encode(self, text, add_special_tokens=True):
        import re

        return [
            sum(ord(c) * 131**i for i, c in enumerate(piece)) % 100003
            for piece in re.findall(r"\s*\S+|\s+", text)
        ]


SYSTEM_PROMPT = "You are a careful assistant for a travel agency."


def _chat(system, user, *rest):
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        *rest,
    ]


def _engine(tokenizer, **kwargs):
    engine = SimpleEngine("test-model", **kwargs)
    engine._loaded = True
    engine._supports_system_kv_cache = True
    engine._model = SimpleNamespace(model=_ChainModel(), tokenizer=tokenizer)
    return engine


async def _answer(engine, tokenizer, messages, max_tokens=6):
    chunks = [
        chunk
        async for chunk in engine.stream_chat(
            messages, max_tokens=max_tokens, temperature=0.0, top_p=1.0
        )
    ]
    prompt = tokenizer.apply_chat_template(messages)
    expected = _reference_generation(tokenizer.encode(prompt, False), max_tokens)
    return "".join(c.new_text for c in chunks), expected, chunks[-1]


def _patched():
    return (
        patch(
            "mlx_lm.models.cache.make_prompt_cache",
            side_effect=lambda *_a, **_k: [KVCache()],
        ),
        patch("mlx_lm.stream_generate", side_effect=_fake_stream_generate),
    )


@pytest.mark.anyio
@pytest.mark.parametrize("trie", [False, True])
async def test_requests_sharing_a_system_prompt_answer_like_a_cold_prefill(trie):
    """A, B (system hit), A again, B again, then A's conversation continued: every
    answer is what a cold prefill of that request produces."""
    tokenizer = _CharTokenizer()
    engine = _engine(tokenizer, prefix_trie_cache=trie)
    conv_a = _chat(SYSTEM_PROMPT, "Name three primary colours.")
    conv_b = _chat(SYSTEM_PROMPT, "How many days are in a leap year?")
    first_answer = _reference_generation(
        tokenizer.encode(tokenizer.apply_chat_template(conv_a), False), 6
    )
    conv_a2 = _chat(
        SYSTEM_PROMPT,
        "Name three primary colours.",
        {"role": "assistant", "content": first_answer},
        {"role": "user", "content": "And three secondary ones?"},
    )

    patch_cache, patch_generate = _patched()
    with patch_cache, patch_generate:
        for messages in (conv_a, conv_b, conv_a, conv_b, conv_a2, conv_a, conv_b):
            got, expected, last = await _answer(engine, tokenizer, messages)
            assert got == expected

    assert engine._system_kv_cache_stats["hits"] >= 1
    if trie:
        assert engine._prefix_trie_cache_stats["hits"] >= 1


@pytest.mark.anyio
async def test_system_prompt_whose_tokens_are_not_a_prefix_of_the_request_is_not_reused():
    """The probe cuts the rendered prompt mid-word; the cut text tokenizes to
    different ids than the same characters inside the full prompt. The snapshot
    would hold KV of tokens the request does not contain, so it must be skipped."""
    tokenizer = _WordTokenizer()
    engine = _engine(tokenizer)
    uncached_prompts = []

    async def uncached_stream_generate(*, prompt, **_kwargs):
        uncached_prompts.append(prompt)
        yield GenerationOutput(text="", new_text="", finished=True)

    engine.stream_generate = uncached_stream_generate
    patch_cache, patch_generate = _patched()
    with patch_cache, patch_generate:
        for user in ("first question", "second question", "first question"):
            messages = _chat(SYSTEM_PROMPT, user)
            _ = [
                c
                async for c in engine.stream_chat(
                    messages, max_tokens=6, temperature=0.0, top_p=1.0
                )
            ]

    assert len(uncached_prompts) == 3
    assert engine._system_kv_cache_stats == {
        "hits": 0,
        "misses": 0,
        "stores": 0,
        "evictions": 0,
    }
    assert len(engine._system_kv_cache) == 0


@pytest.mark.anyio
async def test_different_system_prompts_do_not_share_snapshots():
    """Same user turn behind two system prompts of equal length."""
    tokenizer = _CharTokenizer()
    engine = _engine(tokenizer)
    system_a = "You are a careful assistant for a travel agency."
    system_b = "You are a rushing assistant for a travel agency."
    assert len(system_a) == len(system_b)
    patch_cache, patch_generate = _patched()
    with patch_cache, patch_generate:
        for system in (system_a, system_b, system_a, system_b):
            got, expected, _ = await _answer(
                engine, tokenizer, _chat(system, "Name a colour.")
            )
            assert got == expected

    assert engine._system_kv_cache_stats["stores"] == 2
