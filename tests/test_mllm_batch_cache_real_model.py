# SPDX-License-Identifier: Apache-2.0
"""Opt-in real-checkpoint check of the MLLM batch generator's text-only prefix cache.

What a text-only request stores in the KV prefix cache must be the KV of exactly
its prompt, and a request served from it must answer like the same request served
cold (temperature 0), through the production ``MLLMScheduler`` /
``MLLMBatchGenerator``, including after a request whose speculative (MTP) verify
step advanced the cache past the emitted tokens.

Gemma-3 mixes sliding-window and global layers, which is the case that matters::

    VLLM_MLX_REAL_CACHE_MODEL_TESTS=1 \\
    VLLM_MLX_REAL_MLLM_MODEL=/path/to/gemma-3-4b-it-qat-4bit \\
    python -m pytest tests/test_mllm_batch_cache_real_model.py --run-slow -m slow

Two ways of cutting the same prompt into forward calls give logits that differ by up
to about one unit in bf16 (a 193-token call and a 193+18-token call round the same
row differently), enough to flip a near tie at temperature 0 and unrelated to the
cache. The reference for a hit is therefore a cold request that is prefilled in the
same two calls (``prefill_step_size`` = the cached tokens), and the stored entries
are compared with the KV of a cold prefill bit for bit.

No locally runnable checkpoint ships an MTP head, so the MTP test gives the real
model a draft head of its own: it proposes the true greedy successor (taken from a
cold run of the same prompt) for two out of three steps and a wrong token for the
third, so both the accept and the reject branch of the verify step run on real KV.
"""

import os
from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core")

from mlx_lm.models.cache import make_prompt_cache  # noqa: E402

from vllm_mlx.mllm_scheduler import MLLMScheduler, MLLMSchedulerConfig  # noqa: E402

pytestmark = pytest.mark.slow

MAX_TOKENS = 20
SYSTEM = (
    "You are a meticulous assistant. Answer in plain prose and keep every "
    "answer under three sentences. Never use lists. "
) * 8


def _model_path():
    if os.environ.get("VLLM_MLX_REAL_CACHE_MODEL_TESTS") != "1":
        pytest.skip("set VLLM_MLX_REAL_CACHE_MODEL_TESTS=1 for real checkpoints")
    value = os.environ.get("VLLM_MLX_REAL_MLLM_MODEL")
    if not value or not Path(value).joinpath("config.json").is_file():
        pytest.fail("VLLM_MLX_REAL_MLLM_MODEL must name a complete local model")
    return value


@pytest.fixture(scope="module")
def loaded():
    from mlx_vlm import load

    return load(_model_path())


def _prompt(processor, *turns, system=SYSTEM):
    """Chat-template text for alternating user/assistant ``turns``."""
    messages = [{"role": "system", "content": system}]
    for index, text in enumerate(turns):
        messages.append(
            {"role": "user" if index % 2 == 0 else "assistant", "content": text}
        )
    tokenizer = getattr(processor, "tokenizer", processor)
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def _scheduler(loaded, *, prefix_cache=True, chunked=0, step=1024):
    model, processor = loaded
    return MLLMScheduler(
        model,
        processor,
        MLLMSchedulerConfig(
            max_num_seqs=4,
            prefill_batch_size=4,
            completion_batch_size=4,
            enable_vision_cache=False,
            enable_prefix_cache=prefix_cache,
            use_memory_aware_cache=prefix_cache,
            prefix_cache_memory_mb=2048,
            chunked_prefill_tokens=chunked,
            prefill_step_size=step,
        ),
    )


def _run(scheduler, prompts):
    """Run ``{id: prompt}`` to completion: ``({id: tokens}, {id: cached_tokens})``."""
    for request_id, prompt in prompts.items():
        scheduler.add_request(
            prompt=prompt,
            max_tokens=MAX_TOKENS,
            temperature=0.0,
            request_id=request_id,
        )
    tokens = {request_id: [] for request_id in prompts}
    cached = {}
    done = set()
    for _ in range(MAX_TOKENS + 800):  # interleaved prefill takes a step per chunk
        output = scheduler.step()
        for out in output.outputs:
            tokens[out.request_id].extend(out.new_token_ids)
            cached[out.request_id] = out.cached_tokens
            if out.finished:
                done.add(out.request_id)
        if len(done) == len(prompts):
            return tokens, cached
    raise AssertionError(f"requests did not finish: {tokens}")


def _cold(loaded, prompt, first_call=1024):
    """The same request with no prefix cache, prefilled in calls of ``first_call``."""
    scheduler = _scheduler(loaded, prefix_cache=False, step=first_call)
    return _run(scheduler, {"cold": prompt})[0]["cold"]


def _assert_stored_entries_are_prompt_kv(loaded, generator):
    """Every stored entry is, bit for bit, the KV of a cold prefill of its key."""
    entries = generator.prefix_cache._entries
    assert entries, "nothing was stored; the check would be vacuous"
    model, _ = loaded
    language_model = model.language_model
    for key, entry in entries.items():
        ids = list(key)
        reference = make_prompt_cache(language_model)
        language_model(mx.array([ids]), cache=reference)
        for index, (stored, cold) in enumerate(zip(entry.cache, reference)):
            assert int(stored.offset) == len(ids), (index, stored.offset, len(ids))
            for name in ("keys", "values"):
                got = getattr(stored, name)[..., : len(ids), :]
                want = getattr(cold, name)[..., : len(ids), :]
                assert mx.array_equal(got, want).item(), (index, name)


def test_repeat_and_extension_answer_like_a_cold_request(loaded):
    _, processor = loaded
    scheduler = _scheduler(loaded)
    base = _prompt(processor, "Why is the sky blue?")
    extended = _prompt(
        processor, "Why is the sky blue?", "Rayleigh scattering.", "And at sunset?"
    )

    first, _ = _run(scheduler, {"base": base})
    assert first["base"] == _cold(loaded, base)
    _assert_stored_entries_are_prompt_kv(loaded, scheduler.batch_generator)

    for name, prompt in (("repeat", base), ("extension", extended)):
        tokens, cached = _run(scheduler, {name: prompt})
        assert cached[name], f"{name}: the prefix cache was not used"
        assert tokens[name] == _cold(loaded, prompt, cached[name]), name


@pytest.mark.parametrize("chunked", [0, 64], ids=["blocking", "interleaved"])
def test_prompt_longer_than_the_sliding_window(loaded, chunked):
    """Gemma-3's window layers have wrapped; what is stored must still be exact.

    A wrapped window cannot be cut back, so the completion store is refused (and an
    interleaved prefill stores the wrapped state as it is). Whatever is stored holds
    exactly its key's positions, and a request that hits it still starts like a
    cold one.
    """
    _, processor = loaded
    tokenizer = getattr(processor, "tokenizer", processor)
    items = " ".join(
        f"Item {i}: the quick brown fox number {i} jumps over lazy dog {i * 7}."
        for i in range(110)
    )
    base = _prompt(processor, items + " Summarise the items.")
    extended = _prompt(
        processor, items + " Summarise the items.", "Done.", "Now count them."
    )
    assert len(tokenizer.encode(base)) > 1100

    scheduler = _scheduler(loaded, chunked=chunked)
    _run(scheduler, {"base": base})
    for key, entry in scheduler.batch_generator.prefix_cache._entries.items():
        assert [int(c.offset) for c in entry.cache] == [len(key)] * len(entry.cache)
    tokens, _ = _run(scheduler, {"extension": extended})
    assert tokens["extension"][:2] == _cold(loaded, extended)[:2]


def _install_oracle_mtp(scheduler, successor):
    """Give the real language model a draft head and install the MTP step.

    Returns the generator and a function that puts plain decoding back.
    """
    from vllm_mlx.mllm_batch_generator import install_mtp_mllm

    scheduler._ensure_batch_generator()
    generator = scheduler.batch_generator
    inner = generator.language_model
    plain = (inner, generator._step, generator._next)
    vocab = int(inner.lm_head.weight.shape[0])
    calls = [0]

    class WithDraftHead:
        def __getattr__(self, name):
            return getattr(inner, name)

        def __call__(self, tokens, cache=None, return_hidden=False, **kwargs):
            if not return_hidden:
                return inner(tokens, cache=cache, **kwargs)
            hidden = inner.model(tokens, cache=cache)
            return inner.lm_head(hidden), hidden

        def mtp_forward(self, hidden, next_token_ids, mtp_cache=None):
            calls[0] += 1
            proposal = successor.get(int(next_token_ids[0, 0].item()), 0)
            if calls[0] % 3 == 0:  # every third draft is wrong
                proposal += 1
            logits = mx.full((1, 1, vocab), -10.0)
            logits[0, 0, proposal] = 10.0
            return logits

    wrapper = WithDraftHead()
    generator.language_model = wrapper
    install_mtp_mllm(generator, wrapper)

    def uninstall():
        generator.language_model, generator._step, generator._next = plain

    return generator, uninstall


def test_mtp_stored_prompt_kv_serves_cold_answers(loaded):
    """The entry stored by a request that ran speculative steps is its prompt KV.

    The MTP request itself is not compared token for token with a plain one (a
    two-token verify forward and a one-token forward round differently and can flip
    a near tie); what must hold is that the entry it stored is its prompt's KV and
    that plain requests served from it answer like cold ones.
    """
    _, processor = loaded
    base = _prompt(processor, "Why is the sky blue?")
    extended = _prompt(
        processor, "Why is the sky blue?", "Rayleigh scattering.", "And at sunset?"
    )

    # The draft head proposes the true successor of a token from the cold answer.
    cold_ids = _cold(loaded, base)
    tokenizer = getattr(processor, "tokenizer", processor)
    prompt_ids = tokenizer.encode(base)
    sequence = prompt_ids + cold_ids
    successor = {}
    for current, following in zip(
        sequence[len(prompt_ids) - 1 :], sequence[len(prompt_ids) :]
    ):
        successor.setdefault(current, following)

    scheduler = _scheduler(loaded)
    generator, uninstall = _install_oracle_mtp(scheduler, successor)
    served, _ = _run(scheduler, {"mtp": base})
    stats = generator.get_mtp_stats()
    assert stats["accepted"] > 0 and stats["rejected"] > 0, stats
    assert served["mtp"][:8] == cold_ids[:8]
    _assert_stored_entries_are_prompt_kv(loaded, generator)
    uninstall()

    for name, prompt in (("repeat", base), ("extension", extended)):
        tokens, cached = _run(scheduler, {name: prompt})
        assert cached[name], f"{name}: the prefix cache was not used"
        assert tokens[name] == _cold(loaded, prompt, cached[name]), name
