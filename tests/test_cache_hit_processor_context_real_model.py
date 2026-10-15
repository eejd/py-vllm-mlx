# SPDX-License-Identifier: Apache-2.0
"""Opt-in real-checkpoint check: a prefix-cache hit answers like the cold request
under ``repetition_penalty``.

The penalty looks at the last tokens of the sequence the batch generator has
seen.  A cache hit prefills only the prompt's remainder, so unless the scheduler
also tells it which tokens the cached prefix holds, the penalty sees one token
of the prompt where the cold request sees all of it and the greedy answer
changes.  The prompt asks the model to copy its last sentence, so the answer
wants exactly the tokens the penalty is meant to discourage.

Run against a local text checkpoint::

    VLLM_MLX_REAL_CACHE_MODEL_TESTS=1 \\
    VLLM_MLX_REAL_LLM_MODEL=/path/to/Llama-3.2-1B-Instruct-4bit \\
    python -m pytest tests/test_cache_hit_processor_context_real_model.py \\
        -m slow --run-slow
"""

import asyncio
import os
from pathlib import Path

import pytest

pytest.importorskip("mlx.core")

from vllm_mlx.engine.batched import BatchedEngine  # noqa: E402

pytestmark = pytest.mark.slow

MAX_TOKENS = 48
PASSAGE = " ".join(
    f"The {noun} {verb} over the {place} while the {other} watched in silence."
    for noun, verb, place, other in [
        ("heron", "flew", "frozen lake", "fox"),
        ("fox", "ran", "snowy hill", "owl"),
        ("owl", "glided", "dark forest", "deer"),
        ("deer", "leapt", "narrow stream", "hare"),
        ("hare", "darted", "open meadow", "heron"),
        ("badger", "dug", "old orchard", "wren"),
        ("wren", "sang", "stone wall", "badger"),
        ("otter", "swam", "quiet river", "heron"),
        ("stoat", "slipped", "long hedge", "wren"),
        ("crow", "circled", "bare field", "otter"),
    ]
)


def _model_path():
    if os.environ.get("VLLM_MLX_REAL_CACHE_MODEL_TESTS") != "1":
        pytest.skip("set VLLM_MLX_REAL_CACHE_MODEL_TESTS=1 for real checkpoints")
    value = os.environ.get("VLLM_MLX_REAL_LLM_MODEL")
    if not value or not Path(value).joinpath("config.json").is_file():
        pytest.fail("VLLM_MLX_REAL_LLM_MODEL must name a complete local model")
    return value


@pytest.fixture(scope="module")
def engine():
    loop = asyncio.new_event_loop()
    started = BatchedEngine(_model_path())
    loop.run_until_complete(started.start())
    yield loop, started
    loop.run_until_complete(started.stop())
    loop.close()


def _ask(loop, engine, messages, **kw):
    out = loop.run_until_complete(
        engine.chat(messages, max_tokens=MAX_TOKENS, temperature=0.0, **kw)
    )
    return out.text, out.cached_tokens


MESSAGES = [
    {
        "role": "user",
        "content": f"{PASSAGE}\n\nCopy the last sentence of the passage exactly.",
    }
]


@pytest.mark.parametrize("penalty", [1.3, 2.0])
def test_a_hit_answers_exactly_like_the_cold_request_with_a_repetition_penalty(
    engine, penalty
):
    loop, running = engine
    running.clear_runtime_caches()
    unpenalised, _ = _ask(loop, running, MESSAGES)
    running.clear_runtime_caches()
    cold, cold_cached = _ask(loop, running, MESSAGES, repetition_penalty=penalty)
    assert not cold_cached
    assert cold != unpenalised, "the penalty must change the answer"

    hits = [_ask(loop, running, MESSAGES, repetition_penalty=penalty) for _ in range(2)]

    assert all(
        cached for _, cached in hits
    ), "the repeats must be served from the cache"
    assert [text for text, _ in hits] == [cold, cold]
