# SPDX-License-Identifier: Apache-2.0
"""No reasoning parser raises on any arrangement of its markers.

The server calls ``extract_reasoning`` (and the streaming hooks) without a guard: an
exception is an HTTP 500 for a request whose generation was merely odd. A degenerate
generation is the model's problem, not the client's, so a parser must give some answer.
(The server does not wrap the calls: masking a parser crash would hide the parser bug;
the fix belongs in the parser, and this test is what keeps it there.)
"""

import random

import pytest

from vllm_mlx.reasoning import get_parser, list_parsers

WORDS = [
    "<|channel>", "<|channel>response", "<|channel>thought", "<channel|>", "<think>",
    "</think>", "<|channel|>", "analysis", "final", "<|message|>", "<|end|>",
    "<|return|>", "<|call|>", "<|start|>", "commentary to=functions.f",
    "<|constrain|>", "response", "thought", "x", "Ok", " ", "\n", "<", "|>", "<|",
    "[THINK]", "[/THINK]",
]  # fmt: skip


@pytest.mark.parametrize("name", sorted(list_parsers()))
def test_extract_reasoning_and_the_streaming_hooks_never_raise(name):
    rng = random.Random(3)
    for _ in range(2000):
        raw = "".join(rng.choice(WORDS) for _ in range(rng.randint(0, 9)))
        get_parser(name)().extract_reasoning(raw)

        parser = get_parser(name)()
        parser.reset_state()
        cuts = (
            sorted(rng.sample(range(1, len(raw)), min(len(raw) - 1, rng.randint(0, 4))))
            if len(raw) > 1
            else []
        )
        seen = ""
        for start, end in zip([0, *cuts], [*cuts, len(raw)]):
            delta = raw[start:end]
            previous, seen = seen, seen + delta
            parser.extract_reasoning_streaming(previous, seen, delta)
        parser.finalize_stream()
