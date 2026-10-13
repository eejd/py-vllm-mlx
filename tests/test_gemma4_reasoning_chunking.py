# SPDX-License-Identifier: Apache-2.0
"""Gemma 4 streaming reasoning agrees with extract_reasoning() for every chunking.

Real Gemma 4 emits ``<|channel>`` and ``thought`` as separate tokens, so a stream of
single tokens always worked. A delta that carries several tokens (speculative / MTP
decoding, a batching detokenizer) can hold a marker together with text on either side
of it. The parser used to drop such a delta (py-vllm-mlx#72), and when the delta also
held ``<channel|>`` it never left the thinking phase, so everything after it, a tool
call included, streamed as reasoning.
"""

import itertools
import re

import pytest

from vllm_mlx.reasoning import get_parser

CALL = '<|tool_call>call:read_file{path:<|"|>/tmp/a.py<|"|>}<tool_call|>'

# raw output -> (reasoning, content) as extract_reasoning() reads it, for one
# thought/response cycle (a degenerate cycle cannot be taken back once streamed).
RAW = {
    "issue-72": (
        "<|channel>thought\nLet me read it.<channel|>" + CALL,
        "Let me read it.",
        CALL,
    ),
    "standard": (
        "<|channel>thought\nLet me think step by step.\n<channel|>The answer is 42.",
        "Let me think step by step.",
        "The answer is 42.",
    ),
    "response-marker": (
        "<|channel>thought\nAnalyzing.\n<|channel>response\nThe result is 7.",
        "Analyzing.",
        "The result is 7.",
    ),
    "empty-thought": (
        "<|channel>thought\n<channel|>Only content here.",
        None,
        "Only content here.",
    ),
    "no-markers": (
        "Just a regular answer, 1 < 2 and 3 > 2.",
        None,
        "Just a regular answer, 1 < 2 and 3 > 2.",
    ),
    "cut-off-thinking": (
        "<|channel>thought\nStill thinking...",
        "Still thinking...",
        None,
    ),
    "angle-brackets-in-thought": (
        "<|channel>thought\nif a < b and <c> then <|x|><channel|>done",
        "if a < b and <c> then <|x|>",
        "done",
    ),
    "multibyte": ("<|channel>thought\n日本語 🤔<channel|>答えは42", "日本語 🤔", "答えは42"),
    # a channel name is only a name when it stands alone or a non-letter follows it
    "word-starting-with-thought": (
        "<|channel>thought\nOk.<channel|>thoughtful day",
        "Ok.",
        "thoughtful day",
    ),
    # extract_reasoning() drops a leading name that a non-letter follows, a space included
    "name-then-space": (
        "<|channel>thought\nOk.<channel|>response times are fine",
        "Ok.",
        "times are fine",
    ),
    "word-starting-with-response": (
        "<|channel>thought\nOk.<channel|>responses vary",
        "Ok.",
        "responses vary",
    ),
    "name-then-punctuation": (
        "<|channel>thought\nOk.<channel|>thought, then more",
        "Ok.",
        ", then more",
    ),
    "name-alone-on-a-line": (
        "<|channel>thought\nA\nresponse\nB<channel|>C\nthought\nD",
        "A\nB",
        "C\nD",
    ),
    "thought-label-then-word": ("<|channel>thoughtful plan<channel|>x", "thoughtful plan", "x"),
    "name-on-a-padded-line": (
        "<|channel>thought\nA\n  thought  \nB<channel|>x",
        "A\nB",
        "x",
    ),
    "empty-thought": ("<|channel>thought\n<channel|>", None, None),
    "whitespace-at-the-edges": (
        "<|channel>thought\n\n  Think.  \n\n<channel|>\n\n  Answer.  \n\n",
        "Think.",
        "Answer.",
    ),
    "blank-lines-inside": (
        "<|channel>thought\nA\n\nB<channel|>C\n\n\nD",
        "A\n\nB",
        "C\n\n\nD",
    ),
    # the stream ends in the middle of a marker or a channel name
    "cut-after-start-marker": ("<|channel>thought\nabc<|channel>", "abc", None),
    "cut-in-response-marker": ("<|channel>thought\nabc<|channel>respo", "abcrespo", None),
    "cut-in-label": ("<|channel>thou", "thou", None),
    "cut-after-end-marker": ("<|channel>thought\nabc<channel|>", "abc", None),
    "cut-in-end-marker": ("<|channel>thought\nabc<channel|", "abc<channel|", None),
    "cut-on-a-name": ("<|channel>thought\nabc<channel|>thou", "abc", "thou"),
    "cut-after-name": ("<|channel>thought\nabc<channel|>thought", "abc", None),
}

# a marker is one token in a real tokenizer; the label and the prose are ordinary tokens
_TOKEN_RE = re.compile(r"<\|channel>|<channel\|>|<\|tool_call>|<tool_call\|>|\s+|\w+|.")


def _tokens(raw: str) -> list[str]:
    return _TOKEN_RE.findall(raw)


def _fixed(raw: str, size: int) -> list[str]:
    return [raw[i : i + size] for i in range(0, len(raw), size)]


def _grouped(raw: str, size: int) -> list[str]:
    toks = _tokens(raw)
    return ["".join(toks[i : i + size]) for i in range(0, len(toks), size)]


def _stream(raw_chunks: list[str]) -> tuple[str, str]:
    parser = get_parser("gemma4")()
    parser.reset_state()
    accumulated, reasoning, content = "", [], []
    for delta in raw_chunks:
        previous, accumulated = accumulated, accumulated + delta
        msg = parser.extract_reasoning_streaming(previous, accumulated, delta)
        if msg is not None:
            reasoning.append(msg.reasoning or "")
            content.append(msg.content or "")
    msg = parser.finalize_stream()
    if msg is not None:
        reasoning.append(msg.reasoning or "")
        content.append(msg.content or "")
    return "".join(reasoning), "".join(content)


def _chunkings(raw: str):
    yield "whole", [raw]
    yield "chars", list(raw)
    yield "tokens", _tokens(raw)
    for size in (2, 3, 5, 8, 13):
        yield f"fixed{size}", _fixed(raw, size)
    for size in (2, 3, 4):
        yield f"tokens-x{size}", _grouped(raw, size)
    # every single split point
    for cut in range(1, len(raw)):
        yield f"cut{cut}", [raw[:cut], raw[cut:]]


CASES = [
    pytest.param(name, label, chunks, id=f"{name}-{label}")
    for name, (raw, _r, _c) in RAW.items()
    for label, chunks in _chunkings(raw)
]


@pytest.mark.parametrize("name,label,chunks", CASES)
def test_streaming_matches_extract_reasoning(name, label, chunks):
    raw, want_reasoning, want_content = RAW[name]
    assert "".join(chunks) == raw
    assert _stream(chunks) == (want_reasoning or "", want_content or "")


@pytest.mark.parametrize("name", list(RAW))
def test_expected_values_are_what_extract_reasoning_returns(name):
    """The table above is the non-streaming answer, not a separate opinion."""
    raw, want_reasoning, want_content = RAW[name]
    reasoning, content = get_parser("gemma4")().extract_reasoning(raw)
    assert (reasoning, content) == (want_reasoning, want_content)


def test_start_marker_with_text_after_it_is_not_dropped():
    """#72, row 3: the delta that carries ``<|channel>`` plus the thought text."""
    got = _stream(
        ["<|channel>thought\nLet me read it.", "<channel|>", CALL],
    )
    assert got == ("Let me read it.", CALL)


def test_start_and_end_marker_in_one_delta_leave_the_thinking_phase():
    """#72, row 2: after the end marker, later text is content, not reasoning."""
    got = _stream(["<|channel>thought\nLet me read it.<channel|>", CALL])
    assert got == ("Let me read it.", CALL)


def test_whole_output_in_one_delta():
    """#72, row 1: nothing is dropped."""
    raw = RAW["issue-72"][0]
    assert _stream([raw]) == ("Let me read it.", CALL)


@pytest.mark.parametrize("cuts", list(itertools.combinations(range(1, 12), 2)))
def test_three_way_splits_of_the_markers(cuts):
    raw = "<|channel>thought\nab<channel|>cd"
    a, b = cuts
    assert _stream([raw[:a], raw[a:b], raw[b:]]) == ("ab", "cd")


def test_same_delta_can_carry_reasoning_and_content():
    parser = get_parser("gemma4")()
    parser.reset_state()
    msg = parser.extract_reasoning_streaming(
        "", "<|channel>thought\nab<channel|>cd", "<|channel>thought\nab<channel|>cd"
    )
    assert (msg.reasoning, msg.content) == ("ab", "cd")


def test_degenerate_cycling_keeps_the_last_answer_as_content():
    """Content emitted before a re-entry cannot be retracted; the final answer
    after the last <channel|> is content and no marker leaks in any chunking."""
    raw = (
        "<|channel>thought\nLoop1<channel|>"
        "<|channel>thought\nLoop2<channel|>Final response"
    )
    for label, chunks in _chunkings(raw):
        reasoning, content = _stream(chunks)
        assert content.endswith("Final response"), label
        assert "Loop2" in reasoning, label
        assert "<|channel>" not in reasoning + content, label
        assert "<channel|>" not in reasoning + content, label


@pytest.mark.parametrize(
    "tail,phase_text",
    [("1 +", "content"), ("a <", "content"), ("a <|chan", "content")],
)
def test_partial_marker_held_at_end_of_plain_content_is_content(tail, phase_text):
    """No channel was ever opened, so the withheld text is content, not reasoning."""
    parser = get_parser("gemma4")()
    parser.reset_state()
    msg = parser.extract_reasoning_streaming("", tail, tail)
    flushed = parser.finalize_stream()
    content = (msg.content if msg else "") or ""
    content += (flushed.content if flushed else "") or ""
    assert content == tail
    assert not (msg and msg.reasoning) and not (flushed and flushed.reasoning)


def test_random_single_cycle_outputs_stream_like_extract_reasoning():
    """Seeded fuzz: channel names, lookalike words, blank space and marker fragments in
    both channels, cut anywhere, delivered in random pieces."""
    import random

    words = ["thought", "response", "thoughtful", "responses", "x", "Ok", "\n", " ",
             "  ", ".", ",", "a b", "<", "<|", "|>", "\n\n", "thou", "respo"]
    rng = random.Random(7)
    checked = 0
    for _ in range(6000):
        def body():
            return "".join(rng.choice(words) for _ in range(rng.randint(0, 7)))

        sep = rng.choice(["<channel|>", "<|channel>response"])
        raw = "<|channel>" + rng.choice(["thought", "", " "]) + body()
        if rng.random() < 0.8:
            raw += sep + body()
        if rng.random() < 0.3:
            raw = raw[: rng.randint(1, len(raw))]
        starts = raw.count("<|channel>")
        if (
            raw.startswith("<|channel>response")
            or starts - (1 if "<|channel>response" in raw[1:] else 0) > 1
            or raw.count("<channel|>") > 1
            or ("<|channel>" in raw[1:] and "<|channel>response" not in raw[1:])
            or len(raw) < 2
        ):
            continue
        cuts = sorted(rng.sample(range(1, len(raw)), min(len(raw) - 1, rng.randint(0, 6))))
        chunks = [raw[i:j] for i, j in zip([0, *cuts], [*cuts, len(raw)])]
        reasoning, content = get_parser("gemma4")().extract_reasoning(raw)
        assert _stream(chunks) == (reasoning or "", content or ""), (raw, chunks)
        checked += 1
    assert checked > 2000
