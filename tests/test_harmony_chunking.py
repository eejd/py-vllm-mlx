# SPDX-License-Identifier: Apache-2.0
"""The harmony parsers read a stream the way they read the whole text, whatever the deltas.

A delta can hold several tokens (speculative decoding, a batching detokenizer): a control
token with text after it, the end of one message and the start of the next, a whole
message. The reasoning parsers used to key on single-token deltas and dropped such text.
Streaming must give what ``extract_reasoning`` / ``extract_tool_calls`` give for the whole
output, for every chunking, with the closing token present or missing (it is a stop token,
so the decoded text usually has none).
"""

import json
import re

import pytest

from vllm_mlx.reasoning import get_parser
from vllm_mlx.reasoning.harmony_channels import (
    Body,
    Close,
    HarmonyScanner,
    Open,
    parse_messages,
)
from vllm_mlx.tool_parsers.harmony_tool_parser import HarmonyToolParser

ARGS = json.dumps({"q": "a < b", "n": 2})
CALL = (
    "<|channel|>commentary to=functions.search <|constrain|>json"
    f"<|message|>{ARGS}<|call|>"
)

RAW = {
    "analysis-final": (
        "<|channel|>analysis<|message|>Thinking it over.<|end|>"
        "<|start|>assistant<|channel|>final<|message|>The answer.<|return|>"
    ),
    "analysis-final-at-eos": (
        "<|channel|>analysis<|message|>Thinking it over.<|end|>"
        "<|start|>assistant<|channel|>final<|message|>The answer."
    ),
    "analysis-call": (
        "<|channel|>analysis<|message|>Need a search.<|end|>"
        "<|start|>assistant" + CALL
    ),
    "analysis-call-at-eos": (
        "<|channel|>analysis<|message|>Need a search.<|end|>"
        "<|start|>assistant" + CALL.removesuffix("<|call|>")
    ),
    "analysis-call-final": (
        "<|channel|>analysis<|message|>Need a search.<|end|>"
        "<|start|>assistant"
        + CALL
        + "<|start|>assistant<|channel|>final<|message|>Found it.<|return|>"
    ),
    "two-calls": (
        "<|channel|>analysis<|message|>Two searches.<|end|>"
        + CALL
        + CALL.replace("search", "fetch")
    ),
    "newline-separated": (
        "<|channel|>analysis\n<|message|>Need to call.\n<|end|>\n"
        "<|channel|>commentary to=functions.search\n<|constrain|>json\n"
        f"<|message|>{ARGS}\n<|call|>\n"
        "<|channel|>final\n<|message|>Found it.\n<|return|>"
    ),
    "final-with-constrain": (
        "<|channel|>analysis<|message|>Think.<|end|>"
        "<|channel|>final <|constrain|>JSON<|message|>Done.<|return|>"
    ),
    "two-analysis-blocks": (
        "<|channel|>analysis<|message|>First.<|end|>"
        "<|start|>assistant<|channel|>analysis<|message|>Second.<|end|>"
        "<|start|>assistant<|channel|>final<|message|>Done.<|return|>"
    ),
    "preamble-then-call": (
        "<|channel|>analysis<|message|>Think.<|end|>"
        "<|start|>assistant<|channel|>commentary<|message|>Looking it up.<|end|>"
        "<|start|>assistant" + CALL
    ),
    "recipient-in-role-header": (
        "<|channel|>analysis<|message|>Think.<|end|>"
        "<|start|>assistant to=functions.search<|channel|>commentary json"
        f"<|message|>{ARGS}<|call|>"
    ),
    "cut-off-in-analysis": "<|channel|>analysis<|message|>Still think",
    "cut-off-in-call": (
        "<|channel|>analysis<|message|>Think.<|end|>"
        "<|channel|>commentary to=functions.search<|message|>" + ARGS[:-4]
    ),
    "call-mentioned-in-analysis": (
        "<|channel|>analysis<|message|>I could call to=functions.search with {} but no.<|end|>"
        "<|channel|>final<|message|>No call.<|return|>"
    ),
}

# a control token is one token in a real tokenizer, the rest is ordinary text
_TOKEN_RE = re.compile(r"<\|[a-z]+\|>|\s+|\w+|.")


def _tokens(raw):
    return _TOKEN_RE.findall(raw)


def _chunkings(raw):
    yield "whole", [raw]
    yield "chars", list(raw)
    yield "tokens", _tokens(raw)
    for size in (2, 3, 5, 7):
        yield f"fixed{size}", [raw[i : i + size] for i in range(0, len(raw), size)]
    toks = _tokens(raw)
    for size in (2, 3, 4):
        yield f"tokens-x{size}", [
            "".join(toks[i : i + size]) for i in range(0, len(toks), size)
        ]
    for cut in range(1, len(raw)):
        yield f"cut{cut}", [raw[:cut], raw[cut:]]


CASES = [
    pytest.param(name, label, chunks, id=f"{name}-{label}")
    for name, raw in RAW.items()
    for label, chunks in _chunkings(raw)
]


def _stream_reasoning(parser, chunks):
    parser.reset_state()
    seen, reasoning, content = "", [], []
    for delta in chunks:
        previous, seen = seen, seen + delta
        msg = parser.extract_reasoning_streaming(previous, seen, delta)
        if msg is not None:
            reasoning.append(msg.reasoning or "")
            content.append(msg.content or "")
    msg = parser.finalize_stream()
    if msg is not None:
        reasoning.append(msg.reasoning or "")
        content.append(msg.content or "")
    return "".join(reasoning), "".join(content)


@pytest.mark.parametrize("parser_name", ["gpt_oss", "harmony"])
@pytest.mark.parametrize("name,label,chunks", CASES)
def test_reasoning_stream_equals_whole_text(parser_name, name, label, chunks):
    raw = RAW[name]
    reasoning, content = get_parser(parser_name)().extract_reasoning(raw)
    if parser_name == "gpt_oss" and reasoning is None and content == raw:
        pytest.skip("gpt_oss returns an output with no analysis or final as is")

    streamed = _stream_reasoning(get_parser(parser_name)(), chunks)

    assert streamed == (reasoning or "", content or "")


def _stream_tool_parser(chunks):
    parser = HarmonyToolParser()
    parser.reset()
    seen, calls, content = "", [], []
    for delta in chunks:
        previous, seen = seen, seen + delta
        result = parser.extract_tool_calls_streaming(previous, seen, delta)
        if result:
            calls += result.get("tool_calls") or []
            content.append(result.get("content") or "")
    final = parser.finalize_streaming(seen)
    if final:
        calls += final.get("tool_calls") or []
        content.append(final.get("content") or "")
    return (
        [
            (c["function"]["name"], json.loads(c["function"]["arguments"]))
            for c in calls
        ],
        [c["index"] for c in calls],
        "".join(content),
    )


@pytest.mark.parametrize("name,label,chunks", CASES)
def test_tool_parser_stream_equals_whole_text(name, label, chunks):
    raw = RAW[name]
    whole = HarmonyToolParser().extract_tool_calls(raw)

    calls, indexes, content = _stream_tool_parser(chunks)

    assert calls == [(c["name"], json.loads(c["arguments"])) for c in whole.tool_calls]
    assert indexes == list(range(len(calls)))
    assert content == (whole.content or "")
    # the analysis, the call text and every control token stay out of the content
    assert not re.search(r"<\|[a-z]+\|>|to=functions", content)


# --- what the reader makes of a message ------------------------------------------------


def test_scanner_reads_header_recipient_and_constrain():
    messages, plain = parse_messages(
        "\n<|start|>assistant<|channel|>commentary to=functions.f <|constrain|>json"
        '<|message|>{"a":1}<|call|>'
    )

    assert plain == "\n"
    assert [(m.channel, m.recipient, m.body, m.terminator) for m in messages] == [
        ("commentary", "functions.f", '{"a":1}', "<|call|>")
    ]


def test_scanner_ends_a_message_where_the_next_one_starts():
    events = HarmonyScanner().feed(
        "<|channel|>analysis<|message|>a<|channel|>final<|message|>b"
    )

    assert [type(e).__name__ for e in events] == [
        "Open",
        "Body",
        "Close",
        "Open",
        "Body",
    ]
    assert isinstance(events[2], Close) and events[2].terminator == "<|channel|>"


def test_scanner_holds_a_partial_token_until_it_is_decided():
    scanner = HarmonyScanner()

    first = scanner.feed("<|channel|>analysis<|message|>x<|en")
    second = scanner.feed("d|>")

    assert [e.text for e in first if isinstance(e, Body)] == ["x"]
    assert any(isinstance(e, Close) for e in second)


def test_scanner_gives_back_a_partial_token_that_never_completed():
    scanner = HarmonyScanner()

    events = scanner.feed("<|channel|>analysis<|message|>a <|en")
    events += scanner.finish()

    assert "".join(e.text for e in events if isinstance(e, Body)) == "a <|en"
    assert isinstance(events[0], Open)


def test_scanner_reports_the_constrain_type_apart_from_the_channel():
    events = HarmonyScanner().feed(
        "<|channel|>final <|constrain|>JSON<|message|>{}<|return|>"
    )

    assert events[0] == Open("final", None, "JSON")


def test_scanner_does_not_start_a_message_inside_a_body():
    messages, _ = parse_messages("<|channel|>analysis<|message|>a<|message|>b<|end|>")

    assert [(m.channel, m.body) for m in messages] == [("analysis", "ab")]


# --- absolute expectations (the stream/whole comparisons above cannot see both being wrong)


@pytest.mark.parametrize("parser_name", ["gpt_oss", "harmony"])
def test_a_partial_token_at_the_end_of_the_text_is_text_not_lost(parser_name):
    text = "<|channel|>analysis<|message|>a < b <|"

    for _label, chunks in _chunkings(text):
        assert _stream_reasoning(get_parser(parser_name)(), chunks) == ("a < b <|", "")
    assert get_parser(parser_name)().extract_reasoning(text) == ("a < b <|", None)


def test_a_recipient_in_the_role_header_makes_a_call():
    text = (
        "<|start|>assistant to=functions.search<|channel|>commentary json"
        '<|message|>{"q": 1}<|call|>'
    )

    whole = HarmonyToolParser().extract_tool_calls(text)
    streamed = _stream_tool_parser([text])

    assert [(c["name"], json.loads(c["arguments"])) for c in whole.tool_calls] == [
        ("search", {"q": 1})
    ]
    assert streamed[0] == [("search", {"q": 1})]


@pytest.mark.parametrize("channel", ["analysis", "commentary", "final"])
def test_a_message_addressed_to_a_function_is_a_call_in_any_channel(channel):
    text = (
        f"<|channel|>{channel} to=functions.search <|constrain|>json"
        '<|message|>{"q": 1}<|call|>'
    )

    whole = HarmonyToolParser().extract_tool_calls(text)
    reasoning, content = get_parser("harmony")().extract_reasoning(text)

    assert [c["name"] for c in whole.tool_calls] == ["search"]
    assert (reasoning, content) == (None, None)  # the arguments are neither
    assert _stream_tool_parser(list(text))[0] == [("search", {"q": 1})]


def test_a_commentary_message_to_no_one_is_content_for_every_reader():
    text = (
        "<|channel|>analysis<|message|>Need.<|end|>"
        "<|channel|>commentary<|message|>Looking it up.<|end|>"
        "<|channel|>final<|message|>Done.<|end|>"
    )

    assert get_parser("harmony")().extract_reasoning(text) == (
        "Need.",
        "Looking it up.\nDone.",
    )
    assert HarmonyToolParser().extract_tool_calls(text).content == (
        "Looking it up.\nDone."
    )
    assert _stream_tool_parser(_tokens(text))[2] == "Looking it up.\nDone."


@pytest.mark.parametrize(
    "body", ["  \n Think.  \n", "    indented()\n  more   ", "\n\n x \n\n"]
)
def test_message_bodies_are_stripped_the_same_way_streaming_and_whole(body):
    text = (
        f"<|channel|>analysis<|message|>{body}<|end|><|channel|>final<|message|>{body}"
    )

    for name in ("gpt_oss", "harmony"):
        want = get_parser(name)().extract_reasoning(text)
        assert want == (body.strip(), body.strip())
        for label, chunks in _chunkings(text):
            assert _stream_reasoning(get_parser(name)(), chunks) == want, label
    for label, chunks in _chunkings(text):
        assert _stream_tool_parser(chunks)[2] == body.strip(), label


@pytest.mark.parametrize("channel", ["analysis", "commentary"])
@pytest.mark.parametrize("name", ["a-b", "get.weather", "x"])
def test_any_name_after_functions_is_a_call(channel, name):
    """vLLM checks only the prefix: a hyphen is valid in a function name."""
    text = (
        f"<|channel|>{channel} to=functions.{name} <|constrain|>json"
        '<|message|>{"q": 1}<|call|>'
    )

    whole = HarmonyToolParser().extract_tool_calls(text)

    assert [c["name"] for c in whole.tool_calls] == [name]
    assert get_parser("harmony")().extract_reasoning(text) == (None, None)
    assert get_parser("gpt_oss")().extract_reasoning(text) == (None, None)
    assert _stream_tool_parser(_tokens(text))[0] == [(name, {"q": 1})]
    for parser_name in ("gpt_oss", "harmony"):
        assert _stream_reasoning(get_parser(parser_name)(), _tokens(text)) == ("", "")


def test_only_calls_is_nothing_to_separate_for_the_reasoning_parsers():
    text = '<|channel|>commentary to=functions.f<|message|>{"a":'

    assert get_parser("gpt_oss")().extract_reasoning(text) == (None, None)
    assert _stream_reasoning(get_parser("gpt_oss")(), _tokens(text)) == ("", "")


def test_the_analysis_messages_come_out_whole_and_nothing_else_does():
    from vllm_mlx.reasoning.harmony_channels import strip_reasoning_messages

    call = '<|start|>assistant to=functions.f<|channel|>commentary json<|message|>{"a":1}<|call|>'
    text = (
        "<|channel|>analysis<|message|>one<|end|>"
        "<|start|>assistant<|channel|>analysis<|message|>two<|end|>"
        + call
        + "<|start|>assistant<|channel|>analysis<|message|>cut off"
    )

    # each analysis message leaves with its role header and terminator; the call keeps
    # its own role header
    assert strip_reasoning_messages(text) == call
    assert strip_reasoning_messages("plain <|channel|>final<|message|>x") == (
        "plain <|channel|>final<|message|>x"
    )
