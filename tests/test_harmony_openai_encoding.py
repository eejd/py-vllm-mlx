# SPDX-License-Identifier: Apache-2.0
"""The parsers read what ``openai_harmony`` writes, the way vLLM's Harmony parser reads it.

The text here is not hand-written: messages are rendered with the ``openai_harmony``
encoding (the one gpt-oss was trained on) and decoded to text, so the headers are the real
ones (``<|start|>assistant to=functions.f<|channel|>commentary json<|message|>``, a
completed message ending in ``<|end|>``). Expected values follow vLLM's mapping: a message
addressed to a function is a tool call whatever its channel, analysis is the reasoning,
final and a commentary message addressed to no one (a preamble) are the content.
"""

import json

import pytest

openai_harmony = pytest.importorskip("openai_harmony")

from openai_harmony import (  # noqa: E402
    HarmonyEncodingName,
    Message,
    Role,
    load_harmony_encoding,
)

from vllm_mlx.reasoning import get_parser  # noqa: E402
from vllm_mlx.tool_parsers.harmony_tool_parser import HarmonyToolParser  # noqa: E402

ENCODING = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
ARGS = json.dumps({"q": "x < y"})


def _message(channel, text, recipient=None, content_type=None):
    message = Message.from_role_and_content(Role.ASSISTANT, text).with_channel(channel)
    if recipient:
        message = message.with_recipient(recipient)
    if content_type:
        message = message.with_content_type(content_type)
    return message


def _render(messages, *, drop_last_stop=False):
    text = ENCODING.decode_utf8([t for m in messages for t in ENCODING.render(m)])
    # the prompt ends with "<|start|>assistant": generation continues after it. (A role
    # header that already carries the recipient is kept whole: a model writes the
    # recipient after the channel.)
    if text.startswith("<|start|>assistant<|channel|>"):
        text = text.removeprefix("<|start|>assistant")
    if drop_last_stop:
        for stop in ("<|end|>", "<|call|>", "<|return|>"):
            text = text.removesuffix(stop)  # a generation stops without decoding it
    return text


CALL = ("commentary", ARGS, "functions.search", "json")
CASES = {
    "analysis-final": ([("analysis", "Think."), ("final", "Answer.")], 0),
    "analysis-call": ([("analysis", "Need."), CALL], 1),
    "analysis-preamble-call": (
        [("analysis", "Need."), ("commentary", "Looking up."), CALL],
        1,
    ),
    "analysis-preamble-only": (
        [("analysis", "Need."), ("commentary", "Looking up.")],
        0,
    ),
    "call-in-the-analysis-channel": (
        [("analysis", ARGS, "functions.search", "json")],
        1,
    ),
    "analysis-call-final": ([("analysis", "N."), CALL, ("final", "Done.")], 1),
    "two-calls": ([CALL, ("commentary", ARGS, "functions.fetch", "json")], 2),
    "indented-final": ([("final", "    indented()\n  more   ")], 0),
}


def _expected(spec):
    reasoning, content, calls = [], [], []
    for channel, text, *rest in spec:
        recipient = rest[0] if rest else None
        if recipient and recipient.startswith("functions."):
            calls.append((recipient.removeprefix("functions."), json.loads(text)))
        elif channel == "analysis":
            reasoning.append(text)
        elif channel == "final" or (channel == "commentary" and recipient is None):
            content.append(text)
    return "\n".join(reasoning), "\n".join(content), calls


def _chunkings(text):
    tokens = [
        ENCODING.decode_utf8([t]) for t in ENCODING.encode(text, allowed_special="all")
    ]
    yield "whole", [text]
    yield "tokens", tokens
    yield "chars", list(text)
    yield "pairs", ["".join(tokens[i : i + 2]) for i in range(0, len(tokens), 2)]
    for cut in range(1, len(text), 7):
        yield f"cut{cut}", [text[:cut], text[cut:]]


def _stream_reasoning(name, chunks):
    parser = get_parser(name)()
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


def _stream_tools(chunks):
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
        "".join(content),
    )


CASE_PARAMS = [
    pytest.param(name, at_eos, id=f"{name}{'-at-eos' if at_eos else ''}")
    for name in CASES
    for at_eos in (False, True)
]


@pytest.mark.parametrize("name,at_eos", CASE_PARAMS)
@pytest.mark.parametrize("parser_name", ["gpt_oss", "harmony"])
def test_reasoning_parsers_read_the_rendered_messages(parser_name, name, at_eos):
    spec, _ = CASES[name]
    text = _render([_message(*m) for m in spec], drop_last_stop=at_eos)
    reasoning, content, _calls = _expected(spec)

    whole = get_parser(parser_name)().extract_reasoning(text)

    if parser_name == "gpt_oss" and not reasoning and not content:
        # nothing to separate (only calls): nothing, never the raw control tokens
        assert whole == (None, None)
    else:
        assert (whole[0] or "", whole[1] or "") == (reasoning.strip(), content.strip())
    for label, chunks in _chunkings(text):
        assert "".join(chunks) == text
        assert _stream_reasoning(parser_name, chunks) == (
            reasoning.strip(),
            content.strip(),
        ), label


@pytest.mark.parametrize("name,at_eos", CASE_PARAMS)
def test_tool_parser_reads_the_rendered_messages(name, at_eos):
    spec, count = CASES[name]
    text = _render([_message(*m) for m in spec], drop_last_stop=at_eos)
    _reasoning, content, calls = _expected(spec)
    assert len(calls) == count

    whole = HarmonyToolParser().extract_tool_calls(text)

    assert [(c["name"], json.loads(c["arguments"])) for c in whole.tool_calls] == calls
    assert (whole.content or "") == content.strip()
    for label, chunks in _chunkings(text):
        streamed_calls, streamed_content = _stream_tools(chunks)
        assert streamed_calls == calls, label
        assert streamed_content == content.strip(), label
        assert "<|" not in streamed_content and "to=functions" not in streamed_content
