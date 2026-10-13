# SPDX-License-Identifier: Apache-2.0
"""Every tool parser: the stream says what the whole text says, whatever the chunking.

A client must get the same calls and the same content whether it asked for a stream or
not, and whatever the deltas were: one token each, several tokens at once, whole lines,
the whole output in one. Each parser here gets its own call format, plain text, text
before and after the call, two calls and a call cut off at the end, is driven through the
real streaming handlers (chat, ``/v1/messages``, ``/v1/responses``) with a fake engine, and
compared with the non-streaming answer for the same output. Special tokens are single
tokens in a real tokenizer, so they arrive whole; everything else may be cut anywhere.

The parsers listed in ``KNOWN_VIOLATIONS`` do not hold the property today. Each is an
``xfail(strict=True)``: fixing one turns its test into a failure that says to delete the
entry, so the list can only shrink.
"""

import json
import re

import pytest

import vllm_mlx.server as srv
from tests.test_reasoning_tool_call_contract import _run
from vllm_mlx.tool_parsers import ToolParserManager

W = "hello world"
_J = json.dumps({"name": "echo_args", "arguments": {"text": W}})
_XML = (
    "<tool_call>\n<function=echo_args>\n<parameter=text>\nhello world\n</parameter>\n"
    "</function>\n</tool_call>"
)
_ARG_KV = "<tool_call>echo_args<arg_key>text</arg_key><arg_value>hello world</arg_value></tool_call>"

# parser name -> one call of the format that parser reads
CALLS = {
    "auto": "<tool_call>\n" + _J + "\n</tool_call>",
    "deepseek": (
        "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>echo_args\n"
        '```json\n{"text": "hello world"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>'
    ),
    "deepseek_v4": (
        '<｜DSML｜tool_calls>\n<｜DSML｜invoke name="echo_args">\n'
        '<｜DSML｜parameter name="text" string="true">hello world</｜DSML｜parameter>\n'
        "</｜DSML｜invoke>\n</｜DSML｜tool_calls>"
    ),
    "functionary": '<function=echo_args>{"text": "hello world"}</function>',
    "gemma4": '<|tool_call>call:echo_args{text:<|"|>hello world<|"|>}<tool_call|>',
    "glm47": _ARG_KV,
    "granite": "<|tool_call|>[" + _J + "]",
    "hermes": "<tool_call>\n" + _J + "\n</tool_call>",
    "kimi": (
        "<|tool_calls_section_begin|><|tool_call_begin|>functions.echo_args:0"
        '<|tool_call_argument_begin|>{"text": "hello world"}<|tool_call_end|>'
        "<|tool_calls_section_end|>"
    ),
    "lfm2": "<|tool_call_start|>[echo_args(text='hello world')]<|tool_call_end|>",
    "llama": '<|python_tag|>{"name": "echo_args", "parameters": {"text": "hello world"}}',
    "minicpm": '<function name="echo_args"><param name="text">hello world</param></function>',
    "minimax": (
        '<minimax:tool_call>\n<invoke name="echo_args">\n'
        '<parameter name="text">hello world</parameter>\n</invoke>\n</minimax:tool_call>'
    ),
    "mistral": '[TOOL_CALLS]echo_args[ARGS]{"text": "hello world"}',
    "nemotron": _XML,
    "phi4_mini_json": "functools[" + _J + "]",
    "poolside_v1": _ARG_KV,
    "qwen": "<tool_call>\n" + _J + "\n</tool_call>",
    "qwen3_coder": _XML,
    "step3p5": _XML,
    "xlam": "[" + _J + "]",
}


def _scenarios(call: str) -> dict[str, str]:
    return {
        "plain": "Just a plain answer, 1 < 2.",
        "call": call,
        "text-then-call": "Let me do that.\n" + call,
        "call-then-text": call + "\nAll done.",
        "two-calls": call + "\n" + call,
        "cut-off": call[: -max(3, len(call) // 5)],
    }


# scenarios that only make sense for one parser's format
_TAG = "<|python_tag|>"


def _extra_scenarios(name: str, call: str) -> dict[str, str]:
    if name != "llama":
        return {}
    # text after a call's separator: content unless it can still become another call
    return {
        "separator-text": call + "; and then some prose",
        "separator-tag-text": call + "; " + _TAG + "and then some prose",
        "separator-tag-array": call + "; " + _TAG + "[1, 2]",
        "separator-literal": call + "; true",
        "separator-partial-literal": call + "; t",
        "separator-malformed-object": call + "; {x: 1}",
        "separator-non-call-object": call + '; {"a": 1}',
        "separator-dangling": call + ";",
        "separator-then-cut-off-call": call + "; " + _TAG + call[len(_TAG) : -8],
        "separator-then-call": call + "; " + _TAG + call[len(_TAG) :],
    }


def _all_scenarios(name: str) -> dict[str, str]:
    call = CALLS[name]
    return {**_scenarios(call), **_extra_scenarios(name, call)}


# a special token is one token in a real tokenizer; the rest is cut anywhere
_TOKEN_RE = re.compile(r"<[^<>\s]*>|\[[A-Z_]+\]|\s+|\w+|.")


def _chunks(raw: str, chunking: str) -> list[str]:
    if chunking == "whole":
        return [raw]
    if chunking == "lines":
        return raw.splitlines(keepends=True)
    tokens = _TOKEN_RE.findall(raw)
    if chunking == "tokens":
        return tokens
    return ["".join(tokens[i : i + 3]) for i in range(0, len(tokens), 3)]


CHUNKINGS = ("whole", "tokens", "lines", "tokens-x3")
SITES = ("chat", "anthropic", "responses")

# What is wrong where the stream and the whole text disagree. Measured, not guessed: each
# entry below is a violation this test observed (chat, /v1/messages and /v1/responses).
REASONS = {
    "markup-in-content": (
        "the parser declares no STREAMING_MARKERS, so the streaming gate never opens: the "
        "call markup streams as content and the call is found only by the end-of-stream "
        "re-parse"
    ),
    "text-dropped": (
        "text that arrives in the same delta as a call, or after it, is dropped by the "
        "stream; the whole text returns it as content"
    ),
    "text-extra": "the stream shows text the whole-text parse drops",
    "cut-off": (
        "a call cut off at the end of generation: the whole text returns its text as "
        "content, the stream drops it (py-vllm-mlx#32)"
    ),
    "truncated-call-emitted": (
        "a call cut off at the end of generation is streamed as a call with truncated "
        "arguments (py-vllm-mlx#32)"
    ),
    "call-reemitted": (
        "with two calls the first is streamed twice (a second tool_calls entry with "
        "index 0), so a client that merges by index gets corrupt arguments"
    ),
    "stream-error": "the stream emits a call the whole text does not, with invalid arguments",
    "other": "the stream and the whole text differ in another way",
}

# (parser, scenario) -> (reasons, sites, chunkings) that violate it. Each case listed is an
# xfail(strict=True): when one is fixed its test fails until the entry is deleted, so the
# list can only shrink. Parsers not listed (hermes aside from the cases below, llama,
# lfm2/minicpm outside a cut-off call, ...) hold the property for every scenario here.
KNOWN_VIOLATIONS: dict[tuple[str, str], tuple[str, tuple[str, ...], tuple[str, ...]]] = {
    ('auto', 'call-then-text'): ('text-dropped', SITES, CHUNKINGS),
    ('auto', 'cut-off'): ('cut-off', ('anthropic', 'chat'), CHUNKINGS),
    ('auto', 'text-then-call'): ('text-dropped', SITES, ('whole',)),
    ('auto', 'two-calls'): ('call-reemitted', ('chat',), ('lines', 'tokens', 'tokens-x3')),
    ('deepseek', 'call'): ('markup-in-content', SITES, CHUNKINGS),
    ('deepseek', 'call-then-text'): ('markup-in-content', SITES, CHUNKINGS),
    ('deepseek', 'text-then-call'): ('markup-in-content', SITES, CHUNKINGS),
    ('deepseek', 'two-calls'): ('markup-in-content', SITES, CHUNKINGS),
    ('deepseek_v4', 'call-then-text'): ('stream-error', ('chat',), ('lines', 'tokens', 'tokens-x3')),
    ('deepseek_v4', 'two-calls'): ('call-reemitted', ('chat',), ('lines', 'tokens', 'tokens-x3')),
    ('functionary', 'call-then-text'): ('text-dropped', SITES, CHUNKINGS),
    ('functionary', 'cut-off'): ('cut-off', ('anthropic', 'chat'), CHUNKINGS),
    ('functionary', 'text-then-call'): ('text-dropped', SITES, ('whole',)),
    ('functionary', 'two-calls'): ('call-reemitted', ('chat',), ('lines', 'tokens', 'tokens-x3')),
    ('gemma4', 'cut-off'): ('cut-off', ('anthropic', 'chat'), CHUNKINGS),
    ('gemma4', 'text-then-call'): ('text-dropped', SITES, ('whole',)),
    ('glm47', 'cut-off'): ('cut-off', ('anthropic', 'chat'), CHUNKINGS),
    ('glm47', 'text-then-call'): ('text-extra', SITES, ('lines', 'tokens', 'tokens-x3')),
    ('glm47', 'two-calls'): ('call-reemitted', ('chat',), ('lines', 'tokens', 'tokens-x3')),
    ('granite', 'call'): ('markup-in-content', SITES, CHUNKINGS),
    ('granite', 'call-then-text'): ('markup-in-content', SITES, CHUNKINGS),
    ('granite', 'text-then-call'): ('markup-in-content', SITES, CHUNKINGS),
    ('granite', 'two-calls'): ('markup-in-content', SITES, CHUNKINGS),
    ('hermes', 'call-then-text'): ('text-dropped', SITES, ('whole',)),
    ('hermes', 'cut-off'): ('cut-off', ('anthropic', 'chat'), CHUNKINGS),
    ('hermes', 'text-then-call'): ('text-dropped', SITES, ('whole',)),
    ('kimi', 'call'): ('markup-in-content', SITES, CHUNKINGS),
    ('kimi', 'call-then-text'): ('markup-in-content', SITES, CHUNKINGS),
    ('kimi', 'text-then-call'): ('markup-in-content', SITES, CHUNKINGS),
    ('kimi', 'two-calls'): ('markup-in-content', SITES, CHUNKINGS),
    ('lfm2', 'cut-off'): ('cut-off', ('anthropic', 'chat'), CHUNKINGS),
    ('minicpm', 'cut-off'): ('cut-off', ('anthropic', 'chat'), CHUNKINGS),
    ('minimax', 'call-then-text'): ('text-dropped', SITES, CHUNKINGS),
    ('minimax', 'cut-off'): ('cut-off', ('anthropic', 'chat'), CHUNKINGS),
    ('minimax', 'text-then-call'): ('text-dropped', SITES, ('whole',)),
    ('minimax', 'two-calls'): ('other', ('chat',), ('lines', 'tokens', 'tokens-x3')),
    ('mistral', 'call-then-text'): ('stream-error+text-dropped', ('anthropic', 'chat'), CHUNKINGS),
    ('mistral', 'cut-off'): ('cut-off+truncated-call-emitted', ('anthropic', 'chat'), CHUNKINGS),
    ('mistral', 'two-calls'): ('call-reemitted', ('chat',), ('whole',)),
    ('nemotron', 'call-then-text'): ('text-dropped', SITES, CHUNKINGS),
    ('nemotron', 'cut-off'): ('cut-off', ('anthropic', 'chat'), CHUNKINGS),
    ('nemotron', 'text-then-call'): ('text-dropped', SITES, ('whole',)),
    ('nemotron', 'two-calls'): ('call-reemitted', ('chat',), ('lines', 'tokens', 'tokens-x3')),
    ('poolside_v1', 'call-then-text'): ('text-extra', ('anthropic',), CHUNKINGS),
    ('poolside_v1', 'cut-off'): ('cut-off+truncated-call-emitted', ('anthropic', 'chat'), CHUNKINGS),
    ('qwen', 'call-then-text'): ('text-dropped', SITES, CHUNKINGS),
    ('qwen', 'text-then-call'): ('text-dropped', SITES, ('whole',)),
    ('qwen3_coder', 'cut-off'): ('truncated-call-emitted', ('chat',), CHUNKINGS),
    ('qwen3_coder', 'text-then-call'): ('text-dropped', SITES, ('whole',)),
    ('step3p5', 'call-then-text'): ('text-dropped', SITES, CHUNKINGS),
    ('step3p5', 'text-then-call'): ('text-dropped', SITES, ('whole',)),
    ('xlam', 'call'): ('markup-in-content', SITES, CHUNKINGS),
    ('xlam', 'call-then-text'): ('markup-in-content', SITES, CHUNKINGS),
    ('xlam', 'text-then-call'): ('markup-in-content', SITES, CHUNKINGS),
    ('xlam', 'two-calls'): ('markup-in-content', SITES, CHUNKINGS),
}


def _configure(monkeypatch, name: str) -> None:
    import tests.test_reasoning_tool_call_contract as contract

    monkeypatch.setattr(contract, "_chunks", _chunks)
    monkeypatch.setattr(srv, "_max_request_tokens", 32768)
    monkeypatch.setattr(srv, "_enable_auto_tool_choice", True)
    monkeypatch.setattr(srv, "_tool_call_parser", name)
    monkeypatch.setattr(
        srv, "_tool_parser_instance", ToolParserManager.get_tool_parser(name)()
    )
    monkeypatch.setattr(srv, "_reasoning_parser", None)
    monkeypatch.setattr(srv, "_reasoning_parser_name", None)


PAIRS = [
    pytest.param(name, scenario, id=f"{name}-{scenario}")
    for name in sorted(CALLS)
    for scenario in _all_scenarios(name)
]


def test_every_registered_parser_class_is_covered():
    classes = {
        ToolParserManager.get_tool_parser(name)
        for name in ToolParserManager.list_registered()
    }
    covered = {ToolParserManager.get_tool_parser(name) for name in CALLS}
    # harmony is covered by tests/test_harmony_chunking.py (it needs its own control
    # tokens and a reasoning parser to be interesting)
    uncovered = {c.__name__ for c in classes - covered}
    assert uncovered == {"HarmonyToolParser"}


@pytest.mark.parametrize("chunking", CHUNKINGS)
@pytest.mark.parametrize("site", SITES)
@pytest.mark.parametrize("parser_name,scenario", PAIRS)
@pytest.mark.anyio
async def test_stream_equals_non_streaming(
    request, monkeypatch, parser_name, scenario, site, chunking
):
    known = KNOWN_VIOLATIONS.get((parser_name, scenario))
    if known and site in known[1] and chunking in known[2]:
        request.applymarker(
            pytest.mark.xfail(
                strict=True,
                reason="; ".join(REASONS[code] for code in known[0].split("+"))
                + " -- delete the KNOWN_VIOLATIONS entry if this now passes",
            )
        )
    _configure(monkeypatch, parser_name)
    raw = _all_scenarios(parser_name)[scenario]

    whole = await _run("chat", monkeypatch, raw, "whole", stream=False)
    streamed = await _run(site, monkeypatch, raw, chunking, stream=True)

    assert streamed.calls == whole.calls
    assert streamed.content.strip() == whole.content.strip()
