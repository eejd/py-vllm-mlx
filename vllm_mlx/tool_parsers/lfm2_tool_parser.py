# SPDX-License-Identifier: Apache-2.0
"""
Liquid LFM2 / LFM2.5 tool call parser for vllm-mlx.

The LFM2 chat template renders (and the model emits) tool calls as a Python
list of keyword-only calls between two marker tokens::

    <|tool_call_start|>[get_weather(city='Paris'), get_weather(city='Tokyo')]<|tool_call_end|>

Parallel calls share one block. String values are single-quoted with Python
escapes; nested dicts/lists are rendered as JSON (so ``true``/``false``/``null``
can appear inside them) and other scalars through ``str()`` (``True``/``None``).
The markers are plain (non-special) tokens, so they survive decoding.
"""

import ast
import copy
import json
from collections.abc import Sequence
from typing import Any

from .abstract_tool_parser import (
    ExtractedToolCallInformation,
    ToolParser,
    ToolParserManager,
)
from .hermes_tool_parser import generate_tool_id

TOOL_CALL_START = "<|tool_call_start|>"
TOOL_CALL_END = "<|tool_call_end|>"

_JSON_LITERALS = {"true": True, "false": False, "null": None}


class _JSONLiterals(ast.NodeTransformer):
    """Turn the JSON spellings ``true``/``false``/``null`` into constants."""

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id in _JSON_LITERALS:
            return ast.copy_location(ast.Constant(_JSON_LITERALS[node.id]), node)
        return node


def _call_name(func: ast.expr) -> str | None:
    """``name`` or dotted ``a.b.c`` for the callee, else None."""
    parts: list[str] = []
    while isinstance(func, ast.Attribute):
        parts.append(func.attr)
        func = func.value
    if not isinstance(func, ast.Name):
        return None
    parts.append(func.id)
    return ".".join(reversed(parts))


def _literal(node: ast.expr) -> Any:
    """Evaluate a keyword value to a JSON-serialisable Python value."""
    tree = ast.fix_missing_locations(_JSONLiterals().visit(copy.deepcopy(node)))
    value = ast.literal_eval(tree)
    if isinstance(value, set):
        value = sorted(value, key=str)
    json.dumps(value)  # raises for complex/bytes/etc.
    return value


def _parse_payload(payload: str) -> list[tuple[str, dict[str, Any]]] | None:
    """Parse the text between the markers into ``[(name, arguments), ...]``.

    Accepts the canonical ``[call, call]`` list and a bare single ``call``.
    Returns None when anything in the payload is not a keyword-only call with
    literal values (positional arguments, ``**kwargs``, names, expressions), so
    a half-understood block never produces a half-correct call.
    """
    src = payload.strip()
    if not src:
        return None
    try:
        body = ast.parse(src, mode="eval").body
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return None
    nodes = body.elts if isinstance(body, ast.List) else [body]
    if not nodes:
        return None
    calls: list[tuple[str, dict[str, Any]]] = []
    for node in nodes:
        if not isinstance(node, ast.Call) or node.args:
            return None
        name = _call_name(node.func)
        if not name:
            return None
        args: dict[str, Any] = {}
        for kw in node.keywords:
            if kw.arg is None:
                return None
            try:
                args[kw.arg] = _literal(kw.value)
            except (ValueError, TypeError, SyntaxError, RecursionError, MemoryError):
                return None
        calls.append((name, args))
    return calls


def _split(text: str) -> list[tuple[str, int, int, str, bool]]:
    """Split text into ``(kind, start, end, body, closed)`` segments.

    ``kind`` is ``"text"`` or ``"block"``. ``start``/``end`` are offsets into ``text``;
    a block spans its start marker through its end marker, and ``body`` is its payload.
    ``closed`` is False only for a trailing block that has not seen its end marker yet.
    """
    segments: list[tuple[str, int, int, str, bool]] = []
    pos = 0
    while True:
        start = text.find(TOOL_CALL_START, pos)
        if start == -1:
            if pos < len(text):
                segments.append(("text", pos, len(text), text[pos:], True))
            return segments
        if start > pos:
            segments.append(("text", pos, start, text[pos:start], True))
        body = start + len(TOOL_CALL_START)
        end = text.find(TOOL_CALL_END, body)
        if end == -1:
            segments.append(("block", start, len(text), text[body:], False))
            return segments
        block_end = end + len(TOOL_CALL_END)
        segments.append(("block", start, block_end, text[body:end], True))
        pos = block_end


def _make_call(name: str, args: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": generate_tool_id(),
        "name": name,
        "arguments": json.dumps(args, ensure_ascii=False),
    }


def _partial_marker_suffix(text: str) -> int:
    """Length of a trailing proper prefix of the start marker (``"<|tool_c"``), else 0."""
    for n in range(min(len(TOOL_CALL_START) - 1, len(text)), 0, -1):
        if text.endswith(TOOL_CALL_START[:n]):
            return n
    return 0


def _ends_inside_block(text: str) -> bool:
    start = text.rfind(TOOL_CALL_START)
    return start != -1 and text.find(TOOL_CALL_END, start + len(TOOL_CALL_START)) == -1


@ToolParserManager.register_module(["lfm2", "lfm2.5"])
class Lfm2ToolParser(ToolParser):
    """Tool call parser for Liquid LFM2 / LFM2.5 models.

    Used with ``--enable-auto-tool-choice --tool-call-parser lfm2``. The model
    opens ``<think>`` in the generation prompt, so pair it with
    ``--reasoning-parser qwen3`` (think text is also stripped here as a fallback).
    """

    # The LFM2 template renders assistant ``tool_calls`` and ``tool`` turns itself.
    SUPPORTS_NATIVE_TOOL_FORMAT = True
    # Lets the server route deltas here once the (non-special) start marker shows up.
    STREAMING_MARKERS = (TOOL_CALL_START,)

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        text = self.strip_think_tags(model_output)
        tool_calls: list[dict[str, Any]] = []
        content: list[str] = []
        for kind, _start, _end, value, _closed in _split(text):
            if kind == "text":
                content.append(value)
                continue
            parsed = _parse_payload(value)
            if parsed is None:
                # Not a call we understand: keep it visible rather than drop it.
                content.append(value.strip())
                continue
            tool_calls.extend(_make_call(n, a) for n, a in parsed)
        cleaned = "".join(content).strip()
        return ExtractedToolCallInformation(
            tools_called=bool(tool_calls),
            tool_calls=tool_calls,
            content=cleaned if cleaned or not tool_calls else None,
        )

    @staticmethod
    def _format_streaming(
        calls: list[dict[str, Any]], start_index: int
    ) -> dict[str, Any]:
        """Render calls into the streaming delta shape; ``index`` stays absolute."""
        return {
            "tool_calls": [
                {
                    "index": start_index + i,
                    "id": tc["id"],
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": tc["arguments"]},
                }
                for i, tc in enumerate(calls)
            ]
        }

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int] | None = None,
        current_token_ids: Sequence[int] | None = None,
        delta_token_ids: Sequence[int] | None = None,
        request: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Emit the calls of a block when its end marker arrives, each exactly once.

        Counting is done on accumulated text, so a marker split across deltas is
        seen once, and ``index`` is the absolute call position so a client that
        concatenates per index never sees a call twice.
        """
        # Text the stream has already handed to the client. A trailing piece that could be
        # the start of the marker is held back until the next delta shows what it is, so a
        # marker split across deltas never leaks as prose.
        seen = len(previous_text)
        if not _ends_inside_block(previous_text):
            seen -= _partial_marker_suffix(previous_text)
        segments = _split(current_text)
        held = (
            _partial_marker_suffix(current_text)
            if segments and segments[-1][0] == "text"
            else 0
        )

        running = 0  # calls in closed blocks so far, i.e. the next absolute index
        first_new: int | None = None
        new_calls: list[dict[str, Any]] = []
        content: list[str] = []
        after_block = False
        for kind, start, end, body, closed in segments:
            if kind == "text":
                # Prose is user-visible. Whatever part of it is new in this delta goes out
                # now, including prose between two blocks that arrived in one delta.
                stop = len(body) - (held if end == len(current_text) else 0)
                offset = max(0, seen - start)
                part = body[offset:stop]
                # Whitespace right after a call is not content, but whitespace inside prose
                # that follows it is (a word separator arriving as its own delta).
                if part and (part.strip() or not after_block or body[:offset].strip()):
                    content.append(part)
                continue
            after_block = True
            if not closed:
                continue  # still streaming; buffered until its end marker
            parsed = _parse_payload(body)
            already_sent = end <= seen
            if parsed is None:
                if not already_sent:
                    content.append(body.strip())
                continue
            if not already_sent:
                if first_new is None:
                    first_new = running
                new_calls.extend(_make_call(n, a) for n, a in parsed)
            running += len(parsed)

        result: dict[str, Any] = {}
        if new_calls:
            result = self._format_streaming(new_calls, first_new or 0)
        text = "".join(content)
        if text:
            result["content"] = text
        return result or None

    def finalize_streaming(self, current_text: str) -> dict[str, Any] | None:
        """Resolve a block the model never closed (stopped before the end marker)."""
        segments = _split(current_text)
        if segments and segments[-1][0] == "text":
            # Generation ended on something that looked like the start of a marker.
            held = _partial_marker_suffix(current_text)
            return {"content": current_text[-held:]} if held else None
        if not segments or segments[-1][0] != "block" or segments[-1][4]:
            return None
        already = sum(
            len(_parse_payload(body) or [])
            for kind, _s, _e, body, _c in segments[:-1]
            if kind == "block"
        )
        parsed = _parse_payload(segments[-1][3])
        if parsed is None:
            return None
        return self._format_streaming([_make_call(n, a) for n, a in parsed], already)
