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
            except (ValueError, TypeError, SyntaxError, RecursionError):
                return None
        calls.append((name, args))
    return calls


def _split(text: str) -> list[tuple[str, str, bool]]:
    """Split text into ``("text", s, True)`` / ``("block", payload, closed)`` segments.

    ``closed`` is False only for a trailing block that has not seen its end
    marker yet.
    """
    segments: list[tuple[str, str, bool]] = []
    pos = 0
    while True:
        start = text.find(TOOL_CALL_START, pos)
        if start == -1:
            if pos < len(text):
                segments.append(("text", text[pos:], True))
            return segments
        if start > pos:
            segments.append(("text", text[pos:start], True))
        body = start + len(TOOL_CALL_START)
        end = text.find(TOOL_CALL_END, body)
        if end == -1:
            segments.append(("block", text[body:], False))
            return segments
        segments.append(("block", text[body:end], True))
        pos = end + len(TOOL_CALL_END)


def _make_call(name: str, args: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": generate_tool_id(),
        "name": name,
        "arguments": json.dumps(args, ensure_ascii=False),
    }


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
        for kind, value, _closed in _split(text):
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
        start_pos = current_text.find(TOOL_CALL_START)
        if start_pos == -1:
            return {"content": delta_text}

        prev_end_count = previous_text.count(TOOL_CALL_END)
        if current_text.count(TOOL_CALL_END) > prev_end_count:
            emitted_calls = 0
            new_calls: list[dict[str, Any]] = []
            new_text: list[str] = []
            first_new_index = 0
            closed_seen = 0
            for kind, value, closed in _split(current_text):
                if kind != "block" or not closed:
                    continue
                parsed = _parse_payload(value)
                if closed_seen < prev_end_count:
                    emitted_calls += len(parsed or [])
                elif parsed is None:
                    new_text.append(value.strip())
                else:
                    if not new_calls:
                        first_new_index = emitted_calls
                    new_calls.extend(_make_call(n, a) for n, a in parsed)
                    emitted_calls += len(parsed)
                closed_seen += 1
            result: dict[str, Any] = {}
            if new_calls:
                result = self._format_streaming(new_calls, first_new_index)
            tail = self._text_after_last_block(previous_text, current_text)
            content = "".join(new_text) + tail
            if content.strip():
                result["content"] = content
            return result or None

        in_block = current_text.count(TOOL_CALL_START) > current_text.count(
            TOOL_CALL_END
        )
        if in_block:
            # Text that shared a delta with the start marker is still the user's.
            if start_pos >= len(previous_text) and TOOL_CALL_START not in previous_text:
                lead = current_text[len(previous_text) : start_pos]
                if lead.strip():
                    return {"content": lead}
            return None
        # Between or after calls: pass prose through, drop pure whitespace.
        return {"content": delta_text} if delta_text.strip() else None

    @staticmethod
    def _text_after_last_block(previous_text: str, current_text: str) -> str:
        """Text after the last end marker that this delta is the first to carry."""
        last_end = current_text.rfind(TOOL_CALL_END)
        if last_end == -1:
            return ""
        after_start = last_end + len(TOOL_CALL_END)
        if current_text.count(TOOL_CALL_START) > current_text.count(TOOL_CALL_END):
            return ""
        return current_text[max(len(previous_text), after_start) :]

    def finalize_streaming(self, current_text: str) -> dict[str, Any] | None:
        """Resolve a block the model never closed (stopped before the end marker)."""
        segments = _split(current_text)
        if not segments or segments[-1][0] != "block" or segments[-1][2]:
            return None
        already = sum(
            len(_parse_payload(v) or []) for k, v, c in segments[:-1] if k == "block"
        )
        parsed = _parse_payload(segments[-1][1])
        if parsed is None:
            return None
        return self._format_streaming([_make_call(n, a) for n, a in parsed], already)
