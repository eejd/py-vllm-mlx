# SPDX-License-Identifier: Apache-2.0
"""
MiniCPM5 tool call parser for vllm-mlx.

The MiniCPM5 chat template tells the model to call tools with consecutive XML
elements and no wrapper token::

    <function name="get_weather"><param name="city">Paris</param></function>

A value containing ``<``, ``&`` or a newline is wrapped in ``<![CDATA[...]]>``.
``<function`` and ``<param`` are special tokens in this tokenizer but they are
not stripped from API output, so they reach the parser as text.

Values are not typed in the markup, and the template replays non-string
arguments as Python reprs (``True``, ``{'a': 1}``), so a value is typed from the
request's tool schema when there is one and otherwise parsed as JSON, then as a
Python literal, falling back to the raw string.
"""

import json
import re
from collections.abc import Sequence
from typing import Any

from .abstract_tool_parser import (
    ExtractedToolCallInformation,
    ToolParser,
    ToolParserManager,
)
from .hermes_tool_parser import _parse_param_value, generate_tool_id

# A CDATA section is matched atomically and any other text must not start a tag we are
# looking for the end of, so a value such as ``<![CDATA[x = "</function>"]]>`` cannot end
# its element early and an unterminated block fails fast instead of backtracking.
_CDATA = r"(?><!\[CDATA\[.*?\]\]>)"
_FUNCTION_RE = re.compile(
    r'<function\s+name="([^"]+)"\s*>((?:'
    + _CDATA
    + r"|(?!</function>|<!\[CDATA\[).)*+)</function>",
    re.DOTALL,
)
_OPEN_TAIL_RE = re.compile(r'<function\s+name="([^"]+)"\s*>(.*)\Z', re.DOTALL)
_PARAM_RE = re.compile(
    r'<param\s+name="([^"]+)"\s*>((?:'
    + _CDATA
    + r"|(?!</param>|<!\[CDATA\[).)*+)</param>",
    re.DOTALL,
)
_CDATA_RE = re.compile(r"\A\s*<!\[CDATA\[(.*)\]\]>\s*\Z", re.DOTALL)
_CDATA_SECTION_RE = re.compile(r"<!\[CDATA\[.*?\]\]>", re.DOTALL)
# Legacy MiniCPM tokens some checkpoints still emit around a call, and a closing tag with
# no element to close.
_STRAY_TOKENS_RE = re.compile(r"</?tool_call>|</function>")

_FUNCTION_OPEN = "<function"
_FUNCTION_CLOSE = "</function>"


def _schema_types(schema: Any) -> set[str]:
    """JSON-schema type names a parameter accepts (``type`` or ``anyOf``)."""
    if not isinstance(schema, dict):
        return set()
    types: set[str] = set()
    declared = schema.get("type")
    if isinstance(declared, str):
        types.add(declared)
    elif isinstance(declared, list):
        types.update(t for t in declared if isinstance(t, str))
    for key in ("anyOf", "oneOf"):
        for option in schema.get(key) or []:
            types |= _schema_types(option)
    return types


def _param_schemas(request: dict[str, Any] | None, name: str) -> dict[str, Any]:
    """``properties`` of the named tool in the request, or {}."""
    for tool in (request or {}).get("tools") or []:
        func = tool.get("function", tool) if isinstance(tool, dict) else {}
        if isinstance(func, dict) and func.get("name") == name:
            props = (func.get("parameters") or {}).get("properties")
            return props if isinstance(props, dict) else {}
    return {}


def _typed_value(raw: str, schema: Any) -> Any:
    cdata = _CDATA_RE.match(raw)
    value = cdata.group(1) if cdata else raw.strip()
    types = _schema_types(schema)
    nullable = "null" in types
    if not cdata and nullable and value in ("None", "null"):
        # How the template replays (and the model writes) a null argument.
        return None
    if cdata or (types and types <= {"string", "null"}):
        # CDATA is the model's explicit "this is text"; a string-only schema agrees.
        return value
    return _parse_param_value(value)


def _build_call(name: str, body: str, request: dict[str, Any] | None) -> dict[str, Any]:
    schemas = _param_schemas(request, name)
    args = {
        pname: _typed_value(raw, schemas.get(pname))
        for pname, raw in _PARAM_RE.findall(body)
    }
    return {
        "id": generate_tool_id(),
        "name": name,
        "arguments": json.dumps(args, ensure_ascii=False),
    }


@ToolParserManager.register_module(["minicpm", "minicpm5"])
class MiniCPMToolParser(ToolParser):
    """Tool call parser for MiniCPM5.

    Used with ``--enable-auto-tool-choice --tool-call-parser minicpm``. Pair it
    with ``--reasoning-parser qwen3`` (think text is also stripped here as a
    fallback). Not for MiniCPM4, whose format differs.
    """

    # The template renders assistant ``tool_calls`` and tool results natively.
    SUPPORTS_NATIVE_TOOL_FORMAT = True
    STREAMING_MARKERS = (_FUNCTION_OPEN,)

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        text = self.strip_think_tags(model_output)
        calls = [
            _build_call(m.group(1), m.group(2), request)
            for m in _FUNCTION_RE.finditer(text)
        ]
        if not calls:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=text
            )
        content = _STRAY_TOKENS_RE.sub("", _FUNCTION_RE.sub("", text)).strip()
        return ExtractedToolCallInformation(
            tools_called=True, tool_calls=calls, content=content or None
        )

    def __init__(self, tokenizer=None):
        super().__init__(tokenizer)
        self._reset_stream_state()

    def _reset_stream_state(self) -> None:
        # Elements already closed in the text seen so far, where the last one ended, and how
        # much text that covers. Streaming advances this incrementally so each delta only
        # scans the new window instead of re-matching the whole output.
        self._n_closed = 0
        self._last_end = 0
        self._seen_len = 0

    def reset(self) -> None:
        super().reset()
        self._reset_stream_state()

    def _sync(self, previous_text: str) -> None:
        """Make the cached state describe ``previous_text`` (a no-op in normal streaming)."""
        if self._seen_len == len(previous_text):
            return
        matches = (
            list(_FUNCTION_RE.finditer(previous_text))
            if _FUNCTION_CLOSE in previous_text
            else []
        )
        self._n_closed = len(matches)
        self._last_end = matches[-1].end() if matches else 0
        self._seen_len = len(previous_text)

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
        """Emit each ``<function>`` element once, when its closing tag arrives.

        The index of a call is the number of *elements* closed before it (not the number
        of ``</function>`` strings: a stray one, or one inside CDATA, closes nothing).
        Elements are tracked on accumulated text, so tags that split across deltas work and
        a call is never re-sent. The server only routes deltas here once ``<function`` has
        appeared, so think text before it has already gone out as content.
        """
        self._sync(previous_text)
        if _FUNCTION_OPEN not in current_text:
            self._seen_len = len(current_text)
            return {"content": delta_text}

        prev_closed = self._n_closed
        # A newly completed element must end inside this delta (plus the bytes a tag could
        # straddle), so only then is the matcher run.
        window = max(0, len(previous_text) - len(_FUNCTION_CLOSE) + 1)
        new = (
            list(_FUNCTION_RE.finditer(current_text, self._last_end))
            if _FUNCTION_CLOSE in current_text[window:]
            else []
        )
        if new:
            self._n_closed += len(new)
            self._last_end = new[-1].end()
        self._seen_len = len(current_text)
        element_open = current_text.find(_FUNCTION_OPEN, self._last_end) != -1

        if new:
            result: dict[str, Any] = self._format_streaming(
                [_build_call(m.group(1), m.group(2), request) for m in new],
                prev_closed,
            )
            if not element_open:
                text = _STRAY_TOKENS_RE.sub(
                    "", current_text[max(len(previous_text), self._last_end) :]
                )
                if text.strip():
                    result["content"] = text
            return result

        if element_open:
            # Inside an element. Prose that shared a delta with its opening tag is still
            # the user's.
            first_open = current_text.find(_FUNCTION_OPEN, self._last_end)
            if first_open >= len(previous_text):
                lead = current_text[len(previous_text) : first_open]
                if lead.strip():
                    return {"content": lead}
            return None
        # Between or after calls: pass prose through, drop whitespace and stray tokens.
        text = _STRAY_TOKENS_RE.sub("", delta_text)
        if (
            not text.strip()
            and not current_text[self._last_end : len(previous_text)].strip()
        ):
            return None  # whitespace right after a call is not content
        return {"content": text} if text else None

    def finalize_streaming(self, current_text: str) -> dict[str, Any] | None:
        """Resolve an element the model never closed, when it is unambiguous.

        The server calls this with the text only, so a schema cannot be applied: values
        are typed by the same fallback chain as when the request has no tools.
        """
        matches = list(_FUNCTION_RE.finditer(current_text))
        tail_from = matches[-1].end() if matches else 0
        tail = _OPEN_TAIL_RE.search(current_text, tail_from)
        if not tail:
            return None
        body = _CDATA_SECTION_RE.sub("", tail.group(2))
        if (
            _FUNCTION_OPEN in body
            or "<![CDATA[" in body
            or body.count("<param") != body.count("</param>")
        ):
            # A second element, a CDATA section the model never closed, or a param that is
            # still open: guessing would merge or truncate arguments.
            return None
        return self._format_streaming(
            [_build_call(tail.group(1), tail.group(2), None)], len(matches)
        )
