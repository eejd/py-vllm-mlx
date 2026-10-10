# SPDX-License-Identifier: Apache-2.0
"""
Microsoft Phi-4-mini tool call parser for vllm-mlx.

Phi-4-mini is prompted (see ``vllm_mlx/templates/tool_chat_template_phi4_mini.jinja``)
to answer a tool call with the ``functools`` marker followed by one JSON list that
holds every call of the turn; there is no closing marker besides the list's own
``]``::

    functools[{"name": "get_weather", "arguments": {"city": "Paris"}}]

The registered names follow vLLM's ``phi4_mini_json`` parser. This implementation
differs from it in four ways:

* the end of the list is found with a string-aware bracket scan, not the non-greedy
  regex ``functools\\[(.*?)\\]``, which cuts the list at the first ``]`` and so loses
  any call whose arguments contain a list or a ``]`` inside a string;
* a block is accepted whole or not at all. Invalid JSON, a call that is not an
  object, a name that is not an identifier or ``arguments`` that are not an object
  leave the block in the content, so ``arguments`` is always valid JSON;
* prose around a block stays content (vLLM returns ``None`` whenever a call was found);
* it streams, with the fork's emit-once discipline.
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
from .hermes_tool_parser import generate_tool_id

MARKER = "functools"
OPEN = MARKER + "["

# OpenAI's function-name charset, minus a leading digit or hyphen: names that cannot
# be mistaken for prose, markup or a path.
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]{0,127}\Z")


def _array_end(text: str, start: int) -> int | None:
    """Index just past the ``]`` that closes the list opening at ``text[start]``.

    ``]`` and ``[`` inside a JSON string do not count. None while the list is open.
    """
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return i + 1
    return None


def _parse_calls(array_text: str) -> list[tuple[str, dict[str, Any]]] | None:
    """``[{"name": ..., "arguments": {...}}, ...]`` -> ``[(name, arguments), ...]``.

    ``parameters`` is accepted for ``arguments`` (vLLM does too) and a missing key
    means a call without arguments. Returns None when any part of the list is not a
    call this parser can stand behind.
    """
    try:
        raw = json.loads(array_text)
    except (ValueError, RecursionError):
        return None
    if not isinstance(raw, list) or not raw:
        return None
    calls: list[tuple[str, dict[str, Any]]] = []
    for item in raw:
        if not isinstance(item, dict):
            return None
        name = item.get("name")
        if not isinstance(name, str) or not _NAME.match(name):
            return None
        args = item["arguments"] if "arguments" in item else item.get("parameters", {})
        if isinstance(args, str):
            # A model that double-encodes its arguments is still unambiguous.
            try:
                args = json.loads(args)
            except (ValueError, RecursionError):
                return None
        if not isinstance(args, dict):
            return None
        calls.append((name, args))
    return calls


def _split(text: str) -> list[tuple[str, int, int, str, bool]]:
    """Split text into ``(kind, start, end, body, closed)`` segments.

    ``kind`` is ``"text"`` or ``"block"``. ``start``/``end`` are offsets into ``text``.
    A block spans ``functools`` through its closing ``]``; ``body`` is the list text.
    ``closed`` is False only for a trailing block whose list has not ended yet.
    ``functools`` not followed by ``[`` is ordinary text.
    """
    segments: list[tuple[str, int, int, str, bool]] = []
    pos = 0
    while True:
        start = text.find(OPEN, pos)
        if start == -1:
            if pos < len(text):
                segments.append(("text", pos, len(text), text[pos:], True))
            return segments
        if start > pos:
            segments.append(("text", pos, start, text[pos:start], True))
        list_start = start + len(MARKER)
        end = _array_end(text, list_start)
        if end is None:
            segments.append(("block", start, len(text), text[list_start:], False))
            return segments
        segments.append(("block", start, end, text[list_start:end], True))
        pos = end


def _make_call(name: str, args: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": generate_tool_id(),
        "name": name,
        "arguments": json.dumps(args, ensure_ascii=False),
    }


def _partial_marker_suffix(text: str) -> int:
    """Length of a trailing proper prefix of ``functools[`` (``"functo"``), else 0."""
    for n in range(min(len(OPEN) - 1, len(text)), 0, -1):
        if text.endswith(OPEN[:n]):
            return n
    return 0


def _ends_inside_block(text: str) -> bool:
    start = text.rfind(OPEN)
    return start != -1 and _array_end(text, start + len(MARKER)) is None


@ToolParserManager.register_module(["phi4_mini_json", "phi4_mini", "phi4"])
class Phi4MiniToolParser(ToolParser):
    """Tool call parser for Microsoft Phi-4-mini (``functools[...]`` JSON list).

    Used with ``--enable-auto-tool-choice --tool-call-parser phi4_mini_json``. The
    model's own chat template ignores ``tools=``; pair the parser with
    ``--chat-template vllm_mlx/templates/tool_chat_template_phi4_mini.jinja`` (shipped
    in the package) or the model is never told about any tool.
    """

    # The shipped template renders assistant ``tool_calls`` and ``tool`` turns itself.
    SUPPORTS_NATIVE_TOOL_FORMAT = True
    # The marker is plain text, so route deltas here once it shows up.
    STREAMING_MARKERS = (MARKER,)
    # Receive every delta and be finalized at the end of every response, so a list the
    # model never closed is resolved even when the last delta also completed another
    # (the server otherwise only finalizes after a delta that produced nothing).
    REQUIRES_EAGER_STREAMING = True

    def __init__(self, tokenizer=None):
        super().__init__(tokenizer)
        self._reset_stream_state()

    def _reset_stream_state(self) -> None:
        self._prose_sent = False
        # Before the marker has appeared, streaming is plain prose and each delta is
        # handled in O(1); ``_held`` is the partial-marker tail not yet handed out.
        self._marker_seen = False
        self._held = 0
        self._seen_len = 0

    def reset(self) -> None:
        super().reset()
        self._reset_stream_state()

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        tool_calls: list[dict[str, Any]] = []
        content: list[str] = []
        for kind, start, end, body, _closed in _split(model_output):
            if kind == "text":
                content.append(body)
                continue
            parsed = _parse_calls(body)
            if parsed is None:
                # Not a call list we understand: keep it visible rather than drop it.
                content.append(model_output[start:end])
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
        """Emit the calls of a list when its closing ``]`` arrives, each exactly once.

        Counting is done on accumulated text, so a marker split across deltas is
        seen once, and ``index`` is the absolute call position so a client that
        concatenates per index never sees a call twice.
        """
        if not self._marker_seen and self._seen_len == len(previous_text):
            window = current_text[max(0, len(previous_text) - len(OPEN) + 1) :]
            if OPEN not in window:
                held = _partial_marker_suffix(current_text)
                part = current_text[
                    len(previous_text) - self._held : len(current_text) - held
                ]
                self._held, self._seen_len = held, len(current_text)
                if not self._prose_sent:
                    part = part.lstrip()
                if part:
                    self._prose_sent = True
                    return {"content": part}
                return None
        self._marker_seen = True
        # Text the stream has already handed to the client. A trailing piece that could
        # be the start of the marker is held back until the next delta shows what it
        # is, so a marker split across deltas never leaks as prose.
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
        prose_sent = self._prose_sent
        for kind, start, end, body, closed in segments:
            if kind == "text":
                # Prose is user-visible. Whatever part of it is new in this delta goes
                # out now, including prose between two blocks that arrived together.
                stop = len(body) - (held if end == len(current_text) else 0)
                part = body[max(0, seen - start) : stop]
                # Leading whitespace is not content (the non-streaming result strips
                # it), but once prose has gone out whitespace is a word separator.
                if not prose_sent:
                    part = part.lstrip()
                if part:
                    content.append(part)
                    prose_sent = True
                continue
            if not closed:
                continue  # still streaming; buffered until its closing bracket
            parsed = _parse_calls(body)
            already_sent = end <= seen
            if parsed is None:
                if not already_sent:
                    content.append(current_text[start:end])
                    prose_sent = True
                continue
            if not already_sent:
                if first_new is None:
                    first_new = running
                new_calls.extend(_make_call(n, a) for n, a in parsed)
            running += len(parsed)

        result: dict[str, Any] = {}
        if new_calls:
            result = self._format_streaming(new_calls, first_new or 0)
        self._prose_sent = prose_sent
        text = "".join(content)
        if text:
            result["content"] = text
        return result or None

    def finalize_streaming(self, current_text: str) -> dict[str, Any] | None:
        """Flush what streaming held back when generation ended.

        That is a trailing prefix of the marker, or a list the model never closed
        (stopped at ``max_tokens``). The latter is not a call, so it goes out as the
        content ``extract_tool_calls`` would have returned for it.
        """
        segments = _split(current_text)
        if not segments:
            return None
        kind, start, _end, _body, closed = segments[-1]
        if kind == "text":
            held = _partial_marker_suffix(current_text)
            return {"content": current_text[-held:]} if held else None
        if closed:
            return None
        return {"content": current_text[start:]}
