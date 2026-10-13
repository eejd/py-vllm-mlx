# SPDX-License-Identifier: Apache-2.0
"""
Mistral tool call parser for vllm-mlx.

Handles Mistral's tool calling format:
- Format: [TOOL_CALLS] [{"name": "func", "arguments": {...}}]
- Or newer: [TOOL_CALLS]func_name{"arg": "value"}
- Or newest (Ministral 3, Devstral Small 2, Dec 2025 tokenizers):
  [TOOL_CALLS]func_name[ARGS]{"arg": "value"}
  Confirmed directly in these models' chat_template.jinja:
  {{- '[TOOL_CALLS]' + tool['function']['name'] + '[ARGS]' + arguments }}

Used with models like Mistral-7B-Instruct, Devstral, Ministral 3, etc.
"""

import json
import re
from collections.abc import Sequence
from random import choices
from string import ascii_letters, digits
from typing import Any

from .abstract_tool_parser import (
    ExtractedToolCallInformation,
    ToolParser,
    ToolParserManager,
)

ALPHANUMERIC = ascii_letters + digits

_TOOL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


def generate_mistral_tool_id() -> str:
    """
    Generate a random Mistral-compatible tool call ID.

    Mistral Tool Call IDs must be alphanumeric with a length of 9.
    """
    return "".join(choices(ALPHANUMERIC, k=9))


def _is_plain_tool_name(name: str) -> bool:
    """Return True for names that are safe to dispatch as function calls."""
    return bool(_TOOL_NAME_PATTERN.match(name))


_JSON_DECODER = json.JSONDecoder()


def _json_prefix_end(text: str) -> int | None:
    """Length of the first complete JSON value at the start of ``text``.

    ``text`` must already be stripped of leading whitespace. Returns None when it does
    not begin with a complete JSON value (empty, malformed, or cut off).

    A v11+ call (``name[ARGS]{...}``) has no terminator of its own, so a model may add
    ordinary text after the closing brace. That text is not part of the arguments:
    taking the whole remainder as the arguments (or rejecting the call because the
    remainder is not JSON) breaks the call. Upstream vLLM does the same since
    vllm-project/vllm#48975.
    """
    try:
        return _JSON_DECODER.raw_decode(text)[1]
    except json.JSONDecodeError:
        return None


class _ArgsScanner:
    """Follows one call's JSON arguments value through text that arrives in pieces.

    Used by the streaming path (one delta at a time) and by the marker splitter of the
    complete-output path, so both agree on where a call's arguments end and where the
    next call begins:

    - a ``[TOOL_CALLS]`` marker inside a quoted string is argument data, not a new
      call (untrusted model output must not forge a second dispatchable call);
    - the arguments end where the outermost object/array (or a top-level string)
      closes; any other top-level value (a number, ``true``) has no end to find and
      runs to the next marker;
    - once the value has ended, quotes in whatever the model wrote next are prose, not
      JSON, so they cannot hide the next call's marker.

    The ``[ARGS]`` separator ahead of the value (``name[ARGS]{...}``) is skipped
    rather than counted as an array.
    """

    __slots__ = ("in_string", "escaped", "depth", "done")

    BOT_TOKEN = "[TOOL_CALLS]"
    ARGS_TOKEN = "[ARGS]"

    def __init__(self) -> None:
        self.in_string = False
        self.escaped = False
        self.depth = 0
        self.done = False

    def scan(self, text: str, i: int = 0) -> tuple[int, int]:
        """Scan ``text`` from ``i``; return ``(args_end, marker)``.

        ``text[i:args_end]`` is arguments text (or the name/``[ARGS]`` that precedes
        it); ``marker`` is the position of the ``[TOOL_CALLS]`` marker that opens the
        next call, -1 when there is none. Text between ``args_end`` and ``marker`` (or
        the end) follows a complete arguments value.
        """
        n = len(text)
        while i < n:
            if self.done:
                return i, text.find(self.BOT_TOKEN, i)
            ch = text[i]
            if self.escaped:
                self.escaped = False
            elif self.in_string:
                if ch == "\\":
                    self.escaped = True
                elif ch == '"':
                    self.in_string = False
                    self.done = self.depth == 0
            elif ch == '"':
                self.in_string = True
            elif text.startswith(self.BOT_TOKEN, i):
                return i, i
            elif self.depth == 0 and text.startswith(self.ARGS_TOKEN, i):
                i += len(self.ARGS_TOKEN)
                continue
            elif ch in "{[":
                self.depth += 1
            elif ch in "}]":
                if self.depth > 0:
                    self.depth -= 1
                    self.done = self.depth == 0
            i += 1
        return n, -1


@ToolParserManager.register_module("mistral")
class MistralToolParser(ToolParser):
    """
    Tool call parser for Mistral models.

    Supports both old and new Mistral tool call formats:
    - Old (< v11): [TOOL_CALLS] [{"name": "add", "arguments": {"a": 1, "b": 2}}]
    - New (>= v11): [TOOL_CALLS]add{"a": 1, "b": 2}

    Used when --enable-auto-tool-choice --tool-call-parser mistral are set.
    """

    # Mistral chat templates support native tool message format
    SUPPORTS_NATIVE_TOOL_FORMAT = True

    BOT_TOKEN = "[TOOL_CALLS]"
    ARGS_TOKEN = "[ARGS]"
    TOOL_CALL_REGEX = re.compile(r"\[{.*}\]", re.DOTALL)
    _NAME_BUFFER_LIMIT = 256

    def __init__(self, tokenizer=None):
        super().__init__(tokenizer)
        self.bot_token_id = self.vocab.get(self.BOT_TOKEN) if self.vocab else None
        # Streaming state for the name/arguments boundary within the current
        # tool call. See _feed_name.
        self._args_started: bool = False
        # Where the current call's JSON arguments end, carried across deltas so
        # text the model adds after them is not streamed as arguments and a
        # [TOOL_CALLS] marker inside a JSON string stays argument data.
        self._scanner = _ArgsScanner()
        self._name_buffer: str = ""
        # Set when the boundary marker never arrives and the withheld text
        # was flushed as content; subsequent deltas pass through as content.
        self._name_buffer_overflow: bool = False
        # One id per active tool call, generated when the call starts and
        # attached to whichever delta is the first to carry real content
        # (name and/or arguments) — that may not be the delta containing
        # BOT_TOKEN itself, since the name can still be buffered pending the
        # [ARGS]/`{` boundary. See _append_call_delta.
        self._current_tool_call_id: str | None = None
        self._tool_call_id_emitted: bool = False

    def reset(self) -> None:
        super().reset()
        self._reset_call_state()
        self._current_tool_call_id = None

    def _reset_call_state(self) -> None:
        """Forget the name/arguments/id state of the current streaming call."""
        self._args_started = False
        self._scanner = _ArgsScanner()
        self._name_buffer = ""
        self._name_buffer_overflow = False
        self._tool_call_id_emitted = False

    def _start_new_tool_call(self) -> None:
        """Begin a new streaming tool call: bump the index and reset the
        per-call name/arguments and id state."""
        self.current_tool_id += 1
        self._reset_call_state()
        self._current_tool_call_id = generate_mistral_tool_id()

    def _split_on_tool_call_markers(self, text: str) -> list[str]:
        """Split on [TOOL_CALLS] occurrences that are not argument data.

        A marker appearing inside a quoted string value is argument data,
        not a new call — splitting there would let untrusted model output
        forge a second dispatchable call.

        The scan starts at the first marker, not at index 0: the text before
        the first marker is prose, not JSON, so an odd number of double quotes
        there must not leave a string open when the marker arrives (that would
        hide the call entirely). For the same reason each call is scanned with
        its own state, so quotes in text the model writes after one call's
        arguments cannot hide the next call's marker either.
        """
        token = self.BOT_TOKEN
        first = text.find(token)
        if first == -1:
            return [text]

        parts: list[str] = [text[:first]]
        start = first + len(token)
        scanner = _ArgsScanner()
        pos = start
        while True:
            _, marker = scanner.scan(text, pos)
            if marker == -1:
                break
            parts.append(text[start:marker])
            start = pos = marker + len(token)
            scanner = _ArgsScanner()
        parts.append(text[start:])
        return parts

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        """
        Extract tool calls from a complete Mistral model response.

        Args:
            model_output: The complete model output string
            request: Optional request context

        Returns:
            ExtractedToolCallInformation with parsed tool calls
        """
        # If the tool call token is not present, return as text response
        if self.BOT_TOKEN not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        content_and_raw_tool_calls = self._split_on_tool_call_markers(model_output)
        content = content_and_raw_tool_calls[0].strip()
        raw_tool_calls = content_and_raw_tool_calls[1:]

        tool_calls = []

        for raw_tool_call in raw_tool_calls:
            raw_tool_call = raw_tool_call.strip()
            if not raw_tool_call:
                continue

            # Try newest format first: func_name[ARGS]{"arg": "value"}.
            # The marker is the boundary only when it comes before the first
            # `{` — a legacy call whose JSON arguments contain the literal
            # "[ARGS]" substring must keep the `{` boundary.
            args_idx = raw_tool_call.find(self.ARGS_TOKEN)
            brace_idx = raw_tool_call.find("{")
            if (
                not raw_tool_call.startswith("[")
                and args_idx != -1
                and (brace_idx == -1 or args_idx < brace_idx)
            ):
                tool_name = raw_tool_call[:args_idx].strip()
                args_str = raw_tool_call[args_idx + len(self.ARGS_TOKEN) :].strip()

                if tool_name and _is_plain_tool_name(tool_name):
                    args_end = _json_prefix_end(args_str)
                    if args_end is None:
                        # Malformed arguments — reject rather than emit a
                        # corrupted or forged call.
                        continue
                    # Text after the arguments is not part of the call.
                    tool_calls.append(
                        {
                            "id": generate_mistral_tool_id(),
                            "name": tool_name,
                            "arguments": args_str[:args_end],
                        }
                    )
                continue

            # Try new format: func_name{"arg": "value"}
            if not raw_tool_call.startswith("[") and "{" in raw_tool_call:
                end_name = raw_tool_call.find("{")
                tool_name = raw_tool_call[:end_name].strip()
                args_str = raw_tool_call[end_name:]
                # Drop text after the arguments. A value that does not parse
                # (generation cut off) stays as it is: the server completes
                # its closing braces.
                args_end = _json_prefix_end(args_str)
                if args_end is not None:
                    args_str = args_str[:args_end]

                if tool_name:
                    tool_calls.append(
                        {
                            "id": generate_mistral_tool_id(),
                            "name": tool_name,
                            "arguments": args_str,
                        }
                    )
                continue

            # Try old format: [{"name": "func", "arguments": {...}}]
            try:
                parsed, _ = _JSON_DECODER.raw_decode(raw_tool_call)
                if isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, dict) and "name" in item:
                            args = item.get("arguments", {})
                            tool_calls.append(
                                {
                                    "id": generate_mistral_tool_id(),
                                    "name": item["name"],
                                    "arguments": (
                                        json.dumps(args, ensure_ascii=False)
                                        if isinstance(args, dict)
                                        else str(args)
                                    ),
                                }
                            )
                continue
            except json.JSONDecodeError:
                pass

            # Fallback: try regex to extract JSON array
            try:
                match = self.TOOL_CALL_REGEX.search(raw_tool_call)
                if match:
                    parsed = json.loads(match.group(0))
                    if isinstance(parsed, list):
                        for item in parsed:
                            if isinstance(item, dict) and "name" in item:
                                args = item.get("arguments", {})
                                tool_calls.append(
                                    {
                                        "id": generate_mistral_tool_id(),
                                        "name": item["name"],
                                        "arguments": (
                                            json.dumps(args, ensure_ascii=False)
                                            if isinstance(args, dict)
                                            else str(args)
                                        ),
                                    }
                                )
            except (json.JSONDecodeError, AttributeError):
                # If all parsing fails, treat as content
                if raw_tool_call:
                    content = (
                        (content + " " + raw_tool_call).strip()
                        if content
                        else raw_tool_call
                    )

        if tool_calls:
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=content if content else None,
            )
        else:
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output,
            )

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
        """
        Extract tool calls from streaming Mistral model output.

        For streaming, we detect when [TOOL_CALLS] appears and start
        accumulating tool call data.
        """
        # Text of a call that is past its name/arguments boundary is arguments
        # text (see _feed_args): a [TOOL_CALLS] marker inside a quoted JSON
        # string value is argument data, not a new call; a marker outside a
        # string starts the next call (its own index), and text after a
        # complete arguments value is dropped, as in extract_tool_calls.
        result: dict[str, Any] = {}
        if self._args_started:
            self._feed_call(delta_text, result)
            return result or None

        # Check if tool call token is in current output
        if self.BOT_TOKEN not in current_text:
            # Not a tool call yet, return content delta
            return {"content": delta_text}

        # Tool call detected
        if self.BOT_TOKEN in delta_text:
            # This delta contains the start of tool calls
            content_part, _, tool_part = delta_text.partition(self.BOT_TOKEN)
            if content_part:
                result["content"] = content_part
            self._start_new_tool_call()
            self._feed_call(tool_part, result)
            return result or None

        # We're in the middle of a tool call
        if self.current_tool_id >= 0:
            self._feed_call(delta_text, result)
            return result or None

        return None

    def _feed_call(self, text: str, result: dict[str, Any]) -> None:
        """Feed ``text`` of the current tool call into the streaming state.

        The tool-call deltas (and content, when the call turns out to be prose)
        it yields are collected in ``result``. A [TOOL_CALLS] marker inside
        ``text`` that opens the next call moves the rest of ``text`` to that
        call, however many calls the delta carries.
        """
        while text:
            if self._name_buffer_overflow:
                # The boundary never arrived; everything is plain text now.
                result["content"] = result.get("content", "") + text
                return
            if self._args_started:
                text = self._feed_args(text, result)
            else:
                text = self._feed_name(text, result)

    def _feed_name(self, text: str, result: dict[str, Any]) -> str:
        """Feed text of a call that has not reached its name/arguments boundary.

        Once the boundary (the `[ARGS]` marker, or a bare `{` for older
        checkpoints) has been seen, every later delta is argument text and is
        never re-classified — JSON string content (bare keys/values like `city`
        or `Paris`) has no distinguishing leading punctuation, so re-evaluating
        each delta in isolation (the previous approach) misclassified
        mid-argument fragments as more of the function name.

        Returns the part of the text that follows the boundary (argument text),
        or "" while the boundary has not arrived.
        """
        # Buffer until we can find the boundary marker — it may itself be
        # split across two deltas (e.g. "...[AR" / "GS]...").
        self._name_buffer += text
        buffer = self._name_buffer
        args_idx = buffer.find(self.ARGS_TOKEN)
        brace_idx = buffer.find("{")
        if args_idx != -1 and (brace_idx == -1 or args_idx < brace_idx):
            # The [ARGS] marker is the boundary when it precedes the first
            # `{`. A legacy call whose JSON arguments contain the literal
            # "[ARGS]" substring must keep the `{` boundary.
            name_end = args_idx
            args_start = args_idx + len(self.ARGS_TOKEN)
        elif brace_idx != -1:
            name_end = args_start = brace_idx
        else:
            # Marker not seen yet — withhold rather than guess. If it never
            # arrives (truncation, or the model deviating into prose), flush
            # the withheld text as content so the response is not silently
            # lost.
            if len(buffer) > self._NAME_BUFFER_LIMIT:
                self._name_buffer = ""
                self._name_buffer_overflow = True
                result["content"] = result.get("content", "") + buffer
            return ""

        self._name_buffer = ""
        self._args_started = True
        name = buffer[:name_end].strip()
        if name:
            self._append_call_delta(result, {"name": name})
        return buffer[args_start:]

    def _feed_args(self, text: str, result: dict[str, Any]) -> str:
        """Feed argument text of the current call.

        Returns the text of the next call when a [TOOL_CALLS] marker opens one
        (that call is started), else "".
        """
        args_end, marker = self._scanner.scan(text)
        if args_end:
            self._append_call_delta(result, {"arguments": text[:args_end]})
        if marker == -1:
            return ""
        self._start_new_tool_call()
        return text[marker + len(self.BOT_TOKEN) :]

    def _append_call_delta(
        self, result: dict[str, Any], function: dict[str, str]
    ) -> None:
        """Add a name/arguments delta of the current call to ``result``.

        Deltas of one call within one result are merged into a single entry.
        The call's id goes on the first entry that carries real content (the
        name may still have been buffered pending the [ARGS]/`{` boundary when
        the BOT_TOKEN delta arrived, so that need not be the BOT_TOKEN delta),
        exactly once.
        """
        calls = result.setdefault("tool_calls", [])
        if calls and calls[-1]["index"] == self.current_tool_id:
            merged = calls[-1]["function"]
            for key, value in function.items():
                merged[key] = merged.get(key, "") + value
            return
        call: dict[str, Any] = {
            "index": self.current_tool_id,
            "type": "function",
            "function": dict(function),
        }
        if not self._tool_call_id_emitted:
            call["id"] = self._current_tool_call_id
            self._tool_call_id_emitted = True
        calls.append(call)
