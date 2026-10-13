# SPDX-License-Identifier: Apache-2.0
"""
Harmony tool call parser for GPT-OSS models.

Harmony uses control tokens and channels for tool calling:

    <|channel|>commentary to=functions.get_weather
    <|constrain|>json
    <|message|>{"location": "San Francisco"}
    <|call|>

The final response is in the 'final' channel:

    <|channel|>final
    <|message|>The weather is 72F.
    <|return|>
"""

import json
import re
import uuid
from collections.abc import Sequence
from typing import Any

from ..reasoning.harmony_channels import (
    CHANNEL,
    Body,
    Close,
    HarmonyScanner,
    Open,
    Plain,
    parse_messages,
)
from .abstract_tool_parser import (
    ExtractedToolCallInformation,
    ToolParser,
    ToolParserManager,
)


def _generate_tool_id() -> str:
    """Generate a unique tool call ID."""
    return f"call_{uuid.uuid4().hex[:8]}"


_FUNCTION_RECIPIENT = re.compile(r"functions\.(\w+)")


def _function_name(channel: str | None, recipient: str | None) -> str | None:
    """The function a commentary message is addressed to, if it is a call."""
    if channel != "commentary" or not recipient:
        return None
    match = _FUNCTION_RECIPIENT.fullmatch(recipient)
    return match.group(1) if match else None


def _build_call(name: str, body: str, terminator: str | None) -> dict[str, Any] | None:
    """A tool call from a commentary message, or None when it is not one.

    A message that stopped at an explicit terminator (``<|call|>``, ``<|end|>``,
    ``<|return|>``, ``<|start|>``) with arguments that are not JSON keeps the raw
    arguments. One that ran to the end of the text or into another channel is
    judged strictly: arguments that do not parse mean it was cut off, not a call.
    """
    args_str = body.strip()
    try:
        arguments = json.loads(args_str)
    except json.JSONDecodeError:
        if terminator is None or terminator == CHANNEL:
            return None
        return {"id": _generate_tool_id(), "name": name, "arguments": args_str}
    return {
        "id": _generate_tool_id(),
        "name": name,
        "arguments": (
            json.dumps(arguments, ensure_ascii=False)
            if isinstance(arguments, dict)
            else str(arguments)
        ),
    }


@ToolParserManager.register_module(["harmony", "gpt-oss"])
class HarmonyToolParser(ToolParser):
    """
    Tool call parser for GPT-OSS models using Harmony format.

    Harmony uses control tokens and 3 channels:
    - analysis: internal reasoning (handled by reasoning parser)
    - commentary: tool calls addressed with to=functions.{name}
    - final: user-facing response

    Used when --enable-auto-tool-choice --tool-call-parser harmony are set.
    """

    SUPPORTS_NATIVE_TOOL_FORMAT = False
    # The commentary channel is where the calls are, and a reasoning parser keeps it
    # out of its content. This parser reads the channels itself (analysis is never
    # content or a call), so with a reasoning parser it is handed the raw output, control
    # tokens intact, instead of the reasoning parser's content.
    READS_RAW_OUTPUT = True
    # Channels are its own business from the first delta: it is what keeps the analysis
    # and the call text out of the content when there is no reasoning parser.
    REQUIRES_EAGER_STREAMING = True

    def __init__(self, tokenizer=None):
        super().__init__(tokenizer)
        self._reset_stream()

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        """
        Extract tool calls from a complete Harmony model response.

        Parses commentary channel messages addressed to ``functions.*`` for tool
        calls and the final channel for the user-facing content. The closing
        token of a message is usually missing from the text (``<|call|>`` and
        ``<|return|>`` are stop tokens), so a message that runs to the end of
        the text is read like a terminated one.
        """
        messages, plain = parse_messages(model_output)

        tool_calls = []
        for message in messages:
            name = _function_name(message.channel, message.recipient)
            if name is None:
                continue
            call = _build_call(name, message.body, message.terminator)
            if call is not None:
                tool_calls.append(call)

        final = "\n".join(
            m.body.strip() for m in messages if m.channel == "final" and m.body.strip()
        )
        # Text outside any message is an answer only when the model used no
        # channels at all; an analysis or a cut-off call is not content.
        content = final or (None if messages else plain.strip()) or None

        if tool_calls:
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=content,
            )

        return ExtractedToolCallInformation(
            tools_called=False,
            tool_calls=[],
            content=content or "",
        )

    # --- streaming -----------------------------------------------------------------

    def _reset_stream(self) -> None:
        self._scanner = HarmonyScanner()
        self._fed = 0  # length of the text the scanner has been given
        self._call: tuple[str, list[str]] | None = None  # open call: name, body parts
        self._call_index = 0
        self._final_started = False  # inside a final message, nothing emitted yet
        self._final_emitted = False

    def _consume(
        self, events: list, content: list[str], calls: list[dict[str, Any]]
    ) -> None:
        for event in events:
            if isinstance(event, Open):
                name = _function_name(event.channel, event.recipient)
                self._call = (name, []) if name else None
                self._final_started = event.channel == "final"
            elif isinstance(event, Body):
                if self._call is not None:
                    self._call[1].append(event.text)
                elif event.channel == "final":
                    text = event.text
                    if self._final_started:
                        text = text.lstrip()
                        if not text:
                            continue
                        self._final_started = False
                        if self._final_emitted:
                            text = "\n" + text
                    self._final_emitted = True
                    content.append(text)
            elif isinstance(event, Close):
                if self._call is not None:
                    name, parts = self._call
                    self._call = None
                    call = _build_call(name, "".join(parts), event.terminator)
                    if call is not None:
                        calls.append(self._number(call))
            elif isinstance(event, Plain) and event.leading:
                content.append(event.text)

    def _number(self, call: dict[str, Any]) -> dict[str, Any]:
        numbered = {
            "index": self._call_index,
            "id": call["id"],
            "type": "function",
            "function": {"name": call["name"], "arguments": call["arguments"]},
        }
        self._call_index += 1
        return numbered

    def _result(
        self, content: list[str], calls: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        result: dict[str, Any] = {}
        if content and "".join(content):
            result["content"] = "".join(content)
        if calls:
            result["tool_calls"] = calls
        return result or None

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
        Extract tool calls from streaming Harmony model output.

        The deltas are read as they come, whatever their size: a delta may hold
        several tokens, a control token with text after it, or the end of a message
        and the start of the next. A commentary message addressed to a function
        becomes a tool call when it ends (an explicit terminator, or the start of the
        next message); the final channel is emitted as content and everything else
        (analysis, preambles) is suppressed. A call still open when the stream ends is
        resolved by ``finalize_streaming``.
        """
        # The text before this delta, taken from the accumulated text: that is the
        # one callers always keep right.
        before = (
            current_text[: len(current_text) - len(delta_text)]
            if current_text.endswith(delta_text)
            else previous_text
        )
        if self._fed != len(before):
            # The text before this delta was not seen through this parser (it was only
            # accumulated, or the parser was reset): read it, emit nothing for it.
            self._reset_stream()
            self._consume(self._scanner.feed(before), [], [])
            self._call_index = 0
        self._fed = len(before) + len(delta_text)

        content: list[str] = []
        calls: list[dict[str, Any]] = []
        self._consume(self._scanner.feed(delta_text), content, calls)
        return self._result(content, calls)

    def finalize_streaming(self, current_text: str) -> dict[str, Any] | None:
        """End of the stream: a call that never got a terminator is read as is."""
        if self._fed != len(current_text):
            self._reset_stream()
            self._consume(self._scanner.feed(current_text), [], [])
            self._call_index = 0
            self._fed = len(current_text)
        content: list[str] = []
        calls: list[dict[str, Any]] = []
        self._consume(self._scanner.finish(), content, calls)
        if self._call is not None:
            name, parts = self._call
            self._call = None
            call = _build_call(name, "".join(parts), None)
            if call is not None:
                calls.append(self._number(call))
        return self._result(content, calls)

    def reset(self) -> None:
        """Reset parser state for a new request."""
        super().reset()
        self._reset_stream()
