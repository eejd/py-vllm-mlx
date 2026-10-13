# SPDX-License-Identifier: Apache-2.0
"""
Incremental reader for the Harmony message format (gpt-oss).

A Harmony generation is a sequence of messages::

    <|channel|>analysis<|message|>thinking<|end|>
    <|start|>assistant<|channel|>commentary to=functions.f <|constrain|>json<|message|>{...}<|call|>
    <|start|>assistant<|channel|>final<|message|>The answer.<|return|>

A message is a header (channel name, optional ``to=<recipient>``, optional
``<|constrain|>`` type) followed by ``<|message|>`` and a body that ends at
``<|end|>``, ``<|return|>`` or ``<|call|>``. The terminator is often missing from the
decoded text: those three are stop tokens, and a generation stops on them without
detokenizing them. A body also ends, unterminated, where the next message starts
(``<|start|>`` or ``<|channel|>``) or where the text ends.

``HarmonyScanner`` turns deltas of any size into events; ``parse_messages`` runs the same
reader over a complete text. The reasoning parsers and the tool parser share it, so a
stream and the whole text give the same messages whatever the delta boundaries.
"""

import re
from dataclasses import dataclass

START = "<|start|>"
CHANNEL = "<|channel|>"
MESSAGE = "<|message|>"
CONSTRAIN = "<|constrain|>"
END = "<|end|>"
RETURN = "<|return|>"
CALL = "<|call|>"

CONTROL_TOKENS = (START, CHANNEL, MESSAGE, CONSTRAIN, END, RETURN, CALL)
TERMINATORS = (END, RETURN, CALL)

_TOKEN_RE = re.compile("|".join(re.escape(t) for t in CONTROL_TOKENS))
_RECIPIENT_RE = re.compile(r"\bto=(\S+)")


@dataclass
class Open:
    """A message header is complete and its body starts."""

    channel: str | None
    recipient: str | None
    constrain: str | None
    start: int = 0  # offset of the header's first token in the text fed so far


@dataclass
class Body:
    """Text of the open message's body."""

    channel: str | None
    recipient: str | None
    text: str


@dataclass
class Close:
    """The open message ended. ``terminator`` is the token that ended it."""

    channel: str | None
    recipient: str | None
    terminator: str
    end: int = 0  # offset just past the message (past its terminator if it has one)


@dataclass
class Plain:
    """Text outside any message. ``leading``: no control token has been seen yet."""

    text: str
    leading: bool


Event = Open | Body | Close | Plain


def _trailing_partial_token_len(text: str) -> int:
    """Length of a trailing proper prefix of a control token, to withhold."""
    longest = 0
    for token in CONTROL_TOKENS:
        for i in range(min(len(token) - 1, len(text)), longest, -1):
            if text.endswith(token[:i]):
                longest = i
                break
    return longest


class HarmonyScanner:
    """Feed deltas, get events. State is the position inside the current message."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._mode = "idle"  # "idle" | "header" | "body"
        self._header_kind = ""  # "role" (after <|start|>) | "channel"
        self._held = ""  # trailing text that may still become a control token
        self._header = ""
        self._role_header = ""  # the role header the current channel header follows
        self._constrain: str | None = None
        self._in_constrain = False
        self._channel: str | None = None
        self._recipient: str | None = None
        self.seen_token = False
        self._fed = 0  # characters fed so far
        self._msg_start = 0  # where the open (or opening) message's header began

    @property
    def held(self) -> str:
        """Text withheld because it could still become a control token."""
        return self._held

    def feed(self, delta: str) -> list[Event]:
        text = self._held + delta
        base = self._fed - len(self._held)
        self._fed += len(delta)
        hold = _trailing_partial_token_len(text)
        self._held = text[len(text) - hold :] if hold else ""
        if hold:
            text = text[: len(text) - hold]
        return self._scan(text, base)

    def finish(self) -> list[Event]:
        """End of text: a withheld partial token was ordinary text after all."""
        text, self._held = self._held, ""
        return self._scan(text, self._fed - len(text))

    def _begin_header(self, kind: str, role_header: str = "") -> None:
        self._mode = "header"
        self._header_kind = kind
        self._header = ""
        self._role_header = role_header
        self._constrain = None
        self._in_constrain = False

    def _end_message(self, terminator: str, end: int, events: list[Event]) -> None:
        if self._mode == "body":
            events.append(Close(self._channel, self._recipient, terminator, end))
        self._mode = "idle"
        self._channel = self._recipient = None

    def _text(self, text: str, events: list[Event]) -> None:
        if not text:
            return
        if self._mode == "idle":
            events.append(Plain(text, not self.seen_token))
        elif self._mode == "header":
            if self._in_constrain:
                self._constrain = (self._constrain or "") + text
            else:
                self._header += text
        else:
            events.append(Body(self._channel, self._recipient, text))

    def _open_body(self, events: list[Event]) -> None:
        if self._mode == "header":
            words = self._header.split()
            self._channel = words[0] if words else None
            recipient = _RECIPIENT_RE.search(self._header) or _RECIPIENT_RE.search(
                self._role_header
            )
            self._recipient = recipient.group(1) if recipient else None
            constrain = (self._constrain or "").strip() or None
        else:
            self._channel = self._recipient = None
            constrain = None
        events.append(Open(self._channel, self._recipient, constrain, self._msg_start))
        self._mode = "body"

    def _scan(self, text: str, base: int) -> list[Event]:
        events: list[Event] = []
        pos = 0
        for match in _TOKEN_RE.finditer(text):
            self._text(text[pos : match.start()], events)
            pos = match.end()
            token = match.group()
            at, after = base + match.start(), base + match.end()
            self.seen_token = True
            if token == START:
                self._end_message(token, at, events)
                self._begin_header("role")
                self._msg_start = at
            elif token == CHANNEL:
                after_role_header = (
                    self._mode == "header" and self._header_kind == "role"
                )
                role_header = self._header if after_role_header else ""
                self._end_message(token, at, events)
                self._begin_header("channel", role_header)
                if not after_role_header:
                    self._msg_start = at
            elif token == CONSTRAIN:
                if self._mode == "header":
                    self._in_constrain = True
                    self._constrain = ""
            elif token == MESSAGE:
                if self._mode != "body":  # a stray one inside a body is not text
                    self._open_body(events)
            else:  # END, RETURN, CALL
                self._end_message(token, after, events)
        self._text(text[pos:], events)
        return events


# What a message is to the API, as vLLM's Harmony parser reads it: a message addressed to a
# function is a tool call whatever its channel, analysis is the reasoning, final and a
# commentary message addressed to no one (a preamble to the user) are the content, anything
# else is ignored.
TOOL, REASONING, CONTENT, IGNORE = "tool", "reasoning", "content", "ignore"


class BodyTrimmer:
    """``str.strip`` of one message body, streamed: leading whitespace is dropped until
    text starts, trailing whitespace is held until more text follows it and dropped when
    the message ends. ``feed`` returns the part that may be emitted."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._started = False
        self._held = ""

    def feed(self, text: str) -> str:
        if not self._started:
            text = text.lstrip()
            if not text:
                return ""
            self._started = True
        joined = self._held + text
        body = joined.rstrip()
        self._held = joined[len(body) :]
        return body


def function_name(recipient: str | None) -> str | None:
    """``name`` for a recipient ``functions.name``, else None."""
    if recipient and recipient.startswith("functions."):
        name = recipient[len("functions.") :]
        if name.isidentifier():
            return name
    return None


def segment_kind(channel: str | None, recipient: str | None) -> str:
    if function_name(recipient) is not None:
        return TOOL
    if channel == "analysis":
        return REASONING
    if channel == "final" or (channel == "commentary" and recipient is None):
        return CONTENT
    return IGNORE


@dataclass
class Message:
    """A whole message: ``terminator`` is None when the text ended inside it."""

    channel: str | None
    recipient: str | None
    body: str
    terminator: str | None
    start: int = 0
    end: int = 0


def parse_messages(text: str) -> tuple[list[Message], str]:
    """Read a complete text: its messages and the plain text outside them."""
    scanner = HarmonyScanner()
    events = scanner.feed(text) + scanner.finish()
    messages: list[Message] = []
    plain: list[str] = []
    current: Message | None = None
    for event in events:
        if isinstance(event, Open):
            current = Message(
                event.channel, event.recipient, "", None, event.start, len(text)
            )
            messages.append(current)
        elif isinstance(event, Body) and current is not None:
            current.body += event.text
        elif isinstance(event, Close) and current is not None:
            current.terminator = event.terminator
            current.end = event.end
            current = None
        elif isinstance(event, Plain):
            plain.append(event.text)
    return messages, "".join(plain)


def strip_reasoning_messages(text: str) -> str:
    """``text`` without its reasoning messages (header, body and terminator), the rest
    byte for byte: what a tool parser may read without ever seeing the analysis."""
    messages, _ = parse_messages(text)
    out: list[str] = []
    cursor = 0
    for message in messages:
        if segment_kind(message.channel, message.recipient) == REASONING:
            out.append(text[cursor : message.start])
            cursor = message.end
    out.append(text[cursor:])
    return "".join(out)
