# SPDX-License-Identifier: Apache-2.0
"""
Reasoning parser for Gemma 4 models.

Gemma 4 uses a channel-based protocol for reasoning:

    <|channel>thought
    ...thinking content...
    <channel|>
    ...response content...

Where:
    <|channel> = token 100 (channel switch marker)
    <channel|> = token 101 (end-of-channel marker)

The channel names "thought" and "response" appear as text after the
special tokens and should be stripped from the output.

Some model variants may use <|channel>response instead of <channel|>
to transition from thinking to response mode. This parser handles both.

When thinking is disabled or not triggered, output contains no tags.

Degenerate cycling:
    On long prompts with tools, Gemma 4 may oscillate between thought and
    response channels many times, producing garbage reasoning before finally
    emitting valid content/tool_calls. The parser handles this by splitting
    at the LAST <channel|> so all cycles go into reasoning_content and only
    the final response goes into content. Channel tokens are stripped from
    both sides.
"""

import re

from .base import DeltaMessage
from .think_parser import BaseThinkingReasoningParser

# Channel names that follow <|channel> — stripped from output
_CHANNEL_NAMES = ("thought", "response")
_RESPONSE_MARKER = "<|channel>response"
# "<|channel>response" is the marker only when no letter follows: "<|channel>responses"
# is "<|channel>" and the word "responses".
_RESPONSE_RE = re.compile(r"<\|channel>response(?!\w)")
# Everything the scanner recognises, longest first so the response marker wins
# over the bare start marker it begins with.
_MARKER_RE = re.compile(r"<\|channel>response(?!\w)|<\|channel>|<channel\|>")
# Markers whose proper prefixes are withheld at a delta boundary. A complete
# "<|channel>" is withheld too: it may still become "<|channel>response".
_PARTIAL_MARKERS = (_RESPONSE_MARKER, "<channel|>", "<|channel>")


def _strip_channel_tokens(text: str) -> str:
    """Remove all channel special tokens and bare channel names from text.

    Handles degenerate model output with multiple thought/response cycles
    by stripping all protocol tokens, leaving only the actual text content.
    """
    # Remove special tokens
    text = text.replace("<channel|>", "")
    text = text.replace("<|channel>", "")
    # Remove channel names on standalone lines
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        s = line.strip()
        if s in ("thought", "response"):
            continue
        cleaned.append(line)
    text = "\n".join(cleaned)
    # Strip leading channel name
    text = text.strip()
    for name in ("thought", "response"):
        if text.startswith(name + "\n"):
            text = text[len(name) + 1 :]
            break
        if text.startswith(name) and (
            len(text) == len(name) or not text[len(name)].isalpha()
        ):
            text = text[len(name) :]
            break
    return text.strip()


class _SegmentFilter:
    """Streaming form of ``_strip_channel_tokens`` for the text of one channel segment.

    The text between two markers is read the way the whole-text extraction reads it:
    lines that are only a channel name ("thought", "response") are dropped, a leading
    channel name is dropped when a non-letter follows it ("thought, so" but not
    "thoughtful"), and the ends are stripped. Text that could still turn out to be a
    channel name (a line start, the first word) is held until the next characters
    decide it, whatever the delta boundaries; whitespace at the end is held until more
    text follows in the segment and dropped when the segment ends.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._line = ""  # the current line, held while it may still be a lone name
        self._line_free = False  # it cannot be one: stream it as it comes
        self._head = ""  # leading text, held until the first word is decided
        self._head_done = False
        self._tail = ""  # trailing whitespace, held

    @staticmethod
    def _may_be_name(text: str) -> bool:
        word = text.strip()
        return not word or any(name.startswith(word) for name in _CHANNEL_NAMES)

    def feed(self, text: str) -> str:
        out: list[str] = []
        for i, part in enumerate(text.split("\n")):
            if i:
                out.append(self._end_line())
            out.append(self._extend_line(part))
        return self._trim("".join(out))

    def flush(self) -> str:
        """The segment ends: whatever is held is decided now. The filter is reset."""
        piece = ""
        if not self._line_free and self._line.strip() not in _CHANNEL_NAMES:
            piece = self._line
        self._line = ""
        out = self._trim(piece)
        self.reset()
        return out

    def _extend_line(self, part: str) -> str:
        if self._line_free:
            return part
        self._line += part
        if self._may_be_name(self._line):
            return ""
        self._line_free = True
        line, self._line = self._line, ""
        return line

    def _end_line(self) -> str:
        """A newline ended the line: a lone channel name goes, with its newline."""
        free, line = self._line_free, self._line
        self._line_free, self._line = False, ""
        if free:
            return "\n"
        return "" if line.strip() in _CHANNEL_NAMES else line + "\n"

    @staticmethod
    def _strip_name(text: str) -> str:
        """Drop a leading channel name that a non-letter follows; text is lstripped."""
        for name in _CHANNEL_NAMES:
            if text.startswith(name):
                rest = text[len(name) :]
                if not rest or not rest[0].isalpha():
                    return rest.lstrip()
        return text

    def _trim(self, text: str) -> str:
        """Strip the segment: the leading end once the first word is decided, the
        trailing end as whitespace is held back until text follows it."""
        if not self._head_done:
            self._head += text
            head = self._head.lstrip()
            if not head:
                return ""  # nothing but whitespace yet
            self._head_done = True
            self._head = ""
            text = self._strip_name(head)
        joined = self._tail + text
        body = joined.rstrip()
        self._tail = joined[len(body) :]
        return body


class Gemma4ReasoningParser(BaseThinkingReasoningParser):
    """
    Reasoning parser for Gemma 4 models.

    Handles two transition formats:
    1. <|channel>thought...<channel|>response  (standard: token 100 + 101)
    2. <|channel>thought...<|channel>response   (alternative: token 100 + 100)

    Channel names ("thought", "response") are stripped from output.

    Example:
        Input:  "<|channel>thought\\nLet me think...<channel|>The answer is 42."
        Output: reasoning="Let me think...", content="The answer is 42."

    When no tags are present, the entire output is treated as content.

    Degenerate cycling (long prompts + tools):
        Uses rpartition to split at the LAST <channel|>, so all intermediate
        thought/response cycles go into reasoning and only the final response
        goes into content.

    Streaming buffering:
        Partial markers at a delta boundary (e.g. "<|channel>" without a
        following "response" yet) are buffered internally so they don't
        leak into reasoning/content. The buffer is either consumed when
        the marker completes in a later delta, or flushed as reasoning
        via finalize_stream() when the stream ends.
    """

    @property
    def start_token(self) -> str:
        return "<|channel>"

    @property
    def end_token(self) -> str:
        return "<channel|>"

    def __init__(self, tokenizer=None):
        super().__init__(tokenizer)
        # Trailing text withheld because it could complete into a marker
        # (e.g. "<|channel>" that might become "<|channel>response" next delta).
        self._pending: str = ""
        self._filter = _SegmentFilter()
        self._seen_start = False  # a "<|channel>" has opened a channel in this stream

    def reset_state(self, implicit_mode: bool = False):  # noqa: ARG002
        super().reset_state()
        self._pending = ""
        self._filter.reset()
        self._seen_start = False

    @staticmethod
    def _trailing_partial_marker_len(text: str) -> int:
        """
        Return length of trailing substring of `text` that is a proper prefix
        of any marker (<channel|>, <|channel>response, <|channel>).

        Only counts PROPER prefixes: a marker that is already complete in
        `text` needs no buffering. Returns 0 if no partial match.
        """
        # a complete "<|channel>response" may still continue into a word
        max_len = len(_RESPONSE_MARKER) if text.endswith(_RESPONSE_MARKER) else 0
        for marker in _PARTIAL_MARKERS:
            for i in range(min(len(marker) - 1, len(text)), max_len, -1):
                if text.endswith(marker[:i]):
                    max_len = i
                    break
        return max_len

    def finalize_stream(self) -> DeltaMessage | None:
        """
        End of stream: decide whatever is still held back.

        A partial marker withheld at the end (the model was cut off after
        "<|channel>" or "<|channel>respo") is read as the whole-text extraction
        reads it: a complete marker is a marker, anything else is text of the
        open channel. Text held while it could still have been a channel name,
        and trailing whitespace, are settled the same way.
        """
        pending, self._pending = self._pending, ""
        reasoning: list[str] = []
        content: list[str] = []
        self._scan(pending, reasoning, content)
        self._route(self._filter.flush(), reasoning, content)
        if not reasoning and not content:
            return None
        return DeltaMessage(
            reasoning="".join(reasoning) or None, content="".join(content) or None
        )

    def extract_reasoning(
        self,
        model_output: str,
    ) -> tuple[str | None, str | None]:
        """
        Extract reasoning from complete output.

        Uses rpartition (LAST <channel|>) to handle degenerate cycling:
        all intermediate thought/response cycles go into reasoning,
        only the final response goes into content. Channel tokens
        are stripped from both sides.
        """
        text = model_output

        # Standard format: <|channel>thought...<channel|>content
        # Use rpartition on LAST <channel|> to handle multiple cycles
        if self.start_token in text and self.end_token in text:
            _, _, after_start = text.partition(self.start_token)
            reasoning, _, content = after_start.rpartition(self.end_token)
            reasoning = _strip_channel_tokens(reasoning)
            content = _strip_channel_tokens(content)
            return reasoning or None, content or None

        # Alternative format: <|channel>thought...<|channel>response...
        # Use rfind for the LAST <|channel>response marker
        responses = list(_RESPONSE_RE.finditer(text))
        if text.count(self.start_token) >= 2 and responses:
            _, _, after_start = text.partition(self.start_token)
            last_resp = list(_RESPONSE_RE.finditer(after_start))[-1]
            reasoning = after_start[: last_resp.start()]
            content = after_start[last_resp.end() :]
            reasoning = _strip_channel_tokens(reasoning)
            content = _strip_channel_tokens(content)
            return reasoning or None, content or None

        # Only closing tag (think injected in prompt)
        if self.end_token in text:
            reasoning, _, content = text.rpartition(self.end_token)
            reasoning = _strip_channel_tokens(reasoning)
            content = _strip_channel_tokens(content)
            return reasoning or None, content or None

        # Only start tag (incomplete reasoning, no end yet)
        if self.start_token in text:
            _, _, reasoning = text.partition(self.start_token)
            reasoning = _strip_channel_tokens(reasoning)
            return reasoning or None, None

        # No tags at all — pure content
        return None, model_output

    def extract_reasoning_streaming(
        self,
        previous_text: str,  # noqa: ARG002
        current_text: str,  # noqa: ARG002
        delta_text: str,
    ) -> DeltaMessage | None:
        """
        Extract reasoning from a streaming delta.

        Only ``delta_text`` is read; the parser keeps its own state, so the
        cost per delta does not depend on how much has been generated. The
        delta is scanned marker by marker, so it may carry any mix of text and
        markers (several tokens at once, as with speculative decoding):

        - No markers yet: content (Gemma 4 doesn't inject tags in the prompt)
        - <|channel>: enter reasoning (the "thought" channel name is stripped)
        - <channel|> or <|channel>response: switch to content
        - <|channel> from content (degenerate cycling): back to reasoning

        The text of a channel comes out as ``extract_reasoning()`` returns it
        for one thought/response cycle: channel names stripped, ends stripped.
        A partial marker, text that could still be a channel name, and trailing
        whitespace are held back until the next delta decides them (or
        finalize_stream() does).
        """
        text = self._pending + delta_text
        hold = self._trailing_partial_marker_len(text)
        self._pending = text[len(text) - hold :] if hold else ""
        if hold:
            text = text[: len(text) - hold]
        if not text:
            return None

        reasoning: list[str] = []
        content: list[str] = []
        self._scan(text, reasoning, content)
        if not reasoning and not content:
            return None
        return DeltaMessage(
            reasoning="".join(reasoning) or None, content="".join(content) or None
        )

    def _scan(self, text: str, reasoning: list[str], content: list[str]) -> None:
        """Route ``text`` (no partial marker at its end) marker by marker."""
        pos = 0
        for marker in _MARKER_RE.finditer(text):
            self._segment(text[pos : marker.start()], reasoning, content)
            # the open channel ends here: whatever it held back is decided now
            self._route(self._filter.flush(), reasoning, content)
            pos = marker.end()
            token = marker.group()
            if token == self.start_token:
                self._seen_start = True
                self._phase = "thinking"
            elif token == "<channel|>" or self._seen_start:
                self._phase = "content"
            else:
                # a lone "<|channel>response" is the start of a channel like any other,
                # as extract_reasoning() reads it; it is the switch to the answer only
                # after a channel was opened (<|channel>thought...<|channel>response)
                self._seen_start = True
                self._phase = "thinking"
                self._segment("response", reasoning, content)
        self._segment(text[pos:], reasoning, content)

    def _segment(self, text: str, reasoning: list[str], content: list[str]) -> None:
        """Text of the open channel. Before any marker it is plain content, as is."""
        if self._phase == "pre_think":
            if text:
                content.append(text)
        else:
            self._route(self._filter.feed(text), reasoning, content)

    def _route(self, text: str, reasoning: list[str], content: list[str]) -> None:
        """Hand text that has been through the segment filter to the open channel."""
        if not text:
            return
        if self._phase == "thinking":
            reasoning.append(text)
        else:
            content.append(text)
