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
_THOUGHT_PREFIX = "thought"
_RESPONSE_MARKER = "<|channel>response"
# Full "open-thinking" marker. Kept as a buffering target so that partial
# prefixes arriving mid-stream (e.g. "<|channel>th", "<|channel>thoug") are
# held back until the label completes, instead of leaking 'th', 'tho', etc.
# into the reasoning output.
_THOUGHT_MARKER = "<|channel>thought"
# Everything the scanner recognises, longest first so the response marker wins
# over the bare start marker it begins with.
_MARKER_RE = re.compile(r"<\|channel>response|<\|channel>|<channel\|>")
# Markers whose proper prefixes are withheld at a delta boundary.
_PARTIAL_MARKERS = (_RESPONSE_MARKER, _THOUGHT_MARKER, "<channel|>", "<|channel>")


def _strip_channel_name(text: str, prefix: str) -> str:
    """Strip channel name and leading whitespace/newline from text start."""
    if text.startswith(prefix):
        text = text[len(prefix) :]
    return text.lstrip("\n")


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
        # What to strip from the start of the next text emitted in the current
        # channel: "" nothing, "label" a "thought" channel name and the
        # newlines after it, "newlines" leading newlines, "names" newlines and
        # stray channel names (the text after a <channel|>).
        self._strip: str = ""

    def reset_state(self, implicit_mode: bool = False):  # noqa: ARG002
        super().reset_state()
        self._pending = ""
        self._strip = ""

    @staticmethod
    def _trailing_partial_marker_len(text: str) -> int:
        """
        Return length of trailing substring of `text` that is a proper prefix
        of any marker (<channel|>, <|channel>response, <|channel>thought,
        <|channel>).

        Only counts PROPER prefixes: a marker that is already complete in
        `text` needs no buffering. Returns 0 if no partial match.
        """
        max_len = 0
        for marker in _PARTIAL_MARKERS:
            for i in range(min(len(marker) - 1, len(text)), max_len, -1):
                if text.endswith(marker[:i]):
                    max_len = i
                    break
        return max_len

    def finalize_stream(self) -> DeltaMessage | None:
        """
        Flush any buffered partial marker at the end of stream.

        If the stream ends while we have a partial marker buffered (e.g.
        model emitted "<|channel>" as its last token and got truncated by
        max_tokens), emit it so the client doesn't lose the text: as content
        when no reasoning channel is open, as reasoning otherwise.
        """
        if not self._pending:
            return None
        pending = self._pending
        self._pending = ""
        if self._phase in ("pre_think", "content"):
            return DeltaMessage(content=pending)
        return DeltaMessage(reasoning=pending)

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
        if text.count(self.start_token) >= 2 and _RESPONSE_MARKER in text:
            _, _, after_start = text.partition(self.start_token)
            last_resp = after_start.rfind(_RESPONSE_MARKER)
            reasoning = after_start[:last_resp]
            content = after_start[last_resp + len(_RESPONSE_MARKER) :]
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
        - <|channel>thought: enter reasoning, strip the channel name
        - <channel|> or <|channel>response: switch to content
        - <|channel> from content (degenerate cycling): back to reasoning

        A partial marker at the end of a delta is withheld until the next
        delta decides it (or finalize_stream() releases it).
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
        pos = 0
        for marker in _MARKER_RE.finditer(text):
            self._emit(text[pos : marker.start()], reasoning, content)
            pos = marker.end()
            token = marker.group()
            if token == self.end_token:
                self._phase = "content"
                self._strip = "names"
            elif token == _RESPONSE_MARKER:
                self._phase = "content"
                self._strip = "newlines"
            else:
                self._phase = "thinking"
                self._strip = "label"
        self._emit(text[pos:], reasoning, content)

        if not reasoning and not content:
            return None
        return DeltaMessage(
            reasoning="".join(reasoning) or None, content="".join(content) or None
        )

    def _emit(self, text: str, reasoning: list[str], content: list[str]) -> None:
        """Route text that sits between markers to the open channel."""
        if not text:
            return
        if self._phase == "pre_think":
            content.append(text)
            return
        if self._strip == "label":
            if text.startswith(_THOUGHT_PREFIX):
                text = text[len(_THOUGHT_PREFIX) :]
                self._strip = "newlines"
            else:
                self._strip = ""
        if self._strip in ("newlines", "names"):
            text = text.lstrip("\n")
            if self._strip == "names":
                text = _strip_channel_name(text, _THOUGHT_PREFIX)
                text = _strip_channel_name(text, "response")
            if text:
                self._strip = ""
        if text:
            (reasoning if self._phase == "thinking" else content).append(text)
