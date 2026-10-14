# SPDX-License-Identifier: Apache-2.0
"""
Reasoning parser for GPT-OSS models using Harmony format.

Harmony uses channels for reasoning vs final content:

    <|channel|>analysis
    <|message|>Let me think about this...
    <|end|>
    <|channel|>final
    <|message|>The answer is 42.
    <|return|>

The analysis channel contains reasoning, and the final channel
contains the user-facing response.
"""

from .base import DeltaMessage, ReasoningParser
from .harmony_channels import (
    CONTENT,
    REASONING,
    Body,
    BodyTrimmer,
    HarmonyScanner,
    Open,
    Plain,
    parse_messages,
    segment_kind,
)


def split_messages(model_output: str) -> tuple[str | None, str | None]:
    """(reasoning, content) of a complete Harmony output, as vLLM's Harmony parser reads
    it: the analysis messages, and the final and call-less commentary (preamble)
    messages. Several messages of a kind are joined with a newline; calls are neither.
    """
    messages, _ = parse_messages(model_output)
    reasoning = "\n".join(
        m.body.strip()
        for m in messages
        if m.body.strip() and segment_kind(m.channel, m.recipient) == REASONING
    )
    content = "\n".join(
        m.body.strip()
        for m in messages
        if m.body.strip() and segment_kind(m.channel, m.recipient) == CONTENT
    )
    return reasoning or None, content or None


class HarmonyStreamingReader:
    """Routes the bodies of a Harmony stream: analysis is reasoning, final and preamble
    messages are content.

    Shared by the Harmony reasoning parsers. Messages addressed to a function (the
    calls) and every other channel are not routed: the tool parser reads them from the
    raw stream. Several messages of a kind are joined with a newline, as the whole-text
    extraction does; each message body is stripped, as the whole-text extraction strips
    it (leading whitespace dropped, trailing whitespace held until more text follows).
    ``plain_text_is_content`` says what text outside any message is:
    content for a parser whose whole-text extraction returns an untagged output as
    content, dropped otherwise.
    """

    def __init__(self, plain_text_is_content: bool) -> None:
        self._plain_text_is_content = plain_text_is_content
        self._scanner = HarmonyScanner()
        self._trim = BodyTrimmer()
        self.reset()

    def reset(self) -> None:
        self._scanner.reset()
        self._trim.reset()
        self._emitted = {REASONING: False, CONTENT: False}
        self._message_started = False

    def feed(self, delta: str, *, final: bool = False) -> DeltaMessage | None:
        events = self._scanner.feed(delta)
        if final:
            events += self._scanner.finish()
        reasoning: list[str] = []
        content: list[str] = []
        for event in events:
            if isinstance(event, Open):
                self._message_started = False
                self._trim.reset()
            elif isinstance(event, Body):
                kind = segment_kind(event.channel, event.recipient)
                if kind not in self._emitted:
                    continue
                text = self._trim.feed(event.text)
                if not text:
                    continue
                if not self._message_started:
                    self._message_started = True
                    if self._emitted[kind]:
                        text = "\n" + text
                self._emitted[kind] = True
                (reasoning if kind == REASONING else content).append(text)
            elif (
                isinstance(event, Plain)
                and event.leading
                and self._plain_text_is_content
            ):
                content.append(event.text)
        if not reasoning and not content:
            return None
        return DeltaMessage(
            reasoning="".join(reasoning) or None, content="".join(content) or None
        )


class HarmonyReasoningParser(ReasoningParser):
    """
    Reasoning parser for GPT-OSS models using Harmony format.

    Extracts reasoning from the 'analysis' channel and content from the 'final'
    channel and from commentary messages addressed to no one (preambles to the user);
    messages addressed to a function are tool calls, handled by the tool parser.

    Example:
        Input: "<|channel|>analysis<|message|>Thinking...<|end|>
                <|channel|>final<|message|>Result.<|return|>"
        Output: reasoning="Thinking...", content="Result."

    ``<|end|>``, ``<|return|>`` and ``<|call|>`` are stop tokens, which a generation
    stops on without decoding them, so the text often ends without one: a message that
    runs to the end of the text is read like a terminated one.
    """

    # Its markers are control tokens: streaming sites must not filter them out first.
    READS_CONTROL_TOKENS = True
    # With thinking disabled the server still parses an output that opens a channel
    # (the model emits analysis whatever the template kwarg says): this is the marker it
    # looks for, so the analysis never reaches the client as content.
    start_token = "<|channel|>"

    def __init__(self, tokenizer=None):
        super().__init__(tokenizer)
        self._reader = HarmonyStreamingReader(plain_text_is_content=False)

    def extract_reasoning(
        self,
        model_output: str,
    ) -> tuple[str | None, str | None]:
        """
        Extract reasoning from complete Harmony output.

        Collects all analysis channel messages as reasoning and the
        final channel messages as content.

        Args:
            model_output: Complete model output text.

        Returns:
            (reasoning, content) tuple. Either may be None.
        """
        return split_messages(model_output)

    def extract_reasoning_streaming(
        self,
        previous_text: str,  # noqa: ARG002
        current_text: str,  # noqa: ARG002
        delta_text: str,
    ) -> DeltaMessage | None:
        """
        Extract reasoning from streaming Harmony output.

        Reads ``delta_text`` only, whatever its size: a delta may hold several
        tokens, a control token and text after it, or the end of one message and the
        start of the next.
        """
        return self._reader.feed(delta_text)

    def finalize_stream(self) -> DeltaMessage | None:
        """Release a trailing partial token that never completed."""
        return self._reader.feed("", final=True)

    def reset_state(self, implicit_mode: bool = False):  # noqa: ARG002
        """Reset streaming state for a new request."""
        self._reader.reset()
