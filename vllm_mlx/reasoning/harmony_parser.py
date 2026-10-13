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
from .harmony_channels import Body, HarmonyScanner, Open, Plain, parse_messages


class HarmonyStreamingReader:
    """Routes the bodies of a Harmony stream: analysis is reasoning, final is content.

    Shared by the Harmony reasoning parsers. Commentary (tool calls, preambles) and
    every other channel is not routed: the tool parser reads it from the raw stream.
    Several analysis (or final) messages are joined with a newline, as the whole-text
    extraction does. ``plain_text_is_content`` says what text outside any message is:
    content for a parser whose whole-text extraction returns an untagged output as
    content, dropped otherwise.
    """

    def __init__(self, plain_text_is_content: bool) -> None:
        self._plain_text_is_content = plain_text_is_content
        self._scanner = HarmonyScanner()
        self.reset()

    def reset(self) -> None:
        self._scanner.reset()
        self._emitted = {"analysis": False, "final": False}
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
            elif isinstance(event, Body):
                if event.channel not in self._emitted:
                    continue
                text = event.text
                if not self._message_started:
                    text = text.lstrip()
                    if not text:
                        continue
                    self._message_started = True
                    if self._emitted[event.channel]:
                        text = "\n" + text
                self._emitted[event.channel] = True
                (reasoning if event.channel == "analysis" else content).append(text)
            elif (
                isinstance(event, Plain) and event.leading and self._plain_text_is_content
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

    Extracts reasoning from the 'analysis' channel and content from
    the 'final' channel. Commentary channels (tool calls) are ignored
    since they are handled by the tool parser.

    Example:
        Input: "<|channel|>analysis<|message|>Thinking...<|end|>
                <|channel|>final<|message|>Result.<|return|>"
        Output: reasoning="Thinking...", content="Result."

    ``<|end|>``, ``<|return|>`` and ``<|call|>`` are stop tokens, which a generation
    stops on without decoding them, so the text often ends without one: a message that
    runs to the end of the text is read like a terminated one.
    """

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
        messages, _ = parse_messages(model_output)
        reasoning = "\n".join(
            m.body.strip() for m in messages if m.channel == "analysis" and m.body.strip()
        )
        content = "\n".join(
            m.body.strip() for m in messages if m.channel == "final" and m.body.strip()
        )
        return reasoning or None, content or None

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
