# SPDX-License-Identifier: Apache-2.0
"""
Reasoning parser for GPT-OSS models using channel-based format.

GPT-OSS models use a channel-based token format instead of <think>...</think> tags:
    <|channel|>analysis<|message|>[reasoning]<|start|>assistant<|channel|>final<|message|>[content]<|return|>

Some models also emit an extended format with a constrain token:
    <|channel|>final <|constrain|>JSON<|message|>[content]<|return|>

This parser extracts reasoning from the 'analysis' channel and content from
the 'final' channel, stripping all structural tokens from API responses.
"""

from .base import DeltaMessage, ReasoningParser
from .harmony_parser import HarmonyStreamingReader, split_messages


class GptOssReasoningParser(ReasoningParser):
    """
    Reasoning parser for GPT-OSS models.

    GPT-OSS uses channel-based tokens:
        <|channel|>analysis<|message|>[reasoning]
        <|start|>assistant<|channel|>final<|message|>[content]<|return|>

    The 'analysis' channel maps to reasoning, 'final' to content.

    Also handles extended format with constrain token:
        <|channel|>final <|constrain|>JSON<|message|>[content]<|return|>

    An output with no channel at all is plain content. The closing token of a
    message is usually missing from the text (it is a stop token); a message that
    runs to the end of the text is read like a terminated one.
    """

    # Its markers are control tokens: streaming sites must not filter them out first.
    READS_CONTROL_TOKENS = True

    def __init__(self, tokenizer=None):
        super().__init__(tokenizer)
        self._reader = HarmonyStreamingReader(plain_text_is_content=True)

    def extract_reasoning(
        self,
        model_output: str,
    ) -> tuple[str | None, str | None]:
        """
        Extract reasoning and content from complete model output.

        Args:
            model_output: Complete text output from the model.

        Returns:
            (reasoning, content) tuple. Either may be None.
        """
        if not model_output or "<|channel|>" not in model_output:
            return None, model_output if model_output else None

        reasoning, content = split_messages(model_output)

        # If no channels found, return as plain content
        if reasoning is None and content is None:
            return None, model_output

        return reasoning, content

    def extract_reasoning_streaming(
        self,
        previous_text: str,  # noqa: ARG002
        current_text: str,  # noqa: ARG002
        delta_text: str,
    ) -> DeltaMessage | None:
        """
        Extract reasoning from streaming delta.

        Reads ``delta_text`` only, whatever its size: a delta may hold several
        tokens, a control token and text after it, or the end of one message and the
        start of the next.

        Args:
            previous_text: Accumulated text before this delta.
            current_text: Accumulated text including this delta.
            delta_text: Just the new text in this streaming chunk.

        Returns:
            DeltaMessage with reasoning and/or content, or None to skip.
        """
        return self._reader.feed(delta_text)

    def finalize_stream(self) -> DeltaMessage | None:
        """Release a trailing partial token that never completed."""
        return self._reader.feed("", final=True)

    def reset_state(self, implicit_mode: bool = False):  # noqa: ARG002
        """Reset streaming state for a new request."""
        self._reader.reset()
