# SPDX-License-Identifier: Apache-2.0
"""Safety normalization for messages before Jinja chat-template rendering."""

import functools
import json
import logging
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from typing import Any, TypeVar

import jinja2

from ..request import ChatTemplateRejected

T = TypeVar("T")
logger = logging.getLogger(__name__)


@contextmanager
def chat_template_rejections() -> Iterator[None]:
    """Turn a template's rejection of the request's messages into ``ChatTemplateRejected``.

    Chat templates validate their input by raising from inside the template
    (``raise_exception(...)`` becomes ``jinja2.TemplateError``) or by failing while
    rendering the client's messages (``UndefinedError``). Both mean the request is
    wrong for this model, so the server answers 400 with the template's message, as
    vLLM does (``safe_apply_chat_template``). A ``TemplateSyntaxError`` means the
    template itself does not compile, whatever the request says: that stays a server
    error, as does everything that is not a Jinja error.
    """
    try:
        yield
    except jinja2.TemplateSyntaxError:
        raise
    except jinja2.TemplateError as exc:
        logger.warning("Chat template rejected the request: %s", exc)
        raise ChatTemplateRejected(str(exc)) from exc


def reject_template_errors(
    method: Callable[..., Awaitable[T]],
) -> Callable[..., Awaitable[T]]:
    """Run an async engine method under ``chat_template_rejections``."""

    @functools.wraps(method)
    async def wrapper(*args: Any, **kwargs: Any) -> T:
        with chat_template_rejections():
            return await method(*args, **kwargs)

    return wrapper


def _close_dangling_think_before_tool_call(content: str) -> str:
    """Keep raw tool XML out of an unterminated ``<think>`` section.

    Qwen 3.6 can produce assistant history where ``<think>`` is opened and a
    raw ``<tool_call>`` follows before ``</think>``. Rendering that history as-is
    conditions the next turn as though the tool call is still reasoning. Close
    the dangling thinking span immediately before the first tool call.

    This mirrors the template-side repair described by Cheuk-Yiu Chan:
    https://allanchan339.github.io/bug-fixes/2026/05/02/Qwen36-27B-updated-jinja.html
    """
    if "<tool_call>" not in content or "<think>" not in content:
        return content

    last_think = content.rfind("<think>")
    last_close = content.rfind("</think>")
    tool_pos = content.find("<tool_call>")
    if last_close >= last_think and last_close != -1:
        return content
    if tool_pos > last_think:
        return content[:tool_pos] + "</think>" + content[tool_pos:]
    return content + "</think>"


def _message_to_dict(message: Any) -> dict[str, Any] | Any:
    """Convert OpenAI message model objects without stringifying them."""
    if isinstance(message, dict):
        return dict(message)
    model_dump = getattr(message, "model_dump", None)
    if callable(model_dump):
        return {
            key: value
            for key, value in model_dump(exclude_none=True).items()
            if value is not None
        }
    legacy_dict = getattr(message, "dict", None)
    if callable(legacy_dict):
        return {k: v for k, v in legacy_dict().items() if v is not None}
    return message


def normalize_messages_for_chat_template(messages: list[Any]) -> list[dict]:
    """Return a JSON-safe copy of messages for chat-template rendering.

    Normalizations:
    - close dangling ``<think>`` spans before raw ``<tool_call>`` XML in
      assistant content
    - convert OpenAI tool-call argument JSON strings to mappings for templates
      that iterate argument keys
    """
    normalized = json.loads(
        json.dumps([_message_to_dict(message) for message in messages], default=str)
    )
    for message in normalized:
        if not isinstance(message, dict):
            continue
        if message.get("role") != "assistant":
            continue

        content = message.get("content")
        if isinstance(content, str):
            message["content"] = _close_dangling_think_before_tool_call(content)

        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function")
            if not isinstance(function, dict):
                continue
            arguments: Any = function.get("arguments")
            if not isinstance(arguments, str):
                continue
            try:
                parsed = json.loads(arguments)
            except (json.JSONDecodeError, ValueError, TypeError):
                parsed = {"value": arguments}
            if not isinstance(parsed, dict):
                parsed = {"value": parsed}
            function["arguments"] = parsed
    return normalized


def build_system_prompt_cache_prefix(
    tokenizer: Any,
    messages: list[Any],
    *,
    template_kwargs: dict[str, Any],
    normalized_messages: list[dict] | None = None,
) -> str | None:
    """Render the stable system prefix used by the SimpleEngine KV cache."""
    normalized = (
        normalized_messages
        if normalized_messages is not None
        else normalize_messages_for_chat_template(messages)
    )
    system_messages = [
        dict(message)
        for message in normalized
        if isinstance(message, dict) and message.get("role") == "system"
    ]
    if not system_messages:
        return None

    def _render(user_content: str) -> Any:
        probe_messages = [
            *system_messages,
            {"role": "user", "content": user_content},
        ]
        return tokenizer.apply_chat_template(probe_messages, **template_kwargs)

    try:
        rendered_a = _render("Alpha")
        rendered_b = _render("Bravo")
    except Exception:
        return None

    if not isinstance(rendered_a, str) or not isinstance(rendered_b, str):
        return None

    for boundary, (char_a, char_b) in enumerate(zip(rendered_a, rendered_b)):
        if char_a != char_b:
            break
    else:
        return None

    if boundary < 16:
        return None
    return rendered_a[:boundary]
