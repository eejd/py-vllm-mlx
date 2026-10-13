# SPDX-License-Identifier: Apache-2.0
"""Does the active chat template tell the model which tools exist?

``--enable-auto-tool-choice`` only parses what the model writes; it is the chat
template that puts the request's ``tools`` in front of the model. Many conversions
ship a template that never reads ``tools`` (Phi-4-mini's, the Hermes-3-3B and Gemma-3
ChatML/turn templates). Served that way the model answers in prose and every request
ends with no tool call, with nothing in the logs to say why. This module renders a
probe once at startup so the server can say so.
"""

import logging
from collections.abc import Callable
from typing import Any

from ..templates import BUNDLED_TOOL_TEMPLATES

logger = logging.getLogger(__name__)

# A name no template could contain unless it renders the tool it was given.
_PROBE_TOOL = "vllm_mlx_template_probe_tool"
_PROBE_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": _PROBE_TOOL,
            "description": "Probe tool used to check that the chat template renders tools.",
            "parameters": {
                "type": "object",
                "properties": {"probe_argument": {"type": "string"}},
                "required": ["probe_argument"],
            },
        },
    }
]
_PROBE_MESSAGES = [{"role": "user", "content": "probe"}]


def template_renders_tools(render: Callable[..., str]) -> bool | None:
    """Whether ``render(messages, tools=...)`` puts the tools in the prompt.

    ``render`` renders a chat prompt exactly as the engine would for a request with
    tools. Returns ``True`` when the probe tool's name is in the prompt, ``False``
    when the template ignored ``tools``, and ``None`` when the template cannot be
    probed (it raised, or returned something that is not text): that is no evidence
    either way.
    """
    try:
        prompt = render(_PROBE_MESSAGES, tools=_PROBE_TOOLS)
    except Exception:  # noqa: BLE001 - any template failure just means "unknown"
        logger.debug("chat template tool probe failed", exc_info=True)
        return None
    if not isinstance(prompt, str):
        return None
    return _PROBE_TOOL in prompt


def tools_ignored_warning(parser_name: str) -> str:
    """The startup warning for a template that ignores ``tools``."""
    template = BUNDLED_TOOL_TEMPLATES.get(parser_name)
    if template is not None:
        fix = f"This package ships one for this parser: --chat-template {template}"
    else:
        fix = (
            "Serve it with a template that renders `tools` (--chat-template PATH; "
            "vLLM's examples/tool_chat_template_*.jinja are the usual starting point)"
        )
    return (
        f"--enable-auto-tool-choice is on (--tool-call-parser {parser_name}) but the "
        "chat template this model is served with does not render the request's "
        "`tools`: the model is never told which tools exist, so it will answer in "
        f"prose and no request will produce a tool call. {fix}"
    )


def engine_template_renderer(engine: Any) -> Callable[..., str] | None:
    """A ``render(messages, tools=...)`` that renders as ``engine`` does, else None.

    The batched engine owns its renderer (it picks the processor or the tokenizer).
    For the others the tokenizer is used, or the tokenizer inside a multimodal
    processor when the processor carries no template of its own (Gemma-3).
    """
    apply_template = getattr(engine, "_apply_chat_template", None)
    if callable(apply_template):
        return lambda messages, tools=None: apply_template(messages, tools=tools)

    tokenizer = getattr(engine, "tokenizer", None)
    for candidate in (tokenizer, getattr(tokenizer, "tokenizer", None)):
        if candidate is None or not getattr(candidate, "chat_template", None):
            continue
        if callable(getattr(candidate, "apply_chat_template", None)):
            return lambda messages, tools=None, _tok=candidate: _tok.apply_chat_template(
                messages, tools=tools, tokenize=False, add_generation_prompt=True
            )
    return None
