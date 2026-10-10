# SPDX-License-Identifier: Apache-2.0
"""``--chat-template``: replace the chat template a model ships with.

Some models cannot be served as shipped. Phi-4-mini's own template ignores the
OpenAI ``tools=`` argument, so the model is never told which tools exist. vLLM
solves this with ``--chat-template``; this module is the same option here.

The override is installed on the tokenizer (and the processor and its tokenizer for
multimodal models) right after the model loads. Every prompt-building path reads the
template from there, so one assignment reaches the Simple and Batched engines, the
MLLM text route, the system-prompt cache probes and the template fingerprint the
server keys its caches on.
"""

import argparse
import codecs
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# A value containing none of these cannot be a Jinja template, so a value that is not
# an existing file is a mistyped path, not an inline template.
_JINJA_CHARS = ("{", "}", "\n")


CHAT_TEMPLATE_HELP = (
    "Replace the chat template the model ships with: a path to a Jinja template "
    "file, or the template itself on one line (\\n escapes are decoded). Applies to "
    "text and multimodal models. Needed for Phi-4-mini tool calling, whose own "
    "template ignores tools: use the template shipped in the package, "
    "vllm_mlx/templates/tool_chat_template_phi4_mini.jinja. Not supported with "
    "--models-config."
)


def resolve_chat_template(value: str) -> str:
    """Return the template text for a ``--chat-template`` value.

    ``value`` is a path to a template file, or the template itself in single-line
    form (``\\n`` escapes are decoded, as in vLLM). Raises ``ValueError`` when it is
    neither or the text is not valid Jinja.
    """
    if not value or not value.strip():
        raise ValueError("chat template is empty")

    path = os.path.expanduser(value)
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            template = f.read()
    elif not any(c in value for c in _JINJA_CHARS):
        raise ValueError(
            f"chat template {value!r} is not an existing file and does not look "
            "like an inline Jinja template (no '{', '}' or newline)"
        )
    else:
        template = codecs.decode(value, "unicode_escape")

    from jinja2 import Environment, TemplateSyntaxError

    try:
        Environment().parse(template)
    except TemplateSyntaxError as exc:
        raise ValueError(
            f"chat template is not valid Jinja: {exc.message} (line {exc.lineno})"
        ) from exc
    return template


def chat_template_arg(value: str) -> str:
    """Argparse ``type=`` for ``--chat-template``: resolves to the template text."""
    try:
        return resolve_chat_template(value)
    except (OSError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"--chat-template: {exc}") from exc


def apply_chat_template_override(target: Any, template: str | None) -> None:
    """Install ``template`` on a tokenizer or processor (no-op when it is None).

    A processor and the tokenizer inside it each carry their own copy and
    ``apply_chat_template`` reads whichever object it was called on, so both get it.
    """
    if template is None or target is None:
        return
    target.chat_template = template
    inner = getattr(target, "tokenizer", None)
    if inner is not None and inner is not target:
        inner.chat_template = template
    logger.info("Chat template overridden by --chat-template (%d chars)", len(template))
