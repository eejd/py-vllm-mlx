# SPDX-License-Identifier: Apache-2.0
"""Chat templates shipped with vllm-mlx, for use with ``--chat-template``."""

from pathlib import Path

TEMPLATE_DIR = Path(__file__).resolve().parent

PHI4_MINI_TOOL_TEMPLATE = TEMPLATE_DIR / "tool_chat_template_phi4_mini.jinja"
HERMES_TOOL_TEMPLATE = TEMPLATE_DIR / "tool_chat_template_hermes.jinja"

# ``--tool-call-parser`` name -> the shipped template that tells a model of that family
# which tools exist, for checkpoints whose own template ignores ``tools=``.
BUNDLED_TOOL_TEMPLATES: dict[str, Path] = {
    "hermes": HERMES_TOOL_TEMPLATE,
    "nous": HERMES_TOOL_TEMPLATE,
    "phi4_mini_json": PHI4_MINI_TOOL_TEMPLATE,
    "phi4_mini": PHI4_MINI_TOOL_TEMPLATE,
    "phi4": PHI4_MINI_TOOL_TEMPLATE,
}
