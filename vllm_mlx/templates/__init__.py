# SPDX-License-Identifier: Apache-2.0
"""Chat templates shipped with vllm-mlx, for use with ``--chat-template``."""

from pathlib import Path

TEMPLATE_DIR = Path(__file__).resolve().parent

PHI4_MINI_TOOL_TEMPLATE = TEMPLATE_DIR / "tool_chat_template_phi4_mini.jinja"
