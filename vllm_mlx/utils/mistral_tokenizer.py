# SPDX-License-Identifier: Apache-2.0
"""
Keep Mistral ``tekken`` checkpoints on a tokenizer that can read a rendered prompt.

``transformers`` (5.x) loads ``MistralCommonBackend`` for any checkpoint that ships
``tekken.json`` (Ministral 3, Devstral Small 2, Mistral Small 3.x, Nemo, ...) even
when a ``tokenizer.json`` is present. That backend is built to be driven with
``apply_chat_template(..., tokenize=True)``: it renders and encodes the chat in one
step. This server renders the chat to a *string* (``tokenize=False``) and encodes
that string later, and the mistral-common backend then encodes the string as plain
text: ``<s>`` and ``[INST]`` become 3 and 4 ordinary tokens each, ``[/INST]``,
``[TOOL_CALLS]``, ``[SYSTEM_PROMPT]`` likewise, and the BOS id is dropped. The model
is prompted with a malformed chat and answers in kind.

The ``tokenizer.json`` of those checkpoints is the same tokenizer in the Hugging
Face fast format: control tokens are added tokens, so the rendered string encodes to
the very ids mistral-common would produce, and ``chat_template.jinja`` renders the
same prompt. ``equivalent_fast_tokenizer`` loads it and only returns it once it is
shown identical to the mistral-common tokenizer it replaces (every id decodes to the
same text, special-token ids and BOS/EOS agree, and a battery of plain-text probes
encodes to the same ids). A fast tokenizer is also what constrained decoding needs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# What ``transformers`` needs to be told to load the fast tokenizer of a tekken
# checkpoint: skip the mistral-common auto-detection, and apply the Mistral
# pre-tokenizer regex fix (a no-op when tokenizer.json already carries it, which
# is also what silences a misleading "incorrect regex" warning on such repos).
FAST_TOKENIZER_KWARGS: dict[str, Any] = {
    "mistral_format": False,
    "fix_mistral_regex": True,
}

# Plain-text strings that exercise the pre-tokenizer: contractions, digits,
# whitespace runs, CRLF, tabs, CJK/Cyrillic, combining marks, emoji, markup, quotes.
_PROBES = (
    "Hello, world!",
    "I'm sure it's fine; they'll don't can't we've",
    "12345 6789 3.14159 1,000,000",
    "  leading and trailing  ",
    "\n\n\ndef f(x):\n\treturn x**2\r\n",
    "日本語のテキスト 世界 привет мир",
    "emoji 😀👍🏽 e\u0301 ñ",
    "<html><body>x</body></html>",
    'tool_calls[0].function.name == "echo_args"',
    "don\u2019t \u201ccurly\u201d \u2014 dash\u2026",
    "a   b    c\t\td",
    "{\"text\": \"hello world\", \"count\": 42, \"flag\": true}",
)


def is_mistral_common_tokenizer(tokenizer: Any) -> bool:
    """True for ``transformers``' ``MistralCommonBackend`` (found by name, no import)."""
    return any(
        cls.__name__ == "MistralCommonBackend" for cls in type(tokenizer).__mro__
    )


def _decode_all(tokenizer: Any, size: int) -> list[str]:
    return [tokenizer.decode([i], skip_special_tokens=False) for i in range(size)]


def _mismatch(fast: Any, reference: Any) -> str | None:
    """Why ``fast`` is not the same tokenizer as ``reference``; None when it is."""
    if not getattr(fast, "is_fast", False):
        return "not a fast tokenizer"
    if not fast.chat_template:
        return "no chat template"
    size = len(reference)
    if len(fast) != size:
        return f"vocabulary size {len(fast)} != {size}"
    for name in ("bos_token_id", "eos_token_id"):
        if getattr(fast, name, None) != getattr(reference, name, None):
            return f"{name} differs"
    if _decode_all(fast, size) != _decode_all(reference, size):
        return "token texts differ"
    for probe in _PROBES:
        if fast.encode(probe, add_special_tokens=False) != reference.encode(
            probe, add_special_tokens=False
        ):
            return f"encoding of {probe!r} differs"
    return None


def reference_tokenizer(model_path: str | Path) -> Any | None:
    """The mistral-common tokenizer ``transformers`` picks for ``model_path``.

    None when the checkpoint is not a tekken one, or ``transformers`` picks a
    different tokenizer for it (``mistral-common`` not installed): nothing to fix.
    """
    path = Path(model_path)
    if not (path / "tekken.json").is_file():
        return None
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path)
    return tokenizer if is_mistral_common_tokenizer(tokenizer) else None


def equivalent_fast_tokenizer(model_path: str | Path, reference: Any) -> Any | None:
    """The checkpoint's ``tokenizer.json`` tokenizer, if identical to ``reference``.

    ``reference`` is the mistral-common tokenizer being replaced. Returns None (and
    logs why) when there is no ``tokenizer.json`` or it is not equivalent, in which
    case the caller keeps ``reference``.
    """
    path = Path(model_path)
    if not (path / "tokenizer.json").is_file():
        logger.warning(
            "%s ships tekken.json but no tokenizer.json: keeping the mistral-common "
            "tokenizer, which tokenizes control tokens ([INST], [TOOL_CALLS], <s>, ...) "
            "in a rendered prompt string as plain text",
            path,
        )
        return None
    from transformers import AutoTokenizer

    try:
        fast = AutoTokenizer.from_pretrained(path, **FAST_TOKENIZER_KWARGS)
        reason = _mismatch(fast, reference)
    except Exception as exc:  # unreadable/odd tokenizer.json: keep the reference
        reason = f"{type(exc).__name__}: {exc}"
        fast = None
    if reason is not None:
        logger.warning(
            "%s: tokenizer.json is not equivalent to the mistral-common tokenizer "
            "(%s); keeping the latter, which tokenizes control tokens in a rendered "
            "prompt string as plain text",
            path,
            reason,
        )
        return None
    logger.info(
        "Mistral tekken checkpoint: using the tokenizer.json fast tokenizer "
        "(verified identical to mistral-common over %d token ids)",
        len(fast),
    )
    return fast


def fast_tokenizer_config(
    model_path: str | Path, tokenizer_config: dict | None
) -> dict | None:
    """``tokenizer_config`` for ``mlx_lm`` that selects the equivalent fast tokenizer.

    Returned unchanged unless ``model_path`` is a tekken checkpoint for which
    ``equivalent_fast_tokenizer`` succeeds.
    """
    reference = reference_tokenizer(model_path)
    if reference is None or equivalent_fast_tokenizer(model_path, reference) is None:
        return tokenizer_config
    return {**(tokenizer_config or {}), **FAST_TOKENIZER_KWARGS}


def adopt_fast_tokenizer(
    processor: Any, model_path: str | Path, eos_token_ids: Any = None
) -> bool:
    """Replace a mistral-common ``processor.tokenizer`` by the equivalent fast one.

    ``mlx_vlm.load`` wires a streaming detokenizer and stopping criteria onto the
    processor's tokenizer, and gives up on that when it is the mistral-common one
    (it has no ``vocab``), leaving ``mlx_vlm.generate`` without either. With the
    replacement in place they are wired the way ``mlx_vlm.utils.load_processor``
    does it; ``eos_token_ids`` is the model config's, as in ``mlx_vlm.load``.
    Returns whether the tokenizer was replaced.
    """
    reference = getattr(processor, "tokenizer", None)
    if reference is None or not is_mistral_common_tokenizer(reference):
        return False
    fast = equivalent_fast_tokenizer(model_path, reference)
    if fast is None:
        return False

    from mlx_vlm.tokenizer_utils import load_tokenizer
    from mlx_vlm.utils import StoppingCriteria

    processor.tokenizer = fast
    detokenizer_class = load_tokenizer(Path(model_path), return_tokenizer=False)
    processor.detokenizer = detokenizer_class(fast)
    fast.stopping_criteria = StoppingCriteria(
        eos_token_ids,
        fast,
        additional_eos_token_ids=getattr(processor, "additional_eos_token_ids", ()),
    )
    return True
