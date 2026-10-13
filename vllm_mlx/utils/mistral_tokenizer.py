# SPDX-License-Identifier: Apache-2.0
"""
Keep Mistral ``tekken`` checkpoints on a tokenizer that can read a rendered prompt.

``transformers`` (5.x) loads ``MistralCommonBackend`` for any checkpoint that ships
``tekken.json`` (Ministral 3, Devstral Small 2, Mistral Small 3.x, Nemo, ...) even
when a ``tokenizer.json`` is present. That backend is built to be driven with
``apply_chat_template(..., tokenize=True)``: it renders and encodes the chat in one
step. This server renders the chat to a *string* (``tokenize=False``) and encodes that
string later, and the mistral-common backend then encodes the string as plain text:
``<s>`` and ``[INST]`` become 3 and 4 ordinary tokens each, ``[/INST]``,
``[TOOL_CALLS]``, ``[SYSTEM_PROMPT]`` likewise, and the BOS id is dropped. The model
is prompted with a malformed chat and answers in kind.

The ``tokenizer.json`` of those checkpoints is the same tokenizer in the Hugging
Face fast format: control tokens are added tokens, so the rendered string encodes to
the very ids mistral-common would produce, and ``chat_template.jinja`` renders the
same prompt. ``equivalent_fast_tokenizer`` loads it and only returns it once it is
shown identical to the tekken tokenizer it replaces: every id maps to the same bytes
(read from ``tekken.json`` itself, so ids that are only partial UTF-8 are covered),
the control tokens carry the same names, BOS/EOS agree, and a battery of plain-text
probes encodes to the same ids as mistral-common's. A fast tokenizer is also what
constrained decoding needs.
"""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# What ``transformers`` needs to be told to load the fast tokenizer of a tekken
# checkpoint: skip the mistral-common auto-detection, and apply the Mistral
# pre-tokenizer regex fix (a no-op when tokenizer.json already carries the right
# pattern, which also silences a misleading "incorrect regex" warning on such repos;
# it rescues a tokenizer.json that still has the old pattern, which the equivalence
# check would otherwise reject).
FAST_TOKENIZER_KWARGS: dict[str, Any] = {
    "mistral_format": False,
    "fix_mistral_regex": True,
}

_CODE = '''
def fibonacci(n: int) -> list[int]:
    """Return the first n Fibonacci numbers."""
    a, b, out = 0, 1, []
    for _ in range(n):
        out.append(a)
        a, b = b, a + b
    return out  # e.g. fibonacci(10) -> [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
'''

# Plain-text strings that exercise the pre-tokenizer and the merges: contractions,
# digits, whitespace runs, CRLF, tabs, CJK/Cyrillic/Arabic, combining marks, emoji,
# markup, quotes, code, JSON.
_PROBES = (
    "Hello, world!",
    "I'm sure it's fine; they'll don't can't we've",
    "12345 6789 3.14159 1,000,000 0x7fffffff 2026-10-13T12:00:00Z",
    "  leading and trailing  ",
    "\n\n\ndef f(x):\n\treturn x**2\r\n",
    "日本語のテキスト 世界 привет мир مرحبا بالعالم",
    "emoji 😀👍🏽 e\u0301 ñ ß ǆ",
    '<html><body class="a">x &amp; y</body></html>',
    'tool_calls[0].function.name == "echo_args"',
    "don\u2019t \u201ccurly\u201d \u2014 dash\u2026",
    "a   b    c\t\td",
    '{"text": "hello world", "count": 42, "flag": true, "nested": {"a": [1, 2, 3]}}',
    _CODE,
    "The quick brown fox jumps over the lazy dog. " * 20,
    "ÀÉÎÕÜ àéîõü Œœ Ææ ÿ Ÿ",
)


def is_mistral_common_tokenizer(tokenizer: Any) -> bool:
    """True for ``transformers``' ``MistralCommonBackend`` (found by name, no import)."""
    return any(
        cls.__name__ == "MistralCommonBackend" for cls in type(tokenizer).__mro__
    )


def holds_mistral_common_tokenizer(processor: Any) -> bool:
    """True when ``processor`` is, or wraps in ``.tokenizer``, a mistral-common one."""
    return is_mistral_common_tokenizer(processor) or is_mistral_common_tokenizer(
        getattr(processor, "tokenizer", None)
    )


def _unicode_to_byte() -> dict[str, int]:
    """Inverse of the GPT-2 byte-to-unicode table byte-level BPE vocabularies use."""
    keep = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    chars = keep[:]
    extra = 0
    for byte in range(256):
        if byte not in keep:
            keep.append(byte)
            chars.append(256 + extra)
            extra += 1
    return {chr(c): b for b, c in zip(keep, chars)}


def _token_mismatch(fast: Any, tekken_path: Path) -> str | None:
    """Compare every id of ``fast`` with ``tekken.json``: control-token names, bytes."""
    with open(tekken_path, encoding="utf-8") as handle:
        tekken = json.load(handle)
    config = tekken["config"]
    n_special = config["default_num_special_tokens"]
    n_regular = config["default_vocab_size"] - n_special

    token_of_id = {i: token for token, i in fast.get_vocab().items()}
    to_byte = _unicode_to_byte()

    for special in tekken["special_tokens"]:
        rank, name = special["rank"], special.get("token_str")
        if name is not None and token_of_id.get(rank) != name:
            return f"id {rank} is {token_of_id.get(rank)!r}, tekken has {name!r}"
    for entry in tekken["vocab"][:n_regular]:
        token_id = n_special + entry["rank"]
        token = token_of_id.get(token_id)
        try:
            data = bytes(to_byte[char] for char in token)
        except (KeyError, TypeError):
            return f"id {token_id} ({token!r}) is not a byte-level token"
        if data != base64.b64decode(entry["token_bytes"]):
            return f"id {token_id} is {data!r}, tekken has a different token"
    return None


def _mismatch(fast: Any, reference: Any, tekken_path: Path) -> str | None:
    """Why ``fast`` is not the same tokenizer as ``reference``; None when it is."""
    if not getattr(fast, "is_fast", False):
        return "not a fast tokenizer"
    if not fast.chat_template:
        return "no chat template"
    if len(fast) != len(reference):
        return f"vocabulary size {len(fast)} != {len(reference)}"
    for name in ("bos_token_id", "eos_token_id"):
        if getattr(fast, name, None) != getattr(reference, name, None):
            return f"{name} differs"
    reason = _token_mismatch(fast, tekken_path)
    if reason is not None:
        return reason
    for probe in _PROBES:
        if fast.encode(probe, add_special_tokens=False) != reference.encode(
            probe, add_special_tokens=False
        ):
            return f"encoding of {probe[:40]!r} differs"
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
        reason = _mismatch(fast, reference, path / "tekken.json")
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
        "(verified identical to tekken.json over %d token ids)",
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


# model_type -> (module, class) of the mlx-vlm processor for Pixtral-style models.
_PROCESSOR_CLASSES = {
    "mistral3": ("mlx_vlm.models.mistral3.processing_mistral3", "Mistral3Processor"),
    "pixtral": ("mlx_vlm.models.pixtral.processing_pixtral", "PixtralProcessor"),
}


def _processor_around(backend: Any, fast: Any, path: Path) -> Any | None:
    """A multimodal processor for ``fast``, for when ``mlx_vlm`` handed back the bare
    mistral-common backend as the processor (its Mistral3/Pixtral processor patch did
    not apply). Returns None when the processor cannot be rebuilt."""
    import importlib

    try:
        model_type = json.loads((path / "config.json").read_text()).get("model_type")
        module, name = _PROCESSOR_CLASSES[model_type]
        processor_class = getattr(importlib.import_module(module), name)
        settings = {}
        config_file = path / "processor_config.json"
        if config_file.is_file():
            loaded = json.loads(config_file.read_text())
            settings = {
                key: loaded[key]
                for key in (
                    "patch_size",
                    "spatial_merge_size",
                    "image_token",
                    "image_break_token",
                    "image_end_token",
                )
                if key in loaded
            }
        image_processor = getattr(backend, "image_processor", None)
        if image_processor is None:
            from transformers import AutoImageProcessor

            image_processor = AutoImageProcessor.from_pretrained(path)
        return processor_class(
            image_processor=image_processor, tokenizer=fast, **settings
        )
    except Exception as exc:
        logger.warning(
            "%s: cannot build a multimodal processor around the fast tokenizer "
            "(%s: %s)",
            path,
            type(exc).__name__,
            exc,
        )
        return None


def adopt_fast_tokenizer(
    processor: Any, model_path: str | Path, eos_token_ids: Any = None
) -> Any:
    """Return ``processor`` with a mistral-common tokenizer replaced by the fast one.

    ``mlx_vlm.load`` hands back a Mistral3/Pixtral processor whose ``.tokenizer`` is
    the mistral-common backend (its tokenizer is swapped in place), or, if mlx-vlm's
    processor patch did not apply, the backend itself as the processor (then a
    processor is built around the fast tokenizer). Either way a streaming
    detokenizer and stopping criteria are wired onto the result the way
    ``mlx_vlm.utils.load_processor`` does it; ``mlx_vlm.load`` gives up on that for
    the mistral-common tokenizer (it has no ``vocab``), leaving ``mlx_vlm.generate``
    without either. ``eos_token_ids`` is the model config's, as in ``mlx_vlm.load``.

    A processor that holds no mistral-common tokenizer is returned untouched. When it
    does and the replacement is impossible, a warning says why and the processor is
    returned as it was.
    """
    path = Path(model_path)
    if is_mistral_common_tokenizer(processor):
        reference, backend_is_processor = processor, True
    elif is_mistral_common_tokenizer(getattr(processor, "tokenizer", None)):
        reference, backend_is_processor = processor.tokenizer, False
    else:
        return processor

    fast = equivalent_fast_tokenizer(path, reference)
    if fast is None:
        return processor  # equivalent_fast_tokenizer said why
    if backend_is_processor:
        rebuilt = _processor_around(processor, fast, path)
        if rebuilt is None:
            return processor
        processor = rebuilt
    else:
        processor.tokenizer = fast

    from mlx_vlm.tokenizer_utils import load_tokenizer
    from mlx_vlm.utils import StoppingCriteria

    detokenizer_class = load_tokenizer(path, return_tokenizer=False)
    processor.detokenizer = detokenizer_class(fast)
    fast.stopping_criteria = StoppingCriteria(
        eos_token_ids,
        fast,
        additional_eos_token_ids=getattr(processor, "additional_eos_token_ids", ()),
    )
    return processor
