# SPDX-License-Identifier: Apache-2.0
"""
Tokenizer utilities with fallback support for non-standard tokenizers.

Some models (e.g., Nemotron) use non-standard tokenizer configurations
that transformers doesn't recognize. This module provides fallback loading
directly from tokenizer.json.
"""

import copy
import inspect
import json
import logging
from pathlib import Path

from .chat_template_override import apply_chat_template_override
from .chat_templates import DEFAULT_CHATML_TEMPLATE, NEMOTRON_CHAT_TEMPLATE

logger = logging.getLogger(__name__)

# Models that require tokenizer fallback
FALLBACK_MODELS = [
    "nemotron",
    "NVIDIA-Nemotron",
]


def prompt_adds_special_tokens(tokenizer, prompt: str) -> bool:
    """Whether ``tokenizer.encode(prompt)`` should add its own special tokens.

    A prompt that was already rendered by the chat template carries the BOS
    as literal text (Llama-3 ``<|begin_of_text|>``, Gemma ``<bos>``, ...).
    Tokenizing it with the tokenizer's default ``add_special_tokens=True``
    prepends a *second* BOS the template never asked for -- the model then sees
    ``[BOS, BOS, ...]`` instead of the ``[BOS, ...]`` that
    ``apply_chat_template(tokenize=True)`` (and upstream vLLM, whose chat
    tokenization uses ``add_special_tokens=False``) produce.

    This is the rule ``mlx_lm.generate.stream_generate`` applies to string
    prompts, character for character: skip the tokenizer's special tokens iff
    ``bos_token`` is set and the prompt starts with it (no stripping, so leading
    whitespace defeats it, and an empty ``bos_token`` matches every prompt).  A
    raw completion prompt without a leading BOS text still gets one from the
    tokenizer.  Every tokenization of a prompt string in the engines goes
    through this rule (or :func:`encode_prompt`) so counts, cache keys and
    prefix boundaries agree.
    """
    bos_token = getattr(tokenizer, "bos_token", None)
    return not (isinstance(bos_token, str) and prompt.startswith(bos_token))


def resolve_add_special_tokens(
    tokenizer, prompt: str, add_special_tokens: bool | None = None
) -> bool:
    """The ``add_special_tokens`` value for a prompt of the given request kind.

    ``add_special_tokens`` is the request kind, as in vLLM's tokenize params:

    * ``False`` -- a prompt the chat template already rendered (the template
      owns every marker, BOS included).  vLLM tokenizes chat prompts with
      ``add_special_tokens=False`` (``vllm/renderers/base.py``
      ``default_chat_tok_params``), so nothing is added.
    * ``None`` -- a raw completion prompt: the tokenizer's default (BOS added),
      except that a prompt which already starts with the BOS text keeps that
      single BOS (:func:`prompt_adds_special_tokens`).
    * ``True`` -- always the tokenizer's default.
    """
    if add_special_tokens is not None:
        return add_special_tokens
    return prompt_adds_special_tokens(tokenizer, prompt)


def _encode_takes_add_special_tokens(encode) -> bool:
    try:
        parameters = inspect.signature(encode).parameters.values()
    except (TypeError, ValueError):
        return True
    return any(
        p.name == "add_special_tokens" or p.kind is inspect.Parameter.VAR_KEYWORD
        for p in parameters
    )


def encode_prompt(
    tokenizer, prompt: str, add_special_tokens: bool | None = None
) -> list[int]:
    """Tokenize ``prompt``; see :func:`resolve_add_special_tokens` for the kinds.

    A duck-typed tokenizer whose ``encode(text)`` takes no ``add_special_tokens``
    keyword is encoded with its own defaults.
    """
    if not _encode_takes_add_special_tokens(tokenizer.encode):
        return tokenizer.encode(prompt)
    return tokenizer.encode(
        prompt,
        add_special_tokens=resolve_add_special_tokens(
            tokenizer, prompt, add_special_tokens
        ),
    )


def _needs_tokenizer_fallback(model_name: str) -> bool:
    """Check if model needs tokenizer fallback."""
    model_lower = model_name.lower()
    return any(pattern.lower() in model_lower for pattern in FALLBACK_MODELS)


def collect_eos_token_ids(tokenizer, model_path=None) -> set:
    """Collect the full EOS/stop token-id set for a model.

    Unions every place an EOS id can be declared:
    - ``tokenizer.eos_token_id`` (int or list)
    - ``tokenizer.eos_token_ids`` (mlx_lm TokenizerWrapper attribute)
    - ``config.json`` / ``generation_config.json`` ``eos_token_id``

    Chat models routinely declare turn terminators only in the config EOS
    list, not as the tokenizer's single ``eos_token`` (e.g. Gemma 4 ends
    chat turns with ``<turn|>``=106 and tool responses with
    ``<|tool_response>``=50 while ``tokenizer.eos_token`` stays ``<eos>``=1).
    Generating with only ``{tokenizer.eos_token_id}`` makes such models run
    straight through end-of-turn until max_tokens.

    Args:
        tokenizer: HF tokenizer (or processor.tokenizer) to read attributes
            from.
        model_path: Model directory containing the config files. Falls back
            to ``tokenizer.name_or_path`` when omitted.

    Returns:
        Set of EOS token ids; may be empty if nothing is declared anywhere.
    """
    eos_ids = set()

    for attr in ("eos_token_id", "eos_token_ids"):
        value = getattr(tokenizer, attr, None)
        if value is None:
            continue
        if isinstance(value, (list, set, tuple)):
            # mlx-lm 0.31 reports {None} for a tokenizer without an EOS id.
            eos_ids.update(int(v) for v in value if v is not None)
        else:
            eos_ids.add(int(value))

    if model_path is None:
        model_path = getattr(tokenizer, "name_or_path", None)
    if model_path:
        for config_name in ("config.json", "generation_config.json"):
            config_path = Path(model_path) / config_name
            if not config_path.exists():
                continue
            try:
                config_eos = json.loads(config_path.read_text()).get("eos_token_id")
            except (OSError, json.JSONDecodeError, UnicodeDecodeError):
                continue
            if isinstance(config_eos, list):
                eos_ids.update(int(v) for v in config_eos if v is not None)
            elif config_eos is not None:
                eos_ids.add(int(config_eos))

    return eos_ids


_BOUNDARY_IDS_ATTR = "_vllm_mlx_boundary_token_ids"


def boundary_token_ids(tokenizer) -> frozenset:
    """Ids of the tokens that delimit a sequence and are never text.

    The EOS ids (``collect_eos_token_ids``) plus the tokenizer's BOS, PAD and UNK.
    vLLM renders none of them by default (``skip_special_tokens=True``). A
    base-style model that ends a document with ``<|end_of_text|>`` where the
    checkpoint declares another EOS (Hermes-3 on Llama-3.2: EOS ``<|im_end|>``,
    PAD ``<|end_of_text|>``) goes on generating ``<|begin_of_text|>`` and the next
    document, and mlx-lm's detokenizers would put both markers into the text.

    The other special tokens (``<|python_tag|>``, ``<|channel|>``, ``[TOOL_CALLS]``,
    ...) are not here: the tool-call and reasoning parsers read them out of the
    text. vLLM hides them too by default but turns that off under a tool parser
    (``adjust_request``); this server's parsers work on text, so they stay.

    The set is read from the config files once per tokenizer object.
    """
    cached = getattr(tokenizer, "__dict__", {}).get(_BOUNDARY_IDS_ATTR)
    if cached is not None:
        return cached
    hidden = set(collect_eos_token_ids(tokenizer))
    for attr in ("bos_token_id", "pad_token_id", "unk_token_id"):
        token_id = getattr(tokenizer, attr, None)
        if isinstance(token_id, int):
            hidden.add(token_id)
    hidden = frozenset(hidden)
    try:
        tokenizer.__dict__[_BOUNDARY_IDS_ATTR] = hidden
    except AttributeError:
        pass
    return hidden


def _visible_ids(tokenizer, skip_special_tokens: bool) -> frozenset:
    """The ids a request leaves out of its text (none for ``skip_special_tokens=False``)."""
    return boundary_token_ids(tokenizer) if skip_special_tokens else frozenset()


class _TokenSkippingDetokenizer:
    """Streaming detokenizer that drops some token ids instead of rendering them.

    mlx-lm's ``stream_generate`` never feeds EOS to the detokenizer because it
    stops there. When generation runs past EOS (``ignore_eos``), or past a
    boundary token the checkpoint does not declare as EOS, those tokens would
    otherwise be decoded into the text (``<|im_end|>``, ``<|end_of_text|>`` and
    friends). vLLM hides them (``skip_special_tokens`` defaults to true), so they
    are skipped here; they are still generated and counted.
    """

    def __init__(self, inner, skip_ids):
        self._inner = inner
        self._skip_ids = skip_ids

    def __copy__(self):
        # TokenizerWrapper.detokenizer hands each caller a copy; the wrapped
        # detokenizer must be copied too or concurrent callers share its state.
        return type(self)(copy.copy(self._inner), self._skip_ids)

    def add_token(self, token, *args, **kwargs):
        if token in self._skip_ids:
            return
        self._inner.add_token(token, *args, **kwargs)

    def __getattr__(self, name):
        if name.startswith("__") or name in ("_inner", "_skip_ids"):
            raise AttributeError(name)
        return getattr(self._inner, name)


class _SkippingDetokenizerClass:
    """``detokenizer_class`` of an mlx-lm 0.31 ``TokenizerWrapper`` that skips ids.

    0.31 builds a fresh detokenizer on every ``tokenizer.detokenizer`` access from
    ``self._detokenizer_class(self)``; 0.32 keeps one instance and copies it.
    """

    def __init__(self, inner_class, skip_ids):
        self.inner_class = inner_class
        self.skip_ids = skip_ids

    def __call__(self, tokenizer):
        return _TokenSkippingDetokenizer(self.inner_class(tokenizer), self.skip_ids)


def _set_detokenizer_skip(wrapper, skip_ids: frozenset) -> bool:
    """Make ``wrapper``'s detokenizer skip exactly ``skip_ids`` (none: render all).

    Works on both shapes of mlx-lm's ``TokenizerWrapper`` (a detokenizer instance
    in ``_detokenizer``, 0.32; a class in ``_detokenizer_class``, 0.31), replacing
    any skip filter set before. Only the wrapper's own attribute is rebound, so a
    view that shares its base's state is changed without touching the base.
    Returns False, leaving the wrapper alone, for a shape it does not know.
    """
    state = wrapper.__dict__
    if "_detokenizer" in state:
        inner = state["_detokenizer"]
        while isinstance(inner, _TokenSkippingDetokenizer):
            inner = inner._inner
        state["_detokenizer"] = (
            _TokenSkippingDetokenizer(inner, skip_ids) if skip_ids else inner
        )
        return True
    if "_detokenizer_class" in state:
        inner = state["_detokenizer_class"]
        while isinstance(inner, _SkippingDetokenizerClass):
            inner = inner.inner_class
        state["_detokenizer_class"] = (
            _SkippingDetokenizerClass(inner, skip_ids) if skip_ids else inner
        )
        return True
    return False


def _as_wrapper(tokenizer):
    from mlx_lm.tokenizer_utils import TokenizerWrapper

    if isinstance(tokenizer, TokenizerWrapper):
        return tokenizer
    return TokenizerWrapper(tokenizer)


def hide_boundary_tokens(tokenizer):
    """``tokenizer`` as an mlx-lm ``TokenizerWrapper`` that never renders
    ``boundary_token_ids`` (see there).

    Every text path that detokenizes through the tokenizer (``mlx_lm.generate``
    and ``stream_generate``, with or without a prompt cache) picks it up. The
    wrapper is modified in place, so call it once on the tokenizer the engine
    owns; a second call is a no-op. A mlx-lm whose wrapper has neither shape of
    ``_set_detokenizer_skip`` is left as it is, with a warning: the text then
    shows the markers, as before, instead of the model failing to load.
    """
    tokenizer = _as_wrapper(tokenizer)
    hidden = boundary_token_ids(tokenizer)
    if hidden and not _set_detokenizer_skip(tokenizer, hidden):
        logger.warning(
            "mlx-lm TokenizerWrapper has an unknown detokenizer layout; generated "
            "BOS/PAD/UNK/EOS tokens will be rendered into the text"
        )
    return tokenizer


def streaming_detokenizer(tokenizer, hidden_ids=None):
    """A fresh per-request streaming detokenizer for ``tokenizer`` that skips
    ``hidden_ids`` (default ``boundary_token_ids(tokenizer)``; empty: renders all).

    For the batching schedulers, which feed the tokens of many requests through
    detokenizers they own instead of ``tokenizer.detokenizer``.
    """
    from mlx_lm.tokenizer_utils import NaiveStreamingDetokenizer

    if hidden_ids is None:
        hidden_ids = boundary_token_ids(tokenizer)
    detokenizer = NaiveStreamingDetokenizer(tokenizer)
    return (
        _TokenSkippingDetokenizer(detokenizer, hidden_ids)
        if hidden_ids
        else detokenizer
    )


def decode_visible(tokenizer, token_ids, skip_special_tokens: bool = True) -> str:
    """``tokenizer.decode(token_ids)`` without the ``boundary_token_ids``
    (unless ``skip_special_tokens`` is False)."""
    hidden = _visible_ids(tokenizer, skip_special_tokens)
    return tokenizer.decode([t for t in token_ids if t not in hidden])


def _tokenizer_view(tokenizer, *, eos_token_ids, skip_ids):
    """A ``TokenizerWrapper`` over ``tokenizer``'s state with its own EOS set and
    detokenizer skip ids; ``tokenizer`` is not modified.

    The result is cheap to build per request (no detokenizer map rebuild).
    """
    from mlx_lm.tokenizer_utils import TokenizerWrapper

    class _View(TokenizerWrapper):
        def __init__(self, base):
            # Share base's state instead of re-running TokenizerWrapper.__init__,
            # which rebuilds the detokenizer token map.
            self.__dict__.update(base.__dict__)
            self._eos_token_ids = set(eos_token_ids)
            _set_detokenizer_skip(self, skip_ids)

    return _View(tokenizer)


def eos_ignoring_tokenizer(tokenizer, skip_special_tokens: bool = True):
    """A view of ``tokenizer`` under which mlx-lm does not stop on EOS.

    Implements ``ignore_eos`` for every code path that drives
    ``mlx_lm.stream_generate``/``generate``: those stop when a token is in
    ``tokenizer.eos_token_ids``, so the view reports an empty set. Its
    detokenizer skips the former EOS ids (and the other boundary tokens) so they
    stay out of the text, unless the request set ``skip_special_tokens=False``.
    """
    base = _as_wrapper(tokenizer)
    skip = (
        frozenset(base.eos_token_ids) | boundary_token_ids(base)
        if skip_special_tokens
        else frozenset()
    )
    return _tokenizer_view(base, eos_token_ids=(), skip_ids=skip)


def unfiltered_tokenizer(tokenizer):
    """A view of ``tokenizer`` that renders every token it is fed
    (``skip_special_tokens=False``); it still stops on EOS."""
    base = _as_wrapper(tokenizer)
    return _tokenizer_view(base, eos_token_ids=base.eos_token_ids, skip_ids=frozenset())


def _needs_strict_false(model_name: str) -> bool:
    """Check if model needs strict=False loading (VLM models with extra weights).

    VLM models (e.g., Qwen3.5) have vision_tower weights that don't match
    the text-only model class.  Loading with strict=True fails and wastes
    memory by loading all weights (~100 GB) before raising ValueError.
    Detect these models up-front to avoid the double-load penalty.
    """
    from mlx_lm.utils import _download, load_config

    try:
        model_path = _download(model_name)
        config = load_config(model_path)
    except Exception:
        return False
    # VLM models have vision_config or text_config with a separate model_type
    if "vision_config" in config and "text_config" in config:
        return True
    return False


def _model_type(model_name: str) -> str:
    """Read model_type from config.json, empty string if unavailable."""
    from mlx_lm.utils import _download, load_config

    try:
        config = load_config(_download(model_name))
    except Exception:
        return ""
    return config.get("text_config", config).get("model_type", "") or ""


def _install_custom_chat_template(model_name: str, tokenizer):
    """Give models without a Jinja chat template a programmatic encoder.

    DeepSeek-V4 ships no ``chat_template``, so the stock path would either raise
    or fall back to naive ``"role: content"`` concatenation. Patching the
    tokenizer covers every caller at once.
    """
    if tokenizer is None or _model_type(model_name) != "deepseek_v4":
        return tokenizer

    from .deepseek_v4_encoding import install as install_deepseek_v4

    return install_deepseek_v4(tokenizer, model_name=model_name)


def _prefer_fast_mistral_tokenizer(model_name: str, tokenizer_config: dict | None):
    """Have mlx-lm load the tokenizer.json tokenizer of Mistral tekken checkpoints.

    See ``mistral_tokenizer``: the mistral-common tokenizer ``transformers`` picks
    by default cannot read the prompt string this server renders.
    """
    from mlx_lm.utils import _download

    from .mistral_tokenizer import fast_tokenizer_config

    try:
        model_path = _download(model_name)
    except Exception:
        return tokenizer_config  # the load that follows reports the real error
    return fast_tokenizer_config(model_path, tokenizer_config)


def _drop_mistral_default_system_prompt(model_name: str, tokenizer) -> None:
    """Render Mistral tekken checkpoints' prompts as mistral-common does.

    See ``mistral_tokenizer.drop_default_system_prompt``. Only a template with a
    ``default_system_message`` makes this locate the checkpoint's files.
    """
    from .mistral_tokenizer import (
        drop_default_system_prompt,
        mentions_default_system_message,
    )

    if not mentions_default_system_message(tokenizer):
        return

    from mlx_lm.utils import _download

    try:
        model_path = _download(model_name)
    except Exception:
        return  # the load that follows reports the real error
    drop_default_system_prompt(tokenizer, model_path)


def load_model_with_fallback(
    model_name: str,
    tokenizer_config: dict = None,
    chat_template: str | None = None,
):
    """
    Load model and tokenizer with fallback for non-standard tokenizers.

    Args:
        model_name: HuggingFace model name or local path
        tokenizer_config: Optional tokenizer configuration
        chat_template: Optional template text (see ``--chat-template``) that replaces
            the one the model ships with

    Returns:
        Tuple of (model, tokenizer)
    """
    tokenizer_config = _prefer_fast_mistral_tokenizer(model_name, tokenizer_config)
    model, tokenizer = _load_model_with_fallback(model_name, tokenizer_config)
    tokenizer = _install_custom_chat_template(model_name, tokenizer)
    _drop_mistral_default_system_prompt(model_name, tokenizer)
    apply_chat_template_override(tokenizer, chat_template)
    tokenizer = hide_boundary_tokens(tokenizer)
    return model, tokenizer


def _load_model_with_fallback(model_name: str, tokenizer_config: dict = None):
    from mlx_lm import load

    tokenizer_config = tokenizer_config or {}

    # Check if model needs fallback (e.g., Nemotron)
    if _needs_tokenizer_fallback(model_name):
        logger.info(
            f"Model {model_name} requires tokenizer fallback, loading directly..."
        )
        return _load_with_tokenizer_fallback(model_name)

    # VLM models (e.g., Qwen3.5) have extra vision weights that cause
    # strict=True to fail.  Skip the first load attempt to avoid loading
    # ~100 GB of weights twice (which can cause OOM on 256 GB systems).
    if _needs_strict_false(model_name):
        logger.info(
            f"Model {model_name} detected as VLM, loading directly with strict=False"
        )
        return _load_strict_false(model_name, tokenizer_config)

    try:
        model, tokenizer = load(model_name, tokenizer_config=tokenizer_config)
    except ValueError as e:
        # Fallback for models with non-standard tokenizers
        if "TokenizersBackend" in str(e) or "Tokenizer class" in str(e):
            logger.warning(f"Standard tokenizer loading failed, using fallback: {e}")
            return _load_with_tokenizer_fallback(model_name)
        # Fallback for models with extra weights (e.g., vision tower, MTP layers).
        # Retry with strict=False to discard extra weights.
        elif "parameters not in model" in str(e):
            logger.warning(
                f"Extra parameters found (e.g., vision tower / MTP weights), "
                f"retrying with strict=False: {e}"
            )
            # Clear traceback references to free memory from the failed first load.
            # Without this, large models (200GB+) cause OOM during retry because
            # the traceback holds references to the first load's weight tensors.
            e.__traceback__ = None
            del e
            import gc

            gc.collect()
            return _load_strict_false(model_name, tokenizer_config)
        else:
            raise

    # After successful load, check if MTP weights exist but were stripped by sanitize()
    _try_inject_mtp_post_load(model, model_name)
    return model, tokenizer


def _load_strict_false(model_name: str, tokenizer_config: dict = None):
    """Load model with strict=False to discard extra weights.

    Handles models with extra parameters that the text-only model class
    doesn't define (e.g., vision tower weights in VLM models like Qwen3.5,
    or MTP layers).  The model's own sanitize() handles key remapping
    (e.g., language_model.* prefix), and strict=False silently drops
    unmatched keys.
    """
    import mlx.core as mx
    from mlx_lm.utils import _download, load_model, load_tokenizer

    model_path = _download(model_name)
    model, config = load_model(model_path, strict=False)

    # Verify weights loaded correctly
    from mlx.utils import tree_flatten

    params = tree_flatten(model.parameters())
    total_params = len(params)
    zero_params = sum(1 for _, v in params if mx.all(v == 0).item())
    logger.info(
        f"[strict=False] Loaded {total_params} parameters, "
        f"{zero_params} all-zero tensors"
    )
    # Spot-check embedding weights
    if hasattr(model, "language_model"):
        emb = model.language_model.model.embed_tokens.weight
        logger.info(
            f"[strict=False] embed_tokens: shape={emb.shape}, "
            f"dtype={emb.dtype}, mean={mx.mean(emb.astype(mx.float32)).item():.4f}"
        )

    tokenizer = load_tokenizer(
        model_path,
        tokenizer_config or {},
        eos_token_ids=config.get("eos_token_id", None),
    )
    _try_inject_mtp(model, model_path, config)
    return model, tokenizer


def _try_inject_mtp(model, model_path, config):
    """Inject MTP support if model has MTP config + weights."""
    # Qwen3-Next: flat num_nextn_predict_layers
    if config.get("num_nextn_predict_layers", 0) > 0:
        # Detect Qwen3.5 vs Qwen3-Next by checking text_config or model_type
        text_config = config.get("text_config", config)
        model_type = text_config.get("model_type", config.get("model_type", ""))
        if "qwen3_5" in model_type:
            from ..patches.qwen3_5_mtp import inject_mtp_support
        else:
            from ..patches.qwen3_next_mtp import inject_mtp_support
        inject_mtp_support(model, model_path, config)
        return

    # Qwen3.5: mtp_num_hidden_layers in text_config
    text_config = config.get("text_config", config)
    num_mtp = text_config.get("mtp_num_hidden_layers", 0)
    if num_mtp > 0:
        from ..patches.qwen3_5_mtp import inject_mtp_support

        inject_mtp_support(model, model_path, config)


def _try_inject_mtp_post_load(model, model_name):
    """Check if MTP weights exist but were stripped by sanitize(), and inject."""
    import json

    from mlx_lm.utils import _download

    model_path = _download(model_name)
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        return
    with open(config_path) as f:
        config = json.load(f)
    # Check for MTP in flat config and nested text_config
    text_config = config.get("text_config", {})
    num_mtp = config.get("num_nextn_predict_layers", 0)
    if num_mtp == 0:
        num_mtp = text_config.get("num_nextn_predict_layers", 0)
    if num_mtp == 0:
        num_mtp = text_config.get("mtp_num_hidden_layers", 0)
    # Also check mtp attribute on language_model for VLM wrappers
    check_model = model
    if hasattr(model, "language_model"):
        check_model = model.language_model
    if num_mtp > 0 and getattr(check_model, "mtp", None) is None:
        mtp_file = Path(model_path) / "mtp" / "weights.safetensors"
        if not mtp_file.exists():
            mtp_file = Path(model_path) / "model-mtp.safetensors"
        if mtp_file.exists():
            logger.info(
                f"[MTP] Found MTP config (layers={num_mtp}) and weights, injecting..."
            )
            _try_inject_mtp(model, model_path, config)
        else:
            logger.info(
                f"[MTP] Config has num_nextn_predict_layers={num_mtp} "
                "but MTP weights not found, skipping MTP."
            )


def _load_with_tokenizer_fallback(model_name: str):
    """Load model with fallback tokenizer for non-standard models like Nemotron."""
    from mlx_lm.utils import load_model

    from .download import ensure_model_downloaded

    logger.info("Loading with tokenizer fallback...")

    # Get model path (with retry/timeout support)
    model_path = ensure_model_downloaded(model_name, is_mllm=False)

    # Load model
    model, _ = load_model(model_path)

    # Try to load tokenizer from tokenizer.json directly
    tokenizer_json = model_path / "tokenizer.json"
    if tokenizer_json.exists():
        from tokenizers import Tokenizer
        from transformers import PreTrainedTokenizerFast

        logger.info("Loading tokenizer from tokenizer.json")
        base_tokenizer = Tokenizer.from_file(str(tokenizer_json))

        # Read tokenizer_config.json for special tokens and chat template
        tokenizer_config_path = model_path / "tokenizer_config.json"
        bos_token = "<s>"
        eos_token = "</s>"
        unk_token = "<unk>"
        chat_template = None

        if tokenizer_config_path.exists():
            with open(tokenizer_config_path) as f:
                config = json.load(f)
                bos_token = config.get("bos_token", bos_token)
                eos_token = config.get("eos_token", eos_token)
                unk_token = config.get("unk_token", unk_token)
                chat_template = config.get("chat_template")

        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=base_tokenizer,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token="<pad>",
        )

        # Set chat template if available
        if chat_template:
            tokenizer.chat_template = chat_template
            logger.info("Chat template loaded from tokenizer_config.json")
        elif _needs_tokenizer_fallback(model_name):
            # Use official Nemotron chat template with thinking support
            tokenizer.chat_template = NEMOTRON_CHAT_TEMPLATE
            logger.info("Using official Nemotron chat template with thinking support")
        else:
            # Default simple ChatML format for other models
            tokenizer.chat_template = DEFAULT_CHATML_TEMPLATE
            logger.info("Using default ChatML chat template")

        logger.info("Tokenizer loaded via fallback successfully")
        return model, tokenizer
    else:
        raise ValueError(f"No tokenizer.json found in {model_path}")
