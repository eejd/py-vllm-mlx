# SPDX-License-Identifier: Apache-2.0
"""A stable, honest snapshot of prefix-cache state for ``GET /v1/cache/stats``.

Benchmarks and operators need to know, per run, what cache the engine actually had: which engine
mode, which cache options really apply, how its memory limit was derived, what it holds and has
hit, and what was persisted. Earlier the answers were scattered or silently missing; this module
puts them in one block (``cache_state``, with a ``schema_version``) and follows one rule: a value
the engine cannot report is the string ``"unreported"`` with a reason beside it, never a default
that looks like data.

Cache-related launch options are recorded by the CLI (:func:`launch_options`). Only some apply in
every engine mode: the memory-aware prefix cache, KV-cache quantization, paged cache and SSD tier
are part of the continuous-batching scheduler. The Simple engine ignores them, so they are
reported as *inert* rather than as active (py-vllm-mlx#37, ash#703).
"""

from __future__ import annotations

import importlib.metadata
from typing import Any

SCHEMA_VERSION = 1
UNREPORTED = "unreported"

# CLI flag -> (argparse attribute, parser default). Flags that only take effect with
# --continuous-batching. tests/test_cache_state.py checks the defaults against the real parser.
CONTINUOUS_BATCHING_ONLY: dict[str, tuple[str, Any]] = {
    "--disable-prefix-cache": ("disable_prefix_cache", False),
    "--cache-memory-mb": ("cache_memory_mb", None),
    "--cache-memory-percent": ("cache_memory_percent", 0.20),
    "--no-memory-aware-cache": ("no_memory_aware_cache", False),
    "--kv-cache-quantization": ("kv_cache_quantization", False),
    "--kv-cache-quantization-bits": ("kv_cache_quantization_bits", 8),
    "--kv-cache-quantization-group-size": ("kv_cache_quantization_group_size", 64),
    "--kv-cache-min-quantize-tokens": ("kv_cache_min_quantize_tokens", 256),
    "--use-paged-cache": ("use_paged_cache", False),
    "--ssd-cache-dir": ("ssd_cache_dir", None),
    "--ssd-cache-max-gb": ("ssd_cache_max_gb", 10.0),
    "--prefix-cache-size": ("prefix_cache_size", 100),
    "--paged-cache-block-size": ("paged_cache_block_size", 64),
    "--max-cache-blocks": ("max_cache_blocks", 1000),
    "--chunked-prefill-tokens": ("chunked_prefill_tokens", 0),
}


def inert_options(args: Any) -> list[str]:
    """Flags the user set to a non-default value that this engine mode ignores."""
    if getattr(args, "continuous_batching", False):
        return []
    return [
        flag
        for flag, (attr, default) in CONTINUOUS_BATCHING_ONLY.items()
        if getattr(args, attr, default) != default
    ]


def registry_ignores_persistence(args: Any) -> bool:
    """True when ``--models-config`` is combined with a non-default --prefix-cache-* option.

    Registry mode never loads or saves a persisted prefix cache, so such options do nothing there.
    """
    if not getattr(args, "models_config", None):
        return False
    return (
        getattr(args, "prefix_cache_dir", None) is not None
        or getattr(args, "prefix_cache_persist", "auto") != "auto"
        or getattr(args, "prefix_cache_reset", "never") != "never"
    )


def launch_options(args: Any) -> dict[str, Any]:
    """The cache-relevant options the server was started with (as requested)."""
    cb = bool(getattr(args, "continuous_batching", False))
    kv_requested = bool(getattr(args, "kv_cache_quantization", False))
    return {
        "continuous_batching": cb,
        "prefix_cache_requested": bool(
            getattr(args, "enable_prefix_cache", True)
            and not getattr(args, "disable_prefix_cache", False)
        ),
        "memory_aware_cache_requested": not getattr(args, "no_memory_aware_cache", False),
        "cache_memory_mb": getattr(args, "cache_memory_mb", None),
        "cache_memory_percent": getattr(args, "cache_memory_percent", None),
        "use_paged_cache": bool(getattr(args, "use_paged_cache", False)),
        "ssd_cache_dir": getattr(args, "ssd_cache_dir", None),
        "ssd_cache_max_gb": getattr(args, "ssd_cache_max_gb", None),
        "prefill_step_size": getattr(args, "prefill_step_size", None),
        "prefix_trie_cache": bool(getattr(args, "prefix_trie_cache", False)),
        "kv_cache_quantization": {
            "requested": kv_requested,
            "bits": getattr(args, "kv_cache_quantization_bits", None),
            "group_size": getattr(args, "kv_cache_quantization_group_size", None),
            "min_quantize_tokens": getattr(args, "kv_cache_min_quantize_tokens", None),
            # Only the batched scheduler's prefix cache quantizes stored KV.
            "effective": kv_requested and cb,
        },
        "inert_options": inert_options(args),
    }


def _dist_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return UNREPORTED


def versions() -> dict[str, str]:
    try:
        from . import __version__ as vllm_mlx_version
    except Exception:  # noqa: BLE001
        vllm_mlx_version = UNREPORTED
    return {
        "vllm_mlx": vllm_mlx_version,
        "mlx": _dist_version("mlx"),
        "mlx_lm": _dist_version("mlx-lm"),
    }


def _unreported(reason: str) -> dict[str, str]:
    return {"value": UNREPORTED, "reason": reason}


_COUNTER_KEYS = (
    "hits",
    "misses",
    "evictions",
    "tokens_saved",
    "discarded_hits",
    "entry_count",
    "current_memory_mb",
    "max_memory_mb",
)


def build(
    *,
    engine: Any,
    launch: dict[str, Any] | None,
    engine_cache: Any,
    persistence: dict[str, Any],
    registry_mode: bool,
    none_reason: str | None = None,
) -> dict[str, Any]:
    """Assemble the ``cache_state`` block for one engine.

    ``none_reason`` replaces the "no engine is loaded" reason when ``engine`` is None for a known
    cause (registry mode keeps no single default engine; its models are listed separately).
    """
    engine_class = type(engine).__name__ if engine is not None else None
    if registry_mode:
        # Registry entries choose their own engine; the CLI's --continuous-batching and cache
        # options describe the defaults, not necessarily any model. The engine class is the truth
        # (and with no default engine, as at the top level of a registry server, there is none).
        batching: Any = {"BatchedEngine": True, "SimpleEngine": False}.get(
            engine_class or "", UNREPORTED
        )
        launch_block: Any = _unreported(
            "registry mode: cache options come from each model's registry entry; see engine.class"
        )
        inert: Any = launch_block
    else:
        batching = bool(launch.get("continuous_batching")) if launch else UNREPORTED
        launch_block = launch if launch is not None else _unreported("server not started via the CLI")
        inert = (launch or {}).get("inert_options", UNREPORTED)

    # The batched MLLM engine nests its prefix-cache stats one level down.
    if isinstance(engine_cache, dict) and "hits" not in engine_cache:
        nested = engine_cache.get("prefix_cache")
        if isinstance(nested, dict) and "hits" in nested:
            engine_cache = nested

    counters: dict[str, Any] | dict[str, str]
    memory_limit: Any
    if isinstance(engine_cache, dict) and "error" not in engine_cache and "hits" in engine_cache:
        counters = {k: engine_cache[k] for k in _COUNTER_KEYS if k in engine_cache}
        memory_limit = engine_cache.get("memory_limit") or _unreported(
            "the cache did not record how its memory limit was derived"
        )
    else:
        why = (
            (none_reason or "no engine is loaded")
            if engine is None
            else "this engine mode exposes no prefix-cache counters"
        )
        counters = _unreported(why)
        memory_limit = _unreported(why)

    return {
        "schema_version": SCHEMA_VERSION,
        "versions": versions(),
        "engine": {
            "class": engine_class,
            "continuous_batching": batching,
            "registry_mode": registry_mode,
        },
        "launch_options": launch_block,
        "inert_options": inert,
        "memory_limit": memory_limit,
        "counters": counters,
        "persistence": persistence,
    }
