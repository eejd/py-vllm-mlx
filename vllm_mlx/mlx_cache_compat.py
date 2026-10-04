# SPDX-License-Identifier: Apache-2.0
"""Version-neutral access to prompt-cache layer state.

Two cache contracts coexist in one vllm-mlx process, and neither can be told
apart by looking at the mlx-lm version alone:

* **legacy** -- mlx-vlm's own ``mlx_vlm.models.cache`` classes, and mlx-lm up to
  0.31: ``meta_state`` exists, ``from_state(state, meta_state)`` takes two
  arguments, and ``KVCache.state`` is the offset-sliced ``(keys, values)``.
* **0.32** -- mlx-lm 0.32: no ``meta_state``, ``from_state(state)`` takes one
  argument, and ``KVCache.state`` is the full ``(keys, values, offset)`` where
  ``keys``/``values`` are the *padded* buffers (``keys.shape[2]`` is a
  multiple of 256, not the token count).

So every decision here is made per layer class, never per library version.
The canonical form that leaves this module is the legacy one -- a
``(state, meta_state)`` pair whose KV tensors are exactly ``offset`` tokens
long -- because the paged prefix cache slices and concatenates it along the
sequence axis. :func:`snapshot_state` and :func:`restore_from_state` are the
only places that translate.
"""

from __future__ import annotations

import copy
import functools
import inspect
from typing import Any

# Cache layers whose valid KV region is ``keys[..., :offset, :]`` and that can
# be rebuilt from a plain ``(keys, values)`` pair. Anything else (rotating,
# quantized, chunked, recurrent, nested) round-trips through its own
# ``state`` / ``from_state`` untouched.
_PLAIN_KV_NAMES = frozenset({"KVCache", "ConcatenateKVCache", "BatchKVCache"})

# Recurrent layers: canonical state is the bare list of arrays (what
# ``ArraysCache.state`` returned up to mlx-lm 0.31). In 0.32 ``.state`` is
# ``(cache, left_padding, lengths)`` and ``from_state`` wants that tuple back.
_RECURRENT_NAMES = frozenset({"ArraysCache", "MambaCache"})


@functools.lru_cache(maxsize=None)
def uses_meta_state(cls: type) -> bool:
    """True if ``cls`` follows the legacy contract (``meta_state`` property)."""
    return isinstance(getattr(cls, "meta_state", None), property)


@functools.lru_cache(maxsize=None)
def _from_state_arity(cls: type) -> int:
    """Number of arguments ``cls.from_state`` takes after ``cls`` (0 if none)."""
    fn = getattr(cls, "from_state", None)
    if not callable(fn):
        return 0
    try:
        params = list(inspect.signature(fn).parameters.values())
    except (TypeError, ValueError):
        return 0
    return sum(
        1
        for p in params
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        and p.default is p.empty
    )


def is_plain_kv(layer: Any) -> bool:
    """True for a single contiguous KV layer (not rotating/quantized/chunked)."""
    return type(layer).__name__ in _PLAIN_KV_NAMES


def kv_view(layer: Any) -> tuple[Any, Any] | None:
    """The valid ``(keys, values)`` of a plain KV layer, or None.

    "Valid" means exactly the cached tokens: the padded tail of the buffer
    that mlx-lm 0.32 exposes through ``.state`` is not included. Returns None
    for layers that are not a single contiguous KV region (rotating, recurrent,
    nested, quantized) and for batch sizes other than one.
    """
    if not is_plain_kv(layer):
        return None
    keys = getattr(layer, "keys", None)
    values = getattr(layer, "values", None)
    if keys is None or values is None or not hasattr(keys, "shape"):
        return None
    if keys.shape[0] != 1:
        return None
    kav = getattr(layer, "keys_and_values", None)
    if callable(kav):
        return kav()
    # Legacy contract: .state is already offset-sliced.
    state = layer.state
    if isinstance(state, (list, tuple)) and len(state) >= 2:
        return state[0], state[1]
    return None


def snapshot_state(layer: Any) -> tuple[Any, Any]:
    """Return the canonical ``(state, meta_state)`` of a cache layer.

    ``meta_state`` is ``None`` for classes that do not have one. For a plain KV
    layer the state is the exact-length ``(keys, values)`` pair with
    ``meta_state == (str(offset),)`` on every contract, so the paged prefix
    cache can slice it block by block.
    """
    cls = type(layer)
    view = kv_view(layer)
    if view is not None:
        return (view[0], view[1]), (str(view[0].shape[-2]),)
    if cls.__name__ in _RECURRENT_NAMES:
        arrays = recurrent_arrays(layer)
        if arrays is not None:
            return list(arrays), ""
    meta = layer.meta_state if uses_meta_state(cls) else None
    return layer.state, meta


def restore_from_state(cls: type, state: Any, meta_state: Any = None) -> Any:
    """Rebuild a cache layer of ``cls`` from :func:`snapshot_state` output.

    Raises ``TypeError`` when ``cls`` offers no way to take the state.
    """
    if cls.__name__ in _PLAIN_KV_NAMES and _is_kv_pair(state):
        obj = _new_plain_kv(cls)
        obj.keys, obj.values = state[0], state[1]
        obj.offset = state[0].shape[-2]
        return obj
    if cls.__name__ in _RECURRENT_NAMES and isinstance(state, list):
        obj = cls(len(state))
        set_recurrent_arrays(obj, state)
        return obj
    arity = _from_state_arity(cls)
    if arity >= 2:
        return cls.from_state(state, meta_state)
    if arity == 1:
        return cls.from_state(state)
    raise TypeError(f"{cls.__name__} has no from_state")


def _is_kv_pair(state: Any) -> bool:
    return (
        isinstance(state, (list, tuple))
        and len(state) == 2
        # (B, heads, seq, dim) -- or (heads, seq, dim) for Qwen3.5-style caches;
        # the sequence axis is second to last either way.
        and all(hasattr(t, "shape") and len(t.shape) in (3, 4) for t in state)
    )


def _new_plain_kv(cls: type) -> Any:
    """A ``KVCache`` for any plain-KV class (a batch of one is a KVCache)."""
    if cls.__name__ == "BatchKVCache":
        from mlx_lm.models.cache import KVCache

        return KVCache()
    return cls()


def recurrent_arrays(layer: Any) -> list[Any] | None:
    """The state arrays of a recurrent (``ArraysCache``-style) layer, or None.

    ``layer.cache`` is the list of arrays on every real contract. ``.state``
    is not: in mlx-lm 0.32 it is ``(cache, left_padding, lengths)``.
    """
    inner = getattr(layer, "cache", None)
    if isinstance(inner, list):
        return inner
    # Duck-typed legacy layer: ``.state`` is the bare list. A container
    # (``CacheList``) also has a list ``.state``, so it is excluded.
    if getattr(layer, "caches", None) is None:
        state = getattr(layer, "state", None)
        if isinstance(state, list):
            return state
    return None


def set_recurrent_arrays(layer: Any, arrays: list[Any]) -> None:
    """Replace the state arrays of a recurrent layer in place."""
    layer.cache = list(arrays)


def copy_state(dst: Any, src: Any) -> None:
    """Make ``dst`` hold the same state as ``src`` (same class).

    Nested containers are copied child by child. Assigning ``dst.state``
    on a ``CacheList`` rebuilds its children through ``from_state`` by class
    name, which raises ``KeyError`` for model-defined child classes.
    """
    children_src = getattr(src, "caches", None)
    if children_src is not None:
        for d, s in zip(dst.caches, children_src):
            copy_state(d, s)
        return
    if uses_meta_state(type(src)):
        dst.meta_state = src.meta_state
    dst.state = src.state


def prompt_cache_format() -> str:
    """``"meta"`` for the legacy ``save_prompt_cache`` file layout, else ``"state-v2"``.

    Persisted prompt caches written under one layout cannot be read under the
    other, so the format is recorded with them and mismatches are discarded.
    """
    try:
        from mlx_lm.models.cache import _BaseCache
    except ImportError:
        return "unknown"
    return "meta" if uses_meta_state(_BaseCache) else "state-v2"


def clone_layer(layer: Any, max_tokens: int | None = None) -> Any:
    """An independent copy of ``layer``, optionally cut to ``max_tokens``.

    A plain KV layer gets fresh ``keys``/``values`` cut to its valid tokens
    (and to ``max_tokens`` if smaller), so the clone never exposes the padded
    tail of an mlx-lm 0.32 buffer nor tokens generated after the prompt. Any
    other layer is rebuilt through its own state, which also gives a recurrent
    layer its own array list instead of sharing the original's.
    """
    view = kv_view(layer)
    if view is not None:
        import mlx.core as mx

        n = view[0].shape[-2]
        if max_tokens is not None:
            n = min(n, max_tokens)
        clone = copy.copy(layer)
        clone.keys = mx.array(view[0][..., :n, :])
        clone.values = mx.array(view[1][..., :n, :])
        clone.offset = n
        return clone
    state, meta = snapshot_state(layer)
    return restore_from_state(type(layer), state, meta)


def snapshot_for_rollback(layer: Any) -> tuple[str, list[Any]]:
    """Capture a layer that cannot be trimmed so it can be restored later.

    Speculative verification advances every layer by two tokens; layers that
    cannot ``trim`` (recurrent state, rotating windows) are put back from this
    snapshot instead. Recurrent layers are captured through their array list,
    because ``ArraysCache.state`` in mlx-lm 0.32 is ``(cache, left_padding,
    lengths)`` and its first element is a list, not an array.
    """
    import mlx.core as mx

    arrays = recurrent_arrays(layer)
    if arrays is not None:
        return "arrays", [mx.array(a) if a is not None else None for a in arrays]
    return "state", [mx.array(s) if s is not None else None for s in layer.state]


def restore_from_rollback(layer: Any, snapshot: tuple[str, list[Any]]) -> None:
    """Put back what :func:`snapshot_for_rollback` captured."""
    kind, data = snapshot
    if kind == "arrays":
        set_recurrent_arrays(layer, data)
    else:
        layer.state = data
