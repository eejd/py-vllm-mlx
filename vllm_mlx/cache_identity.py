# SPDX-License-Identifier: Apache-2.0
"""What a stored KV entry depends on, as a string that two runs can compare.

A prefix-cache entry is keyed by token ids, but the tensors behind a key are a
function of much more than that: the model weights, every config field that
reaches attention (rope theta and scaling, sliding window, head layout, norm
epsilon, tied embeddings, ...), the dtype, the quantization of the weights, how
the KV itself is quantized when held, the ``--max-kv-size`` window, and the
mlx / mlx-lm code that computed it. An entry that outlives the process (the
persisted prefix cache, the SSD tier) is only valid for a run that agrees on
all of it, so each such store carries one :class:`CacheIdentity` and refuses
entries written under another.

What is deliberately *not* part of it: the tokenizer and chat template. Keys are
token ids, and the KV for a token id sequence does not depend on how text maps
to those ids.

Weights are identified by content, not by path or modification time: a
different fine-tune of one architecture, or a new revision downloaded over the
same path, has the same file names and the same tensor shapes. Each parameter
contributes two wrapping integer sums over its raw bits (the sum and the sum of
squares). The reduction runs once over every weight on the device, at the speed
of one pass over the weights, and is exact and independent of reduction order,
so the digest is reproducible. It is a change detector for accidentally
different checkpoints, not a cryptographic hash.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Bumped when the identity inputs or their encoding change, so identities
# computed by older code never compare equal to current ones.
IDENTITY_VERSION = 1

# Per-process token used when the weights cannot be read: an identity that can
# never match a later process (fail closed), yet is stable inside this one.
_UNREADABLE_NONCE = uuid.uuid4().hex

# Upper bound on the bytes one device pass materializes while checksumming.
_CHECKSUM_CHUNK_BYTES = 512 * 1024 * 1024

# Elements of one parameter summed in a single device pass.
_SLICE_ELEMENTS = 1 << 25

# Config attributes that vary per run without changing what the model computes.
_NON_KV_CONFIG_KEYS = frozenset({"_name_or_path", "name_or_path", "model_path"})


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=lambda o: f"<{type(o).__module__}.{type(o).__qualname__}>",
    )


def _sha(*parts: str, length: int = 32) -> str:
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()[:length]


def _config_object(model: Any) -> Any:
    for attr in ("config", "args", "model_config"):
        cfg = getattr(model, attr, None)
        if cfg is not None:
            return cfg
    return model


def config_fields(model: Any) -> dict[str, Any]:
    """Every field of the model's config, as plain data.

    The whole config is used rather than an allow-list: any field the model
    reads (rope scaling, sliding window, attention layout, ...) changes the KV,
    and an allow-list silently stops covering a field the next architecture
    adds.
    """
    cfg = _config_object(model)
    if dataclasses.is_dataclass(cfg) and not isinstance(cfg, type):
        fields = dataclasses.asdict(cfg)
    elif isinstance(cfg, Mapping):
        fields = dict(cfg)
    elif hasattr(cfg, "__dict__"):
        fields = {k: v for k, v in vars(cfg).items() if not k.startswith("_")}
    else:
        fields = {}
    return {k: v for k, v in fields.items() if k not in _NON_KV_CONFIG_KEYS}


def _int_view_dtype(mx: Any, itemsize: int) -> Any:
    return {1: mx.uint8, 2: mx.uint16}.get(itemsize, mx.uint32)


def weights_digest(model: Any) -> str | None:
    """Content digest of every parameter of ``model``, or None if unreadable.

    Names, shapes and dtypes are part of it, so a different dtype of the same
    values differs as well.
    """
    try:
        import mlx.core as mx
        from mlx.utils import tree_flatten

        leaves = [
            (name, arr)
            for name, arr in tree_flatten(model.parameters())
            if isinstance(arr, mx.array)
        ]
        if not leaves:
            raise ValueError("the model exposes no parameter arrays")
        leaves.sort(key=lambda item: item[0])

        # (leaf index, pair of wrapping sums) for every slice of every parameter
        partial: list[tuple[int, mx.array]] = []
        pending: list[mx.array] = []
        pending_bytes = 0

        for index, (_, arr) in enumerate(leaves):
            flat = arr.reshape(-1)
            view_dtype = _int_view_dtype(mx, arr.dtype.size)
            # A big tensor (an embedding is often more than a gigabyte) is summed
            # in slices so the widened copy and its square stay bounded; the
            # sums wrap, so slicing does not change them.
            for start in range(0, max(arr.size, 1), _SLICE_ELEMENTS):
                piece = flat[start : start + _SLICE_ELEMENTS]
                if piece.size == 0:
                    bits = mx.zeros((1,), dtype=mx.uint32)
                else:
                    bits = piece.view(view_dtype).astype(mx.uint32)
                pair = mx.stack([mx.sum(bits), mx.sum(bits * bits)])
                partial.append((index, pair))
                pending.append(pair)
                # the widened copy and its square, 4 bytes each per element
                pending_bytes += piece.size * 8
                if pending_bytes >= _CHECKSUM_CHUNK_BYTES:
                    mx.eval(*pending)
                    pending, pending_bytes = [], 0
        if pending:
            mx.eval(*pending)

        totals = [[0, 0] for _ in leaves]
        for index, pair in partial:
            s1, s2 = pair.tolist()
            totals[index][0] = (totals[index][0] + s1) % (1 << 32)
            totals[index][1] = (totals[index][1] + s2) % (1 << 32)

        h = hashlib.sha256()
        for (name, arr), (s1, s2) in zip(leaves, totals):
            h.update(f"{name}|{arr.dtype}|{tuple(arr.shape)}|{s1}|{s2}\n".encode())
        return h.hexdigest()[:32]
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[cache_identity] cannot read the model's weights (%s: %s); persisted and "
            "SSD-tier cache entries of this run will not be reusable by another run",
            type(exc).__name__,
            exc,
        )
        return None


def runtime_versions(model: Any) -> dict[str, str]:
    """Versions of the code that computes the KV: mlx, mlx-lm, and the package
    that defines the model class (mlx-vlm for a VLM language model)."""
    from importlib import metadata

    packages = ["mlx", "mlx-lm"]
    if type(model).__module__.split(".")[0] == "mlx_vlm":
        packages.append("mlx-vlm")
    versions: dict[str, str] = {}
    for pkg in packages:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = "unavailable"
    return versions


def model_class_path(model: Any) -> str:
    cls = type(model)
    return f"{cls.__module__}.{cls.__qualname__}"


@dataclass(frozen=True)
class CacheIdentity:
    """Everything a stored KV entry is valid for.

    Three parts, kept apart so a mismatch can say which one changed and so a
    store can apply a different rule to each:

    * ``model``: the checkpoint and the software that computed the KV. Always
      required to match.
    * ``window``: the ``--max-kv-size`` the KV was computed under (a bounded
      run keeps rotating layers, an unbounded one does not). Always required to
      match.
    * ``kv``: how this run quantizes the KV it holds. Entries written while held
      quantized carry that rounding, so they are only valid under the same
      setting; entries held exactly are valid under any (they are quantized on
      the way in, like a fresh one).
    """

    model: str
    window: str
    kv: str
    detail: Mapping[str, Any] = field(default_factory=dict, compare=False, hash=False)

    @property
    def digest(self) -> str:
        """All three parts in one name, for stores that cannot mix settings."""
        return _sha(str(IDENTITY_VERSION), self.model, self.window, self.kv)

    def mismatch(self, other_model: str | None, other_window: str | None) -> list[str]:
        """Which always-required parts of a recorded identity differ from this one."""
        parts = []
        if other_model != self.model:
            parts.append("model (weights, config, quantization, mlx/mlx-lm versions)")
        if other_window != self.window:
            parts.append("max-kv-size")
        return parts


def window_identity(max_kv_size: int) -> tuple[str, dict[str, Any]]:
    """The ``--max-kv-size`` window as ``(digest, description)``; 0 = unbounded."""
    desc = {"max_kv_size": max(int(max_kv_size or 0), 0)}
    return _sha(_canonical(desc)), desc


def kv_identity(
    *,
    kv_quantize: bool,
    kv_bits: int,
    kv_group_size: int,
    kv_min_quantize_tokens: int,
) -> tuple[str, dict[str, Any]]:
    """How this run quantizes the KV it holds, as ``(digest, description)``.

    Bits, group size and threshold only matter while quantization is on, so
    changing them with it off does not change the identity.
    """
    desc: dict[str, Any] = {"kv_quantize": bool(kv_quantize)}
    if kv_quantize:
        desc.update(
            kv_bits=kv_bits,
            kv_group_size=kv_group_size,
            kv_min_quantize_tokens=kv_min_quantize_tokens,
        )
    return _sha(_canonical(desc)), desc


def model_identity(
    model: Any, quantization_signature: str
) -> tuple[str, dict[str, Any]]:
    """The model-side identity as ``(digest, description)``."""
    weights = weights_digest(model)
    detail: dict[str, Any] = {
        "model_class": model_class_path(model),
        "config": config_fields(model),
        "quantization": quantization_signature,
        "weights": weights if weights is not None else "unreadable",
        "versions": runtime_versions(model),
    }
    hashed = dict(detail)
    if weights is None:
        hashed["weights"] = f"unreadable:{_UNREADABLE_NONCE}"
    return _sha(_canonical(hashed)), detail
