# SPDX-License-Identifier: Apache-2.0
"""
MLLM (Multimodal Language Model) Prefix Cache Manager.

This module provides advanced caching for MLLM inference, implementing
the LMCache-style approach for multimodal prefix caching:

Features:
- Image content hashing for cache keys (LMCache style)
- Vision embedding caching (skip encoder on hit)
- KV cache state caching with prefix matching
- Token ID tracking for partial prefix reuse
- LRU eviction policy with memory limits
- Stats tracking (hits, misses, tokens saved, encoder skips)

Based on research from:
- LMCache: https://blog.lmcache.ai/2025-07-03-multimodal-models/
- vLLM Prefix Caching: https://docs.vllm.ai/en/stable/design/prefix_caching/
- mlx-lm cache_prompt: https://github.com/ml-explore/mlx-lm
"""

import copy
import hashlib
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from vllm_mlx.mlx_cache_compat import clone_layer, is_plain_kv, recurrent_arrays

logger = logging.getLogger(__name__)


@dataclass
class MLLMCacheStats:
    """Statistics for MLLM cache performance."""

    hits: int = 0
    misses: int = 0
    partial_hits: int = 0  # Prefix matched but not full
    tokens_saved: int = 0
    image_cache_hits: int = 0
    vision_encoder_skips: int = 0  # Times we skipped vision encoder
    total_queries: int = 0
    evictions: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        if self.total_queries == 0:
            return 0.0
        return self.hits / self.total_queries

    def to_dict(self) -> dict:
        """Convert stats to dictionary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "partial_hits": self.partial_hits,
            "hit_rate": self.hit_rate,
            "tokens_saved": self.tokens_saved,
            "image_cache_hits": self.image_cache_hits,
            "vision_encoder_skips": self.vision_encoder_skips,
            "total_queries": self.total_queries,
            "evictions": self.evictions,
        }


@dataclass
class MLLMPrefixCacheEntry:
    """
    Enhanced cache entry storing vision embeddings, KV cache, and token IDs.

    This enables:
    1. Skipping vision encoder on image cache hit (saves ~1-2s per image)
    2. Skipping prefix computation on token match (saves ~0.5s per 1k tokens)
    3. Partial prefix reuse for multi-turn conversations
    """

    # Identity
    image_hash: str  # SHA256 of image content
    prompt_hash: str  # SHA256 of formatted prompt

    # Cached states - the key to performance
    vision_embeddings: Any = None  # Output of vision encoder (skip encoder on hit!)
    kv_cache: list[Any] = field(default_factory=list)  # Language model KV states

    # Token tracking for prefix matching
    token_ids: list[int] = field(default_factory=list)  # Full token sequence
    num_image_tokens: int = 0  # e.g., 256 for Gemma 3
    num_text_tokens: int = 0
    prompt_tokens: int = 0  # Total prompt tokens (for stats)

    # Metadata
    created_at: float = field(default_factory=time.time)
    hit_count: int = 0
    model_name: str = ""

    @property
    def total_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def memory_size(self) -> int:
        """Estimate memory usage in bytes."""
        size = 0
        if self.vision_embeddings is not None:
            if hasattr(self.vision_embeddings, "nbytes"):
                size += self.vision_embeddings.nbytes
        if self.kv_cache is not None:
            for layer_cache in self.kv_cache:
                # mlx-lm 0.32 layers report their own size; their ``.state``
                # nests lists (recurrent layers) that a flat sum would skip.
                try:
                    layer_bytes = layer_cache.nbytes
                except (AttributeError, NotImplementedError):
                    layer_bytes = None
                if isinstance(layer_bytes, int):
                    size += layer_bytes
                    continue
                if hasattr(layer_cache, "state"):
                    state = layer_cache.state
                    if state is not None:
                        for tensor in state:
                            if hasattr(tensor, "nbytes"):
                                size += tensor.nbytes
        return size

    def get_prefix_match_length(self, new_token_ids: list[int]) -> int:
        """
        Find how many tokens match between cached prefix and new input.

        This is the key to prefix caching - if the first N tokens match,
        we can skip computing KV states for those N tokens.
        """
        match_length = 0
        for i, (cached, new) in enumerate(zip(self.token_ids, new_token_ids)):
            if cached != new:
                break
            match_length = i + 1
        return match_length


def is_remote_source(source: str) -> bool:
    """True for an ``http(s)://`` URL.

    A URL names a resource, not its bytes: the same URL can serve another
    picture later, so a cache key built from it would hand the old picture's
    pixels or KV to the new one. Such a source has to be fetched to a local
    file (whose bytes :func:`compute_image_hash` hashes) before it can key
    anything; a cache refuses it.
    """
    return isinstance(source, str) and source.startswith(("http://", "https://"))


def compute_image_hash(image_path: str) -> str:
    """
    Compute hash of image content for cache key.

    Following LMCache approach: hash the actual image bytes, not the path.
    This ensures cache hits even when the same image is loaded from
    different paths or as base64.

    Args:
        image_path: Path to image file, or a ``data:`` URI (which carries its
            own content)

    Returns:
        SHA-256 hex digest of the image content (the whole digest: a prefix
        of it is a key someone can collide on purpose)
    """
    try:
        if not image_path.startswith("data:"):
            path = Path(image_path)
            if path.is_file():
                # Hash file content - this is the LMCache approach
                return hashlib.sha256(path.read_bytes()).hexdigest()
        # A data: URI, or a stand-in name that is no file
        return hashlib.sha256(image_path.encode()).hexdigest()
    except Exception as e:
        logger.warning(f"Failed to hash image: {e}")
        return hashlib.sha256(str(image_path).encode()).hexdigest()


def compute_images_hash(images: list[str]) -> str:
    """
    Compute combined hash for multiple images.

    Args:
        images: List of image paths/URLs

    Returns:
        Combined hash string
    """
    if not images:
        return "no_images"

    # Order matters: the k-th image fills the k-th placeholder, so [A, B] and
    # [B, A] are different requests and must not share a cache entry.
    hashes = [compute_image_hash(img) for img in images]
    combined = "_".join(hashes)
    return hashlib.sha256(combined.encode()).hexdigest()


def _snapshot_layer(layer: Any, keep_tokens: int) -> Any | None:
    """Independent copy of one cache layer cut to ``keep_tokens`` positions, or None."""
    # A container's children and a recurrent layer's state hold the whole
    # sequence folded into fixed-size state: there is no earlier position left
    # to cut back to.
    if (
        getattr(layer, "caches", None) is not None
        or recurrent_arrays(layer) is not None
    ):
        return None
    offset = getattr(layer, "offset", None)
    if not isinstance(offset, int) or offset < keep_tokens:
        return None
    # A window that has wrapped (or a buffer that has advanced its start) has
    # evicted the early positions: keeping the first ``keep_tokens`` would
    # silently return a rotation of the buffer.
    max_size = getattr(layer, "max_size", None)
    if max_size is not None and offset > max_size:
        return None
    if getattr(layer, "start_position", None) not in (None, 0):
        return None
    if is_plain_kv(layer):
        return clone_layer(layer, max_tokens=keep_tokens)
    drop = offset - keep_tokens
    if drop:
        is_trimmable = getattr(layer, "is_trimmable", None)
        if not callable(is_trimmable) or not is_trimmable():
            return None
    snapshot = copy.deepcopy(layer)
    if drop and snapshot.trim(drop) != drop:
        return None
    return snapshot if snapshot.offset == keep_tokens else None


def snapshot_prompt_prefix(
    prompt_cache: list[Any] | None, keep_tokens: int
) -> list[Any] | None:
    """Independent copy of ``prompt_cache`` holding exactly its first ``keep_tokens`` positions.

    The cache a request ran on holds the prompt and everything generated after
    it. A prefix-cache entry must hold the prompt alone, and must share no
    arrays or cursors with the live cache (which generation keeps extending
    in place). Returns None when any layer cannot be cut back to exactly
    ``keep_tokens`` -- recurrent state, a container layer, a sliding window
    that has already evicted positions -- so the caller stores nothing instead
    of a cache of some other length.
    """
    if not prompt_cache or keep_tokens <= 0:
        return None
    snapshot = []
    for layer in prompt_cache:
        part = _snapshot_layer(layer, keep_tokens)
        if part is None:
            return None
        snapshot.append(part)
    return snapshot


class MLLMPrefixCacheManager:
    """
    LRU Cache manager for MLLM prefix states with vision embedding caching.

    Implements the LMCache approach for multimodal caching:
    1. Hash-based identification of image+prompt combinations
    2. Vision embedding caching (skip encoder on hit - saves 1-2s!)
    3. KV cache reuse for matching prefixes
    4. Token ID tracking for partial prefix matching
    5. Memory-based eviction (configurable limit)

    Example:
        >>> cache = MLLMPrefixCacheManager(max_memory_mb=2048)
        >>> # First request - cache miss, full computation
        >>> entry, match_len = cache.fetch(["image.jpg"], prompt, token_ids)
        >>> # ... run full forward pass ...
        >>> cache.store(["image.jpg"], prompt, vision_emb, kv_cache, token_ids)
        >>>
        >>> # Second request with same image - cache hit!
        >>> entry, match_len = cache.fetch(["image.jpg"], prompt, token_ids)
        >>> # entry.vision_embeddings available - skip encoder!
        >>> # match_len > 0 - skip prefix computation!

    Performance (Gemma 3 27B, 256 image tokens):
        - Vision encoder: ~1.5s -> 0s (skip on hit)
        - Prefix computation: ~0.5s/1k tokens -> 0s (skip on match)
        - Multi-turn speedup: 8-12x for subsequent turns
    """

    def __init__(
        self,
        max_entries: int = 50,
        max_memory_mb: int = 2048,
    ):
        """
        Initialize MLLM prefix cache manager.

        Args:
            max_entries: Maximum number of cache entries (default: 50)
            max_memory_mb: Maximum memory in MB (default: 2048)
        """
        self.max_size = max_entries
        self.max_memory = max_memory_mb * 1024 * 1024
        self._cache: OrderedDict[str, MLLMPrefixCacheEntry] = OrderedDict()
        self._current_memory = 0
        self.stats = MLLMCacheStats()

    def _make_cache_key(self, images: list[str], prompt: str) -> str:
        """Create cache key from images and prompt."""
        image_hash = compute_images_hash(images)
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        return f"{image_hash}_{prompt_hash}"

    def _make_image_only_key(self, images: list[str]) -> str:
        """Create cache key for image-only lookup (vision embedding reuse)."""
        return compute_images_hash(images)

    def _evict_by_memory(self, required_size: int) -> None:
        """Evict entries until we have enough memory."""
        while self._current_memory + required_size > self.max_memory and self._cache:
            oldest_key = next(iter(self._cache))
            oldest_entry = self._cache.pop(oldest_key)
            self._current_memory -= oldest_entry.memory_size
            self.stats.evictions += 1
            logger.debug(f"MLLM cache evicted (memory): {oldest_key[:20]}...")

    def _evict_by_count(self) -> None:
        """Evict entries until we're under max_size."""
        while len(self._cache) >= self.max_size and self._cache:
            oldest_key = next(iter(self._cache))
            oldest_entry = self._cache.pop(oldest_key)
            self._current_memory -= oldest_entry.memory_size
            self.stats.evictions += 1
            logger.debug(f"MLLM cache evicted (count): {oldest_key[:20]}...")

    def fetch(
        self,
        images: list[str],
        prompt: str,
        token_ids: list[int] | None = None,
    ) -> tuple[MLLMPrefixCacheEntry | None, int]:
        """
        Fetch cached prefix state with prefix matching.

        This is the main entry point for cache lookups. Returns both
        the cache entry (if found) and the prefix match length.

        Args:
            images: List of image paths
            prompt: Text prompt
            token_ids: Optional token IDs for prefix matching

        Returns:
            Tuple of (entry, prefix_match_length) where:
            - entry: The cache entry if found, None otherwise
            - prefix_match_length: Number of tokens that match (0 if miss)
        """
        self.stats.total_queries += 1
        cache_key = self._make_cache_key(images, prompt)

        if cache_key in self._cache:
            # Full cache hit - exact image+prompt match
            entry = self._cache.pop(cache_key)
            self._cache[cache_key] = entry  # Move to end (LRU)
            entry.hit_count += 1

            self.stats.hits += 1
            if images:
                self.stats.image_cache_hits += 1
            if entry.vision_embeddings is not None:
                self.stats.vision_encoder_skips += 1

            # Calculate prefix match length
            match_length = entry.total_tokens
            if token_ids:
                match_length = entry.get_prefix_match_length(token_ids)
                if match_length < entry.total_tokens:
                    self.stats.partial_hits += 1

            self.stats.tokens_saved += match_length
            logger.debug(
                f"MLLM cache HIT: {cache_key[:32]}..., prefix_match={match_length}"
            )

            return self._detached(entry), match_length

        # Check for image-only match (can reuse vision embeddings)
        if images:
            image_key = self._make_image_only_key(images)
            for key, entry in self._cache.items():
                if (
                    entry.image_hash == image_key
                    and entry.vision_embeddings is not None
                ):
                    # Image match - can reuse vision embeddings!
                    self.stats.partial_hits += 1
                    self.stats.vision_encoder_skips += 1
                    logger.debug(
                        f"MLLM cache PARTIAL HIT (vision only): image={image_key[:16]}"
                    )

                    # Return entry for vision embeddings, but 0 prefix match
                    # (prompt is different, so KV cache can't be reused)
                    return self._detached(entry), 0

        self.stats.misses += 1
        logger.debug(f"MLLM cache MISS: {cache_key[:32]}...")
        return None, 0

    @staticmethod
    def _detached(entry: MLLMPrefixCacheEntry) -> MLLMPrefixCacheEntry:
        """A copy of ``entry`` that shares no mutable state with the stored one.

        The stored ``kv_cache`` is a prompt snapshot another request may fetch
        later. A caller that hands its list to mlx-vlm gets it extended in
        place (every layer ``update_and_fetch`` appends), so the stored list
        must never leave the manager.
        """
        return replace(
            entry,
            kv_cache=copy.deepcopy(entry.kv_cache),
            token_ids=list(entry.token_ids),
        )

    def fetch_cache(
        self,
        images: list[str],
        prompt: str,
    ) -> tuple[list[Any] | None, bool]:
        """
        Fetch cached KV state for image+prompt combination.

        Returns an independent copy of the stored layers: the caller may feed
        it to a model (which extends it in place) without touching the entry.
        A hit says nothing about how many positions the layers hold; the
        Simple engine uses :meth:`fetch_prompt_cache`, which checks that.
        """
        entry, match_len = self.fetch(images, prompt)
        # For legacy API, return hit if entry exists (don't require token match)
        if entry is not None and entry.kv_cache is not None:
            return entry.kv_cache, True
        return None, False

    def fetch_prompt_cache(
        self,
        images: list[str],
        prompt: str,
        token_ids: list[int],
    ) -> tuple[list[Any], list[int]] | None:
        """Reusable prompt KV for this exact request, as ``(layers, cached_ids)``.

        ``layers`` is an independent copy holding exactly the positions of
        ``cached_ids``, and ``cached_ids`` is a strict prefix of ``token_ids``
        (the request's real prompt ids, image placeholders included): the
        generation call only has to prefill ``token_ids[len(cached_ids):]``.
        Returns None (counted as a miss) when nothing is stored under the
        key or the stored ids are not such a prefix -- e.g. the same
        images+prompt text tokenized differently.
        """
        entry = self._cache.get(self._make_cache_key(images, prompt))
        cached_ids = entry.token_ids if entry is not None else None
        if (
            entry is None
            or not entry.kv_cache
            or not cached_ids
            or len(cached_ids) >= len(token_ids)
            or list(token_ids[: len(cached_ids)]) != list(cached_ids)
        ):
            self.stats.total_queries += 1
            self.stats.misses += 1
            return None
        hit, _ = self.fetch(images, prompt, token_ids)
        return hit.kv_cache, hit.token_ids

    def store(
        self,
        images: list[str],
        prompt: str,
        vision_embeddings: Any,
        kv_cache: list[Any],
        token_ids: list[int],
        num_image_tokens: int = 0,
        model_name: str = "",
    ) -> bool:
        """
        Store prefix state in cache.

        The manager takes ownership of ``kv_cache``: it must be a snapshot no
        one else extends (see :meth:`store_prompt_cache`), not a live cache.

        Args:
            images: List of image paths
            prompt: Text prompt
            vision_embeddings: Output of vision encoder (can be None for text-only)
            kv_cache: Language model KV cache states
            token_ids: Token sequence the ``kv_cache`` positions correspond to
            num_image_tokens: Number of image tokens (e.g., 256 for Gemma 3)
            model_name: Model name for validation

        Returns:
            True if an entry was stored; False for a remote ``http(s)://``
            source, whose content is not pinned (see :func:`is_remote_source`).
        """
        if any(is_remote_source(image) for image in images):
            return False
        cache_key = self._make_cache_key(images, prompt)
        replaced = self._cache.pop(cache_key, None)
        if replaced is not None:
            self._current_memory -= replaced.memory_size

        entry = MLLMPrefixCacheEntry(
            image_hash=compute_images_hash(images),
            prompt_hash=hashlib.sha256(prompt.encode()).hexdigest(),
            vision_embeddings=vision_embeddings,
            kv_cache=kv_cache,
            token_ids=token_ids,
            num_image_tokens=num_image_tokens,
            num_text_tokens=len(token_ids) - num_image_tokens,
            prompt_tokens=len(token_ids),
            model_name=model_name,
        )

        # Evict by memory first
        self._evict_by_memory(entry.memory_size)

        # Then evict by count
        self._evict_by_count()

        self._cache[cache_key] = entry
        self._current_memory += entry.memory_size

        logger.debug(
            f"MLLM cache STORED: key={cache_key[:32]}..., "
            f"tokens={len(token_ids)}, vision_emb={vision_embeddings is not None}, "
            f"memory={entry.memory_size / 1024 / 1024:.1f}MB"
        )
        return True

    def store_cache(
        self,
        images: list[str],
        prompt: str,
        cache: list[Any] | None,
        num_tokens: int = 0,
    ) -> None:
        """
        Legacy API: Store KV cache for future reuse.

        Stores an independent copy, so a cache the caller keeps using (a live
        one that generation extends in place) cannot change the entry. It
        does not cut the copy back to a prompt length -- that is
        :meth:`store_prompt_cache`.
        """
        # Don't store empty or None caches
        if cache is None or (isinstance(cache, list) and len(cache) == 0):
            return

        self.store(
            images=images,
            prompt=prompt,
            vision_embeddings=None,
            kv_cache=copy.deepcopy(cache),
            token_ids=[0] * num_tokens,  # Dummy token IDs
            num_image_tokens=0,
        )

    def store_prompt_cache(
        self,
        images: list[str],
        prompt: str,
        live_cache: list[Any] | None,
        token_ids: list[int],
        model_name: str = "",
    ) -> bool:
        """Store the prompt KV of a finished request; True if an entry was stored.

        ``live_cache`` is the cache the request ran on: it holds the prompt
        *and* what was generated, and is not touched here. The entry is an
        independent snapshot of its first ``len(token_ids) - 1`` positions
        (see :func:`snapshot_prompt_prefix`): all but the last prompt token,
        which a hit prefills itself because the logits of the last prompt
        position are what the first generated token is sampled from.
        Nothing is stored when the layers cannot be cut back to that length.
        """
        keep = len(token_ids) - 1
        snapshot = snapshot_prompt_prefix(live_cache, keep)
        if snapshot is None:
            return False
        return self.store(
            images=images,
            prompt=prompt,
            vision_embeddings=None,
            kv_cache=snapshot,
            token_ids=list(token_ids[:keep]),
            model_name=model_name,
        )

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        stats = self.stats.to_dict()
        stats["entries"] = len(self._cache)
        stats["max_entries"] = self.max_size
        stats["memory_used_mb"] = self._current_memory / 1024 / 1024
        stats["max_memory_mb"] = self.max_memory / 1024 / 1024
        return stats

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self.stats = MLLMCacheStats()

    def clear(self) -> None:
        """Clear all cached entries and reset stats."""
        self._cache.clear()
        self._current_memory = 0
        self.reset_stats()

    def __len__(self) -> int:
        """Return number of cached entries."""
        return len(self._cache)

    def __repr__(self) -> str:
        mem_mb = self._current_memory / 1024 / 1024
        return f"<MLLMPrefixCacheManager entries={len(self)} memory={mem_mb:.1f}MB>"


# Short alias for convenience
MLLMCacheManager = MLLMPrefixCacheManager

# Legacy aliases for backwards compatibility
VLMCacheStats = MLLMCacheStats
VLMPrefixCacheEntry = MLLMPrefixCacheEntry
VLMCacheEntry = MLLMPrefixCacheEntry
VLMPrefixCacheManager = MLLMPrefixCacheManager
VLMCacheManager = MLLMPrefixCacheManager
