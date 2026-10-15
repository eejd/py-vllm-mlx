# SPDX-License-Identifier: Apache-2.0
"""
Prefix Cache Manager for vllm-mlx.

Wraps mlx-lm's LRUPromptCache to provide prefix caching functionality,
allowing reuse of computed KV cache for common prompt prefixes.

This module provides two implementations:
- PrefixCacheManager: Original trie-based LRU cache (for backward compatibility)
- BlockAwarePrefixCache: Block-based cache with PagedCacheManager integration
"""

import copy
import functools
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    mx = None
    HAS_MLX = False

from .mlx_cache_compat import restore_from_state
from .paged_cache import BlockTable, PagedCacheManager, compute_chain_hashes

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Entry in the prefix cache."""

    prompt_cache: List[Any]  # The cached KV state
    count: int  # Reference count for sharing


@dataclass
class PrefixCacheStats:
    """Statistics for prefix cache performance."""

    hits: int = 0
    misses: int = 0
    tokens_saved: int = 0
    total_queries: int = 0
    evictions: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        if self.total_queries == 0:
            return 0.0
        return self.hits / self.total_queries

    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hit_rate,
            "tokens_saved": self.tokens_saved,
            "total_queries": self.total_queries,
            "evictions": self.evictions,
        }


class PrefixCacheManager:
    """
    Manages prefix caching for vllm-mlx using a trie-based LRU cache.

    This implementation is inspired by mlx-lm's LRUPromptCache but adapted
    for vllm-mlx's batching architecture.

    The cache stores KV states keyed by token sequences, allowing:
    - Exact match: Full prompt found in cache
    - Shorter match: Partial prefix found, process remaining tokens
    - Longer match: Cached prefix longer than request, trim excess

    Example:
        cache_manager = PrefixCacheManager(model, max_entries=100)

        # Check for cached prefix
        cache, remaining_tokens = cache_manager.fetch_cache(tokens)
        if cache:
            # Use cached KV, only process remaining_tokens
            pass

        # After generation, store cache for reuse
        cache_manager.store_cache(full_tokens, prompt_cache)
    """

    def __init__(self, model: Any, max_entries: int = 100):
        """
        Initialize the prefix cache manager.

        Args:
            model: The MLX model (used for cache key identification)
            max_entries: Maximum number of cached entries before LRU eviction
        """
        self.model = model
        self.model_key = id(model)
        self.max_size = max_entries

        # Trie-based cache: nested dicts with token keys
        # Structure: {model_key: {token1: {token2: {..., "cache": CacheEntry}}}}
        self._cache: Dict[Any, Dict] = {}

        # LRU tracking: OrderedDict keyed by (model_key, tuple(tokens)), insertion
        # order = least-recently-used first. move_to_end() and popitem() are O(1).
        self._lru: OrderedDict = OrderedDict()

        # Statistics
        self.stats = PrefixCacheStats()

    def _search(
        self, tokens: List[int]
    ) -> Tuple[Optional[List[int]], Optional[List[int]], Optional[List[int]], int]:
        """
        Search for cached prefix matching tokens.

        Returns:
            Tuple of (exact, shorter, longer, common_prefix_len)
            - exact: Tokens if exact match found
            - shorter: Tokens of shorter cached prefix
            - longer: Tokens of longer cached prefix
            - common_prefix_len: Length of common prefix with longer match
        """
        if self.model_key not in self._cache:
            return None, None, None, 0

        current = self._cache[self.model_key]
        path = []

        # Traverse trie following token sequence
        for i, tok in enumerate(tokens):
            if tok not in current:
                # No match for this token
                # Check if we have a shorter prefix with cache
                if "cache" in current:
                    return None, list(path), None, 0
                return None, None, None, 0

            path.append(tok)
            current = current[tok]

        # Reached end of tokens
        if "cache" in current:
            # Exact match
            return list(tokens), None, None, 0

        # Check for longer cached prefix
        # DFS to find shortest extension with cache
        stack = [(current, list(path))]
        while stack:
            node, node_path = stack.pop()
            if "cache" in node:
                return None, None, node_path, len(tokens)
            for tok, child in node.items():
                if tok != "cache":
                    stack.append((child, node_path + [tok]))

        return None, None, None, 0

    def fetch_cache(self, tokens: List[int]) -> Tuple[Optional[List[Any]], List[int]]:
        """
        Find cached prefix for the given tokens.

        Args:
            tokens: Input token sequence

        Returns:
            Tuple of (cache, remaining_tokens)
            - cache: Cached KV state if found, None otherwise
            - remaining_tokens: Tokens that still need processing
        """
        self.stats.total_queries += 1
        tokens_tuple = tuple(tokens)

        exact, shorter, longer, common_len = self._search(tokens)

        if exact:
            # Exact match - return full cache
            cache_entry = self._get_cache_entry(exact)
            if cache_entry:
                self.stats.hits += 1
                self.stats.tokens_saved += len(tokens)
                self._touch_lru(tokens_tuple)
                # No copy needed - MLX arrays are immutable
                return cache_entry.prompt_cache, []

        if shorter:
            # Shorter prefix cached - return cache and remaining tokens
            cache_entry = self._get_cache_entry(shorter)
            if cache_entry:
                self.stats.hits += 1
                self.stats.tokens_saved += len(shorter)
                self._touch_lru(tuple(shorter))
                remaining = tokens[len(shorter) :]
                # No copy needed - MLX arrays are immutable
                return cache_entry.prompt_cache, remaining

        if longer:
            # Longer prefix cached - trim to match and return
            cache_entry = self._get_cache_entry(longer)
            if cache_entry:
                # Check if cache supports trimming
                prompt_cache = cache_entry.prompt_cache
                if self._can_trim_cache(prompt_cache):
                    trim_amount = len(longer) - len(tokens)
                    trimmed_cache = self._trim_cache(
                        copy.deepcopy(prompt_cache), trim_amount
                    )
                    self.stats.hits += 1
                    self.stats.tokens_saved += len(tokens)
                    return trimmed_cache, []

        # No cache hit
        self.stats.misses += 1
        return None, tokens

    def store_cache(self, tokens: List[int], prompt_cache: List[Any]) -> None:
        """
        Store computed cache for future reuse.

        Args:
            tokens: Token sequence that was processed
            prompt_cache: The computed KV cache to store
        """
        if not tokens:
            return

        tokens_tuple = tuple(tokens)

        # Build trie path
        if self.model_key not in self._cache:
            self._cache[self.model_key] = {}

        current = self._cache[self.model_key]
        for tok in tokens:
            if tok not in current:
                current[tok] = {}
            current = current[tok]

        # Store or update cache entry
        key = (self.model_key, tokens_tuple)
        if "cache" in current:
            current["cache"].count += 1
            # Move to most-recently-used position — O(1) with OrderedDict
            self._lru.move_to_end(key)
        else:
            current["cache"] = CacheEntry(prompt_cache, 1)
            self._lru[key] = None

        # Evict if over capacity
        while len(self._lru) > self.max_size:
            self._evict_lru()

    def _get_cache_entry(self, tokens: List[int]) -> Optional[CacheEntry]:
        """Get cache entry for given tokens."""
        if self.model_key not in self._cache:
            return None

        current = self._cache[self.model_key]
        for tok in tokens:
            if tok not in current:
                return None
            current = current[tok]

        return current.get("cache")

    def _touch_lru(self, tokens_tuple: tuple) -> None:
        """Move entry to most-recently-used position — O(1) with OrderedDict."""
        key = (self.model_key, tokens_tuple)
        if key in self._lru:
            self._lru.move_to_end(key)
        else:
            self._lru[key] = None

    def _evict_lru(self) -> None:
        """Evict least recently used entry — O(1) popitem from OrderedDict."""
        if not self._lru:
            return

        (model_key, tokens_tuple), _ = self._lru.popitem(last=False)
        self._delete_cache(model_key, list(tokens_tuple))
        self.stats.evictions += 1

    def _delete_cache(self, model_key: Any, tokens: List[int]) -> None:
        """Delete cache entry and clean up empty trie branches."""
        if model_key not in self._cache:
            return

        # Navigate to entry
        path = [(self._cache[model_key], None)]
        current = self._cache[model_key]

        for tok in tokens:
            if tok not in current:
                return
            path.append((current[tok], tok))
            current = current[tok]

        # Delete cache entry
        if "cache" in current:
            del current["cache"]

        # Clean up empty branches (bottom-up)
        for i in range(len(path) - 1, 0, -1):
            node, tok = path[i]
            parent, _ = path[i - 1]
            if not node:  # Empty dict
                del parent[tok]

    def _can_trim_cache(self, prompt_cache: List[Any]) -> bool:
        """Check if cache can be trimmed."""
        if not prompt_cache:
            return False
        # Check if first cache layer has is_trimmable method
        first_cache = prompt_cache[0]
        if hasattr(first_cache, "is_trimmable"):
            trimmable = first_cache.is_trimmable()
            if not trimmable:
                logger.debug(
                    "Prefix cache reuse skipped: cache is not trimmable "
                    "(RotatingKVCache does not support trimming)"
                )
            return trimmable
        return hasattr(first_cache, "trim")

    def _trim_cache(self, prompt_cache: List[Any], num_tokens: int) -> List[Any]:
        """Trim cache by removing num_tokens from the end."""
        for cache in prompt_cache:
            if hasattr(cache, "trim"):
                cache.trim(num_tokens)
        return prompt_cache

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.stats.to_dict()

    def reset_stats(self) -> None:
        """Reset statistics."""
        self.stats = PrefixCacheStats()

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        self._lru.clear()
        self.reset_stats()

    def __len__(self) -> int:
        """Return number of cached entries."""
        return len(self._lru)


# =============================================================================
# Block-Aware Prefix Cache (uses PagedCacheManager)
# =============================================================================


def _under_pool_lock(method):
    """Run a BlockAwarePrefixCache method holding the paged pool's lock.

    ``fetch_cache`` and ``store_cache`` look blocks up and then pin them. A block
    that is found free but not yet referenced can be recycled by another thread
    (the scheduler adds requests and stores finished ones from different
    threads), and the caller would then hold a block that carries somebody
    else's tokens. The pool's lock is re-entrant, so the whole operation can
    hold it.
    """

    @functools.wraps(method)
    def locked(self, *args, **kwargs):
        with self.paged_cache.lock:
            return method(self, *args, **kwargs)

    return locked


def _own(value: Any) -> Any:
    """Copy every array of a (nested) cache state into a buffer of its own.

    A slice of an evaluated MLX array is a view that keeps the whole parent
    buffer alive. A cached block that merely sliced the finished request's KV
    would pin all of that request's KV for as long as the block lives, so the
    pool size (``max_cache_blocks``) would bound nothing: the memory held
    could be many times the blocks' nominal size. Non-array values (the integer
    cursors of a rotating layer) pass through.
    """
    if mx is not None and isinstance(value, mx.array):
        return mx.array(value)
    if isinstance(value, tuple):
        return tuple(_own(v) for v in value)
    if isinstance(value, list):
        return [_own(v) for v in value]
    return value


def _arrays_in(value: Any) -> List[Any]:
    """Every array in a (nested) cache state."""
    if mx is not None and isinstance(value, mx.array):
        return [value]
    if isinstance(value, (tuple, list)):
        return [a for v in value for a in _arrays_in(v)]
    return []


def _is_array_state(state: Any) -> bool:
    """A layer state made of arrays only (no integer cursors, no nesting)."""
    return (
        isinstance(state, (list, tuple))
        and bool(state)
        and all(
            v is None or (mx is not None and isinstance(v, mx.array)) for v in state
        )
    )


@dataclass
class BlockCacheEntry:
    """Entry mapping a token sequence to cache blocks."""

    block_table: BlockTable
    cache_data: List[Any]  # Actual KV cache data per block
    last_access: float


class BlockAwarePrefixCache:
    """
    Prefix cache that uses PagedCacheManager for block-based storage.

    Features:
    - Block-level prefix sharing (64 tokens per block)
    - Block identity is a chain hash (parent block hash + tokens + extra keys),
      as in vLLM: a block is reused only behind the exact prefix it was
      computed after
    - Copy-on-Write for efficient forking
    - Reference counting for memory efficiency

    This is the recommended cache for production use when memory
    efficiency for concurrent requests is important.

    Example:
        paged_manager = PagedCacheManager(block_size=64, max_blocks=1000)
        cache = BlockAwarePrefixCache(model, paged_manager)

        # Check for cached prefix
        block_table, remaining_tokens = cache.fetch_cache(request_id, tokens)

        # After generation, store cache
        cache.store_cache(request_id, tokens, kv_cache_data)

        # Clean up when request completes
        cache.release_cache(request_id)
    """

    def __init__(
        self,
        model: Any,
        paged_cache_manager: PagedCacheManager,
    ):
        """
        Initialize block-aware prefix cache.

        Args:
            model: The MLX model (used for identification)
            paged_cache_manager: The PagedCacheManager instance for block management
        """
        self.model = model
        self.model_key = id(model)
        self.paged_cache = paged_cache_manager
        self.block_size = paged_cache_manager.block_size

        # Request to block table mapping
        self._request_tables: Dict[str, BlockCacheEntry] = {}

        # Layouts already reported (once per kind, not per request)
        self._warned_layouts: set = set()
        self._warned_recurrent = False

        # Layouts already reported (once per kind, not per request)
        self._warned_layouts: set = set()
        self._warned_recurrent = False

        # Statistics
        self._hits = 0
        self._misses = 0
        self._tokens_saved = 0

    @_under_pool_lock
    def fetch_cache(
        self,
        request_id: str,
        tokens: List[int],
        extra_keys: Optional[Tuple[Any, ...]] = None,
    ) -> Tuple[Optional[BlockTable], List[int]]:
        """
        Find cached prefix blocks for the given tokens.

        A block is a hit only behind the exact blocks it was computed after:
        blocks are looked up by chain hash (parent block hash + tokens + extra
        keys), so identical tokens cached behind a different prefix, or at a
        different position, are never served. The KV of a token depends on
        every token before it; reusing it behind another prefix answers from
        a context that never existed.

        Args:
            request_id: Unique request identifier
            tokens: Input token sequence
            extra_keys: Per-request inputs that change the KV of identical
                tokens (multimodal content, cache salt, adapter id)

        Returns:
            Tuple of (block_table, remaining_tokens)
            - block_table: BlockTable if prefix found, None otherwise
            - remaining_tokens: Tokens that need processing
        """
        if not tokens:
            return None, tokens

        # A request id that comes back (retry) must not leak the refs of its
        # previous table.
        self.paged_cache.delete_block_table(request_id)
        self._request_tables.pop(request_id, None)

        matched, _ = self.paged_cache.get_computed_blocks(tokens, extra_keys)

        # The prefix has to end on a block that can be restored as of its end
        # (a recurrent layer's snapshot exists only where it was taken);
        # blocks stored without tensors (reference-only data) have nothing to
        # restore and are plain lookups.
        usable = list(matched)
        while usable and not self._is_restorable_end(usable[-1]):
            usable.pop()

        if usable:
            self.paged_cache.touch(usable)
            block_table = self.paged_cache.create_block_table(request_id)
            for block in usable:
                block_table.block_ids.append(block.block_id)
                block_table.num_tokens += block.token_count

            num_prefix_tokens = block_table.num_tokens
            remaining = tokens[num_prefix_tokens:]
            self._hits += 1
            self._tokens_saved += num_prefix_tokens

            logger.debug(
                f"Cache hit for {request_id}: "
                f"{len(usable)} blocks, {num_prefix_tokens} tokens"
            )

            return block_table, remaining

        # No cache hit
        self._misses += 1
        logger.debug(f"Cache miss for {request_id}")
        return None, tokens

    @staticmethod
    def _is_restorable_end(block: Any) -> bool:
        """Can a prefix that ends with ``block`` be rebuilt as of its end?

        Every layer needs an entry in the last block: the slice of a
        sequence-backed layer, or the snapshot of a recurrent layer taken at
        exactly this block's last token.
        """
        data = block.cache_data
        return data is None or all(entry is not None for entry in data)

    def _layout_supported(self, cache_data: List[Any]) -> bool:
        """Can every layer of this cache be stored as blocks (and rebuilt)?

        A layer's state is sliced along the sequence axis (plain KV), or kept
        whole as the snapshot of the block where the request ended (recurrent
        arrays). A layer whose state carries integer cursors (mlx-lm 0.32
        rotating/chunked layers) or nested tuples (quantized KV) is neither:
        slicing it fails. Nothing is cached for such a model, said once instead
        of a warning per finished request.
        """
        for layer_state in cache_data:
            state = layer_state.get("state") if isinstance(layer_state, dict) else None
            if not _is_array_state(state):
                name = (
                    layer_state.get("class_name")
                    if isinstance(layer_state, dict)
                    else type(layer_state).__name__
                )
                if name not in self._warned_layouts:
                    self._warned_layouts.add(name)
                    logger.info(
                        "Paged prefix cache: %s layers cannot be stored as "
                        "blocks; requests of this model are not cached",
                        name,
                    )
                return False
        if not self._warned_recurrent and any(
            self._cache_state_seq_axis(layer_state["state"]) is None
            for layer_state in cache_data
        ):
            self._warned_recurrent = True
            logger.info(
                "Paged prefix cache: this model has recurrent layers; their "
                "state exists only where a request ended, so a prefix hit "
                "needs a stored request that ended on a block boundary"
            )
        return True

    def _cache_extent(self, cache_data: List[Any]) -> int:
        """Number of token positions the extracted cache actually holds.

        Taken from the tensors, never from the token list: the KV of a
        finished request normally covers all of ``prompt + output`` (the
        scheduler's contract), but a path that leaves it one position short
        (a sampled token that was never fed back, a draft/MTP rollback) would
        otherwise get a block hashed for N tokens whose slice holds N-1. That
        block would be reconstructed one position short and every later token
        would sit at the wrong position.

        Returns 0 when the extent cannot be established (no sequence-backed
        layer, or layers that disagree): nothing is then claimed.
        """
        extents = set()
        for layer_state in cache_data:
            state = layer_state.get("state")
            axis = self._cache_state_seq_axis(state)
            if axis is not None:
                extents.add(state[0].shape[axis])
        return extents.pop() if len(extents) == 1 else 0

    @_under_pool_lock
    def store_cache(
        self,
        request_id: str,
        tokens: List[int],
        cache_data: List[Any],
        extra_keys: Optional[Tuple[Any, ...]] = None,
    ) -> Optional[BlockTable]:
        """
        Store computed cache for future reuse.

        This method stores actual tensor data (not references) when cache_data
        contains extracted states from mlx-lm's KVCache.state property.

        Only full blocks whose KV exists are cached, each under its chain hash
        (see :meth:`fetch_cache`); a block already cached behind the same
        prefix is shared, never one that merely holds the same tokens.

        Args:
            request_id: Unique request identifier
            tokens: Token sequence that was processed
            cache_data: The computed KV cache to store. Can be:
                - List of KVCache objects (legacy, stores references)
                - List of dicts with 'state': (keys, values) tensors (new, stores slices)
            extra_keys: Per-request inputs that change the KV of identical
                tokens (multimodal content, cache salt, adapter id)

        Returns:
            BlockTable for the stored cache, or None on failure
        """
        if not tokens:
            return None

        # Check if cache_data contains extracted tensor states
        is_tensor_data = bool(
            cache_data
            and isinstance(cache_data, list)
            and isinstance(cache_data[0], dict)
            and "state" in cache_data[0]
        )

        if is_tensor_data and not self._layout_supported(cache_data):
            return None

        # Positions whose KV really exists. Reference-only data (legacy) has
        # no tensors to bound it, nothing is sliced from it either.
        extent = self._cache_extent(cache_data) if is_tensor_data else len(tokens)
        covered = min(len(tokens), extent)
        num_full_blocks = covered // self.block_size
        hashes = compute_chain_hashes(
            tokens, self.block_size, extra_keys, num_full_blocks
        )

        # Get or create block table
        block_table = self.paged_cache.get_block_table(request_id)
        if not block_table:
            block_table = self.paged_cache.create_block_table(request_id)

        # Keep the blocks of the table that are exactly this sequence's chain
        # (a table from fetch_cache, or from an earlier store of the same
        # request); anything else is dropped rather than extended.
        keep = 0
        for block_id in block_table.block_ids[:num_full_blocks]:
            block = self.paged_cache.allocated_blocks.get(block_id)
            if block is None or block.block_hash != hashes[keep]:
                break
            keep += 1
        self._truncate_table(block_table, keep)

        for i in range(keep, num_full_blocks):
            start = i * self.block_size
            end = start + self.block_size
            block_tokens = tokens[start:end]
            parent_hash = hashes[i - 1] if i > 0 else None

            # Deduplicate against a block cached behind the same prefix
            existing_block = self.paged_cache.find_cached_block(
                block_tokens, parent_hash, extra_keys
            )
            if existing_block is not None:
                self.paged_cache.touch([existing_block])
                block_table.block_ids.append(existing_block.block_id)
                block_table.num_tokens += self.block_size
                if is_tensor_data and HAS_MLX:
                    self._complete_shared_block(
                        existing_block, cache_data, start, end, covered
                    )
                continue

            # Allocate new block
            block = self.paged_cache.allocate_block()
            if not block:
                logger.warning(f"Cannot allocate block for {request_id}")
                break

            # Extract and store actual tensor slices for this block
            if is_tensor_data:
                block_kv_data = (
                    self._extract_block_tensor_slice(cache_data, start, end, covered)
                    if HAS_MLX
                    else None
                )
                if not block_kv_data:
                    # No KV for this block: it must not be findable
                    self.paged_cache.free_block(block.block_id)
                    break
                block.cache_data = block_kv_data
                logger.debug(
                    f"Stored tensor slice for block {block.block_id}: "
                    f"tokens [{start}:{end}], {len(block_kv_data)} layers"
                )

            self.paged_cache.add_block_to_table(block_table, block, self.block_size)
            self.paged_cache.register_block_hash(
                block, block_tokens, parent_hash, extra_keys
            )

        # Store entry for request (for legacy compatibility)
        self._request_tables[request_id] = BlockCacheEntry(
            block_table=block_table,
            cache_data=cache_data,
            last_access=time.time(),
        )

        blocks_with_data = sum(
            1
            for bid in block_table.block_ids
            if self.paged_cache.allocated_blocks.get(bid)
            and self.paged_cache.allocated_blocks[bid].cache_data is not None
        )

        logger.debug(
            f"Stored cache for {request_id}: "
            f"{len(block_table.block_ids)} blocks ({blocks_with_data} with tensor data), "
            f"{block_table.num_tokens} tokens"
        )

        return block_table

    def _truncate_table(self, block_table: BlockTable, keep: int) -> None:
        """Drop every block of the table after the first ``keep``."""
        dropped = block_table.block_ids[keep:]
        if not dropped:
            return
        del block_table.block_ids[keep:]
        block_table.num_tokens = keep * self.block_size
        for block_id in reversed(dropped):
            self.paged_cache.free_block(block_id)

    def _complete_shared_block(
        self,
        block: Any,
        cache_data: List[Dict[str, Any]],
        start_idx: int,
        end_idx: int,
        covered: int,
    ) -> None:
        """Fill in what a shared block was stored without.

        The block holds the same tokens behind the same prefix as this request,
        so this request's KV for those positions is the block's KV. Two gaps
        can be closed from it: a block cached as a bare lookup (no tensors),
        and the recurrent snapshots of a block cached mid-sequence, which
        exist only where a request ended (``end_idx == covered``) and let the
        block end a hit.
        """
        data = block.cache_data
        if data is not None and (
            end_idx != covered or all(entry is not None for entry in data)
        ):
            return
        fresh = self._extract_block_tensor_slice(
            cache_data, start_idx, end_idx, covered
        )
        if not fresh:
            return
        if data is None:
            block.cache_data = fresh
            return
        if len(fresh) != len(data):
            return
        for idx, entry in enumerate(data):
            if entry is None and fresh[idx] is not None:
                data[idx] = fresh[idx]

    def _extract_block_tensor_slice(
        self,
        cache_data: List[Dict[str, Any]],
        start_idx: int,
        end_idx: int,
        total_tokens: int,
    ) -> Optional[List[Optional[Dict[str, Any]]]]:
        """
        Extract per-layer cache data for a single block.

        Args:
            cache_data: List of extracted layer states
            start_idx: Start token index in the sequence
            end_idx: End token index in the sequence
            total_tokens: Total number of tokens covered by cache_data

        Returns:
            Per-layer block cache state, or None on failure
        """
        if not HAS_MLX or not cache_data:
            return None

        try:
            block_slices: List[Optional[Dict[str, Any]]] = []
            for layer_state in cache_data:
                if "state" not in layer_state:
                    block_slices.append(None)
                    continue

                state = layer_state["state"]
                meta_state = layer_state.get("meta_state")
                class_ref = layer_state.get("class_ref")
                class_name = layer_state.get("class_name")

                seq_axis = self._cache_state_seq_axis(state)
                if seq_axis is not None:
                    state_slice = self._slice_concat_cache_state(
                        state, start_idx, end_idx
                    )
                    block_slices.append(
                        {
                            "state": state_slice,
                            "meta_state": meta_state,
                            "class_ref": class_ref,
                            "class_name": class_name,
                            "storage": "concat",
                            "seq_axis": seq_axis,
                        }
                    )
                    continue

                if end_idx == total_tokens:
                    block_slices.append(
                        {
                            "state": _own(state),
                            "meta_state": meta_state,
                            "class_ref": class_ref,
                            "class_name": class_name,
                            "storage": "latest",
                        }
                    )
                else:
                    block_slices.append(None)

            if not any(entry is not None for entry in block_slices):
                return None

            # Materialize the copies now: a lazy copy would still reference
            # the parent buffer until something evaluates it.
            arrays = [
                a for entry in block_slices if entry for a in _arrays_in(entry["state"])
            ]
            if arrays:
                mx.eval(*arrays)
            return block_slices

        except Exception as e:
            logger.warning(f"Failed to extract block tensor slice: {e}")
            return None

    def _cache_state_seq_axis(self, state: Any) -> Optional[int]:
        """Return the sequence axis for cache states that support block concat."""
        if not isinstance(state, (list, tuple)) or not state:
            return None
        if not _is_array_state(state):
            return None  # integer cursors / nested states: not block-sliceable

        ndims = {
            len(tensor.shape)
            for tensor in state
            if tensor is not None and hasattr(tensor, "shape")
        }
        if len(ndims) != 1:
            return None

        ndim = next(iter(ndims))
        if ndim == 4:
            return 2

        # Qwen3.5-style KV caches use (heads, seq, dim).
        if ndim == 3 and len(state) == 2:
            return 1

        return None

    def _slice_concat_cache_state(
        self,
        state: Tuple[Any, ...] | List[Any],
        start_idx: int,
        end_idx: int,
    ) -> Tuple[Any, ...] | List[Any]:
        """Slice a sequence-backed cache state across the token axis."""
        seq_axis = self._cache_state_seq_axis(state)
        if seq_axis is None:
            raise ValueError("Cache state does not support sequence concatenation")

        seq_len = state[0].shape[seq_axis]
        actual_end = min(end_idx, seq_len)
        if start_idx >= actual_end:
            raise ValueError(
                f"Block slice [{start_idx}:{end_idx}] exceeds seq_len {seq_len}"
            )

        def _slice_tensor(tensor: Any) -> Any:
            slices = [slice(None)] * len(tensor.shape)
            slices[seq_axis] = slice(start_idx, actual_end)
            return tensor[tuple(slices)]

        sliced = [_own(_slice_tensor(tensor)) for tensor in state]
        return tuple(sliced) if isinstance(state, tuple) else sliced

    def _concat_cache_states(
        self,
        states: List[Tuple[Any, ...] | List[Any]],
        seq_axis: int,
    ) -> Optional[Tuple[Any, ...] | List[Any]]:
        """Concatenate state fragments for a sequence-backed cache layer."""
        if not states:
            return None
        arity = len(states[0])
        concatenated = []
        for idx in range(arity):
            parts = [state[idx] for state in states]
            if any(part is None for part in parts):
                return None
            concatenated.append(mx.concatenate(parts, axis=seq_axis))
        return tuple(concatenated) if isinstance(states[0], tuple) else concatenated

    def get_cache_for_generation(
        self,
        request_id: str,
    ) -> Tuple[Optional[List[Any]], bool]:
        """
        Get cache data for generation, applying COW if needed.

        Args:
            request_id: Request identifier

        Returns:
            Tuple of (cache_data, was_copied)
        """
        entry = self._request_tables.get(request_id)
        if not entry:
            return None, False

        # Get blocks with COW
        blocks, was_copied = self.paged_cache.get_blocks_for_generation(
            entry.block_table
        )

        if was_copied:
            # Deep copy cache data for modified blocks
            cache_data = copy.deepcopy(entry.cache_data)
        else:
            cache_data = entry.cache_data

        entry.last_access = time.time()
        return cache_data, was_copied

    def release_cache(self, request_id: str) -> None:
        """
        Release cache blocks for a completed request.

        Drops the request's references (also those of a table that only came
        from :meth:`fetch_cache`). Blocks nobody references any more stay
        cached, evictable least recently used first, until the pool needs them.

        Args:
            request_id: Request identifier
        """
        self._request_tables.pop(request_id, None)
        if self.paged_cache.get_block_table(request_id) is not None:
            self.paged_cache.delete_block_table(request_id)
            logger.debug(f"Released cache for {request_id}")

    def fork_cache(
        self,
        source_request_id: str,
        new_request_id: str,
    ) -> Optional[BlockTable]:
        """
        Fork cache from one request to another (COW).

        Args:
            source_request_id: Source request ID
            new_request_id: New request ID

        Returns:
            Forked BlockTable, or None if source not found
        """
        source_entry = self._request_tables.get(source_request_id)
        if not source_entry:
            return None

        # Fork block table (increments ref counts)
        forked_table = self.paged_cache.fork_block_table(
            source_entry.block_table,
            new_request_id,
        )

        # Create new entry with reference to same cache data
        self._request_tables[new_request_id] = BlockCacheEntry(
            block_table=forked_table,
            cache_data=source_entry.cache_data,  # Shared reference
            last_access=time.time(),
        )

        logger.debug(f"Forked cache: {source_request_id} -> {new_request_id}")

        return forked_table

    def reconstruct_cache(
        self,
        block_table: BlockTable,
    ) -> Optional[List[Any]]:
        """
        Reconstruct cache objects from stored block tensor data.

        Sequence-backed caches are concatenated block-by-block. Recurrent
        caches such as ArraysCache are restored from the latest sequence
        boundary snapshot that was actually stored.

        Args:
            block_table: BlockTable containing block IDs to reconstruct from

        Returns:
            List of reconstructed KVCache objects (one per layer),
            or None if reconstruction fails
        """
        if not block_table or not block_table.block_ids:
            return None

        if not HAS_MLX:
            logger.warning("Cannot reconstruct cache: MLX not available")
            return None

        try:
            # Collect cache data from all blocks
            all_block_data = []
            for block_id in block_table.block_ids:
                block = self.paged_cache.allocated_blocks.get(block_id)
                if not block:
                    logger.warning(f"Block {block_id} not found in allocated blocks")
                    return None

                if block.cache_data is None:
                    logger.debug(f"Block {block_id} has no tensor data stored")
                    return None

                all_block_data.append(block.cache_data)

            if not all_block_data:
                return None

            # Every block must describe the same layers
            num_layers = len(all_block_data[0])
            if num_layers == 0 or any(len(d) != num_layers for d in all_block_data):
                return None

            reconstructed_caches = []
            for layer_idx in range(num_layers):
                layer_entries = [block_data[layer_idx] for block_data in all_block_data]

                # The restored layer must describe the state after the LAST
                # block. A sequence-backed layer needs every block's slice (a
                # missing one would shift every later position); a recurrent
                # layer's snapshot is only valid at the token where it was
                # taken, so it must come from the last block, never from an
                # earlier one (that would pair the KV of N tokens with the
                # recurrent state of fewer).
                layer_meta = layer_entries[-1]
                if layer_meta is None:
                    return None

                state = layer_meta["state"]
                if layer_meta["storage"] == "concat":
                    if any(entry is None for entry in layer_entries):
                        return None
                    state = self._concat_cache_states(
                        [entry["state"] for entry in layer_entries],
                        layer_meta["seq_axis"],
                    )

                if state is None:
                    return None

                cache_cls = layer_meta.get("class_ref")
                meta_state = layer_meta.get("meta_state")

                if cache_cls is not None and hasattr(cache_cls, "from_state"):
                    cache = restore_from_state(cache_cls, state, meta_state)
                else:
                    from mlx_lm.models.cache import KVCache

                    if len(state) != 2:
                        return None
                    cache = KVCache()
                    cache.keys, cache.values = state
                    seq_axis = self._cache_state_seq_axis(state)
                    if seq_axis is None:
                        return None
                    cache.offset = cache.keys.shape[seq_axis]

                # A layer that carries its own position (a rotating window
                # rebuilt from the meta_state of the whole stored sequence)
                # must land exactly where the table says the prefix ends.
                offset = getattr(cache, "offset", None)
                if isinstance(offset, int) and offset != block_table.num_tokens:
                    logger.debug(
                        f"Reconstructed layer {layer_idx} is at offset {offset}, "
                        f"not {block_table.num_tokens}: not restorable"
                    )
                    return None

                reconstructed_caches.append(cache)

            if not reconstructed_caches:
                return None

            logger.debug(
                f"Reconstructed cache: {len(reconstructed_caches)} layers, "
                f"{block_table.num_tokens} tokens from {len(block_table.block_ids)} blocks"
            )

            return reconstructed_caches

        except Exception as e:
            logger.warning(f"Failed to reconstruct cache: {e}")
            import traceback

            logger.debug(traceback.format_exc())
            return None

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        paged_stats = self.paged_cache.get_memory_usage()
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": (
                self._hits / (self._hits + self._misses)
                if (self._hits + self._misses) > 0
                else 0
            ),
            "tokens_saved": self._tokens_saved,
            "active_requests": len(self._request_tables),
            **paged_stats,
        }

    def reset_stats(self) -> None:
        """Reset statistics."""
        self._hits = 0
        self._misses = 0
        self._tokens_saved = 0
        self.paged_cache.reset_stats()

    def clear(self) -> None:
        """Clear all cached data."""
        self._request_tables.clear()
        self.paged_cache.clear()
        self.reset_stats()

    def __len__(self) -> int:
        """Return number of active request entries."""
        return len(self._request_tables)
