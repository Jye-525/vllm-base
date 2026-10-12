# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import math
import os
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

from vllm.distributed.kv_events import (
    MEDIUM_GPU,
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    KVCacheEvent,
)
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashList,
    BlockHashListWithBlockSize,
    BlockHashWithGroupId,
    ExternalBlockHash,
    FreeKVCacheBlockQueue,
    KVCacheBlock,
    generate_block_hash_extra_keys,
    get_block_hash,
    get_group_id,
    make_block_hash_with_group_id,
    maybe_convert_block_hash,
)
from vllm.v1.core.tier_logger import _get_tier_logger
from vllm.v1.request import Request

logger = init_logger(__name__)

class BlockHashToBlockMap:
    """
    Cache of blocks that are used for prefix caching. It caches blocks
    from hash directly to a block or multiple blocks
    (i.e. {block_hash: KVCacheBlocks})
    - Mostly block_hash maps to a single KVCacheBlock, and KVCacheBlocks
        would simply be a KVCacheBlock.
    - Otherwise, KVCacheBlocks is a dict from {block_id: KVCacheBlock}

    A cached block is a full block with a block hash that can be used
    for prefix caching.
    The cached block may be used by running requests or in the
    free_block_queue that could potentially be evicted.

    NOTE #1: We currently don't de-duplicate the blocks in the cache,
    meaning that if a block becomes full and is cached, we don't check
    if there is already an identical block in the cache. This is because
    we want to make sure the allocated block IDs won't change so that
    block tables are append-only.
    NOTE #2: The union type is introduced in order to reduce GC costs
    from the inner dict.
    """

    def __init__(self):
        self._cache: dict[
            BlockHashWithGroupId, KVCacheBlock | dict[int, KVCacheBlock]
        ] = {}

    def get_one_block(self, key: BlockHashWithGroupId) -> KVCacheBlock | None:
        """
        Gets any block with the given block hash key.
        """
        blocks = self._cache.get(key)
        if blocks is not None:
            if isinstance(blocks, KVCacheBlock):
                return blocks
            if isinstance(blocks, dict):
                return next(iter(blocks.values()))
            self._unexpected_blocks_type(blocks)
        return None

    def insert(self, key: BlockHashWithGroupId, block: KVCacheBlock) -> None:
        """
        Inserts the KVCacheBlock to the cache
        """
        blocks = self._cache.get(key)
        if blocks is None:
            # When key is not found, attach a single block to the key
            self._cache[key] = block
        elif isinstance(blocks, KVCacheBlock):
            # If there's a block with the same key, merge the original block
            # and the new block into a dict
            self._cache[key] = {blocks.block_id: blocks, block.block_id: block}
        elif isinstance(blocks, dict):
            # If it's already a dict, simply insert the block
            blocks[block.block_id] = block
        else:
            self._unexpected_blocks_type(blocks)

    def pop(self, key: BlockHashWithGroupId, block_id: int) -> KVCacheBlock | None:
        """
        Checks if block_hash exists and pop block_id from the cache
        """
        blocks = self._cache.pop(key, None)
        if blocks is None:
            # block_hash not found in the cache
            return None
        # TODO(Jialin): If key is found, block_id should always present
        # in blocks. We currently keep the original behaviour for safety.
        #
        # Will add block_id == blocks.block_id assertion and
        # use del blocks[block_id] instead as followup.
        if isinstance(blocks, KVCacheBlock):
            if blocks.block_id == block_id:
                return blocks
            # If the single block ID doesn't match, we should put the
            # block back (it should happen rarely)
            self._cache[key] = blocks
            return None
        if isinstance(blocks, dict):
            # Try to pop block_id from the block dict, and if dict still
            # contain blocks, put back to the cache.
            block = blocks.pop(block_id, None)
            if len(blocks) > 0:
                self._cache[key] = blocks
            return block
        self._unexpected_blocks_type(blocks)
        return None

    def __len__(self) -> int:
        return len(self._cache)

    def _unexpected_blocks_type(self, blocks: Any) -> None:
        raise AssertionError(f"Invalid KV cache block type {type(blocks)}")


class BlockPool:
    """BlockPool that manages KVCacheBlocks.
    It provides methods to allocate, free and cache the kv cache blocks. The
    free_block_queue stores the free blocks in eviction order to enable
    allocation, free, and cache eviction. The cached_block_hash_to_block
    maps between block hash and cached block to support finding cached blocks
    by their block hash.

    Args:
        num_gpu_blocks: The number of blocks in the pool.
        enable_caching: Whether to enable prefix caching.
        hash_block_size: The block size of which the block hashes are computed.
            The actual block size usually equals hash_block_size, but in cases
            where different KV cache groups have different block sizes, the
            actual block size can be a multiple of hash_block_size.
        enable_kv_cache_events: Whether to enable kv cache events.
        metrics_collector: Optional metrics collector for tracking block residency.
    """

    def __init__(
        self,
        num_gpu_blocks: int,
        enable_caching: bool,
        hash_block_size: int,
        enable_kv_cache_events: bool = False,
        metrics_collector: KVCacheMetricsCollector | None = None,
    ):
        assert isinstance(num_gpu_blocks, int) and num_gpu_blocks > 0
        self.num_gpu_blocks = num_gpu_blocks
        self.enable_caching = enable_caching
        self.hash_block_size = hash_block_size
        # All kv-cache blocks.
        self.blocks: list[KVCacheBlock] = [
            KVCacheBlock(idx) for idx in range(num_gpu_blocks)
        ]
        # Free block queue that constructs and manipulates a doubly linked
        # list of free blocks (including eviction candidates when caching is
        # enabled).
        self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)

        # Cache for block lookup
        self.cached_block_hash_to_block: BlockHashToBlockMap = BlockHashToBlockMap()

        # To represent a placeholder block with block_id=0.
        # The ref_cnt of null_block is not maintained, needs special care to
        # avoid freeing it.
        self.null_block = self.free_block_queue.popleft()
        self.null_block.is_null = True

        self.enable_kv_cache_events = enable_kv_cache_events
        self.kv_event_queue: list[KVCacheEvent] = []

        self.metrics_collector = metrics_collector

        # Optional observational hook (survival diagnostics). It must provide
        # on_allocate(num_blocks) and on_evict(raw_block_hash) and must not
        # change allocation or eviction decisions.
        self.observer: Any | None = None

        # Raw hashes whose last cached copy was evicted since the last drain.
        # None (the default) records nothing; the scheduler turns it on only
        # while re-ranking waiting requests per admission.
        self._eviction_log: list[BlockHash] | None = None
        # Cached blocks whose hash was dropped to reuse them (cumulative); the
        # scheduler logs the per-step difference.
        self.num_evicted_cached_blocks = 0

        # Peek eviction uses waiting-queue-aware victim selection; LRU uses
        # stock popleft_n. The scheduler configures this at startup.
        self._eviction_policy: str = "lru"

        # Waiting-queue demand maintained incrementally by the
        # scheduler. Higher counts mean that more pending requests share the
        # cumulative prefix ending at that hash. An empty dict degenerates
        # Peek eviction to plain LRU.
        self._demand_counts: dict[BlockHash, float] = {}
        self._demand_max_depths: dict[BlockHash, int] = {}
        self._block_hash_last_access: dict[BlockHash, float] = {}
        self._peek_eviction_mode: str = "plain"
        self._peek_eviction_recency_weight: float = 100.0
        self._peek_eviction_decay_tau: float = 30.0
        self._peek_eviction_scan_cap: int = 0

        # Orphan tracking is intentionally disabled for the demand-aware
        # experiment. vLLM already frees request blocks in reverse order, so
        # suffix blocks normally become eviction candidates before prefix
        # blocks. Keeping an explicit orphan heuristic adds stale-state risk
        # and can conflict with Peek's demand priority.
        #
        # OrphanKey = tuple[BlockHash, int]
        # self._children: dict[OrphanKey, set[OrphanKey]] = defaultdict(set)
        # self._orphaned: set[OrphanKey] = set()

        # Peek-eviction effectiveness counters. Pure observability —
        # they do not influence eviction.
        self.peek_hits: int = 0
        # >0 while scheduler ranking probes the cache; such lookups are not
        # real hits and must not inflate peek_hits.
        self._probe_depth: int = 0
        self.peek_protected: int = 0
        self.peek_publishes: int = 0

        self._eviction_trace_path = os.getenv("VLLM_EVICTION_TRACE_PATH")
        self._eviction_trace_seq = 0
        self._eviction_trace_context: dict[str, Any] = {}

    def set_eviction_trace_context(
        self,
        *,
        request_id: str | None = None,
        num_new_tokens: int | None = None,
        num_computed_tokens: int | None = None,
    ) -> None:
        if not self._eviction_trace_path:
            return
        self._eviction_trace_context = {
            "request_id": request_id,
            "num_new_tokens": num_new_tokens,
            "num_computed_tokens": num_computed_tokens,
        }

    def clear_eviction_trace_context(self) -> None:
        self._eviction_trace_context = {}

    def _trace_block_hash(self, block: KVCacheBlock) -> str:
        if block.block_hash is None:
            return ""
        return bytes(get_block_hash(block.block_hash)).hex()

    def _trace_eviction_selection(
        self,
        num_blocks: int,
        victims: list[KVCacheBlock],
        *,
        walked: int | None = None,
    ) -> None:
        if not self._eviction_trace_path:
            return
        if not any(block.block_hash is not None for block in victims):
            return
        self._eviction_trace_seq += 1
        rows = []
        for idx, block in enumerate(victims):
            raw_hash = (
                get_block_hash(block.block_hash)
                if block.block_hash is not None else None
            )
            demand = (
                self._demand_counts.get(raw_hash, 0.0)
                if raw_hash is not None else 0.0
            )
            priority = (
                self._get_peek_eviction_priority(raw_hash)
                if raw_hash is not None else 0.0
            )
            if block.block_hash is None:
                tier = "uncached"
            elif self._eviction_policy == "peek" and demand > 0.0:
                tier = "demanded"
            elif self._eviction_policy == "peek":
                tier = "zero_demand"
            else:
                tier = "lru_cached"
            rows.append({
                "victim_index": idx,
                "block_id": block.block_id,
                "block_hash": self._trace_block_hash(block),
                "tier": tier,
                "demand": demand,
                "priority": priority,
                "ref_cnt": block.ref_cnt,
            })
        record = {
            "source": "vllm",
            "event": "eviction_selection",
            "event_index": self._eviction_trace_seq,
            "policy": self._eviction_policy,
            "requested_blocks": num_blocks,
            "selected_blocks": len(victims),
            "walked_blocks": walked,
            "free_blocks_before_reuse": self.free_block_queue.num_free_blocks,
            "peek_protected": self.peek_protected,
            "peek_publishes": self.peek_publishes,
            **self._eviction_trace_context,
            "victims": rows,
        }
        trace_dir = os.path.dirname(self._eviction_trace_path)
        if trace_dir:
            os.makedirs(trace_dir, exist_ok=True)
        with open(self._eviction_trace_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")

    def set_peek_demand_counts(
        self,
        counts: dict[BlockHash, float],
        max_depths: Mapping[BlockHash, int] | None = None,
    ) -> None:
        """Publish per-block demand counts used by PEEK eviction modes.

        Pass an empty dict to fall back to plain LRU ordering (e.g. when
        the waiting queue is empty).
        """
        self._demand_counts = counts
        self._demand_max_depths = (
            {h: int(depth) for h, depth in max_depths.items() if h in counts}
            if max_depths is not None else {}
        )
        self.peek_protected = len(counts)
        self.peek_publishes += 1

    def update_peek_demand_counts(
        self,
        counts: Mapping[BlockHash, int],
        *,
        max_depths: Mapping[BlockHash, int] | None = None,
    ) -> None:
        """Incrementally update eviction state from absolute demand counts.

        ``counts`` only needs to contain hashes changed by a waiting-request
        insertion, removal, or path update. ``max_depths`` supplies PEEK's
        lazily maintained deepest chain position for cluster mode.
        """
        if not counts:
            return
        for block_hash, count in counts.items():
            if count <= 0:
                self._demand_counts.pop(block_hash, None)
                self._demand_max_depths.pop(block_hash, None)
            else:
                self._demand_counts[block_hash] = float(count)
                if max_depths is not None:
                    depth = max_depths.get(block_hash)
                    if depth is not None:
                        self._demand_max_depths[block_hash] = int(depth)
        self.peek_protected = len(self._demand_counts)
        self.peek_publishes += 1

    def _get_peek_eviction_priority(
        self,
        block_hash: BlockHash,
        now: float | None = None,
    ) -> float:
        """Return PEEK's mode-specific protection priority for a hash."""
        demand = self._demand_counts.get(block_hash, 0.0)
        if demand <= 0.0:
            return 0.0
        if self._peek_eviction_mode == "cluster":
            depth = self._demand_max_depths.get(block_hash, 1)
            return demand * (1 + depth)
        if self._peek_eviction_mode == "recency":
            last_access = self._block_hash_last_access.get(block_hash)
            current_time = now if now is not None else time.time()
            age = max(0.0, current_time - last_access) if last_access else 0.0
            return float(int(demand - self._peek_eviction_recency_weight * age))
        if self._peek_eviction_mode == "decay":
            last_access = self._block_hash_last_access.get(block_hash)
            current_time = now if now is not None else time.time()
            age = max(0.0, current_time - last_access) if last_access else 0.0
            decay = (
                math.exp(-age / self._peek_eviction_decay_tau)
                if self._peek_eviction_decay_tau > 0.0 else 1.0
            )
            return float(int(demand * decay * 1000))
        return demand

    def get_cached_block(
        self, block_hash: BlockHash, kv_cache_group_ids: list[int]
    ) -> list[KVCacheBlock] | None:
        """Get the cached block by the block hash for each group in
        `kv_cache_group_ids`, or None if cache miss for any group.
        If there are duplicated blocks, we return the first block in the cache.

        Args:
            block_hash: The hash value of the block.
            kv_cache_group_ids: The ids of the KV cache groups.

        Returns:
            The cached blocks if exists, or None.
        """
        cached_blocks = []
        for group_id in kv_cache_group_ids:
            block_hash_with_group_id = make_block_hash_with_group_id(
                block_hash, group_id
            )
            block = self.cached_block_hash_to_block.get_one_block(
                block_hash_with_group_id
            )
            if not block:
                return None
            cached_blocks.append(block)
        if not self._probe_depth and block_hash in self._demand_counts:
            self.peek_hits += 1
        return cached_blocks

    @contextmanager
    def probing(self) -> Iterator[None]:
        """Treat get_cached_block() calls in this scope as ranking probes."""
        self._probe_depth += 1
        try:
            yield
        finally:
            self._probe_depth -= 1

    def start_eviction_log(self) -> None:
        self._eviction_log = []

    def drain_eviction_log(self) -> list[BlockHash]:
        """Return and clear the hashes evicted since the last drain."""
        evicted = self._eviction_log or []
        if self._eviction_log is not None:
            self._eviction_log = []
        return evicted

    def stop_eviction_log(self) -> None:
        self._eviction_log = None

    def is_cached_block(self, block_hash: BlockHash,
                        kv_cache_group_ids: list[int]) -> bool:
        """Return whether the block hash exists for all requested groups.

        Unlike get_cached_block(), this helper is for scheduling/ranking probes
        and does not update any cache-hit counters.
        """
        for group_id in kv_cache_group_ids:
            block_hash_with_group_id = make_block_hash_with_group_id(
                block_hash, group_id)
            if not self.cached_block_hash_to_block.get_one_block(
                    block_hash_with_group_id):
                return False
        return True

    def cache_full_blocks(
        self,
        request: Request,
        blocks: list[KVCacheBlock],
        num_cached_blocks: int,
        num_full_blocks: int,
        block_size: int,
        kv_cache_group_id: int,
        block_mask: list[bool] | None = None,
    ) -> None:
        """Cache a list of full blocks for prefix caching.
        This function takes a list of blocks that will have their block hash
        metadata to be updated and cached. Given a request, it updates the
        metadata for each block and caching it in the
        `cached_block_hash_to_block`.
        The block hashes values are computed by the Request object immediately
        when it is created and when new tokens are appended.

        Args:
            request: The request to cache the blocks.
            blocks: All blocks in the request.
            num_cached_blocks: The number of blocks that are already cached.
            num_full_blocks: The number of blocks that are full and should
                be cached after this function.
            block_size: Number of tokens in each block.
            kv_cache_group_id: The id of the KV cache group.
            block_mask: Optional mask aligned with
                ``blocks[num_cached_blocks:num_full_blocks]``. When provided,
                blocks where the mask is False are skipped (treated like null
                blocks). Used by groups whose ``find_longest_cache_hit`` only
                consults a subset of blocks (e.g. SWA tail-window), so blocks
                that can never serve a hit stay out of the prefix-cache hash
                map.
        """
        if num_cached_blocks >= num_full_blocks:
            return
        new_full_blocks = blocks[num_cached_blocks:num_full_blocks]
        assert len(request.block_hashes) >= num_full_blocks
        assert block_mask is None or len(block_mask) == len(new_full_blocks)
        if block_size == self.hash_block_size:
            # Common case.
            block_hashes: BlockHashList = request.block_hashes
        else:
            # block_size is a multiple of hash_block_size. This happens when
            # different KV cache groups have different block sizes.
            assert block_size % self.hash_block_size == 0
            # Recalculate block_hashes at the granularity of block_size, using
            # the original block_hashes (at the granularity of hash_block_size).
            block_hashes = BlockHashListWithBlockSize(
                request.block_hashes, self.hash_block_size, block_size
            )

        new_block_hashes = block_hashes[num_cached_blocks:]
        new_hashes: list[ExternalBlockHash] | None = (
            [] if self.enable_kv_cache_events else None
        )
        track_last_access = self._peek_eviction_mode in ("recency", "decay")
        now = time.time() if track_last_access else 0.0
        for i, blk in enumerate(new_full_blocks):
            # Some blocks may be null or masked out when enabling sparse attention
            # like sliding window attention, or Mamba models with prefix-caching
            # in align mode. We skip null blocks here.
            if blk.is_null or (block_mask is not None and not block_mask[i]):
                continue
            assert blk.block_hash is None
            block_hash = new_block_hashes[i]

            # Update and added the full block to the cache.
            block_hash_with_group_id = make_block_hash_with_group_id(
                block_hash, kv_cache_group_id
            )
            blk.block_hash = block_hash_with_group_id
            self.cached_block_hash_to_block.insert(block_hash_with_group_id, blk)
            if track_last_access:
                self._block_hash_last_access[block_hash] = now

            # Orphan tracking disabled; keep this as historical reference.
            # block_key = (block_hash, kv_cache_group_id)
            # self._orphaned.discard(block_key)

            # Register parent->child link for per-group orphan tracking.
            # Disabled for the demand-aware experiment.
            j = num_cached_blocks + i
            # if j > 0:
            #     parent_hash = block_hashes[j - 1]
            #     parent_key = (parent_hash, kv_cache_group_id)
            #     self._children[parent_key].add(block_key)

            # --- EXP 2 HOOK: GPU store ---
            if (_tlog := _get_tier_logger()):
                _parent_hash = block_hashes[j - 1] if j > 0 else None
                _tlog.log_event(
                    "gpu_store", block_hash, "gpu",
                    parent_hash=_parent_hash,
                    request_id=request.request_id,
                )
            # --- END EXP 2 HOOK ---

            if new_hashes is not None:
                new_hashes.append(maybe_convert_block_hash(block_hash))

        if self.enable_kv_cache_events:
            if num_cached_blocks == 0:
                parent_block_hash: ExternalBlockHash | None = None
            else:
                parent_block_hash = maybe_convert_block_hash(
                    block_hashes[num_cached_blocks - 1]
                )

            # Calculate token range for the blocks being cached
            start_token_idx = num_cached_blocks * block_size
            end_token_idx = num_full_blocks * block_size

            # Generate extra keys for each block individually.
            # Each block may have different extra_keys (e.g., different MM
            # features, or cache_salt only for the first block).
            # Skip null/masked-out blocks to match the length of new_hashes.
            extra_keys_list: list[tuple[Any, ...] | None] = []
            curr_mm_idx = 0
            for i in range(num_cached_blocks, num_full_blocks):
                if blocks[i].is_null:
                    continue
                if block_mask is not None and not block_mask[i - num_cached_blocks]:
                    continue
                block_start = i * block_size
                block_end = block_start + block_size
                extra_keys, curr_mm_idx = generate_block_hash_extra_keys(
                    request, block_start, block_end, curr_mm_idx
                )
                extra_keys_list.append(extra_keys)

            self.kv_event_queue.append(
                BlockStored(
                    block_hashes=new_hashes,
                    parent_block_hash=parent_block_hash,
                    token_ids=request.all_token_ids[start_token_idx:end_token_idx],
                    block_size=block_size,
                    lora_id=request.lora_request.adapter_id
                    if request.lora_request
                    else None,
                    medium=MEDIUM_GPU,
                    lora_name=request.lora_request.name
                    if request.lora_request
                    else None,
                    extra_keys=extra_keys_list if extra_keys_list else None,
                    group_idx=kv_cache_group_id,
                )
            )

    def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
        """Get new blocks from the free block pool.

        Note that we do not check block cache in this function.

        Under Peek eviction, positive-demand blocks are
        protected first because the current waiting queue can reuse them.
        Blocks with no published demand are cheap victims and are evicted in
        LRU order. If those cheap victims are insufficient, demanded blocks
        are evicted by ascending priority. Under "lru" (the baseline), blocks
        are evicted in plain free-queue order.

        Args:
            num_blocks: The number of blocks to allocate.

        Returns:
            A list of new block.
        """
        if num_blocks > self.get_num_free_blocks():
            raise ValueError(
                f"Cannot get {num_blocks} free blocks from the pool")
        if self.observer is not None:
            self.observer.on_allocate(num_blocks)

        if self.enable_caching and self._eviction_policy == "peek":
            ret = self._select_victims_peek(num_blocks)
        else:
            ret = self.free_block_queue.popleft_n(num_blocks)
            self._trace_eviction_selection(num_blocks, ret, walked=num_blocks)

        # In order to only iterate the list once, we duplicated code a bit
        if self.enable_caching:
            for block in ret:
                self._maybe_evict_cached_block(block)
                assert block.ref_cnt == 0
                block.ref_cnt += 1
                if self.metrics_collector:
                    self.metrics_collector.on_block_allocated(block)
        else:
            for block in ret:
                assert block.ref_cnt == 0
                block.ref_cnt += 1
                if self.metrics_collector:
                    self.metrics_collector.on_block_allocated(block)
        return ret

    def _select_victims_peek(
        self, num_blocks: int
    ) -> list[KVCacheBlock]:
        """Select victims using Peek's bounded demand-aware free-list scan.

        Uncached and zero-demand blocks are selected first in LRU order.
        If the bounded scan finds too few, demanded blocks in the scanned
        prefix are selected by ascending mode-specific protection priority,
        with LRU position breaking ties. This mirrors Peek's vLLM hook.
        """
        t0 = time.perf_counter()
        queue = self.free_block_queue
        if not self._demand_counts:
            victims = queue.popleft_n(num_blocks)
            self._trace_eviction_selection(num_blocks, victims, walked=num_blocks)
            return victims

        cap = (
            self._peek_eviction_scan_cap
            if self._peek_eviction_scan_cap > 0
            else max(num_blocks * 4, 256)
        )
        victims: list[KVCacheBlock] = []
        demanded: list[tuple[float, int, KVCacheBlock]] = []
        block = queue.fake_free_list_head.next_free_block
        tail = queue.fake_free_list_tail
        walked = 0
        while (
            block is not None
            and block is not tail
            and len(victims) < num_blocks
            and walked < cap
        ):
            block_hash = block.block_hash
            if block_hash is None:
                victims.append(block)
            else:
                raw_hash = get_block_hash(block_hash)
                demand = self._demand_counts.get(raw_hash, 0.0)
                if demand <= 0.0:
                    victims.append(block)
                else:
                    priority = self._get_peek_eviction_priority(raw_hash)
                    demanded.append((priority, walked, block))
            block = block.next_free_block
            walked += 1

        if len(victims) < num_blocks:
            demanded.sort(key=lambda item: (item[0], item[1]))
            needed = num_blocks - len(victims)
            victims.extend(block for _, _, block in demanded[:needed])

        if len(victims) < num_blocks:
            # Defensive fallback for a corrupted or unexpectedly short scan.
            return queue.popleft_n(num_blocks)

        for block in victims:
            queue.remove(block)

        self._trace_eviction_selection(num_blocks, victims, walked=walked)

        elapsed_us = (time.perf_counter() - t0) * 1e6
        logger.debug(
            "peek_eviction: %.1f us | requested=%d walked=%d cap=%d "
            "demanded_seen=%d free_remaining=%d protected=%d hits=%d "
            "publishes=%d",
            elapsed_us,
            num_blocks,
            walked,
            cap,
            len(demanded),
            queue.num_free_blocks,
            self.peek_protected,
            self.peek_hits,
            self.peek_publishes,
        )

        return victims

    def _maybe_evict_cached_block(self, block: KVCacheBlock) -> bool:
        """
        If a block is cached in `cached_block_hash_to_block`, we reset its hash
        metadata and evict it from the cache.

        Args:
            block: The block to evict.

        Returns:
            True if the block is evicted, False otherwise.
        """
        # Clean up metrics tracking first to prevent leaks
        if self.metrics_collector:
            self.metrics_collector.on_block_evicted(block)

        block_hash = block.block_hash
        if block_hash is None:
            # The block doesn't have hash, eviction is not needed
            return False

        if self.cached_block_hash_to_block.pop(block_hash, block.block_id) is None:
            # block not found in cached_block_hash_to_block,
            # eviction is not needed
            return False

        # Blocks are not de-duplicated: report a hash only when its last
        # cached copy goes, i.e. when the content stops being reusable.
        if (
            self.observer is not None or self._eviction_log is not None
        ) and self.cached_block_hash_to_block.get_one_block(block_hash) is None:
            raw_hash = get_block_hash(block_hash)
            if self.observer is not None:
                self.observer.on_evict(raw_hash)
            if self._eviction_log is not None:
                self._eviction_log.append(raw_hash)

        self.num_evicted_cached_blocks += 1
        block.reset_hash()

        # Orphan tracking disabled for demand-aware eviction. Historical
        # implementation kept per-group parent->child state here, but the
        # heuristic can become stale and conflict with Peek demand.
        # raw_hash = get_block_hash(block_hash)
        # group_id = get_group_id(block_hash)
        # block_key = (raw_hash, group_id)
        # self._orphaned.discard(block_key)
        # for child in self._children.get(block_key, set()):
        #     self._orphaned.add(child)
        # self._children.pop(block_key, None)

        # --- EXP 2 HOOK: GPU evict ---
        if (_tlog := _get_tier_logger()):
            _tlog.log_event("gpu_evict", get_block_hash(block_hash), "gpu")
        # --- END EXP 2 HOOK ---

        if self.enable_kv_cache_events:
            self.kv_event_queue.append(
                BlockRemoved(
                    block_hashes=[maybe_convert_block_hash(get_block_hash(block_hash))],
                    medium=MEDIUM_GPU,
                    group_idx=get_group_id(block_hash),
                )
            )
        return True

    def touch(self, blocks: Sequence[KVCacheBlock]) -> None:
        """Touch a block increases its reference count by 1, and may remove
        the block from the free queue. This is used when a block is hit by
        another request with the same prefix.

        Args:
            blocks: A list of blocks to touch.
        """
        track_last_access = self._peek_eviction_mode in ("recency", "decay")
        now = time.time() if track_last_access else 0.0
        for block in blocks:
            if track_last_access and block.block_hash is not None:
                self._block_hash_last_access[get_block_hash(block.block_hash)] = now
            # ref_cnt=0 means this block is in the free list (i.e. eviction
            # candidate), so remove it.
            if block.ref_cnt == 0 and not block.is_null:
                self.free_block_queue.remove(block)
            block.ref_cnt += 1
            if self.metrics_collector:
                self.metrics_collector.on_block_accessed(block)

    def free_blocks(self, ordered_blocks: Iterable[KVCacheBlock]) -> None:
        """Free a list of blocks. The blocks should be ordered by their
        eviction priority, where the first block will be evicted first.

        Args:
            ordered_blocks: A list of blocks to free ordered by their eviction
                priority.
        """
        # Materialize the iterable to allow multiple passes.
        blocks_list = list(ordered_blocks)
        for block in blocks_list:
            block.ref_cnt -= 1
        self.free_block_queue.append_n(
            [block for block in blocks_list if block.ref_cnt == 0 and not block.is_null]
        )

    def evict_blocks(self, block_ids: set[int]) -> None:
        """evict blocks from the prefix cache by their block IDs.

        only evicts blocks that are currently cached (have a hash). blocks
        with ref_cnt > 0 are not freed from the block pool, only evicted
        from the prefix cache hash table.

        Args:
            block_ids: Set of block IDs to evict from cache.
        """
        for block_id in block_ids:
            assert block_id < len(self.blocks), (
                f"Invalid block_id {block_id} >= {len(self.blocks)}. "
                f"This indicates a bug in the KV connector - workers should "
                f"only report block IDs that were allocated by the scheduler."
            )
            block = self.blocks[block_id]
            self._maybe_evict_cached_block(block)

    def reset_prefix_cache(self) -> bool:
        """Reset prefix cache. This function may be used in RLHF
        flows to invalid prefix caching after the weights are updated,
        or used for resetting prefix caching status for benchmarking.

        Returns:
            bool: True if the prefix cache is successfully reset,
            False otherwise.
        """
        num_used_blocks = self.num_gpu_blocks - self.get_num_free_blocks()
        if num_used_blocks != 1:  # The null block is always marked as used
            logger.warning(
                "Failed to reset prefix cache because some "
                "blocks (%d) are not freed yet",
                num_used_blocks - 1,
            )
            return False

        # Remove all hashes so that no new blocks will hit.
        self.cached_block_hash_to_block = BlockHashToBlockMap()

        # Clear published Peek demand. Orphan tracking is disabled.
        # self._children.clear()
        # self._orphaned.clear()
        self._demand_counts = {}
        self._demand_max_depths = {}
        self._block_hash_last_access = {}

        # Remove all hashes from all blocks.
        for block in self.blocks:
            block.reset_hash()

        if self.metrics_collector:
            self.metrics_collector.reset()

        logger.info("Successfully reset prefix cache")

        if self.enable_kv_cache_events:
            self.kv_event_queue.append(AllBlocksCleared())

        return True

    def get_num_free_blocks(self) -> int:
        """Get the number of free blocks in the pool.

        Returns:
            The number of free blocks.
        """
        return self.free_block_queue.num_free_blocks

    def get_usage(self) -> float:
        """Get the KV cache usage.

        Returns:
            The KV cache usage (between 0.0 and 1.0).
        """

        # Subtract 1 to account for null block.
        total_gpu_blocks = self.num_gpu_blocks - 1
        if not total_gpu_blocks:
            return 0
        return 1.0 - (self.get_num_free_blocks() / total_gpu_blocks)

    def take_events(self) -> list[KVCacheEvent]:
        """Atomically takes all events and clears the queue.

        Returns:
            A list of KV cache events.
        """
        if not self.enable_kv_cache_events:
            return []
        events = self.kv_event_queue
        self.kv_event_queue = []
        return events
