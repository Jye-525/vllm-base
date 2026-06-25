# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import os
import time
from collections.abc import Iterable, Sequence
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

        # Eviction policy: "score" uses waiting-queue-aware victim
        # selection (_select_victims_scored); "lru" uses stock popleft_n
        # (the baseline). Set by the scheduler at startup; defaults to
        # plain LRU so the pool behaves like stock vLLM until told
        # otherwise.
        self._eviction_policy: str = "lru"

        # Waiting-queue demand scores, published by the scheduler once
        # per schedule() step. score[h] = sum gamma**p over the
        # positions p of waiting requests whose contiguous cached prefix
        # includes h. Higher = reused sooner / by more waiters = keep.
        # An empty dict degenerates scored eviction to plain LRU.
        self._demand_scores: dict[BlockHash, float] = {}
        self._evictable_hash_counts: dict[BlockHash, int] = {}

        # Orphan tracking is intentionally disabled for the demand-aware
        # experiment. vLLM already frees request blocks in reverse order, so
        # suffix blocks normally become eviction candidates before prefix
        # blocks. Keeping an explicit orphan heuristic adds stale-state risk
        # and can conflict with the demand score.
        #
        # OrphanKey = tuple[BlockHash, int]
        # self._children: dict[OrphanKey, set[OrphanKey]] = defaultdict(set)
        # self._orphaned: set[OrphanKey] = set()

        # Scored-eviction effectiveness counters. Pure observability —
        # they do not influence eviction.
        self.scored_hits: int = 0          # hits on positively-scored blocks
        self.scored_protected: int = 0     # size of last published score dict
        self.scored_publishes: int = 0

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
            score = (
                self._demand_scores.get(raw_hash, 0.0)
                if raw_hash is not None else 0.0
            )
            if block.block_hash is None:
                tier = "uncached"
            elif self._eviction_policy == "score" and score > 0.0:
                tier = "demanded"
            elif self._eviction_policy == "score":
                tier = "zero_demand"
            else:
                tier = "lru_cached"
            rows.append({
                "victim_index": idx,
                "block_id": block.block_id,
                "block_hash": self._trace_block_hash(block),
                "tier": tier,
                "score": score,
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
            "scored_protected": self.scored_protected,
            "scored_publishes": self.scored_publishes,
            **self._eviction_trace_context,
            "victims": rows,
        }
        trace_dir = os.path.dirname(self._eviction_trace_path)
        if trace_dir:
            os.makedirs(trace_dir, exist_ok=True)
        with open(self._eviction_trace_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")

    def _mark_evictable(self, block_hash: BlockHashWithGroupId | None) -> None:
        if block_hash is None:
            return
        raw_hash = get_block_hash(block_hash)
        self._evictable_hash_counts[raw_hash] = (
            self._evictable_hash_counts.get(raw_hash, 0) + 1)

    def _unmark_evictable(self, block_hash: BlockHashWithGroupId | None) -> None:
        if block_hash is None:
            return
        raw_hash = get_block_hash(block_hash)
        count = self._evictable_hash_counts.get(raw_hash, 0)
        if count <= 1:
            self._evictable_hash_counts.pop(raw_hash, None)
        else:
            self._evictable_hash_counts[raw_hash] = count - 1

    def is_hash_evictable(self, block_hash: BlockHash) -> bool:
        return self._evictable_hash_counts.get(block_hash, 0) > 0

    def set_demand_scores(self, scores: dict[BlockHash, float]) -> None:
        """Publish the per-block demand scores used by scored eviction.

        Pass an empty dict to fall back to plain LRU ordering (e.g. when
        the waiting queue is empty).
        """
        self._demand_scores = scores
        self.scored_protected = len(scores)
        self.scored_publishes += 1

    def classify_chain(
        self,
        hashes: Sequence[BlockHash],
        kv_cache_group_ids: Sequence[int] = (0,),
    ) -> tuple[int, int, list[BlockHash]]:
        """Single-pass classifier for a waiter's prefix chain.

        Returns (cached_prefix_len, active_hits, evictable_hashes):
          cached_prefix_len: length of the longest *contiguous* prefix of
                             `hashes` for which every hash is cached in
                             every group in `kv_cache_group_ids`. vLLM v1
                             only reuses contiguous prefixes, so this is
                             the safe truncation point for protection.
          active_hits:       count of hashes *within* the contiguous
                             cached prefix whose block (in the first
                             group) has ref_cnt > 0 — cached and held by
                             a running request, therefore NOT in the
                             free queue. Used by the scheduler to
                             discount free-queue cost. Hashes past the
                             first miss are excluded because the waiter
                             will allocate fresh blocks for those
                             positions regardless of what is cached
                             downstream (contiguous-only matching).
          evictable_hashes:  hashes within the contiguous cached prefix that
                             have at least one cached block with ref_cnt == 0.
                             Demand scores only affect free-queue eviction
                             candidates, so publishing pinned-only hashes adds
                             scheduler overhead without changing victim choice.
        """
        cached_prefix_len = 0
        active_hits = 0
        first_group = kv_cache_group_ids[0]
        cache = self.cached_block_hash_to_block
        evictable_hashes: list[BlockHash] = []
        for h in hashes:
            block_in_first_group: KVCacheBlock | None = None
            cached_all_groups = True
            for gid in kv_cache_group_ids:
                key = make_block_hash_with_group_id(h, gid)
                blk = cache.get_one_block(key)
                if gid == first_group:
                    block_in_first_group = blk
                if blk is None:
                    cached_all_groups = False
                    break
            if not cached_all_groups:
                break
            cached_prefix_len += 1
            if (block_in_first_group is not None
                    and block_in_first_group.ref_cnt > 0):
                active_hits += 1
            if self.is_hash_evictable(h):
                evictable_hashes.append(h)
        return cached_prefix_len, active_hits, evictable_hashes

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
        if block_hash in self._demand_scores:
            self.scored_hits += 1
        return cached_blocks

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
            if blk.ref_cnt == 0:
                self._mark_evictable(block_hash_with_group_id)

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

        Under the "score" eviction policy, positive-demand blocks are
        protected first because the current waiting queue can reuse them.
        Blocks with no published demand are cheap victims and are evicted in
        LRU order. If those cheap victims are insufficient, demanded blocks
        are evicted by ascending score. Under "lru" (the baseline), blocks
        are evicted in plain free-queue order.

        Args:
            num_blocks: The number of blocks to allocate.

        Returns:
            A list of new block.
        """
        if num_blocks > self.get_num_free_blocks():
            raise ValueError(
                f"Cannot get {num_blocks} free blocks from the pool")

        if self.enable_caching and self._eviction_policy == "score":
            ret = self._select_victims_scored(num_blocks)
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

    def _select_victims_scored(
        self, num_blocks: int
    ) -> list[KVCacheBlock]:
        """Select ``num_blocks`` eviction victims using the published
        per-block demand scores fused with LRU history.

        Eviction priority (evict first -> last):
          * "Cheap" blocks — uncached, zero-demand (no waiting request
            needs them) — taken in LRU order (free-queue head -> tail).
          * Demanded blocks (score > 0) — only under pressure, in
            ascending score order (lowest protected-value first); LRU
            breaks exact ties.

        The free list is traversed once from the head (LRU order) without
        mutating it. Cheap victims (tiers 1-2) are collected in queue
        order; if they alone satisfy the request the walk stops early and
        demanded blocks are never disturbed (common fast path). Only under
        pressure do we sort the demanded candidates and evict the
        lowest-scoring ones. Chosen victims are unlinked with
        ``FreeKVCacheBlockQueue.remove()`` so survivors keep their LRU
        position.
        """
        t0 = time.perf_counter()
        queue = self.free_block_queue
        scores = self._demand_scores

        victims: list[KVCacheBlock] = []
        # (block, score) for cached blocks with positive demand, in LRU order.
        demanded: list[tuple[KVCacheBlock, float]] = []
        zero_demand_cached = 0
        uncached_seen = 0

        # Traverse head -> tail. The fake tail has next_free_block is None.
        walked = 0
        block = queue.fake_free_list_head.next_free_block
        while (block is not None and block.next_free_block is not None
               and len(victims) < num_blocks):
            nxt = block.next_free_block
            walked += 1
            bh = block.block_hash
            if bh is None:
                # Uncached — safe to reuse immediately.
                uncached_seen += 1
                victims.append(block)
            else:
                raw_hash = get_block_hash(bh)
                score = scores.get(raw_hash, 0.0)
                if score > 0.0:
                    demanded.append((block, score))  # tier 3 candidate
                else:
                    zero_demand_cached += 1
                    victims.append(block)            # tier 1: zero demand (LRU)
            block = nxt

        # Tier 3: only consumed when cheap victims were insufficient. In
        # that case the loop above ran to the tail, so `demanded` holds
        # every demanded block. Sort ascending by score (stable sort keeps
        # LRU order among ties) and evict the lowest-scoring ones.
        still_needed = num_blocks - len(victims)
        used_demanded = 0
        demanded_victim_scores: list[float] = []
        if still_needed > 0 and demanded:
            demanded.sort(key=lambda t: t[1])
            take_with_scores = demanded[:still_needed]
            take = [blk for blk, _ in take_with_scores]
            demanded_victim_scores = [score for _, score in take_with_scores]
            victims.extend(take)
            used_demanded = len(take)

        # Unlink victims; survivors retain their LRU position.
        for blk in victims:
            queue.remove(blk)

        self._trace_eviction_selection(num_blocks, victims, walked=walked)

        if demanded:
            demanded_scores = [score for _, score in demanded]
            demanded_min = min(demanded_scores)
            demanded_avg = sum(demanded_scores) / len(demanded_scores)
            demanded_max = max(demanded_scores)
        else:
            demanded_min = demanded_avg = demanded_max = 0.0
        if demanded_victim_scores:
            victim_score_min = min(demanded_victim_scores)
            victim_score_avg = (
                sum(demanded_victim_scores) / len(demanded_victim_scores))
            victim_score_max = max(demanded_victim_scores)
        else:
            victim_score_min = victim_score_avg = victim_score_max = 0.0

        elapsed_us = (time.perf_counter() - t0) * 1e6
        logger.info(
            "scored_eviction: %.1f us | requested=%d walked=%d "
            "demanded_seen=%d demanded_evicted=%d free_remaining=%d "
            "uncached_seen=%d zero_demand_cached=%d "
            "demanded_score_min=%.6f demanded_score_avg=%.6f "
            "demanded_score_max=%.6f victim_score_min=%.6f "
            "victim_score_avg=%.6f victim_score_max=%.6f "
            "scored_protected=%d scored_hits=%d publishes=%d",
            elapsed_us, num_blocks, walked,
            len(demanded), used_demanded, queue.num_free_blocks,
            uncached_seen, zero_demand_cached, demanded_min, demanded_avg,
            demanded_max, victim_score_min, victim_score_avg,
            victim_score_max,
            self.scored_protected, self.scored_hits, self.scored_publishes,
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

        if block.ref_cnt == 0:
            self._unmark_evictable(block_hash)
        block.reset_hash()

        # Orphan tracking disabled for demand-aware eviction. Historical
        # implementation kept per-group parent->child state here, but the
        # heuristic can become stale and conflict with demand scores.
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
        for block in blocks:
            # ref_cnt=0 means this block is in the free list (i.e. eviction
            # candidate), so remove it.
            if block.ref_cnt == 0 and not block.is_null:
                self.free_block_queue.remove(block)
                self._unmark_evictable(block.block_hash)
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
            was_in_use = block.ref_cnt > 0
            block.ref_cnt -= 1
            if (was_in_use and block.ref_cnt == 0 and not block.is_null
                    and block.block_hash is not None):
                self._mark_evictable(block.block_hash)
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

        # Clear published demand scores. Orphan tracking is disabled.
        # self._children.clear()
        # self._orphaned.clear()
        self._demand_scores = {}
        self._evictable_hash_counts = {}

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
