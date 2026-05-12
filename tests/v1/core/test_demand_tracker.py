# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for DemandTracker and demand-aware eviction in BlockPool."""

import pytest

from vllm.v1.core.demand_tracker import DemandTracker
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    FreeKVCacheBlockQueue,
    KVCacheBlock,
    make_block_hash_with_group_id,
)
from vllm.v1.core.block_pool import BlockPool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bh(tag: bytes) -> BlockHash:
    """Create a BlockHash from raw bytes."""
    return BlockHash(tag)


def _bhg(tag: bytes, group_id: int = 0):
    """Create a BlockHashWithGroupId for assigning to KVCacheBlock."""
    return make_block_hash_with_group_id(_bh(tag), group_id)


# ---------------------------------------------------------------------------
# DemandTracker unit tests
# ---------------------------------------------------------------------------

class TestDemandTracker:

    def test_empty_tracker(self):
        t = DemandTracker()
        assert t.get_demand(_bh(b"x")) == 0
        assert not t.has_demand(_bh(b"x"))
        assert t.num_tracked_hashes == 0

    def test_add_single_request(self):
        t = DemandTracker()
        hashes = [_bh(b"a"), _bh(b"b"), _bh(b"c")]
        t.add_request(hashes)

        for h in hashes:
            assert t.get_demand(h) == 1
            assert t.has_demand(h)
        assert t.num_tracked_hashes == 3

    def test_shared_prefix(self):
        """Two requests sharing prefix blocks should show demand=2."""
        t = DemandTracker()
        req1 = [_bh(b"a"), _bh(b"b"), _bh(b"c")]
        req2 = [_bh(b"a"), _bh(b"b"), _bh(b"d")]

        t.add_request(req1)
        t.add_request(req2)

        assert t.get_demand(_bh(b"a")) == 2  # shared
        assert t.get_demand(_bh(b"b")) == 2  # shared
        assert t.get_demand(_bh(b"c")) == 1  # req1 only
        assert t.get_demand(_bh(b"d")) == 1  # req2 only

    def test_remove_request(self):
        t = DemandTracker()
        req1 = [_bh(b"a"), _bh(b"b")]
        req2 = [_bh(b"a"), _bh(b"b")]

        t.add_request(req1)
        t.add_request(req2)
        t.remove_request(req1)

        assert t.get_demand(_bh(b"a")) == 1
        assert t.get_demand(_bh(b"b")) == 1

    def test_remove_cleans_up(self):
        """After removing all requests, hashes should be fully cleaned up."""
        t = DemandTracker()
        hashes = [_bh(b"a"), _bh(b"b")]
        t.add_request(hashes)
        t.remove_request(hashes)

        assert t.get_demand(_bh(b"a")) == 0
        assert not t.has_demand(_bh(b"a"))
        assert t.num_tracked_hashes == 0

    def test_remove_nonexistent_is_safe(self):
        """Removing hashes that were never added should not raise."""
        t = DemandTracker()
        t.remove_request([_bh(b"z")])
        assert t.get_demand(_bh(b"z")) == 0

    def test_many_requests(self):
        t = DemandTracker()
        common = _bh(b"sys_prompt")
        for i in range(100):
            t.add_request([common, _bh(f"unique_{i}".encode())])

        assert t.get_demand(common) == 100
        assert t.num_tracked_hashes == 101  # 1 common + 100 unique

    def test_add_remove_interleaved(self):
        t = DemandTracker()
        h = _bh(b"block")

        t.add_request([h])
        t.add_request([h])
        assert t.get_demand(h) == 2

        t.remove_request([h])
        assert t.get_demand(h) == 1

        t.add_request([h])
        assert t.get_demand(h) == 2

        t.remove_request([h])
        t.remove_request([h])
        assert t.get_demand(h) == 0
        assert t.num_tracked_hashes == 0


# ---------------------------------------------------------------------------
# BlockPool demand-aware eviction integration tests
# ---------------------------------------------------------------------------

class TestDemandAwareEviction:
    """Test that BlockPool._pop_blocks_demand_aware prefers evicting
    zero-demand blocks over positive-demand blocks."""

    def _make_pool(self, num_blocks: int = 10) -> BlockPool:
        tracker = DemandTracker()
        pool = BlockPool(
            num_gpu_blocks=num_blocks,
            enable_caching=True,
            hash_block_size=16,
            demand_tracker=tracker,
        )
        return pool

    def _cache_block(self, pool: BlockPool, block: KVCacheBlock,
                     tag: bytes, group_id: int = 0):
        """Simulate caching a block: assign a hash and register in cache map."""
        bhg = _bhg(tag, group_id)
        block.block_hash = bhg
        pool.cached_block_hash_to_block.insert(bhg, block)

    def test_evicts_zero_demand_before_positive(self):
        """With mixed demand, zero-demand blocks should be evicted first."""
        pool = self._make_pool(6)
        tracker = pool.demand_tracker

        # Allocate all blocks (leaves null_block allocated, 5 free)
        # The free queue has blocks[1..5]
        blocks = pool.free_block_queue.popleft_n(5)
        assert pool.free_block_queue.num_free_blocks == 0

        # Cache all blocks and free them back (simulates cached eviction
        # candidates)
        for i, block in enumerate(blocks):
            self._cache_block(pool, block, f"block_{i}".encode())
            block.ref_cnt = 0
            pool.free_block_queue.append(block)

        # Mark blocks 0,1 as demanded by a waiting request
        demanded_hashes = [_bh(b"block_0"), _bh(b"block_1")]
        tracker.add_request(demanded_hashes)

        # Request 2 new blocks — should get zero-demand blocks first
        new_blocks = pool._pop_blocks_demand_aware(2)
        assert len(new_blocks) == 2

        # The evicted blocks should NOT be the demanded ones
        evicted_tags = set()
        for b in new_blocks:
            if b.block_hash is not None:
                from vllm.v1.core.kv_cache_utils import get_block_hash
                evicted_tags.add(get_block_hash(b.block_hash))

        assert _bh(b"block_0") not in evicted_tags
        assert _bh(b"block_1") not in evicted_tags

    def test_falls_back_to_demanded_when_needed(self):
        """When all free blocks have demand, they should still be evicted."""
        pool = self._make_pool(4)
        tracker = pool.demand_tracker

        # 3 free blocks (block 0 is null)
        blocks = pool.free_block_queue.popleft_n(3)

        for i, block in enumerate(blocks):
            self._cache_block(pool, block, f"d_{i}".encode())
            block.ref_cnt = 0
            pool.free_block_queue.append(block)

        # All blocks have demand
        tracker.add_request([_bh(b"d_0"), _bh(b"d_1"), _bh(b"d_2")])

        # Should still succeed — falls back to demanded blocks
        new_blocks = pool._pop_blocks_demand_aware(2)
        assert len(new_blocks) == 2

    def test_no_demand_tracker_uses_regular_eviction(self):
        """Without demand tracker, BlockPool uses standard popleft_n."""
        pool = BlockPool(
            num_gpu_blocks=6,
            enable_caching=True,
            hash_block_size=16,
            demand_tracker=None,
        )
        # Should use regular path without error
        blocks = pool.get_new_blocks(2)
        assert len(blocks) == 2

    def test_uncached_blocks_evicted_first(self):
        """Blocks without hashes (not cached) should be evicted immediately."""
        pool = self._make_pool(6)
        tracker = pool.demand_tracker

        # Pop 3 blocks, cache 2 of them, leave 1 uncached
        blocks = pool.free_block_queue.popleft_n(3)

        # Cache blocks 0 and 1 with demand
        self._cache_block(pool, blocks[0], b"cached_0")
        self._cache_block(pool, blocks[1], b"cached_1")
        tracker.add_request([_bh(b"cached_0"), _bh(b"cached_1")])

        # Block 2 stays uncached (no block_hash)
        # Free all back
        for b in blocks:
            b.ref_cnt = 0
            pool.free_block_queue.append(b)

        # Eviction should pick the uncached block first
        new_blocks = pool._pop_blocks_demand_aware(1)
        assert len(new_blocks) == 1
        assert new_blocks[0].block_hash is None  # the uncached one

    def test_orphaned_blocks_evicted_first(self):
        """Orphaned blocks (parent evicted) should be evicted before
        zero-demand cached blocks."""
        pool = self._make_pool(6)
        tracker = pool.demand_tracker

        blocks = pool.free_block_queue.popleft_n(3)

        # Cache all 3 blocks
        for i, block in enumerate(blocks):
            self._cache_block(pool, block, f"o_{i}".encode())
            block.ref_cnt = 0
            pool.free_block_queue.append(block)

        # Mark block 1 as orphaned
        pool._orphaned.add(_bh(b"o_1"))

        # Evict 1 block — should be the orphaned one
        new_blocks = pool._pop_blocks_demand_aware(1)
        assert len(new_blocks) == 1

        from vllm.v1.core.kv_cache_utils import get_block_hash
        evicted_hash = get_block_hash(new_blocks[0].block_hash)
        assert evicted_hash == _bh(b"o_1")


# ---------------------------------------------------------------------------
# Imminent-aware eviction (top-K waiters protected over far-future waiters)
# ---------------------------------------------------------------------------

class TestImminentEviction:
    """When the imminent set is published, demanded blocks NOT in it
    should be evicted before demanded blocks that are."""

    def _make_pool(self, num_blocks: int = 10) -> BlockPool:
        tracker = DemandTracker()
        return BlockPool(
            num_gpu_blocks=num_blocks,
            enable_caching=True,
            hash_block_size=16,
            demand_tracker=tracker,
        )

    def _cache(self, pool: BlockPool, block: KVCacheBlock, tag: bytes):
        bhg = _bhg(tag)
        block.block_hash = bhg
        # Use the real BlockHashToBlockMap API.
        pool.cached_block_hash_to_block.insert(bhg, block)

    def test_empty_imminent_matches_old_behavior(self):
        """Without an imminent set, all demanded blocks form a single
        deferred band and the fallback walks them in queue order."""
        pool = self._make_pool(5)
        tracker = pool.demand_tracker

        blocks = pool.free_block_queue.popleft_n(3)
        for i, b in enumerate(blocks):
            self._cache(pool, b, f"x_{i}".encode())
            b.ref_cnt = 0
            pool.free_block_queue.append(b)

        # All demanded — fallback path will run.
        tracker.add_request([_bh(b"x_0"), _bh(b"x_1"), _bh(b"x_2")])
        assert pool._imminent_hashes == set()

        evicted = pool._pop_blocks_demand_aware(1)
        assert len(evicted) == 1
        # Without imminent split, fallback popleft_n returns the head
        # of the queue (oldest, x_0).
        from vllm.v1.core.kv_cache_utils import get_block_hash
        assert get_block_hash(evicted[0].block_hash) == _bh(b"x_0")

    def test_imminent_protects_top_k_under_pressure(self):
        """When all candidates are demanded, the fallback must evict
        a band-3 (far) block before any band-4 (imminent) block."""
        pool = self._make_pool(5)
        tracker = pool.demand_tracker

        blocks = pool.free_block_queue.popleft_n(3)
        for i, b in enumerate(blocks):
            self._cache(pool, b, f"y_{i}".encode())
            b.ref_cnt = 0
            pool.free_block_queue.append(b)

        # All three are globally demanded.
        tracker.add_request([_bh(b"y_0"), _bh(b"y_1"), _bh(b"y_2")])
        # But only y_1 and y_2 are imminent (top-K waiter's prefix).
        # y_0 is at the head of the queue (oldest) — without the
        # imminent split it would be picked first; with the split it
        # must NOT be picked because y_0 is imminent. Wait: assertion
        # below — we mark y_1 as the *only* imminent one to make the
        # test unambiguous.
        pool.set_imminent_hashes({_bh(b"y_1")})

        evicted = pool._pop_blocks_demand_aware(1)
        assert len(evicted) == 1
        # y_0 and y_2 are band 3 (not in imminent); y_1 is band 4.
        # Fallback walks band 3 first in queue order: y_0 is at head.
        from vllm.v1.core.kv_cache_utils import get_block_hash
        evicted_hash = get_block_hash(evicted[0].block_hash)
        assert evicted_hash != _bh(b"y_1"), (
            "imminent block was evicted while a far block was available"
        )
        assert evicted_hash in {_bh(b"y_0"), _bh(b"y_2")}

    def test_imminent_band4_only_when_no_band3(self):
        """When every demanded block is imminent, fallback still
        succeeds and returns one of them (last-resort eviction)."""
        pool = self._make_pool(5)
        tracker = pool.demand_tracker

        blocks = pool.free_block_queue.popleft_n(2)
        for i, b in enumerate(blocks):
            self._cache(pool, b, f"z_{i}".encode())
            b.ref_cnt = 0
            pool.free_block_queue.append(b)

        tracker.add_request([_bh(b"z_0"), _bh(b"z_1")])
        pool.set_imminent_hashes({_bh(b"z_0"), _bh(b"z_1")})

        evicted = pool._pop_blocks_demand_aware(1)
        assert len(evicted) == 1  # last-resort path still works

    def test_zero_demand_still_evicted_first_with_imminent_set(self):
        """Imminent split applies only to demanded blocks; band-2
        (zero-demand) eviction must still take precedence."""
        pool = self._make_pool(6)
        tracker = pool.demand_tracker

        blocks = pool.free_block_queue.popleft_n(3)
        for i, b in enumerate(blocks):
            self._cache(pool, b, f"m_{i}".encode())
            b.ref_cnt = 0
            pool.free_block_queue.append(b)

        # Only m_2 is demanded, and it's imminent.
        tracker.add_request([_bh(b"m_2")])
        pool.set_imminent_hashes({_bh(b"m_2")})

        evicted = pool._pop_blocks_demand_aware(1)
        assert len(evicted) == 1
        from vllm.v1.core.kv_cache_utils import get_block_hash
        # Must be a zero-demand block, never m_2.
        evicted_hash = get_block_hash(evicted[0].block_hash)
        assert evicted_hash in {_bh(b"m_0"), _bh(b"m_1")}


# ---------------------------------------------------------------------------
# classify_chain: per-waiter contiguous-prefix + active-hits classifier
# ---------------------------------------------------------------------------

class TestClassifyChain:
    """BlockPool.classify_chain returns (cached_prefix_len, active_hits)
    in one pass over a waiter's block_hashes."""

    def _make_pool(self, num_blocks: int = 8) -> BlockPool:
        return BlockPool(
            num_gpu_blocks=num_blocks,
            enable_caching=True,
            hash_block_size=16,
            demand_tracker=DemandTracker(),
        )

    def _cache(self, pool: BlockPool, block: KVCacheBlock, tag: bytes,
               ref_cnt: int = 0):
        bhg = _bhg(tag)
        block.block_hash = bhg
        block.ref_cnt = ref_cnt
        pool.cached_block_hash_to_block.insert(bhg, block)

    def test_all_uncached_returns_zero(self):
        pool = self._make_pool()
        hashes = [_bh(b"a"), _bh(b"b"), _bh(b"c")]
        prefix_len, active = pool.classify_chain(hashes)
        assert prefix_len == 0
        assert active == 0

    def test_full_idle_prefix(self):
        """All cached, all ref_cnt == 0 → full prefix, zero active."""
        pool = self._make_pool()
        blocks = pool.free_block_queue.popleft_n(3)
        tags = [b"a", b"b", b"c"]
        for blk, tag in zip(blocks, tags):
            self._cache(pool, blk, tag, ref_cnt=0)
        prefix_len, active = pool.classify_chain([_bh(t) for t in tags])
        assert prefix_len == 3
        assert active == 0

    def test_full_active_prefix(self):
        """All cached, all ref_cnt > 0 → full prefix, all active."""
        pool = self._make_pool()
        blocks = pool.free_block_queue.popleft_n(3)
        tags = [b"a", b"b", b"c"]
        for blk, tag in zip(blocks, tags):
            self._cache(pool, blk, tag, ref_cnt=1)
        prefix_len, active = pool.classify_chain([_bh(t) for t in tags])
        assert prefix_len == 3
        assert active == 3

    def test_gap_truncates_prefix(self):
        """A missing block ends the contiguous prefix; downstream
        cached blocks do NOT extend prefix_len or active_hits."""
        pool = self._make_pool()
        blocks = pool.free_block_queue.popleft_n(3)
        # Cache positions 0, 1, 3 (skip 2).
        self._cache(pool, blocks[0], b"h0", ref_cnt=0)
        self._cache(pool, blocks[1], b"h1", ref_cnt=1)
        # blocks[2] is "h3" — cached and active but past the gap.
        self._cache(pool, blocks[2], b"h3", ref_cnt=1)
        chain = [_bh(b"h0"), _bh(b"h1"), _bh(b"h2"), _bh(b"h3"), _bh(b"h4")]
        prefix_len, active = pool.classify_chain(chain)
        # Prefix stops at h2; h3 cached-active beyond is ignored.
        assert prefix_len == 2
        assert active == 1  # only h1 is active and within prefix

    def test_mixed_idle_and_active_within_prefix(self):
        pool = self._make_pool()
        blocks = pool.free_block_queue.popleft_n(4)
        self._cache(pool, blocks[0], b"x0", ref_cnt=2)  # active
        self._cache(pool, blocks[1], b"x1", ref_cnt=0)  # idle
        self._cache(pool, blocks[2], b"x2", ref_cnt=3)  # active
        self._cache(pool, blocks[3], b"x3", ref_cnt=0)  # idle
        chain = [_bh(b"x0"), _bh(b"x1"), _bh(b"x2"), _bh(b"x3")]
        prefix_len, active = pool.classify_chain(chain)
        assert prefix_len == 4
        assert active == 2

    def test_empty_chain(self):
        pool = self._make_pool()
        prefix_len, active = pool.classify_chain([])
        assert prefix_len == 0
        assert active == 0


# ---------------------------------------------------------------------------
# Imminent-protection effectiveness counters
# ---------------------------------------------------------------------------

class TestImminentCounters:
    """Counters that track whether published imminent protections
    actually paid off as get_cached_block hits."""

    def _make_pool(self) -> BlockPool:
        return BlockPool(
            num_gpu_blocks=6,
            enable_caching=True,
            hash_block_size=16,
            demand_tracker=DemandTracker(),
        )

    def test_publish_increments_counters(self):
        pool = self._make_pool()
        assert pool.imminent_publishes == 0
        assert pool.imminent_total_protected == 0
        pool.set_imminent_hashes({_bh(b"a"), _bh(b"b")})
        assert pool.imminent_publishes == 1
        assert pool.imminent_total_protected == 2
        pool.set_imminent_hashes(set())
        assert pool.imminent_publishes == 2
        assert pool.imminent_total_protected == 0

    def test_hit_increments_only_when_protected(self):
        pool = self._make_pool()
        blocks = pool.free_block_queue.popleft_n(2)
        bhg_a = _bhg(b"a")
        bhg_b = _bhg(b"b")
        blocks[0].block_hash = bhg_a
        pool.cached_block_hash_to_block.insert(bhg_a, blocks[0])
        blocks[1].block_hash = bhg_b
        pool.cached_block_hash_to_block.insert(bhg_b, blocks[1])

        pool.set_imminent_hashes({_bh(b"a")})
        assert pool.imminent_hits == 0

        # Hit on a protected hash → counter increments.
        result = pool.get_cached_block(_bh(b"a"), [0])
        assert result is not None
        assert pool.imminent_hits == 1

        # Hit on an unprotected hash → no change.
        result = pool.get_cached_block(_bh(b"b"), [0])
        assert result is not None
        assert pool.imminent_hits == 1

        # Miss on a protected hash → no change (only successful lookups count).
        result = pool.get_cached_block(_bh(b"never_cached"), [0])
        assert result is None
        assert pool.imminent_hits == 1
