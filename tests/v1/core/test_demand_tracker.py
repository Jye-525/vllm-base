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

    # NOTE: test_orphaned_blocks_evicted_first removed — orphan probe (EXP 4)
    # was intentionally not ported from the v0.15.0-era KVReuse fork.
