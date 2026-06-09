# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for waiting-queue-aware (demand-score) GPU prefix-cache eviction.

Covers:
  * compute_demand_scores      — additive mixture scoring (position, reuse,
                                  chain-loss depth, epsilon cutoff)
  * BlockPool._select_victims_scored — tiered victim selection (cheap blocks
                                  first in LRU order, demanded blocks by
                                  ascending score, survivor LRU preserved)
  * BlockPool.get_new_blocks    — "score" vs "lru" policy routing
"""

import pytest

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.eviction_scorer import compute_demand_scores
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    KVCacheBlock,
    get_block_hash,
    make_block_hash_with_group_id,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bh(tag: bytes) -> BlockHash:
    return BlockHash(tag)


def _bhg(tag: bytes, group_id: int = 0):
    return make_block_hash_with_group_id(_bh(tag), group_id)


def _raw(block: KVCacheBlock) -> BlockHash:
    return get_block_hash(block.block_hash)


def _make_pool(num_blocks: int, policy: str = "score") -> BlockPool:
    pool = BlockPool(
        num_gpu_blocks=num_blocks,
        enable_caching=True,
        hash_block_size=16,
    )
    pool._eviction_policy = policy
    return pool


def _cache(pool: BlockPool, block: KVCacheBlock, tag: bytes) -> None:
    """Assign a hash to a free block, register it in the cache, ref_cnt=0."""
    bhg = _bhg(tag)
    block.block_hash = bhg
    pool.cached_block_hash_to_block.insert(bhg, block)
    block.ref_cnt = 0


def _pool_with_cached(tags: list[bytes]) -> tuple[BlockPool, dict]:
    """Build a pool whose free queue holds one cached block per tag, in the
    given order (head == tags[0] == least-recently-used). num_gpu_blocks is
    sized so no extra free blocks linger."""
    pool = _make_pool(len(tags) + 1)  # +1 for the null block
    blocks = pool.free_block_queue.popleft_n(len(tags))  # queue now empty
    by_tag = {}
    for blk, tag in zip(blocks, tags):
        _cache(pool, blk, tag)
        pool.free_block_queue.append(blk)  # re-append in tag order (LRU order)
        by_tag[tag] = blk
    return pool, by_tag


# ---------------------------------------------------------------------------
# compute_demand_scores — pure scoring formula
# ---------------------------------------------------------------------------

A, B, C, D, E = b"A", b"B", b"C", b"D", b"E"
# Waiters: R0->{A,B,C}, R1->{A,B}, R2->{A,D}, R3->{E}  (queue order)
_AE_PREFIXES = [[A, B, C], [A, B], [A, D], [E]]


class TestComputeDemandScores:

    def test_empty_queue(self):
        assert compute_demand_scores([], 0.9) == {}

    def test_ae_example_additive_default(self):
        s = compute_demand_scores(_AE_PREFIXES, 0.9)
        assert s[A] == pytest.approx(1.0)
        assert s[B] == pytest.approx((1.0 + 2 / 3 + 2 / 3) / 3)
        assert s[C] == pytest.approx((1.0 + 1 / 3 + 1 / 3) / 3)
        assert s[D] == pytest.approx((0.81 + 1 / 3 + 0.5) / 3)
        assert s[E] == pytest.approx((0.729 + 1 / 3 + 1.0) / 3)

    def test_position_weight_only_is_nearest_reuse(self):
        s = compute_demand_scores(
            _AE_PREFIXES,
            0.9,
            position_weight=1.0,
            reuse_weight=0.0,
            depth_weight=0.0,
        )
        assert s[A] == pytest.approx(1.0)
        assert s[B] == pytest.approx(1.0)
        assert s[C] == pytest.approx(1.0)
        assert s[D] == pytest.approx(0.81)
        assert s[E] == pytest.approx(0.729)

    def test_reuse_weight_only_is_normalized_frequency(self):
        s = compute_demand_scores(
            _AE_PREFIXES,
            0.9,
            position_weight=0.0,
            reuse_weight=1.0,
            depth_weight=0.0,
        )
        assert s[A] == pytest.approx(1.0)
        assert s[B] == pytest.approx(2 / 3)
        assert s[C] == pytest.approx(1 / 3)
        assert s[D] == pytest.approx(1 / 3)
        assert s[E] == pytest.approx(1 / 3)

    def test_chain_loss_depth_weight_only(self):
        s = compute_demand_scores(
            [[A, B, C]],
            0.9,
            position_weight=0.0,
            reuse_weight=0.0,
            depth_weight=1.0,
        )
        assert s[A] == pytest.approx(1.0)
        assert s[B] == pytest.approx(2 / 3)
        assert s[C] == pytest.approx(1 / 3)

    def test_raw_chain_loss_depth_mode(self):
        s = compute_demand_scores(
            [[A, B, C]],
            0.9,
            position_weight=0.0,
            reuse_weight=0.0,
            depth_weight=1.0,
            depth_mode="chain_loss",
        )
        assert s[A] == pytest.approx(3.0)
        assert s[B] == pytest.approx(2.0)
        assert s[C] == pytest.approx(1.0)

    def test_empty_prefix_still_advances_position(self):
        """A waiter with no cached prefix consumes its position so later
        waiters keep the correct gamma ** p weight."""
        s = compute_demand_scores(
            [[B], [], [A]],
            0.5,
            position_weight=1.0,
            reuse_weight=0.0,
            depth_weight=0.0,
        )
        assert s[A] == pytest.approx(0.25)  # pos 2 (not pos 1)

    def test_epsilon_cutoff_bounds_positions(self):
        """Once gamma ** p < epsilon, far waiters stop contributing."""
        prefixes = [[bytes([i])] for i in range(100)]  # unique hash per pos
        s = compute_demand_scores(prefixes, 0.5, epsilon=1e-3)
        # 0.5**9 = 0.00195 >= 1e-3 ; 0.5**10 = 0.00097 < 1e-3 -> positions 0..9
        assert len(s) == 10


# ---------------------------------------------------------------------------
# BlockPool._select_victims_scored — tiered victim selection
# ---------------------------------------------------------------------------

class TestSelectVictimsScored:

    def test_zero_demand_evicted_before_demanded(self):
        pool, _ = _pool_with_cached([A, B, C])
        pool.set_demand_scores({_bh(A): 2.0, _bh(B): 1.0})  # C has no demand
        victims = pool._select_victims_scored(1)
        assert len(victims) == 1
        assert _raw(victims[0]) == _bh(C)

    def test_lowest_score_evicted_among_demanded(self):
        pool, _ = _pool_with_cached([A, B, C])
        pool.set_demand_scores({_bh(A): 3.0, _bh(B): 1.0, _bh(C): 2.0})
        victims = pool._select_victims_scored(1)
        assert _raw(victims[0]) == _bh(B)  # lowest score, even mid-queue

    def test_orphaned_is_cheap_demanded_protected(self):
        """An orphaned block (always score 0) is evicted before a demanded
        block that sits earlier in the queue."""
        pool, _ = _pool_with_cached([A, B])  # A first (LRU), B last
        pool.set_demand_scores({_bh(A): 5.0})  # A demanded; B has no demand
        pool._orphaned.add(_bh(B))
        victims = pool._select_victims_scored(1)
        assert _raw(victims[0]) == _bh(B)

    def test_uncached_block_evicted_first(self):
        pool = _make_pool(4)  # null + 3 blocks
        b1, b2, b3 = pool.free_block_queue.popleft_n(3)
        _cache(pool, b1, A)
        _cache(pool, b2, B)
        b3.ref_cnt = 0  # b3 stays uncached (no hash)
        for blk in (b1, b2, b3):  # queue order: A, B, uncached
            pool.free_block_queue.append(blk)
        pool.set_demand_scores({_bh(A): 2.0, _bh(B): 1.0})
        victims = pool._select_victims_scored(1)
        assert victims[0].block_hash is None  # the uncached block

    def test_survivors_keep_lru_order(self):
        pool, _ = _pool_with_cached([A, B, C, D])  # queue: A B C D
        pool.set_demand_scores({_bh(A): 5.0})  # only A demanded
        victims = pool._select_victims_scored(2)
        # cheap victims taken in LRU order: B then C; A deferred, stop at 2.
        assert [_raw(v) for v in victims] == [_bh(B), _bh(C)]
        remaining = pool.free_block_queue.get_all_free_blocks()
        assert [_raw(b) for b in remaining] == [_bh(A), _bh(D)]
        assert pool.free_block_queue.num_free_blocks == 2

    def test_falls_back_to_demanded_under_pressure(self):
        """When every free block is demanded, the lowest scorers are still
        evicted so the allocation can proceed."""
        pool, _ = _pool_with_cached([A, B, C])
        pool.set_demand_scores({_bh(A): 3.0, _bh(B): 1.0, _bh(C): 2.0})
        victims = pool._select_victims_scored(2)
        assert len(victims) == 2
        assert {_raw(v) for v in victims} == {_bh(B), _bh(C)}  # two lowest

    def test_empty_scores_is_plain_lru(self):
        """No published scores -> every cached block is cheap -> pure LRU."""
        pool, _ = _pool_with_cached([A, B, C])
        pool.set_demand_scores({})
        victims = pool._select_victims_scored(2)
        assert [_raw(v) for v in victims] == [_bh(A), _bh(B)]  # queue head order


# ---------------------------------------------------------------------------
# BlockPool.get_new_blocks — policy routing
# ---------------------------------------------------------------------------

class TestGetNewBlocksPolicy:

    def test_lru_policy_uses_popleft_order(self):
        pool = _make_pool(5, policy="lru")
        ids_before = [b.block_id
                      for b in pool.free_block_queue.get_all_free_blocks()]
        new = pool.get_new_blocks(2)
        assert [b.block_id for b in new] == ids_before[:2]  # stock LRU head

    def test_score_policy_protects_demanded_via_public_api(self):
        pool = _make_pool(4)  # score policy
        b1, b2, b3 = pool.free_block_queue.popleft_n(3)
        _cache(pool, b1, A)        # A demanded
        b2.ref_cnt = 0             # b2 uncached
        pool.free_block_queue.append(b1)  # queue: A, uncached
        pool.free_block_queue.append(b2)
        pool.set_demand_scores({_bh(A): 5.0})
        new = pool.get_new_blocks(1)
        assert new[0].block_id == b2.block_id      # uncached evicted
        assert b1.ref_cnt == 0                     # A untouched ...
        assert b1.block_hash is not None           # ... and still cached
