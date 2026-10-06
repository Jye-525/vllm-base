# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for PEEK-compatible waiting-queue-aware GPU cache eviction."""

import pytest

from vllm.v1.core import block_pool as block_pool_module
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    KVCacheBlock,
    get_block_hash,
    make_block_hash_with_group_id,
)


def _bh(tag: bytes) -> BlockHash:
    return BlockHash(tag)


def _bhg(tag: bytes, group_id: int = 0):
    return make_block_hash_with_group_id(_bh(tag), group_id)


def _raw(block: KVCacheBlock) -> BlockHash:
    assert block.block_hash is not None
    return get_block_hash(block.block_hash)


def _make_pool(
    num_blocks: int,
    policy: str = "peek",
    mode: str = "plain",
) -> BlockPool:
    pool = BlockPool(
        num_gpu_blocks=num_blocks,
        enable_caching=True,
        hash_block_size=16,
    )
    pool._eviction_policy = policy
    pool._peek_eviction_mode = mode
    return pool


def _cache(pool: BlockPool, block: KVCacheBlock, tag: bytes) -> None:
    bhg = _bhg(tag)
    block.block_hash = bhg
    pool.cached_block_hash_to_block.insert(bhg, block)
    block.ref_cnt = 0


def _pool_with_cached(
    tags: list[bytes],
    mode: str = "plain",
) -> tuple[BlockPool, dict[bytes, KVCacheBlock]]:
    pool = _make_pool(len(tags) + 1, mode=mode)
    blocks = pool.free_block_queue.popleft_n(len(tags))
    by_tag = {}
    for block, tag in zip(blocks, tags):
        _cache(pool, block, tag)
        pool.free_block_queue.append(block)
        by_tag[tag] = block
    return pool, by_tag


A, B, C, D = b"A", b"B", b"C", b"D"


def test_incremental_demand_count_and_depth_updates() -> None:
    pool = _make_pool(2, mode="cluster")
    pool.update_peek_demand_counts(
        {_bh(A): 2, _bh(B): 1},
        max_depths={_bh(A): 1, _bh(B): 4},
    )
    assert pool._demand_counts == {_bh(A): 2.0, _bh(B): 1.0}
    assert pool._demand_max_depths == {_bh(A): 1, _bh(B): 4}

    pool.update_peek_demand_counts({_bh(A): 0}, max_depths={_bh(B): 4})
    assert pool._demand_counts == {_bh(B): 1.0}
    assert pool._demand_max_depths == {_bh(B): 4}


def test_ranking_probe_does_not_count_peek_hits() -> None:
    pool, _ = _pool_with_cached([A, B])
    pool.set_peek_demand_counts({_bh(A): 1.0})
    hits = pool.peek_hits
    with pool.probing():
        assert pool.get_cached_block(_bh(A), [0]) is not None
    assert pool.peek_hits == hits
    assert pool.get_cached_block(_bh(A), [0]) is not None
    assert pool.peek_hits == hits + 1


def test_zero_demand_evicted_before_demanded() -> None:
    pool, _ = _pool_with_cached([A, B, C])
    pool.set_peek_demand_counts({_bh(A): 2.0, _bh(B): 1.0})
    victims = pool._select_victims_peek(1)
    assert [_raw(victim) for victim in victims] == [_bh(C)]


def test_plain_mode_evicts_lowest_demand_then_lru() -> None:
    pool, _ = _pool_with_cached([A, B, C], mode="plain")
    pool.set_peek_demand_counts({_bh(A): 3.0, _bh(B): 1.0, _bh(C): 2.0})
    victims = pool._select_victims_peek(1)
    assert [_raw(victim) for victim in victims] == [_bh(B)]


def test_cluster_mode_multiplies_demand_by_chain_depth() -> None:
    pool, _ = _pool_with_cached([A, B], mode="cluster")
    pool.set_peek_demand_counts(
        {_bh(A): 2.0, _bh(B): 1.0},
        max_depths={_bh(A): 1, _bh(B): 5},
    )
    victims = pool._select_victims_peek(1)
    # A: 2 * (1 + 1) = 4; B: 1 * (1 + 5) = 6.
    assert [_raw(victim) for victim in victims] == [_bh(A)]


def test_recency_mode_applies_linear_age_penalty() -> None:
    pool, _ = _pool_with_cached([A, B], mode="recency")
    pool._peek_eviction_recency_weight = 10.0
    pool.set_peek_demand_counts({_bh(A): 5.0, _bh(B): 1.0})
    pool._block_hash_last_access = {_bh(A): 99.0, _bh(B): 100.0}
    assert pool._get_peek_eviction_priority(_bh(A), now=100.0) == -5.0
    assert pool._get_peek_eviction_priority(_bh(B), now=100.0) == 1.0


def test_decay_mode_applies_exponential_age_penalty() -> None:
    pool, _ = _pool_with_cached([A, B], mode="decay")
    pool._peek_eviction_decay_tau = 1.0
    pool.set_peek_demand_counts({_bh(A): 5.0, _bh(B): 1.0})
    pool._block_hash_last_access = {_bh(A): 90.0, _bh(B): 100.0}
    assert pool._get_peek_eviction_priority(_bh(A), now=100.0) == 0.0
    assert pool._get_peek_eviction_priority(_bh(B), now=100.0) == 1000.0


@pytest.mark.parametrize("mode", ["recency", "decay"])
def test_touch_records_last_access(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    pool, blocks = _pool_with_cached([A], mode=mode)
    monkeypatch.setattr(block_pool_module.time, "time", lambda: 123.0)
    pool.touch([blocks[A]])
    assert pool._block_hash_last_access[_bh(A)] == 123.0


def test_survivors_keep_lru_order() -> None:
    pool, _ = _pool_with_cached([A, B, C, D])
    pool.set_peek_demand_counts({_bh(A): 5.0})
    victims = pool._select_victims_peek(2)
    assert [_raw(victim) for victim in victims] == [_bh(B), _bh(C)]
    remaining = pool.free_block_queue.get_all_free_blocks()
    assert [_raw(block) for block in remaining] == [_bh(A), _bh(D)]


def test_falls_back_to_demanded_under_pressure() -> None:
    pool, _ = _pool_with_cached([A, B, C])
    pool.set_peek_demand_counts({_bh(A): 3.0, _bh(B): 1.0, _bh(C): 2.0})
    victims = pool._select_victims_peek(2)
    assert {_raw(victim) for victim in victims} == {_bh(B), _bh(C)}


def test_empty_demand_is_plain_lru() -> None:
    pool, _ = _pool_with_cached([A, B, C])
    pool.set_peek_demand_counts({})
    victims = pool._select_victims_peek(2)
    assert [_raw(victim) for victim in victims] == [_bh(A), _bh(B)]


def test_lru_policy_uses_popleft_order() -> None:
    pool = _make_pool(5, policy="lru")
    ids_before = [
        block.block_id for block in pool.free_block_queue.get_all_free_blocks()
    ]
    new = pool.get_new_blocks(2)
    assert [block.block_id for block in new] == ids_before[:2]
