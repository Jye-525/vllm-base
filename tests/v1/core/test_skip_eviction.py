# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for skip-aware GPU cache eviction (protects LPM-skipped prefixes)."""

from types import SimpleNamespace
from typing import Any

import pytest

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    KVCacheBlock,
    get_block_hash,
    make_block_hash_with_group_id,
)

A, B, C, D, E = b"A", b"B", b"C", b"D", b"E"


def _bh(tag: bytes) -> BlockHash:
    return BlockHash(tag)


def _raw(block: KVCacheBlock) -> BlockHash:
    assert block.block_hash is not None
    return get_block_hash(block.block_hash)


def _request(
    request_id: str, tags: list[bytes], num_blocks: int | None = None
) -> tuple[Any, int]:
    """A (request, protected prefix blocks) pair; the whole prompt by default.
    set_skip_protected reads only request_id and block_hashes."""
    request = SimpleNamespace(
        request_id=request_id, block_hashes=[_bh(t) for t in tags]
    )
    return request, len(tags) if num_blocks is None else num_blocks


def _pool_with_cached(
    tags: list[bytes], extra_blocks: int = 0
) -> tuple[BlockPool, dict[bytes, KVCacheBlock]]:
    """A pool whose free queue holds cached blocks in `tags` order (LRU
    first), then `extra_blocks` uncached free blocks."""
    pool = BlockPool(
        num_gpu_blocks=len(tags) + extra_blocks + 1,
        enable_caching=True,
        hash_block_size=16,
    )
    pool._eviction_policy = "skip"
    uncached = pool.free_block_queue.popleft_n(extra_blocks)
    blocks = pool.free_block_queue.popleft_n(len(tags))
    by_tag = {}
    for block, tag in zip(blocks, tags):
        key = make_block_hash_with_group_id(_bh(tag), 0)
        block.block_hash = key
        pool.cached_block_hash_to_block.insert(key, block)
        by_tag[tag] = block
    pool.free_block_queue.append_n(uncached + blocks)
    return pool, by_tag


def _lru_order(pool: BlockPool) -> list[BlockHash | None]:
    return [
        _raw(block) if block.block_hash is not None else None
        for block in pool.free_block_queue.get_all_free_blocks()
    ]


def _assert_invariants(pool: BlockPool) -> None:
    main = pool.free_block_queue.get_all_free_blocks()
    side = pool._skip_side_pool
    assert pool.get_num_free_blocks() == len(main) + len(side)
    assert not {block.block_id for block in main} & set(side)
    for block in side.values():
        assert block.ref_cnt == 0
        assert block.block_hash in pool._skip_protect_counts
    for block in main:
        assert block.block_hash not in pool._skip_protect_counts


def test_protected_blocks_stay_free_and_are_evicted_last() -> None:
    pool, by_tag = _pool_with_cached([A, B, C, D])
    pool.set_skip_protected([_request("R1", [A, B])])

    assert pool.num_skip_protected_free_blocks == 2
    assert pool.get_num_free_blocks() == 4
    assert _lru_order(pool) == [_bh(C), _bh(D)]
    _assert_invariants(pool)

    victims = pool.get_new_blocks(3)

    # Plain LRU first, then the protected request's deepest block.
    assert victims == [by_tag[C], by_tag[D], by_tag[B]]
    assert pool.num_skip_protected_evictions == 1
    assert pool._skip_side_pool == {by_tag[A].block_id: by_tag[A]}
    assert pool.get_num_free_blocks() == 1
    _assert_invariants(pool)


def test_only_the_cached_prefix_is_protected() -> None:
    pool, _ = _pool_with_cached([A, B, C])
    # B was evicted earlier, so C is unreachable as a prefix: protect only A.
    pool.set_skip_protected([_request("R1", [A, D, C], num_blocks=1)])

    assert pool.num_skip_protected_free_blocks == 1
    assert _lru_order(pool) == [_bh(B), _bh(C)]
    _assert_invariants(pool)


def test_grown_prefix_is_protected_again() -> None:
    pool, _ = _pool_with_cached([A, B, C])
    pool.set_skip_protected([_request("R1", [A, B, C], num_blocks=1)])
    assert pool.num_skip_protected_free_blocks == 1

    # Skipped again with a longer cached prefix.
    pool.set_skip_protected([_request("R1", [A, B, C])])

    assert pool.num_skip_protected_free_blocks == 3
    assert _lru_order(pool) == []
    _assert_invariants(pool)


def test_lowest_rank_deepest_block_is_evicted_first() -> None:
    pool, by_tag = _pool_with_cached([A, B, C, D], extra_blocks=1)
    pool.set_skip_protected([_request("R1", [A, B]), _request("R2", [C, D])])

    victims = pool.get_new_blocks(3)

    assert victims[0].block_hash is None  # the unprotected block
    assert victims[1:] == [by_tag[D], by_tag[C]]
    assert pool.get_new_blocks(1) == [by_tag[B]]
    _assert_invariants(pool)


def test_shared_block_takes_the_best_rank() -> None:
    pool, by_tag = _pool_with_cached([A, B, C])
    # A is block 0 of both prompts; R1 outranks R2.
    pool.set_skip_protected([_request("R1", [A, B]), _request("R2", [A, C])])

    assert pool.get_new_blocks(1) == [by_tag[C]]
    assert pool.get_new_blocks(1) == [by_tag[B]]
    assert pool.get_new_blocks(1) == [by_tag[A]]


def test_reordering_moves_no_block() -> None:
    pool, by_tag = _pool_with_cached([A, B, C, D])
    r1, r2 = _request("R1", [A, B]), _request("R2", [C, D])
    pool.set_skip_protected([r1, r2])
    pool.set_skip_protected([r2, r1])

    assert pool.num_skip_protected_free_blocks == 4
    # R1 is now the least important: its deepest block goes first.
    assert pool.get_new_blocks(1) == [by_tag[B]]


def test_release_returns_blocks_to_recent_end_deepest_first() -> None:
    pool, _ = _pool_with_cached([A, B, C, D])
    pool.set_skip_protected([_request("R1", [A, B]), _request("R2", [A])])

    pool.set_skip_protected([_request("R2", [A])])

    # B is no longer protected; A is still protected through R2.
    assert _lru_order(pool) == [_bh(C), _bh(D), _bh(B)]
    assert pool.num_skip_protected_free_blocks == 1
    _assert_invariants(pool)

    pool.set_skip_protected([])

    assert _lru_order(pool) == [_bh(C), _bh(D), _bh(B), _bh(A)]
    assert pool.num_skip_protected_free_blocks == 0
    assert not pool._skip_protect_counts
    _assert_invariants(pool)


def test_freed_block_with_protected_hash_goes_to_side_pool() -> None:
    pool, by_tag = _pool_with_cached([A, B])
    pool.touch([by_tag[A], by_tag[B]])
    pool.set_skip_protected([_request("R1", [A])])

    # Held blocks are not moved; they are routed when freed.
    assert pool.num_skip_protected_free_blocks == 0
    pool.free_blocks([by_tag[B], by_tag[A]])

    assert pool._skip_side_pool == {by_tag[A].block_id: by_tag[A]}
    assert _lru_order(pool) == [_bh(B)]
    _assert_invariants(pool)


def test_touch_takes_block_out_of_side_pool() -> None:
    pool, by_tag = _pool_with_cached([A, B])
    pool.set_skip_protected([_request("R1", [A])])

    pool.touch([by_tag[A]])

    assert by_tag[A].ref_cnt == 1
    assert pool.num_skip_protected_free_blocks == 0
    assert pool.get_num_free_blocks() == 1
    _assert_invariants(pool)


def test_evict_blocks_moves_uncached_block_back_to_lru() -> None:
    pool, by_tag = _pool_with_cached([A, B])
    pool.set_skip_protected([_request("R1", [A])])

    pool.evict_blocks({by_tag[A].block_id})

    assert by_tag[A].block_hash is None
    # The uncached block is the next victim.
    assert _lru_order(pool) == [None, _bh(B)]
    assert pool.num_skip_protected_free_blocks == 0
    assert pool.get_num_free_blocks() == 2
    _assert_invariants(pool)


def test_reset_prefix_cache_drops_protection() -> None:
    pool, _ = _pool_with_cached([A, B])
    pool.set_skip_protected([_request("R1", [A, B])])

    assert pool.reset_prefix_cache()

    assert pool.num_skip_protected_free_blocks == 0
    assert pool.get_num_free_blocks() == 2
    assert not pool._skip_protect_counts
    assert pool._skip_protected_ids == []
    _assert_invariants(pool)


def test_freed_blocks_join_the_eviction_order() -> None:
    pool, by_tag = _pool_with_cached([A, B, C, D])
    pool.touch([by_tag[D]])
    pool.set_skip_protected([_request("R1", [A, B]), _request("R2", [C, D])])
    # Build the victim heap, then free D into the side pool.
    assert pool.get_new_blocks(1) == [by_tag[C]]
    pool.free_blocks([by_tag[D]])

    assert pool.get_new_blocks(1) == [by_tag[D]]
    assert pool.get_new_blocks(1) == [by_tag[B]]
    assert pool.num_skip_protected_evictions == 3
    _assert_invariants(pool)


def test_cannot_allocate_more_than_free_including_protected() -> None:
    pool, _ = _pool_with_cached([A, B])
    pool.set_skip_protected([_request("R1", [A, B])])

    assert len(pool.get_new_blocks(2)) == 2
    with pytest.raises(ValueError):
        pool.get_new_blocks(1)
