# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.core.sched.waiting_prefix_index import (
    PendingPrefixIndex,
    PendingRankInfo,
)


def _hash(tag: str) -> BlockHash:
    return BlockHash(tag.encode())


def test_insert_remove_and_shared_prefix() -> None:
    index = PendingPrefixIndex()
    a, b, c, d, e = map(_hash, "abcde")

    assert index.upsert("r1", [a, b, c]) == {a: 1, b: 1, c: 1}
    assert index.upsert("r2", [a, b, d]) == {a: 2, b: 2, d: 1}
    assert index.upsert("r3", [a, e]) == {a: 3, e: 1}
    assert dict(index.demand_max_depths) == {a: 1, b: 2, c: 3, d: 3, e: 2}

    assert index.longest_shared_prefix_blocks("r1") == 2
    assert index.cluster_info("r1") == (b, 2, 2)
    assert index.cluster_info("r3") == (a, 1, 3)
    assert index.request_score("r1") == 6

    assert index.discard("r2") == {a: 2, b: 1, d: 0}
    assert dict(index.demand_max_depths) == {a: 1, b: 2, c: 3, e: 2}
    assert index.longest_shared_prefix_blocks("r1") == 1
    assert index.cluster_info("r1") == (a, 1, 2)


def test_upsert_updates_only_changed_hash_counts() -> None:
    index = PendingPrefixIndex()
    a, b, c = map(_hash, "abc")

    index.upsert("r1", [a, b])
    version = index.version
    assert index.upsert("r1", [a, b]) == {}
    assert index.version == version

    assert index.upsert("r1", [a, b, c]) == {c: 1}
    assert index.request_path("r1") == (a, b, c)
    assert dict(index.demand_counts) == {a: 1, b: 1, c: 1}


def test_empty_path_still_tracks_request_membership() -> None:
    index = PendingPrefixIndex()

    assert index.upsert("empty", []) == {}
    assert "empty" in index
    assert len(index) == 1
    assert index.discard("empty") == {}
    assert len(index) == 0


def test_max_depth_is_lazy_until_demand_reaches_zero() -> None:
    index = PendingPrefixIndex()
    a = _hash("a")

    index.upsert("deep", [_hash("x"), _hash("y"), a])
    index.upsert("shallow", [a])
    assert index.demand_max_depths[a] == 3

    index.discard("deep")
    assert index.pending_count(a) == 1
    assert index.demand_max_depths[a] == 3

    index.discard("shallow")
    assert a not in index.demand_max_depths


def test_ready_counts_are_independent_from_pending_demand() -> None:
    index = PendingPrefixIndex()
    a, b = map(_hash, "ab")

    index.upsert("ready", [a, b], ready=True)
    index.upsert("parked", [a, b], ready=False)

    assert index.pending_count(b) == 2
    assert index.ready_count(b) == 1
    assert index.rank_info_many(["ready", "parked"], 2) == {
        "ready": PendingRankInfo(2, None, 0, 1, b)
    }

    version = index.version
    assert index.set_ready("parked", True)
    assert index.version == version + 1
    assert index.pending_count(b) == 2
    assert index.ready_count(b) == 2
    assert index.rank_info_many(["ready", "parked"], 2)["ready"] == (
        PendingRankInfo(4, b, 2, 2, b)
    )

    assert index.upsert("parked", [a, b], ready=False) == {}
    assert index.pending_count(b) == 2
    assert index.ready_count(b) == 1


def test_rank_info_can_skip_unused_request_score() -> None:
    index = PendingPrefixIndex()
    a, b = map(_hash, "ab")
    index.upsert("r1", [a, b])
    index.upsert("r2", [a, b])

    without_scores = index.rank_info_many(
        ["r1", "r2"], 1, include_request_score=False
    )
    assert without_scores["r1"] == PendingRankInfo(0, b, 2, 2, a)
    assert index.rank_info_many(["r1", "r2"], 1)["r1"] == (
        PendingRankInfo(4, b, 2, 2, a)
    )


def test_rank_info_cache_is_invalidated_by_index_changes() -> None:
    index = PendingPrefixIndex()
    a, b = map(_hash, "ab")
    index.upsert("r1", [a, b])

    assert index.rank_info_many(["r1"], 1)["r1"].cluster_size == 1
    index.upsert("r2", [a, b])
    assert index.rank_info_many(["r1", "r2"], 1)["r1"].cluster_size == 2


def _naive_amortized(index: PendingPrefixIndex, request_id: str, start: int) -> float:
    path = index.request_path(request_id) or ()
    return sum(1 / max(index.ready_count(h), 1) for h in path[start:])


def test_amortized_blocks_divides_each_block_by_its_ready_sharers() -> None:
    index = PendingPrefixIndex()
    a, b, c, d, e, f, g = map(_hash, "abcdefg")
    index.upsert("r1", [a, b, c, d])
    index.upsert("r2", [a, b, e])
    index.upsert("r3", [a, f])
    # Not rank-ready: demand only, it does not share the cost.
    index.upsert("parked", [a, b, g], ready=False)

    # Counts along r1: a=3, b=2, c=1, d=1.
    assert index.amortized_blocks("r1", 0) == pytest.approx(1 / 3 + 1 / 2 + 2)
    assert index.amortized_blocks("r1", 1) == pytest.approx(1 / 2 + 2)
    assert index.amortized_blocks("r1", 3) == pytest.approx(1)
    assert index.amortized_blocks("r1", 4) == 0.0
    assert index.amortized_blocks("r3", 0) == pytest.approx(1 / 3 + 1)
    assert index.amortized_blocks("missing", 0) == 0.0
    for request_id in ("r1", "r2", "r3", "parked"):
        path = index.request_path(request_id)
        assert path is not None
        for start in range(len(path) + 1):
            assert index.amortized_blocks(request_id, start) == pytest.approx(
                _naive_amortized(index, request_id, start)
            )


def test_amortized_blocks_follows_index_changes() -> None:
    index = PendingPrefixIndex()
    path = [_hash(f"b{i}") for i in range(8)]
    index.upsert("long", path)
    index.upsert("mid", path[:5])
    index.upsert("short", path[:2])

    # Runs of counts 3, 2, 1 along the long path.
    assert index.amortized_blocks("long", 0) == pytest.approx(2 / 3 + 3 / 2 + 3)
    index.discard("mid")
    assert index.amortized_blocks("long", 0) == pytest.approx(2 / 2 + 6)
    index.set_ready("short", False)
    assert index.amortized_blocks("long", 1) == pytest.approx(7)
