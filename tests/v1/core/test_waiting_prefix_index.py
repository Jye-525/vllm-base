# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import random

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
    assert index.pending_count(a) == 3
    assert index.pending_count(b) == 2

    assert index.discard("r2") == {a: 2, b: 1, d: 0}
    assert dict(index.demand_max_depths) == {a: 1, b: 2, c: 3, e: 2}
    assert index.pending_count(b) == 1
    assert index.pending_count(d) == 0


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

    without_scores = index.rank_info_many(["r1", "r2"], 1, include_request_score=False)
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


def _naive_rank_info(
    index: PendingPrefixIndex, request_id: str, claim_depth: int, with_score: bool
) -> PendingRankInfo:
    """The linear-scan definition the cached binary search must reproduce."""
    path = index.request_path(request_id) or ()
    cluster_hash, cluster_depth, cluster_size = None, 0, 1
    for depth in range(len(path), 0, -1):
        count = index.ready_count(path[depth - 1])
        if count >= 2:
            cluster_hash, cluster_depth, cluster_size = path[depth - 1], depth, count
            break
    claim = path[claim_depth - 1] if 0 < claim_depth <= len(path) else None
    score = sum(index.ready_count(h) for h in path) if with_score else 0
    return PendingRankInfo(score, cluster_hash, cluster_depth, cluster_size, claim)


def test_cached_values_match_a_full_recompute_under_random_changes() -> None:
    """Cumulative (chained) paths drawn from a few prefix trees; after every
    insert, removal, path change or readiness change, the cached rank info
    and amortized sums equal the plain definitions (exact floats)."""
    rng = random.Random(11)
    index = PendingPrefixIndex()
    live: dict[str, bool] = {}

    def path_of(tree: int, branch: list[int]) -> list[BlockHash]:
        out, prefix = [], f"t{tree}"
        for step in branch:
            prefix += f"/{step}"
            out.append(_hash(prefix))
        return out

    for op in range(1500):
        roll = rng.random()
        rid = f"r{rng.randrange(60)}"
        if roll < 0.55:
            branch = [rng.randrange(3) for _ in range(rng.randrange(0, 12))]
            ready = rng.random() < 0.85
            index.upsert(rid, path_of(rng.randrange(4), branch), ready=ready)
            live[rid] = ready
        elif roll < 0.8 and rid in live:
            index.discard(rid)
            del live[rid]
        elif rid in live:
            live[rid] = not live[rid]
            index.set_ready(rid, live[rid])
        if op % 3:
            continue
        ids = sorted(live)
        for claim_depth, with_score in ((2, True), (2, False), (0, True)):
            infos = index.rank_info_many(
                ids, claim_depth, include_request_score=with_score
            )
            assert list(infos) == [r for r in ids if live[r]]
            for request_id, info in infos.items():
                assert info == _naive_rank_info(
                    index, request_id, claim_depth, with_score
                )
        for request_id in ids:
            path = index.request_path(request_id) or ()
            for start in {0, len(path) // 2, len(path)}:
                assert index.amortized_blocks(request_id, start) == (
                    _naive_runs_amortized(index, request_id, start)
                )


def _naive_runs_amortized(index: PendingPrefixIndex, request_id: str, start: int):
    """The uncached run sum, term for term (so floats compare exactly)."""
    path = index.request_path(request_id) or ()
    total, i = 0.0, max(start, 0)
    while i < len(path):
        count = max(index.ready_count(path[i]), 1)
        j = i + 1
        while j < len(path) and index.ready_count(path[j]) >= count:
            j += 1
        total += (j - i) / count
        i = j
    return total


def test_root_versions_are_dropped_with_their_last_request() -> None:
    index = PendingPrefixIndex()
    a, b = map(_hash, "ab")
    index.upsert("r1", [a, b])
    index.upsert("r2", [a])
    index.discard("r1")
    assert a in index._root_versions
    index.discard("r2")
    assert index._root_versions == {}
    assert index._amortized_cache == {}
