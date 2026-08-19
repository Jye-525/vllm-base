# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.core.sched.request_queue import FCFSRequestQueue
from vllm.v1.core.sched.waiting_prefix_index import PendingRankInfo
from vllm.v1.core.sched.waiting_ranker import rank_clpm, rank_lpm


def _request(request_id: str, arrival_time: float = 0.0):
    return SimpleNamespace(request_id=request_id, arrival_time=arrival_time)


def _info(
    *, score: int = 1, size: int = 1, claim: bytes | None = None
) -> PendingRankInfo:
    block_hash = BlockHash(claim) if claim is not None else None
    return PendingRankInfo(
        request_score=score,
        cluster_hash=block_hash if size >= 2 else None,
        cluster_depth=1 if size >= 2 else 0,
        cluster_size=size,
        claim_hash=block_hash,
    )


def test_lpm_keeps_one_cold_pioneer_and_deprioritizes_siblings() -> None:
    pioneer = _request("pioneer")
    sibling = _request("sibling")
    other = _request("other")
    requests = [pioneer, sibling, other]
    hits = {"pioneer": 0, "sibling": 32, "other": 16}
    infos = {
        "pioneer": _info(score=4, size=2, claim=b"shared"),
        "sibling": _info(score=4, size=2, claim=b"shared"),
        "other": _info(claim=b"other"),
    }

    plan = rank_lpm(requests, hits, infos, check_threshold=32)

    assert [r.request_id for r in plan.ordered_requests] == [
        "other",
        "pioneer",
        "sibling",
    ]


def test_clpm_interleaves_cache_and_age_lanes() -> None:
    fast1 = _request("fast1", 20.0)
    fast2 = _request("fast2", 30.0)
    old = _request("old", 10.0)
    requests = [old, fast2, fast1]
    hits = {"old": 40, "fast2": 80, "fast1": 100}
    infos = {
        request.request_id: _info(score=2, size=2, claim=b"shared")
        for request in requests
    }

    plan = rank_clpm(
        requests,
        hits,
        infos,
        check_threshold=32,
        big_lane_share=0.7,
        age_alpha=0.0,
        group_major=False,
        dynamic_lane=False,
        slo_budget_s=2.0,
        previous_b_share=0.3,
        lane_a_pass=0.0,
        lane_b_pass=0.0,
        now=30.0,
    )

    assert [r.request_id for r in plan.ordered_requests] == [
        "fast1",
        "fast2",
        "old",
    ]
    assert plan.lane_credit["old"].lane == "b"


def test_queue_reorders_only_rankable_slots_in_bounded_prefix() -> None:
    queue = FCFSRequestQueue(
        [_request("a"), _request("parked"), _request("b"), _request("tail")]
    )

    queue.reorder_rankable_prefix(
        3,
        [_request("b"), _request("a")],
        {"a", "b"},
    )

    assert [request.request_id for request in queue] == [
        "b",
        "parked",
        "a",
        "tail",
    ]
