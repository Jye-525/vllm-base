# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.core.sched.request_queue import FCFSRequestQueue
from vllm.v1.core.sched.waiting_prefix_index import PendingRankInfo
from vllm.v1.core.sched.waiting_ranker import (
    rank_clpm,
    rank_dspf,
    rank_lpm,
    rank_sjf,
    rank_spf,
)


def _request(request_id: str, arrival_time: float = 0.0):
    return SimpleNamespace(request_id=request_id, arrival_time=arrival_time)


def _info(
    *, score: int = 1, size: int = 1, depth: int = 1, claim: bytes | None = None
) -> PendingRankInfo:
    block_hash = BlockHash(claim) if claim is not None else None
    return PendingRankInfo(
        request_score=score,
        cluster_hash=block_hash if size >= 2 else None,
        cluster_depth=depth if size >= 2 else 0,
        cluster_size=size,
        claim_hash=block_hash,
    )


def test_lpm_orders_only_by_prefix_hits() -> None:
    requests = [_request("first"), _request("second"), _request("third")]
    hits = {"first": 0, "second": 32, "third": 16}

    plan = rank_lpm(requests, hits)

    assert [r.request_id for r in plan.ordered_requests] == [
        "second",
        "third",
        "first",
    ]
    assert plan.lane_credit == {}


def test_lpm_ties_preserve_queue_order_not_arrival() -> None:
    requests = [_request("young", 20), _request("old", 10)]

    plan = rank_lpm(requests, {"young": 64, "old": 64})

    assert [r.request_id for r in plan.ordered_requests] == ["young", "old"]


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


def _job(
    request_id: str,
    *,
    arrival_time: float = 0.0,
    num_tokens: int = 0,
    max_tokens: int = 0,
    num_output_tokens: int = 0,
):
    return SimpleNamespace(
        request_id=request_id,
        arrival_time=arrival_time,
        num_tokens=num_tokens,
        max_tokens=max_tokens,
        num_output_tokens=num_output_tokens,
    )


def _ids(plan) -> list[str]:
    return [request.request_id for request in plan.ordered_requests]


def test_spf_orders_by_uncached_tokens() -> None:
    requests = [
        _job("long_cold", num_tokens=100),
        _job("long_warm", num_tokens=100),
        _job("short_cold", num_tokens=40),
    ]
    hits = {"long_cold": 0, "long_warm": 80, "short_cold": 0}

    plan = rank_spf(requests, hits, age_alpha=0.0)

    # Uncached: long_warm 20, short_cold 40, long_cold 100.
    assert _ids(plan) == ["long_warm", "short_cold", "long_cold"]
    assert plan.lane_credit == {}


def test_spf_ties_preserve_queue_order() -> None:
    requests = [
        _job("young", arrival_time=20, num_tokens=50),
        _job("old", arrival_time=10, num_tokens=50),
    ]

    plan = rank_spf(requests, {"young": 0, "old": 0}, age_alpha=0.0)

    assert _ids(plan) == ["young", "old"]


def test_spf_aging_overtakes_exactly_past_length_gap() -> None:
    # Length gap 100 tokens; alpha 10 tokens/s needs more than 10 s of age.
    def order(old_arrival: float) -> list[str]:
        requests = [
            _job("short", arrival_time=100.0, num_tokens=20),
            _job("long", arrival_time=old_arrival, num_tokens=120),
        ]
        return _ids(rank_spf(requests, {"short": 0, "long": 0}, age_alpha=10.0))

    assert order(old_arrival=91.0) == ["short", "long"]
    assert order(old_arrival=89.0) == ["long", "short"]


def test_sjf_orders_by_remaining_output_tokens() -> None:
    requests = [
        _job("big", max_tokens=500),
        _job("small", max_tokens=50),
        # A preempted request that already produced 450 of 480 tokens.
        _job("nearly_done", max_tokens=480, num_output_tokens=450),
    ]

    plan = rank_sjf(requests, age_alpha=0.0)

    assert _ids(plan) == ["nearly_done", "small", "big"]


def test_sjf_aging_promotes_old_long_job() -> None:
    requests = [
        _job("short", arrival_time=100.0, max_tokens=10),
        _job("long", arrival_time=0.0, max_tokens=1000),
    ]

    assert _ids(rank_sjf(requests, age_alpha=0.0)) == ["short", "long"]
    # 100 s of extra wait at 20 tokens/s outweighs a 990-token gap.
    assert _ids(rank_sjf(requests, age_alpha=20.0)) == ["long", "short"]


def _dspf(
    jobs: list[tuple[str, int, int]],
    head_cost: int = 0,
    token_budget: int = 10**9,
    max_placements: int = 10**9,
):
    """Rank (id, cost, deadline) jobs in queue order; one token per cost unit."""
    requests = [_request(job_id) for job_id, _, _ in jobs]
    costs = {job_id: cost for job_id, cost, _ in jobs}
    deadlines = {job_id: deadline for job_id, _, deadline in jobs}
    return rank_dspf(
        requests, costs, deadlines, head_cost, costs, token_budget, max_placements
    )


def _starts(ranking, jobs, head_cost: int = 0) -> dict[str, int]:
    costs = {job_id: cost for job_id, cost, _ in jobs}
    starts, clock = {}, head_cost
    for request in ranking.plan.ordered_requests:
        starts[request.request_id] = clock
        clock += costs[request.request_id]
    return starts


def test_dspf_protects_every_deadline_not_only_the_earliest() -> None:
    # Reviewer counterexample: guarding only the earliest deadline ran the
    # nine small jobs, then A, and started B at 19, past its deadline 12.
    jobs = [("A", 10, 10), ("B", 2, 12)]
    jobs += [(f"s{i}", 1, 12 + i) for i in range(1, 10)]

    ranking = _dspf(jobs)

    assert ranking.feasible
    assert _ids(ranking.plan) == [f"s{i}" for i in range(1, 9)] + ["B", "A", "s9"]
    starts = _starts(ranking, jobs)
    assert all(starts[job_id] <= deadline for job_id, _, deadline in jobs)
    assert ranking.edd_placements == 2


def test_dspf_is_spf_when_no_deadline_binds() -> None:
    jobs = [("big", 9, 1000), ("tie1", 2, 1000), ("small", 1, 1000), ("tie2", 2, 1000)]

    ranking = _dspf(jobs)

    # Cheapest first; equal costs keep queue order.
    assert _ids(ranking.plan) == ["small", "tie1", "tie2", "big"]
    assert ranking.edd_placements == 0


def test_dspf_infeasible_set_is_served_in_edd_order() -> None:
    # Deadline + cost: late_big 5, late_small 6, fresh 101.
    jobs = [("fresh", 1, 100), ("late_big", 8, -3), ("late_small", 1, 5)]

    ranking = _dspf(jobs)

    assert not ranking.feasible
    assert _ids(ranking.plan) == ["late_big", "late_small", "fresh"]


def test_dspf_counts_head_work_before_ranked_requests() -> None:
    # Y must start by 3. With no head work the cheap X fits before it; with
    # 3 units of preempted work ahead, X would push Y to 4.
    jobs = [("X", 1, 100), ("Y", 5, 3)]

    assert _ids(_dspf(jobs).plan) == ["X", "Y"]
    assert _ids(_dspf(jobs, head_cost=3).plan) == ["Y", "X"]


def test_dspf_plans_one_step_then_follows_edd_order() -> None:
    jobs = [("late", 4, 50), ("early", 4, 10), ("cheap", 1, 1000)]

    by_count = _dspf(jobs, max_placements=1)
    by_tokens = _dspf(jobs, token_budget=1)

    # One placement (the cheapest), then EDD: early (14) before late (54).
    assert _ids(by_count.plan) == ["cheap", "early", "late"]
    assert _ids(by_tokens.plan) == ["cheap", "early", "late"]
    assert _ids(_dspf(jobs, max_placements=0).plan) == ["early", "late", "cheap"]


def test_dspf_empty_window() -> None:
    ranking = _dspf([])

    assert ranking.feasible
    assert ranking.plan.ordered_requests == []
