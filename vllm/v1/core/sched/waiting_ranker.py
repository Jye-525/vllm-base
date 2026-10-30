# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure ranking logic for cache-aware waiting-request scheduling."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np

from vllm.v1.core.sched.waiting_prefix_index import PendingRankInfo
from vllm.v1.request import Request

Lane = Literal["a", "b"]


@dataclass(frozen=True, slots=True)
class LaneCredit:
    lane: Lane
    stride: float


@dataclass(frozen=True, slots=True)
class RankPlan:
    ordered_requests: list[Request]
    lane_credit: dict[str, LaneCredit]
    dynamic_b_share: float | None = None


def _sections(
    requests: list[Request],
    prefix_hits: dict[str, int],
    rank_info: dict[str, PendingRankInfo],
    check_threshold: int,
) -> dict[str, int]:
    """Classify warm requests, cold pioneers, and later cold siblings."""
    sections: dict[str, int] = {}
    claimed: set[object] = set()
    for request in requests:
        request_id = request.request_id
        info = rank_info[request_id]
        if check_threshold < 0 or prefix_hits[request_id] > check_threshold:
            sections[request_id] = 0
        elif info.claim_hash is None or info.claim_hash not in claimed:
            sections[request_id] = 1
            if info.claim_hash is not None:
                claimed.add(info.claim_hash)
        else:
            sections[request_id] = 2
    return sections


def rank_lpm(
    requests: list[Request],
    prefix_hits: dict[str, int],
) -> RankPlan:
    """Order by prefix hits, preserving current queue order on ties."""
    ordered = sorted(requests, key=lambda request: -prefix_hits[request.request_id])
    return RankPlan(ordered, {})


def _rank_by_aged_length(
    requests: list[Request],
    lengths: Mapping[str, float],
    age_alpha: float,
) -> RankPlan:
    """Order by ``length - age_alpha * wait``, shortest first.

    Aging is linear and shared, so ``length - age_alpha * (now - arrival)``
    orders the same as ``length + age_alpha * arrival``; no clock is needed.
    The sort is stable, so ties keep the current queue order.
    """
    ordered = sorted(
        requests,
        key=lambda request: (
            lengths[request.request_id] + age_alpha * request.arrival_time
        ),
    )
    return RankPlan(ordered, {})


def rank_spf(
    requests: list[Request],
    prefix_hits: dict[str, int],
    age_alpha: float,
) -> RankPlan:
    """Shortest prefill first: fewest uncached tokens, with linear aging.

    A preempted request's ``num_tokens`` includes its generated tokens, which
    must be replayed, so the same formula covers it.
    """
    uncached = {
        request.request_id: request.num_tokens - prefix_hits[request.request_id]
        for request in requests
    }
    return _rank_by_aged_length(requests, uncached, age_alpha)


def rank_sspf(
    requests: list[Request],
    amortized_uncached: dict[str, float],
    age_alpha: float,
) -> RankPlan:
    """Sharing-amortized SPF: fewest amortized uncached tokens, with aging.

    ``amortized_uncached`` prices each uncached block at its size divided by
    the number of waiting requests that contain it, so serving one member
    of a group that shares an uncached prefix is cheap when the rest of the
    group follows. With no sharing it equals the uncached tokens of SPF.
    """
    return _rank_by_aged_length(requests, amortized_uncached, age_alpha)


@dataclass(frozen=True, slots=True)
class DSPFRanking:
    plan: RankPlan
    # The ranked set could meet every deadline from the current work clock.
    feasible: bool
    # Placements that took the EDD head instead of the cheapest request.
    edd_placements: int


def rank_dspf(
    requests: list[Request],
    costs: dict[str, int],
    deadlines: dict[str, int],
    head_cost: int,
    prefill_tokens: dict[str, int],
    token_budget: int,
    max_placements: int,
) -> DSPFRanking:
    """Deadline-guarded shortest prefill first.

    ``costs`` are prefill FLOPs. ``deadlines`` are start deadlines on the
    committed-work clock, relative to its current value (V - W): a request
    must be admitted before committed work passes its deadline, so it must
    complete by deadline + cost. ``head_cost`` is work admitted before any
    ranked request (preempted and other unranked requests).

    Each placement takes the cheapest request unless that would leave a
    request earlier in EDD order (deadline + cost) unable to complete in
    time; then it takes the EDD head. EDD minimises maximum lateness when
    every request is available (Jackson's rule), so a set that can meet all
    deadlines keeps doing so, and a set that cannot is served in EDD order.
    Placement stops once the planned requests fill the step
    (``max_placements`` requests or ``token_budget`` prefill tokens); the
    rest follow in EDD order and are re-ranked next step. Sorts are stable,
    so ties keep the current queue order. Arithmetic is float64: summed
    FLOPs of a long queue can exceed int64.
    """
    n = len(requests)
    ids = [request.request_id for request in requests]
    cost = np.array([costs[i] for i in ids], dtype=np.float64)
    due = np.array([deadlines[i] + costs[i] for i in ids], dtype=np.float64)
    by_cost = np.argsort(cost, kind="stable")
    edd = np.argsort(due, kind="stable")
    position = np.empty(n, dtype=np.int64)
    position[edd] = np.arange(n)
    # In EDD order: completion deadline minus the work up to and including
    # each request. The set is feasible from ``base`` iff base <= min(slack).
    slack = due[edd] - np.cumsum(cost[edd])
    unplaced = np.ones(n, dtype=bool)
    base = float(head_cost)
    feasible = n == 0 or base <= slack.min()

    order: list[int] = []
    edd_placements = 0
    planned_tokens = 0
    cheapest = edd_head = 0
    while len(order) < min(n, max_placements) and planned_tokens < token_budget:
        while not unplaced[position[by_cost[cheapest]]]:
            cheapest += 1
        while not unplaced[edd_head]:
            edd_head += 1
        candidate = by_cost[cheapest]
        pick = edd[edd_head]
        if base <= slack[unplaced].min():
            k = position[candidate]
            ahead = slack[:k][unplaced[:k]]
            if ahead.size == 0 or base + cost[candidate] <= ahead.min():
                pick = candidate
        if pick != candidate:
            edd_placements += 1
        # Placing ``pick`` first moves the start of everything behind it in
        # EDD order by its cost and removes it from their prefix: their
        # slack against ``base`` is unchanged.
        k = position[pick]
        unplaced[k] = False
        slack[k + 1 :] += cost[pick]
        base += cost[pick]
        order.append(int(pick))
        planned_tokens += prefill_tokens[ids[pick]]
    order.extend(int(edd[k]) for k in range(n) if unplaced[k])
    return DSPFRanking(
        RankPlan([requests[i] for i in order], {}), bool(feasible), edd_placements
    )


def rank_sjf(requests: list[Request], age_alpha: float) -> RankPlan:
    """Shortest job first on remaining output tokens, with linear aging.

    ``max_tokens`` is the true output length only when the client sets
    ``ignore_eos``; otherwise it is an upper bound.
    """
    remaining = {
        request.request_id: request.max_tokens - request.num_output_tokens
        for request in requests
    }
    return _rank_by_aged_length(requests, remaining, age_alpha)


def _group_major_order(
    requests: list[Request],
    sections: dict[str, int],
    prefix_hits: dict[str, int],
    rank_info: dict[str, PendingRankInfo],
    positions: dict[str, int],
) -> list[Request]:
    groups: dict[tuple[int, object], list[Request]] = {}
    for request in requests:
        request_id = request.request_id
        info = rank_info[request_id]
        key: tuple[int, object] = (
            sections[request_id],
            info.cluster_hash
            if info.cluster_hash is not None
            else ("request", request_id),
        )
        groups.setdefault(key, []).append(request)

    def group_key(item: tuple[tuple[int, object], list[Request]]) -> tuple:
        (section, _), members = item
        infos = [rank_info[request.request_id] for request in members]
        group_score = max(info.cluster_depth * info.cluster_size for info in infos)
        return (
            section,
            -group_score,
            min(request.arrival_time for request in members),
            min(positions[request.request_id] for request in members),
        )

    ordered: list[Request] = []
    for _, members in sorted(groups.items(), key=group_key):
        ordered.extend(
            sorted(
                members,
                key=lambda request: (
                    -prefix_hits[request.request_id],
                    request.arrival_time,
                    positions[request.request_id],
                ),
            )
        )
    return ordered


def rank_clpm(
    requests: list[Request],
    prefix_hits: dict[str, int],
    rank_info: dict[str, PendingRankInfo],
    *,
    check_threshold: int,
    big_lane_share: float,
    age_alpha: float,
    group_major: bool,
    dynamic_lane: bool,
    slo_budget_s: float,
    previous_b_share: float,
    lane_a_pass: float,
    lane_b_pass: float,
    now: float,
) -> RankPlan:
    """PEEK clustered LPM with cache lane A and age lane B."""
    sections = _sections(requests, prefix_hits, rank_info, check_threshold)
    positions = {request.request_id: pos for pos, request in enumerate(requests)}

    def cache_key(request: Request) -> tuple:
        request_id = request.request_id
        info = rank_info[request_id]
        effective_hit = prefix_hits[request_id]
        if age_alpha > 0.0:
            effective_hit += int(
                age_alpha * max(0.0, now - request.arrival_time)
            )
        return (
            sections[request_id],
            -effective_hit,
            -info.request_score,
            -info.cluster_size,
            positions[request_id],
        )

    b_share = 1.0 - big_lane_share
    next_dynamic_b_share = None
    if dynamic_lane:
        singletons = [
            request
            for request in requests
            if rank_info[request.request_id].cluster_size < 2
        ]
        singleton_fraction = len(singletons) / len(requests)
        oldest_singleton_age = max(
            (max(0.0, now - request.arrival_time) for request in singletons),
            default=0.0,
        )
        age_pressure = min(oldest_singleton_age / slo_budget_s, 1.0)
        raw_b_share = min(
            0.6,
            max(0.1, 0.15 + 0.5 * singleton_fraction + 0.3 * age_pressure),
        )
        b_share = 0.7 * previous_b_share + 0.3 * raw_b_share
        next_dynamic_b_share = b_share
    a_share = 1.0 - b_share

    def order_lane_a() -> list[Request]:
        if group_major:
            return _group_major_order(
                requests, sections, prefix_hits, rank_info, positions
            )
        return sorted(requests, key=cache_key)

    def order_lane_b() -> list[Request]:
        return sorted(
            requests,
            key=lambda request: (
                sections[request.request_id],
                request.arrival_time,
                -prefix_hits[request.request_id],
                positions[request.request_id],
            ),
        )

    if a_share <= 0.0:
        lane_b = order_lane_b()
        return RankPlan(
            lane_b,
            {r.request_id: LaneCredit("b", 1.0) for r in lane_b},
            next_dynamic_b_share,
        )
    if b_share <= 0.0:
        lane_a = order_lane_a()
        return RankPlan(
            lane_a,
            {r.request_id: LaneCredit("a", 1.0) for r in lane_a},
            next_dynamic_b_share,
        )

    lane_a = order_lane_a()
    lane_b = order_lane_b()
    a_stride = 1.0 / a_share
    b_stride = 1.0 / b_share
    local_a_pass = lane_a_pass
    local_b_pass = lane_b_pass
    a_pos = b_pos = 0
    emitted: set[str] = set()
    ordered: list[Request] = []
    credits: dict[str, LaneCredit] = {}

    while len(ordered) < len(requests):
        while a_pos < len(lane_a) and lane_a[a_pos].request_id in emitted:
            a_pos += 1
        while b_pos < len(lane_b) and lane_b[b_pos].request_id in emitted:
            b_pos += 1

        next_a = local_a_pass + a_stride if a_pos < len(lane_a) else float("inf")
        next_b = local_b_pass + b_stride if b_pos < len(lane_b) else float("inf")
        if next_a <= next_b:
            request = lane_a[a_pos]
            a_pos += 1
            local_a_pass = next_a
            credit = LaneCredit("a", a_stride)
        else:
            request = lane_b[b_pos]
            b_pos += 1
            local_b_pass = next_b
            credit = LaneCredit("b", b_stride)
        emitted.add(request.request_id)
        ordered.append(request)
        credits[request.request_id] = credit

    return RankPlan(ordered, credits, next_dynamic_b_share)
