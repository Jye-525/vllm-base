# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure ranking logic for cache-aware waiting-request scheduling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

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
    rank_info: dict[str, PendingRankInfo],
    check_threshold: int,
) -> RankPlan:
    """PEEK LPM: prefix hits first, but keep cold sibling bursts at the tail."""
    sections = _sections(requests, prefix_hits, rank_info, check_threshold)
    positions = {request.request_id: pos for pos, request in enumerate(requests)}
    ordered = sorted(
        requests,
        key=lambda request: (
            1 if sections[request.request_id] == 2 else 0,
            0
            if sections[request.request_id] == 2
            else -prefix_hits[request.request_id],
            positions[request.request_id],
        ),
    )
    return RankPlan(ordered, {})


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
        effective_hit = prefix_hits[request_id] + int(
            age_alpha * max(0.0, now - request.arrival_time)
        )
        return (
            sections[request_id],
            -effective_hit,
            -info.request_score,
            -info.cluster_size,
            positions[request_id],
        )

    lane_a = (
        _group_major_order(requests, sections, prefix_hits, rank_info, positions)
        if group_major
        else sorted(requests, key=cache_key)
    )
    lane_b = sorted(
        requests,
        key=lambda request: (
            sections[request.request_id],
            request.arrival_time,
            -prefix_hits[request.request_id],
            positions[request.request_id],
        ),
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

    if a_share <= 0.0:
        return RankPlan(
            lane_b,
            {r.request_id: LaneCredit("b", 1.0) for r in lane_b},
            next_dynamic_b_share,
        )
    if b_share <= 0.0:
        return RankPlan(
            lane_a,
            {r.request_id: LaneCredit("a", 1.0) for r in lane_a},
            next_dynamic_b_share,
        )

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
