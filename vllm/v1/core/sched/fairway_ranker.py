# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fairway waiting-queue ordering (Alikhani et al., arXiv:2611.02082, Alg. 1).

Port of the KVReuse simulator's ``simulator/sched-sim/fairway_order.py``;
keys, stable sorts and tie rules are the same, so both engines order the
same waiting queue identically. Functions take mappings keyed by request id.

Per scheduling step, for each waiting request r with ``len`` tokens and ``h``
cached prefix tokens:

    c      = max(1, len - h)                      work (uncached tokens)
    SLO    = max(floor, mult * est(len))          patience
    score  = h / c + wait / SLO                   Fast-lane score

The Fast lane orders by score (highest first, ties oldest first), the Fair
lane by arrival. The two lanes are merged by work charges (stride
scheduling): the Fast lane's charge starts at ``w * c`` of its first request,
the Fair lane's at ``c`` of the oldest request; the lane with the smaller
charge (Fast on ties) places its next unplaced request and adds that
request's work, times ``w`` for the Fast lane. Charges restart every step.

The paper fixes w = 4, mult = 3 and floor = 1 s for every workload.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Mapping
from dataclasses import dataclass
from functools import cache
from typing import TypeVar

T = TypeVar("T", bound=Hashable)

WAIT_MODES = ("relative", "absolute", "none")


@dataclass(frozen=True)
class FairwayParams:
    weight: float = 4.0  # Fast-lane work weight w
    patience_mult: float = 3.0  # patience = max(floor, mult * est(len))
    patience_floor_s: float = 1.0
    wait: str = "relative"  # relative (paper) | absolute (s) | none (ratio only)
    fair_lane: bool = True

    def __post_init__(self) -> None:
        if self.weight <= 0:
            raise ValueError("Fairway weight must be > 0")
        if self.patience_mult < 0 or self.patience_floor_s < 0:
            raise ValueError("Fairway patience multiplier and floor must be >= 0")
        if (
            self.wait == "relative"
            and self.patience_floor_s == 0
            and self.patience_mult == 0
        ):
            raise ValueError(
                "relative wait needs a positive patience floor or multiplier"
            )
        if self.wait not in WAIT_MODES:
            raise ValueError(
                f"unknown Fairway wait mode {self.wait!r}; one of {WAIT_MODES}"
            )


def prefill_time_estimator(
    chunk_time: Callable[[int, int], float], budget: int
) -> Callable[[int], float]:
    """est(num_tokens): seconds to prefill a prompt alone on an idle engine.

    ``chunk_time(start, end)`` is the forward-pass time in seconds of one
    prefill-only step computing prompt tokens [start, end) after the
    ``start`` already computed. The prompt runs as ``budget``-token chunks,
    one step each. This stands in for the paper's one-off GPU profiling.
    """
    if budget <= 0:
        raise ValueError("token budget must be > 0")

    @cache
    def est(num_tokens: int) -> float:
        total, start = 0.0, 0
        while start < num_tokens:
            end = min(num_tokens, start + budget)
            total += chunk_time(start, end)
            start = end
        return total

    return est


def patience(
    num_tokens: int, est: Callable[[int], float], params: FairwayParams
) -> float:
    """Seconds a request may wait before its relative wait reaches 1."""
    return max(params.patience_floor_s, params.patience_mult * est(num_tokens))


def fast_score(
    hit_tokens: int,
    num_tokens: int,
    wait_s: float,
    est: Callable[[int], float],
    params: FairwayParams,
) -> float:
    """KV reuse ratio plus the wait term (higher ranks first)."""
    work = max(1, num_tokens - hit_tokens)
    score = hit_tokens / work
    if params.wait == "relative":
        score += wait_s / patience(num_tokens, est, params)
    elif params.wait == "absolute":
        score += wait_s
    return score


def merge_by_work_charge(
    fast: list[T], fair: list[T], work: Mapping[T, int], weight: float
) -> list[T]:
    """Interleave two orders of the same requests by work charges.

    ``fast`` and ``fair`` hold the same requests; ``work`` maps each to its
    uncached tokens (>= 1). Returns one order containing each request once.
    """
    if len(fast) != len(fair):
        raise ValueError("both lanes must hold the same requests")
    if not fast:
        return []
    placed: set[T] = set()
    order: list[T] = []
    a = weight * work[fast[0]]
    f = work[fair[0]]
    i = j = 0
    while len(order) < len(fast):
        if a <= f:
            while fast[i] in placed:
                i += 1
            r = fast[i]
            a += weight * work[r]
        else:
            while fair[j] in placed:
                j += 1
            r = fair[j]
            f += work[r]
        placed.add(r)
        order.append(r)
    return order


def rank_fairway(
    items: Iterable[T],
    hit_tokens: Mapping[T, int],
    num_tokens: Mapping[T, int],
    arrival: Mapping[T, float],
    now: float,
    est: Callable[[int], float],
    params: FairwayParams,
) -> list[T]:
    """Fairway order of the waiting requests ``items``.

    ``hit_tokens``, ``num_tokens`` and ``arrival`` (seconds) map each item to
    its cached prefix, its token count and its arrival time; ``now`` is the
    current time. Sorts are stable, so pass ``items`` oldest first to break
    equal arrival times by arrival order.
    """
    items = list(items)
    if len(items) <= 1:
        return items
    fair = sorted(items, key=lambda r: arrival[r])
    score = {
        r: fast_score(hit_tokens[r], num_tokens[r], now - arrival[r], est, params)
        for r in items
    }
    fast = sorted(fair, key=lambda r: -score[r])
    if not params.fair_lane:
        return fast
    work = {r: max(1, num_tokens[r] - hit_tokens[r]) for r in items}
    return merge_by_work_charge(fast, fair, work, params.weight)
