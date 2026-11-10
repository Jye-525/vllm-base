# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cross-step state of the related-work waiting-queue orderings.

Fairway, RAGCache and k-LPM (and the SSPF overtaking guard) are ports of the
KVReuse simulator's engine-independent modules (``fairway_ranker``,
``ragcache_ranker``, ``klpm_ranker``, ``overtake_guard``). This class wires
them to the scheduler and keeps what they remember between steps:

- arrival order: ``(arrival_time, sequence)`` keys, the simulator's
  ``arrival_key`` (equal arrival times break by the order received);
- the ranking clock (``time.time``; tests swap it for a fake clock);
- RAGCache's bypass window, k-LPM's cycle and plan, and the overtaking
  guard's overtaken work, all advanced once per fresh admission.

Only fresh requests (``RequestStatus.WAITING``) are ranked or counted:
preempted requests resume at the queue head, as in the simulator.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping
from typing import TYPE_CHECKING

from vllm.v1.core.sched.fairway_prefill_model import load_chunk_time
from vllm.v1.core.sched.fairway_ranker import (
    FairwayParams,
    prefill_time_estimator,
    rank_fairway,
)
from vllm.v1.core.sched.klpm_ranker import KLPMCycle, rank_klpm
from vllm.v1.core.sched.overtake_guard import OvertakeGuard, order_with_guard
from vllm.v1.core.sched.ragcache_ranker import BypassWindow, rank_ragcache
from vllm.v1.core.sched.waiting_ranker import RankPlan
from vllm.v1.request import Request, RequestStatus

if TYPE_CHECKING:
    from vllm.config import SchedulerConfig

ArrivalKey = tuple[float, int]


class RankingState:
    """Arrival order, clock and per-admission state of the related-work
    orderings. ``policy`` is the scheduling policy's value (``"fairway"``,
    ``"ragcache"``, ``"klpm"``, ``"sspf"``, ...)."""

    def __init__(
        self,
        config: SchedulerConfig,
        policy: str,
        token_budget: int,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.policy = policy
        self.clock = clock
        self._seq: dict[str, int] = {}
        self._next_seq = 0

        self.fairway: FairwayParams | None = None
        self.fairway_est: Callable[[int], float] | None = None
        if policy == "fairway":
            if not config.fairway_prefill_model:
                raise ValueError(
                    "the 'fairway' scheduling policy needs fairway_prefill_model "
                    "(a forward-sim model JSON)"
                )
            self.fairway = FairwayParams(
                weight=config.fairway_weight,
                patience_mult=config.fairway_patience_mult,
                patience_floor_s=config.fairway_patience_floor_s,
                wait=config.fairway_wait,
                fair_lane=config.fairway_fair_lane,
            )
            self.set_fairway_chunk_time(
                load_chunk_time(config.fairway_prefill_model), token_budget
            )

        self.ragcache_window = config.ragcache_window
        self.ragcache_mode = config.ragcache_window_mode
        self.bypass: BypassWindow | None = None
        if (
            policy == "ragcache"
            and self.ragcache_mode == "bypass"
            and (self.ragcache_window > 0)
        ):
            self.bypass = BypassWindow(self.ragcache_window)

        self.klpm: KLPMCycle | None = None
        # k-LPM: the fresh order of the last ranking and the index of its next
        # fresh admission. Admissions that follow it keep it valid.
        self._klpm_plan: list[str] = []
        self._klpm_next = 0
        if policy == "klpm":
            self.klpm = KLPMCycle(config.klpm_k)

        self.guard: OvertakeGuard | None = None
        self._guard_due: set[str] = set()
        if config.sspf_overtake_guard_k > 0:
            self.guard = OvertakeGuard(config.sspf_overtake_guard_k)

    # ------------------------------------------------------------ arrivals

    def on_arrival(self, request_id: str) -> None:
        if request_id not in self._seq:
            self._seq[request_id] = self._next_seq
            self._next_seq += 1

    def arrival_key(self, request: Request) -> ArrivalKey:
        return (request.arrival_time, self._seq[request.request_id])

    def forget(self, request_id: str) -> None:
        """The request finished or was aborted."""
        self._seq.pop(request_id, None)
        if self.bypass is not None:
            self.bypass.forget(request_id)
        if self.guard is not None:
            self.guard.forget(request_id)
            self._guard_due.discard(request_id)

    @property
    def tracks_admissions(self) -> bool:
        """Whether a fresh admission must be reported (with the fresh
        requests still waiting)."""
        return (
            self.bypass is not None or self.klpm is not None or self.guard is not None
        )

    # ------------------------------------------------------------ fairway

    def set_fairway_chunk_time(
        self, chunk_time: Callable[[int, int], float], token_budget: int
    ) -> None:
        """Price est(len) with ``chunk_time(start, end)`` in ``token_budget``
        chunks (tests give the simulator's step time here)."""
        self.fairway_est = prefill_time_estimator(chunk_time, token_budget)

    def rank_fairway(
        self, candidates: list[Request], prefix_hits: Mapping[str, int]
    ) -> RankPlan:
        assert self.fairway is not None and self.fairway_est is not None
        oldest = sorted(candidates, key=self.arrival_key)
        ids = [r.request_id for r in oldest]
        order = rank_fairway(
            ids,
            prefix_hits,
            {r.request_id: r.num_tokens for r in oldest},
            {r.request_id: r.arrival_time for r in oldest},
            self.clock(),
            self.fairway_est,
            self.fairway,
        )
        return _plan(candidates, order)

    # ------------------------------------------------------------ ragcache

    def rank_ragcache(
        self, candidates: list[Request], prefix_hits: Mapping[str, int]
    ) -> RankPlan:
        order = rank_ragcache(
            [r.request_id for r in candidates],
            prefix_hits,
            {r.request_id: r.num_tokens for r in candidates},
            {r.request_id: self.arrival_key(r) for r in candidates},
            self.ragcache_window,
            self.ragcache_mode,
            self.bypass,
        )
        return _plan(candidates, order)

    # ------------------------------------------------------------ k-LPM

    def rank_klpm(
        self, candidates: list[Request], prefix_hits: Mapping[str, int]
    ) -> RankPlan:
        assert self.klpm is not None
        order = rank_klpm(
            [r.request_id for r in candidates],
            prefix_hits,
            {r.request_id: self.arrival_key(r) for r in candidates},
            self.klpm.k,
            self.klpm.pos,
        )
        self._klpm_plan = order
        self._klpm_next = 0
        return _plan(candidates, order)

    def begin_step(self) -> None:
        """A new scheduler step: the previous step's k-LPM plan is stale
        (this step ranks again, unless there is at most one candidate)."""
        self._klpm_plan = []
        self._klpm_next = 0

    def on_set_aside(self) -> bool:
        """A waiting request was set aside (blocked, LoRA limit, unknown
        connector hits). Returns whether the queue must be re-ranked before
        the next admission: k-LPM's plan assumed it would be admitted."""
        return self.klpm is not None and bool(self._klpm_plan)

    # ------------------------------------------------------------ SSPF guard

    def guard_order(self, plan: RankPlan, prefix_hits: Mapping[str, int]) -> RankPlan:
        """Move the due requests of an SSPF order to its front, oldest first."""
        if self.guard is None:
            return plan
        self._guard_due = set()
        if not self.guard.over:  # nothing is due before any request was overtaken
            return plan
        ordered = plan.ordered_requests
        by_id = {r.request_id: r for r in ordered}
        order, due = order_with_guard(
            [r.request_id for r in ordered],
            {r.request_id: r.num_tokens - prefix_hits[r.request_id] for r in ordered},
            {r.request_id: self.arrival_key(r) for r in ordered},
            self.guard,
        )
        self._guard_due = set(due)
        return RankPlan(
            [by_id[i] for i in order], plan.lane_credit, plan.dynamic_b_share
        )

    # ------------------------------------------------------------ admissions

    def on_fresh_admission(
        self,
        request: Request,
        num_computed_tokens: int,
        waiting: Iterable[Request],
    ) -> bool:
        """``request`` (fresh) was just admitted with ``num_computed_tokens``
        cached; ``waiting`` holds every request still in the waiting queues.
        Returns whether the queue must be re-ranked before the next
        admission (k-LPM: the admission did not follow the plan)."""
        request_id = request.request_id
        key = self.arrival_key(request)
        fresh: dict[str, ArrivalKey] | None = None
        if self.bypass is not None or self.guard is not None:
            fresh = {
                q.request_id: self.arrival_key(q)
                for q in waiting
                if q.status == RequestStatus.WAITING
            }
        if self.bypass is not None:
            assert fresh is not None
            self.bypass.on_admit(request_id, key, fresh)
        if self.guard is not None:
            assert fresh is not None
            self.guard.on_admit(
                request_id,
                request.num_tokens - num_computed_tokens,
                key,
                fresh,
                was_due=request_id in self._guard_due,
            )
            self._guard_due.discard(request_id)
        if self.klpm is None:
            return False
        self.klpm.on_admit()
        plan, i = self._klpm_plan, self._klpm_next
        if not plan:
            return False  # nothing was ranked this step (at most one candidate)
        if i < len(plan) and plan[i] == request_id:
            self._klpm_next = i + 1
            return False
        # Admitted out of plan (an unranked or set-aside request): the plan's
        # slots no longer match the cycle, so rank again from the new position.
        self._klpm_plan = []
        return True


def _plan(candidates: list[Request], order: list[str]) -> RankPlan:
    by_id = {r.request_id: r for r in candidates}
    return RankPlan([by_id[i] for i in order], {})
