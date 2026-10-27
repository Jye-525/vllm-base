# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSPF: deadline-guarded shortest prefill first.

Costs are prefill FLOPs (prefill_cost.py). ``work`` (W) is a committed-work
clock: the estimated prefill FLOPs of every admission so far, where a resumed
request counts again for its recompute. The first step a waiting request is
ranked, it gets a start deadline V: when serving the work already waiting,
then this step's new requests cheapest first, would finish it. Ranking
(``rank_dspf``) takes the cheapest request unless that would make another
miss its deadline.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from vllm.logger import init_logger
from vllm.v1.core.sched.prefill_cost import PrefillCostShape
from vllm.v1.core.sched.request_queue import FCFSRequestQueue
from vllm.v1.core.sched.waiting_ranker import DSPFRanking, rank_dspf
from vllm.v1.request import Request, RequestStatus

logger = init_logger(__name__)


def check_dspf_scope(**features: bool) -> None:
    """Raise if any named unsupported feature is in use."""
    unsupported = [name for name, present in features.items() if present]
    if unsupported:
        raise ValueError(
            "The 'dspf' scheduling policy does not support " + ", ".join(unsupported)
        )


class DSPFScheduler:
    """DSPF's per-request deadlines, cost snapshot and work clock.

    ``prefix_len`` is the read-only prefix-cache lookup; admission does the
    authoritative one and reports it through ``cost_grew``.
    """

    def __init__(
        self,
        shape: PrefillCostShape,
        window_size: int,
        prefix_len: Callable[[Request], int],
    ) -> None:
        self.shape = shape
        self.window_size = window_size
        self._prefix_len = prefix_len
        self.work = 0
        # Start deadline V on the work clock of each waiting request.
        self.deadlines: dict[str, int] = {}
        # Costs from the latest ranking; admission compares against these.
        self.planned_costs: dict[str, int] = {}
        # Cached tokens found by admission's authoritative lookup when a
        # request cost more than planned; they override the read-only lookup
        # for the rest of the step.
        self._admission_hits: dict[str, int] = {}
        # Logged per step from the step's first ranking (-1: none ran).
        self.step_feasible = -1
        self.step_edd = 0
        self.step_late = 0

    def start_step(self) -> None:
        self._admission_hits.clear()
        self.step_feasible = -1
        self.step_edd = self.step_late = 0

    def cost(self, request: Request) -> tuple[int, int]:
        """Prefill FLOPs and uncached tokens of a waiting request."""
        hits = self._admission_hits.get(request.request_id)
        if hits is None:
            hits = request.num_computed_tokens or self._prefix_len(request)
        uncached = request.num_tokens - hits
        return self.shape.prefill_flops(uncached, hits), uncached

    def rank(
        self,
        waiting: FCFSRequestQueue,
        skipped_waiting: Iterable[Request],
        token_budget: int,
        free_slots: int,
        record_step: bool,
    ) -> DSPFRanking:
        """Give first-time candidates deadlines, refresh the cost snapshot,
        and reorder the ranked window of ``waiting``.

        Runs in full even with zero or one candidate, so admission always
        compares against costs from this call. Candidates are fresh waiting
        requests; preempted, partially computed and skipped-queue requests
        keep their slots and count as head work admitted before them
        (conservative when they are not at the head).
        """
        window = len(waiting) if self.window_size == 0 else self.window_size
        head_cost = sum(self.cost(request)[0] for request in skipped_waiting)
        candidates: list[Request] = []
        costs: dict[str, int] = {}
        uncached: dict[str, int] = {}
        for request in waiting.iter_first_n(window):
            cost, tokens = self.cost(request)
            if (
                request.status == RequestStatus.WAITING
                and request.num_computed_tokens == 0
            ):
                candidates.append(request)
                costs[request.request_id] = cost
                uncached[request.request_id] = tokens
            else:
                head_cost += cost
        self._assign_deadlines(candidates, costs, head_cost)

        relative = {
            request_id: self.deadlines[request_id] - self.work for request_id in costs
        }
        ranking = rank_dspf(
            candidates, costs, relative, head_cost, uncached, token_budget, free_slots
        )
        if len(candidates) > 1:
            waiting.reorder_rankable_prefix(
                self.window_size, ranking.plan.ordered_requests, set(costs)
            )
        self.planned_costs = costs
        if record_step:
            self.step_feasible = int(ranking.feasible)
            self.step_edd = ranking.edd_placements
            self.step_late = sum(deadline < 0 for deadline in relative.values())
        return ranking

    def _assign_deadlines(
        self, candidates: list[Request], costs: dict[str, int], head_cost: int
    ) -> None:
        """Deadline of each first-time candidate: when serving the work
        already waiting (head work and candidates with deadlines, at their
        current cost), then the new ones cheapest first, would finish it."""
        new = [r for r in candidates if r.request_id not in self.deadlines]
        if not new:
            return
        finish = self.work + head_cost
        finish += sum(
            costs[r.request_id] for r in candidates if r.request_id in self.deadlines
        )
        for request in sorted(new, key=lambda r: costs[r.request_id]):
            finish += costs[request.request_id]
            self.deadlines[request.request_id] = finish

    def cost_grew(self, request: Request, num_local_hit_tokens: int) -> bool:
        """Whether admission's authoritative lookup made ``request`` cost
        more than its ranking assumed (an admission earlier in this step
        evicted its blocks). If so the caller re-ranks, using this cost for
        ``request``, and retries the queue head. At most once per request
        per step; a second increase is admitted as is."""
        planned = self.planned_costs.get(request.request_id)
        if planned is None:
            return False
        actual = self.shape.prefill_flops(
            request.num_tokens - num_local_hit_tokens, num_local_hit_tokens
        )
        if actual <= planned:
            return False
        if request.request_id in self._admission_hits:
            logger.debug(
                "DSPF: %s cost grew again after a re-rank; admitting it.",
                request.request_id,
            )
            return False
        self._admission_hits[request.request_id] = num_local_hit_tokens
        return True

    def admit(self, request: Request, num_computed_tokens: int) -> str:
        """Charge an admitted prefill to the work clock; return log fields."""
        cost = self.shape.prefill_flops(
            request.num_tokens - num_computed_tokens, num_computed_tokens
        )
        deadline = self.deadlines.pop(request.request_id, None)
        self.planned_costs.pop(request.request_id, None)
        slack = (
            ""
            if deadline is None
            else f" dspf_slack_gflop={(deadline - self.work) / 1e9:.1f}"
        )
        self.work += cost
        return f" dspf_cost_gflop={cost / 1e9:.1f}{slack}"

    def forget(self, request_id: str) -> None:
        """A request finished or was aborted. Other deadlines stay as they
        are: they may include only part of its original cost."""
        self.deadlines.pop(request_id, None)
        self.planned_costs.pop(request_id, None)
