# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefix-aware predictive admission derived from PKAS evict-optimal-7."""

from __future__ import annotations

from bisect import bisect_left, bisect_right, insort
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


class BatchCostModel:
    """Small measured latency curves for prefill and decode batches."""

    def __init__(self) -> None:
        self.curves: dict[bool, dict[int, tuple[float, float, int]]] = {
            False: {},
            True: {},
        }
        self.prefill_seconds = 0.0
        self.prefill_tokens = 0
        self.estimated_replay_tokens = 0
        self.actual_replay_tokens = 0

    def observe(self, prefill: int, decode: int, seconds: float) -> None:
        tokens = prefill + decode
        if tokens <= 0 or seconds <= 0:
            return
        curve = self.curves[prefill > 0]
        bucket = tokens.bit_length()
        old = curve.get(bucket)
        if old is None:
            curve[bucket] = (float(tokens), seconds, 1)
        else:
            x, y, count = old
            curve[bucket] = (
                0.8 * x + 0.2 * tokens,
                0.8 * y + 0.2 * seconds,
                count + 1,
            )
        if prefill:
            self.prefill_seconds += seconds
            self.prefill_tokens += prefill

    def estimate(self, prefill: int, decode: int) -> float | None:
        tokens = prefill + decode
        if tokens == 0:
            return 0.0
        points = sorted(
            (x, y) for x, y, n in self.curves[prefill > 0].values() if n >= 3
        )
        if not points:
            return None
        index = bisect_left([x for x, _ in points], tokens)
        if index == 0:
            # Below observed sizes, do not invent an unmeasured launch cost.
            x, y = points[0]
            return y * tokens / x
        if index == len(points):
            x, y = points[-1]
            return y * tokens / x
        x0, y0 = points[index - 1]
        x1, y1 = points[index]
        return y0 + (y1 - y0) * (tokens - x0) / (x1 - x0)

    def replay_budget(
        self,
        base_prefill: int,
        base_decode: int,
        new_prefill: int,
        new_decode: int,
        ratio: float,
    ) -> tuple[int, float]:
        if ratio <= 0 or not self.prefill_tokens:
            return 0, 0.0
        base = self.estimate(base_prefill, base_decode)
        solo = self.estimate(new_prefill, new_decode)
        combined = self.estimate(
            base_prefill + new_prefill, base_decode + new_decode
        )
        if base is None or solo is None or combined is None:
            return 0, 0.0
        saving = max(0.0, base + solo - combined)
        seconds_per_token = self.prefill_seconds / self.prefill_tokens
        correction = max(
            1.0, self.actual_replay_tokens / max(1, self.estimated_replay_tokens)
        )
        budget = min(saving, ratio * solo) / (seconds_per_token * correction)
        return int(budget), saving


class CausalOutputLengthEstimator:
    """Upper-quantile forecast from recent completed outputs only."""

    def __init__(self) -> None:
        self.history: deque[int] = deque()
        self.sorted_lengths: list[int] = []

    @property
    def count(self) -> int:
        return len(self.history)

    def observe(self, output_tokens: int) -> None:
        if len(self.history) == 256:
            oldest = self.history.popleft()
            self.sorted_lengths.pop(bisect_left(self.sorted_lengths, oldest))
        self.history.append(output_tokens)
        insort(self.sorted_lengths, output_tokens)

    def predict(self) -> int:
        if self.count < 32:
            return 512
        return max(1, self.sorted_lengths[int(0.9 * (self.count - 1))])

    def target(self, observed_tokens: int) -> int:
        predicted = self.predict()
        if observed_tokens < predicted:
            return predicted
        first = bisect_right(self.sorted_lengths, observed_tokens)
        survivors = self.count - first
        if survivors >= 8:
            return self.sorted_lengths[first + int(0.9 * (survivors - 1))]
        return observed_tokens + predicted


@dataclass(frozen=True, slots=True)
class RequestSnapshot:
    request_id: str
    prompt_tokens: int
    computed_tokens: int
    output_tokens: int
    block_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class AdmissionDecision:
    feasible: bool
    predicted_output_tokens: int
    prefix_hit_blocks: int
    peak_pinned_blocks: int
    capacity_blocks: int
    peak_is_upper_bound: bool = False
    replay_allowed: bool = False
    estimated_replay_tokens: int = 0


@dataclass(slots=True)
class _VirtualRequest:
    request_id: str
    prefill_end: int
    progress: int
    target: int
    real_block_ids: tuple[int, ...]
    allocated_blocks: int
    synthetic_blocks: int = 0


class PredictiveAdmissionController:
    """Simulate active KV growth before admitting one waiting request."""

    def __init__(
        self,
        *,
        block_size: int,
        capacity_blocks: int,
        token_budget: int,
        long_prefill_token_threshold: int,
        max_model_len: int,
    ) -> None:
        self.block_size = block_size
        self.capacity_blocks = capacity_blocks
        self.token_budget = token_budget
        self.long_prefill_token_threshold = long_prefill_token_threshold
        self.max_model_len = max_model_len
        self.estimator = CausalOutputLengthEstimator()

    def observe_completion(self, output_tokens: int) -> None:
        self.estimator.observe(output_tokens)

    def evaluate(
        self,
        running: Sequence[RequestSnapshot],
        candidate: RequestSnapshot,
        *,
        candidate_hit_block_ids: Sequence[int],
        block_ref_counts: Mapping[int, int],
        initial_pinned_blocks: int | None = None,
        use_capacity_bound: bool = True,
        use_fast_forward: bool = True,
        replay_budget_tokens: int = 0,
    ) -> AdmissionDecision:
        predicted = self.estimator.predict()
        # Keep the scheduler snapshot immutable and record only reference
        # changes caused by this speculative simulation.
        ref_deltas: dict[int, int] = {}
        live_blocks = (
            sum(count > 0 for count in block_ref_counts.values())
            if initial_pinned_blocks is None
            else initial_pinned_blocks
        )
        virtual: list[_VirtualRequest] = []

        def adjust_ref(block_id: int, delta: int) -> None:
            nonlocal live_blocks
            before = block_ref_counts.get(block_id, 0) + ref_deltas.get(block_id, 0)
            after = before + delta
            assert after >= 0
            ref_deltas[block_id] = ref_deltas.get(block_id, 0) + delta
            if before == 0 and after > 0:
                live_blocks += 1
            elif before > 0 and after == 0:
                live_blocks -= 1

        def target_for(request: RequestSnapshot) -> int:
            total_output_target = self.estimator.target(request.output_tokens)
            return min(
                request.prompt_tokens + total_output_target - 1,
                self.max_model_len - 1,
            )

        for request in running:
            block_ids = tuple(request.block_ids)
            virtual.append(
                _VirtualRequest(
                    request.request_id,
                    request.prompt_tokens + request.output_tokens,
                    request.computed_tokens,
                    target_for(request),
                    block_ids,
                    len(block_ids),
                )
            )

        hit_ids = list(dict.fromkeys(candidate_hit_block_ids))
        prefix_hit_count = len(hit_ids)
        for block_id in hit_ids:
            adjust_ref(block_id, 1)

        def allocate_to_progress(request: _VirtualRequest, progress: int) -> None:
            nonlocal live_blocks
            required = (progress + self.block_size - 1) // self.block_size
            # Predicted future blocks cannot be shared yet, so their identities
            # are irrelevant. Counting them avoids one Python object per block.
            new_blocks = max(0, required - request.allocated_blocks)
            request.allocated_blocks += new_blocks
            request.synthetic_blocks += new_blocks
            live_blocks += new_blocks

        candidate_virtual = _VirtualRequest(
            candidate.request_id,
            candidate.prompt_tokens + candidate.output_tokens,
            candidate.computed_tokens,
            target_for(candidate),
            tuple(hit_ids),
            len(hit_ids),
        )
        allocate_to_progress(candidate_virtual, candidate_virtual.progress)
        virtual.append(candidate_virtual)

        peak = live_blocks
        if peak > self.capacity_blocks:
            return AdmissionDecision(
                False, predicted, prefix_hit_count, peak, self.capacity_blocks
            )

        if use_capacity_bound:
            upper_bound = live_blocks + sum(
                max(
                    0,
                    (request.target + self.block_size - 1) // self.block_size
                    - request.allocated_blocks,
                )
                for request in virtual
            )
            if upper_bound <= self.capacity_blocks:
                return AdmissionDecision(
                    True,
                    predicted,
                    prefix_hit_count,
                    upper_bound,
                    self.capacity_blocks,
                    peak_is_upper_bound=True,
                )

        def release(request: _VirtualRequest) -> None:
            nonlocal live_blocks
            for block_id in request.real_block_ids:
                adjust_ref(block_id, -1)
            live_blocks -= request.synthetic_blocks

        def overflow_decision() -> AdmissionDecision:
            replay_tokens = 0
            allowed = False
            # LPM preempts the newest (tail) request and frees its blocks in
            # reverse order. Price the prefix lost if the deficit consumes
            # those newly evictable blocks before the candidate resumes.
            if replay_budget_tokens > 0 and len(virtual) > 1:
                deficit = live_blocks - self.capacity_blocks
                victim = candidate_virtual
                for index in range(victim.allocated_blocks - 1, -1, -1):
                    if index < len(victim.real_block_ids):
                        block_id = victim.real_block_ids[index]
                        refs = block_ref_counts.get(block_id, 0) + ref_deltas.get(
                            block_id, 0
                        )
                        if refs > 1:
                            continue
                    deficit -= 1
                    if deficit == 0:
                        # An incomplete tail block is not a reusable cache hit.
                        prefix = min(
                            index * self.block_size,
                            victim.progress // self.block_size * self.block_size,
                        )
                        replay_tokens = max(0, victim.progress - prefix)
                        # One block-equivalent charges for restarting a victim.
                        allowed = (
                            replay_tokens + self.block_size <= replay_budget_tokens
                        )
                        break
            return AdmissionDecision(
                False,
                predicted,
                prefix_hit_count,
                peak,
                self.capacity_blocks,
                replay_allowed=allowed,
                estimated_replay_tokens=replay_tokens,
            )

        while True:
            # Snapshots already include this step's allocations and progress.
            # Check its peak before releasing requests that finish in this step.
            if candidate_virtual.progress >= candidate_virtual.target:
                return AdmissionDecision(
                    True, predicted, prefix_hit_count, peak, self.capacity_blocks
                )
            survivors: list[_VirtualRequest] = []
            for request in virtual:
                if request.progress >= request.target:
                    release(request)
                else:
                    survivors.append(request)
            virtual = survivors
            # Once every active request is decoding, all receive one token per
            # step in supported configurations. Jump to the next completion.
            if (
                use_fast_forward
                and self.token_budget >= len(virtual)
                and all(request.progress >= request.prefill_end for request in virtual)
            ):
                jump = min(request.target - request.progress for request in virtual)
                if replay_budget_tokens > 0:
                    def growth(steps: int) -> int:
                        return sum(
                            max(
                                0,
                                (request.progress + steps + self.block_size - 1)
                                // self.block_size - request.allocated_blocks,
                            )
                            for request in virtual
                        )

                    if live_blocks + growth(jump) > self.capacity_blocks:
                        low, high = 1, jump
                        while low < high:
                            middle = (low + high) // 2
                            if live_blocks + growth(middle) > self.capacity_blocks:
                                high = middle
                            else:
                                low = middle + 1
                        jump = low
                for request in virtual:
                    request.progress += jump
                    allocate_to_progress(request, request.progress)
                peak = max(peak, live_blocks)
                if peak > self.capacity_blocks:
                    return overflow_decision()
            else:
                budget = self.token_budget
                made_progress = False
                for request in virtual:
                    if budget == 0:
                        break
                    if request.progress < request.prefill_end:
                        amount = request.prefill_end - request.progress
                        threshold = self.long_prefill_token_threshold
                        if 0 < threshold < amount:
                            amount = threshold
                    else:
                        amount = 1
                    amount = min(amount, budget, request.target - request.progress)
                    if amount <= 0:
                        continue
                    request.progress += amount
                    budget -= amount
                    made_progress = True
                    allocate_to_progress(request, request.progress)
                if not made_progress:
                    return AdmissionDecision(
                        False,
                        predicted,
                        prefix_hit_count,
                        peak,
                        self.capacity_blocks,
                    )
                peak = max(peak, live_blocks)
                if peak > self.capacity_blocks:
                    return overflow_decision()
