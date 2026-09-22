# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefix-aware predictive admission derived from PKAS evict-optimal-7."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


class OnlineOutputLengthEstimator:
    """PKAS moving estimator using completed output lengths."""

    def __init__(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def observe(self, output_tokens: int) -> None:
        self.count += 1
        delta = output_tokens - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (output_tokens - self.mean)

    def predict(self) -> int:
        if self.count <= 1:
            return 1
        sample_stddev = math.sqrt(self.m2 / (self.count - 1))
        return math.ceil(abs(self.mean - sample_stddev))


@dataclass(frozen=True, slots=True)
class RequestSnapshot:
    request_id: str
    prompt_tokens: int
    computed_tokens: int
    output_tokens: int
    max_output_tokens: int
    block_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class AdmissionDecision:
    feasible: bool
    predicted_output_tokens: int
    prefix_hit_blocks: int
    peak_pinned_blocks: int
    capacity_blocks: int


@dataclass(slots=True)
class _VirtualRequest:
    request_id: str
    prompt_tokens: int
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
    ) -> None:
        self.block_size = block_size
        self.capacity_blocks = capacity_blocks
        self.token_budget = token_budget
        self.long_prefill_token_threshold = long_prefill_token_threshold
        self.estimator = OnlineOutputLengthEstimator()

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
            before = block_ref_counts.get(block_id, 0) + ref_deltas.get(
                block_id, 0
            )
            after = before + delta
            assert after >= 0
            ref_deltas[block_id] = ref_deltas.get(block_id, 0) + delta
            if before == 0 and after > 0:
                live_blocks += 1
            elif before > 0 and after == 0:
                live_blocks -= 1

        def target_for(request: RequestSnapshot) -> int:
            total_output_target = min(
                request.max_output_tokens,
                max(predicted, request.output_tokens + 1),
            )
            return max(
                request.computed_tokens + 1,
                request.prompt_tokens + total_output_target,
            )

        for request in running:
            block_ids = tuple(request.block_ids)
            virtual.append(
                _VirtualRequest(
                    request.request_id,
                    request.prompt_tokens,
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
            candidate.prompt_tokens,
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

        def release(request: _VirtualRequest) -> None:
            nonlocal live_blocks
            for block_id in request.real_block_ids:
                adjust_ref(block_id, -1)
            live_blocks -= request.synthetic_blocks

        while candidate_virtual in virtual:
            # Once every active request is decoding, all receive one token per
            # step in supported configurations. Jump to the next completion.
            if self.token_budget >= len(virtual) and all(
                request.progress >= request.prompt_tokens for request in virtual
            ):
                jump = min(request.target - request.progress for request in virtual)
                for request in virtual:
                    request.progress += jump
                    allocate_to_progress(request, request.progress)
                peak = max(peak, live_blocks)
                if peak > self.capacity_blocks:
                    return AdmissionDecision(
                        False,
                        predicted,
                        prefix_hit_count,
                        peak,
                        self.capacity_blocks,
                    )
            else:
                budget = self.token_budget
                made_progress = False
                for request in virtual:
                    if budget == 0:
                        break
                    if request.progress < request.prompt_tokens:
                        amount = request.prompt_tokens - request.progress
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
                    return AdmissionDecision(
                        False,
                        predicted,
                        prefix_hit_count,
                        peak,
                        self.capacity_blocks,
                    )

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

        raise AssertionError("candidate disappeared without an admission decision")
