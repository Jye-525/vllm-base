# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefix-aware predictive admission derived from PKAS evict-optimal-7."""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right, insort
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol


class OutputLengthEstimator(Protocol):
    """Causal output-length forecast fed by completed requests only."""

    @property
    def count(self) -> int: ...

    def reset(self) -> None: ...

    def observe(self, output_tokens: int) -> None: ...

    def predict(self) -> int: ...

    def target(self, observed_tokens: int) -> int: ...


class CausalOutputLengthEstimator:
    """Upper-quantile forecast from recent completed outputs only.

    Legacy predictor (``pkas_output_predictor="legacy"``), kept to reproduce
    runs made before 2026-10-12. Its 512-token cold forecast, 32-sample
    warm-up and 8-sample tail threshold are fixed choices without a
    derivation; use ``QuantileOutputLengthEstimator`` for new experiments.
    """

    def __init__(self, quantile: float = 0.9, history: int = 256) -> None:
        if not 0 < quantile <= 1:
            raise ValueError("quantile must be in (0, 1]")
        if history < 8:
            raise ValueError("history must be at least 8")
        self.quantile = quantile
        self.max_history = history
        self.min_samples = min(32, history // 2)
        self.history: deque[int] = deque()
        self.sorted_lengths: list[int] = []

    @property
    def count(self) -> int:
        return len(self.history)

    def reset(self) -> None:
        self.history.clear()
        self.sorted_lengths.clear()

    def observe(self, output_tokens: int) -> None:
        if len(self.history) == self.max_history:
            oldest = self.history.popleft()
            self.sorted_lengths.pop(bisect_left(self.sorted_lengths, oldest))
        self.history.append(output_tokens)
        insort(self.sorted_lengths, output_tokens)

    def predict(self) -> int:
        if self.count < self.min_samples:
            return 512
        return max(1, self.sorted_lengths[int(self.quantile * (self.count - 1))])

    def target(self, observed_tokens: int) -> int:
        predicted = self.predict()
        if observed_tokens < predicted:
            return predicted
        first = bisect_right(self.sorted_lengths, observed_tokens)
        survivors = self.count - first
        if survivors >= 8:
            return self.sorted_lengths[first + int(self.quantile * (survivors - 1))]
        return observed_tokens + predicted


class PaperOutputLengthEstimator:
    """PKAS paper predictor (Eq. 1): ``ceil(|M1 - sqrt(M2 / (N - 1)))|``.

    M1 and M2 are Welford's running mean and sum of squared deviations over
    every completed output since the last reset. With at most one completion
    the forecast is one token. A request that has already produced at least
    the forecast is expected to produce one more token.
    """

    def __init__(self) -> None:
        self.reset()

    @property
    def count(self) -> int:
        return self._count

    def reset(self) -> None:
        self._count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def observe(self, output_tokens: int) -> None:
        self._count += 1
        delta = output_tokens - self.mean
        self.mean += delta / self._count
        self.m2 += delta * (output_tokens - self.mean)

    def predict(self) -> int:
        if self._count <= 1:
            return 1
        sample_stddev = math.sqrt(self.m2 / (self._count - 1))
        # Every request produces at least one token.
        return max(1, math.ceil(abs(self.mean - sample_stddev)))

    def target(self, observed_tokens: int) -> int:
        predicted = self.predict()
        if observed_tokens < predicted:
            return predicted
        return observed_tokens + 1


def min_quantile_samples(quantile: float, confidence: float) -> int:
    """Fewest samples whose maximum bounds the ``quantile`` with ``confidence``.

    All n i.i.d. samples fall below the true q-quantile with probability q**n,
    so the sample maximum is a distribution-free upper confidence bound once
    q**n <= 1 - confidence. Below this size no order statistic can bound the
    quantile, and the empirical quantile is not used.
    """
    if not 0 < quantile < 1:
        raise ValueError("quantile must be in (0, 1)")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    return max(1, math.ceil(math.log(1 - confidence) / math.log(quantile)))


class QuantileOutputLengthEstimator:
    """Empirical quantile of completed outputs with derived sample-size rules.

    ``history`` limits the samples to the most recent completions; 0 keeps
    all of them. Until ``min_quantile_samples(quantile, confidence)`` requests
    have completed there is no evidence for a forecast, and the paper's rule
    applies: one more token. A request that outlives the forecast uses the
    same quantile of the completed outputs longer than it when there are
    enough of them, else the longest completed output, else one more token.
    """

    def __init__(
        self, quantile: float = 0.5, history: int = 0, confidence: float = 0.95
    ) -> None:
        if history < 0:
            raise ValueError("history must be nonnegative (0 keeps all outputs)")
        self.quantile = quantile
        self.confidence = confidence
        self.max_history = history
        self.min_samples = min_quantile_samples(quantile, confidence)
        if history and history < self.min_samples:
            raise ValueError(
                f"history {history} cannot hold the {self.min_samples} samples "
                "needed to bound this quantile"
            )
        self.history: deque[int] = deque()
        self.sorted_lengths: list[int] = []

    @property
    def count(self) -> int:
        return len(self.history)

    def reset(self) -> None:
        self.history.clear()
        self.sorted_lengths.clear()

    def observe(self, output_tokens: int) -> None:
        if self.max_history and len(self.history) == self.max_history:
            oldest = self.history.popleft()
            self.sorted_lengths.pop(bisect_left(self.sorted_lengths, oldest))
        self.history.append(output_tokens)
        insort(self.sorted_lengths, output_tokens)

    def predict(self) -> int:
        if self.count < self.min_samples:
            return 1
        return max(1, self.sorted_lengths[int(self.quantile * (self.count - 1))])

    def target(self, observed_tokens: int) -> int:
        predicted = self.predict()
        if observed_tokens < predicted:
            return predicted
        if self.count < self.min_samples:
            return observed_tokens + 1
        first = bisect_right(self.sorted_lengths, observed_tokens)
        longer = self.count - first
        if longer >= self.min_samples:
            return self.sorted_lengths[first + int(self.quantile * (longer - 1))]
        if longer:
            return self.sorted_lengths[-1]
        return observed_tokens + 1


def make_output_length_estimator(
    predictor: str,
    *,
    quantile: float,
    history: int | None,
    confidence: float,
) -> OutputLengthEstimator:
    if predictor == "legacy":
        return CausalOutputLengthEstimator(
            quantile, 256 if history is None else history
        )
    if predictor == "paper":
        return PaperOutputLengthEstimator()
    if predictor == "quantile":
        return QuantileOutputLengthEstimator(
            quantile, 0 if history is None else history, confidence
        )
    raise ValueError(f"Unknown PKAS output predictor: {predictor}")


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
        output_predictor: str = "legacy",
        output_quantile: float = 0.9,
        output_history: int | None = None,
        output_confidence: float = 0.95,
    ) -> None:
        self.block_size = block_size
        self.capacity_blocks = capacity_blocks
        self.token_budget = token_budget
        self.long_prefill_token_threshold = long_prefill_token_threshold
        self.max_model_len = max_model_len
        self.estimator: OutputLengthEstimator = make_output_length_estimator(
            output_predictor,
            quantile=output_quantile,
            history=output_history,
            confidence=output_confidence,
        )

    def reset_history(self) -> None:
        self.estimator.reset()

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
            return AdmissionDecision(
                False, predicted, prefix_hit_count, peak, self.capacity_blocks
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

    def evaluate_repo(
        self,
        running: Sequence[RequestSnapshot],
        candidate: RequestSnapshot,
        *,
        candidate_hit_block_ids: Sequence[int],
        block_ref_counts: Mapping[int, int],
        initial_pinned_blocks: int | None = None,
    ) -> AdmissionDecision:
        """Copy of ``evaluate`` that follows the PKAS repository code.

        Mirrors ``_simulate_running_queue`` of the PKAS repository's
        evict-optimal-7 scheduler rather than the paper text. Every running
        request is expected to produce a full forecast more tokens, whatever
        it has produced already (the repository computes ``pt + est -
        progress[0]`` and ``progress[0]`` is the prompt length), and the
        event-driven fast-forward with its prefill accounting is the
        repository's. KV accounting stays prefix-aware: hit and shared blocks
        are reference counted, so a block is freed only with its last
        reference, and evictable cached blocks count as free.
        """
        predicted = self.estimator.predict()
        ref_deltas: dict[int, int] = {}
        live_blocks = (
            sum(count > 0 for count in block_ref_counts.values())
            if initial_pinned_blocks is None
            else initial_pinned_blocks
        )

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

        def allocate_to_progress(request: _VirtualRequest, progress: int) -> None:
            nonlocal live_blocks
            required = (progress + self.block_size - 1) // self.block_size
            new_blocks = max(0, required - request.allocated_blocks)
            request.allocated_blocks += new_blocks
            request.synthetic_blocks += new_blocks
            live_blocks += new_blocks

        def release(request: _VirtualRequest) -> None:
            nonlocal live_blocks
            for block_id in request.real_block_ids:
                adjust_ref(block_id, -1)
            live_blocks -= request.synthetic_blocks

        def decision(feasible: bool) -> AdmissionDecision:
            return AdmissionDecision(
                feasible, predicted, len(hit_ids), peak, self.capacity_blocks
            )

        # The candidate must fit its next forward pass, after its prefix hit.
        hit_ids = list(dict.fromkeys(candidate_hit_block_ids))
        for block_id in hit_ids:
            adjust_ref(block_id, 1)
        prefill_end = candidate.prompt_tokens + candidate.output_tokens
        candidate_virtual = _VirtualRequest(
            candidate.request_id,
            prefill_end,
            candidate.computed_tokens,
            prefill_end + predicted,
            tuple(hit_ids),
            len(hit_ids),
        )
        allocate_to_progress(candidate_virtual, candidate_virtual.progress)
        peak = live_blocks
        if peak > self.capacity_blocks:
            return decision(False)

        # Remaining tokens: the candidate's rest of prefill plus a forecast;
        # one forecast for each running request (its ``target`` field here).
        remaining = prefill_end + predicted - candidate.computed_tokens
        prefill_left = prefill_end - candidate.computed_tokens
        order = sorted(
            (
                _VirtualRequest(
                    request.request_id,
                    request.prompt_tokens + request.output_tokens,
                    request.computed_tokens,
                    max(1, predicted),
                    tuple(request.block_ids),
                    len(request.block_ids),
                )
                for request in running
            ),
            key=lambda request: request.target,
        )
        prefill_rate = max(1, self.token_budget - len(order))
        iteration = 0
        counted_prefill = 0
        prefill_iterations = 0
        index = 0
        while index < len(order):
            if remaining - counted_prefill - (iteration - prefill_iterations) == 0:
                return decision(True)
            request = order[index]
            step = min(request.target, remaining) - iteration
            if step <= 0:
                release(request)
                index += 1
                continue
            for other in order[index:]:
                other.progress += step
                allocate_to_progress(other, other.progress)
            peak = max(peak, live_blocks)
            if peak > self.capacity_blocks:
                return decision(False)
            prefill_steps = -(-prefill_left // prefill_rate)
            if prefill_steps < step:
                grow = prefill_left + step - prefill_steps
                counted_prefill = prefill_left
                prefill_iterations = prefill_steps
                prefill_left = 0
            else:
                grow = prefill_rate * step
                counted_prefill += grow
                prefill_iterations += step
                prefill_left -= grow
            candidate_virtual.progress += grow
            allocate_to_progress(candidate_virtual, candidate_virtual.progress)
            peak = max(peak, live_blocks)
            if peak > self.capacity_blocks:
                return decision(False)
            iteration += step
        return decision(True)
