# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import pytest

from vllm.v1.core.sched.predictive_admission import (
    OnlineOutputLengthEstimator,
    PredictiveAdmissionController,
    RequestSnapshot,
)

pytestmark = pytest.mark.cpu_test


def make_request(
    request_id: str,
    *,
    prompt: int,
    computed: int,
    blocks: tuple[int, ...] = (),
    output: int = 0,
    max_output: int = 16,
) -> RequestSnapshot:
    return RequestSnapshot(
        request_id=request_id,
        prompt_tokens=prompt,
        computed_tokens=computed,
        output_tokens=output,
        max_output_tokens=max_output,
        block_ids=blocks,
    )


def make_controller(capacity: int) -> PredictiveAdmissionController:
    return PredictiveAdmissionController(
        block_size=4,
        capacity_blocks=capacity,
        token_budget=8,
        long_prefill_token_threshold=0,
    )


def test_online_estimator_matches_pkas_rule():
    estimator = OnlineOutputLengthEstimator()
    assert estimator.predict() == 1
    estimator.observe(10)
    assert estimator.predict() == 1
    estimator.observe(14)
    expected = math.ceil(abs(12 - math.sqrt(8)))
    assert estimator.predict() == expected

    zero_estimator = OnlineOutputLengthEstimator()
    zero_estimator.observe(0)
    zero_estimator.observe(0)
    assert zero_estimator.predict() == 0


def test_candidate_growth_is_rejected_at_capacity_peak():
    decision = make_controller(capacity=2).evaluate(
        [],
        make_request("candidate", prompt=8, computed=8),
        candidate_hit_block_ids=(),
        block_ref_counts={},
    )
    assert not decision.feasible
    assert decision.peak_pinned_blocks == 3


def test_shared_prefix_is_counted_once():
    controller = make_controller(capacity=3)
    decision = controller.evaluate(
        [make_request("running", prompt=4, computed=4, blocks=(7,))],
        make_request("candidate", prompt=4, computed=4),
        candidate_hit_block_ids=(7,),
        block_ref_counts={7: 1},
    )
    assert decision.feasible
    assert decision.prefix_hit_blocks == 1
    assert decision.peak_pinned_blocks == 3


def test_unpinned_prefix_hit_becomes_pinned():
    decision = make_controller(capacity=1).evaluate(
        [],
        make_request("candidate", prompt=4, computed=4),
        candidate_hit_block_ids=(7,),
        block_ref_counts={7: 0},
    )
    assert not decision.feasible
    assert decision.peak_pinned_blocks == 2


def test_chunked_prefill_then_decode_is_simulated():
    controller = PredictiveAdmissionController(
        block_size=4,
        capacity_blocks=4,
        token_budget=4,
        long_prefill_token_threshold=4,
    )
    decision = controller.evaluate(
        [],
        make_request("candidate", prompt=8, computed=4),
        candidate_hit_block_ids=(),
        block_ref_counts={},
    )
    assert decision.feasible
    assert decision.peak_pinned_blocks == 3


def test_completed_request_releases_blocks_before_later_growth():
    controller = PredictiveAdmissionController(
        block_size=4,
        capacity_blocks=3,
        token_budget=1,
        long_prefill_token_threshold=0,
    )
    decision = controller.evaluate(
        [make_request("running", prompt=4, computed=4, blocks=(7,))],
        make_request("candidate", prompt=8, computed=4),
        candidate_hit_block_ids=(),
        block_ref_counts={7: 1},
    )
    assert decision.feasible
    assert decision.peak_pinned_blocks == 3
