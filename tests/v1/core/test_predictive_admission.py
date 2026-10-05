# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.v1.core.sched.predictive_admission import (
    BatchCostModel,
    CausalOutputLengthEstimator,
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
) -> RequestSnapshot:
    return RequestSnapshot(
        request_id=request_id,
        prompt_tokens=prompt,
        computed_tokens=computed,
        output_tokens=output,
        block_ids=blocks,
    )


def train(controller: PredictiveAdmissionController, output: int) -> None:
    for _ in range(32):
        controller.observe_completion(output)


def make_controller(capacity: int, predicted: int = 1) -> PredictiveAdmissionController:
    controller = PredictiveAdmissionController(
        block_size=4,
        capacity_blocks=capacity,
        token_budget=8,
        long_prefill_token_threshold=0,
        max_model_len=128,
    )
    train(controller, predicted)
    return controller


def test_causal_estimator_uses_completed_lengths_and_survival():
    estimator = CausalOutputLengthEstimator()
    assert estimator.predict() == 512
    for length in range(1, 32):
        estimator.observe(length)
    assert estimator.predict() == 512
    estimator = CausalOutputLengthEstimator()
    for length in range(1, 101):
        estimator.observe(length)
    assert estimator.count == 100
    assert estimator.predict() == 90
    assert estimator.target(0) == 90
    assert estimator.target(90) == 99
    assert estimator.target(100) == 190
    for _ in range(256):
        estimator.observe(20)
    assert estimator.count == 256
    assert estimator.predict() == 20


def test_causal_estimator_quantile_history_and_reset():
    median = CausalOutputLengthEstimator(quantile=0.5)
    for length in range(1, 101):
        median.observe(length)
    assert median.predict() == 50
    assert median.target(50) == 75

    recent = CausalOutputLengthEstimator(quantile=0.9, history=16)
    assert recent.min_samples == 8
    for _ in range(7):
        recent.observe(3000)
    assert recent.predict() == 512
    recent.observe(3000)
    assert recent.predict() == 3000
    for _ in range(16):
        recent.observe(64)
    assert recent.count == 16
    assert recent.predict() == 64

    recent.reset()
    assert recent.count == 0
    assert recent.sorted_lengths == []
    assert recent.predict() == 512


def test_controller_passes_estimator_settings_and_resets():
    controller = PredictiveAdmissionController(
        block_size=4,
        capacity_blocks=8,
        token_budget=8,
        long_prefill_token_threshold=0,
        max_model_len=128,
        output_quantile=0.5,
        output_history=32,
    )
    assert controller.estimator.quantile == 0.5
    assert controller.estimator.max_history == 32
    train(controller, 7)
    assert controller.estimator.predict() == 7
    controller.reset_history()
    assert controller.estimator.count == 0


@pytest.mark.parametrize("kwargs", [{"quantile": 0.0}, {"quantile": 1.5},
                                    {"history": 4}])
def test_causal_estimator_rejects_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        CausalOutputLengthEstimator(**kwargs)


def test_causal_admission_uses_cold_forecast():
    controller = PredictiveAdmissionController(
        block_size=4, capacity_blocks=2, token_budget=8,
        long_prefill_token_threshold=0, max_model_len=128,
    )
    decision = controller.evaluate(
        [], make_request("candidate", prompt=4, computed=4),
        candidate_hit_block_ids=(), block_ref_counts={},
    )
    assert decision.predicted_output_tokens == 512
    assert not decision.feasible


def test_candidate_growth_is_rejected_at_capacity_peak():
    controller = make_controller(capacity=2, predicted=2)
    decision = controller.evaluate(
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
    assert decision.peak_pinned_blocks == 1


def test_unpinned_prefix_hit_becomes_pinned():
    controller = make_controller(capacity=1, predicted=2)
    decision = controller.evaluate(
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
        max_model_len=128,
    )
    train(controller, 1)
    decision = controller.evaluate(
        [],
        make_request("candidate", prompt=8, computed=4),
        candidate_hit_block_ids=(),
        block_ref_counts={},
    )
    assert decision.feasible
    assert decision.peak_pinned_blocks == 2


def test_completed_request_releases_blocks_before_later_growth():
    controller = PredictiveAdmissionController(
        block_size=4,
        capacity_blocks=3,
        token_budget=1,
        long_prefill_token_threshold=0,
        max_model_len=128,
    )
    train(controller, 1)
    decision = controller.evaluate(
        [make_request("running", prompt=4, computed=4, blocks=(7,))],
        make_request("candidate", prompt=8, computed=4),
        candidate_hit_block_ids=(),
        block_ref_counts={7: 1},
    )
    assert decision.feasible
    assert decision.peak_pinned_blocks == 3


@pytest.mark.parametrize("prompt,output", [(4, 1), (1, 4)])
@pytest.mark.parametrize("use_bound", [False, True])
def test_last_sampled_token_needs_no_kv_slot(prompt, output, use_bound):
    controller = make_controller(1, predicted=output)
    decision = controller.evaluate(
        [],
        make_request("candidate", prompt=prompt, computed=prompt),
        candidate_hit_block_ids=(),
        block_ref_counts={},
        use_capacity_bound=use_bound,
    )
    assert decision.feasible
    assert decision.peak_pinned_blocks == 1


def test_current_step_peak_precedes_completion_release():
    controller = make_controller(1)
    decision = controller.evaluate(
        [make_request("running", prompt=4, computed=4, blocks=(7,))],
        make_request("candidate", prompt=4, computed=4),
        candidate_hit_block_ids=(),
        block_ref_counts={7: 1},
    )
    assert not decision.feasible
    assert decision.peak_pinned_blocks == 2


def test_upper_bound_failure_uses_completion_releases():
    controller = make_controller(2)
    decision = controller.evaluate(
        [make_request("running", prompt=4, computed=4, blocks=(7,))],
        make_request("candidate", prompt=8, computed=4),
        candidate_hit_block_ids=(),
        block_ref_counts={7: 1},
    )
    assert decision.feasible
    assert not decision.peak_is_upper_bound
    assert decision.peak_pinned_blocks == 2


def test_model_length_caps_predicted_growth():
    controller = make_controller(1)
    controller.max_model_len = 5
    decision = controller.evaluate(
        [],
        make_request("candidate", prompt=4, computed=4),
        candidate_hit_block_ids=(),
        block_ref_counts={},
    )
    assert decision.feasible
    assert decision.peak_pinned_blocks == 1


def test_observed_output_extends_underpredicted_target():
    controller = make_controller(1)
    decision = controller.evaluate(
        [],
        make_request("candidate", prompt=2, computed=5, output=3),
        candidate_hit_block_ids=(),
        block_ref_counts={},
    )
    assert not decision.feasible


def test_fast_forward_matches_stepwise_simulation():
    # Include shared blocks, unpinned hits, partial prefill, resumed output
    # backlog, and steps too small to decode every request at once.
    for budget in (1, 4, 8):
        for predicted in (1, 4, 9):
            for capacity in range(2, 11):
                for progress in (4, 6, 8):
                    controller = make_controller(capacity, predicted=predicted)
                    controller.token_budget = budget
                    running = [
                        make_request(
                            "running", prompt=4, computed=4, blocks=(7,), output=1
                        )
                    ]
                    candidate = make_request(
                        "candidate", prompt=8, computed=progress, output=3
                    )
                    kwargs = dict(
                        candidate_hit_block_ids=(7,),
                        block_ref_counts={7: 1},
                        use_capacity_bound=False,
                    )
                    reference = controller.evaluate(
                        running, candidate, use_fast_forward=False, **kwargs
                    )
                    fast = controller.evaluate(running, candidate, **kwargs)
                    assert fast.feasible == reference.feasible
                    # In a rejected jump, the endpoint may exceed the first
                    # overflowing step. Successful trajectories have equal peaks.
                    if fast.feasible:
                        assert fast.peak_pinned_blocks == reference.peak_pinned_blocks
                    bound = controller.evaluate(
                        running,
                        candidate,
                        candidate_hit_block_ids=(7,),
                        block_ref_counts={7: 1},
                    )
                    assert bound.feasible == reference.feasible
                    if bound.peak_is_upper_bound:
                        assert bound.peak_pinned_blocks >= reference.peak_pinned_blocks


@pytest.mark.parametrize("capacity,feasible", [(4, False), (5, True)])
def test_shared_block_survives_until_last_reference_finishes(capacity, feasible):
    controller = make_controller(capacity)
    decision = controller.evaluate(
        [
            make_request("first", prompt=4, computed=4, blocks=(7,)),
            make_request(
                "last", prompt=4, computed=4, blocks=(7,), output=1
            ),
        ],
        make_request("candidate", prompt=12, computed=4),
        candidate_hit_block_ids=(),
        block_ref_counts={7: 2},
        use_capacity_bound=False,
    )
    # First completes in the current step. Last retains block 7 and grows
    # one private block while the candidate grows to three private blocks.
    assert decision.feasible is feasible
    assert decision.peak_pinned_blocks == 5


@pytest.mark.parametrize("budget,allowed", [(0, False), (4, False), (5, True)])
def test_replay_budget_prices_private_tail_and_restart(budget, allowed):
    controller = make_controller(3, predicted=4)
    decision = controller.evaluate(
        [make_request("running", prompt=4, computed=4, blocks=(7,))],
        make_request("candidate", prompt=4, computed=4),
        candidate_hit_block_ids=(),
        block_ref_counts={7: 1},
        replay_budget_tokens=budget,
    )
    # Next decode allocates two new blocks: one over capacity. Evicting
    # the candidate's last block loses one token; restart costs four more.
    assert not decision.feasible
    assert decision.replay_allowed is allowed
    if budget:
        assert decision.estimated_replay_tokens == 1


def test_replay_budget_never_overrides_current_step_capacity():
    controller = make_controller(1)
    decision = controller.evaluate(
        [make_request("running", prompt=4, computed=4, blocks=(7,))],
        make_request("candidate", prompt=4, computed=4),
        candidate_hit_block_ids=(),
        block_ref_counts={7: 1},
        replay_budget_tokens=100,
    )
    assert not decision.feasible and not decision.replay_allowed


def test_replay_budget_cannot_rescue_oversized_solo_request():
    controller = make_controller(1, predicted=4)
    decision = controller.evaluate(
        [],
        make_request("candidate", prompt=4, computed=4),
        candidate_hit_block_ids=(),
        block_ref_counts={},
        replay_budget_tokens=100,
    )
    assert not decision.feasible and not decision.replay_allowed


def test_replay_estimate_includes_shared_suffix_after_first_missing_block():
    controller = make_controller(3, predicted=4)
    controller.token_budget = 4
    decision = controller.evaluate(
        [make_request("running", prompt=12, computed=4, blocks=(8,))],
        make_request("candidate", prompt=8, computed=8, output=1),
        candidate_hit_block_ids=(7, 8),
        block_ref_counts={7: 0, 8: 1},
        replay_budget_tokens=100,
    )
    # Running grows by two blocks; candidate's shared tail cannot be freed.
    # Evicting its first block loses the entire contiguous eight-token prefix.
    assert not decision.feasible
    assert decision.replay_allowed
    assert decision.estimated_replay_tokens == 8


def test_replay_budget_rejects_when_tail_cannot_cover_deficit():
    controller = make_controller(2, predicted=4)
    decision = controller.evaluate(
        [make_request("running", prompt=12, computed=4, blocks=(7,))],
        make_request("candidate", prompt=4, computed=4, output=1),
        candidate_hit_block_ids=(7,),
        block_ref_counts={7: 1},
        replay_budget_tokens=100,
    )
    assert not decision.feasible and not decision.replay_allowed


def test_cost_fast_forward_locates_first_overflow_like_stepwise():
    for capacity in range(2, 12):
        for budget in (1, 5, 20, 100):
            controller = make_controller(capacity, predicted=16)
            args = (
                [make_request("running", prompt=4, computed=4, blocks=(7,))],
                make_request("candidate", prompt=4, computed=4),
            )
            kwargs = dict(
                candidate_hit_block_ids=(), block_ref_counts={7: 1},
                replay_budget_tokens=budget, use_capacity_bound=False,
            )
            assert controller.evaluate(*args, **kwargs) == controller.evaluate(
                *args, **kwargs, use_fast_forward=False,
            )


def test_batch_cost_needs_measurements_and_zero_ratio_stays_strict():
    model = BatchCostModel()
    assert model.replay_budget(100, 0, 100, 0, 0.05) == (0, 0.0)
    for _ in range(3):
        model.observe(100, 0, 0.1)
        model.observe(200, 0, 0.12)
    assert model.replay_budget(100, 0, 100, 0, 0) == (0, 0.0)
    assert model.replay_budget(0, 0, 100, 0, 0.05)[0] == 0


def test_batch_cost_saving_limits_budget_and_replay_corrects_it():
    model = BatchCostModel()
    for _ in range(3):
        model.observe(100, 0, 0.1)
        model.observe(200, 0, 0.12)
    budget, saving = model.replay_budget(100, 0, 100, 0, 0.05)
    assert saving == pytest.approx(0.08)
    assert budget == 6
    model.estimated_replay_tokens = 10
    model.actual_replay_tokens = 30
    assert model.replay_budget(100, 0, 100, 0, 0.05)[0] == 2
    saturated = BatchCostModel()
    for _ in range(3):
        saturated.observe(100, 0, 0.1)
        saturated.observe(200, 0, 0.2)
    assert saturated.replay_budget(100, 0, 100, 0, 0.05)[0] == 0


def test_batch_cost_decode_curve_is_separate_from_prefill():
    model = BatchCostModel()
    for _ in range(3):
        model.observe(100, 0, 0.1)
        model.observe(0, 8, 0.02)
        model.observe(0, 16, 0.025)
    assert model.estimate(0, 8) == pytest.approx(0.02)
    assert model.estimate(8, 0) == pytest.approx(0.008)
    assert model.estimate(0, 12) == pytest.approx(0.0225)
    assert model.replay_budget(0, 8, 0, 8, 0.05)[1] == pytest.approx(0.015)
