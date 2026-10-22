# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.v1.core.sched.predictive_admission import (
    CausalOutputLengthEstimator,
    PaperOutputLengthEstimator,
    PredictiveAdmissionController,
    QuantileOutputLengthEstimator,
    RequestSnapshot,
    min_quantile_samples,
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


def test_paper_estimator_follows_eq1():
    estimator = PaperOutputLengthEstimator()
    assert estimator.predict() == 1
    estimator.observe(100)
    assert estimator.predict() == 1  # N <= 1
    for length in (200, 300):
        estimator.observe(length)
    # mean 200, sample std 100: ceil(|200 - 100|) = 100.
    assert estimator.predict() == 100
    assert estimator.target(50) == 100
    # At or past the forecast: one more token.
    assert estimator.target(100) == 101
    assert estimator.target(150) == 151
    estimator.reset()
    assert estimator.count == 0 and estimator.predict() == 1


def test_paper_estimator_uses_absolute_value_for_heavy_tails():
    estimator = PaperOutputLengthEstimator()
    for length in (1, 1, 1, 97):
        estimator.observe(length)
    # mean 25, sample std 48: ceil(|25 - 48|) = 23.
    assert estimator.predict() == 23


def test_min_quantile_samples_is_distribution_free_bound():
    # Smallest n with q**n <= 1 - confidence.
    assert min_quantile_samples(0.5, 0.95) == 5
    assert min_quantile_samples(0.9, 0.95) == 29
    assert 0.5**5 <= 0.05 < 0.5**4
    for quantile, confidence in ((0.0, 0.95), (1.0, 0.95), (0.5, 1.0)):
        with pytest.raises(ValueError):
            min_quantile_samples(quantile, confidence)


def test_quantile_estimator_waits_for_a_bounding_sample():
    estimator = QuantileOutputLengthEstimator(0.5)  # needs 5 samples
    for length in (10, 20, 30, 40):
        estimator.observe(length)
    assert estimator.predict() == 1
    assert estimator.target(0) == 1
    assert estimator.target(25) == 26
    estimator.observe(50)
    assert estimator.predict() == 30
    assert estimator.target(10) == 30
    # Two completed outputs are longer than 30: too few to bound the
    # conditional median, so use the longest completed output.
    assert estimator.target(30) == 50
    assert estimator.target(50) == 51


def test_quantile_estimator_conditional_tail_and_history():
    estimator = QuantileOutputLengthEstimator(0.5)
    for length in range(1, 21):
        estimator.observe(length)
    assert estimator.max_history == 0 and estimator.count == 20
    assert estimator.predict() == 10  # median of 1..20 (index 9)
    assert estimator.target(5) == 10
    # Past the forecast: 8 outputs exceed 12, enough to bound their median,
    # which is 16 (index 3 of 13..20).
    assert estimator.target(12) == 16
    windowed = QuantileOutputLengthEstimator(0.5, history=6)
    for length in range(1, 21):
        windowed.observe(length)
    assert windowed.sorted_lengths == list(range(15, 21))
    with pytest.raises(ValueError):
        QuantileOutputLengthEstimator(0.5, history=4)


@pytest.mark.parametrize("predictor,kind", [
    ("paper", PaperOutputLengthEstimator),
    ("quantile", QuantileOutputLengthEstimator),
    ("legacy", CausalOutputLengthEstimator),
])
def test_controller_selects_output_predictor(predictor, kind):
    controller = PredictiveAdmissionController(
        block_size=4, capacity_blocks=8, token_budget=8,
        long_prefill_token_threshold=0, max_model_len=128,
        output_predictor=predictor, output_quantile=0.5,
    )
    assert isinstance(controller.estimator, kind)
    if predictor == "quantile":
        assert controller.estimator.max_history == 0
    if predictor == "legacy":
        assert controller.estimator.max_history == 256
    with pytest.raises(ValueError):
        PredictiveAdmissionController(
            block_size=4, capacity_blocks=8, token_budget=8,
            long_prefill_token_threshold=0, max_model_len=128,
            output_predictor="unknown",
        )


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


def test_fast_forward_matches_stepwise_simulation():
    # Jumping to the next completion can overshoot the first overflowing step,
    # so only the admission decision, not the reported peak, must match.
    for capacity in range(2, 12):
        controller = make_controller(capacity, predicted=16)
        args = (
            [make_request("running", prompt=4, computed=4, blocks=(7,))],
            make_request("candidate", prompt=4, computed=4),
        )
        kwargs = dict(
            candidate_hit_block_ids=(), block_ref_counts={7: 1},
            use_capacity_bound=False,
        )
        assert controller.evaluate(*args, **kwargs).feasible == controller.evaluate(
            *args, **kwargs, use_fast_forward=False,
        ).feasible


def make_paper_controller(
    capacity: int, predicted: int
) -> PredictiveAdmissionController:
    controller = PredictiveAdmissionController(
        block_size=4,
        capacity_blocks=capacity,
        token_budget=8,
        long_prefill_token_threshold=0,
        max_model_len=128,
        output_predictor="paper",
    )
    for _ in range(2):
        controller.observe_completion(predicted)
    return controller


@pytest.mark.parametrize("capacity,repo_feasible", [(5, False), (6, True)])
def test_repo_running_request_reserves_full_forecast_past_its_output(
    capacity, repo_feasible
):
    # The running request has produced 8 tokens against a 4-token forecast.
    # The paper text expects one more token; the repository reserves 4 more.
    controller = make_paper_controller(capacity, predicted=4)
    kwargs = dict(
        running=[make_request("running", prompt=4, computed=12, blocks=(1, 2, 3),
                              output=8)],
        candidate=make_request("candidate", prompt=4, computed=4),
        candidate_hit_block_ids=(),
        block_ref_counts={1: 1, 2: 1, 3: 1},
    )
    assert controller.evaluate(**kwargs).feasible
    decision = controller.evaluate_repo(**kwargs)
    assert decision.feasible is repo_feasible
    assert decision.peak_pinned_blocks == 6


@pytest.mark.parametrize("capacity,feasible", [(2, False), (3, True)])
def test_repo_shared_prefix_is_counted_once(capacity, feasible):
    controller = make_paper_controller(capacity, predicted=1)
    decision = controller.evaluate_repo(
        [make_request("running", prompt=4, computed=4, blocks=(7,))],
        make_request("candidate", prompt=4, computed=4),
        candidate_hit_block_ids=(7,),
        block_ref_counts={7: 1},
    )
    assert decision.feasible is feasible
    assert decision.prefix_hit_blocks == 1
    # Shared block 7 plus one decode block each; unshared it would be 4.
    assert decision.peak_pinned_blocks == 3


@pytest.mark.parametrize("capacity,feasible", [(0, False), (1, True)])
def test_repo_unpinned_prefix_hit_becomes_pinned(capacity, feasible):
    controller = make_paper_controller(capacity, predicted=2)
    decision = controller.evaluate_repo(
        [],
        make_request("candidate", prompt=4, computed=4),
        candidate_hit_block_ids=(7,),
        block_ref_counts={7: 0},
    )
    assert decision.feasible is feasible
    assert decision.peak_pinned_blocks == 1


def test_repo_without_running_requests_checks_only_the_next_chunk():
    controller = make_paper_controller(1, predicted=1)
    kwargs = dict(
        running=[],
        candidate=make_request("candidate", prompt=8, computed=4),
        candidate_hit_block_ids=(),
        block_ref_counts={},
    )
    assert not controller.evaluate(**kwargs).feasible
    decision = controller.evaluate_repo(**kwargs)
    assert decision.feasible
    assert decision.peak_pinned_blocks == 1


@pytest.mark.parametrize("capacity,feasible", [(7, False), (8, True)])
def test_repo_fast_forward_prefill_and_running_completion(capacity, feasible):
    # Forecast 2, budget 8: in 2 iterations the running request grows 2
    # tokens and the candidate prefills 2 * (8 - 1) = 14 tokens (22 of 32,
    # 6 blocks); the running request then finishes and the candidate is
    # admitted with prefill left, as in the repository.
    controller = make_paper_controller(capacity, predicted=2)
    decision = controller.evaluate_repo(
        [make_request("running", prompt=4, computed=4, blocks=(1,))],
        make_request("candidate", prompt=32, computed=8),
        candidate_hit_block_ids=(),
        block_ref_counts={1: 1},
    )
    assert decision.feasible is feasible
    assert decision.peak_pinned_blocks == 8
