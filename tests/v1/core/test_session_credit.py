# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SSPF session-history credit (KVReuse ``simulator/kvsim.py`` ``add_waiting``
and ``Sim.rank``): credit = leading prompt blocks shared with the session's
previous arrived turn, fixed at arrival; each credited block past the usable
hit lowers the price by ``(1 - w) * block_size`` tokens."""

import pytest

from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.sched.session_credit import (
    SessionCreditTracker,
    common_prefix_blocks,
    session_id_of,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request

from .utils import EOS_TOKEN_ID, create_scheduler

BS = 16
DOC = list(range(1, 65))  # 4 full blocks


@pytest.fixture(autouse=True, scope="module")
def _fixed_none_hash():
    init_none_hash(sha256)


def _request(
    request_id: str,
    tokens: list[int],
    session_id: str | None = None,
    max_tokens: int = 4,
) -> Request:
    extra = None if session_id is None else {"session_id": session_id}
    params = SamplingParams(max_tokens=max_tokens, ignore_eos=True, extra_args=extra)
    params.update_from_generation_config({}, EOS_TOKEN_ID)
    return Request(
        request_id=request_id,
        prompt_token_ids=tokens,
        sampling_params=params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(BS, sha256),
    )


def _scheduler(weight: float = 0.5, **kwargs) -> Scheduler:
    return create_scheduler(
        enable_prefix_caching=True,
        scheduling_policy="sspf",
        lpm_window_size=0,
        spf_age_alpha=0.0,
        sspf_session_credit_weight=weight,
        **kwargs,
    )


def _step(scheduler: Scheduler, token: int = 7) -> None:
    output = scheduler.schedule()
    req_ids = list(output.num_scheduled_tokens)
    sampled = []
    for rid in req_ids:
        request = scheduler.requests[rid]
        done = request.num_computed_tokens >= request.num_tokens
        sampled.append([token] if done else [])
    scheduler.update_from_output(
        output,
        ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
            sampled_token_ids=sampled,
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )


# ---------------------------------------------------------------- tracker


def test_common_prefix_blocks():
    assert common_prefix_blocks([1, 2, 3], [1, 2, 4, 5]) == 2
    assert common_prefix_blocks([1, 2], [1, 2, 3]) == 2
    assert common_prefix_blocks([9], [1]) == 0
    assert common_prefix_blocks([], [1]) == 0


def test_credit_is_against_the_previous_arrived_turn():
    tracker = SessionCreditTracker()
    assert tracker.on_arrival("t1", "s", (1, 2)) == 0
    assert tracker.on_arrival("t2", "s", (1, 2, 3)) == 2
    assert tracker.on_arrival("t3", "s", (1, 2, 3, 4)) == 3  # vs t2, not t1
    assert tracker.on_arrival("x1", "other", (1, 2, 3)) == 0  # sessions apart
    assert [tracker.credit(r) for r in ("t1", "t2", "t3")] == [0, 2, 3]
    assert tracker.drop("t3") == 3
    assert tracker.credit("t3") == 0


def test_session_table_is_bounded_lru():
    tracker = SessionCreditTracker(max_sessions=2)
    tracker.on_arrival("a1", "a", (1,))
    tracker.on_arrival("b1", "b", (2,))
    tracker.on_arrival("a2", "a", (1,))  # touches a: b is now the oldest
    tracker.on_arrival("c1", "c", (3,))  # evicts b
    assert len(tracker) == 2
    assert tracker.on_arrival("b2", "b", (2,)) == 0
    assert tracker.on_arrival("a3", "a", (1,)) == 0  # evicted by b2
    tracker.clear_sessions()
    assert len(tracker) == 0


def test_session_id_comes_from_extra_args_and_vllm_xargs():
    assert session_id_of(_request("r", DOC, "s1")) == "s1"
    assert session_id_of(_request("r", DOC)) is None
    params = CompletionRequest(
        model="m", prompt=[1, 2, 3], vllm_xargs={"session_id": "s9"}
    ).to_sampling_params(max_tokens=4)
    assert params.extra_args == {"session_id": "s9"}


# ---------------------------------------------------------------- scheduler


def test_credit_lowers_the_price_of_evicted_history():
    costs = {}
    for weight in (1.0, 0.5):
        scheduler = _scheduler(weight, max_num_seqs=1)
        scheduler.add_request(_request("t1", DOC, "s"))
        scheduler.add_request(_request("t2", DOC + list(range(100, 132)), "s"))
        scheduler.add_request(_request("single", list(range(500, 548))))
        scheduler.schedule()  # admits t1 (32 tokens; ties t2 at w=0.5, first)
        costs[weight] = dict(scheduler._sspf_costs)
    # t2: 4 blocks shared with t1 (1/2 each) + 2 own blocks = 4 blocks = 64.
    assert costs[1.0] == {"t2": 64.0, "single": 48.0}
    # Credit 4 blocks, no hit: 64 - 0.5 * 16 * 4 = 32.
    assert costs[0.5] == {"t2": 32.0, "single": 48.0}


def test_generated_blocks_are_never_credited():
    scheduler = _scheduler()
    scheduler.add_request(_request("t1", DOC, "s", max_tokens=32))
    for _ in range(18):  # prefill, then 17 outputs: one full generated block
        _step(scheduler)
    t1 = scheduler.requests["t1"]
    assert len(t1.block_hashes) == 5
    # The next turn's prompt continues with exactly t1's generated block.
    scheduler.add_request(_request("t2", DOC + [7] * 16 + [8] * 16, "s"))
    assert scheduler._session_credit.credit("t2") == 4


def test_credit_is_dropped_at_admission_and_not_recomputed():
    scheduler = _scheduler()
    scheduler.add_request(_request("t1", DOC, "s"))
    scheduler.add_request(_request("t2", DOC + list(range(100, 116)), "s"))
    tracker = scheduler._session_credit
    assert tracker.credit("t2") == 4
    _step(scheduler)  # both admitted
    assert tracker.credit("t2") == 0
    t2 = scheduler.requests["t2"]
    scheduler.running.remove(t2)
    scheduler._preempt_request(t2, 0.0)  # re-tracked, credit stays 0
    assert tracker.credit("t2") == 0


def test_requests_without_session_id_get_no_credit():
    scheduler = _scheduler()
    scheduler.add_request(_request("a", DOC))
    scheduler.add_request(_request("b", DOC))
    assert len(scheduler._session_credit) == 0
    assert scheduler._session_credit.credit("b") == 0


def test_weight_one_creates_no_tracker():
    scheduler = _scheduler(1.0)
    assert scheduler._session_credit is None
    scheduler.add_request(_request("a", DOC, "s"))
    scheduler.add_request(_request("b", DOC, "s"))
    _step(scheduler)


def test_credit_requires_sspf():
    with pytest.raises(ValueError, match="sspf_session_credit_weight"):
        create_scheduler(
            enable_prefix_caching=True,
            scheduling_policy="spf",
            sspf_session_credit_weight=0.5,
        )


@pytest.mark.parametrize("reset_running", [False, True])
def test_prefix_cache_reset_forgets_sessions(reset_running):
    scheduler = _scheduler()
    scheduler.add_request(_request("warm", DOC, "s"))
    if reset_running:
        _step(scheduler)  # warm is running and holds KV
    assert len(scheduler._session_credit) == 1
    assert scheduler.reset_prefix_cache(reset_running_requests=reset_running)
    assert len(scheduler._session_credit) == 0
