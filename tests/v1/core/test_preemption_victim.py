# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Least-done preemption victim (SSPF series), against stock LIFO.

The simulator's rule (KVReuse ``simulator/sched-sim/kvsim.py``
``pick_victim``): preempt the running request with the fewest computed
tokens, the earliest admitted on ties, repeating until the allocation fits or
the requester itself is chosen; then go on with the next running request.
Decision parity is tested in ``simulator/sched-sim/test_vllm_parity.py``.
"""

import pytest

from vllm.config import SchedulerConfig
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request

from .utils import EOS_TOKEN_ID, create_scheduler

BS = 16


@pytest.fixture(autouse=True, scope="module")
def _fixed_none_hash():
    init_none_hash(sha256)


def _request(request_id: str, num_tokens: int, start: int) -> Request:
    params = SamplingParams(max_tokens=32, ignore_eos=True)
    params.update_from_generation_config({}, EOS_TOKEN_ID)
    return Request(
        request_id=request_id,
        prompt_token_ids=list(range(start, start + num_tokens)),
        sampling_params=params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(BS, sha256),
    )


def _step(scheduler: Scheduler) -> tuple[dict[str, int], list[str]]:
    """Run one step; return scheduled tokens and preemption victims in order."""
    victims: list[str] = []
    preempt = scheduler._preempt_request

    def record(request, *args, **kwargs):
        victims.append(request.request_id)
        return preempt(request, *args, **kwargs)

    scheduler._preempt_request = record
    try:
        output = scheduler.schedule()
    finally:
        del scheduler._preempt_request
    req_ids = list(output.num_scheduled_tokens)
    sampled = []
    for rid in req_ids:
        request = scheduler.requests[rid]
        done = request.num_computed_tokens >= request.num_tokens
        sampled.append([7] if done else [])
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
    return dict(output.num_scheduled_tokens), victims


def _running(victim: str, num_blocks: int, prompts: list[tuple[str, int]]):
    """Admit every prompt in step 1 (their prefills fill the usable pool)."""
    scheduler = create_scheduler(
        num_blocks=num_blocks,  # includes the null block
        preemption_victim=victim,
    )
    for i, (rid, num_tokens) in enumerate(prompts):
        scheduler.add_request(_request(rid, num_tokens, 1000 * (i + 1)))
    scheduled, victims = _step(scheduler)
    assert list(scheduled) == [rid for rid, _ in prompts]
    assert victims == []
    return scheduler


def test_lifo_is_the_default():
    assert SchedulerConfig.preemption_victim == "lifo"


@pytest.mark.parametrize(
    ("victim", "expected_victims", "expected_scheduled"),
    [
        # a (48 computed) needs a 4th block; the pool (6 usable) is full.
        # LIFO preempts c, freeing 2 blocks: a and b both grow.
        ("lifo", ["c"], {"a": 1, "b": 1}),
        # least_done preempts b (16 computed), then c needs a block and is
        # itself the least done of {a, c}: it is preempted too.
        ("least_done", ["b", "c"], {"a": 1}),
    ],
)
def test_victim_rule(victim, expected_victims, expected_scheduled):
    scheduler = _running(victim, 7, [("a", 48), ("b", 16), ("c", 32)])
    scheduled, victims = _step(scheduler)
    assert victims == expected_victims
    assert scheduled == expected_scheduled


@pytest.mark.parametrize(
    ("victim", "expected_victims", "expected_scheduled"),
    [
        ("lifo", ["c"], {"a": 1, "b": 1}),
        # a and b tie at 16 computed tokens: the earliest admitted, a, goes
        # first and preempts itself. The loop goes on: b takes a's block,
        # then c needs one and unschedules b (the least done).
        ("least_done", ["a", "b"], {"c": 1}),
    ],
)
def test_least_done_ties_go_to_the_earliest_admitted(
    victim, expected_victims, expected_scheduled
):
    scheduler = _running(victim, 6, [("a", 16), ("b", 16), ("c", 48)])
    scheduled, victims = _step(scheduler)
    assert victims == expected_victims
    assert scheduled == expected_scheduled


@pytest.mark.parametrize(
    ("victim", "expected_victims", "expected_scheduled"),
    [
        # a (16 computed) needs a 2nd block; the pool (5 usable) is full.
        # LIFO preempts c, freeing 2 blocks: a and b both grow.
        ("lifo", ["c"], {"a": 1, "b": 1}),
        # least_done: a is the least done and preempts itself; b and c are
        # mid-block and still decode in the same step.
        ("least_done", ["a"], {"b": 1, "c": 1}),
    ],
)
def test_least_done_self_preemption_keeps_scheduling(
    victim, expected_victims, expected_scheduled
):
    scheduler = _running(victim, 6, [("a", 16), ("b", 20), ("c", 20)])
    scheduled, victims = _step(scheduler)
    assert victims == expected_victims
    assert scheduled == expected_scheduled
    assert {r.request_id for r in scheduler.running} == set(expected_scheduled)


@pytest.mark.parametrize(
    ("victim", "expected_victims", "expected_scheduled"),
    [
        ("lifo", ["b"], {"a": 1}),
        # a (20 computed) was scheduled earlier this step without a new
        # block; b needs one. least_done preempts a and returns its budget
        # and blocks, so b grows instead.
        ("least_done", ["a"], {"b": 1}),
    ],
)
def test_least_done_unschedules_a_victim_scheduled_earlier(
    victim, expected_victims, expected_scheduled
):
    scheduler = _running(victim, 6, [("a", 20), ("b", 48)])
    scheduled, victims = _step(scheduler)
    assert victims == expected_victims
    assert scheduled == expected_scheduled
    assert {r.request_id for r in scheduler.running} == set(expected_scheduled)


def test_priority_policy_rejects_least_done():
    with pytest.raises(ValueError, match="preemption_victim"):
        create_scheduler(scheduling_policy="priority", preemption_victim="least_done")


def test_pkas_rejects_least_done():
    with pytest.raises(ValueError, match="preemption_victim"):
        create_scheduler(
            scheduling_policy="lpm",
            enable_prefix_caching=True,
            admission_policy="pkas",
            preemption_victim="least_done",
        )


def test_async_scheduling_rejects_least_done():
    with pytest.raises(ValueError, match="least_done"):
        create_scheduler(async_scheduling=True, preemption_victim="least_done")
