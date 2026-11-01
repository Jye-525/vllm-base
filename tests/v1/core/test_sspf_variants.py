# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SSPF series: pricing, sharing semantics and per-step timing fields.

The pricing rules mirror the trace-driven simulator (KVReuse
``simulator/kvsim.py``); decision parity with it is tested there
(``simulator/test_vllm_parity.py``).
"""

import pytest

from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.sched.waiting_ranker import sspf_cost
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus

from .utils import EOS_TOKEN_ID, create_scheduler

BS = 16
TIMING_FIELDS = (
    "scheduler_ns",
    "running_ns",
    "rank_ns",
    "admit_ns",
    "arrival_ns",
    "output_processing_ns",
)


@pytest.fixture(autouse=True, scope="module")
def _fixed_none_hash():
    # Once per module: without PYTHONHASHSEED every call draws a new random
    # root hash, and requests built after different calls share no block.
    init_none_hash(sha256)


def _request(request_id: str, tokens: list[int], max_tokens: int = 4) -> Request:
    params = SamplingParams(max_tokens=max_tokens, ignore_eos=True)
    params.update_from_generation_config({}, EOS_TOKEN_ID)
    return Request(
        request_id=request_id,
        prompt_token_ids=tokens,
        sampling_params=params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(BS, sha256),
    )


def _scheduler(policy: str = "sspf", **kwargs) -> Scheduler:
    return create_scheduler(
        enable_prefix_caching=True,
        scheduling_policy=policy,
        lpm_window_size=0,
        spf_age_alpha=0.0,
        **kwargs,
    )


def _model_output(scheduler: Scheduler, output) -> ModelRunnerOutput:
    """One sampled token for every request whose prefill completed."""
    req_ids = list(output.num_scheduled_tokens)
    sampled = []
    for rid in req_ids:
        request = scheduler.requests[rid]
        done = request.num_computed_tokens >= request.num_tokens
        sampled.append([7] if done else [])
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
        sampled_token_ids=sampled,
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


# ---------------------------------------------------------------- pricing


@pytest.mark.parametrize(
    ("num_tokens", "num_full_blocks", "amortized", "expected"),
    [
        (80, 5, 4 / 4 + 1, 32.0),  # 4 shared blocks / 4 sharers + 1 own block
        (48, 3, 3.0, 48.0),  # single: equals SPF's uncached tokens
        (70, 4, 1.5, 30.0),  # 1.5 blocks + private 6-token tail
        (15, 0, 0.0, 15.0),  # tail only
    ],
)
def test_sspf_cost_matches_simulator_formula(
    num_tokens, num_full_blocks, amortized, expected
):
    # simulator: acost * BS + ntok - len(phash) * BS
    assert sspf_cost(num_tokens, num_full_blocks, BS, amortized) == expected


@pytest.mark.parametrize(
    ("hit_blocks", "credit_blocks", "weight", "expected"),
    [
        (0, 4, 1.0, 80.0),  # weight 1: plain SSPF, credit ignored
        (4, 4, 0.5, 80.0),  # whole credited history still cached: no discount
        (1, 4, 0.5, 56.0),  # 3 evicted credited blocks: 80 - 0.5 * 16 * 3
        (0, 4, 0.0, 16.0),  # w = 0 fully offsets evicted history
    ],
)
def test_sspf_cost_session_credit(hit_blocks, credit_blocks, weight, expected):
    # 5 full blocks, all private (amortized 5), price 80 before credit.
    assert sspf_cost(80, 5, BS, 5.0, hit_blocks, credit_blocks, weight) == expected


def test_sspf_cost_session_credit_is_not_clipped():
    # A shared credited block can leave a negative price (frozen rule).
    assert sspf_cost(64, 4, BS, 4 / 4, 0, 4, 0.0) == pytest.approx(16 - 64)


def test_sspf_without_sharing_orders_as_spf():
    prompts = {
        "a": list(range(1, 81)),
        "b": list(range(100, 132)),
        "c": list(range(200, 250)),
        "d": list(range(300, 333)),
    }
    orders = {}
    for policy in ("spf", "sspf"):
        scheduler = _scheduler(policy, max_num_seqs=1)
        for rid, tokens in prompts.items():
            scheduler.add_request(_request(rid, tokens))
        output = scheduler.schedule()
        orders[policy] = [r.req_id for r in output.scheduled_new_reqs] + [
            r.request_id for r in scheduler.waiting
        ]
    assert orders["sspf"] == orders["spf"] == ["b", "d", "c", "a"]


def test_preempted_request_counts_as_sharer():
    scheduler = _scheduler()
    doc = list(range(1, 65))
    a = _request("a", doc + list(range(100, 116)))
    b = _request("b", doc + list(range(200, 216)))
    scheduler.add_request(a)
    scheduler.add_request(b)
    index = scheduler.pending_prefix_index
    # Both waiting: each of the 4 shared blocks costs 1/2.
    assert index.amortized_blocks("b", 0) == pytest.approx(4 / 2 + 1)

    # A preempted request is re-tracked as a ready sharer (simulator: the
    # index holds fresh and preempted requests).
    index.discard("a")
    assert index.amortized_blocks("b", 0) == pytest.approx(4 + 1)
    a.status = RequestStatus.PREEMPTED
    scheduler._track_waiting_request(a)
    assert index.is_ready("a")
    assert index.amortized_blocks("b", 0) == pytest.approx(4 / 2 + 1)


def test_same_step_siblings_hit_the_leaders_published_blocks():
    """Design section B: G1 computes the document; G2-G4, admitted in the same
    step, hit it because allocation publishes scheduled full blocks."""
    scheduler = _scheduler()
    doc = list(range(1, 65))
    for i in range(4):
        own = list(range(1000 * (i + 1), 1000 * (i + 1) + 16))
        scheduler.add_request(_request(f"g{i}", doc + own))
    scheduler.add_request(_request("single", list(range(9000, 9048))))

    output = scheduler.schedule()

    hits = {
        rid: scheduler.requests[rid].num_computed_tokens - n
        for rid, n in output.num_scheduled_tokens.items()
    }
    # Group price 4/4 + 1 = 2 blocks < single's 3 blocks: the group goes first.
    assert list(hits) == ["g0", "g1", "g2", "g3", "single"]
    assert hits == {"g0": 0, "g1": 64, "g2": 64, "g3": 64, "single": 0}


def test_multichunk_leader_raises_sibling_price_until_published():
    """Design D.3.4: while the leader prefills in chunks it has left the
    queue, so its siblings split the unpublished shared blocks among
    themselves (1/2 instead of 1/3) until the blocks are cached."""
    scheduler = _scheduler(max_num_seqs=1, long_prefill_token_threshold=32)
    doc = list(range(1, 65))
    for rid, own in (("leader", 100), ("sib1", 200), ("sib2", 300)):
        scheduler.add_request(_request(rid, doc + list(range(own, own + 16))))
    index = scheduler.pending_prefix_index
    manager = scheduler.kv_cache_manager
    assert index.amortized_blocks("sib1", 0) == pytest.approx(4 / 3 + 1)

    output = scheduler.schedule()  # leader admitted, 32 tokens: blocks 0-1
    assert [r.req_id for r in output.scheduled_new_reqs] == ["leader"]
    sib1 = scheduler.requests["sib1"]
    hit_blocks = manager.get_computed_prefix_len(sib1) // BS
    assert hit_blocks == 2
    assert index.amortized_blocks("sib1", hit_blocks) == pytest.approx(2 / 2 + 1)


def test_sspf_prices_each_candidate_once_per_step(monkeypatch):
    scheduler = _scheduler(max_num_seqs=1)
    for rid, start in (("a", 1), ("b", 100), ("c", 200)):
        scheduler.add_request(_request(rid, list(range(start, start + 40))))
    index = scheduler.pending_prefix_index
    calls = []
    original = index.amortized_blocks

    def counted(request_id, start_block):
        calls.append(request_id)
        return original(request_id, start_block)

    monkeypatch.setattr(index, "amortized_blocks", counted)
    scheduler.schedule()
    assert sorted(calls) == ["a", "b", "c"]


# ---------------------------------------------------------------- timing


def _capture_batch_logs(monkeypatch) -> list[tuple[dict, int]]:
    records: list[tuple[dict, int]] = []

    def capture(record: dict, recomputed: int) -> None:
        records.append((dict(record), recomputed))

    monkeypatch.setattr(Scheduler, "_log_schedule_batch", staticmethod(capture))
    return records


@pytest.mark.parametrize("policy", ["fcfs", "lpm", "clpm", "spf", "sspf"])
def test_every_policy_logs_scheduling_times_once_per_step(monkeypatch, policy):
    records = _capture_batch_logs(monkeypatch)
    scheduler = _scheduler(policy)
    scheduler.add_request(_request("a", list(range(1, 41)), max_tokens=2))
    scheduler.add_request(_request("b", list(range(100, 140)), max_tokens=2))

    for _ in range(3):
        output = scheduler.schedule()
        scheduler.update_from_output(output, _model_output(scheduler, output))

    assert [r["step"] for r, _ in records] == [1, 2, 3]
    for record, _ in records:
        for field in TIMING_FIELDS:
            assert record[field] >= 0, field
        assert record["rank_ns"] <= record["scheduler_ns"]
        assert record["running_ns"] + record["admit_ns"] <= record["scheduler_ns"]
    # Both arrivals happened before step 1 and are charged to it only.
    assert records[0][0]["arrival_ns"] > 0
    assert records[1][0]["arrival_ns"] == 0


def test_empty_step_flush_keeps_cost_accounting(monkeypatch):
    records = _capture_batch_logs(monkeypatch)
    scheduler = _scheduler()
    assert scheduler._preemption_cost_supported

    scheduler.schedule()  # nothing to schedule: no accounting for this step
    scheduler.schedule()  # its line is flushed without update_from_output

    assert len(records) == 1
    assert records[0][1] == 0
    assert scheduler._preemption_cost_supported


def test_unfinished_accounted_step_still_disables_cost_accounting(monkeypatch):
    _capture_batch_logs(monkeypatch)
    scheduler = _scheduler()
    scheduler.add_request(_request("a", list(range(1, 41))))
    scheduler.schedule()  # tokens scheduled, output never delivered
    scheduler.schedule()  # overlapping steps: accounting must stop
    assert not scheduler._preemption_cost_supported
