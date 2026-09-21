# SPDX-License-Identifier: Apache-2.0
"""Real allocator/scheduler CPU fixtures; no model weights or GPU execution."""

import dataclasses
import json

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import OPTConfig, PreTrainedTokenizerFast

from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus

from .utils import create_requests, create_scheduler

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def local_model(tmp_path):
    model = tmp_path / "model"
    OPTConfig(architectures=["OPTForCausalLM"]).save_pretrained(model)
    PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"<unk>": 0}, unk_token="<unk>")),
        unk_token="<unk>",
    ).save_pretrained(model)
    return str(model)


def model_output(scheduler, output):
    ids = list(output.num_scheduled_tokens)
    return ModelRunnerOutput(
        req_ids=ids, req_id_to_index={rid: i for i, rid in enumerate(ids)},
        sampled_token_ids=[
            [1000] if scheduler.requests[rid].num_computed_tokens >= scheduler.requests[rid].num_tokens else []
            for rid in ids], logprobs=None, prompt_logprobs_dict={}, pooler_output=[],
    )


def state(scheduler):
    pool = scheduler.kv_cache_manager.block_pool
    return (
        [r.request_id for r in scheduler.running],
        [r.request_id for r in scheduler.waiting],
        [(r.request_id, r.num_computed_tokens, r.num_preemptions, r.status) for r in scheduler.requests.values()],
        [(b.block_id, b.ref_cnt, b.block_hash) for b in pool.blocks],
        [b.block_id for b in pool.free_block_queue.get_all_free_blocks()],
        dict(pool._demand_counts),
        dataclasses.asdict(scheduler.kv_cache_manager.prefix_cache_stats),
    )


@pytest.mark.parametrize("prefix,shared,policy,async_scheduling", [
    (False, False, "fcfs", False), (True, False, "fcfs", False),
    (True, True, "fcfs", False), (True, True, "lpm", False),
    (True, True, "clpm", False), (False, False, "priority", False),
    (True, False, "fcfs", True),
])
def test_diagnostics_preserve_scheduler_state(tmp_path, monkeypatch, local_model, prefix, shared, policy, async_scheduling):
    settings = dict(model=local_model, num_blocks=10, max_num_seqs=4,
                    max_num_batched_tokens=64, max_model_len=256,
                    enable_prefix_caching=prefix, scheduling_policy=policy,
                    async_scheduling=async_scheduling)
    monkeypatch.delenv("VLLM_PREEMPTION_DIAGNOSTICS_DIR", raising=False)
    baseline = create_scheduler(**settings)
    monkeypatch.setenv("VLLM_PREEMPTION_DIAGNOSTICS_DIR", str(tmp_path / "diagnostics"))
    observed = create_scheduler(**settings)
    for scheduler in (baseline, observed):
        for req in create_requests(4, num_tokens=31, max_tokens=48, same_prompt=shared):
            req.arrival_time = 1.0 + int(req.request_id)
            scheduler.add_request(req)
    assert baseline.preemption_diagnostics is None
    for _ in range(250):
        expected, actual = baseline.schedule(), observed.schedule()
        assert dataclasses.asdict(expected) == dataclasses.asdict(actual)
        assert state(baseline) == state(observed)
        baseline.update_from_output(expected, model_output(baseline, expected))
        observed.update_from_output(actual, model_output(observed, actual))
        assert state(baseline) == state(observed)
        if not baseline.requests:
            break
    assert not baseline.requests
    diag = observed.preemption_diagnostics
    assert diag.enabled and not diag.requests
    assert diag.totals["pressure_preemptions"] > 0
    baseline.shutdown()
    observed.shutdown()
    rows = [json.loads(line) for path in (tmp_path / "diagnostics").glob("*.jsonl") for line in path.read_text().splitlines()]
    episodes = [r for r in rows if r["kind"] == "episode"]
    assert episodes
    if async_scheduling:
        assert "async_scheduling" in rows[0]["unsupported"]
        assert not any(r["kind"] == "recovery" for r in rows)
    elif policy == "priority":
        assert all("priority_shadow" in e["unsupported"] for e in episodes)
    else:
        assert all(e["tail_matches_actual"] for e in episodes)
        assert diag.totals["completed_scheduled_tokens"] > 0


def test_forced_reset_and_cancellation(tmp_path, monkeypatch, local_model):
    monkeypatch.setenv("VLLM_PREEMPTION_DIAGNOSTICS_DIR", str(tmp_path / "diagnostics"))
    scheduler = create_scheduler(model=local_model, enable_prefix_caching=True)
    req = create_requests(1, num_tokens=33)[0]
    scheduler.add_request(req)
    output = scheduler.schedule()
    scheduler.update_from_output(output, model_output(scheduler, output))
    assert scheduler.reset_prefix_cache(reset_running_requests=True)
    assert scheduler.preemption_diagnostics.boundary_id == 1
    scheduler.finish_requests(req.request_id, RequestStatus.FINISHED_ABORTED)
    diag = scheduler.preemption_diagnostics
    assert not diag.requests
    assert diag.totals["episodes"] == 0
    assert diag.totals["pressure_preemptions"] == 0
    scheduler.shutdown()


@pytest.mark.parametrize("prefix,hole,expected_hits", [
    (False, None, 0), (True, None, 48), (True, 0, 0), (True, 1, 16),
])
def test_real_prefix_survival_and_chunked_partial_replay(
    tmp_path, monkeypatch, local_model, prefix, hole, expected_hits
):
    import time

    monkeypatch.setenv("VLLM_PREEMPTION_DIAGNOSTICS_DIR", str(tmp_path / "diagnostics"))
    scheduler = create_scheduler(
        model=local_model, num_blocks=12, max_num_batched_tokens=16,
        max_model_len=128, enable_prefix_caching=prefix,
    )
    req = create_requests(1, num_tokens=49, max_tokens=16)[0]
    scheduler.add_request(req)
    while req.num_computed_tokens < 49:
        output = scheduler.schedule()
        scheduler.update_from_output(output, model_output(scheduler, output))
    blocks = list(scheduler.kv_cache_manager.get_blocks(req.request_id).blocks[0])
    scheduler.running.remove(req)
    scheduler._preempt_request(req, time.monotonic())
    scheduler.prev_step_scheduled_req_ids.discard(req.request_id)
    if hole is not None:
        scheduler.kv_cache_manager.block_pool._maybe_evict_cached_block(blocks[hole])
    diagnostics = scheduler.preemption_diagnostics
    for _ in range(5):
        output = scheduler.schedule()
        scheduler.update_from_output(output, model_output(scheduler, output))
        recovery = diagnostics.requests[req.request_id]["recovery"]
        assert recovery["local_prefix_hit_tokens"] == expected_hits
        if recovery["recovered_at"] is not None:
            break
    assert recovery["recovered_at"] is not None
    assert recovery["prompt_replay"] == 49-expected_hits
    assert recovery["output_replay"] == 0
    scheduler.finish_requests(req.request_id, RequestStatus.FINISHED_STOPPED)
    scheduler.shutdown()


def test_idle_reset_flushes_diagnostic_checkpoint(tmp_path, monkeypatch, local_model):
    directory = tmp_path / "diagnostics"
    monkeypatch.setenv("VLLM_PREEMPTION_DIAGNOSTICS_DIR", str(directory))
    scheduler = create_scheduler(model=local_model, enable_prefix_caching=True)
    req = create_requests(1, num_tokens=17, max_tokens=1)[0]
    scheduler.add_request(req)
    output = scheduler.schedule()
    scheduler.update_from_output(output, model_output(scheduler, output))
    assert not scheduler.requests
    assert scheduler.reset_prefix_cache(reset_running_requests=True)
    row = json.loads(next(directory.glob("*.jsonl")).read_text().splitlines()[-1])
    assert row["kind"] == "cache_reset" and row["succeeded"]
    assert not row["active_requests"] and not row["outstanding_batch"]
    assert row["totals"]["requests"] == 1
    assert row["totals"]["completed_scheduled_tokens"] == 17
    scheduler.shutdown()
