# SPDX-License-Identifier: Apache-2.0
"""Survival diagnostics: component units plus a real CPU scheduler fixture."""

import json
from types import SimpleNamespace as NS

import pytest

from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    FreeKVCacheBlockQueue,
    KVCacheBlock,
    make_block_hash_with_group_id,
)
from vllm.v1.core.sched.survival_diagnostics import (
    EvictionAttributor,
    VictimLedger,
    VictimRecord,
    WaitingDemandIndex,
)

from .test_preemption_diagnostics_scheduler import local_model, model_output  # noqa: F401
from .utils import create_requests, create_scheduler

pytestmark = pytest.mark.cpu_test


def raw(name: str) -> BlockHash:
    return BlockHash(name.encode())


def cached_block(block_id: int, name: str | None) -> KVCacheBlock:
    block = KVCacheBlock(block_id=block_id)
    if name is not None:
        block.block_hash = make_block_hash_with_group_id(raw(name), 0)
    return block


def test_waiting_demand_index_counts_overlapping_requests():
    demand = WaitingDemandIndex()
    demand.enter("a", [raw("x"), raw("y")])
    demand.enter("b", [raw("y")])
    demand.exit("a")
    assert not demand.demanded(raw("x"))
    assert demand.demanded(raw("y"))
    demand.exit("b")
    demand.exit("b")  # idempotent
    assert not demand.demanded(raw("y"))


def test_attributor_counts_victim_evictions_and_classifies_free_queue():
    # Free queue in LRU order: victim block, demanded block, no-demand block,
    # uncached block.
    blocks = [
        cached_block(0, "v"), cached_block(1, "d"),
        cached_block(2, "n"), cached_block(3, None),
    ]
    pool = NS(free_block_queue=FreeKVCacheBlockQueue(blocks))
    demand = WaitingDemandIndex()
    demand.enter("new", [raw("d")])
    attributor = EvictionAttributor(pool, demand)
    attributor.track("victim", [raw("v"), raw("w")])

    attributor.on_allocate(2)
    attributor.on_evict(raw("v"))
    attributor.on_evict(raw("other"))
    census = attributor.census()

    assert attributor.allocated_blocks_total == 2
    assert attributor.step_victim_evictions == 1
    assert attributor.step_other_evictions == 1
    assert (census.free_blocks, census.victim_blocks, census.demanded_blocks,
            census.no_demand_blocks, census.uncached_blocks) == (4, 1, 1, 1, 1)
    assert census.lower_value_blocks == 2
    assert attributor.untrack("victim") == 1
    assert attributor.untrack("victim") == 0


def test_attributor_shared_block_counts_for_every_owning_victim():
    attributor = EvictionAttributor(NS(), WaitingDemandIndex())
    attributor.track("a", [raw("s")])
    attributor.track("b", [raw("s")])
    attributor.on_evict(raw("s"))
    assert attributor.untrack("a") == 1
    assert attributor.untrack("b") == 1


def victim_record(**overrides) -> VictimRecord:
    fields = dict(
        request_id="r", ordinal=1, preempt_step=10, preempt_time=0.0,
        computed_tokens=100, prompt_tokens=80, output_tokens=20,
        older_free_blocks=0, exclusive_cached_blocks=6, shared_cached_blocks=0,
        uncached_blocks=1, cached_context_tokens=96, allocated_at_preempt=50,
        waiters_at_preempt=1,
    )
    fields.update(overrides)
    return VictimRecord(**fields)


def test_ledger_derives_wait_allocation_and_loss():
    ledger = VictimLedger()
    ledger.add(victim_record())
    ledger.add(victim_record(request_id="s", preempt_step=12))
    row = ledger.close(
        "r", outcome="resumed", step=14, allocated_before_step=58,
        evicted_blocks=2, resume_hit_tokens=64,
    )
    assert row["wait_steps"] == 4
    assert row["allocations_during_wait"] == 8
    assert row["allocation_rate"] == 2.0
    assert row["lost_tokens"] == 36
    assert row["victim_blocks_evicted"] == 2
    assert row["max_waiters_during_wait"] == 2
    assert "extra" not in row
    assert ledger.close("r", outcome="resumed", step=15, allocated_before_step=0,
                        evicted_blocks=0, resume_hit_tokens=0) is None


def test_ledger_unresumed_victim_has_no_loss_value():
    ledger = VictimLedger()
    ledger.add(victim_record())
    row = ledger.close("r", outcome="finished_while_waiting", step=11,
                       allocated_before_step=50, evicted_blocks=0,
                       resume_hit_tokens=None)
    assert row["lost_tokens"] is None
    assert row["allocation_rate"] == 0.0


def survival_rows(directory):
    return [
        json.loads(line)
        for path in (directory / "survival").glob("survival-*.jsonl")
        for line in path.read_text().splitlines()
    ]


@pytest.mark.parametrize("shared", [False, True])
def test_scheduler_records_every_pressure_preemption(
    tmp_path, monkeypatch, local_model, shared  # noqa: F811
):
    directory = tmp_path / "diagnostics"
    monkeypatch.setenv("VLLM_PREEMPTION_DIAGNOSTICS_DIR", str(directory))
    scheduler = create_scheduler(
        model=local_model, num_blocks=10, max_num_seqs=4,
        max_num_batched_tokens=64, max_model_len=256,
        enable_prefix_caching=True,
    )
    assert scheduler.survival_diagnostics is not None
    for req in create_requests(4, num_tokens=31, max_tokens=48, same_prompt=shared):
        scheduler.add_request(req)
    for _ in range(250):
        output = scheduler.schedule()
        scheduler.update_from_output(output, model_output(scheduler, output))
        if not scheduler.requests:
            break
    preemptions = scheduler.preemption_diagnostics.totals["pressure_preemptions"]
    scheduler.shutdown()

    rows = survival_rows(directory)
    victims = [r for r in rows if r["kind"] == "victim"]
    assert preemptions > 0
    assert len(victims) == preemptions
    for row in victims:
        assert row["outcome"] == "resumed"
        assert row["lost_tokens"] == max(
            0, row["computed_tokens"] - row["resume_hit_tokens"]
        )
        tracked = row["exclusive_cached_blocks"] + row["shared_cached_blocks"]
        assert 0 <= row["victim_blocks_evicted"] <= tracked
        assert row["wait_steps"] >= 0
    for row in (r for r in rows if r["kind"] == "evictions"):
        assert 0 <= row["avoidable_evictions"] <= row["victim_evictions"]
    assert rows[-1]["kind"] == "shutdown"


def test_forced_reset_is_not_a_victim(tmp_path, monkeypatch, local_model):  # noqa: F811
    directory = tmp_path / "diagnostics"
    monkeypatch.setenv("VLLM_PREEMPTION_DIAGNOSTICS_DIR", str(directory))
    scheduler = create_scheduler(model=local_model, enable_prefix_caching=True)
    req = create_requests(1, num_tokens=33)[0]
    scheduler.add_request(req)
    output = scheduler.schedule()
    scheduler.update_from_output(output, model_output(scheduler, output))
    assert scheduler.reset_prefix_cache(reset_running_requests=True)
    assert not scheduler.survival_diagnostics.ledger.open
    scheduler.shutdown()
    kinds = [r["kind"] for r in survival_rows(directory)]
    assert "victim" not in kinds and "cache_reset" in kinds
