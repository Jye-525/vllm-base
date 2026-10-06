# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for KVReuse orphan bookkeeping in InstrumentedCPUOffloadingManager."""

from vllm.v1.kv_offload.base import OffloadKey, ReqContext, make_offload_key
from vllm.v1.kv_offload.cpu.instrumented_manager import (
    InstrumentedCPUOffloadingManager,
)

_CTX = ReqContext(req_id="r")


def _keys(*int_ids: int) -> list[OffloadKey]:
    return [make_offload_key(str(i).encode(), 0) for i in int_ids]


def test_orphan_tracking_off_without_log(monkeypatch):
    monkeypatch.delenv("VLLM_ORPHAN_LOG", raising=False)
    manager = InstrumentedCPUOffloadingManager(num_blocks=2, cache_policy="lru")
    assert not manager._track_orphans
    manager.prepare_store(_keys(1, 2), _CTX)
    assert not manager._children
    assert not manager._orphaned


def test_orphan_tracking_marks_children_of_evicted_parent(monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_ORPHAN_LOG", str(tmp_path / "orphans.csv"))
    manager = InstrumentedCPUOffloadingManager(num_blocks=2, cache_policy="lru")
    try:
        assert manager._track_orphans
        k1, k2, k3 = _keys(1, 2, 3)
        manager.prepare_store([k1, k2], _CTX)
        manager.complete_store([k1, k2], _CTX)
        # Storing k3 evicts the LRU block k1, which orphans its child k2.
        output = manager.prepare_store([k3], _CTX)
        assert output is not None
        assert output.evicted_keys == [k1]
        assert manager._orphaned == {k2}
        assert manager.num_cpu_blocks_in_use() == 2
    finally:
        assert manager._orphan_counter is not None
        manager._orphan_counter.stop()
