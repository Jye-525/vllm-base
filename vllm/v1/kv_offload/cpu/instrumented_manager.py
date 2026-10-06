# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Instrumented variant of CPUOffloadingManager that emits tier_logger events.

The tier logger is gated on env var VLLM_TIER_LOG; when unset, every hook
becomes a no-op (single isinstance check), so this manager is safe to use as
the default.

When VLLM_ORPHAN_LOG is set, it also maintains the EXP 4 orphan counter
state (_children / _orphaned) that orphan_probe.OrphanCounter samples; when
unset, that bookkeeping is skipped entirely.
"""

from collections import defaultdict
from collections.abc import Collection

from vllm.v1.core.orphan_probe import maybe_start_orphan_counter
from vllm.v1.core.tier_logger import _get_tier_logger
from vllm.v1.kv_offload.base import (
    LoadStoreSpec,
    OffloadKey,
    PrepareStoreOutput,
    ReqContext,
)
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager


class InstrumentedCPUOffloadingManager(CPUOffloadingManager):
    """CPU KV cache offloading manager with per-event tracing hooks."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # --- EXP 4: orphan counter state (only used when tracking) ---
        # parent key -> set of child keys currently resident in CPU
        self._children: dict[OffloadKey, set[OffloadKey]] = defaultdict(set)
        # CPU blocks whose parent has been evicted from CPU (orphaned)
        self._orphaned: set[OffloadKey] = set()
        # Background CSV writer, started only if VLLM_ORPHAN_LOG is set.
        self._orphan_counter = maybe_start_orphan_counter(self)
        self._track_orphans = self._orphan_counter is not None
        # --- END EXP 4 ---

    def num_cpu_blocks_in_use(self) -> int:
        return self._num_blocks - self._get_num_free_blocks()

    def prepare_load(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> LoadStoreSpec:
        result = super().prepare_load(keys, req_context)
        tlog = _get_tier_logger()
        if tlog is not None:
            for key in keys:
                tlog.log_event(
                    "cpu_hit", key, "cpu", request_id=req_context.req_id
                )
        return result

    def prepare_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> PrepareStoreOutput | None:
        if self._track_orphans:
            keys = list(keys)
            # --- EXP 4: register consecutive parent→child links.
            # Blocks arrive in prefix-chain order, so keys[i-1] is the parent
            # of keys[i].  This covers within-batch links; the parent of
            # keys[0] (a cross-batch link) is tracked only when that parent
            # is itself new in this batch.
            for i in range(1, len(keys)):
                self._children[keys[i - 1]].add(keys[i])
            # --- END EXP 4 ---

        result = super().prepare_store(keys, req_context)
        if result is not None:
            tlog = _get_tier_logger()
            for key in result.evicted_keys:
                if self._track_orphans:
                    # --- EXP 4: update orphan counter ---
                    self._orphaned.discard(key)
                    for child in self._children.get(key, ()):
                        if self._policy.get(child) is not None:
                            self._orphaned.add(child)
                    self._children.pop(key, None)
                    # --- END EXP 4 ---
                if tlog is not None:
                    tlog.log_event(
                        "cpu_evict", key, "cpu",
                        request_id=req_context.req_id,
                    )
        return result

    def complete_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
        success: bool = True,
    ) -> None:
        if not success:
            if self._track_orphans:
                # --- EXP 4: clean up orphan state for blocks the failed
                # store is about to drop (mirrors the base class predicate).
                for key in keys:
                    block = self._policy.get(key)
                    if block is not None and not block.is_ready:
                        self._orphaned.discard(key)
                        self._children.pop(key, None)
                # --- END EXP 4 ---
            super().complete_store(keys, req_context, success)
            return

        tlog = _get_tier_logger()
        # Capture keys that will transition to ready (mirroring the base
        # class predicate) before super() mutates ref_cnt.
        stored: list[OffloadKey] = []
        if tlog is not None:
            for key in keys:
                block = self._policy.get(key)
                if block is not None and not block.is_ready:
                    stored.append(key)
        super().complete_store(keys, req_context, success)
        if tlog is not None:
            for key in stored:
                tlog.log_event(
                    "cpu_store", key, "cpu",
                    request_id=req_context.req_id,
                )
