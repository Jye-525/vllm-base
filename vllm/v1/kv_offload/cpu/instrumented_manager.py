# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Instrumented variant of CPUOffloadingManager that emits tier_logger events.

The tier logger is gated on env var VLLM_TIER_LOG; when unset, every hook
becomes a no-op (single isinstance check), so this manager is safe to use as
the default.
"""

from collections.abc import Collection

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
        result = super().prepare_store(keys, req_context)
        if result is not None:
            tlog = _get_tier_logger()
            if tlog is not None:
                for key in result.evicted_keys:
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
