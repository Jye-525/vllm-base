# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Observational survival diagnostics for preempted requests (JSONL schema v1).

With prefix caching, a preempted request's blocks stay in the free queue with
their hashes until eviction reclaims them. This recorder measures, per victim,
how many of those blocks survive until the request is readmitted, and, per
scheduler step, whether evicted victim blocks could have been served by
lower-value free blocks instead. It never changes scheduling or eviction.

Records:
  ``victim``     one per preemption, emitted when the victim is readmitted,
                 finishes while waiting, or is still waiting at shutdown.
  ``evictions``  one per scheduler step in which victim blocks were evicted.

The scheduler enables it together with preemption diagnostics and writes to
the ``survival/`` subdirectory of ``VLLM_PREEMPTION_DIAGNOSTICS_DIR``.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from vllm.v1.core.kv_cache_utils import get_block_hash
from vllm.v1.core.sched.preemption_diagnostics import logger, observed

if TYPE_CHECKING:
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.kv_cache_utils import BlockHash, KVCacheBlock
    from vllm.v1.request import Request


class JsonlWriter:
    """Exclusive-create JSONL stream with a common record envelope."""

    def __init__(self, directory: str, prefix: str):
        self.instance_id = uuid.uuid4().hex
        Path(directory).mkdir(parents=True, exist_ok=True)
        filename = f"{prefix}-{os.getpid()}-{self.instance_id}.jsonl"
        self._stream = (Path(directory) / filename).open("x")

    def emit(self, kind: str, step_id: int, **fields) -> None:
        self._stream.write(json.dumps(dict(
            version=1, kind=kind, instance_id=self.instance_id,
            step_id=step_id, timestamp=time.monotonic(), **fields,
        )) + "\n")

    def flush(self) -> None:
        self._stream.flush()

    def close(self) -> None:
        try:
            self._stream.close()
        except OSError:
            pass


class WaitingDemandIndex:
    """Multiset of block hashes requested by requests in the waiting queue.

    Maintained incrementally on queue entry and exit, so a step-end query
    costs one dictionary lookup per free block.
    """

    def __init__(self) -> None:
        self._counts: Counter[BlockHash] = Counter()
        self._members: dict[str, tuple[BlockHash, ...]] = {}

    def enter(self, request_id: str, hashes: Iterable[BlockHash]) -> None:
        self.exit(request_id)
        members = tuple(hashes)
        self._members[request_id] = members
        self._counts.update(members)

    def exit(self, request_id: str) -> None:
        members = self._members.pop(request_id, None)
        if members is None:
            return
        self._counts.subtract(members)
        for block_hash in members:
            if self._counts[block_hash] <= 0:
                del self._counts[block_hash]

    def demanded(self, block_hash: BlockHash) -> bool:
        return block_hash in self._counts


@dataclass
class FreeQueueCensus:
    """Classification of the blocks currently in the free queue."""

    free_blocks: int = 0
    victim_blocks: int = 0
    demanded_blocks: int = 0
    no_demand_blocks: int = 0
    uncached_blocks: int = 0

    @property
    def lower_value_blocks(self) -> int:
        """Blocks that no waiting request (victim or new) is known to need.

        Counted at step end, so it may include blocks freed or released from
        demand later in the same step: ``avoidable_evictions`` is an upper
        bound on the evictions a victim-aware policy could have redirected.
        """
        return self.no_demand_blocks + self.uncached_blocks


class GuardedObserver:
    """Block-pool observer that can never raise into allocation or eviction.

    An exception inside ``BlockPool.get_new_blocks`` would leave popped
    blocks out of the free queue, so any failure disables the recorder.
    """

    def __init__(self, recorder: SurvivalRecorder, target: EvictionAttributor):
        self._recorder = recorder
        self._target = target

    def on_allocate(self, num_blocks: int) -> None:
        try:
            self._target.on_allocate(num_blocks)
        except Exception as exc:
            self._recorder.disable(exc)

    def on_evict(self, block_hash: BlockHash) -> None:
        try:
            self._target.on_evict(block_hash)
        except Exception as exc:
            self._recorder.disable(exc)


class EvictionAttributor:
    """Attributes evictions to waiting victims and audits LRU ordering.

    A victim block evicted while lower-value blocks remain in the free queue
    is *avoidable*: the allocation could have taken a lower-value block.
    """

    def __init__(self, block_pool: BlockPool, demand: WaitingDemandIndex):
        self._block_pool = block_pool
        self._demand = demand
        self._owners: dict[BlockHash, set[str]] = {}
        self._hashes_of: dict[str, tuple[BlockHash, ...]] = {}
        # Hashes per victim whose last cached copy was evicted while it waited.
        self.evicted_per_victim: dict[str, set[BlockHash]] = {}
        self.step_victim_evictions = 0
        self.step_other_evictions = 0
        self.allocated_blocks_total = 0

    def track(self, request_id: str, hashes: Iterable[BlockHash]) -> None:
        self.untrack(request_id)
        owned = tuple(hashes)
        self._hashes_of[request_id] = owned
        for block_hash in owned:
            self._owners.setdefault(block_hash, set()).add(request_id)

    def untrack(self, request_id: str) -> int:
        """Stop tracking a victim; return how many of its blocks were evicted."""
        for block_hash in self._hashes_of.pop(request_id, ()):
            owners = self._owners.get(block_hash)
            if owners is not None:
                owners.discard(request_id)
                if not owners:
                    del self._owners[block_hash]
        return len(self.evicted_per_victim.pop(request_id, ()))

    def clear(self) -> None:
        self._owners.clear()
        self._hashes_of.clear()
        self.evicted_per_victim.clear()

    # BlockPool observer interface.
    def on_allocate(self, num_blocks: int) -> None:
        self.allocated_blocks_total += num_blocks

    def on_evict(self, block_hash: BlockHash) -> None:
        owners = self._owners.get(block_hash)
        if not owners:
            self.step_other_evictions += 1
            return
        self.step_victim_evictions += 1
        for request_id in owners:
            self.evicted_per_victim.setdefault(request_id, set()).add(block_hash)

    def reset_step(self) -> None:
        self.step_victim_evictions = 0
        self.step_other_evictions = 0

    def census(self) -> FreeQueueCensus:
        result = FreeQueueCensus()
        for block in self._iter_free_blocks():
            result.free_blocks += 1
            if block.block_hash is None:
                result.uncached_blocks += 1
                continue
            raw_hash = get_block_hash(block.block_hash)
            if raw_hash in self._owners:
                result.victim_blocks += 1
            elif self._demand.demanded(raw_hash):
                result.demanded_blocks += 1
            else:
                result.no_demand_blocks += 1
        return result

    def _iter_free_blocks(self) -> Iterator[KVCacheBlock]:
        queue = self._block_pool.free_block_queue
        block = queue.fake_free_list_head.next_free_block
        while block is not None and block is not queue.fake_free_list_tail:
            yield block
            block = block.next_free_block


@dataclass
class VictimSnapshot:
    """State captured immediately before a victim's blocks are freed."""

    request_id: str
    blocks: list[KVCacheBlock]
    computed_tokens: int
    prompt_tokens: int
    output_tokens: int
    older_free_blocks: int


@dataclass
class VictimRecord:
    """One open preemption, closed when the victim leaves the waiting queue."""

    request_id: str
    ordinal: int
    preempt_step: int
    preempt_time: float
    computed_tokens: int
    prompt_tokens: int
    output_tokens: int
    older_free_blocks: int
    exclusive_cached_blocks: int
    shared_cached_blocks: int
    uncached_blocks: int
    # Tokens in full cached blocks. computed_tokens - cached_context_tokens
    # (the partial tail) is lost even when every block survives.
    cached_context_tokens: int
    allocated_at_preempt: int
    waiters_at_preempt: int
    max_waiters_during_wait: int = 0
    extra: dict = field(default_factory=dict)


class VictimLedger:
    """Open victim records and the derived survival metrics."""

    def __init__(self) -> None:
        self.open: dict[str, VictimRecord] = {}

    def add(self, record: VictimRecord) -> None:
        self.open[record.request_id] = record
        waiting = len(self.open)
        for other in self.open.values():
            other.max_waiters_during_wait = max(
                other.max_waiters_during_wait, waiting
            )

    def close(
        self,
        request_id: str,
        *,
        outcome: str,
        step: int,
        allocated_before_step: int,
        evicted_blocks: int,
        resume_hit_tokens: int | None,
    ) -> dict | None:
        record = self.open.pop(request_id, None)
        if record is None:
            return None
        row = asdict(record)
        row.update(record.extra)
        del row["extra"]
        wait_steps = step - record.preempt_step
        allocations = max(0, allocated_before_step - record.allocated_at_preempt)
        row.update(
            outcome=outcome,
            close_step=step,
            wait_steps=wait_steps,
            wait_s=time.monotonic() - record.preempt_time,
            allocations_during_wait=allocations,
            allocation_rate=allocations / wait_steps if wait_steps > 0 else None,
            victim_blocks_evicted=evicted_blocks,
            resume_hit_tokens=resume_hit_tokens,
            lost_tokens=(
                max(0, record.computed_tokens - resume_hit_tokens)
                if resume_hit_tokens is not None else None
            ),
        )
        return row

    def clear(self) -> list[str]:
        request_ids = list(self.open)
        self.open.clear()
        return request_ids


class SurvivalRecorder:
    """Scheduler-facing facade. Fails closed: errors disable the recorder."""

    def __init__(
        self,
        directory: str,
        kv_cache_manager: KVCacheManager,
        configuration: dict,
    ):
        self.enabled = True
        self.cost: Counter[str] = Counter()
        self.step = 0
        self._writer: JsonlWriter | None = None
        self._kv_cache_manager = kv_cache_manager
        block_pool = kv_cache_manager.block_pool
        self.demand = WaitingDemandIndex()
        self.attributor = EvictionAttributor(block_pool, self.demand)
        self.ledger = VictimLedger()
        self._allocated_at_step_start = 0
        self._observer: GuardedObserver | None = None
        self._block_size = configuration.get("block_size", 0)
        try:
            self._writer = JsonlWriter(directory, "survival")
            self._writer.emit("startup", self.step, configuration=configuration)
            self._writer.flush()
            self._observer = GuardedObserver(self, self.attributor)
            block_pool.observer = self._observer
        except Exception as exc:
            self.disable(exc)

    def disable(self, exc: Exception) -> None:
        if self.enabled:
            logger.warning("Survival diagnostics disabled: %s", exc)
        self._teardown()

    def _teardown(self) -> None:
        self.enabled = False
        block_pool = self._kv_cache_manager.block_pool
        if self._observer is not None and block_pool.observer is self._observer:
            block_pool.observer = None
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    @observed
    def begin_step(self) -> None:
        self.step += 1
        self.attributor.reset_step()
        self._allocated_at_step_start = self.attributor.allocated_blocks_total

    @observed
    def end_step(self) -> None:
        attributor = self.attributor
        if attributor.step_victim_evictions == 0:
            return
        census = attributor.census()
        self._writer.emit(
            "evictions", self.step,
            victim_evictions=attributor.step_victim_evictions,
            other_evictions=attributor.step_other_evictions,
            avoidable_evictions=min(
                attributor.step_victim_evictions, census.lower_value_blocks
            ),
            waiting_victims=len(self.ledger.open),
            **asdict(census),
            lower_value_blocks=census.lower_value_blocks,
        )
        self._writer.flush()

    @observed
    def request_waiting(self, request: Request) -> None:
        self.demand.enter(request.request_id, request.block_hashes)

    def capture_victim(self, request: Request) -> VictimSnapshot | None:
        """Call before the victim's blocks are freed."""
        if not self.enabled:
            return None
        try:
            groups = self._kv_cache_manager.get_blocks(request.request_id).blocks
            return VictimSnapshot(
                request_id=request.request_id,
                blocks=[b for group in groups for b in group if not b.is_null],
                computed_tokens=request.num_computed_tokens,
                prompt_tokens=request.num_prompt_tokens,
                output_tokens=request.num_output_tokens,
                older_free_blocks=(
                    self._kv_cache_manager.block_pool.get_num_free_blocks()
                ),
            )
        except Exception as exc:
            self.disable(exc)
            return None

    @observed
    def preempted(self, request: Request, snapshot: VictimSnapshot | None) -> None:
        """Call after the request's blocks are freed by any preemption.

        ``snapshot`` is None for preemptions that are not tracked as victims
        (for example a forced prefix-cache reset); the request still re-enters
        the waiting queue and its blocks still count as waiting demand.
        """
        self.demand.enter(request.request_id, request.block_hashes)
        if snapshot is None:
            return
        exclusive = shared = uncached = 0
        hashes = []
        for block in snapshot.blocks:
            if block.block_hash is None:
                uncached += 1
                continue
            hashes.append(get_block_hash(block.block_hash))
            if block.ref_cnt == 0:
                exclusive += 1
            else:
                shared += 1
        self.attributor.track(request.request_id, hashes)
        self.ledger.add(VictimRecord(
            request_id=request.request_id,
            ordinal=request.num_preemptions + 1,
            preempt_step=self.step,
            preempt_time=time.monotonic(),
            computed_tokens=snapshot.computed_tokens,
            prompt_tokens=snapshot.prompt_tokens,
            output_tokens=snapshot.output_tokens,
            older_free_blocks=snapshot.older_free_blocks,
            exclusive_cached_blocks=exclusive,
            shared_cached_blocks=shared,
            uncached_blocks=uncached,
            cached_context_tokens=(exclusive + shared) * self._block_size,
            allocated_at_preempt=self.attributor.allocated_blocks_total,
            waiters_at_preempt=len(self.ledger.open) + 1,  # including itself
        ))

    @observed
    def admitted(self, request: Request, local_hit_tokens: int) -> None:
        self.demand.exit(request.request_id)
        self._close(request.request_id, "resumed", local_hit_tokens)

    @observed
    def finished(self, request: Request) -> None:
        self.demand.exit(request.request_id)
        self._close(request.request_id, "finished_while_waiting", None)

    @observed
    def cache_reset(self, succeeded: bool) -> None:
        if succeeded:
            # Cached blocks are gone; open victims can no longer be measured.
            for request_id in list(self.ledger.open):
                self._close(request_id, "cache_reset", None)
            self.attributor.clear()
        self._writer.emit("cache_reset", self.step, succeeded=succeeded)
        self._writer.flush()

    @observed
    def shutdown(self) -> None:
        for request_id in list(self.ledger.open):
            self._close(request_id, "waiting_at_shutdown", None)
        self._writer.emit("shutdown", self.step, overhead=dict(self.cost))
        self._writer.flush()
        self._teardown()

    def _close(
        self, request_id: str, outcome: str, resume_hit_tokens: int | None
    ) -> None:
        if request_id not in self.ledger.open:
            return
        row = self.ledger.close(
            request_id,
            outcome=outcome,
            step=self.step,
            allocated_before_step=self._allocated_at_step_start,
            evicted_blocks=self.attributor.untrack(request_id),
            resume_hit_tokens=resume_hit_tokens,
        )
        self._writer.emit("victim", self.step, **row)
        self._writer.flush()  # survive a hard kill of the server
