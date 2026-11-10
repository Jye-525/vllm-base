# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Demand-aware retention of free cached KV blocks (SSPF series).

Eviction policy ``demand`` (``kv_cache_eviction_policy="demand"``). A free
cached block whose hash has predicted reuse sits in a side pool and is
evicted only after every other free block (plain LRU first). Predicted reuse
is either

* queued demand: a waiting request (fresh or preempted, including a
  preempted request's generated blocks) contains the hash, or
* a session hold: the full blocks of a finished session turn, kept for
  ``ttl_factor`` x the session's predicted tool gap.

Within the side pool, held turns go first (latest hold expiry, then deepest
block), then queued demand (worst best-sharer rank, then deepest block);
block id breaks ties. A hash with queued demand is ranked as demand even if
also held. This is the KVReuse simulator's ``eviction="pred"`` with
``pe_tier="hold_last"`` under rule R (``simulator/kvsim.py`` ``_pick_pred``,
``_rank_of``):

* ranks come from the waiting order at the last ranking (a snapshot of the
  waiting requests' index paths), resolved lazily to hash -> best position;
  generated blocks of preempted requests rank 0;
* until the next snapshot, the path of a request preempted since ranks 0,
  and a hash whose demand ended since (and is not held) ranks infinite;
  admissions do not change ranks.

Keys of side-pool blocks live in lazy min-heaps, one per tier: every key
change pushes the block's current key, and stale entries are dropped when
popped. Keys start with their tier, so the hold tier's heap is drained first
and the demand tier's keys (which need the waiting-order ranks) are built
only when the held blocks cannot supply the victims (the simulator's
``_pick_pred`` shortcut).
"""

from __future__ import annotations

import heapq
import math
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from itertools import islice
from typing import TYPE_CHECKING

from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    KVCacheBlock,
    get_block_hash,
    make_block_hash_with_group_id,
)

if TYPE_CHECKING:
    from vllm.v1.core.block_pool import BlockPool

HOLD_TIER = 0  # evicted before queued demand (pe_tier="hold_last")
DEMAND_TIER = 1
# Tool-gap estimates, as literals so the float results equal the simulator's.
# Per session: 0.5 * old + 0.5 * observed; for sessions with no gap yet, a
# global estimate: 0.9 * old + 0.1 * observed.
GAP_SESSION_OLD, GAP_SESSION_NEW = 0.5, 0.5
GAP_GLOBAL_OLD, GAP_GLOBAL_NEW = 0.9, 0.1

Key = tuple[int, float, int, int]


class DemandRetention:
    def __init__(
        self,
        block_pool: BlockPool,
        demand_counts: Mapping[BlockHash, int],
        demand_depths: Mapping[BlockHash, int],
        *,
        ttl_factor: float = 2.0,
        initial_gap_s: float = 5.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._pool = block_pool
        # Live views of the waiting-prefix index: pending sharers per hash and
        # each hash's (1-based) depth in a request path.
        self._demand = demand_counts
        self._depths = demand_depths
        self.ttl_factor = ttl_factor
        self.initial_gap_s = initial_gap_s
        self.clock = clock
        # Free cached blocks with predicted reuse, by block id.
        self.side: dict[int, KVCacheBlock] = {}
        # Lazy key heaps by tier (None: rebuild from the side pool when needed).
        self._heaps: dict[int, list[tuple[Key, int]] | None] = {
            HOLD_TIER: None,
            DEMAND_TIER: None,
        }
        # Hashes with queued demand, as last reported by on_demand_change.
        self._demanded: set[BlockHash] = set()
        # Session holds: hash -> [holding turns, depth]; hash -> latest expiry
        # (a running maximum, not lowered when one of several holds drops).
        self._hold: dict[BlockHash, list[int]] = {}
        self._hold_t: dict[BlockHash, float] = {}
        self._session_hold: dict[str, tuple[float, Sequence[BlockHash]]] = {}
        self._hold_expiry: list[tuple[float, int, str]] = []
        self._hold_seq = 0
        self._session_end: dict[str, float] = {}
        self._gap_session: dict[str, float] = {}
        self._gap_all: float | None = None
        # Rule R ranks.
        self._snapshot: list[tuple[Sequence[BlockHash], int]] = []
        self._rank_map: dict[BlockHash, int] | None = None
        self._rank_extra: dict[BlockHash, float] = {}
        # Diagnostics (cumulative; the scheduler logs per-step differences).
        self.num_hold_evictions = 0
        self.num_demand_evictions = 0
        self.elapsed_ns = 0

    # ------------------------------------------------------------- state

    def wanted(self, block_hash: BlockHash) -> bool:
        return self._demand.get(block_hash, 0) > 0 or block_hash in self._hold

    def _rank(self, block_hash: BlockHash) -> float:
        extra = self._rank_extra.get(block_hash)
        if extra is not None:
            return extra
        if self._rank_map is None:
            ranks: dict[BlockHash, int] = {}
            for pos, (path, num_prompt_blocks) in enumerate(self._snapshot):
                # Assigned hashes form a prefix of each path (cumulative
                # hashes): skip it by binary search.
                lo, hi = 0, len(path)
                while lo < hi:
                    mid = (lo + hi) // 2
                    if path[mid] in ranks:
                        lo = mid + 1
                    else:
                        hi = mid
                # Every hash from lo on is new (distinct within a path), so
                # bulk inserts give the per-hash loop's map: prompt blocks
                # rank pos, generated blocks (preempted requests) rank 0.
                split = max(lo, num_prompt_blocks)
                if lo < split:
                    ranks.update(dict.fromkeys(islice(path, lo, split), pos))
                if split < len(path):
                    ranks.update(dict.fromkeys(islice(path, split, None), 0))
            self._rank_map = ranks
        return self._rank_map.get(block_hash, math.inf)

    def _key(self, block: KVCacheBlock) -> Key:
        assert block.block_hash is not None
        h = get_block_hash(block.block_hash)
        if self._demand.get(h, 0) > 0:
            depth = self._depths.get(h, 1) - 1
            return (DEMAND_TIER, -self._rank(h), -depth, block.block_id)
        _, depth = self._hold[h]
        return (HOLD_TIER, -self._hold_t[h], -depth, block.block_id)

    def _tier(self, block: KVCacheBlock) -> int:
        """The first element of ``_key(block)``, without resolving ranks."""
        assert block.block_hash is not None
        h = get_block_hash(block.block_hash)
        return DEMAND_TIER if self._demand.get(h, 0) > 0 else HOLD_TIER

    def _push(self, block: KVCacheBlock) -> None:
        tier = self._tier(block)
        heap = self._heaps[tier]
        if heap is None:
            return
        if len(heap) > 2 * len(self.side) + 1024:
            self._heaps[tier] = None  # mostly stale: rebuild when needed
        else:
            heapq.heappush(heap, (self._key(block), block.block_id))

    def _copies(self, block_hash: BlockHash) -> list[KVCacheBlock]:
        return self._pool.cached_block_hash_to_block.get_all_blocks(
            make_block_hash_with_group_id(block_hash, 0)
        )

    def _refresh(self, block_hash: BlockHash) -> None:
        """Move the hash's free cached copies to or from the side pool and
        re-key the ones that stay (the simulator's _want_on / _want_off)."""
        queue = self._pool.free_block_queue
        if self.wanted(block_hash):
            for block in self._copies(block_hash):
                if block.ref_cnt != 0:
                    continue
                if block.block_id not in self.side:
                    queue.remove(block)
                    self.side[block.block_id] = block
                self._push(block)
        else:
            for block in self._copies(block_hash):
                if self.side.pop(block.block_id, None) is not None:
                    queue.append(block)

    # ------------------------------------------------------------- events

    def add(self, block: KVCacheBlock) -> None:
        """A freed cached block whose hash is wanted."""
        self.side[block.block_id] = block
        self._push(block)

    def on_demand_change(
        self, counts: Mapping[BlockHash, int], path: Sequence[BlockHash]
    ) -> None:
        """Waiting-index counts changed for ``counts`` (hash -> new count),
        all from one request ``path``: apply in path order, so copies return
        to the LRU tail shallowest first, as in the simulator."""
        start = time.perf_counter_ns()
        seen = 0
        for h in path:
            count = counts.get(h)
            if count is None:
                continue
            seen += 1
            self._on_count(h, count)
        if seen < len(counts):  # hashes no longer on the path (replaced path)
            on_path = set(path)
            for h, count in counts.items():
                if h not in on_path:
                    self._on_count(h, count)
        self.elapsed_ns += time.perf_counter_ns() - start

    def _on_count(self, h: BlockHash, count: int) -> None:
        demanded = self._demanded
        if count > 0:
            if h in demanded:
                # Demand stays positive: the hash stays wanted, in the demand
                # tier, with the same rank and depth (cumulative hashes have
                # one depth), and its free copies are already in the side
                # pool. Its key is unchanged, so there is nothing to refresh.
                return
            demanded.add(h)
        else:
            demanded.discard(h)
            if h not in self._hold:
                self._rank_extra[h] = math.inf  # demand ended (rule R)
        self._refresh(h)

    def on_preempt(self, path: Iterable[BlockHash]) -> None:
        """The request resumes at the queue head: its path ranks 0."""
        start = time.perf_counter_ns()
        for h in path:
            self._rank_extra[h] = 0
            for block in self._copies(h):
                if block.block_id in self.side:
                    self._push(block)
        self.elapsed_ns += time.perf_counter_ns() - start

    def set_snapshot(self, snapshot: list[tuple[Sequence[BlockHash], int]]) -> None:
        """New waiting order: (index path, prompt full blocks) per request."""
        self._snapshot = snapshot
        self._rank_map = None
        self._rank_extra = {}
        self._heaps = {HOLD_TIER: None, DEMAND_TIER: None}

    def pop_victims(self, num_blocks: int) -> list[KVCacheBlock]:
        """Take side-pool blocks once the LRU queue is empty, in key order."""
        start = time.perf_counter_ns()
        victims: list[KVCacheBlock] = []
        for tier in (HOLD_TIER, DEMAND_TIER):
            while len(victims) < num_blocks:
                heap = self._heaps[tier]
                if heap is None:
                    heap = [
                        (self._key(b), b.block_id)
                        for b in self.side.values()
                        if self._tier(b) == tier
                    ]
                    heapq.heapify(heap)
                    self._heaps[tier] = heap
                if not heap:
                    break
                key, block_id = heapq.heappop(heap)
                block = self.side.get(block_id)
                if block is None or self._key(block) != key:
                    continue  # left the side pool, or re-keyed (a newer entry exists)
                del self.side[block_id]
                victims.append(block)
                if tier == HOLD_TIER:
                    self.num_hold_evictions += 1
                else:
                    self.num_demand_evictions += 1
        if len(victims) < num_blocks:
            raise RuntimeError("demand retention: side pool has too few blocks")
        self.elapsed_ns += time.perf_counter_ns() - start
        return victims

    # ------------------------------------------------------------- holds

    def turn_arrived(self, session_id: str, arrival_time: float) -> None:
        """Learn the session's tool gap; the turn's own demand replaces the
        hold (call drop_hold next)."""
        end = self._session_end.get(session_id)
        if end is None:
            return
        gap = max(arrival_time - end, 0.0)
        prev = self._gap_session.get(session_id)
        self._gap_session[session_id] = (
            gap if prev is None else GAP_SESSION_OLD * prev + GAP_SESSION_NEW * gap
        )
        self._gap_all = (
            gap
            if self._gap_all is None
            else GAP_GLOBAL_OLD * self._gap_all + GAP_GLOBAL_NEW * gap
        )

    def hold_turn(self, session_id: str, block_hashes: Sequence[BlockHash]) -> None:
        """A session turn finished: hold its full blocks until its predicted
        return. Call before its blocks are freed."""
        start = time.perf_counter_ns()
        now = self.clock()
        self._session_end[session_id] = now
        self.drop_hold(session_id)
        gap = self._gap_session.get(
            session_id,
            self._gap_all if self._gap_all is not None else self.initial_gap_s,
        )
        expiry = now + self.ttl_factor * gap
        hashes = tuple(block_hashes)
        for depth, h in enumerate(hashes):
            entry = self._hold.get(h)
            if entry is None:
                self._hold[h] = [1, depth]
            else:
                entry[0] += 1
            self._hold_t[h] = max(self._hold_t.get(h, 0.0), expiry)
            self._refresh(h)
        self._session_hold[session_id] = (expiry, hashes)
        self._hold_seq += 1
        heapq.heappush(self._hold_expiry, (expiry, self._hold_seq, session_id))
        self.elapsed_ns += time.perf_counter_ns() - start

    def drop_hold(self, session_id: str) -> None:
        held = self._session_hold.pop(session_id, None)
        if held is None:
            return
        for h in held[1]:
            entry = self._hold[h]
            if entry[0] == 1:
                del self._hold[h]
                del self._hold_t[h]
                self._refresh(h)
            else:
                entry[0] -= 1

    def expire_holds(self) -> None:
        start = time.perf_counter_ns()
        now = self.clock()
        heap = self._hold_expiry
        while heap and heap[0][0] <= now:
            expiry, _, session_id = heapq.heappop(heap)
            held = self._session_hold.get(session_id)
            if held is not None and held[0] == expiry:
                self.drop_hold(session_id)
        self.elapsed_ns += time.perf_counter_ns() - start

    # ------------------------------------------------------------- reset

    def reset(self) -> list[KVCacheBlock]:
        """Forget all predictions; return the side-pool blocks to free."""
        blocks = list(self.side.values())
        self.side.clear()
        self._heaps = {HOLD_TIER: None, DEMAND_TIER: None}
        self._hold.clear()
        self._hold_t.clear()
        self._session_hold.clear()
        self._hold_expiry.clear()
        self._session_end.clear()
        self._gap_session.clear()
        self._gap_all = None
        self._snapshot = []
        self._rank_map = None
        self._rank_extra = {}
        return blocks
