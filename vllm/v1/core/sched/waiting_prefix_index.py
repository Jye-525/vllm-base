# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Incremental block-prefix index for waiting requests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from vllm.v1.core.kv_cache_utils import BlockHash


@dataclass(frozen=True, slots=True)
class PendingRankInfo:
    """Prefix-sharing metadata used by the waiting-request ranker."""

    request_score: int
    cluster_hash: BlockHash | None
    cluster_depth: int
    cluster_size: int
    claim_hash: BlockHash | None


class PendingPrefixIndex:
    """Track block-prefix sharing across pending requests.

    vLLM block hashes are cumulative: each hash incorporates the preceding
    block hash. Therefore, the number of request paths containing a hash is
    also the number of pending requests sharing the prefix ending at that
    block. Updates return the absolute counts of changed hashes so consumers
    can maintain derived state incrementally.

    Derived per-request values (cLPM rank info, the SSPF run sum) depend only
    on the ready counts along the request's own path. Every request that
    contains one of those hashes shares the path's first hash (its *root*),
    so a change to any request's path or readiness bumps only its root's
    version, and a cached value stays valid while its root's version does.
    """

    def __init__(self) -> None:
        self._request_paths: dict[str, tuple[BlockHash, ...]] = {}
        self._request_ready: dict[str, bool] = {}
        self._pending_counts: dict[BlockHash, int] = {}
        # PEEK cluster eviction uses the deepest position at which a demanded
        # hash occurs. Keep the maximum lazily while demand remains positive,
        # matching PEEK's vLLM implementation.
        self._pending_max_depths: dict[BlockHash, int] = {}
        self._ready_counts: dict[BlockHash, int] = {}
        self._version = 0
        # Root (first path hash) -> version, bumped whenever a request whose
        # path starts there is inserted, removed, replaced or changes
        # readiness. Dropped once no pending path uses the root.
        self._root_versions: dict[BlockHash, int] = {}
        # request id -> (root version, info), per (claim depth, score flag).
        self._rank_info_cache: dict[
            tuple[int, bool], dict[str, tuple[int, PendingRankInfo]]
        ] = {}
        # request id -> (start block, root version, amortized blocks).
        self._amortized_cache: dict[str, tuple[int, int, float]] = {}

    def __len__(self) -> int:
        return len(self._request_paths)

    def __contains__(self, request_id: str) -> bool:
        return request_id in self._request_paths

    @property
    def version(self) -> int:
        return self._version

    @property
    def demand_counts(self) -> Mapping[BlockHash, int]:
        """Return a read-only live view of per-prefix pending counts."""
        return MappingProxyType(self._pending_counts)

    @property
    def demand_max_depths(self) -> Mapping[BlockHash, int]:
        """Return the deepest pending-chain position for each demanded hash."""
        return MappingProxyType(self._pending_max_depths)

    def request_path(self, request_id: str) -> tuple[BlockHash, ...] | None:
        return self._request_paths.get(request_id)

    def pending_count(self, block_hash: BlockHash) -> int:
        return self._pending_counts.get(block_hash, 0)

    def ready_count(self, block_hash: BlockHash) -> int:
        """Return the number of rank-ready requests containing a prefix."""
        return self._ready_counts.get(block_hash, 0)

    def is_ready(self, request_id: str) -> bool:
        return self._request_ready.get(request_id, False)

    @staticmethod
    def _apply_count_delta(
        counts: dict[BlockHash, int],
        block_hashes: set[BlockHash],
        delta: int,
    ) -> None:
        for block_hash in block_hashes:
            count = counts.get(block_hash, 0) + delta
            if count:
                counts[block_hash] = count
            else:
                counts.pop(block_hash, None)

    def _touch(self, path: Sequence[BlockHash] | None) -> None:
        """A request with this path changed: its root's derived values are stale."""
        if path:
            root = path[0]
            self._root_versions[root] = self._root_versions.get(root, 0) + 1

    def _root_version(self, path: Sequence[BlockHash]) -> int:
        return self._root_versions.get(path[0], 0) if path else -1

    def _forget_derived(self, request_id: str) -> None:
        self._amortized_cache.pop(request_id, None)
        for cache in self._rank_info_cache.values():
            cache.pop(request_id, None)

    def _drop_unused_root(self, path: Sequence[BlockHash] | None) -> None:
        """Forget the version of a root no pending path uses any more (no
        cached value can refer to it: its requests' entries are gone)."""
        if path and path[0] not in self._pending_counts:
            self._root_versions.pop(path[0], None)

    def upsert(
        self,
        request_id: str,
        block_hashes: Sequence[BlockHash],
        *,
        ready: bool = True,
    ) -> dict[BlockHash, int]:
        """Insert or replace a request path.

        Returns a mapping from every changed hash to its new absolute count.
        An unchanged path is a no-op.
        """
        new_path = tuple(block_hashes)
        old_path = self._request_paths.get(request_id)
        old_ready = self._request_ready.get(request_id, False)
        if old_path == new_path and old_ready == ready:
            return {}

        if old_path == new_path:
            path_hashes = set(new_path)
            self._apply_count_delta(self._ready_counts, path_hashes, 1 if ready else -1)
            self._request_ready[request_id] = ready
            self._version += 1
            self._touch(new_path)
            return {}

        old_hashes = set(old_path or ())
        new_hashes = set(new_path)
        changed_hashes = old_hashes ^ new_hashes

        self._apply_count_delta(self._pending_counts, old_hashes - new_hashes, -1)
        self._apply_count_delta(self._pending_counts, new_hashes - old_hashes, 1)
        for block_hash in old_hashes - new_hashes:
            if block_hash not in self._pending_counts:
                self._pending_max_depths.pop(block_hash, None)
        for depth, block_hash in enumerate(new_path, 1):
            self._pending_max_depths[block_hash] = max(
                depth, self._pending_max_depths.get(block_hash, 0)
            )
        if old_ready:
            self._apply_count_delta(self._ready_counts, old_hashes, -1)
        if ready:
            self._apply_count_delta(self._ready_counts, new_hashes, 1)

        self._request_paths[request_id] = new_path
        self._request_ready[request_id] = ready
        self._version += 1
        self._touch(old_path)
        self._touch(new_path)
        self._forget_derived(request_id)
        self._drop_unused_root(old_path)
        return {
            block_hash: self._pending_counts.get(block_hash, 0)
            for block_hash in changed_hashes
        }

    def discard(self, request_id: str) -> dict[BlockHash, int]:
        """Remove a request if present and return changed absolute counts."""
        path = self._request_paths.pop(request_id, None)
        if path is None:
            return {}

        ready = self._request_ready.pop(request_id)

        changed_counts: dict[BlockHash, int] = {}
        path_hashes = set(path)
        for block_hash in path_hashes:
            count = self._pending_counts[block_hash] - 1
            if count:
                self._pending_counts[block_hash] = count
            else:
                del self._pending_counts[block_hash]
                self._pending_max_depths.pop(block_hash, None)
            changed_counts[block_hash] = count

        if ready:
            self._apply_count_delta(self._ready_counts, path_hashes, -1)

        self._version += 1
        self._touch(path)
        self._forget_derived(request_id)
        self._drop_unused_root(path)
        return changed_counts

    def clear(self) -> dict[BlockHash, int]:
        """Remove every request and return zero counts for removed hashes."""
        if not self._request_paths:
            return {}
        changed_counts = dict.fromkeys(self._pending_counts, 0)
        self._request_paths.clear()
        self._request_ready.clear()
        self._pending_counts.clear()
        self._pending_max_depths.clear()
        self._ready_counts.clear()
        self._version += 1
        self._root_versions.clear()
        self._rank_info_cache.clear()
        self._amortized_cache.clear()
        return changed_counts

    def set_ready(self, request_id: str, ready: bool) -> bool:
        """Change rank eligibility without changing eviction demand.

        Returns whether the index changed.
        """
        path = self._request_paths.get(request_id)
        if path is None:
            return False
        old_ready = self._request_ready[request_id]
        if old_ready == ready:
            return False
        self._apply_count_delta(self._ready_counts, set(path), 1 if ready else -1)
        self._request_ready[request_id] = ready
        self._version += 1
        self._touch(path)
        return True

    def rank_info_many(
        self,
        request_ids: Sequence[str],
        claim_depth_blocks: int,
        *,
        include_request_score: bool = True,
    ) -> dict[str, PendingRankInfo]:
        """Build ready-only sharing metadata for a batch of requests.

        A request's metadata stays cached while its root's version does: only
        requests sharing its path's first hash can change the counts on it.
        """
        cache = self._rank_info_cache.setdefault(
            (claim_depth_blocks, include_request_score), {}
        )
        out: dict[str, PendingRankInfo] = {}
        for request_id in request_ids:
            path = self._request_paths.get(request_id)
            if path is None or not self._request_ready[request_id]:
                continue
            version = self._root_version(path)
            hit = cache.get(request_id)
            if hit is not None and hit[0] == version:
                out[request_id] = hit[1]
                continue
            info = self._rank_info(path, claim_depth_blocks, include_request_score)
            cache[request_id] = (version, info)
            out[request_id] = info
        return out

    def _rank_info(
        self,
        path: tuple[BlockHash, ...],
        claim_depth_blocks: int,
        include_request_score: bool,
    ) -> PendingRankInfo:
        counts = self._ready_counts
        # Deepest block shared by at least two ready requests. Ready counts
        # never increase along a (cumulative) path, so the blocks with a
        # count of two or more form a prefix: binary-search its end.
        lo, hi = 0, len(path)
        while lo < hi:
            mid = (lo + hi) // 2
            if counts.get(path[mid], 0) >= 2:
                lo = mid + 1
            else:
                hi = mid
        if lo:
            cluster_hash: BlockHash | None = path[lo - 1]
            cluster_depth = lo
            cluster_size = counts[path[lo - 1]]
        else:
            cluster_hash, cluster_depth, cluster_size = None, 0, 1
        claim_hash = (
            path[claim_depth_blocks - 1]
            if claim_depth_blocks > 0 and len(path) >= claim_depth_blocks
            else None
        )
        return PendingRankInfo(
            request_score=(self._count_sum(path) if include_request_score else 0),
            cluster_hash=cluster_hash,
            cluster_depth=cluster_depth,
            cluster_size=cluster_size,
            claim_hash=claim_hash,
        )

    def _runs(self, path: tuple[BlockHash, ...], start: int):
        """(run length, raw ready count) of the runs of equal counts along
        ``path[start:]``, each run's end found by binary search (counts never
        increase along a path). A count below 1 is its own run of the floored
        count, as ``amortized_blocks`` has always summed it."""
        counts = self._ready_counts
        i = max(start, 0)
        n = len(path)
        while i < n:
            count = max(counts.get(path[i], 0), 1)
            # First position after i whose count is below this run's.
            lo, hi = i + 1, n
            while lo < hi:
                mid = (lo + hi) // 2
                if counts.get(path[mid], 0) >= count:
                    lo = mid + 1
                else:
                    hi = mid
            yield lo - i, count, counts.get(path[i], 0)
            i = lo

    def _count_sum(self, path: tuple[BlockHash, ...]) -> int:
        """``sum(ready_count(h) for h in path)`` by runs of equal counts."""
        total = 0
        for length, _, raw in self._runs(path, 0):
            total += length * raw
        return total

    def amortized_blocks(self, request_id: str, start_block: int) -> float:
        """Return ``sum(1 / ready_count(h))`` over the path from ``start_block``.

        Each block counts once per rank-ready request that contains it, so a
        block shared by n waiting requests costs 1/n. Hashes are cumulative,
        so counts never increase along a path: the sum is taken one run of
        equal counts at a time, each run's end found by binary search.
        Counts are floored at 1 (a request that is not ready yet). The sum is
        cached per request while its start block and root version hold.
        """
        path = self._request_paths.get(request_id)
        if path is None:
            return 0.0
        version = self._root_version(path)
        hit = self._amortized_cache.get(request_id)
        if hit is not None and hit[0] == start_block and hit[1] == version:
            return hit[2]
        total = 0.0
        for length, count, _ in self._runs(path, start_block):
            total += length / count
        self._amortized_cache[request_id] = (start_block, version, total)
        return total
