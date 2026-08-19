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
    """

    def __init__(self) -> None:
        self._request_paths: dict[str, tuple[BlockHash, ...]] = {}
        self._request_ready: dict[str, bool] = {}
        self._pending_counts: dict[BlockHash, int] = {}
        self._ready_counts: dict[BlockHash, int] = {}
        self._version = 0

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
            return {}

        old_hashes = set(old_path or ())
        new_hashes = set(new_path)
        changed_hashes = old_hashes ^ new_hashes

        self._apply_count_delta(self._pending_counts, old_hashes - new_hashes, -1)
        self._apply_count_delta(self._pending_counts, new_hashes - old_hashes, 1)
        if old_ready:
            self._apply_count_delta(self._ready_counts, old_hashes, -1)
        if ready:
            self._apply_count_delta(self._ready_counts, new_hashes, 1)

        self._request_paths[request_id] = new_path
        self._request_ready[request_id] = ready
        self._version += 1
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
            changed_counts[block_hash] = count

        if ready:
            self._apply_count_delta(self._ready_counts, path_hashes, -1)

        self._version += 1
        return changed_counts

    def clear(self) -> dict[BlockHash, int]:
        """Remove every request and return zero counts for removed hashes."""
        if not self._request_paths:
            return {}
        changed_counts = dict.fromkeys(self._pending_counts, 0)
        self._request_paths.clear()
        self._request_ready.clear()
        self._pending_counts.clear()
        self._ready_counts.clear()
        self._version += 1
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
        return True

    def rank_info_many(
        self,
        request_ids: Sequence[str],
        claim_depth_blocks: int,
    ) -> dict[str, PendingRankInfo]:
        """Build ready-only sharing metadata for a batch of requests."""
        result: dict[str, PendingRankInfo] = {}
        for request_id in request_ids:
            path = self._request_paths.get(request_id)
            if path is None or not self._request_ready[request_id]:
                continue

            cluster_hash = None
            cluster_depth = 0
            cluster_size = 1
            for depth in range(len(path), 0, -1):
                block_hash = path[depth - 1]
                count = self._ready_counts.get(block_hash, 0)
                if count >= 2:
                    cluster_hash = block_hash
                    cluster_depth = depth
                    cluster_size = count
                    break

            claim_hash = (
                path[claim_depth_blocks - 1]
                if claim_depth_blocks > 0 and len(path) >= claim_depth_blocks
                else None
            )
            result[request_id] = PendingRankInfo(
                request_score=sum(self._ready_counts.get(h, 0) for h in path),
                cluster_hash=cluster_hash,
                cluster_depth=cluster_depth,
                cluster_size=cluster_size,
                claim_hash=claim_hash,
            )
        return result

    def longest_shared_prefix_blocks(self, request_id: str) -> int:
        """Return the deepest prefix shared with another pending request."""
        path = self._request_paths.get(request_id)
        if path is None:
            return 0
        return max(
            (
                depth
                for depth, block_hash in enumerate(path, 1)
                if self._pending_counts[block_hash] >= 2
            ),
            default=0,
        )

    def cluster_info(self, request_id: str) -> tuple[BlockHash, int, int] | None:
        """Return ``(cluster_hash, depth_blocks, cluster_size)``.

        The cluster is the request's deepest prefix shared with at least one
        other pending request.
        """
        path = self._request_paths.get(request_id)
        if path is None:
            return None
        for depth in range(len(path), 0, -1):
            block_hash = path[depth - 1]
            count = self._pending_counts[block_hash]
            if count >= 2:
                return block_hash, depth, count
        return None

    def request_score(self, request_id: str) -> int:
        """Return the block-granularity pending-sharing score."""
        path = self._request_paths.get(request_id)
        if path is None:
            return 0
        return sum(self._pending_counts[h] for h in path)
