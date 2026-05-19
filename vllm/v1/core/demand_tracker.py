# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Demand-aware eviction tracker for GPU prefix cache.

Tracks how many waiting requests need each block hash, enabling the
BlockPool to prefer evicting blocks that no future request will use.
"""

from collections.abc import Sequence

from vllm.v1.core.kv_cache_utils import BlockHash


class DemandTracker:
    """Counts how many waiting requests reference each block hash.

    Updated incrementally as requests enter/leave the waiting queue.
    Used by BlockPool to make demand-aware eviction decisions:
    blocks with zero demand are evicted before blocks with positive demand.
    """

    def __init__(self) -> None:
        # BlockHash -> number of waiting requests that contain this hash
        self._demand: dict[BlockHash, int] = {}

    def add_request(self, block_hashes: Sequence[BlockHash]) -> None:
        """Called when a request enters the waiting queue."""
        demand = self._demand
        for bh in block_hashes:
            demand[bh] = demand.get(bh, 0) + 1

    def remove_request(self, block_hashes: Sequence[BlockHash]) -> None:
        """Called when a request leaves the waiting queue."""
        demand = self._demand
        for bh in block_hashes:
            count = demand.get(bh, 0) - 1
            if count <= 0:
                demand.pop(bh, None)
            else:
                demand[bh] = count

    def get_demand(self, block_hash: BlockHash) -> int:
        """Get the number of waiting requests that need this block."""
        return self._demand.get(block_hash, 0)

    def has_demand(self, block_hash: BlockHash) -> bool:
        """Check if any waiting request needs this block."""
        return block_hash in self._demand

    @property
    def num_tracked_hashes(self) -> int:
        """Number of unique block hashes currently tracked."""
        return len(self._demand)
