# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import OrderedDict, defaultdict
from collections.abc import Iterable

from vllm.v1.kv_offload.base import OffloadKey
from vllm.v1.kv_offload.cpu.policies.base import BlockStatus, CachePolicy


class LFUCachePolicy(CachePolicy):
    """LFU cache policy with frequency buckets and FIFO tie-breaking.

    Each block tracks an access frequency. On eviction, the block with the
    lowest frequency is removed first; ties within a frequency bucket are
    broken by insertion order (FIFO).
    """

    def __init__(self, cache_capacity: int):
        # cache_capacity unused by LFU but accepted for a uniform constructor
        self.blocks: dict[OffloadKey, BlockStatus] = {}
        self.freq: dict[OffloadKey, int] = {}
        # frequency -> OrderedDict[key, None] (ordered set, FIFO within bucket)
        self.freq_buckets: dict[int, OrderedDict[OffloadKey, None]] = (
            defaultdict(OrderedDict)
        )

    def get(self, key: OffloadKey) -> BlockStatus | None:
        return self.blocks.get(key)

    def insert(self, key: OffloadKey, block: BlockStatus) -> None:
        self.blocks[key] = block
        self.freq[key] = 1
        self.freq_buckets[1][key] = None

    def remove(self, key: OffloadKey) -> None:
        del self.blocks[key]
        f = self.freq.pop(key)
        del self.freq_buckets[f][key]
        if not self.freq_buckets[f]:
            del self.freq_buckets[f]

    def touch(self, keys: Iterable[OffloadKey]) -> None:
        for key in reversed(list(keys)):
            if key in self.blocks:
                self._increment_freq(key)

    def _increment_freq(self, key: OffloadKey) -> None:
        old = self.freq[key]
        new = old + 1
        self.freq[key] = new
        del self.freq_buckets[old][key]
        if not self.freq_buckets[old]:
            del self.freq_buckets[old]
        self.freq_buckets[new][key] = None

    def clear(self) -> None:
        self.blocks.clear()
        self.freq.clear()
        self.freq_buckets.clear()

    def evict(
        self, n: int, protected: set[OffloadKey]
    ) -> list[tuple[OffloadKey, BlockStatus]] | None:
        if n == 0:
            return []
        candidates: list[tuple[OffloadKey, BlockStatus]] = []
        # Walk frequency buckets from lowest, collecting evictable blocks.
        for f in sorted(self.freq_buckets.keys()):
            for key in self.freq_buckets[f]:
                block = self.blocks[key]
                if block.ref_cnt == 0 and key not in protected:
                    candidates.append((key, block))
                    if len(candidates) == n:
                        break
            if len(candidates) == n:
                break
        if len(candidates) < n:
            return None
        for key, _ in candidates:
            # Reuse remove() to keep bucket state consistent.
            self.remove(key)
        return candidates
