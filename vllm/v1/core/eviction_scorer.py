# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Waiting-queue-aware demand scoring for GPU prefix-cache eviction.

The scheduler turns the waiting queue into a per-block "demand score": for a
cached block hash ``h``,

    demand_score(h) = sum over waiters p that reuse h of  gamma ** p

where ``p`` is the position of the waiter in the scheduling queue (``p == 0``
is the request scheduled next). A block reused by a near waiter, or by many
waiters, scores higher and is protected from eviction first.

``gamma`` is a single dial between two regimes:
  - ``gamma -> 0``  : only the next waiter matters  -> "soonest-reuse wins"
                      (Belady-style nearest-reuse).
  - ``gamma -> 1``  : every waiter counts equally    -> "popularity wins"
                      (the score becomes the demand count).

This module is intentionally free of vLLM internals so it can be unit-tested in
isolation: it consumes the contiguous cached prefix of each waiter (in queue
order) and returns the score dict. The scheduler is responsible for computing
those prefixes (via ``BlockPool.classify_chain``) and for bounding how many
waiters are scanned.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # BlockHash is `NewType("BlockHash", bytes)`; only needed for typing, so
    # keep this module free of heavy vLLM imports (unit-testable in isolation).
    from vllm.v1.core.kv_cache_utils import BlockHash


def compute_demand_scores(
    waiter_prefixes: Iterable[Sequence[BlockHash]],
    gamma: float,
    epsilon: float = 1e-3,
) -> dict[BlockHash, float]:
    """Compute per-block demand scores from the waiting queue.

    Args:
        waiter_prefixes: One entry per waiting request, **in queue order**
            (index 0 == scheduled next). Each entry is the waiter's contiguous
            cached prefix — the block hashes it would reuse. Entries may be
            empty (the waiter has no cached prefix); they still advance the
            position so later waiters keep the correct ``gamma ** p`` weight.
        gamma: Position-decay factor in ``(0, 1]``. Smaller favors imminence;
            ``1.0`` weights all positions equally (pure demand count).
        epsilon: Stop accumulating once ``gamma ** p < epsilon`` — far waiters
            contribute negligibly. Ignored when ``gamma >= 1`` (weight never
            decays), so the caller must bound the iterable in that case.

    Returns:
        Mapping from block hash to its demand score. Hashes reused by no waiter
        are absent (implicitly score 0).
    """
    scores: dict[BlockHash, float] = {}
    weight = 1.0
    for prefix in waiter_prefixes:
        if weight < epsilon:
            break
        if prefix:
            for h in prefix:
                scores[h] = scores.get(h, 0.0) + weight
        weight *= gamma
    return scores
