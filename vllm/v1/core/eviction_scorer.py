# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Waiting-queue-aware demand scoring for GPU prefix-cache eviction.

The scheduler turns the waiting queue into a per-block "demand score". The
score is an additive mixture of three scheduler-visible signals:

    score(h) = w_pos   * nearest_reuse(h)
             + w_reuse * reuse_count(h)
             + w_depth * chain_loss_depth(h)

By default, all three components are normalized to make the mixture weights
interpretable.

``nearest_reuse(h)`` is the highest ``gamma ** p`` among waiting requests that
reuse ``h``. ``reuse_count(h)`` is the normalized number of waiters that reuse
``h``. ``chain_loss_depth(h)`` is derived from the block's position in each
waiter's contiguous cached prefix. For a prefix of length ``L`` and block index
``j`` (0-based), the default depth contribution is:

    ((L - j) / L) ** alpha

This is a chain-loss signal: earlier blocks in a cached prefix protect more
downstream reuse if retained. A raw ``(L - j) ** alpha`` mode is also available
for experiments where depth should intentionally dominate the mixture scale.

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
    position_weight: float = 1.0 / 3.0,
    reuse_weight: float = 1.0 / 3.0,
    depth_weight: float = 1.0 / 3.0,
    depth_alpha: float = 1.0,
    depth_mode: str = "chain_loss_norm",
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
        position_weight: Mixture weight for nearest queued reuse. The provided
            weights are normalized internally, so they need not sum to 1.
        reuse_weight: Mixture weight for upcoming reuse frequency.
        depth_weight: Mixture weight for chain-loss depth.
        depth_alpha: Exponent applied to the chain-loss depth signal.
        depth_mode: ``"chain_loss_norm"`` uses ``((L - j) / L) ** alpha``;
            ``"chain_loss"`` uses raw ``(L - j) ** alpha``.

    Returns:
        Mapping from block hash to its demand score. Hashes reused by no waiter
        are absent (implicitly score 0).
    """
    position_weight = max(0.0, position_weight)
    reuse_weight = max(0.0, reuse_weight)
    depth_weight = max(0.0, depth_weight)
    total_weight = position_weight + reuse_weight + depth_weight
    if total_weight <= 0.0:
        position_weight = reuse_weight = depth_weight = 1.0 / 3.0
    else:
        position_weight /= total_weight
        reuse_weight /= total_weight
        depth_weight /= total_weight

    position_scores: dict[BlockHash, float] = {}
    reuse_counts: dict[BlockHash, int] = {}
    depth_scores: dict[BlockHash, float] = {}
    weight = 1.0
    for prefix in waiter_prefixes:
        if weight < epsilon:
            break
        if prefix:
            prefix_len = len(prefix)
            for j, h in enumerate(prefix):
                position_scores[h] = max(position_scores.get(h, 0.0), weight)
                reuse_counts[h] = reuse_counts.get(h, 0) + 1
                if depth_mode == "chain_loss":
                    depth = float(prefix_len - j) ** depth_alpha
                else:
                    depth = ((prefix_len - j) / prefix_len) ** depth_alpha
                depth_scores[h] = max(depth_scores.get(h, 0.0), depth)
        weight *= gamma

    if not reuse_counts:
        return {}

    max_reuse = max(reuse_counts.values())
    scores: dict[BlockHash, float] = {}
    for h, count in reuse_counts.items():
        reuse_score = count / max_reuse
        scores[h] = (
            position_weight * position_scores[h]
            + reuse_weight * reuse_score
            + depth_weight * depth_scores[h]
        )
    return scores
