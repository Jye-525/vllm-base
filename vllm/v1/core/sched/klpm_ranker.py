# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""k-LPM waiting-queue ordering (Dexter et al., NeurIPS 2025, arXiv:2502.04677,
Algorithm 1).

Port of the KVReuse simulator's ``simulator/sched-sim/klpm_order.py``; keys,
stable sorts and tie rules are the same. Functions take mappings keyed by
request id.

Algorithm 1 admits the oldest waiting request, then the k - 1 requests with
the longest cached prefix, and repeats; k = 1 is FCFS. The cycle counts
**fresh admissions, not scheduler steps**: ``KLPMCycle`` keeps the position
across steps and the scheduler calls ``on_admit`` once per fresh admission.
Restarting the cycle every step would admit the oldest request first in
every step, i.e. plain FCFS whenever about one request is admitted per step.

Ties in cached length keep the input order (a stable sort, as LPM does).
Preempted requests are excluded from both the count and the oldest pick:
vLLM resumes them at the queue head.

``rank_klpm`` plans the whole order assuming admissions take a prefix of it.
Removing the plan's head and advancing the cycle by one leaves the same
tail, so the plan stays valid while admissions follow it. When the scheduler
sets a request aside instead (blocked, LoRA, KV connector), it re-ranks from
the advanced cycle position.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Iterable, Mapping
from typing import Any, TypeVar

T = TypeVar("T", bound=Hashable)


def check_k(k: Any) -> int:
    """k as an int >= 1 (accepts a whole float such as 4.0)."""
    if (
        isinstance(k, bool)
        or not isinstance(k, (int, float))
        or not math.isfinite(k)
        or k != int(k)
        or k < 1
    ):
        raise ValueError(f"k-LPM k must be a whole number >= 1, got {k!r}")
    return int(k)


class KLPMCycle:
    """Position in the k-cycle, in fresh admissions since the cycle started."""

    def __init__(self, k: int) -> None:
        self.k = check_k(k)
        self.pos = 0  # 0: the next admission is the oldest request

    def on_admit(self) -> None:
        """A fresh request was just admitted."""
        self.pos = (self.pos + 1) % self.k

    def next_is_oldest(self) -> bool:
        return self.pos == 0


def rank_klpm(
    items: Iterable[T],
    hit_tokens: Mapping[T, int],
    arrival: Mapping[T, Any],
    k: int,
    pos: int = 0,
) -> list[T]:
    """k-LPM order of the fresh waiting requests ``items``.

    ``hit_tokens`` maps each item to its cached prefix length, ``arrival`` to
    a comparable arrival key (ties broken). Position j of the result (0-based)
    is the oldest unplaced request when ``(pos + j) % k == 0``, otherwise the
    unplaced request with the longest cached prefix.
    """
    k = check_k(k)
    items = list(items)
    lpm = sorted(items, key=lambda r: -hit_tokens[r])  # stable
    oldest = sorted(items, key=lambda r: arrival[r])
    out: list[T] = []
    placed: set[T] = set()
    i = j = 0  # cursors into lpm and oldest
    for n in range(len(lpm)):
        if (pos + n) % k == 0:
            while oldest[j] in placed:
                j += 1
            r = oldest[j]
        else:
            while lpm[i] in placed:
                i += 1
            r = lpm[i]
        placed.add(r)
        out.append(r)
    return out
