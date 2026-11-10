# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RAGCache cache-aware reordering (Jin et al., ACM TOCS 44(1), 2025, §5.2).

Port of the KVReuse simulator's ``simulator/sched-sim/ragcache_order.py``;
keys, stable sorts and tie rules are the same. Functions take mappings keyed
by request id.

Priority = cached length / computation length (uncached tokens), highest
first, ties oldest first. The paper bounds starvation with a window of 32
requests but gives no exact rule; two readings are provided:

- ``bypass`` (default): a waiting request overtaken by ``window`` admissions
  of later arrivals is *due*; due requests rank first, oldest first.
  ``BypassWindow`` keeps the counters; the scheduler calls ``on_admit`` for
  each fresh admission. The bound counts admissions, not tokens or seconds.
- ``slide``: only the ``window`` oldest waiting requests are reordered; the
  rest follow in arrival order. No hard bound (newcomers keep entering).

``window == 0`` is the ratio alone, with no starvation guard.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping
from typing import Any, TypeVar

T = TypeVar("T", bound=Hashable)

WINDOW_MODES = ("bypass", "slide")


def reuse_ratio(hit_tokens: int, num_tokens: int) -> float:
    """Cached tokens per uncached token (at least one token is computed)."""
    return hit_tokens / max(1, num_tokens - hit_tokens)


class BypassWindow:
    """Per-request count of admissions of later arrivals (bypass reading)."""

    def __init__(self, window: int) -> None:
        if window <= 0:
            raise ValueError("bypass window must be > 0")
        self.window = window
        self.overtaken: dict[Hashable, int] = {}

    def on_admit(
        self,
        admitted: Hashable,
        admitted_arrival: Any,
        waiting_arrival: Mapping[Hashable, Any],
    ) -> None:
        """``admitted`` was just admitted. ``waiting_arrival`` maps each fresh
        request still waiting to its arrival key (comparable; ties broken,
        e.g. (time, sequence)); every one that arrived earlier than
        ``admitted_arrival`` was overtaken once more."""
        self.overtaken.pop(admitted, None)
        for r, key in waiting_arrival.items():
            if key < admitted_arrival:
                self.overtaken[r] = self.overtaken.get(r, 0) + 1

    def is_due(self, item: Hashable) -> bool:
        return self.overtaken.get(item, 0) >= self.window

    def forget(self, item: Hashable) -> None:
        self.overtaken.pop(item, None)


def _by_ratio(
    items: list[T], hit_tokens: Mapping[T, int], num_tokens: Mapping[T, int]
) -> list[T]:
    # stable: items oldest first gives ties oldest first
    return sorted(items, key=lambda r: -reuse_ratio(hit_tokens[r], num_tokens[r]))


def rank_ragcache(
    items: Iterable[T],
    hit_tokens: Mapping[T, int],
    num_tokens: Mapping[T, int],
    arrival: Mapping[T, Any],
    window: int = 32,
    mode: str = "bypass",
    bypass: BypassWindow | None = None,
) -> list[T]:
    """RAGCache order of the waiting requests ``items``.

    ``arrival`` maps each item to a comparable arrival key. ``mode`` is
    ``bypass`` (needs the ``BypassWindow`` state) or ``slide``; ``window``
    0 ranks by the ratio alone.
    """
    if mode not in WINDOW_MODES:
        raise ValueError(
            f"unknown RAGCache window mode {mode!r}; one of {WINDOW_MODES}"
        )
    oldest = sorted(items, key=lambda r: arrival[r])
    if window <= 0:
        return _by_ratio(oldest, hit_tokens, num_tokens)
    if mode == "slide":
        return _by_ratio(oldest[:window], hit_tokens, num_tokens) + oldest[window:]
    if bypass is None:
        raise ValueError("bypass mode needs a BypassWindow")
    due = [r for r in oldest if bypass.is_due(r)]
    rest = [r for r in oldest if not bypass.is_due(r)]
    return due + _by_ratio(rest, hit_tokens, num_tokens)
