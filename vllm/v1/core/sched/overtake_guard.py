# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Overtaking guard for SSPF: a bound on how far a request can be overtaken.

Port of the KVReuse simulator's ``simulator/sched-sim/overtake_guard.py``
(``og_k``); same rule and tie order. Requests are keyed by request id.

Overtaken work O(r) of a waiting request r is the uncached prompt tokens of
requests that arrived after r but were admitted before it. Requests that
arrived earlier never count: FCFS would have served them first as well, so
O(r) is only the delay caused by reordering. Request r is *due* once

    O(r) >= k * U(r)

where U(r) is r's own uncached tokens now (at least one 16-token block). Due
requests go to the front of the fresh waiting queue, oldest first; the rest
keep the base (SSPF) order. Admission is unchanged, so a due request at the
front is admitted as soon as its KV fits, and no request is overtaken by more
than k times its own work before it reaches the front.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping
from typing import Any, TypeVar

T = TypeVar("T", bound=Hashable)

# One KV block: a fully cached request still counts as one block of work.
MIN_UNCACHED = 16


class OvertakeGuard:
    """Overtaken work of the waiting requests, updated at every fresh admission."""

    def __init__(self, k: float) -> None:
        if not k > 0:
            raise ValueError("overtaking guard k must be > 0")
        self.k = float(k)
        self.over: dict[Hashable, int] = {}
        self.promoted = 0  # diagnostic: requests admitted while due

    def on_admit(
        self,
        admitted: Hashable,
        work: int,
        admitted_key: Any,
        waiting_keys: Mapping[Hashable, Any],
        was_due: bool = False,
    ) -> None:
        """Record the fresh admission of ``admitted`` with ``work`` uncached
        tokens. ``admitted_key`` and ``waiting_keys`` (request -> key) are
        arrival order keys; every request still waiting that arrived before
        ``admitted`` has been overtaken by ``work`` more tokens."""
        self.over.pop(admitted, None)
        self.promoted += bool(was_due)
        for q, key in waiting_keys.items():
            if key < admitted_key:
                self.over[q] = self.over.get(q, 0) + work

    def is_due(self, item: Hashable, uncached_tokens: int) -> bool:
        return self.over.get(item, 0) >= self.k * max(MIN_UNCACHED, uncached_tokens)

    def forget(self, item: Hashable) -> None:
        self.over.pop(item, None)


def order_with_guard(
    items: Iterable[T],
    uncached_tokens: Mapping[T, int],
    arrival: Mapping[T, Any],
    guard: OvertakeGuard,
) -> tuple[list[T], list[T]]:
    """``items`` in base order, with the due ones moved to the front, oldest
    first. Returns (new order, due items)."""
    items = list(items)
    due = [r for r in items if guard.is_due(r, uncached_tokens[r])]
    if not due:
        return items, []
    first = set(due)
    due.sort(key=lambda r: arrival[r])
    return due + [r for r in items if r not in first], due
