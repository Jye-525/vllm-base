# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Session-history credit for SSPF (temporal sharing across agent turns).

A returning turn of a session reuses the previous turn's prompt as its own
prefix. If that history was evicted during the tool gap, plain SSPF charges
the turn for it again. The credit is the number of leading prompt blocks the
turn shares with the previous *arrived* turn of its session, computed once
at arrival. At ranking, each credited block past the usable cache hit is
discounted by ``(1 - w) * block_size`` tokens (``sspf_session_credit_weight``
is ``w``), the simulator's fixed-subtraction rule (KVReuse
``simulator/kvsim.py`` ``add_waiting`` and ``Sim.rank``).

The client supplies the session id through ``vllm_xargs`` (OpenAI API), which
reaches ``SamplingParams.extra_args["session_id"]``. Requests without one get
no credit. The session table keeps the most recently active sessions only,
so a long-running server cannot grow it without bound.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence

from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.request import Request

SESSION_ID_KEY = "session_id"
# Sessions remembered at once. A session idle while this many others arrive
# loses its credit on return; benchmark workloads stay far below it. Each
# entry keeps its prompt's block-hash tuple (~65 bytes per block), so very
# long contexts at the full bound can hold a few hundred MB.
MAX_SESSIONS = 4096


def session_id_of(request: Request) -> str | None:
    """The client-supplied session id, or None."""
    params = request.sampling_params
    if params is None or not params.extra_args:
        return None
    session_id = params.extra_args.get(SESSION_ID_KEY)
    return None if session_id is None else str(session_id)


def common_prefix_blocks(a: Sequence[BlockHash], b: Sequence[BlockHash]) -> int:
    """Leading blocks two prompts share.

    Block hashes are cumulative (each covers its whole prefix), so equal
    hashes at position i imply equal prefixes: the shared positions form a
    prefix, found by binary search.
    """
    lo, hi = 0, min(len(a), len(b))
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if a[mid - 1] == b[mid - 1]:
            lo = mid
        else:
            hi = mid - 1
    return lo


class SessionCreditTracker:
    """Per-session latest prompt and per-request credit, in blocks."""

    def __init__(self, max_sessions: int = MAX_SESSIONS) -> None:
        self._max_sessions = max_sessions
        # session id -> prompt block hashes of its latest arrived turn
        self._latest: OrderedDict[str, Sequence[BlockHash]] = OrderedDict()
        # request id -> credited blocks (only non-zero credits are stored)
        self._credits: dict[str, int] = {}

    def __len__(self) -> int:
        return len(self._latest)

    def on_arrival(
        self,
        request_id: str,
        session_id: str,
        prompt_path: Sequence[BlockHash],
    ) -> int:
        """Record a new turn; return its credit.

        ``prompt_path`` must hold the prompt's full blocks only (the waiting
        index path at arrival), so generated blocks never count.
        """
        previous = self._latest.pop(session_id, None)
        credit = 0 if previous is None else common_prefix_blocks(previous, prompt_path)
        self._latest[session_id] = prompt_path
        if len(self._latest) > self._max_sessions:
            self._latest.popitem(last=False)
        if credit:
            self._credits[request_id] = credit
        return credit

    def credit(self, request_id: str) -> int:
        return self._credits.get(request_id, 0)

    def drop(self, request_id: str) -> int:
        """Forget a request's credit (admitted or finished); return it."""
        return self._credits.pop(request_id, 0)

    def clear_sessions(self) -> None:
        """Forget every session's latest prompt (prefix-cache reset).

        Credits of requests already waiting were fixed at their arrival and
        are kept.
        """
        self._latest.clear()
