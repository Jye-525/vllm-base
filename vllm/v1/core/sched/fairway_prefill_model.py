# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefill-time model for Fairway's patience, read from a forward-sim JSON.

Fairway needs ``est(len)``, the time to prefill a prompt alone on an idle
engine. The KVReuse simulator prices it with forward-sim's fitted step-time
model (``simulator/forward-sim/models/<config>.json``); this module reads
the same file and prices one request's step exactly as
``forward_sim.StepModel.time_of_totals`` does, so vLLM and the simulator
compute the same estimate:

    T = t0 + sum_b s_b [N_d > b] + tN N_d + tC cached_d + tP new + tA attn

with ``attn = new * cached + new^2 / 2``. As in the simulator's forward-sim
adapter (``fwdsim_timer.py``), a one-token step is a decode (``N_d = 1``,
``cached_d = start``) and a longer one a prefill chunk.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any

TERMS = ("t0", "tN", "tC", "tP", "tA")


class ForwardSimChunkTime:
    """``chunk_time(start, end)`` in seconds of a one-request step."""

    def __init__(self, model: Mapping[str, Any], source: str = "<dict>") -> None:
        units = model.get("units", "seconds")
        if units != "seconds":
            raise ValueError(
                f"{source}: forward-sim units must be seconds, got {units!r}"
            )
        coef = model.get("coef")
        if not isinstance(coef, Mapping):
            raise ValueError(f"{source}: forward-sim model has no 'coef' mapping")
        tiles = [int(b) for b in model.get("tiles", [])]
        missing = [n for n in (*TERMS, *(f"s{b}" for b in tiles)) if n not in coef]
        if missing:
            raise ValueError(
                f"{source}: forward-sim model lacks coefficients {missing}"
            )
        self.source = source
        self.c = {n: float(coef[n]) for n in TERMS}
        self.tile_steps = [(b, float(coef[f"s{b}"])) for b in tiles]

    def _time_of_totals(
        self, nd: int, dec_cached: int, pf_new: int, pf_attn: float
    ) -> float:
        # Same expression and order as forward_sim.StepModel.time_of_totals.
        c = self.c
        return (
            c["t0"]
            + sum(s for b, s in self.tile_steps if nd > b)
            + c["tN"] * nd
            + c["tC"] * dec_cached
            + c["tP"] * pf_new
            + c["tA"] * pf_attn
        )

    def __call__(self, start: int, end: int) -> float:
        new = end - start
        if new < 1 or start < 0:
            raise ValueError(f"chunk [{start}, {end}) needs start >= 0 and end > start")
        if new == 1:
            return self._time_of_totals(1, start, 0, 0.0)
        return self._time_of_totals(0, 0, new, new * start + new * new / 2)


def load_chunk_time(path: str) -> Callable[[int, int], float]:
    """Read a forward-sim model JSON file and return its ``chunk_time``."""
    with open(path) as f:
        model = json.load(f)
    return ForwardSimChunkTime(model, source=path)
