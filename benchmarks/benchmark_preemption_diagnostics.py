#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Synthetic CPU harness for diagnostic overhead, not a serving benchmark."""

import argparse
import json
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace as NS

from vllm.v1.core.sched.preemption_diagnostics import PreemptionDiagnostics


def run(steps, pressure, enabled, directory):
    diagnostics = PreemptionDiagnostics(directory, {"synthetic_microbenchmark": True}) if enabled else None
    requests = [NS(request_id=str(i), resumable=False, sampling_params=True,
                   has_encoder_inputs=False, num_computed_tokens=16,
                   num_preemptions=0, num_prompt_tokens=16, num_output_tokens=1)
                for i in range(32)]
    ownership = {r.request_id: [NS(block_id=i*8+j, ref_cnt=1, is_null=False) for j in range(8)]
                 for i, r in enumerate(requests)}
    scheduler = NS(policy=NS(value="fcfs"), running=requests, waiting=[],
                   kv_cache_manager=NS(get_blocks=lambda rid: NS(blocks=(ownership[rid],))))
    if diagnostics:
        for request in requests:
            diagnostics.add_request(request)
    start = time.perf_counter_ns()
    for _ in range(steps):
        if diagnostics:
            diagnostics.begin_step()
        if pressure:
            if diagnostics:
                diagnostics.allocation_failed("0", 12, 0)
                diagnostics.start_episode(scheduler, requests[0], 1, {})
            for request in requests[-1:-3:-1]:
                request.num_preemptions += 1
                if diagnostics:
                    diagnostics.preempt(request, 0, 8, "memory_pressure")
            if diagnostics:
                diagnostics.end_episode(True)
        if diagnostics:
            diagnostics.end_step()
    elapsed = time.perf_counter_ns()-start
    costs = dict(diagnostics.cost) if diagnostics else {}
    if diagnostics:
        diagnostics.shutdown()
    return dict(enabled=enabled, pressure=pressure, steps=steps,
                elapsed_ns=elapsed, ns_per_step=elapsed/steps, overhead_ns=costs,
                candidate_scan_ns=costs.get("candidate_scan_ns", 0),
                candidate_blocks_scanned=(diagnostics.totals["candidate_blocks_scanned"]
                                          if diagnostics else 0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    with tempfile.TemporaryDirectory(prefix="preemption-microbenchmark-") as directory:
        rows = [run(args.steps, pressure, enabled, directory)
                for pressure in (False, True) for enabled in (False, True)]
    result = json.dumps(dict(scope="synthetic CPU diagnostic harness", measurements=rows), indent=2)
    if args.output:
        args.output.write_text(result+"\n")
    print(result)


if __name__ == "__main__":
    main()
