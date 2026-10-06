# SPDX-License-Identifier: Apache-2.0
"""In-process orphan counter for Experiment 4: orphan accumulation measurement.

Maintains an incremental orphan count inside the offloading manager and
periodically appends a single summary row to a CSV file.  This replaces the
heavy snapshot-based OrphanProbe (which copied the entire block dict and wrote
thousands of rows per tick) with an O(1)-per-eviction design that writes one
row every SNAPSHOT_INTERVAL_SEC seconds.

Enable by setting VLLM_ORPHAN_LOG=/path/to/orphan_log.csv
"""

import csv
import os
import threading
import time
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from vllm.v1.kv_offload.cpu.instrumented_manager import (
        InstrumentedCPUOffloadingManager,
    )

SNAPSHOT_INTERVAL_SEC = 10


class OrphanCounter:
    """Background thread that periodically logs orphan counters to a CSV.

    The actual orphan counting is done incrementally inside the offloading
    manager (_orphaned set, updated at eviction time).  This thread only
    reads two integers from the manager and appends one CSV row per tick,
    making it essentially free compared to the snapshot approach.

    Accuracy note: counts a block as orphaned only when its *direct* CPU
    parent is evicted.  Grandchildren become orphaned in subsequent ticks
    as their (now-orphaned) parents are eventually evicted.  This slightly
    underestimates transient orphan counts but is accurate at steady state.
    """

    def __init__(self, log_path: str, manager: "InstrumentedCPUOffloadingManager"):
        """
        Args:
            log_path: Path to the output CSV file (appended, not overwritten).
            manager: The CPU offloading manager.  Exposes _orphaned (set)
                and num_cpu_blocks_in_use() for any cache policy.
        """
        self._log_path = log_path
        self._manager = manager
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="orphan-counter")
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        # Write header row
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow([
                "snapshot_id", "timestamp",
                "total_cpu_blocks", "orphan_count",
            ])

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()

    def _run(self) -> None:
        snapshot_id = 0
        while not self._stop.wait(timeout=SNAPSHOT_INTERVAL_SEC):
            try:
                self._log(snapshot_id)
            except Exception:
                pass  # never crash the main process
            snapshot_id += 1

    def _log(self, snapshot_id: int) -> None:
        ts = time.time()
        mgr = self._manager

        orphan_count = len(mgr._orphaned)
        total_cpu = mgr.num_cpu_blocks_in_use()

        with open(self._log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                snapshot_id, f"{ts:.6f}", total_cpu, orphan_count,
            ])


def maybe_start_orphan_counter(
    manager: "InstrumentedCPUOffloadingManager",
) -> Optional[OrphanCounter]:
    """Start an OrphanCounter if VLLM_ORPHAN_LOG env var is set.

    Returns the counter instance (already started) or None if disabled.
    Call counter.stop() on shutdown if the returned value is not None.
    """
    log_path = os.environ.get("VLLM_ORPHAN_LOG")
    if not log_path:
        return None
    counter = OrphanCounter(log_path=log_path, manager=manager)
    counter.start()
    return counter
