# SPDX-License-Identifier: Apache-2.0
"""CSV logger for cross-tier block tracking (Experiment 2 instrumentation).

Logs GPU and CPU store/evict/hit events for offline coverage analysis.

Enable by setting:
    VLLM_TIER_LOG=/path/to/tier_events.csv

When the env var is unset (default), all logging calls are no-ops.
"""

import atexit
import csv
import os
import threading
import time
from typing import Optional


class TierEventLogger:
    """Thread-safe CSV logger for tier-tracking events."""

    _instance: Optional["TierEventLogger"] = None
    _lock = threading.Lock()

    def __init__(self, path: str):
        self._file = open(path, "w", newline="", buffering=1)
        self._writer = csv.writer(self._file)
        self._writer.writerow([
            "event", "block_hash", "tier", "timestamp",
            "parent_hash", "request_id",
        ])
        self._write_lock = threading.Lock()
        atexit.register(self._close)

    @classmethod
    def get(cls) -> Optional["TierEventLogger"]:
        """Return the singleton logger, or None if logging is disabled."""
        if cls._instance is not None:
            return cls._instance
        path = os.environ.get("VLLM_TIER_LOG")
        if not path:
            return None
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(path)
        return cls._instance

    def log_event(
        self,
        event: str,
        block_hash: bytes,
        tier: str,
        parent_hash: Optional[bytes] = None,
        request_id: Optional[str] = None,
    ) -> None:
        """
        Args:
            event: one of gpu_store, gpu_evict, cpu_store, cpu_evict, cpu_hit
            block_hash: raw bytes of the block hash
            tier: "gpu" or "cpu"
            parent_hash: raw bytes of the parent block hash (optional)
            request_id: ID of the request that caused this event (optional)
        """
        row = [
            event,
            block_hash.hex(),
            tier,
            f"{time.time():.6f}",
            parent_hash.hex() if parent_hash is not None else "",
            request_id if request_id is not None else "",
        ]
        with self._write_lock:
            self._writer.writerow(row)

    def flush(self) -> None:
        with self._write_lock:
            self._file.flush()

    def _close(self) -> None:
        with self._write_lock:
            self._file.flush()
            self._file.close()


def _get_tier_logger() -> Optional[TierEventLogger]:
    """Convenience accessor used at hook sites."""
    return TierEventLogger.get()
