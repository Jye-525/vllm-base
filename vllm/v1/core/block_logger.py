# SPDX-License-Identifier: Apache-2.0
"""CSV logger for prefix caching block events (Experiment 1 instrumentation).

Logs block creation and cache hit events to a CSV file for offline
prefix-tree analysis.

Enable by setting the environment variable:
    VLLM_BLOCK_LOG=/path/to/block_events.csv

When the env var is unset (default), all logging calls are no-ops.
"""

import atexit
import csv
import os
import threading
import time
from typing import Optional


class BlockEventLogger:
    """Thread-safe CSV logger for block hash events."""

    _instance: Optional["BlockEventLogger"] = None
    _lock = threading.Lock()

    def __init__(self, path: str):
        self._file = open(path, "w", newline="", buffering=1)
        self._writer = csv.writer(self._file)
        self._writer.writerow([
            "event",
            "block_hash",
            "parent_hash",
            "request_id",
            "timestamp",
            "block_index",
            "num_tokens",
            "phase",
        ])
        self._write_lock = threading.Lock()
        atexit.register(self._close)

    @classmethod
    def get(cls) -> Optional["BlockEventLogger"]:
        """Return the singleton logger, or None if logging is disabled."""
        if cls._instance is not None:
            return cls._instance
        path = os.environ.get("VLLM_BLOCK_LOG")
        if not path:
            return None
        with cls._lock:
            # Double-check after acquiring lock.
            if cls._instance is None:
                cls._instance = cls(path)
        return cls._instance

    def log_block(
        self,
        block_hash: bytes,
        parent_hash: Optional[bytes],
        request_id: str,
        block_index: int,
        num_tokens: int,
        phase: Optional[str] = None,
    ) -> None:
        row = [
            "block",
            block_hash.hex(),
            parent_hash.hex() if parent_hash else "",
            request_id,
            f"{time.time():.6f}",
            block_index,
            num_tokens,
            phase if phase is not None else "",
        ]
        with self._write_lock:
            self._writer.writerow(row)

    def log_cache_hit(
        self,
        block_hash: bytes,
        request_id: str,
    ) -> None:
        row = [
            "cache_hit",
            block_hash.hex(),
            "",
            request_id,
            f"{time.time():.6f}",
            "",
            "",
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
