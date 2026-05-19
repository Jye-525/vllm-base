# SPDX-License-Identifier: Apache-2.0
"""JSONL logger for per-request prompt and output token ids.

Logs one record per finished request for offline KV cache reuse analysis.
Each record contains the full prompt and output token id sequences along
with timing and prefix-cache stats.

Enable by setting the environment variable:
    VLLM_REQUEST_LOG=/path/to/requests.jsonl

When the env var is unset (default), all logging calls are no-ops.
"""

import atexit
import json
import os
import threading
import time
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from vllm.v1.request import Request


class RequestEventLogger:
    """Thread-safe JSONL logger for finished-request token id records."""

    _instance: Optional["RequestEventLogger"] = None
    _lock = threading.Lock()

    def __init__(self, path: str):
        self._file = open(path, "w", buffering=1)
        self._write_lock = threading.Lock()
        atexit.register(self._close)

    @classmethod
    def get(cls) -> Optional["RequestEventLogger"]:
        """Return the singleton logger, or None if logging is disabled."""
        if cls._instance is not None:
            return cls._instance
        path = os.environ.get("VLLM_REQUEST_LOG")
        if not path:
            return None
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(path)
        return cls._instance

    def log_request(self, request: "Request") -> None:
        finish_reason = request.get_finished_reason()
        record = {
            "request_id": request.request_id,
            "client_index": request.client_index,
            "arrival_time": request.arrival_time,
            "finish_time": time.time(),
            "finish_reason": (
                finish_reason.value
                if finish_reason is not None
                else None
            ),
            "status": str(request.status),
            "num_prompt_tokens": request.num_prompt_tokens,
            "num_output_tokens": request.num_output_tokens,
            "num_cached_tokens": request.num_cached_tokens,
            "num_external_computed_tokens":
                request.num_external_computed_tokens,
            "num_preemptions": request.num_preemptions,
            "prompt_token_ids":
                list(request.prompt_token_ids)
                if request.prompt_token_ids is not None
                else None,
            "output_token_ids": list(request.output_token_ids),
        }
        line = json.dumps(record, separators=(",", ":"))
        with self._write_lock:
            self._file.write(line)
            self._file.write("\n")

    def flush(self) -> None:
        with self._write_lock:
            self._file.flush()

    def _close(self) -> None:
        with self._write_lock:
            self._file.flush()
            self._file.close()
