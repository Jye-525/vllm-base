# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded ownership of post-forward payloads, independent of CUDA transport."""

import threading
from collections import deque
from collections.abc import Callable
from typing import Any


def parse_send_timing(config: Any) -> str:
    timing = config.get_from_extra_config("send_timing", "per_layer")
    if timing not in ("per_layer", "post_prefill"):
        raise ValueError("send_timing must be per_layer or post_prefill")
    if timing == "post_prefill":
        if config.get_from_extra_config("send_type", "PUT_ASYNC") not in (
            "PUT", "PUT_ASYNC"
        ):
            raise ValueError("post_prefill requires PUT or PUT_ASYNC")
    return timing


class BoundedSender:
    """Reserve before preparing payloads; release only after transport completes.

    prepare runs on the submitting/model thread. send runs on a background
    thread and must not return until the payload is safe to release.
    """

    def __init__(self, capacity: int, send: Callable[[Any], bool]) -> None:
        if capacity <= 0:
            raise ValueError("post_prefill_buffer_size must be positive")
        self.capacity = capacity
        self.pending_bytes = 0
        self._send = send
        self._cv = threading.Condition()
        self._queue: deque[tuple[int, Any]] = deque()
        self._error: BaseException | None = None
        self._closed = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def check(self) -> None:
        with self._cv:
            if self._error is not None:
                raise RuntimeError("Post-prefill PUT_ASYNC failed") from self._error

    def submit(self, size: int, prepare: Callable[[], Any]) -> None:
        if size <= 0 or size > self.capacity:
            raise ValueError(
                f"Payload size {size} exceeds post_prefill_buffer_size "
                f"{self.capacity}, or is empty"
            )
        with self._cv:
            self.check()
            if self._closed:
                raise RuntimeError("Post-prefill sender is closed")
            while self.pending_bytes + size > self.capacity:
                self._cv.wait()
                self.check()
                if self._closed:
                    raise RuntimeError("Post-prefill sender is closed")
            self.pending_bytes += size
        try:
            item = prepare()
        except BaseException:
            with self._cv:
                self.pending_bytes -= size
                self._cv.notify_all()
            raise
        with self._cv:
            # Keep ownership even if an earlier send failed during prepare.
            # GPU work in prepare may still be using this payload.
            self._queue.append((size, item))
            self._cv.notify_all()
            self.check()

    def drain(self) -> None:
        with self._cv:
            self.check()
            while self.pending_bytes:
                self._cv.wait()
                self.check()

    def close(self) -> None:
        with self._cv:
            self._closed = True
            self._cv.notify_all()
        self.drain()
        with self._cv:
            self._cv.notify_all()
        self._thread.join()

    def _run(self) -> None:
        while True:
            with self._cv:
                while not self._queue:
                    if self._closed and not self.pending_bytes:
                        return
                    self._cv.wait()
                size, item = self._queue.popleft()
            try:
                if not self._send(item):
                    raise RuntimeError("Peer rejected post-prefill transfer")
            except BaseException as exc:
                with self._cv:
                    # Retain failed/in-flight payloads on fatal errors: an
                    # exception may leave GPU communication still outstanding.
                    self._queue.appendleft((size, item))
                    self._error = exc
                    self._cv.notify_all()
                return
            del item
            with self._cv:
                self.pending_bytes -= size
                self._cv.notify_all()
