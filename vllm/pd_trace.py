# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in PD execution traces. No torch import or I/O on the disabled path.

PD_E2E_TRACE_DIR enables one JSONL file per process. PD_E2E_TRACE_RUN_ID must
be identical across client/router/engines. GPU events are queried by the writer;
tracing never synchronizes a CUDA stream. Missing/failed records invalidate a
trace instead of being silently converted into zero stage time.

PD_E2E_TRACE_DETAIL defaults to "timeline". Set it to "full" to also
split model execution at TP collectives and record all connector hook scopes.
"""

import atexit
from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
import queue
import socket
import threading
import time
import uuid
import weakref


def clock_sample():
    before = time.perf_counter_ns()
    wall = time.time_ns()
    after = time.perf_counter_ns()
    return {"monotonic_before_ns": before, "wall_ns": wall,
            "monotonic_after_ns": after}


@dataclass
class GpuSpan:
    record: dict
    start: object
    end: object
    stream: object
    anchor: object
    device: int


class Trace:
    def __init__(self, directory, run_id, *, capacity=65536, detail=None):
        if not run_id:
            raise ValueError("PD_E2E_TRACE_RUN_ID is required with PD_E2E_TRACE_DIR")
        self.detail = detail if detail is not None else os.environ.get(
            "PD_E2E_TRACE_DETAIL", "timeline")
        if self.detail not in ("timeline", "full"):
            raise ValueError("PD_E2E_TRACE_DETAIL must be timeline or full")
        self.full_detail = self.detail == "full"
        self.process_id = uuid.uuid4().hex
        self.common = {"schema_version": 1, "run_id": run_id,
                       "trace_detail": self.detail,
                       "process_id": self.process_id, "pid": os.getpid(),
                       "host": socket.gethostname()}
        self._queue = queue.Queue(maxsize=capacity)
        self._lock = threading.RLock()
        self._anchors = {}
        self._streams = {}
        self._open_by_stream = {}
        self._event_ids = weakref.WeakKeyDictionary()
        self._open_gpu_spans = set()
        self._last_gpu_calibration = {}
        self._sequence = 0
        self._dropped = 0
        self._errors = []
        self._closed = False
        self._stop = threading.Event()
        self._capacity = capacity
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"{self.common['host']}-{os.getpid()}-{self.process_id}.jsonl"
        # Fail at initialization for unwritable output; never run a benchmark
        # thinking tracing is active when the writer could not open its file.
        self._file = self.path.open("x", encoding="utf-8")
        self._thread = threading.Thread(target=self._write_loop,
                                        name="pd-e2e-trace", daemon=True)
        self._thread.start()
        self.emit("process_start", clock=clock_sample())
        atexit.register(self.close)

    def _id(self):
        with self._lock:
            self._sequence += 1
            return f"{self.process_id}:{self._sequence}"

    def _submit(self, record):
        if self._closed:
            self._dropped += 1
            return
        try:
            self._queue.put_nowait(record)
        except queue.Full:
            self._dropped += 1

    def emit(self, event, *, at_ns=None, **fields):
        event_id = self._id()
        self._submit({**self.common, "event_id": event_id, "event": event,
                      "monotonic_ns": time.perf_counter_ns() if at_ns is None else at_ns,
                      "thread_id": threading.get_ident(), **fields})
        return event_id

    @contextmanager
    def host_span(self, name, **fields):
        start = time.perf_counter_ns()
        status = "ok"
        try:
            yield
        except BaseException:
            status = "failed"
            raise
        finally:
            self.emit("host_span", name=name, start_ns=start,
                      end_ns=time.perf_counter_ns(), status=status, **fields)

    def begin_gpu(self, stage, *, stream=None, dependencies=(), **fields):
        import torch

        stream = stream or torch.cuda.current_stream()
        device = stream.device.index
        with torch.cuda.device(device), self._lock:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("PD E2E tracing requires eager execution (no CUDA graphs)")
            anchor = self._anchors.get(device)
            if anchor is None:
                anchor = torch.cuda.Event(enable_timing=True)
                lower = time.perf_counter_ns()
                anchor.record(stream)
                anchor_id = self.emit("gpu_anchor_submitted", device=device,
                                      lower_bound_ns=lower, clock=clock_sample())
                self._anchors[device] = (anchor, anchor_id, lower)
                self._submit(("anchor", device, anchor, anchor_id, lower))
            anchor, anchor_id, _ = self._anchors[device]
            # Refresh the CPU/GPU mapping without synchronizing. Keep explicit
            # submission/observation bounds instead of assuming enqueue=execute.
            if time.monotonic() - self._last_gpu_calibration.get(device, 0) > 1:
                calibration = torch.cuda.Event(enable_timing=True)
                lower = time.perf_counter_ns()
                calibration.record(stream)
                self._submit(("calibration", device, calibration, anchor_id, lower, anchor))
                self._last_gpu_calibration[device] = time.monotonic()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(stream)
            key = (device, int(stream.cuda_stream))
            previous = self._streams.get(key)
            event_id = self._id()
            self._open_gpu_spans.add(event_id)
            stack = self._open_by_stream.setdefault(key, [])
            parent = stack[-1] if stack else None
            stack.append(event_id)
            record = {**self.common, "event_id": event_id, "event": "gpu_span",
                      "stage": stage, "device": device, "stream_id": key[1],
                      "parent_event_id": parent,
                      "thread_id": threading.get_ident(),
                      "anchor_id": anchor_id, "host_submit_ns": time.perf_counter_ns(),
                      "dependencies": list(dict.fromkeys(
                          [*dependencies, *([previous] if previous else [])])),
                      "measurement": "cuda_event_stream_interval", **fields}
            return GpuSpan(record, start, end, stream, anchor, device)

    def end_gpu(self, span, *, status="ok"):
        import torch

        with torch.cuda.device(span.device), self._lock:
            span.end.record(span.stream)
            span.record["host_end_submit_ns"] = time.perf_counter_ns()
            span.record["status"] = status
            key = (span.device, int(span.stream.cuda_stream))
            stack = self._open_by_stream.get(key, [])
            if stack and stack[-1] == span.record["event_id"]:
                stack.pop()
            else:
                self._errors.append(f"non-nested GPU scope {span.record['event_id']}")
            self._streams[(span.device, int(span.stream.cuda_stream))] = span.record["event_id"]
            self._open_gpu_spans.discard(span.record["event_id"])
            if status != "ok":
                self._errors.append(f"failed GPU span {span.record['event_id']}")
        self._submit(span)
        return span.record["event_id"]

    @contextmanager
    def gpu_span(self, stage, **fields):
        span = self.begin_gpu(stage, **fields)
        status = "ok"
        try:
            yield span
        except BaseException:
            status = "failed"
            raise
        finally:
            self.end_gpu(span, status=status)

    def bind_event(self, cuda_event, trace_event_id):
        # The application owns readiness events; tracing must not retain them
        # for the lifetime of the engine or confuse recycled object IDs.
        with self._lock:
            if len(self._event_ids) >= self._capacity:
                self._dropped += 1
                self._event_ids.clear()
            self._event_ids[cuda_event] = trace_event_id

    def event_dependency(self, cuda_event):
        with self._lock:
            item = self._event_ids.get(cuda_event)
            return [item] if item is not None else []

    def release_event(self, cuda_event):
        with self._lock:
            self._event_ids.pop(cuda_event, None)

    def wait_event(self, stream, event, **fields):
        with self.gpu_span("other", name="cuda_wait", stream=stream,
                           attribution="wait_envelope",
                           dependencies=self.event_dependency(event), **fields):
            stream.wait_event(event)

    def _write_loop(self):
        pending = []
        last_clock = 0
        try:
            while not self._stop.is_set() or not self._queue.empty():
                for _ in range(1024):
                    try:
                        record = self._queue.get(timeout=0.002 if not pending else 0)
                    except queue.Empty:
                        break
                    if isinstance(record, dict):
                        self._write(record)
                    elif len(pending) < self._capacity:
                        pending.append(record)
                    else:
                        self._dropped += 1
                retained = []
                for item in pending:
                    try:
                        import torch
                        device = item.device if isinstance(item, GpuSpan) else item[1]
                        with torch.cuda.device(device):
                            event = item.end if isinstance(item, GpuSpan) else item[2]
                            if not event.query():
                                retained.append(item)
                                continue
                            if isinstance(item, GpuSpan):
                                if not item.anchor.query():
                                    retained.append(item)
                                    continue
                                self._write({**item.record,
                                    "gpu_start_ms": item.anchor.elapsed_time(item.start),
                                    "gpu_end_ms": item.anchor.elapsed_time(item.end),
                                    "completion_observed_ns": time.perf_counter_ns()})
                            elif item[0] == "anchor":
                                self._write({**self.common, "event": "gpu_anchor_observed",
                                    "anchor_id": item[3], "device": device,
                                    "lower_bound_ns": item[4],
                                    "upper_bound_ns": time.perf_counter_ns()})
                            elif item[5].query():
                                self._write({**self.common, "event": "gpu_clock_sample",
                                    "anchor_id": item[3], "device": device,
                                    "gpu_offset_ms": item[5].elapsed_time(item[2]),
                                    "lower_bound_ns": item[4],
                                    "upper_bound_ns": time.perf_counter_ns()})
                            else:
                                retained.append(item)
                    except Exception as exc:
                        self._errors.append(type(exc).__name__ + ": " + str(exc))
                pending = retained
                if time.monotonic() - last_clock > 1:
                    self._write({**self.common, "event": "clock_sample", **clock_sample()})
                    self._file.flush()
                    last_clock = time.monotonic()
                self._stop.wait(0.002)
            self._write({**self.common, "event": "trace_end",
                         "complete": not (pending or self._dropped or self._errors or self._open_gpu_spans),
                         "pending_gpu_events": len(pending), "dropped_records": self._dropped,
                         "unfinished_gpu_spans": len(self._open_gpu_spans),
                         "errors": self._errors})
        except Exception as exc:
            self._errors.append(str(exc))
            import sys
            print(f"PD_E2E_TRACE_FAILED {self.path}: {exc}", file=sys.stderr)
        finally:
            self._file.close()

    def _write(self, record):
        self._file.write(json.dumps(record, separators=(",", ":")) + "\n")

    def close(self):
        if not self._closed:
            self._closed = True
            self._stop.set()
            self._thread.join(timeout=5)


_trace = None
_init_lock = threading.Lock()


def get_trace():
    global _trace
    directory = os.environ.get("PD_E2E_TRACE_DIR")
    if not directory:
        return None
    with _init_lock:
        if _trace is None or _trace.common["pid"] != os.getpid():
            _trace = Trace(directory, os.environ.get("PD_E2E_TRACE_RUN_ID", ""))
    return _trace


def shutdown_trace():
    if _trace is not None and _trace.common["pid"] == os.getpid():
        _trace.close()


_model = threading.local()


def begin_forward(connector, scheduler_output):
    trace = get_trace()
    if trace is None or not scheduler_output.num_scheduled_tokens:
        return
    from vllm.distributed.parallel_state import get_tp_group
    group = get_tp_group()
    config = getattr(connector, "config", None)
    vllm_config = connector._vllm_config
    if (not vllm_config.model_config.enforce_eager
            or vllm_config.scheduler_config.async_scheduling):
        raise ValueError("PD E2E tracing currently requires enforce_eager and synchronous scheduling")
    if not (getattr(connector, "_timing_enabled", False)
            or getattr(config, "enable_timing", False)):
        raise ValueError("PD E2E tracing requires connector enable_timing=true for batch correlation")
    producer = getattr(connector, "is_producer", False) or getattr(config, "role", "") == "producer"
    meta = scheduler_output.kv_connector_metadata
    fields = {"request_ids": list(scheduler_output.num_scheduled_tokens),
              "batch_id": getattr(meta, "batch_id", None),
              "tp_rank": group.rank_in_group, "tp_size": group.world_size,
              "role": "producer" if producer else "consumer"}
    _model.fields = fields
    # Connectors explicitly declare hooks that do no work. Unknown connectors
    # keep both boundaries; never infer that a consumer hook is a no-op.
    noop_hooks = getattr(connector, "pd_trace_noop_hooks", None)
    _model.noop_hooks = frozenset(noop_hooks()) if noop_hooks else frozenset()
    _model.tp_group = group
    _model.collective_sequence = 0
    _model.stage = "prefill" if producer else "decode"
    _model.span = trace.begin_gpu(_model.stage, name="model_slice", **fields)
    trace.emit("forward_begin", **fields)


@contextmanager
def model_hook(name, layer_name=None):
    trace = get_trace()
    active = getattr(_model, "span", None)
    if trace is None or active is None:
        yield
        return
    if not trace.full_detail and name in _model.noop_hooks:
        yield
        return
    active.record.update(ends_before=name, layer_name=layer_name)
    trace.end_gpu(active)
    _model.span = None
    # A hook may itself contain nested codec/copy/communication intervals.
    # Its envelope is a dependency boundary, never an additive stage duration.
    try:
        if trace.full_detail:
            with trace.gpu_span("other", name=name, layer_name=layer_name,
                                attribution="envelope_only", **_model.fields):
                with trace.host_span(name, layer_name=layer_name, **_model.fields):
                    yield
        else:
            # Keep the real hook's host interval (including PUT blocking time)
            # and the model boundary, without another pair of CUDA events.
            with trace.host_span(name, layer_name=layer_name, **_model.fields):
                yield
    finally:
        _model.span = trace.begin_gpu(_model.stage, name="model_slice", **_model.fields)


def end_forward():
    trace = get_trace()
    active = getattr(_model, "span", None)
    if trace is not None and active is not None:
        end_id = trace.end_gpu(active)
        trace.emit("forward_end_submitted", dependencies=[end_id], **_model.fields)
        _model.span = None


@contextmanager
def collective(group, operation):
    """Expose TP joins without labeling model collectives as KV transfer."""
    trace = get_trace()
    active = getattr(_model, "span", None)
    if (trace is None or not trace.full_detail or active is None
            or group is not _model.tp_group):
        yield
        return
    active.record["ends_before"] = operation
    trace.end_gpu(active)
    _model.span = None
    sequence = getattr(_model, "collective_sequence", 0)
    _model.collective_sequence = sequence + 1
    collective_id = f"{_model.fields['role']}:{_model.fields['batch_id']}:{group.unique_name}:{sequence}"
    try:
        with trace.gpu_span(_model.stage, name=operation,
                            attribution="collective_envelope", collective_id=collective_id,
                            collective_members=list(group.ranks), **_model.fields):
            yield
    finally:
        _model.span = trace.begin_gpu(_model.stage, name="model_slice", **_model.fields)
