# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Low-overhead timing support for :class:`P2pNcclConnector`.

The objects in this module are instantiated only when P2P timing is enabled.
They deliberately do not synchronize CUDA. Callers resolve events after an
existing model-output or transport-stream synchronization.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

TimingDetail = Literal["summary", "segments"]
TimingSide = Literal["send", "recv"]

_CLIENT_REQUEST_ID_RE = re.compile(
    r"_([0-9a-f]{32})(?:-\d+)?(?:-[0-9a-f]+)?$"
)


def parse_timing_options(
    config: Any, *, async_scheduling: bool = False
) -> tuple[bool, TimingDetail]:
    """Parse and validate the connector's opt-in timing options."""

    enabled = bool(config.get_from_extra_config("enable_timing", False))
    detail = str(
        config.get_from_extra_config("timing_detail", "summary")
    ).lower()
    if detail not in ("summary", "segments"):
        raise ValueError(
            "kv_connector_extra_config.timing_detail must be "
            "'summary' or 'segments'"
        )
    send_type = config.get_from_extra_config("send_type", "PUT_ASYNC")
    if enabled and send_type == "GET":
        raise ValueError("P2pNcclConnector timing does not support send_type=GET")
    if enabled and async_scheduling:
        raise ValueError(
            "P2pNcclConnector timing requires asynchronous scheduling "
            "to be disabled"
        )
    return enabled, detail  # type: ignore[return-value]


def extract_client_request_id(request_id: str | None) -> str:
    """Extract the router UUID from a vLLM request ID when it is present."""

    if not request_id:
        return ""
    match = _CLIENT_REQUEST_ID_RE.search(request_id)
    return match.group(1) if match else request_id


def request_id_from_tensor_id(tensor_id: str) -> str:
    return tensor_id.split("#", 1)[0]


def layer_name_from_tensor_id(tensor_id: str) -> str:
    parts = tensor_id.split("#", 1)
    return parts[1] if len(parts) == 2 else ""


def nonnegative_duration_ms(start_ns: int, end_ns: int) -> tuple[float, bool]:
    """Return a clamped duration and whether the source clock went backwards."""

    delta = end_ns - start_ns
    return max(0.0, delta / 1e6), delta < 0


@dataclass
class CudaEventSpan:
    start: Any
    end: Any

    def ready(self) -> bool:
        for event in (self.start, self.end):
            query = getattr(event, "query", None)
            if query is not None and not bool(query()):
                return False
        return True

    def elapsed_ms(self) -> float:
        return max(0.0, float(self.start.elapsed_time(self.end)))

    def anchor_offsets_ms(self, anchor: Any) -> tuple[float, float]:
        return (
            float(anchor.elapsed_time(self.start)),
            float(anchor.elapsed_time(self.end)),
        )


@dataclass
class TensorTiming:
    tensor_id: str
    request_id: str
    layer_name: str
    payload_bytes: int
    batch_id: str | None
    queued_perf_ns: int
    queued_wall_ns: int
    control_start_perf_ns: int | None = None
    control_end_perf_ns: int | None = None
    control_start_wall_ns: int | None = None
    control_end_wall_ns: int | None = None
    allocation_start_perf_ns: int | None = None
    allocation_end_perf_ns: int | None = None
    allocation_start_wall_ns: int | None = None
    allocation_end_wall_ns: int | None = None
    nccl_start_perf_ns: int | None = None
    nccl_end_perf_ns: int | None = None
    nccl_start_wall_ns: int | None = None
    nccl_end_wall_ns: int | None = None
    completed_perf_ns: int | None = None
    completed_wall_ns: int | None = None
    nccl_span: CudaEventSpan | None = None
    status: str = "pending"
    error: str | None = None


@dataclass
class RequestTiming:
    request_id: str
    side: TimingSide
    batch_id: str | None = None
    enqueued_count: int = 0
    completed_count: int = 0
    sealed: bool = False
    emitted: bool = False
    tensors: list[TensorTiming] = field(default_factory=list)
    extraction_spans: list[CudaEventSpan] = field(default_factory=list)
    insertion_spans: list[CudaEventSpan] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class P2pRequestTimingTracker:
    """Thread-safe request completion and timing publication tracker."""

    def __init__(
        self,
        *,
        side: TimingSide,
        detail: TimingDetail,
        anchor: Any,
        common: dict[str, Any],
        emit: Callable[[dict[str, Any]], None],
    ) -> None:
        self.side = side
        self.detail = detail
        self.anchor = anchor
        self.common = dict(common)
        self._emit = emit
        self._lock = threading.Lock()
        self._requests: dict[str, RequestTiming] = {}
        self._tensors: dict[str, TensorTiming] = {}

    def bind_request(self, request_id: str, batch_id: str | None) -> None:
        with self._lock:
            state = self._state_locked(request_id)
            if state.batch_id is None:
                state.batch_id = batch_id
            elif batch_id is not None and state.batch_id != batch_id:
                state.warnings.append(
                    "request observed in more than one timing batch; "
                    f"keeping {state.batch_id!r} and ignoring {batch_id!r}"
                )
            for tensor in state.tensors:
                if tensor.batch_id is None:
                    tensor.batch_id = state.batch_id

    def enqueue(
        self,
        *,
        tensor_id: str,
        payload_bytes: int,
        batch_id: str | None,
        queued_perf_ns: int,
        queued_wall_ns: int,
    ) -> None:
        request_id = request_id_from_tensor_id(tensor_id)
        with self._lock:
            state = self._state_locked(request_id)
            if state.batch_id is None:
                state.batch_id = batch_id
            if tensor_id in self._tensors:
                state.warnings.append(
                    f"duplicate tensor timing enqueue ignored: {tensor_id}"
                )
                return
            tensor = TensorTiming(
                tensor_id=tensor_id,
                request_id=request_id,
                layer_name=layer_name_from_tensor_id(tensor_id),
                payload_bytes=max(0, int(payload_bytes)),
                batch_id=state.batch_id or batch_id,
                queued_perf_ns=queued_perf_ns,
                queued_wall_ns=queued_wall_ns,
            )
            state.tensors.append(tensor)
            state.enqueued_count += 1
            self._tensors[tensor_id] = tensor

    def update_tensor(self, tensor_id: str, **values: Any) -> None:
        with self._lock:
            tensor = self._tensors.get(tensor_id)
            if tensor is None:
                return
            for key, value in values.items():
                if hasattr(tensor, key):
                    setattr(tensor, key, value)

    def complete_tensor(
        self,
        tensor_id: str,
        *,
        completed_perf_ns: int,
        completed_wall_ns: int,
        status: str,
        error: str | None = None,
    ) -> None:
        request_id = request_id_from_tensor_id(tensor_id)
        with self._lock:
            tensor = self._tensors.get(tensor_id)
            if tensor is None or tensor.completed_perf_ns is not None:
                return
            tensor.completed_perf_ns = completed_perf_ns
            tensor.completed_wall_ns = completed_wall_ns
            tensor.status = status
            tensor.error = error
            self._state_locked(request_id).completed_count += 1
        self.publish_ready(request_id)

    def add_stage_span(
        self,
        request_id: str,
        stage: Literal["extraction", "insertion"],
        span: CudaEventSpan,
    ) -> None:
        with self._lock:
            state = self._state_locked(request_id)
            spans = (
                state.extraction_spans
                if stage == "extraction"
                else state.insertion_spans
            )
            spans.append(span)

    def seal_request(self, request_id: str, batch_id: str | None) -> None:
        with self._lock:
            state = self._state_locked(request_id)
            if state.batch_id is None:
                state.batch_id = batch_id
            state.sealed = True
        self.publish_ready(request_id)

    def fail_request(self, request_id: str, message: str) -> None:
        with self._lock:
            state = self._state_locked(request_id)
            state.warnings.append(message)
            state.sealed = True
            for tensor in state.tensors:
                if tensor.status == "pending":
                    tensor.status = "failed"
                    tensor.error = message
        self.publish_ready(request_id, allow_incomplete_failure=True)

    def discard_request(self, request_id: str) -> None:
        with self._lock:
            state = self._requests.pop(request_id, None)
            if state is not None:
                for tensor in state.tensors:
                    self._tensors.pop(tensor.tensor_id, None)

    def publish_ready(
        self,
        request_id: str | None = None,
        *,
        allow_incomplete_failure: bool = False,
    ) -> None:
        publications: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
        with self._lock:
            states = (
                [self._requests[request_id]]
                if request_id is not None and request_id in self._requests
                else list(self._requests.values())
                if request_id is None
                else []
            )
            for state in states:
                publication = self._publication_locked(
                    state,
                    allow_incomplete_failure=allow_incomplete_failure,
                )
                if publication is not None:
                    state.emitted = True
                    publications.append(publication)
        for summary, segments in publications:
            try:
                self._emit(summary)
            except Exception:
                # Timing/logging must never fail inference or a transport
                # progress thread.
                continue
            if self.detail == "segments":
                for segment in segments:
                    try:
                        self._emit(segment)
                    except Exception:
                        pass

    def latest_nccl_end_offset_ms(
        self, request_ids: set[str]
    ) -> float | None:
        """Return a ready local-GPU completion offset for analyzer support."""

        values: list[float] = []
        with self._lock:
            spans = [
                tensor.nccl_span
                for request_id in request_ids
                for tensor in self._requests.get(
                    request_id, RequestTiming(request_id, self.side)
                ).tensors
                if tensor.nccl_span is not None
            ]
        for span in spans:
            assert span is not None
            if span.ready():
                values.append(span.anchor_offsets_ms(self.anchor)[1])
        return max(values) if values else None

    def _state_locked(self, request_id: str) -> RequestTiming:
        state = self._requests.get(request_id)
        if state is None:
            state = RequestTiming(request_id=request_id, side=self.side)
            self._requests[request_id] = state
        return state

    def _publication_locked(
        self,
        state: RequestTiming,
        *,
        allow_incomplete_failure: bool,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
        if state.emitted or not state.sealed or state.enqueued_count == 0:
            return None
        complete = state.completed_count == state.enqueued_count
        failed = any(tensor.status == "failed" for tensor in state.tensors)
        if not complete and not (allow_incomplete_failure and failed):
            return None
        spans = [
            *(tensor.nccl_span for tensor in state.tensors),
            *state.extraction_spans,
            *state.insertion_spans,
        ]
        if any(span is not None and not span.ready() for span in spans):
            return None
        return self._build_records_locked(state)

    def _build_records_locked(
        self, state: RequestTiming
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        tensors = list(state.tensors)
        completed = [
            tensor for tensor in tensors if tensor.completed_perf_ns is not None
        ]
        warnings = list(dict.fromkeys(state.warnings))
        clock_anomaly = False

        nccl_gpu_ms = 0.0
        nccl_offsets: list[tuple[float, float]] = []
        for tensor in tensors:
            if tensor.nccl_span is None:
                continue
            nccl_gpu_ms += tensor.nccl_span.elapsed_ms()
            nccl_offsets.append(tensor.nccl_span.anchor_offsets_ms(self.anchor))

        extraction_gpu_ms = sum(
            span.elapsed_ms() for span in state.extraction_spans
        )
        insertion_gpu_ms = sum(span.elapsed_ms() for span in state.insertion_spans)

        control_wall_ms = 0.0
        allocation_wall_ms = 0.0
        for tensor in tensors:
            if (
                tensor.control_start_perf_ns is not None
                and tensor.control_end_perf_ns is not None
            ):
                value, anomaly = nonnegative_duration_ms(
                    tensor.control_start_perf_ns, tensor.control_end_perf_ns
                )
                control_wall_ms += value
                clock_anomaly |= anomaly
            if (
                tensor.allocation_start_perf_ns is not None
                and tensor.allocation_end_perf_ns is not None
            ):
                value, anomaly = nonnegative_duration_ms(
                    tensor.allocation_start_perf_ns,
                    tensor.allocation_end_perf_ns,
                )
                allocation_wall_ms += value
                clock_anomaly |= anomaly
            wall_pairs = (
                (tensor.queued_wall_ns, tensor.completed_wall_ns),
                (tensor.control_start_wall_ns, tensor.control_end_wall_ns),
                (tensor.allocation_start_wall_ns, tensor.allocation_end_wall_ns),
                (tensor.nccl_start_wall_ns, tensor.nccl_end_wall_ns),
            )
            clock_anomaly |= any(
                start is not None and end is not None and end < start
                for start, end in wall_pairs
            )

        queued_perf_ns = min(tensor.queued_perf_ns for tensor in tensors)
        completed_perf_ns = max(
            tensor.completed_perf_ns or tensor.queued_perf_ns for tensor in tensors
        )
        envelope_ms, anomaly = nonnegative_duration_ms(
            queued_perf_ns, completed_perf_ns
        )
        clock_anomaly |= anomaly
        if clock_anomaly:
            warnings.append(
                "one or more monotonic clock intervals were negative and clamped"
            )

        statuses = {tensor.status for tensor in tensors}
        if "failed" in statuses:
            status = "failed"
        elif clock_anomaly:
            status = "clock_anomaly"
        elif len(completed) != len(tensors):
            status = "incomplete"
        else:
            status = "ok"

        request_common = {
            **self.common,
            "request_id": state.request_id,
            "client_request_id": extract_client_request_id(state.request_id),
            "batch_id": state.batch_id,
        }
        summary: dict[str, Any] = {
            **request_common,
            "record_type": "transfer_summary",
            "side": self.side,
            "measurement_scope": "request_rank",
            "payload_bytes": sum(tensor.payload_bytes for tensor in tensors),
            "tensor_count": len(tensors),
            "enqueued_tensor_count": state.enqueued_count,
            "completed_tensor_count": state.completed_count,
            "zmq_control_wall_ms": control_wall_ms,
            "allocation_wall_ms": allocation_wall_ms,
            "queue_to_completion_wall_ms": envelope_ms,
            "host_envelope_start_wall_ns": min(
                tensor.queued_wall_ns for tensor in tensors
            ),
            "host_envelope_end_wall_ns": max(
                tensor.completed_wall_ns or tensor.queued_wall_ns
                for tensor in tensors
            ),
            "extraction_gpu_ms": extraction_gpu_ms,
            "insertion_gpu_ms": insertion_gpu_ms,
            "status": status,
            "warnings": warnings,
            "measurement_semantics": {
                "forward": "model_stream_envelope_not_kernel_sum",
                "receiver": (
                    "posted_to_completion_includes_source_readiness_wait"
                ),
                "batch_compute": "shared_not_divided_across_requests",
            },
        }
        if self.side == "send":
            summary["nccl_send_gpu_ms"] = nccl_gpu_ms
            summary["send_queue_to_completion_wall_ms"] = envelope_ms
            summary["metric_relationships"] = [
                {
                    "send_type": self.common.get("send_type"),
                    "rule": (
                        "do_not_add_to_prefill"
                        if self.common.get("send_type") == "PUT"
                        else "may_overlap_prefill_use_batch_tail"
                    ),
                }
            ]
        else:
            summary["nccl_recv_posted_to_completion_gpu_ms"] = nccl_gpu_ms
            summary["recv_posted_to_completion_wall_ms"] = envelope_ms
            summary["metric_relationships"] = [
                {
                    "metric": "nccl_recv_posted_to_completion_gpu_ms",
                    "includes": "source_readiness_wait",
                    "rule": "not_pure_network_bandwidth",
                },
                {
                    "metric": "insertion_gpu_ms",
                    "precedes": "decode_forward_gpu_ms",
                },
            ]
        if nccl_offsets:
            summary["gpu_anchor_nccl_start_ms"] = min(
                value[0] for value in nccl_offsets
            )
            summary["gpu_anchor_nccl_end_ms"] = max(
                value[1] for value in nccl_offsets
            )

        segments = [
            self._build_segment_record(request_common, tensor)
            for tensor in tensors
        ]
        return summary, segments

    def _build_segment_record(
        self, request_common: dict[str, Any], tensor: TensorTiming
    ) -> dict[str, Any]:
        queue_end_perf_ns = tensor.control_start_perf_ns or tensor.nccl_start_perf_ns
        queue_wall_ms = _optional_duration_ms(
            tensor.queued_perf_ns, queue_end_perf_ns
        )
        control_wall_ms = _optional_duration_ms(
            tensor.control_start_perf_ns, tensor.control_end_perf_ns
        )
        allocation_wall_ms = _optional_duration_ms(
            tensor.allocation_start_perf_ns, tensor.allocation_end_perf_ns
        )
        nccl_host_wall_ms = _optional_duration_ms(
            tensor.nccl_start_perf_ns, tensor.nccl_end_perf_ns
        )
        completion_wall_ms = _optional_duration_ms(
            tensor.queued_perf_ns, tensor.completed_perf_ns
        )
        record: dict[str, Any] = {
            **request_common,
            "record_type": "transfer_segment",
            "side": self.side,
            "measurement_scope": "layer_tensor_rank",
            "transfer_id": tensor.tensor_id,
            "layer_name": tensor.layer_name,
            "payload_bytes": tensor.payload_bytes,
            "wall_start_ns": tensor.queued_wall_ns,
            "wall_end_ns": tensor.completed_wall_ns,
            "queue_wall_start_ns": tensor.queued_wall_ns,
            "control_wall_start_ns": tensor.control_start_wall_ns,
            "control_wall_end_ns": tensor.control_end_wall_ns,
            "allocation_wall_start_ns": tensor.allocation_start_wall_ns,
            "allocation_wall_end_ns": tensor.allocation_end_wall_ns,
            "nccl_wall_start_ns": tensor.nccl_start_wall_ns,
            "nccl_wall_end_ns": tensor.nccl_end_wall_ns,
            "completion_wall_ns": tensor.completed_wall_ns,
            "queue_wall_ms": queue_wall_ms,
            "zmq_control_wall_ms": control_wall_ms,
            "allocation_wall_ms": allocation_wall_ms,
            "nccl_host_wall_ms": nccl_host_wall_ms,
            "queue_to_completion_wall_ms": completion_wall_ms,
            "status": tensor.status,
            "error": tensor.error,
            "measurement_semantics": (
                "receiver_interval_includes_source_readiness_wait"
                if self.side == "recv"
                else "sender_nccl_stream_interval"
            ),
        }
        if tensor.nccl_span is not None:
            start_ms, end_ms = tensor.nccl_span.anchor_offsets_ms(self.anchor)
            record.update(
                {
                    "nccl_gpu_ms": tensor.nccl_span.elapsed_ms(),
                    "gpu_anchor_start_ms": start_ms,
                    "gpu_anchor_end_ms": end_ms,
                }
            )
        return record


def timing_log_record(logger: Any, record: dict[str, Any]) -> None:
    logger.info(
        "P2P_NCCL_TIMING_LOG %s",
        json.dumps(record, sort_keys=True, separators=(",", ":")),
    )


def _optional_duration_ms(
    start_ns: int | None, end_ns: int | None
) -> float | None:
    if start_ns is None or end_ns is None:
        return None
    return nonnegative_duration_ms(start_ns, end_ns)[0]
