# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import regex as re
import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_engine import (
    P2pNcclEngine,
)
from vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_timing import (
    CudaEventSpan,
    extract_client_request_id,
    nonnegative_duration_ms,
    parse_timing_options,
    timing_log_record,
)
from vllm.distributed.parallel_state import get_tp_group, get_world_group
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import MLACommonMetadata
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class ReqMeta:
    # Request Id
    request_id: str
    # Request block ids
    block_ids: torch.Tensor
    # Request num tokens
    num_tokens: int

    @staticmethod
    def make_meta(
        request_id: str, token_ids: list[int], block_ids: list[int], block_size: int
    ) -> "ReqMeta":
        block_ids_tensor = torch.tensor(block_ids)
        return ReqMeta(
            request_id=request_id,
            block_ids=block_ids_tensor,
            num_tokens=len(token_ids),
        )


@dataclass(frozen=True)
class ScheduledReqMeta:
    request_id: str
    token_count: int
    phase: str


@dataclass
class P2pNcclConnectorMetadata(KVConnectorMetadata):
    requests: list[ReqMeta]

    def __init__(self):
        self.requests = []

    def add_request(
        self,
        request_id: str,
        token_ids: list[int],
        block_ids: list[int],
        block_size: int,
    ) -> None:
        self.requests.append(
            ReqMeta.make_meta(request_id, token_ids, block_ids, block_size)
        )


@dataclass
class P2pNcclTimingConnectorMetadata(P2pNcclConnectorMetadata):
    """Timing-only scheduler metadata, absent from the disabled path."""

    timed_requests: list[ScheduledReqMeta]
    batch_request_ids: list[str]
    batch_id: str | None

    def __init__(self):
        super().__init__()
        self.timed_requests = []
        self.batch_request_ids = []
        self.batch_id = None


@dataclass
class _ActiveBatchTiming:
    batch_id: str
    requests: list[ScheduledReqMeta]
    batch_request_ids: list[str]
    forward_start: Any
    host_start_perf_ns: int
    host_start_wall_ns: int


@dataclass
class _PendingBatchTiming:
    batch_id: str
    requests: list[ScheduledReqMeta]
    batch_request_ids: list[str]
    forward_span: CudaEventSpan
    host_start_perf_ns: int
    host_end_perf_ns: int
    host_start_wall_ns: int
    host_end_wall_ns: int


class P2pNcclConnector(KVConnectorBase_V1):
    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )
        self._block_size = vllm_config.cache_config.block_size
        self._requests_need_load: dict[str, Any] = {}
        self.is_producer = self._kv_transfer_config.is_kv_producer
        self.chunked_prefill: dict[str, tuple[list[int], list[int] | None]] = {}

        self._timing_enabled, self._timing_detail = parse_timing_options(
            self._kv_transfer_config,
            async_scheduling=bool(vllm_config.scheduler_config.async_scheduling),
        )
        if self._timing_enabled:
            self._timing_batch_counter = 0

        self._rank = get_world_group().rank if role == KVConnectorRole.WORKER else 0
        self._local_rank = (
            get_world_group().local_rank if role == KVConnectorRole.WORKER else 0
        )
        self._tp_rank = 0
        self._tp_size = int(vllm_config.parallel_config.tensor_parallel_size)
        if role == KVConnectorRole.WORKER:
            tp_group = get_tp_group()
            self._tp_rank = int(tp_group.rank_in_group)
            self._tp_size = int(tp_group.world_size)
        if self._timing_enabled and role == KVConnectorRole.WORKER:
            self._active_timing: _ActiveBatchTiming | None = None
            self._pending_timing: deque[_PendingBatchTiming] = deque()

        self.p2p_nccl_engine = (
            P2pNcclEngine(
                local_rank=self._local_rank,
                config=self._kv_transfer_config,
                hostname="",
                port_offset=self._rank,
                tp_rank=self._tp_rank,
                tp_size=self._tp_size,
            )
            if role == KVConnectorRole.WORKER
            else None
        )

    # ==============================
    # Worker-side methods
    # ==============================

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        """Start loading the KV cache from the connector buffer to vLLM's
        paged KV buffer.

        Args:
            forward_context (ForwardContext): the forward context.
            **kwargs: additional arguments for the load operation

        Note:
            The number of elements in kv_caches and layer_names should be
            the same.
        """

        # Producer timing starts at the same pre-forward hook. Consumers start
        # only after the initial KV receive and insertion below.
        if self.is_producer:
            if self.p2p_nccl_engine is not None and self._timing_enabled:
                metadata = self._get_connector_metadata()
                assert isinstance(metadata, P2pNcclTimingConnectorMetadata)
                for request in metadata.requests:
                    self.p2p_nccl_engine.begin_request_timing(
                        request.request_id, metadata.batch_id
                    )
            self._start_batch_timing()
            return

        assert self.p2p_nccl_engine is not None

        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            self._start_batch_timing()
            return

        def inject_kv_into_layer(
            layer: torch.Tensor,
            kv_cache: torch.Tensor,
            block_ids: torch.Tensor,
            request_id: str,
        ) -> None:
            """
            Inject KV cache data into a given attention layer tensor.

            This function updates `layer` in-place with values from `kv_cache`,
            handling different backend layouts:
              - MLA (Multi-Linear Attention) or FlashInfer: KV tensors are
                indexed along the first dimension.
              - FlashAttention: KV tensors are indexed along the second
                dimension.

            If the number of provided block IDs does not match the number of KV
            blocks, only the overlapping portion is updated, and a warning is
            logged.

            Args:
                layer (torch.Tensor): The attention layer KV tensor to update.
                kv_cache (torch.Tensor): The KV cache tensor to inject.
                block_ids (torch.Tensor): Indices of the blocks to update.
                request_id (str): Request identifier used for logging.

            Returns:
                None. The function modifies `layer` in-place.
            """
            if (
                isinstance(attn_metadata, MLACommonMetadata) or layer.shape[1] == 2
            ):  # MLA or FlashInfer
                num_block = kv_cache.shape[0]
                self.check_tensors_except_dim(layer, kv_cache, 0)
                if len(block_ids) == num_block:
                    layer[block_ids, ...] = kv_cache
                else:
                    layer[block_ids[:num_block], ...] = kv_cache
                    logger.warning(
                        "🚧kv_cache does not match, block_ids:%d, "
                        "num_block:%d, request_id:%s",
                        len(block_ids),
                        num_block,
                        request_id,
                    )

            elif layer.shape[0] == 2:  # FlashAttention
                num_block = kv_cache.shape[1]
                self.check_tensors_except_dim(layer, kv_cache, 1)
                if len(block_ids) == num_block:
                    layer[:, block_ids, ...] = kv_cache
                else:
                    layer[:, block_ids[:num_block], ...] = kv_cache
                    logger.warning(
                        "🚧kv_cache does not match, block_ids:%d, "
                        "num_block:%d, request_id:%s",
                        len(block_ids),
                        num_block,
                        request_id,
                    )

        # Get the metadata
        metadata: KVConnectorMetadata = self._get_connector_metadata()
        assert isinstance(metadata, P2pNcclConnectorMetadata)
        timing_metadata = None
        if self._timing_enabled:
            assert isinstance(metadata, P2pNcclTimingConnectorMetadata)
            timing_metadata = metadata

        if metadata is None:
            return

        # Load the KV for each request each layer
        for request in metadata.requests:
            request_id = request.request_id
            if self._timing_enabled:
                assert timing_metadata is not None
                self.p2p_nccl_engine.begin_request_timing(
                    request_id, timing_metadata.batch_id
                )
            ip, port = self.parse_request_id(request_id, False)
            remote_address = ip + ":" + str(port + self._rank)
            try:
                for layer_name in forward_context.no_compile_layers:
                    layer = forward_context.no_compile_layers[layer_name]

                    # Only process layers that have kv_cache
                    # attribute (attention layers) Skip non-attention
                    # layers like FusedMoE
                    kv_cache = getattr(layer, "kv_cache", None)
                    if kv_cache is None:
                        continue

                    layer = kv_cache

                    kv_cache = self.p2p_nccl_engine.recv_tensor(
                        request.request_id + "#" + layer_name, remote_address
                    )

                    if kv_cache is None:
                        logger.warning("🚧kv_cache is None, %s", request.request_id)
                        continue

                    insertion_span = None
                    if self._timing_enabled:
                        insertion_span = (
                            self.p2p_nccl_engine.start_cuda_timing_span()
                        )
                    inject_kv_into_layer(
                        layer, kv_cache, request.block_ids, request.request_id
                    )
                    if self._timing_enabled:
                        self.p2p_nccl_engine.finish_cuda_timing_span(
                            request_id, "insertion", insertion_span
                        )
            except Exception as exc:
                if self._timing_enabled:
                    self.p2p_nccl_engine.fail_request_timing(
                        request_id, f"consumer_load_failed:{type(exc).__name__}"
                    )
                raise
            if self._timing_enabled:
                assert timing_metadata is not None
                self.p2p_nccl_engine.seal_request_timing(
                    request_id, timing_metadata.batch_id
                )

        self._start_batch_timing()

    def wait_for_layer_load(self, layer_name: str) -> None:
        """Blocking until the KV for a specific layer is loaded into vLLM's
        paged buffer.

        This interface will be useful for layer-by-layer pipelining.

        Args:
            layer_name: the name of that layer
        """
        return

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        """Start saving the KV cache of the layer from vLLM's paged buffer
        to the connector.

        Args:
            layer_name (str): the name of the layer.
            kv_layer (torch.Tensor): the paged KV buffer of the current
                layer in vLLM.
            attn_metadata (AttentionMetadata): the attention metadata.
            **kwargs: additional arguments for the save operation.
        """

        # Only producer/prefill saves KV Cache
        if not self.is_producer:
            return

        assert self.p2p_nccl_engine is not None

        def extract_kv_from_layer(
            layer: torch.Tensor,
            block_ids: torch.Tensor,
        ) -> torch.Tensor:
            """
            Extract KV cache slices from a given attention layer tensor.

            This function handles multiple backend layouts:
              - MLA (Multi-Linear Attention) or FlashInfer: KV tensors are
                indexed along the first dimension.
              - FlashAttention: KV tensors are indexed along the second
                dimension.

            Args:
                layer (torch.Tensor): The KV cache from the attention layer.
                block_ids (torch.Tensor): Indices of blocks to extract.

            Returns:
                torch.Tensor: A tensor containing the extracted KV slices.
                Returns None if the layout is unsupported.
            """
            if (
                isinstance(attn_metadata, MLACommonMetadata) or layer.shape[1] == 2
            ):  # MLA or FlashInfer
                return layer[block_ids, ...]

            if layer.shape[0] == 2:  # FlashAttention
                return layer[:, block_ids, ...]

            return None

        connector_metadata = self._get_connector_metadata()
        assert isinstance(connector_metadata, P2pNcclConnectorMetadata)
        timing_metadata = None
        if self._timing_enabled:
            assert isinstance(
                connector_metadata, P2pNcclTimingConnectorMetadata
            )
            timing_metadata = connector_metadata
        for request in connector_metadata.requests:
            request_id = request.request_id
            ip, port = self.parse_request_id(request_id, True)
            remote_address = ip + ":" + str(port + self._rank)

            extraction_span = None
            if self._timing_enabled:
                extraction_span = self.p2p_nccl_engine.start_cuda_timing_span()
            kv_cache = extract_kv_from_layer(kv_layer, request.block_ids)
            if self._timing_enabled:
                assert timing_metadata is not None
                self.p2p_nccl_engine.finish_cuda_timing_span(
                    request_id, "extraction", extraction_span
                )
                assert timing_metadata is not None
                self.p2p_nccl_engine.send_tensor(
                    request_id + "#" + layer_name,
                    kv_cache,
                    remote_address,
                    batch_id=timing_metadata.batch_id,
                )
            else:
                self.p2p_nccl_engine.send_tensor(
                    request_id + "#" + layer_name, kv_cache, remote_address
                )

    def wait_for_save(self):
        self._end_batch_timing()
        if self.is_producer:
            assert self.p2p_nccl_engine is not None
            if self._timing_enabled:
                metadata = self._get_connector_metadata()
                assert isinstance(metadata, P2pNcclTimingConnectorMetadata)
                for request in metadata.requests:
                    self.p2p_nccl_engine.seal_request_timing(
                        request.request_id, metadata.batch_id
                    )
            self.p2p_nccl_engine.wait_for_sent()

    def on_model_output_ready(self) -> None:
        if not self._timing_enabled:
            return
        if self.p2p_nccl_engine is not None:
            self.p2p_nccl_engine.publish_ready_timing()
        self._drain_ready_batch_timing()

    def get_finished(
        self, finished_req_ids: set[str], **kwargs: Any
    ) -> tuple[set[str] | None, set[str] | None]:
        """
        Notifies worker-side connector ids of requests that have
        finished generating tokens.

        Returns:
            ids of requests that have finished asynchronous transfer,
            tuple of (sending/saving ids, recving/loading ids).
            The finished saves/sends req ids must belong to a set provided in a
            call to this method (this call or a prior one).
        """

        assert self.p2p_nccl_engine is not None

        no_compile_layers = self._vllm_config.compilation_config.static_forward_context
        return self.p2p_nccl_engine.get_finished(finished_req_ids, no_compile_layers)

    # ==============================
    # Scheduler-side methods
    # ==============================

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        """
        Get number of new tokens that can be loaded from the
        external KV cache beyond the num_computed_tokens.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            the number of tokens that can be loaded from the
            external KV cache beyond what is already computed.
        """
        if self.is_producer:
            return 0, False

        prompt_token_ids = request.prompt_token_ids or []
        num_external_tokens = len(prompt_token_ids) - 1 - num_computed_tokens

        if num_external_tokens < 0:
            num_external_tokens = 0

        return num_external_tokens, False

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        """
        Update KVConnector state after block allocation.
        """
        if not self.is_producer and num_external_tokens > 0:
            self._requests_need_load[request.request_id] = (
                request,
                blocks.get_block_ids()[0],
            )

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        """Build the connector metadata for this step.

        This function should NOT modify any fields in the scheduler_output.
        Also, calling this function will reset the state of the connector.

        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
        """

        meta = (
            P2pNcclTimingConnectorMetadata()
            if self._timing_enabled
            else P2pNcclConnectorMetadata()
        )

        for new_req in scheduler_output.scheduled_new_reqs:
            if self.is_producer:
                num_scheduled_tokens = (scheduler_output.num_scheduled_tokens)[
                    new_req.req_id
                ]
                num_tokens = num_scheduled_tokens + new_req.num_computed_tokens
                # the request's prompt is chunked prefill
                if num_tokens < len(new_req.prompt_token_ids or []):
                    # 'CachedRequestData' has no attribute 'prompt_token_ids'
                    self.chunked_prefill[new_req.req_id] = (
                        new_req.block_ids[0],
                        new_req.prompt_token_ids,
                    )
                    continue
                # the request's prompt is not chunked prefill
                meta.add_request(
                    request_id=new_req.req_id,
                    token_ids=new_req.prompt_token_ids or [],
                    block_ids=new_req.block_ids[0],
                    block_size=self._block_size,
                )
                continue
            if new_req.req_id in self._requests_need_load:
                meta.add_request(
                    request_id=new_req.req_id,
                    token_ids=new_req.prompt_token_ids or [],
                    block_ids=new_req.block_ids[0],
                    block_size=self._block_size,
                )
                self._requests_need_load.pop(new_req.req_id)

        cached_reqs = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(cached_reqs.req_ids):
            num_computed_tokens = cached_reqs.num_computed_tokens[i]
            new_block_ids = cached_reqs.new_block_ids[i]
            resumed_from_preemption = req_id in cached_reqs.resumed_req_ids

            if self.is_producer:
                num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
                num_tokens = num_scheduled_tokens + num_computed_tokens
                assert req_id in self.chunked_prefill
                assert new_block_ids is not None
                block_ids = new_block_ids[0]
                if not resumed_from_preemption:
                    block_ids = self.chunked_prefill[req_id][0] + block_ids
                prompt_token_ids = self.chunked_prefill[req_id][1]
                assert prompt_token_ids is not None
                # the request's prompt is chunked prefill again
                if num_tokens < len(prompt_token_ids):
                    self.chunked_prefill[req_id] = (block_ids, prompt_token_ids)
                    continue
                # the request's prompt is all prefilled finally
                meta.add_request(
                    request_id=req_id,
                    token_ids=prompt_token_ids,
                    block_ids=block_ids,
                    block_size=self._block_size,
                )
                self.chunked_prefill.pop(req_id, None)
                continue

            # NOTE(rob): here we rely on the resumed requests being
            # the first N requests in the list scheduled_cache_reqs.
            if not resumed_from_preemption:
                break
            if req_id in self._requests_need_load:
                request, _ = self._requests_need_load.pop(req_id)
                total_tokens = num_computed_tokens + 1
                token_ids = request.all_token_ids[:total_tokens]

                # NOTE(rob): For resumed req, new_block_ids is all
                # of the block_ids for the request.
                assert new_block_ids is not None
                block_ids = new_block_ids[0]

                meta.add_request(
                    request_id=req_id,
                    token_ids=token_ids,
                    block_ids=block_ids,
                    block_size=self._block_size,
                )

        self._requests_need_load.clear()
        if self._timing_enabled:
            self._add_scheduled_timing(meta, scheduler_output)
        return meta

    def _add_scheduled_timing(
        self,
        meta: P2pNcclConnectorMetadata,
        scheduler_output: SchedulerOutput,
    ) -> None:
        assert isinstance(meta, P2pNcclTimingConnectorMetadata)
        transfer_request_ids = {request.request_id for request in meta.requests}
        meta.batch_request_ids = [
            str(request_id) for request_id in scheduler_output.num_scheduled_tokens
        ]
        meta.timed_requests = []
        for request_id, token_count in scheduler_output.num_scheduled_tokens.items():
            if self.is_producer:
                phase = "prefill"
            elif request_id in transfer_request_ids:
                phase = "decode_first"
            else:
                phase = "decode_generation"
            meta.timed_requests.append(
                ScheduledReqMeta(
                    request_id=str(request_id),
                    token_count=int(token_count),
                    phase=phase,
                )
            )
        counter = self._timing_batch_counter
        self._timing_batch_counter += 1
        engine_id = str(self._kv_transfer_config.engine_id or "engine")[:12]
        role = "producer" if self.is_producer else "consumer"
        # This scheduler-created value is copied to every TP worker.
        meta.batch_id = f"p2p-{role}-{engine_id}-{counter:08d}"

    def _start_batch_timing(self) -> None:
        if not self._timing_enabled or self.p2p_nccl_engine is None:
            return
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, P2pNcclTimingConnectorMetadata)
        if not metadata.timed_requests or metadata.batch_id is None:
            return
        try:
            start = torch.cuda.Event(enable_timing=True)
            host_start_perf_ns = time.perf_counter_ns()
            host_start_wall_ns = time.time_ns()
            start.record()
            self._active_timing = _ActiveBatchTiming(
                batch_id=metadata.batch_id,
                requests=list(metadata.timed_requests),
                batch_request_ids=list(metadata.batch_request_ids or ()),
                forward_start=start,
                host_start_perf_ns=host_start_perf_ns,
                host_start_wall_ns=host_start_wall_ns,
            )
        except Exception:
            self._safe_timing_log_exception(
                "Failed to start P2P NCCL batch timing %s", metadata.batch_id
            )

    def _end_batch_timing(self) -> None:
        if not self._timing_enabled:
            return
        active = getattr(self, "_active_timing", None)
        if active is None:
            return
        try:
            host_end_perf_ns = time.perf_counter_ns()
            host_end_wall_ns = time.time_ns()
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            self._pending_timing.append(
                _PendingBatchTiming(
                    batch_id=active.batch_id,
                    requests=active.requests,
                    batch_request_ids=active.batch_request_ids,
                    forward_span=CudaEventSpan(active.forward_start, end),
                    host_start_perf_ns=active.host_start_perf_ns,
                    host_end_perf_ns=host_end_perf_ns,
                    host_start_wall_ns=active.host_start_wall_ns,
                    host_end_wall_ns=host_end_wall_ns,
                )
            )
        except Exception:
            self._safe_timing_log_exception(
                "Failed to finish P2P NCCL batch timing %s", active.batch_id
            )
        finally:
            self._active_timing = None

    def _drain_ready_batch_timing(self) -> None:
        ready: list[_PendingBatchTiming] = []
        retained: deque[_PendingBatchTiming] = deque()
        while self._pending_timing:
            pending = self._pending_timing.popleft()
            try:
                if pending.forward_span.ready():
                    ready.append(pending)
                else:
                    retained.append(pending)
            except Exception:
                retained.append(pending)
                self._safe_timing_log_exception(
                    "Failed to query P2P NCCL batch timing %s", pending.batch_id
                )
        self._pending_timing = retained
        for pending in ready:
            try:
                self._emit_batch_timing(pending)
            except Exception:
                self._safe_timing_log_exception(
                    "Failed to emit P2P NCCL batch timing %s", pending.batch_id
                )

    def _emit_batch_timing(self, pending: _PendingBatchTiming) -> None:
        assert self.p2p_nccl_engine is not None
        anchor = self.p2p_nccl_engine.timing_anchor
        if anchor is None:
            return
        forward_gpu_ms = pending.forward_span.elapsed_ms()
        anchor_start_ms, anchor_end_ms = (
            pending.forward_span.anchor_offsets_ms(anchor)
        )
        host_ms, clock_anomaly = nonnegative_duration_ms(
            pending.host_start_perf_ns, pending.host_end_perf_ns
        )
        clock_anomaly |= pending.host_end_wall_ns < pending.host_start_wall_ns
        phases = {request.phase for request in pending.requests}
        if self.is_producer:
            phase = "prefill"
        elif phases == {"decode_first"}:
            phase = "decode_first"
        elif phases == {"decode_generation"}:
            phase = "decode_generation"
        else:
            phase = "mixed"
        batch_request_ids = list(pending.batch_request_ids) or [
            request.request_id for request in pending.requests
        ]
        client_ids = {
            extract_client_request_id(request_id)
            for request_id in batch_request_ids
        }
        send_type = self._kv_transfer_config.get_from_extra_config(
            "send_type", "PUT_ASYNC"
        )
        record = {
            "schema_version": 1,
            "record_type": "batch_forward",
            "connector": "P2pNcclConnector",
            "role": "producer" if self.is_producer else "consumer",
            "send_type": send_type,
            "request_id": (
                batch_request_ids[0] if len(batch_request_ids) == 1 else None
            ),
            "client_request_id": (
                next(iter(client_ids)) if len(client_ids) == 1 else None
            ),
            "batch_id": pending.batch_id,
            "batch_size": len(batch_request_ids),
            "batch_request_ids": batch_request_ids,
            "requests": [
                {
                    "request_id": request.request_id,
                    "client_request_id": extract_client_request_id(
                        request.request_id
                    ),
                    "token_count": request.token_count,
                    "phase": request.phase,
                }
                for request in pending.requests
            ],
            "phase": phase,
            "tp_rank": self._tp_rank,
            "tp_size": self._tp_size,
            "local_rank": self._local_rank,
            "host": self.p2p_nccl_engine._hostname,
            "device": str(self.p2p_nccl_engine.device),
            "measurement_scope": (
                "exact_per_request"
                if len(batch_request_ids) == 1
                else "batch_shared"
            ),
            "forward_gpu_ms": forward_gpu_ms,
            "forward_host_envelope_ms": host_ms,
            "host_envelope_start_wall_ns": pending.host_start_wall_ns,
            "host_envelope_end_wall_ns": pending.host_end_wall_ns,
            "gpu_anchor_start_ms": anchor_start_ms,
            "gpu_anchor_end_ms": anchor_end_ms,
            "status": "clock_anomaly" if clock_anomaly else "ok",
            "measurement_semantics": {
                "forward": "model_stream_envelope_not_kernel_sum",
                "batch_compute": "shared_not_divided_across_requests",
                "put": "synchronous_send_not_added_to_prefill",
                "put_async": "overlap_prefill_use_batch_tail_not_request_sum",
            },
        }
        timing_log_record(logger, record)

    @staticmethod
    def _safe_timing_log_exception(message: str, *args: Any) -> None:
        try:
            logger.exception(message, *args)
        except Exception:
            pass

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Called when a request has finished, before its blocks are freed.

        Returns:
            True if the request is being saved/sent asynchronously and blocks
            should not be freed until the request_id is returned from
            get_finished().
            Optional KVTransferParams to be included in the request outputs
            returned by the engine.
        """

        self.chunked_prefill.pop(request.request_id, None)

        return False, None

    # ==============================
    # Static methods
    # ==============================

    @staticmethod
    def parse_request_id(request_id: str, is_prefill=True) -> tuple[str, int]:
        # Regular expression to match the string hostname and integer port
        if is_prefill:
            pattern = r"___decode_addr_(.*):(\d+)"
        else:
            pattern = r"___prefill_addr_(.*):(\d+)___"

        # Use re.search to find the pattern in the request_id
        match = re.search(pattern, request_id)
        if match:
            # Extract the ranks
            ip = match.group(1)
            port = int(match.group(2))

            return ip, port
        raise ValueError(f"Request id {request_id} does not contain hostname and port")

    @staticmethod
    def check_tensors_except_dim(tensor1, tensor2, dim):
        shape1 = tensor1.size()
        shape2 = tensor2.size()

        if len(shape1) != len(shape2) or not all(
            s1 == s2 for i, (s1, s2) in enumerate(zip(shape1, shape2)) if i != dim
        ):
            raise NotImplementedError(
                "Currently, only symmetric TP is supported. Asymmetric TP, PP,"
                "and others will be supported in future PRs."
            )
