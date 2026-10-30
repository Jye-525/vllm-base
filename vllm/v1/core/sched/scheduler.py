# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools
import os
import time
from collections import defaultdict, deque
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from vllm import envs
from vllm.compilation.cuda_graph import CUDAGraphStat
from vllm.config import VllmConfig
from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorMetadata,
    ECConnectorRole,
)
from vllm.distributed.ec_transfer.ec_connector.factory import ECConnectorFactory
from vllm.distributed.kv_events import EventPublisherFactory, KVEventBatch
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    KVConnectorBase_V1,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    RoutedExpertsManager,
)
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.multimodal.encoder_budget import MultiModalBudget
from vllm.v1.core.encoder_cache_manager import (
    EncoderCacheManager,
    EncoderDecoderCacheManager,
)
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.kv_cache_utils import BlockHash, KVCacheBlock
from vllm.v1.core.request_logger import RequestEventLogger
from vllm.v1.core.sched.dspf import DSPFScheduler, check_dspf_scope
from vllm.v1.core.sched.interface import PauseState, SchedulerInterface
from vllm.v1.core.sched.output import (
    CachedRequestData,
    GrammarOutput,
    NewRequestData,
    SchedulerOutput,
)
from vllm.v1.core.sched.predictive_admission import (
    PredictiveAdmissionController,
    RequestSnapshot,
)
from vllm.v1.core.sched.preemption_diagnostics import PreemptionDiagnostics
from vllm.v1.core.sched.prefill_cost import PrefillCostShape
from vllm.v1.core.sched.survival_diagnostics import SurvivalRecorder
from vllm.v1.core.sched.request_queue import (
    FCFSRequestQueue,
    RequestQueue,
    SchedulingPolicy,
    create_request_queue,
)
from vllm.v1.core.sched.utils import check_stop, remove_all
from vllm.v1.core.sched.waiting_prefix_index import PendingPrefixIndex
from vllm.v1.core.sched.waiting_ranker import (
    DSPFRanking,
    RankPlan,
    rank_clpm,
    rank_lpm,
    rank_sjf,
    rank_spf,
    rank_sspf,
)
from vllm.v1.engine import EngineCoreEventType, EngineCoreOutput, EngineCoreOutputs
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig
from vllm.v1.metrics.perf import ModelMetrics, PerfStats
from vllm.v1.metrics.stats import PrefixCacheStats, SchedulerStats
from vllm.v1.outputs import DraftTokenIds, KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus, StreamingUpdate
from vllm.v1.spec_decode.metrics import SpecDecodingStats
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.utils import record_function_or_nullcontext

logger = init_logger(__name__)


class _BlockRefCounts(Mapping[int, int]):
    """Read live counts during a synchronous, non-mutating gate evaluation."""

    def __init__(self, blocks: list[KVCacheBlock]) -> None:
        self.blocks = blocks

    def __getitem__(self, block_id: int) -> int:
        if not 0 <= block_id < len(self.blocks):
            raise KeyError(block_id)
        return self.blocks[block_id].ref_cnt

    def __iter__(self) -> Iterator[int]:
        return iter(range(1, len(self.blocks)))

    def __len__(self) -> int:
        return len(self.blocks) - 1


@dataclass(slots=True)
class _PredictiveAdmissionState:
    running: list[RequestSnapshot]
    block_ref_counts: Mapping[int, int]
    pinned_block_count: int


class Scheduler(SchedulerInterface):
    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        block_size: int,
        hash_block_size: int | None = None,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        self.vllm_config = vllm_config
        self.scheduler_config = vllm_config.scheduler_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.kv_cache_config = kv_cache_config
        self.kv_events_config = vllm_config.kv_events_config
        self.parallel_config = vllm_config.parallel_config
        self.log_stats = log_stats
        self.observability_config = vllm_config.observability_config
        self.kv_metrics_collector: KVCacheMetricsCollector | None = None
        if self.observability_config.kv_cache_metrics:
            self.kv_metrics_collector = KVCacheMetricsCollector(
                self.observability_config.kv_cache_metrics_sample,
            )
        self.structured_output_manager = structured_output_manager
        self.is_encoder_decoder = vllm_config.model_config.is_encoder_decoder

        # include_finished_set controls whether a separate set of finished
        # request ids should be included in the EngineCoreOutputs returned
        # by update_from_outputs(). This is currently used in the multi-engine
        # case to track request lifetimes efficiently.
        self.finished_req_ids_dict: dict[int, set[str]] | None = (
            defaultdict(set) if include_finished_set else None
        )
        self.prev_step_scheduled_req_ids: set[str] = set()

        # Scheduling constraints.
        self.max_num_running_reqs = self.scheduler_config.max_num_seqs
        self.max_num_scheduled_tokens = (
            self.scheduler_config.max_num_scheduled_tokens
            if self.scheduler_config.max_num_scheduled_tokens
            else self.scheduler_config.max_num_batched_tokens
        )
        self.max_model_len = vllm_config.model_config.max_model_len
        self.enable_kv_cache_events = (
            self.kv_events_config is not None
            and self.kv_events_config.enable_kv_cache_events
        )

        # Create KVConnector for the Scheduler. Note that each Worker
        # will have a corresponding KVConnector with Role=WORKER.
        # KV Connector pushes/pull of remote KVs for P/D and offloading.
        self.connector = None
        self.connector_prefix_cache_stats: PrefixCacheStats | None = None
        self.recompute_kv_load_failures = True
        if self.vllm_config.kv_transfer_config is not None:
            assert not self.is_encoder_decoder, (
                "Encoder-decoder models are not currently supported with KV connectors"
            )
            self.connector = KVConnectorFactory.create_connector(
                config=self.vllm_config,
                role=KVConnectorRole.SCHEDULER,
                kv_cache_config=self.kv_cache_config,
            )
            if self.log_stats:
                self.connector_prefix_cache_stats = PrefixCacheStats()
            kv_load_failure_policy = (
                self.vllm_config.kv_transfer_config.kv_load_failure_policy
            )
            self.recompute_kv_load_failures = kv_load_failure_policy == "recompute"

        self.kv_event_publisher = EventPublisherFactory.create(
            self.kv_events_config,
            self.parallel_config.data_parallel_index,
        )
        self.ec_connector = None
        if self.vllm_config.ec_transfer_config is not None:
            self.ec_connector = ECConnectorFactory.create_connector(
                config=self.vllm_config, role=ECConnectorRole.SCHEDULER
            )

        num_gpu_blocks = self.cache_config.num_gpu_blocks
        assert num_gpu_blocks is not None and num_gpu_blocks > 0

        self.block_size = block_size
        self.dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size
        self.pcp_world_size = vllm_config.parallel_config.prefill_context_parallel_size

        # req_id -> Request
        self.requests: dict[str, Request] = {}
        # Scheduling policy
        try:
            self.policy = SchedulingPolicy(self.scheduler_config.policy)
        except ValueError as e:
            raise ValueError(
                f"Unknown scheduling policy: {self.scheduler_config.policy}"
            ) from e
        if (
            self.policy
            in (
                SchedulingPolicy.LPM,
                SchedulingPolicy.CLPM,
            )
            and not self.cache_config.enable_prefix_caching
        ):
            logger.warning_once(
                "LPM/cLPM require prefix caching. Falling back to FCFS "
                "because prefix caching is disabled."
            )
            self.policy = SchedulingPolicy.FCFS
        # Priority queues for requests.
        self.waiting = create_request_queue(self.policy)
        # requests skipped in waiting flow due async deps or constraints.
        self.skipped_waiting = create_request_queue(self.policy)
        self.pending_prefix_index = PendingPrefixIndex()
        self._clpm_lane_a_pass = 0.0
        self._clpm_lane_b_pass = 0.0
        self._clpm_dynamic_b_share = 0.3
        self._init_waiting_rankers()
        self._dspf: DSPFScheduler | None = None
        if self.policy == SchedulingPolicy.DSPF:
            self._dspf = self._init_dspf()
        self.running: list[Request] = []

        # The request IDs that are finished in between the previous and the
        # current steps. This is used to notify the workers about the finished
        # requests so that they can free the cached states for those requests.
        # This is flushed at the end of each scheduling step.
        self.finished_req_ids: set[str] = set()

        # Counter for requests waiting for streaming input. Used to calculate
        # number of unfinished requests
        self.num_waiting_for_streaming_input: int = 0

        # KV Connector: requests in process of async KV loading or recving
        self.finished_recving_kv_req_ids: set[str] = set()
        self.failed_recving_kv_req_ids: set[str] = set()

        # Encoder-related.
        # Calculate encoder cache size if applicable
        supports_mm_inputs = mm_registry.supports_multimodal_inputs(
            vllm_config.model_config
        )
        mm_budget = (
            MultiModalBudget(vllm_config, mm_registry) if supports_mm_inputs else None
        )

        # NOTE: Text-only encoder-decoder models are implemented as
        # multi-modal models for convenience
        # Example: https://github.com/vllm-project/bart-plugin
        if self.is_encoder_decoder:
            assert mm_budget and len(mm_budget.mm_max_toks_per_item) <= 1, (
                "Encoder-decoder models are expected to implement the "
                "multimodal interface with at most one modality."
            )

        self.max_num_encoder_input_tokens = (
            mm_budget.encoder_compute_budget if mm_budget else 0
        )
        encoder_cache_size = mm_budget.encoder_cache_size if mm_budget else 0
        self.encoder_cache_manager = (
            EncoderDecoderCacheManager(cache_size=encoder_cache_size)
            if self.is_encoder_decoder
            else EncoderCacheManager(cache_size=encoder_cache_size)
        )

        speculative_config = vllm_config.speculative_config
        self.use_eagle = False
        self.num_spec_tokens = self.num_lookahead_tokens = 0
        if speculative_config:
            self.num_spec_tokens = speculative_config.num_speculative_tokens
            if speculative_config.use_eagle():
                self.use_eagle = True
                self.num_lookahead_tokens = self.num_spec_tokens
            if speculative_config.uses_draft_model():
                self.num_lookahead_tokens = self.num_spec_tokens

        # Create the KV cache manager.
        if hash_block_size is None:
            hash_block_size = block_size
        self.kv_cache_manager = KVCacheManager(
            kv_cache_config=kv_cache_config,
            max_model_len=self.max_model_len,
            max_num_batched_tokens=self.scheduler_config.max_num_batched_tokens,
            enable_caching=self.cache_config.enable_prefix_caching,
            use_eagle=self.use_eagle,
            log_stats=self.log_stats,
            enable_kv_cache_events=self.enable_kv_cache_events,
            dcp_world_size=self.dcp_world_size,
            pcp_world_size=self.pcp_world_size,
            hash_block_size=hash_block_size,
            metrics_collector=self.kv_metrics_collector,
        )
        # Bind GPU block pool to the KV connector. This must happen after
        # kv_cache_manager is constructed so block_pool is available.
        if self.connector is not None:
            self.connector.bind_gpu_block_pool(self.kv_cache_manager.block_pool)

        # Peek eviction consumes cumulative block-hash demand maintained by
        # PendingPrefixIndex. Keep stock LRU when prefix caching is disabled.
        policy = self.scheduler_config.kv_cache_eviction_policy
        if not self.cache_config.enable_prefix_caching and policy != "lru":
            logger.warning_once(
                "%s eviction requires prefix caching. Falling back to LRU "
                "because prefix caching is disabled.",
                policy,
            )
            policy = "lru"
        self._eviction_policy = policy
        self._peek_eviction_mode = self.scheduler_config.peek_eviction_mode
        block_pool = self.kv_cache_manager.block_pool
        block_pool._eviction_policy = self._eviction_policy
        block_pool._peek_eviction_mode = self._peek_eviction_mode
        block_pool._peek_eviction_recency_weight = (
            self.scheduler_config.peek_eviction_recency_weight
        )
        block_pool._peek_eviction_decay_tau = (
            self.scheduler_config.peek_eviction_decay_tau
        )
        block_pool._peek_eviction_scan_cap = (
            self.scheduler_config.peek_eviction_scan_cap
        )
        # Skip-aware eviction protects the prefixes of waiting requests that
        # LPM admission skipped. It matches request.block_hashes against
        # group-0 block hashes, so it needs one group hashed at its block size.
        self._skip_eviction = self._eviction_policy == "skip"
        self._skip_protected: list[Request] = []
        # Requests skipped this step that have a cached prefix to protect.
        self._step_skip_protectable: list[Request] = []
        self._skip_eviction_max_protected = (
            self.scheduler_config.skip_eviction_max_protected
        )
        if self._skip_eviction:
            groups = kv_cache_config.kv_cache_groups
            if (
                len(groups) != 1
                or groups[0].kv_cache_spec.block_size != block_pool.hash_block_size
            ):
                raise ValueError(
                    "kv_cache_eviction_policy='skip' requires a single KV cache "
                    "group whose block size equals the hash block size"
                )
            if self.connector is not None:
                # Connectors such as simple KV offload walk free_block_queue,
                # which no longer holds the protected free blocks.
                raise ValueError(
                    "kv_cache_eviction_policy='skip' does not support KV connectors"
                )
            logger.info(
                "Skip-aware eviction: protect the prefixes of up to %d skipped "
                "waiting requests, evicting them after all other free blocks.",
                self._skip_eviction_max_protected,
            )

        self.use_pp = self.parallel_config.pipeline_parallel_size > 1
        self.use_v2_model_runner = vllm_config.use_v2_model_runner
        self.scheduler_reserve_full_isl = (
            self.scheduler_config.scheduler_reserve_full_isl
        )

        self.has_mamba_layers = kv_cache_config.has_mamba_layers
        self.needs_kv_cache_zeroing = kv_cache_config.needs_kv_cache_zeroing
        self.need_mamba_block_aligned_split = (
            self.has_mamba_layers and self.cache_config.mamba_cache_mode == "align"
        )
        self.predictive_admission: PredictiveAdmissionController | None = None
        self._pkas_block_ids: dict[
            str, tuple[Sequence[KVCacheBlock], tuple[int, ...]]
        ] = {}
        self._pkas_ref_counts = _BlockRefCounts(block_pool.blocks)
        self._pkas_step_stats: dict[str, int] | None = None
        self._pkas_last_gate_target: dict[str, int] = {}
        # Scheduler step number; every scheduler log line carries it.
        self._step_id = 0
        self._step_evicted_blocks_start = 0
        # Rejections are reused until an admission, completion, or preemption
        # changes the epoch, or until the step-based expiry lets waiting time
        # (running requests nearing completion) change the outcome.
        self._pkas_epoch = 0
        self._pkas_reject_cache: dict[str, tuple[int, int]] = {}
        if self.scheduler_config.admission_policy == "pkas":
            unsupported = []
            # Without prefix caching LPM has fallen back to FCFS above, which
            # is PKAS's original setting; FCFS is accepted only then.
            configured = SchedulingPolicy(self.scheduler_config.policy)
            if configured != SchedulingPolicy.LPM and not (
                configured == SchedulingPolicy.FCFS
                and not self.cache_config.enable_prefix_caching
            ):
                unsupported.append("scheduling_policy_must_be_lpm")
            if (
                self.scheduler_config.async_scheduling
                or type(self).__name__ == "AsyncScheduler"
            ):
                unsupported.append("async_scheduling")
            if speculative_config is not None:
                unsupported.append("speculative_decoding")
            if self.connector is not None or self.ec_connector is not None:
                unsupported.append("offload_or_transfer")
            if self.use_pp or self.dcp_world_size > 1 or self.pcp_world_size > 1:
                unsupported.append("pipeline_or_context_parallelism")
            if self.scheduler_config.runner_type != "generate":
                unsupported.append("non_generation_runner")
            if self.is_encoder_decoder or self.scheduler_config.is_multimodal_model:
                unsupported.append("encoder_or_multimodal")
            groups = kv_cache_config.kv_cache_groups
            if (
                len(groups) != 1
                or type(groups[0].kv_cache_spec) is not FullAttentionSpec
                or groups[0].kv_cache_spec.sliding_window is not None
                or groups[0].kv_cache_spec.attention_chunk_size is not None
            ):
                unsupported.append("hybrid_or_non_full_attention")
            if self.scheduler_reserve_full_isl:
                # PKAS's own simulation reserves the prompt and predicted
                # decode, counting KV that running requests free first.
                unsupported.append("full_isl_admission")
            if unsupported:
                raise ValueError(
                    "PKAS predictive admission does not support: "
                    + ", ".join(unsupported)
                )
            self.predictive_admission = PredictiveAdmissionController(
                block_size=self.block_size,
                capacity_blocks=block_pool.num_gpu_blocks - 1,
                token_budget=self.max_num_scheduled_tokens,
                max_model_len=self.max_model_len,
                long_prefill_token_threshold=(
                    self.scheduler_config.long_prefill_token_threshold
                ),
                output_predictor=self.scheduler_config.pkas_output_predictor,
                output_quantile=self.scheduler_config.pkas_output_quantile,
                output_history=self.scheduler_config.pkas_output_history,
                output_confidence=self.scheduler_config.pkas_output_confidence,
            )
        # PKAS repository variant: its simulation and its chunk clamp.
        self._pkas_repo = (
            self.predictive_admission is not None
            and self.scheduler_config.pkas_simulation == "repo"
        )
        self.perf_metrics: ModelMetrics | None = None
        if self.log_stats and vllm_config.observability_config.enable_mfu_metrics:
            self.perf_metrics = ModelMetrics(vllm_config)

        self.enable_return_routed_experts = (
            vllm_config.model_config.enable_return_routed_experts
        )

        if self.enable_return_routed_experts:
            assert self.dcp_world_size == 1 and self.pcp_world_size == 1, (
                "enable_return_routed_experts does not support context parallelism "
                "(dcp_world_size > 1 or pcp_world_size > 1)"
            )

            self.routed_experts_mgr = RoutedExpertsManager(
                vllm_config=vllm_config,
                kv_cache_config=kv_cache_config,
            )
            # Block-ID snapshot taken at schedule time (before forward),
            # so update_from_output can read slot data even if a later
            # schedule() frees the blocks (async scheduling race).
            self._re_block_ids: dict[str, list[int]] = {}

        self._pause_state: PauseState = PauseState.UNPAUSED

        # Exact token accounting requires completed, non-overlapping text batches.
        groups = kv_cache_config.kv_cache_groups
        self._preemption_cost_supported = not (
            self.scheduler_config.async_scheduling
            or type(self).__name__ == "AsyncScheduler"
            or speculative_config is not None
            or self.connector is not None
            or self.ec_connector is not None
            or self.use_pp
            or self.dcp_world_size > 1
            or self.pcp_world_size > 1
            or self.is_encoder_decoder
            or len(groups) != 1
            or type(groups[0].kv_cache_spec) is not FullAttentionSpec
            or groups[0].kv_cache_spec.sliding_window is not None
            or groups[0].kv_cache_spec.attention_chunk_size is not None
        )
        self._completed_token_high: dict[str, int] = {}
        self._preemption_recovery_high: dict[str, int] = {}
        self._step_preempted_tokens = 0
        self._pending_batch_logs: dict[
            int, tuple[tuple[Any, ...], dict[str, tuple[int, int, int]]]
        ] = {}

        self.preemption_diagnostics: PreemptionDiagnostics | None = None
        if directory := envs.VLLM_PREEMPTION_DIAGNOSTICS_DIR:
            unsupported = []
            if (
                self.scheduler_config.async_scheduling
                or type(self).__name__ == "AsyncScheduler"
            ):
                unsupported.append("async_scheduling")
            if speculative_config is not None:
                unsupported.append("speculative_decoding")
            if self.connector is not None or self.ec_connector is not None:
                unsupported.append("offload_or_transfer")
            if self.use_pp or self.dcp_world_size > 1 or self.pcp_world_size > 1:
                unsupported.append("pipeline_or_context_parallelism")
            groups = kv_cache_config.kv_cache_groups
            if (
                len(groups) != 1
                or type(groups[0].kv_cache_spec) is not FullAttentionSpec
                or groups[0].kv_cache_spec.sliding_window is not None
                or groups[0].kv_cache_spec.attention_chunk_size is not None
            ):
                unsupported.append("hybrid_or_non_full_attention")
            if self.is_encoder_decoder:
                unsupported.append("encoder_decoder")
            self.preemption_diagnostics = PreemptionDiagnostics(
                directory,
                dict(
                    policy=self.policy.value,
                    block_size=block_size,
                    num_blocks=kv_cache_config.num_blocks,
                    prefix_caching=self.cache_config.enable_prefix_caching,
                    chunked_prefill=self.scheduler_config.enable_chunked_prefill,
                    max_num_seqs=self.max_num_running_reqs,
                    max_num_batched_tokens=self.max_num_scheduled_tokens,
                    max_model_len=self.max_model_len,
                    eviction_policy=self._eviction_policy,
                    peek_eviction_mode=self._peek_eviction_mode,
                    async_scheduling=self.scheduler_config.async_scheduling,
                    speculative_decoding=speculative_config is not None,
                    kv_transfer=self.connector is not None,
                    kv_cache_groups=len(groups),
                    pipeline_parallel_size=self.parallel_config.pipeline_parallel_size,
                    decode_context_parallel_size=self.dcp_world_size,
                    prefill_context_parallel_size=self.pcp_world_size,
                ),
                unsupported,
            )
        self.survival_diagnostics: SurvivalRecorder | None = None
        if self.preemption_diagnostics is not None and not unsupported:
            # A subdirectory keeps survival records out of tools that read
            # every preemption-diagnostics JSONL file in the parent directory.
            self.survival_diagnostics = SurvivalRecorder(
                os.path.join(envs.VLLM_PREEMPTION_DIAGNOSTICS_DIR, "survival"),
                self.kv_cache_manager,
                dict(
                    policy=self.policy.value,
                    eviction_policy=self._eviction_policy,
                    block_size=block_size,
                    num_blocks=kv_cache_config.num_blocks,
                    prefix_caching=self.cache_config.enable_prefix_caching,
                    predictive_admission=self.predictive_admission is not None,
                ),
            )

    def _mamba_block_aligned_split(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_local_computed_tokens: int = 0,
        num_external_computed_tokens: int = 0,
    ) -> int:
        assert num_external_computed_tokens == 0, (
            "External KV connector is not verified yet"
        )
        num_computed_tokens = (
            request.num_computed_tokens
            + num_new_local_computed_tokens
            + num_external_computed_tokens
        )
        # Perform block-aligned splitting at prefill phase, including:
        # * non-resumed requests: num_computed_tokens < num_prompt_tokens + 0
        # * resumed requests: num_computed_tokens < (
        #                       num_prompt_tokens + num_output_tokens
        #                     )
        # NOTE: Use `request.num_tokens - 1` to bypass normal decoding.
        if num_computed_tokens < max(request.num_prompt_tokens, request.num_tokens - 1):
            # To enable block-aligned caching of the Mamba state, `num_new_tokens`
            # must be a multiple of `block_size`.
            # As an exception, if `num_new_tokens` is less than `block_size`, the
            # state is simply not cached, requiring no special handling.
            # Additionally, when Eagle mode is enabled, FullAttn prunes the last
            # matching block. To prevent this from causing a Mamba cache miss, the
            # last chunk must be not smaller than `block_size`.
            block_size = self.cache_config.block_size
            last_cache_position = request.num_tokens - request.num_tokens % block_size
            # eagle prune
            if self.use_eagle:
                last_cache_position = max(last_cache_position - block_size, 0)
            num_computed_tokens_after_sched = num_computed_tokens + num_new_tokens
            if num_computed_tokens_after_sched < last_cache_position:
                # align to block_size
                num_new_tokens = num_new_tokens // block_size * block_size
            elif (
                num_computed_tokens
                < last_cache_position
                < num_computed_tokens_after_sched
            ):
                # force to cache the last chunk
                num_new_tokens = last_cache_position - num_computed_tokens
            else:
                # prefill the last few tokens
                pass
        return num_new_tokens

    def _update_pending_demand(self, counts: Mapping[BlockHash, int]) -> None:
        if self._eviction_policy != "peek" or not counts:
            return
        self.kv_cache_manager.block_pool.update_peek_demand_counts(
            counts,
            max_depths=self.pending_prefix_index.demand_max_depths,
        )

    def _track_waiting_request(self, request: Request) -> None:
        counts = self.pending_prefix_index.upsert(
            request.request_id,
            request.block_hashes,
            ready=request.status in (RequestStatus.WAITING, RequestStatus.PREEMPTED),
        )
        self._update_pending_demand(counts)

    def _set_waiting_request_ready(self, request: Request) -> None:
        self.pending_prefix_index.set_ready(
            request.request_id,
            request.status in (RequestStatus.WAITING, RequestStatus.PREEMPTED),
        )

    def _untrack_waiting_request(self, request: Request) -> None:
        counts = self.pending_prefix_index.discard(request.request_id)
        self._update_pending_demand(counts)

    def _republish_pending_demand(self) -> None:
        if self._eviction_policy != "peek":
            return
        counts = {
            block_hash: float(count)
            for block_hash, count in self.pending_prefix_index.demand_counts.items()
        }
        self.kv_cache_manager.block_pool.set_peek_demand_counts(
            counts, self.pending_prefix_index.demand_max_depths
        )

    def _can_rank_waiting_request(
        self,
        request: Request,
        scheduled_loras: set[int],
    ) -> bool:
        """Return whether a waiting request is eligible for prefix ranking.

        This check is intentionally conservative and side-effect free. The
        normal waiting-admission path still performs the authoritative checks
        before scheduling the selected request.
        """
        if request.status != RequestStatus.WAITING and not (
            self._rank_preempted and request.status == RequestStatus.PREEMPTED
        ):
            return False
        if request.num_computed_tokens != 0:
            return False
        if request.skip_reading_prefix_cache:
            return False
        return not (
            self.lora_config
            and request.lora_request
            and len(scheduled_loras) == self.lora_config.max_loras
            and request.lora_request.lora_int_id not in scheduled_loras
        )

    def _init_waiting_rankers(self) -> None:
        """Select the candidate ranker for the configured policy.

        Policies without a ranker (FCFS, PRIORITY) keep their queue order.
        """
        config = self.scheduler_config
        # Default aging: a request that waited 60 s outranks any fresh one,
        # since vLLM bounds prompt and output lengths by max_model_len.
        default_alpha = self.max_model_len / 60.0
        self._spf_age_alpha = (
            default_alpha if config.spf_age_alpha is None else config.spf_age_alpha
        )
        self._sjf_age_alpha = (
            default_alpha if config.sjf_age_alpha is None else config.sjf_age_alpha
        )
        self._rank_preempted = config.rank_preempted_requests
        self._rank_per_admission = config.rank_per_admission
        self._step_rank_ns = 0
        self._step_rerank_count = 0
        # LPM that fell back to FCFS (no prefix caching) never skips.
        self._lpm_admission_skips = (
            config.lpm_admission_skips if self.policy == SchedulingPolicy.LPM else 0
        )
        self._lpm_bypass_counts: dict[str, int] = {}
        # Local prefix hit of a bypassed request when it was first and last
        # skipped (skip-aware eviction protects the last one).
        self._lpm_first_skip_hits: dict[str, int] = {}
        self._lpm_last_skip_hits: dict[str, int] = {}
        self._step_lpm_skips = 0
        self._step_lpm_scan_ns = 0
        self._lpm_scan_start_ns = 0
        # In-flight prefix deferral (SPF/SSPF): it waits for prefix-cache
        # hits, so it is off without prefix caching.
        self._defer_inflight = config.defer_inflight_prefixes
        if self._defer_inflight and not self.cache_config.enable_prefix_caching:
            logger.warning_once(
                "defer_inflight_prefixes requires prefix caching; disabled."
            )
            self._defer_inflight = False
        # Steps each waiting request was deferred, logged at admission.
        self._defer_counts: dict[str, int] = {}
        self._step_defers = 0
        rankers: dict[SchedulingPolicy, Callable[[list[Request]], RankPlan]] = {
            SchedulingPolicy.LPM: self._rank_lpm_candidates,
            SchedulingPolicy.CLPM: self._rank_clpm_candidates,
            SchedulingPolicy.SPF: self._rank_spf_candidates,
            SchedulingPolicy.SSPF: self._rank_sspf_candidates,
            SchedulingPolicy.SJF: self._rank_sjf_candidates,
        }
        self._waiting_ranker = rankers.get(self.policy)
        if self.policy in (SchedulingPolicy.SPF, SchedulingPolicy.SSPF):
            name = self.policy.value.upper()
            if not self.cache_config.enable_prefix_caching:
                logger.warning_once(
                    "%s without prefix caching sees no cache hits.", name
                )
            logger.info(
                "%s scheduling: age_alpha=%.3f tokens/s, window=%d, "
                "rank_preempted=%s, defer_inflight_prefixes=%s",
                name,
                self._spf_age_alpha,
                config.lpm_window_size,
                self._rank_preempted,
                self._defer_inflight,
            )
        elif self.policy == SchedulingPolicy.SJF:
            logger.info(
                "SJF scheduling: age_alpha=%.3f tokens/s, window=%d, "
                "rank_preempted=%s",
                self._sjf_age_alpha,
                config.lpm_window_size,
                self._rank_preempted,
            )
        if self._rank_per_admission:
            logger.info(
                "Waiting-queue ranking: re-rank after admissions that change "
                "waiting prefix hits (policy=%s).",
                self.policy.value,
            )
        if self._lpm_admission_skips:
            logger.info(
                "LPM admission skips: skip up to %d waiting requests that do "
                "not fit in free KV, admit the first that fits, then stop.",
                self._lpm_admission_skips,
            )

    def _can_skip_unfit_request(
        self,
        request: Request,
        request_queue: RequestQueue,
        skipped: list[Request],
    ) -> bool:
        """Whether LPM admission may skip a request that does not fit in KV.

        Only fresh requests at the head of ``self.waiting`` are skipped; a
        preempted request waiting to resume still stops admission.
        """
        return (
            len(skipped) < self._lpm_admission_skips
            and request_queue is self.waiting
            and request.status == RequestStatus.WAITING
            and request.num_computed_tokens == 0
        )

    def _skip_unfit_request(
        self,
        request_queue: RequestQueue,
        skipped: list[Request],
        num_local_hit_tokens: int,
    ) -> None:
        """Move the unfit head request aside for the rest of this step."""
        request = request_queue.pop_request()
        if not skipped:
            self._lpm_scan_start_ns = time.perf_counter_ns()
        skipped.append(request)
        self._lpm_first_skip_hits.setdefault(request.request_id, num_local_hit_tokens)
        self._lpm_last_skip_hits[request.request_id] = num_local_hit_tokens
        self._lpm_bypass_counts[request.request_id] = (
            self._lpm_bypass_counts.get(request.request_id, 0) + 1
        )
        self._step_lpm_skips += 1
        protectable = self._step_skip_protectable
        if (
            self._skip_eviction
            and num_local_hit_tokens >= self.block_size
            and len(protectable) < self._skip_eviction_max_protected
        ):
            # Protect now, so the backfill admitted later in this step cannot
            # evict the prefix of a request it bypassed. This step's skips
            # lead in ranked order, then the older protected requests.
            protectable.append(request)
            this_step = {r.request_id for r in protectable}
            self._set_skip_protected(
                protectable
                + [r for r in self._skip_protected if r.request_id not in this_step]
            )

    def _set_skip_protected(self, requests: list[Request]) -> None:
        """Publish the skip-protected waiting requests, most important first.
        Each protects the cached prefix it had when it was last skipped; a
        request without a full cached block takes no slot."""
        protected = [
            (r, self._lpm_last_skip_hits[r.request_id] // self.block_size)
            for r in requests
        ]
        protected = [(r, n) for r, n in protected if n > 0]
        protected = protected[: self._skip_eviction_max_protected]
        self._skip_protected = [r for r, _ in protected]
        self.kv_cache_manager.block_pool.set_skip_protected(protected)

    def _unprotect_skipped_request(self, request: Request) -> None:
        """A bypassed request was admitted or finished: release its prefix."""
        if self._skip_eviction:
            self._set_skip_protected(
                [r for r in self._skip_protected if r is not request]
            )

    def _refresh_skip_protection(self) -> None:
        """Protect the top bypassed waiting requests in the current LPM order."""
        protected: list[Request] = []
        hits = self._lpm_last_skip_hits
        candidates = sum(hit >= self.block_size for hit in hits.values())
        limit = min(self._skip_eviction_max_protected, candidates)
        for request in self.waiting:
            if len(protected) == limit:
                break
            if hits.get(request.request_id, 0) >= self.block_size:
                protected.append(request)
        self._set_skip_protected(protected)

    def _rerank_waiting_requests(self, scheduled_loras: set[int]) -> None:
        """Re-rank after an admission changed waiting requests' prefix hits."""
        start_ns = time.perf_counter_ns()
        self._reorder_waiting_requests(scheduled_loras)
        self._step_rank_ns += time.perf_counter_ns() - start_ns
        self._step_rerank_count += 1

    def _admission_changed_waiting_hits(
        self,
        request: Request,
        num_local_hit_tokens: int,
        num_scheduled_end: int,
    ) -> bool:
        """Whether admitting ``request`` changed a waiting request's prefix hits.

        Call after ``request`` left the pending index. Allocation caches the
        admitted request's new full blocks (raising hits of waiting requests
        that share them) and may evict cached blocks (lowering hits of
        waiting requests that use them). Block hashes are cumulative, so a
        waiting request that shares any newly cached block also shares the
        first one. The check is conservative: it may report a change that
        does not move any key, which only costs an extra re-rank.
        """
        index = self.pending_prefix_index
        block_pool = self.kv_cache_manager.block_pool
        if any(index.pending_count(h) for h in block_pool.drain_eviction_log()):
            return True
        block_size = block_pool.hash_block_size
        first_new = num_local_hit_tokens // block_size
        num_full = min(num_scheduled_end // block_size, len(request.block_hashes))
        if first_new >= num_full:
            return False
        return index.pending_count(request.block_hashes[first_new]) > 0

    def _reorder_waiting_requests(
        self,
        scheduled_loras: set[int],
    ) -> RankPlan | None:
        """Reorder rankable slots in one bounded waiting-queue prefix."""
        ranker = self._waiting_ranker
        if ranker is None:
            return None
        assert isinstance(self.waiting, FCFSRequestQueue)
        fallback_threshold = self.scheduler_config.lpm_fallback_threshold
        if fallback_threshold > 0 and len(self.waiting) > fallback_threshold:
            return None
        window_size = self.scheduler_config.lpm_window_size
        effective_window = len(self.waiting) if window_size == 0 else window_size
        waiting_window = list(self.waiting.iter_first_n(effective_window))
        candidates = [
            request
            for request in waiting_window
            if self._can_rank_waiting_request(request, scheduled_loras)
        ]
        if len(candidates) <= 1:
            return None

        plan = ranker(candidates)
        self.waiting.reorder_rankable_prefix(
            window_size,
            plan.ordered_requests,
            {request.request_id for request in candidates},
        )
        return plan

    def _prefix_hits(self, candidates: list[Request]) -> dict[str, int]:
        return {
            request.request_id: self.kv_cache_manager.get_computed_prefix_len(request)
            for request in candidates
        }

    def _rank_lpm_candidates(self, candidates: list[Request]) -> RankPlan:
        return rank_lpm(candidates, self._prefix_hits(candidates))

    def _rank_clpm_candidates(self, candidates: list[Request]) -> RankPlan:
        prefix_hits = self._prefix_hits(candidates)
        deprioritize_threshold = self.scheduler_config.lpm_deprioritize_threshold
        claim_depth_blocks = (
            (deprioritize_threshold + self.block_size - 1) // self.block_size
            if deprioritize_threshold > 0
            else 0
        )
        rank_info = self.pending_prefix_index.rank_info_many(
            [request.request_id for request in candidates],
            claim_depth_blocks,
            include_request_score=not self.scheduler_config.clpm_group_major,
        )
        if len(rank_info) != len(candidates):
            raise RuntimeError("Prefix index is inconsistent with waiting queue")

        if not any(info.cluster_size >= 2 for info in rank_info.values()):
            return rank_lpm(candidates, prefix_hits)
        plan = rank_clpm(
            candidates,
            prefix_hits,
            rank_info,
            check_threshold=self.scheduler_config.lpm_check_threshold,
            big_lane_share=self.scheduler_config.clpm_big_lane_share,
            age_alpha=self.scheduler_config.clpm_age_alpha,
            group_major=self.scheduler_config.clpm_group_major,
            dynamic_lane=self.scheduler_config.clpm_dynamic_lane,
            slo_budget_s=self.scheduler_config.clpm_slo_budget_s,
            previous_b_share=self._clpm_dynamic_b_share,
            lane_a_pass=self._clpm_lane_a_pass,
            lane_b_pass=self._clpm_lane_b_pass,
            now=time.time(),
        )
        if plan.dynamic_b_share is not None:
            self._clpm_dynamic_b_share = plan.dynamic_b_share
        return plan

    def _rank_spf_candidates(self, candidates: list[Request]) -> RankPlan:
        return rank_spf(candidates, self._prefix_hits(candidates), self._spf_age_alpha)

    def _rank_sspf_candidates(self, candidates: list[Request]) -> RankPlan:
        """Rank by uncached prompt tokens, each uncached full block divided by
        the number of rank-ready waiting requests that contain it; the
        trailing partial block counts in full."""
        index = self.pending_prefix_index
        block_size = self.kv_cache_manager.block_pool.hash_block_size
        prefix_hits = self._prefix_hits(candidates)
        costs: dict[str, float] = {}
        for request in candidates:
            request_id = request.request_id
            path = index.request_path(request_id)
            if path is None:
                raise RuntimeError("Prefix index is inconsistent with waiting queue")
            hit_blocks = prefix_hits[request_id] // block_size
            costs[request_id] = (
                block_size * index.amortized_blocks(request_id, hit_blocks)
                + request.num_tokens
                - len(path) * block_size
            )
        return rank_sspf(candidates, costs, self._spf_age_alpha)

    def _inflight_prefix_blocks(self) -> set[BlockHash]:
        """Block hashes running requests have not computed yet.

        ``num_computed_tokens`` advances only after scheduling, so this
        includes blocks being computed this step. Those are already in the
        prefix cache (blocks are cached at allocation), so they show up as
        hits, not as a waiting request's next uncached block.
        """
        block_size = self.kv_cache_manager.block_pool.hash_block_size
        blocks: set[BlockHash] = set()
        for request in self.running:
            first_uncomputed = request.num_computed_tokens // block_size
            blocks.update(request.block_hashes[first_uncomputed:])
        return blocks

    def _next_block_inflight(
        self,
        request: Request,
        num_local_hit_tokens: int,
        inflight_blocks: set[BlockHash],
    ) -> bool:
        """Whether a fresh request's next uncached block is in flight."""
        if (
            request.status != RequestStatus.WAITING
            or request.skip_reading_prefix_cache
            or not inflight_blocks
        ):
            return False
        block_hashes = request.block_hashes
        block_size = self.kv_cache_manager.block_pool.hash_block_size
        next_block = num_local_hit_tokens // block_size
        # A block holding the last prompt token is recomputed even when
        # cached (hits stop at num_tokens - 1): waiting for it gains nothing.
        if (next_block + 1) * block_size >= request.num_tokens:
            return False
        return (
            next_block < len(block_hashes)
            and block_hashes[next_block] in inflight_blocks
        )

    def _rank_sjf_candidates(self, candidates: list[Request]) -> RankPlan:
        # Ranks on max_tokens alone: no prefix-cache lookups.
        return rank_sjf(candidates, self._sjf_age_alpha)

    def _init_dspf(self) -> DSPFScheduler:
        """Check DSPF's scope and build its cost model and state."""
        model_config = self.vllm_config.model_config
        check_dspf_scope(
            **{
                "LoRA": self.lora_config is not None,
                "KV connectors": self.connector is not None,
                "encoder cache connectors": self.ec_connector is not None,
                "encoder-decoder models": self.is_encoder_decoder,
                "multimodal models": model_config.is_multimodal_model,
                "pipeline parallelism": self.parallel_config.pipeline_parallel_size
                > 1,
            }
        )
        window_size = self.scheduler_config.lpm_window_size
        if window_size > 0:
            logger.warning_once(
                "DSPF ranks only the first %d waiting requests; later ones get "
                "no deadline until they enter the window. Use "
                "--lpm-window-size 0 to rank the whole queue.",
                window_size,
            )
        return DSPFScheduler(
            PrefillCostShape.from_model_config(model_config),
            window_size,
            # Looked up on each call, so the manager can be created later.
            lambda request: self.kv_cache_manager.get_computed_prefix_len(request),
        )

    def _dspf_rank(self, token_budget: int, record_step: bool) -> DSPFRanking:
        assert self._dspf is not None and isinstance(self.waiting, FCFSRequestQueue)
        return self._dspf.rank(
            self.waiting,
            self.skipped_waiting,
            token_budget,
            self.max_num_running_reqs - len(self.running),
            record_step,
        )

    def _commit_clpm_admission(
        self, request_id: str, rank_plan: RankPlan | None
    ) -> None:
        if rank_plan is None:
            return
        credit = rank_plan.lane_credit.get(request_id)
        if credit is None:
            return
        if credit.lane == "a":
            self._clpm_lane_a_pass += credit.stride
        else:
            self._clpm_lane_b_pass += credit.stride

        min_pass = min(self._clpm_lane_a_pass, self._clpm_lane_b_pass)
        if min_pass > 1_000_000:
            self._clpm_lane_a_pass -= min_pass
            self._clpm_lane_b_pass -= min_pass

    def _predictive_block_ids(
        self, request_id: str, blocks: Sequence[KVCacheBlock]
    ) -> tuple[int, ...]:
        # Supported full-attention tables only append until free/preemption.
        cached = self._pkas_block_ids.get(request_id)
        if cached is None or cached[0] is not blocks:
            ids = tuple(block.block_id for block in blocks)
        else:
            ids = cached[1]
            if len(ids) != len(blocks):
                ids += tuple(block.block_id for block in blocks[len(ids) :])
        self._pkas_block_ids[request_id] = (blocks, ids)
        return ids

    def _predictive_pinned_blocks(self) -> int:
        pool = self.kv_cache_manager.block_pool
        return pool.num_gpu_blocks - 1 - pool.get_num_free_blocks()

    def _build_predictive_admission_state(
        self, num_scheduled_tokens: Mapping[str, int]
    ) -> _PredictiveAdmissionState | None:
        if self.predictive_admission is None:
            return None

        start_ns = time.perf_counter_ns()
        running: list[RequestSnapshot] = []
        for active in self.running:
            blocks = self.kv_cache_manager.get_blocks(active.request_id).blocks[0]
            block_ids = self._predictive_block_ids(active.request_id, blocks)
            running.append(
                RequestSnapshot(
                    request_id=active.request_id,
                    prompt_tokens=active.num_prompt_tokens,
                    computed_tokens=(
                        active.num_computed_tokens
                        + num_scheduled_tokens.get(active.request_id, 0)
                    ),
                    output_tokens=active.num_output_tokens,
                    block_ids=block_ids,
                )
            )
        state = _PredictiveAdmissionState(
            running=running,
            block_ref_counts=self._pkas_ref_counts,
            pinned_block_count=self._predictive_pinned_blocks(),
        )
        assert self._pkas_step_stats is not None
        self._pkas_step_stats["snapshot_ns"] += time.perf_counter_ns() - start_ns
        return state

    def _update_predictive_admission_state(
        self,
        state: _PredictiveAdmissionState | None,
        request: Request,
        computed_tokens: int,
        request_blocks: KVCacheBlocks,
    ) -> None:
        if state is None:
            return

        blocks = request_blocks.blocks[0]
        block_ids = self._predictive_block_ids(request.request_id, blocks)
        state.pinned_block_count = self._predictive_pinned_blocks()
        state.running.append(
            RequestSnapshot(
                request_id=request.request_id,
                prompt_tokens=request.num_prompt_tokens,
                computed_tokens=computed_tokens,
                output_tokens=request.num_output_tokens,
                block_ids=block_ids,
            )
        )

    def _predictive_admission_allows(
        self,
        state: _PredictiveAdmissionState | None,
        request: Request,
        computed_blocks: KVCacheBlocks,
        num_computed_tokens: int,
        num_new_tokens: int,
    ) -> bool:
        controller = self.predictive_admission
        if controller is None:
            return True
        assert state is not None
        assert self._pkas_step_stats is not None

        cached = self._pkas_reject_cache.get(request.request_id)
        if cached is not None:
            epoch, expires = cached
            if epoch == self._pkas_epoch and self._step_id <= expires:
                self._pkas_step_stats["cached_rejections"] += 1
                self._pkas_step_stats["rejections"] += 1
                return False
            del self._pkas_reject_cache[request.request_id]

        hit_blocks = computed_blocks.blocks[0]
        hit_ids = tuple(block.block_id for block in hit_blocks if not block.is_null)
        start_ns = time.perf_counter_ns()
        decision = controller.evaluate(
            state.running,
            RequestSnapshot(
                request_id=request.request_id,
                prompt_tokens=request.num_prompt_tokens,
                computed_tokens=num_computed_tokens + num_new_tokens,
                output_tokens=request.num_output_tokens,
                block_ids=(),
            ),
            candidate_hit_block_ids=hit_ids,
            block_ref_counts=state.block_ref_counts,
            initial_pinned_blocks=state.pinned_block_count,
        )
        stats = self._pkas_step_stats
        stats["evaluate_ns"] += time.perf_counter_ns() - start_ns
        stats["evaluations"] += 1
        stats["bound_accepts"] += int(decision.peak_is_upper_bound)
        stats["simulations"] += int(not decision.peak_is_upper_bound)
        stats["rejections"] += int(not decision.feasible)
        cache_steps = self.scheduler_config.pkas_reject_cache_steps
        if not decision.feasible and cache_steps > 0:
            self._pkas_reject_cache[request.request_id] = (
                self._pkas_epoch,
                self._step_id + cache_steps,
            )
        logger.debug(
            "predictive_admission: request_id=%s feasible=%s "
            "predicted_output_tokens=%d prefix_hit_blocks=%d "
            "peak_pinned_blocks=%d capacity_blocks=%d peak_is_upper_bound=%s",
            request.request_id,
            decision.feasible,
            decision.predicted_output_tokens,
            decision.prefix_hit_blocks,
            decision.peak_pinned_blocks,
            decision.capacity_blocks,
            decision.peak_is_upper_bound,
        )
        if decision.feasible:
            # Reported on the request's admitted_prefix_cache line, and on
            # its preempted line if PKAS's forecast turns out short.
            self._pkas_last_gate_target[request.request_id] = (
                controller.estimator.target(request.num_output_tokens)
            )
        return decision.feasible

    def _predictive_admission_allows_repo(
        self,
        state: _PredictiveAdmissionState | None,
        request: Request,
        computed_blocks: KVCacheBlocks,
        num_computed_tokens: int,
        num_new_tokens: int,
    ) -> bool:
        """Copy of ``_predictive_admission_allows`` that runs the PKAS
        repository simulation (``pkas_simulation="repo"``)."""
        controller = self.predictive_admission
        if controller is None:
            return True
        assert state is not None
        assert self._pkas_step_stats is not None

        cached = self._pkas_reject_cache.get(request.request_id)
        if cached is not None:
            epoch, expires = cached
            if epoch == self._pkas_epoch and self._step_id <= expires:
                self._pkas_step_stats["cached_rejections"] += 1
                self._pkas_step_stats["rejections"] += 1
                return False
            del self._pkas_reject_cache[request.request_id]

        hit_blocks = computed_blocks.blocks[0]
        hit_ids = tuple(block.block_id for block in hit_blocks if not block.is_null)
        start_ns = time.perf_counter_ns()
        decision = controller.evaluate_repo(
            state.running,
            RequestSnapshot(
                request_id=request.request_id,
                prompt_tokens=request.num_prompt_tokens,
                computed_tokens=num_computed_tokens + num_new_tokens,
                output_tokens=request.num_output_tokens,
                block_ids=(),
            ),
            candidate_hit_block_ids=hit_ids,
            block_ref_counts=state.block_ref_counts,
            initial_pinned_blocks=state.pinned_block_count,
        )
        stats = self._pkas_step_stats
        stats["evaluate_ns"] += time.perf_counter_ns() - start_ns
        stats["evaluations"] += 1
        stats["bound_accepts"] += int(decision.peak_is_upper_bound)
        stats["simulations"] += int(not decision.peak_is_upper_bound)
        stats["rejections"] += int(not decision.feasible)
        cache_steps = self.scheduler_config.pkas_reject_cache_steps
        if not decision.feasible and cache_steps > 0:
            self._pkas_reject_cache[request.request_id] = (
                self._pkas_epoch,
                self._step_id + cache_steps,
            )
        logger.debug(
            "predictive_admission: request_id=%s feasible=%s "
            "predicted_output_tokens=%d prefix_hit_blocks=%d "
            "peak_pinned_blocks=%d capacity_blocks=%d peak_is_upper_bound=%s",
            request.request_id,
            decision.feasible,
            decision.predicted_output_tokens,
            decision.prefix_hit_blocks,
            decision.peak_pinned_blocks,
            decision.capacity_blocks,
            decision.peak_is_upper_bound,
        )
        if decision.feasible:
            # The repository reserves a full forecast past the tokens so far.
            self._pkas_last_gate_target[request.request_id] = (
                request.num_output_tokens + controller.estimator.predict()
            )
        return decision.feasible

    def _pkas_repo_clamp_to_free_kv(self, num_new_tokens: int) -> int:
        """PKAS repository behaviour: shrink a running request's chunk to the
        free KV (evictable cached blocks count as free), so a long prefill
        advances by a smaller chunk instead of preempting. With no free block
        the chunk is left as is and allocation preempts as usual."""
        pool = self.kv_cache_manager.block_pool
        free_tokens = pool.get_num_free_blocks() * self.block_size
        if free_tokens > 0:
            return min(num_new_tokens, free_tokens)
        return num_new_tokens

    def schedule(self) -> SchedulerOutput:
        if self._pending_batch_logs:
            # Overlapping execution cannot use the synchronous accounting path.
            self._preemption_cost_supported = False
            for record, _ in self._pending_batch_logs.values():
                record["preempted_tokens"] = -1
                self._log_schedule_batch(record, -1)
            self._pending_batch_logs.clear()
        self._step_preempted_tokens = 0
        self._step_rank_ns = 0
        if self._dspf is not None:
            self._dspf.start_step()
        self._step_rerank_count = 0
        self._step_lpm_skips = 0
        self._step_lpm_scan_ns = 0
        self._step_defers = 0
        self._step_skip_protectable = []
        self._step_id += 1
        self._step_evicted_blocks_start = (
            self.kv_cache_manager.block_pool.num_evicted_cached_blocks
        )
        self._step_skip_protected_evictions_start = (
            self.kv_cache_manager.block_pool.num_skip_protected_evictions
        )
        self._pkas_step_stats = (
            dict.fromkeys(
                (
                    "evaluations",
                    "rejections",
                    "cached_rejections",
                    "bound_accepts",
                    "simulations",
                    "recovery_stops",
                    "blocked_budget",
                    "snapshot_ns",
                    "evaluate_ns",
                ),
                0,
            )
            if self.predictive_admission is not None
            else None
        )
        diagnostics = self.preemption_diagnostics
        if diagnostics is not None:
            diagnostics.begin_step()
        survival = self.survival_diagnostics
        if survival is not None:
            survival.begin_step()
        # NOTE(woosuk) on the scheduling algorithm:
        # There's no "decoding phase" nor "prefill phase" in the scheduler.
        # Each request just has the num_computed_tokens and
        # num_tokens_with_spec. num_tokens_with_spec =
        # len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids).
        # At each step, the scheduler tries to assign tokens to the requests
        # so that each request's num_computed_tokens can catch up its
        # num_tokens_with_spec. This is general enough to cover
        # chunked prefills, prefix caching, speculative decoding,
        # and the "jump decoding" optimization in the future.

        scheduled_new_reqs: list[Request] = []
        scheduled_resumed_reqs: list[Request] = []
        scheduled_running_reqs: list[Request] = []
        preempted_reqs: list[Request] = []

        req_to_new_blocks: dict[str, KVCacheBlocks] = {}
        num_scheduled_tokens: dict[str, int] = {}
        token_budget = self.max_num_scheduled_tokens
        if self._pause_state == PauseState.PAUSED_ALL:
            # Do not schedule any requests when paused.
            token_budget = 0

        # Encoder-related.
        scheduled_encoder_inputs: dict[str, list[int]] = {}
        encoder_compute_budget = self.max_num_encoder_input_tokens
        # Spec decode-related.
        scheduled_spec_decode_tokens: dict[str, list[int]] = {}

        # For logging.
        scheduled_timestamp = time.monotonic()

        self.kv_cache_manager.new_step_starts()

        # First, schedule the RUNNING requests.
        req_index = 0
        while req_index < len(self.running) and token_budget > 0:
            request = self.running[req_index]

            if (
                request.num_output_placeholders > 0
                # This is (num_computed_tokens + 1) - (num_output_placeholders - 1).
                # Since output placeholders are also included in the computed tokens
                # count, we subtract (num_output_placeholders - 1) to remove any draft
                # tokens, so that we can be sure no further steps are needed even if
                # they are all rejected.
                and request.num_computed_tokens + 2 - request.num_output_placeholders
                >= request.num_prompt_tokens + request.max_tokens
            ):
                # Async scheduling: Avoid scheduling an extra step when we are sure that
                # the previous step has reached request.max_tokens. We don't schedule
                # partial draft tokens since this prevents uniform decode optimizations.
                req_index += 1
                continue

            num_new_tokens = (
                request.num_tokens_with_spec
                + request.num_output_placeholders
                - request.num_computed_tokens
            )
            if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
                num_new_tokens = self.scheduler_config.long_prefill_token_threshold
            num_new_tokens = min(num_new_tokens, token_budget)

            # Make sure the input position does not exceed the max model len.
            # This is necessary when using spec decoding.
            num_new_tokens = min(
                num_new_tokens, self.max_model_len - 1 - request.num_computed_tokens
            )

            # Schedule encoder inputs.
            encoder_inputs_to_schedule = None
            external_load_encoder_input: list[int] = []
            new_encoder_compute_budget = encoder_compute_budget
            if request.has_encoder_inputs:
                (
                    encoder_inputs_to_schedule,
                    num_new_tokens,
                    new_encoder_compute_budget,
                    external_load_encoder_input,
                ) = self._try_schedule_encoder_inputs(
                    request,
                    request.num_computed_tokens,
                    num_new_tokens,
                    encoder_compute_budget,
                    shift_computed_tokens=1 if self.use_eagle else 0,
                )

            if self.need_mamba_block_aligned_split:
                num_new_tokens = self._mamba_block_aligned_split(
                    request, num_new_tokens
                )

            if num_new_tokens == 0:
                # The request cannot be scheduled because one of the following
                # reasons:
                # 1. No new tokens to schedule. This may happen when
                #    (1) PP>1 and we have already scheduled all prompt tokens
                #    but they are not finished yet.
                #    (2) Async scheduling and the request has reached to either
                #    its max_total_tokens or max_model_len.
                # 2. The encoder budget is exhausted.
                # 3. The encoder cache is exhausted.
                # 4. Insufficient budget for a block-aligned chunk in hybrid
                #    models with mamba cache mode \"align\".
                # NOTE(woosuk): Here, by doing `continue` instead of `break`,
                # we do not strictly follow the FCFS scheduling policy and
                # allow the lower-priority requests to be scheduled.
                req_index += 1
                continue

            if self._pkas_repo:
                num_new_tokens = self._pkas_repo_clamp_to_free_kv(num_new_tokens)

            # Schedule newly needed KV blocks for the request.
            with record_function_or_nullcontext("schedule: allocate_slots"):
                while True:
                    new_blocks = self.kv_cache_manager.allocate_slots(
                        request,
                        num_new_tokens,
                        num_lookahead_tokens=self.num_lookahead_tokens,
                        failure_observer=(
                            diagnostics.allocation_failed if diagnostics else None
                        ),
                    )

                    if new_blocks is not None:
                        # The request can be scheduled.
                        if diagnostics is not None:
                            diagnostics.end_episode(proceeded=True)
                        break

                    if diagnostics is not None:
                        diagnostics.start_episode(
                            self, request, num_new_tokens, num_scheduled_tokens
                        )
                    # The request cannot be scheduled.
                    # Preempt the lowest-priority request.
                    if self.policy == SchedulingPolicy.PRIORITY:
                        preempted_req = max(
                            self.running,
                            key=lambda r: (r.priority, r.arrival_time),
                        )
                        self.running.remove(preempted_req)
                        if preempted_req in scheduled_running_reqs:
                            preempted_req_id = preempted_req.request_id
                            scheduled_running_reqs.remove(preempted_req)
                            token_budget += num_scheduled_tokens.pop(preempted_req_id)
                            req_to_new_blocks.pop(preempted_req_id)
                            scheduled_spec_decode_tokens.pop(preempted_req_id, None)
                            preempted_encoder_inputs = scheduled_encoder_inputs.pop(
                                preempted_req_id, None
                            )
                            if preempted_encoder_inputs:
                                # Restore encoder compute budget if the preempted
                                # request had encoder inputs scheduled in this step.
                                num_embeds_to_restore = sum(
                                    preempted_req.get_num_encoder_embeds(i)
                                    for i in preempted_encoder_inputs
                                )
                                encoder_compute_budget += num_embeds_to_restore
                            req_index -= 1
                    else:
                        preempted_req = self.running.pop()

                    self._preempt_request(preempted_req, scheduled_timestamp)
                    preempted_reqs.append(preempted_req)
                    if preempted_req == request:
                        # No more request to preempt. Cannot schedule this request.
                        if diagnostics is not None:
                            diagnostics.end_episode(proceeded=False)
                        break

            if new_blocks is None:
                # Cannot schedule this request.
                break

            # Schedule the request.
            scheduled_running_reqs.append(request)
            request_id = request.request_id
            req_to_new_blocks[request_id] = new_blocks
            num_scheduled_tokens[request_id] = num_new_tokens
            token_budget -= num_new_tokens
            req_index += 1

            # Speculative decode related.
            if request.spec_token_ids:
                num_scheduled_spec_tokens = (
                    num_new_tokens
                    + request.num_computed_tokens
                    - request.num_tokens
                    - request.num_output_placeholders
                )
                if num_scheduled_spec_tokens > 0:
                    spec_token_ids = request.spec_token_ids
                    if len(spec_token_ids) > num_scheduled_spec_tokens:
                        spec_token_ids = spec_token_ids[:num_scheduled_spec_tokens]
                    scheduled_spec_decode_tokens[request.request_id] = spec_token_ids

                # New spec tokens will be set in `update_draft_token_ids` before the
                # next step when applicable.
                request.spec_token_ids = []

            # Encoder-related.
            if encoder_inputs_to_schedule:
                scheduled_encoder_inputs[request_id] = encoder_inputs_to_schedule
                # Allocate the encoder cache.
                for i in encoder_inputs_to_schedule:
                    self.encoder_cache_manager.allocate(request, i)
                    if self.ec_connector is not None:
                        self.ec_connector.update_state_after_alloc(request, i)
                encoder_compute_budget = new_encoder_compute_budget
            if external_load_encoder_input:
                for i in external_load_encoder_input:
                    self.encoder_cache_manager.allocate(request, i)
                    if self.ec_connector is not None:
                        self.ec_connector.update_state_after_alloc(request, i)

        # Record the LoRAs in scheduled_running_reqs
        scheduled_loras: set[int] = set()
        if self.lora_config:
            scheduled_loras = set(
                req.lora_request.lora_int_id
                for req in scheduled_running_reqs
                if req.lora_request and req.lora_request.lora_int_id > 0
            )
            assert len(scheduled_loras) <= self.lora_config.max_loras

        prefix_rank_plan: RankPlan | None = None

        # Next, schedule the WAITING requests.
        if not preempted_reqs and self._pause_state == PauseState.UNPAUSED:
            step_skipped_waiting = create_request_queue(self.policy)
            # Ranked policies (LPM, cLPM, SPF, SJF): reorder the bounded
            # rankable prefix of self.waiting once per step (LPM and SPF may
            # re-rank between admissions); admission below still pops from
            # the queue head.
            rank_start_ns = time.perf_counter_ns()
            if self._dspf is not None:
                self._dspf_rank(token_budget, record_step=True)
            else:
                prefix_rank_plan = self._reorder_waiting_requests(scheduled_loras)
            self._step_rank_ns = time.perf_counter_ns() - rank_start_ns
            if self._skip_eviction and self._lpm_bypass_counts:
                self._refresh_skip_protection()
            predictive_admission_state = None
            if (
                (self.waiting or self.skipped_waiting)
                and token_budget > 0
                and len(self.running) < self.max_num_running_reqs
            ):
                predictive_admission_state = self._build_predictive_admission_state(
                    num_scheduled_tokens
                )
            # Per-admission re-ranking: re-rank lazily, only before the next
            # admission attempt, so a step that admits once never re-ranks.
            rerank_pending = False
            if self._rank_per_admission:
                self.kv_cache_manager.block_pool.start_eviction_log()
            # LPM admission skips: unfit requests set aside this step, in
            # ranked order. One admission after a skip ends the step.
            lpm_skipped: list[Request] = []
            # In-flight prefix deferral: requests held back this step, in
            # ranked order, and the blocks running requests have yet to
            # compute (built at the first check).
            deferred: list[Request] = []
            inflight_blocks: set[BlockHash] | None = None
            while (self.waiting or self.skipped_waiting) and token_budget > 0:
                if len(self.running) == self.max_num_running_reqs:
                    break
                if rerank_pending:
                    rerank_pending = False
                    self._rerank_waiting_requests(scheduled_loras)

                request_queue = self._select_waiting_queue_for_scheduling()
                assert request_queue is not None

                request = request_queue.peek_request()

                request_id = request.request_id

                # try to promote blocked statuses while traversing skipped queue.
                if self._is_blocked_waiting_status(
                    request.status
                ) and not self._try_promote_blocked_waiting_request(request):
                    if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                        logger.debug(
                            "%s is still in WAITING_FOR_REMOTE_KVS state.",
                            request_id,
                        )
                    request_queue.pop_request()
                    step_skipped_waiting.prepend_request(request)
                    continue

                # Check that adding the request still respects the max_loras
                # constraint.
                if (
                    self.lora_config
                    and request.lora_request
                    and (
                        len(scheduled_loras) == self.lora_config.max_loras
                        and request.lora_request.lora_int_id not in scheduled_loras
                    )
                ):
                    # Scheduling would exceed max_loras, skip.
                    request_queue.pop_request()
                    step_skipped_waiting.prepend_request(request)
                    continue

                num_external_computed_tokens = 0
                load_kv_async = False
                connector_prefix_cache_queries, connector_prefix_cache_hits = 0, 0

                # Get already-cached tokens.
                if request.num_computed_tokens == 0:
                    # Get locally-cached tokens.
                    new_computed_blocks, num_new_local_computed_tokens = (
                        self.kv_cache_manager.get_computed_blocks(request)
                    )
                    if self._dspf is not None and self._dspf.cost_grew(
                        request, num_new_local_computed_tokens
                    ):
                        # An earlier admission this step evicted its blocks:
                        # re-rank with its real cost and retry the head.
                        rerank_start_ns = time.perf_counter_ns()
                        self._dspf_rank(token_budget, record_step=False)
                        self._step_rank_ns += time.perf_counter_ns() - rerank_start_ns
                        self._step_rerank_count += 1
                        continue

                    if self._defer_inflight and request_queue is self.waiting:
                        if inflight_blocks is None:
                            inflight_blocks = self._inflight_prefix_blocks()
                        if self._next_block_inflight(
                            request, num_new_local_computed_tokens, inflight_blocks
                        ):
                            # Wait for the running request to cache it.
                            request_queue.pop_request()
                            deferred.append(request)
                            self._defer_counts[request_id] = (
                                self._defer_counts.get(request_id, 0) + 1
                            )
                            self._step_defers += 1
                            continue

                    # Get externally-cached tokens if using a KVConnector.
                    if self.connector is not None:
                        ext_tokens, load_kv_async = (
                            self.connector.get_num_new_matched_tokens(
                                request, num_new_local_computed_tokens
                            )
                        )

                        if ext_tokens is None:
                            # The request cannot be scheduled because
                            # the KVConnector couldn't determine
                            # the number of matched tokens.
                            request_queue.pop_request()
                            step_skipped_waiting.prepend_request(request)
                            continue

                        num_external_computed_tokens = ext_tokens

                        connector_prefix_cache_queries = (
                            request.num_tokens - num_new_local_computed_tokens
                        )
                        connector_prefix_cache_hits = num_external_computed_tokens

                    # Total computed tokens (local + external).
                    num_computed_tokens = (
                        num_new_local_computed_tokens + num_external_computed_tokens
                    )
                    assert num_computed_tokens <= request.num_tokens

                    # Track first scheduled prefill, not post-preemption repeat prefills
                    if request.prefill_stats is not None:
                        assert num_computed_tokens <= request.num_prompt_tokens
                        request.prefill_stats.set(
                            num_prompt_tokens=request.num_prompt_tokens,
                            num_local_cached_tokens=num_new_local_computed_tokens,
                            num_external_cached_tokens=num_external_computed_tokens,
                        )
                        if (request_logger := RequestEventLogger.get()) is not None:
                            request_logger.record_prefill(
                                request.request_id,
                                num_computed_tokens,
                                num_external_computed_tokens,
                            )
                else:
                    # KVTransfer: WAITING reqs have num_computed_tokens > 0
                    # after async KV recvs are completed.
                    new_computed_blocks = self.kv_cache_manager.empty_kv_cache_blocks
                    num_new_local_computed_tokens = 0
                    num_computed_tokens = request.num_computed_tokens

                encoder_inputs_to_schedule = None
                external_load_encoder_input = []
                new_encoder_compute_budget = encoder_compute_budget

                if load_kv_async:
                    # KVTransfer: loading remote KV, do not allocate for new work.
                    assert num_external_computed_tokens > 0
                    num_new_tokens = 0
                else:
                    # Number of tokens to be scheduled.
                    # We use `request.num_tokens` instead of
                    # `request.num_prompt_tokens` to consider the resumed
                    # requests, which have output tokens.
                    num_new_tokens = request.num_tokens - num_computed_tokens
                    threshold = self.scheduler_config.long_prefill_token_threshold
                    if 0 < threshold < num_new_tokens:
                        num_new_tokens = threshold

                    # chunked prefill has to be enabled explicitly to allow
                    # pooling requests to be chunked
                    if (
                        not self.scheduler_config.enable_chunked_prefill
                        and num_new_tokens > token_budget
                    ):
                        # If chunked_prefill is disabled,
                        # we can stop the scheduling here.
                        break

                    num_new_tokens = min(num_new_tokens, token_budget)
                    assert num_new_tokens > 0

                    # Schedule encoder inputs.
                    if request.has_encoder_inputs:
                        (
                            encoder_inputs_to_schedule,
                            num_new_tokens,
                            new_encoder_compute_budget,
                            external_load_encoder_input,
                        ) = self._try_schedule_encoder_inputs(
                            request,
                            num_computed_tokens,
                            num_new_tokens,
                            encoder_compute_budget,
                            shift_computed_tokens=1 if self.use_eagle else 0,
                        )
                        if num_new_tokens == 0:
                            # The request cannot be scheduled.
                            break

                if self.need_mamba_block_aligned_split:
                    num_new_tokens = self._mamba_block_aligned_split(
                        request,
                        num_new_tokens,
                        num_new_local_computed_tokens,
                        num_external_computed_tokens,
                    )
                    if num_new_tokens == 0:
                        break

                # Determine if we need to allocate cross-attention blocks.
                num_encoder_tokens = 0
                if (
                    self.is_encoder_decoder
                    and request.has_encoder_inputs
                    and encoder_inputs_to_schedule
                ):
                    num_encoder_tokens = sum(
                        request.get_num_encoder_embeds(i)
                        for i in encoder_inputs_to_schedule
                    )

                # Full-ISL admission check (upstream scheduler_reserve_full_isl).
                # PKAS rejects this setting: its simulation already reserves
                # the rest of the prompt and the predicted decode.
                if self.scheduler_reserve_full_isl:
                    fits, required_blocks, free_blocks = (
                        self.kv_cache_manager.full_sequence_fits(
                            request,
                            num_new_computed_tokens=num_new_local_computed_tokens,
                            new_computed_blocks=new_computed_blocks,
                            num_external_computed_tokens=num_external_computed_tokens,
                            num_encoder_tokens=num_encoder_tokens,
                        )
                    )
                    if not fits:
                        if diagnostics is not None:
                            diagnostics.admission_failed(
                                request_id, required_blocks, free_blocks
                            )
                        if request.has_encoder_inputs:
                            self.encoder_cache_manager.free(request)
                        if self._can_skip_unfit_request(
                            request, request_queue, lpm_skipped
                        ):
                            self._skip_unfit_request(
                                request_queue,
                                lpm_skipped,
                                num_new_local_computed_tokens,
                            )
                            continue
                        # Upstream behaviour: stop admitting this step.
                        break

                admission_allows = (
                    self._predictive_admission_allows_repo
                    if self._pkas_repo
                    else self._predictive_admission_allows
                )
                if not admission_allows(
                    predictive_admission_state,
                    request,
                    new_computed_blocks,
                    num_computed_tokens,
                    num_new_tokens,
                ):
                    # Release encoder-cache references taken while checking
                    # this request, as the allocation-failure path does.
                    if request.has_encoder_inputs:
                        self.encoder_cache_manager.free(request)
                    stats = self._pkas_step_stats
                    assert stats is not None
                    if request.num_preemptions > 0 and num_computed_tokens > 0:
                        # A preempted request is waiting to resume and PKAS
                        # refused it: admission stops behind it.
                        stats["recovery_stops"] += 1
                    stats["blocked_budget"] = token_budget
                    break

                # Handles an edge case when P/D Disaggregation
                # is used with Spec Decoding where an
                # extra block gets allocated which
                # creates a mismatch between the number
                # of local and remote blocks.
                effective_lookahead_tokens = (
                    0 if request.num_computed_tokens == 0 else self.num_lookahead_tokens
                )

                new_blocks = self.kv_cache_manager.allocate_slots(
                    request,
                    num_new_tokens,
                    num_new_computed_tokens=num_new_local_computed_tokens,
                    new_computed_blocks=new_computed_blocks,
                    num_lookahead_tokens=effective_lookahead_tokens,
                    num_external_computed_tokens=num_external_computed_tokens,
                    delay_cache_blocks=load_kv_async,
                    num_encoder_tokens=num_encoder_tokens,
                    # The full-ISL check already ran above, before PKAS.
                    failure_observer=(
                        diagnostics.admission_failed if diagnostics else None
                    ),
                )

                if new_blocks is None:
                    # The request cannot be scheduled.

                    # NOTE: we need to untouch the request from the encode cache
                    # manager
                    if request.has_encoder_inputs:
                        self.encoder_cache_manager.free(request)
                    if self._can_skip_unfit_request(
                        request, request_queue, lpm_skipped
                    ):
                        self._skip_unfit_request(
                            request_queue, lpm_skipped, num_new_local_computed_tokens
                        )
                        continue
                    break

                # KVTransfer: the connector uses this info to determine
                # if a load is needed. Note that
                # This information is used to determine if a load is
                # needed for this request.
                if self.connector is not None:
                    self.connector.update_state_after_alloc(
                        request,
                        self.kv_cache_manager.get_blocks(request_id),
                        num_external_computed_tokens,
                    )
                    if (
                        self.connector_prefix_cache_stats is not None
                        and connector_prefix_cache_queries != 0
                    ):
                        self.connector_prefix_cache_stats.record(
                            num_tokens=connector_prefix_cache_queries,
                            num_hits=connector_prefix_cache_hits,
                            preempted=request.num_preemptions > 0,
                        )

                # Remove the request only after allocation succeeds. Prefix
                # policies have already reordered the bounded queue prefix.
                request = request_queue.pop_request()
                self._pkas_reject_cache.pop(request.request_id, None)
                lpm_bypassed = self._lpm_bypass_counts.pop(request.request_id, 0)
                lpm_first_skip_hit = self._lpm_first_skip_hits.pop(
                    request.request_id, 0
                )
                self._lpm_last_skip_hits.pop(request.request_id, None)
                deferred_steps = self._defer_counts.pop(request.request_id, 0)
                if lpm_bypassed:
                    # Its hit blocks are allocated now; drop the protection.
                    self._unprotect_skipped_request(request)
                self._pkas_epoch += 1
                self._untrack_waiting_request(request)
                if self._rank_per_admission:
                    rerank_pending = (
                        self._admission_changed_waiting_hits(
                            request,
                            num_new_local_computed_tokens,
                            num_computed_tokens + num_new_tokens,
                        )
                        or rerank_pending
                    )
                if load_kv_async:
                    # If loading async, allocate memory and put request
                    # into the WAITING_FOR_REMOTE_KV state.
                    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
                    step_skipped_waiting.prepend_request(request)
                    # Set num_computed_tokens even though KVs are not yet loaded.
                    # request.num_computed_tokens will not be used anywhere until
                    # the request finished the KV transfer.
                    #
                    # If a transfer error is reported by the connector,
                    # request.num_computed_tokens will be re-set accordingly in
                    # _update_requests_with_invalid_blocks.
                    #
                    # When the transfer is finished, either successfully or not,
                    # request.num_computed_tokens will correctly reflect the number
                    # of computed tokens.
                    # _update_waiting_for_remote_kv will then cache
                    # only the successfully loaded tokens.
                    request.num_computed_tokens = num_computed_tokens
                    # Still pending (not ready) until the remote KVs arrive.
                    self._track_waiting_request(request)
                    if lpm_skipped:
                        break
                    continue

                self._commit_clpm_admission(request.request_id, prefix_rank_plan)
                dspf_fields = (
                    self._dspf.admit(request, num_computed_tokens)
                    if self._dspf is not None
                    else ""
                )

                self.kv_cache_manager.record_local_prefix_cache_hit(
                    request,
                    num_new_local_computed_tokens,
                )
                # A resumed request (prompt_tokens includes its output so far)
                # appears again; its first line is the original admission.
                pkas_target = self._pkas_last_gate_target.get(request.request_id)
                logger.info(
                    "admitted_prefix_cache: step=%d request_id=%s prompt_tokens=%d "
                    "local_hit_tokens=%d num_new_tokens=%d waiting_after_pop=%d "
                    "running=%d%s%s%s%s",
                    self._step_id,
                    request.request_id,
                    request.num_tokens,
                    num_new_local_computed_tokens,
                    num_new_tokens,
                    len(self.waiting) + len(lpm_skipped) + len(deferred),
                    len(self.running),
                    "" if pkas_target is None
                    else f" pkas_target_output_tokens={pkas_target}",
                    f" lpm_bypassed={lpm_bypassed}"
                    f" lpm_first_skip_hit={lpm_first_skip_hit}"
                    if lpm_bypassed
                    else "",
                    dspf_fields,
                    f" deferred_steps={deferred_steps}" if deferred_steps else "",
                )

                self.running.append(request)
                if inflight_blocks is not None:
                    # Its uncached blocks are in flight for later requests.
                    hash_block_size = self.kv_cache_manager.block_pool.hash_block_size
                    inflight_blocks.update(
                        request.block_hashes[num_computed_tokens // hash_block_size :]
                    )
                if survival is not None:
                    survival.admitted(request, num_new_local_computed_tokens)
                if self.log_stats:
                    request.record_event(
                        EngineCoreEventType.SCHEDULED, scheduled_timestamp
                    )
                if request.status == RequestStatus.WAITING:
                    scheduled_new_reqs.append(request)
                elif request.status == RequestStatus.PREEMPTED:
                    scheduled_resumed_reqs.append(request)
                    if diagnostics is not None:
                        diagnostics.resumed(request, num_new_local_computed_tokens)
                else:
                    raise RuntimeError(f"Invalid request status: {request.status}")

                if self.lora_config and request.lora_request:
                    scheduled_loras.add(request.lora_request.lora_int_id)
                req_to_new_blocks[request_id] = self.kv_cache_manager.get_blocks(
                    request_id
                )
                num_scheduled_tokens[request_id] = num_new_tokens
                token_budget -= num_new_tokens
                request.status = RequestStatus.RUNNING
                request.num_computed_tokens = num_computed_tokens
                self._update_predictive_admission_state(
                    predictive_admission_state,
                    request,
                    num_computed_tokens + num_new_tokens,
                    req_to_new_blocks[request.request_id],
                )
                # Encoder-related.
                if encoder_inputs_to_schedule:
                    scheduled_encoder_inputs[request_id] = encoder_inputs_to_schedule
                    # Allocate the encoder cache.
                    for i in encoder_inputs_to_schedule:
                        self.encoder_cache_manager.allocate(request, i)
                        if self.ec_connector is not None:
                            self.ec_connector.update_state_after_alloc(request, i)
                    encoder_compute_budget = new_encoder_compute_budget
                # Allocate for external load encoder cache
                if external_load_encoder_input:
                    for i in external_load_encoder_input:
                        self.encoder_cache_manager.allocate(request, i)
                        if self.ec_connector is not None:
                            self.ec_connector.update_state_after_alloc(request, i)
                if lpm_skipped:
                    # Admitted past unfit requests: stop admitting this step.
                    break

            if self._rank_per_admission:
                self.kv_cache_manager.block_pool.stop_eviction_log()

            # Return the deferred requests to the head in ranked order.
            for deferred_request in reversed(deferred):
                self.waiting.prepend_request(deferred_request)

            if lpm_skipped:
                # Return the skipped requests to the head in ranked order.
                self._step_lpm_scan_ns = (
                    time.perf_counter_ns() - self._lpm_scan_start_ns
                )
                for skipped_request in reversed(lpm_skipped):
                    self.waiting.prepend_request(skipped_request)

            # re-queue requests skipped in this pass ahead of older skipped items.
            if step_skipped_waiting:
                self.skipped_waiting.prepend_requests(step_skipped_waiting)

        # Check if the scheduling constraints are satisfied.
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens

        assert token_budget >= 0
        assert len(self.running) <= self.max_num_running_reqs
        # Since some requests in the RUNNING queue may not be scheduled in
        # this step, the total number of scheduled requests can be smaller than
        # len(self.running).
        assert len(scheduled_new_reqs) + len(scheduled_resumed_reqs) + len(
            scheduled_running_reqs
        ) <= len(self.running)

        prefill_tokens, decode_tokens, prefill_attention, decode_context = (
            self._execution_work({
                rid: (
                    self.requests[rid].num_computed_tokens,
                    self.requests[rid].num_computed_tokens + n,
                    0,
                )
                for rid, n in num_scheduled_tokens.items()
            })
        )
        # One schedule_batch line per step, written once the batch has run
        # (see _complete_batch_log); fields are in the order written.
        batch_record: dict[str, Any] = dict(
            step=self._step_id,
            new=len(scheduled_new_reqs),
            resumed=len(scheduled_resumed_reqs),
            running=len(scheduled_running_reqs),
            preempted=len(preempted_reqs),
            waiting_after=len(self.waiting),
            running_after=len(self.running),
            token_budget_used=total_num_scheduled_tokens,
            prefill_tokens=prefill_tokens,
            decode_tokens=decode_tokens,
            prefill_attention=prefill_attention,
            decode_context=decode_context,
            rank_ns=self._step_rank_ns,
            reranks=self._step_rerank_count,
            lpm_skips=self._step_lpm_skips,
            lpm_scan_ns=self._step_lpm_scan_ns,
            preempted_tokens=self._step_preempted_tokens,
            evicted_blocks=(
                self.kv_cache_manager.block_pool.num_evicted_cached_blocks
                - self._step_evicted_blocks_start
            ),
            pkas=self._pkas_step_stats,
        )
        if self._skip_eviction:
            pool = self.kv_cache_manager.block_pool
            batch_record["skip_protected_reqs"] = len(self._skip_protected)
            batch_record["skip_protected_free_blocks"] = (
                pool.num_skip_protected_free_blocks
            )
            batch_record["skip_protected_evictions"] = (
                pool.num_skip_protected_evictions
                - self._step_skip_protected_evictions_start
            )
        if self._defer_inflight:
            batch_record["deferred"] = self._step_defers
        if self._dspf is not None:
            batch_record["dspf_feasible"] = self._dspf.step_feasible
            batch_record["dspf_edd"] = self._dspf.step_edd
            batch_record["dspf_late"] = self._dspf.step_late

        # Get the longest common prefix among all requests in the running queue.
        # This can be potentially used for cascade attention.
        num_common_prefix_blocks = [0] * len(self.kv_cache_config.kv_cache_groups)
        with record_function_or_nullcontext("schedule: get_num_common_prefix_blocks"):
            if self.running:
                any_request_id = self.running[0].request_id
                num_common_prefix_blocks = (
                    self.kv_cache_manager.get_num_common_prefix_blocks(any_request_id)
                )

        # Construct the scheduler output.
        if self.use_v2_model_runner:
            scheduled_new_reqs = scheduled_new_reqs + scheduled_resumed_reqs
            scheduled_resumed_reqs = []
            new_reqs_data = [
                NewRequestData.from_request(
                    req,
                    req_to_new_blocks[req.request_id].get_block_ids(),
                    req._all_token_ids,
                )
                for req in scheduled_new_reqs
            ]
        else:
            new_reqs_data = [
                NewRequestData.from_request(
                    req, req_to_new_blocks[req.request_id].get_block_ids()
                )
                for req in scheduled_new_reqs
            ]

        with record_function_or_nullcontext("schedule: make_cached_request_data"):
            cached_reqs_data = self._make_cached_request_data(
                scheduled_running_reqs,
                scheduled_resumed_reqs,
                num_scheduled_tokens,
                scheduled_spec_decode_tokens,
                req_to_new_blocks,
            )

        # Record the request ids that were scheduled in this step.
        self.prev_step_scheduled_req_ids.clear()
        self.prev_step_scheduled_req_ids.update(num_scheduled_tokens.keys())

        new_block_ids_to_zero = (
            (self.kv_cache_manager.take_new_block_ids() or None)
            if self.needs_kv_cache_zeroing
            else None
        )

        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=cached_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            scheduled_encoder_inputs=scheduled_encoder_inputs,
            num_common_prefix_blocks=num_common_prefix_blocks,
            preempted_req_ids={req.request_id for req in preempted_reqs},
            # finished_req_ids is an existing state in the scheduler,
            # instead of being newly scheduled in this step.
            # It contains the request IDs that are finished in between
            # the previous and the current steps.
            finished_req_ids=self.finished_req_ids,
            free_encoder_mm_hashes=self.encoder_cache_manager.get_freed_mm_hashes(),
            new_block_ids_to_zero=new_block_ids_to_zero,
        )

        # NOTE(Kuntai): this function is designed for multiple purposes:
        # 1. Plan the KV cache store
        # 2. Wrap up all the KV cache load / save ops into an opaque object
        # 3. Clear the internal states of the connector
        if self.connector is not None:
            meta = self._build_kv_connector_meta(self.connector, scheduler_output)
            scheduler_output.kv_connector_metadata = meta

        # Build the connector meta for ECConnector
        if self.ec_connector is not None:
            ec_meta: ECConnectorMetadata = self.ec_connector.build_connector_meta(
                scheduler_output
            )
            scheduler_output.ec_connector_metadata = ec_meta

        self._preemption_cost_supported = self._preemption_cost_supported and all(
            not request.resumable
            and request.sampling_params is not None
            and not request.has_encoder_inputs
            for request in (self.requests[rid] for rid in num_scheduled_tokens)
        )
        cost_supported = self._preemption_cost_supported
        if cost_supported and total_num_scheduled_tokens:
            intervals = {
                rid: (
                    self.requests[rid].num_computed_tokens,
                    self.requests[rid].num_computed_tokens + n,
                    self._preemption_recovery_high.get(rid, 0),
                )
                for rid, n in num_scheduled_tokens.items()
            }
            self._pending_batch_logs[id(scheduler_output)] = (batch_record, intervals)
        else:
            if not cost_supported:
                batch_record["preempted_tokens"] = -1
            self._log_schedule_batch(batch_record, 0 if cost_supported else -1)

        if diagnostics is not None:
            diagnostics.scheduled(scheduler_output, self.requests)
        with record_function_or_nullcontext("schedule: update_after_schedule"):
            self._update_after_schedule(scheduler_output)
        if diagnostics is not None:
            diagnostics.end_step()
        if survival is not None:
            survival.end_step()
        return scheduler_output

    def _build_kv_connector_meta(
        self, connector: KVConnectorBase_V1, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        return connector.build_connector_meta(scheduler_output)

    @staticmethod
    def _log_schedule_batch(record: dict[str, Any], recomputed_tokens: int) -> None:
        """Write one step: counters, -1 when unavailable; PKAS counters only
        when non-zero, and execution_ns only when the batch was timed."""
        pkas = record["pkas"] or {}
        fields = " ".join(
            f"{key}={value}" for key, value in record.items() if key != "pkas"
        )
        logger.info(
            "schedule_batch: %s recomputed_tokens=%d%s",
            fields,
            recomputed_tokens,
            "".join(f" pkas_{key}={value}" for key, value in pkas.items() if value),
        )

    def observe_batch_execution(
        self, scheduler_output: SchedulerOutput, seconds: float
    ) -> None:
        pending = self._pending_batch_logs.get(id(scheduler_output))
        if pending is not None:
            pending[0]["execution_ns"] = int(seconds * 1e9)

    def _execution_work(
        self, intervals: dict[str, tuple[int, int, int]]
    ) -> tuple[int, int, int, int]:
        """Work of one batch for the step cost model.

        Tokens below a request's prompt-or-output high mark execute as
        prefill (this includes replayed output after preemption); the rest are
        decode. A token at position p attends to p + 1 keys, so the attention
        terms are the sums of p + 1 over the prefill and decode tokens.
        """
        prefill = decode = prefill_attention = decode_context = 0
        for rid, (start, end, _) in intervals.items():
            request = self.requests.get(rid)
            if request is None:
                continue
            split = start + min(end - start, max(
                0, max(request.num_prompt_tokens, request.num_tokens - 1) - start
            ))
            prefill += split - start
            decode += end - split
            prefill_attention += (split * (split + 1) - start * (start + 1)) // 2
            decode_context += (end * (end + 1) - split * (split + 1)) // 2
        return prefill, decode, prefill_attention, decode_context

    def _complete_batch_log(
        self, scheduler_output: SchedulerOutput, model_runner_output: ModelRunnerOutput
    ) -> None:
        pending = self._pending_batch_logs.pop(id(scheduler_output), None)
        if pending is None:
            return
        args, intervals = pending
        recomputed_tokens = 0
        for rid, (start, end, recovery_high) in intervals.items():
            if rid not in model_runner_output.req_id_to_index:
                continue
            recomputed_tokens += max(0, min(end, recovery_high) - start)
            if rid in self.requests:
                self._completed_token_high[rid] = max(
                    self._completed_token_high.get(rid, 0), end
                )
                if end >= recovery_high:
                    self._preemption_recovery_high.pop(rid, None)
        self._log_schedule_batch(args, recomputed_tokens)

    def _preempt_request(
        self, request: Request, timestamp: float, reason: str = "memory_pressure"
    ) -> None:
        """Preempt a request and put it back to the waiting queue.

        NOTE: The request should be popped from the running queue outside of this
        method.
        """
        assert request.status == RequestStatus.RUNNING, (
            "Only running requests can be preempted"
        )
        diagnostics = self.preemption_diagnostics
        free_before = self.kv_cache_manager.block_pool.get_num_free_blocks()
        if reason == "memory_pressure":
            if (
                self._preemption_cost_supported
                and not request.resumable
                and request.sampling_params is not None
                and not request.has_encoder_inputs
                and self._step_preempted_tokens >= 0
            ):
                self._step_preempted_tokens += request.num_computed_tokens
                self._preemption_recovery_high[request.request_id] = (
                    self._completed_token_high.get(request.request_id, 0)
                )
            else:
                self._preemption_cost_supported = False
                self._step_preempted_tokens = -1
        else:
            self._preemption_recovery_high.pop(request.request_id, None)
        self._pkas_block_ids.pop(request.request_id, None)
        self._pkas_reject_cache.pop(request.request_id, None)
        self._pkas_epoch += 1
        survival = self.survival_diagnostics
        victim = (
            survival.capture_victim(request)
            if survival is not None and reason == "memory_pressure"
            else None
        )
        self.kv_cache_manager.free(request)
        free_after = self.kv_cache_manager.block_pool.get_num_free_blocks()
        # Its next admitted_prefix_cache line (same request_id) is the resume.
        pkas = ""
        if self.predictive_admission is not None:
            estimator = self.predictive_admission.estimator
            current_target = (
                request.num_output_tokens + estimator.predict()
                if self._pkas_repo
                else estimator.target(request.num_output_tokens)
            )
            pkas = (
                " pkas_gate_target_output_tokens="
                f"{self._pkas_last_gate_target.get(request.request_id, -1)}"
                f" pkas_current_target_output_tokens={current_target}"
            )
        logger.info(
            "preempted: step=%d request_id=%s reason=%s computed_tokens=%d "
            "prompt_tokens=%d output_tokens=%d released_blocks=%d preemptions=%d%s",
            self._step_id,
            request.request_id,
            reason,
            request.num_computed_tokens,
            request.num_prompt_tokens,
            request.num_output_tokens,
            free_after - free_before,
            request.num_preemptions + 1,
            pkas,
        )
        if survival is not None:
            survival.preempted(request, victim)
        self.encoder_cache_manager.free(request)
        request.status = RequestStatus.PREEMPTED
        request.num_computed_tokens = 0
        if request.spec_token_ids:
            request.spec_token_ids = []
        request.num_preemptions += 1
        if self.log_stats:
            request.record_event(EngineCoreEventType.PREEMPTED, timestamp)

        # Put the request back to the waiting queue.
        self.waiting.prepend_request(request)
        self._track_waiting_request(request)
        if diagnostics is not None:
            diagnostics.preempt(request, free_before, free_after, reason)

    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        # Advance the number of computed tokens for the request AFTER
        # the request is scheduled.
        # 1. The scheduler_output of the current step has to include the
        #    original number of scheduled tokens to determine input IDs.
        # 2. Advance the number of computed tokens here allowing us to
        #    schedule the prefill request again immediately in the next
        #    scheduling step.
        # 3. If some tokens (e.g. spec tokens) are rejected later, the number of
        #    computed tokens will be adjusted in update_from_output.
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        for req_id, num_scheduled_token in num_scheduled_tokens.items():
            request = self.requests[req_id]
            request.num_computed_tokens += num_scheduled_token
            request.is_prefill_chunk = request.num_computed_tokens < (
                request.num_tokens + request.num_output_placeholders
            )
            scheduler_output.has_structured_output_requests |= (
                request.use_structured_output and not request.is_prefill_chunk
            )

        # Snapshot block IDs for routed experts before forward starts.
        # A concurrent schedule() may preempt requests and free blocks
        # before update_from_output runs; the snapshot survives that.
        # Use update() to preserve entries from the previous step that
        # have not yet been consumed by update_from_output (async
        # scheduling may call _update_after_schedule again before the
        # prior update_from_output runs).
        if self.enable_return_routed_experts:
            gid = self.routed_experts_mgr.attn_gid
            self._re_block_ids.update(
                {
                    rid: self.kv_cache_manager.get_blocks(rid).get_block_ids()[gid]
                    for rid in num_scheduled_tokens
                }
            )

        # Clear the finished request IDs.
        # NOTE: We shouldn't do self.finished_req_ids.clear() here because
        # it will also affect the scheduler output.
        self.finished_req_ids = set()

    def _update_request_as_session(
        self, session: Request, update: StreamingUpdate
    ) -> None:
        """
        Updates the waiting session with the next streaming update.

        Discards the last sampled output token from the prior input chunk.
        """

        # Current streaming input behaviour: Keep only computed output tokens
        # (discard final sampled output token).
        num_computed_tokens = session.num_computed_tokens
        kept_output_tokens = session._all_token_ids[
            session.num_prompt_tokens : num_computed_tokens
        ]
        del session._all_token_ids[num_computed_tokens:]
        session._output_token_ids.clear()
        assert session.prompt_token_ids is not None
        # Extend prompt with kept output tokens.
        session.prompt_token_ids.extend(kept_output_tokens)

        if update.mm_features:
            base = session.num_tokens
            for mm_feature in update.mm_features:
                mm_feature.mm_position = replace(
                    mm_feature.mm_position, offset=mm_feature.mm_position.offset + base
                )
            session.mm_features.extend(update.mm_features)

        session._all_token_ids.extend(update.prompt_token_ids or ())
        session.prompt_token_ids.extend(update.prompt_token_ids or ())
        # Update block hashes for the new tokens.
        session.update_block_hashes()
        session.num_prompt_tokens = len(session.prompt_token_ids)
        session.arrival_time = update.arrival_time
        session.sampling_params = update.sampling_params
        if session.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
            self.num_waiting_for_streaming_input -= 1
        session.status = RequestStatus.WAITING

        if self.log_stats:
            session.record_event(EngineCoreEventType.QUEUED)

    def _make_cached_request_data(
        self,
        running_reqs: list[Request],
        resumed_reqs: list[Request],
        num_scheduled_tokens: dict[str, int],
        spec_decode_tokens: dict[str, list[int]],
        req_to_new_blocks: dict[str, KVCacheBlocks],
    ) -> CachedRequestData:
        req_ids: list[str] = []
        new_token_ids: list[list[int]] = []
        new_block_ids: list[tuple[list[int], ...] | None] = []
        all_token_ids: dict[str, list[int]] = {}
        num_computed_tokens: list[int] = []
        num_output_tokens: list[int] = []
        resumed_req_ids = set()

        num_running_reqs = len(running_reqs)
        for idx, req in enumerate(itertools.chain(running_reqs, resumed_reqs)):
            req_id = req.request_id
            req_ids.append(req_id)
            # NOTE: In PP+async scheduling, we consume token ids via a direct GPU
            # broadcast path (`input_batch.prev_sampled_token_ids`), so we can
            # omit this payload.
            if self.use_pp and not self.scheduler_config.async_scheduling:
                # When using PP, the scheduler sends the sampled tokens back,
                # because there's no direct communication between the first-
                # stage worker and the last-stage worker. Otherwise, we don't
                # need to send the sampled tokens back because the model runner
                # will cache them.
                num_tokens = num_scheduled_tokens[req_id] - len(
                    spec_decode_tokens.get(req_id, ())
                )
                token_ids = req.all_token_ids[
                    req.num_computed_tokens : req.num_computed_tokens + num_tokens
                ]
                new_token_ids.append(token_ids)
            scheduled_in_prev_step = req_id in self.prev_step_scheduled_req_ids
            if idx >= num_running_reqs:
                assert not scheduled_in_prev_step
                resumed_req_ids.add(req_id)
            if not scheduled_in_prev_step:
                all_token_ids[req_id] = req.all_token_ids.copy()
            new_block_ids.append(
                req_to_new_blocks[req_id].get_block_ids(allow_none=True)
            )
            num_computed_tokens.append(req.num_computed_tokens)
            num_output_tokens.append(
                req.num_output_tokens + req.num_output_placeholders
            )

        return CachedRequestData(
            req_ids=req_ids,
            resumed_req_ids=resumed_req_ids,
            new_token_ids=new_token_ids,
            all_token_ids=all_token_ids,
            new_block_ids=new_block_ids,
            num_computed_tokens=num_computed_tokens,
            num_output_tokens=num_output_tokens,
        )

    def _try_schedule_encoder_inputs(
        self,
        request: Request,
        num_computed_tokens: int,
        num_new_tokens: int,
        encoder_compute_budget: int,
        shift_computed_tokens: int = 0,
    ) -> tuple[list[int], int, int, list[int]]:
        """
        Determine which encoder inputs need to be scheduled in the current step,
        and update `num_new_tokens` and encoder token budget accordingly.

        An encoder input will be scheduled if:
        - Its output tokens overlap with the range of tokens being computed
        in this step, i.e.,
        [num_computed_tokens, num_computed_tokens + num_new_tokens).
        - It is not already computed and stored in the encoder cache.
        - It is not exist on remote encoder cache (via ECConnector)
        - There is sufficient encoder token budget to process it.
        - The encoder cache has space to store it.

        If an encoder input cannot be scheduled due to cache or budget
        limitations, the method adjusts `num_new_tokens` to schedule only the
        decoder tokens up to just before the unschedulable encoder input.

        Note that num_computed_tokens includes both locally cached
        blocks and externally cached blocks (via KVConnector).
        """
        if num_new_tokens == 0 or not request.has_encoder_inputs:
            return [], num_new_tokens, encoder_compute_budget, []
        encoder_inputs_to_schedule: list[int] = []
        mm_features = request.mm_features
        assert mm_features is not None
        assert len(mm_features) > 0
        external_load_encoder_input = []

        # NOTE: since scheduler operates on the request level (possibly with
        # multiple encoder inputs per request), we need to create temporary
        # trackers for accounting at the encoder input level.
        mm_hashes_to_schedule = set()
        num_embeds_to_schedule = 0
        for i, mm_feature in enumerate(mm_features):
            start_pos = mm_feature.mm_position.offset
            num_encoder_tokens = mm_feature.mm_position.length
            num_encoder_embeds = mm_feature.mm_position.get_num_embeds()
            item_identifier = mm_feature.identifier

            # The encoder output is needed if the two ranges overlap:
            # [num_computed_tokens, num_computed_tokens + num_new_tokens) and
            # [start_pos, start_pos + num_encoder_tokens)
            if (
                start_pos
                >= num_computed_tokens + num_new_tokens + shift_computed_tokens
            ):
                # The encoder input is not needed in this step.
                break

            if self.is_encoder_decoder and num_computed_tokens > 0:
                assert start_pos == 0, (
                    "Encoder input should be processed at the beginning of "
                    "the sequence when encoder-decoder models are used."
                )
                # Encoder input has already been computed
                # The calculation here is a bit different. We don't turn encoder
                # output into tokens that get processed by the decoder and
                # reflected in num_computed_tokens. Instead, start_pos reflects
                # the position where we need to ensure we calculate encoder
                # inputs. This should always be 0 to ensure we calculate encoder
                # inputs before running the decoder.  Once we've calculated some
                # decoder tokens (num_computed_tokens > 0), then we know we
                # already calculated encoder inputs and can skip here.
                continue
            elif start_pos + num_encoder_tokens <= num_computed_tokens:
                # The encoder input is already computed and stored
                # in the decoder's KV cache.
                continue

            if not self.is_encoder_decoder:
                # We are not using the encoder cache for encoder-decoder models,
                # yet.
                if item_identifier in mm_hashes_to_schedule:
                    # The same encoder input has already been scheduled in the
                    # current step.
                    continue

                if self.encoder_cache_manager.check_and_update_cache(request, i):
                    # The encoder input is already computed and cached from a
                    # previous step.
                    continue

            # If no encoder input chunking is allowed, we do not want to
            # partially schedule a multimodal item. If the scheduled range would
            # only cover part of the mm input, roll back to before the mm item.
            if (
                self.scheduler_config.disable_chunked_mm_input
                and num_computed_tokens < start_pos
                and (num_computed_tokens + num_new_tokens)
                < (start_pos + num_encoder_tokens)
            ):
                # Account for EAGLE shift when rolling back to avoid
                # encoder cache miss. This ensures the scheduled range
                # stops before start_pos even with the shift.
                num_new_tokens = max(
                    0, start_pos - (num_computed_tokens + shift_computed_tokens)
                )
                break
            if not self.encoder_cache_manager.can_allocate(
                request, i, encoder_compute_budget, num_embeds_to_schedule
            ):
                # The encoder cache is full or the encoder budget is exhausted.
                # NOTE(woosuk): We assume that the encoder input tokens should
                # be processed altogether, as the encoder usually uses
                # bidirectional attention.
                if num_computed_tokens + shift_computed_tokens < start_pos:
                    # We only schedule the decoder tokens just before the
                    # encoder input.
                    num_new_tokens = start_pos - (
                        num_computed_tokens + shift_computed_tokens
                    )
                else:
                    # Because of prefix caching, num_computed_tokens is greater
                    # than start_pos even though its encoder input is not
                    # available. In this case, we can't schedule any token for
                    # the request in this step.
                    num_new_tokens = 0
                break

            # Calculate the number of embeddings to schedule in the current range
            # of scheduled encoder placeholder tokens.
            start_idx_rel = max(0, num_computed_tokens - start_pos)
            end_idx_rel = min(
                num_encoder_tokens, num_computed_tokens + num_new_tokens - start_pos
            )
            curr_embeds_start, curr_embeds_end = (
                mm_feature.mm_position.get_embeds_indices_in_range(
                    start_idx_rel, end_idx_rel
                )
            )
            # There's no embeddings in the current range of encoder placeholder tokens
            # so we can skip the encoder input.
            if curr_embeds_end - curr_embeds_start == 0:
                continue

            if self.ec_connector is not None and self.ec_connector.has_cache_item(
                item_identifier
            ):
                mm_hashes_to_schedule.add(item_identifier)
                external_load_encoder_input.append(i)
                num_embeds_to_schedule += num_encoder_embeds
                continue

            num_embeds_to_schedule += num_encoder_embeds
            encoder_compute_budget -= num_encoder_embeds
            mm_hashes_to_schedule.add(item_identifier)
            encoder_inputs_to_schedule.append(i)

        return (
            encoder_inputs_to_schedule,
            num_new_tokens,
            encoder_compute_budget,
            external_load_encoder_input,
        )

    def get_grammar_bitmask(
        self, scheduler_output: SchedulerOutput
    ) -> GrammarOutput | None:
        # Collect list of scheduled request ids that use structured output.
        # The corresponding rows of the bitmask will be in this order.
        if not scheduler_output.has_structured_output_requests:
            return None

        structured_output_request_ids = [
            req_id
            for req_id in scheduler_output.num_scheduled_tokens
            if (req := self.requests.get(req_id))
            and (req.use_structured_output and not req.is_prefill_chunk)
        ]
        if not structured_output_request_ids:
            return None

        bitmask = self.structured_output_manager.grammar_bitmask(
            self.requests,
            structured_output_request_ids,
            scheduler_output.scheduled_spec_decode_tokens,
        )
        return GrammarOutput(structured_output_request_ids, bitmask)

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        self._complete_batch_log(scheduler_output, model_runner_output)
        sampled_token_ids = model_runner_output.sampled_token_ids
        logprobs = model_runner_output.logprobs
        prompt_logprobs_dict = model_runner_output.prompt_logprobs_dict
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        pooler_outputs = model_runner_output.pooler_output
        num_nans_in_logits = model_runner_output.num_nans_in_logits
        kv_connector_output = model_runner_output.kv_connector_output
        cudagraph_stats = model_runner_output.cudagraph_stats

        perf_stats: PerfStats | None = None
        if self.perf_metrics and self.perf_metrics.is_enabled():
            perf_stats = self.perf_metrics.get_step_perf_stats_per_gpu(scheduler_output)

        outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)
        spec_decoding_stats: SpecDecodingStats | None = None
        kv_connector_stats: KVConnectorStats | None = (
            kv_connector_output.kv_connector_stats if kv_connector_output else None
        )
        if kv_connector_stats and self.connector:
            kv_stats = self.connector.get_kv_connector_stats()
            if kv_stats:
                kv_connector_stats = kv_connector_stats.aggregate(kv_stats)

        failed_kv_load_req_ids = None
        if kv_connector_output and kv_connector_output.invalid_block_ids:
            # These blocks contain externally computed tokens that failed to
            # load. Identify affected requests and adjust their computed token
            # count to trigger recomputation of the invalid blocks.
            failed_kv_load_req_ids = self._handle_invalid_blocks(
                kv_connector_output.invalid_block_ids,
                num_scheduled_tokens,
            )

        # Persist per-step routed experts into the scheduler-side slot
        # buffer (CPU->CPU fancy-index assign; ~few MB per step).
        # MUST precede the per-request routing reads below: stopped
        # requests may terminate on tokens generated in this very step,
        # whose routing was just D2H'd into model_runner_output.
        routing_data = None
        routing_offsets: dict[str, int] = {}
        if model_runner_output.routed_experts is not None:
            re = model_runner_output.routed_experts
            self.routed_experts_mgr.store_batch(re.routing_data, re.slot_mapping)
            routing_data = re.routing_data.astype(
                self.routed_experts_mgr.routed_experts_by_slot.dtype,
                copy=False,
            )
            # Build offset map using model runner's request order
            # (input_batch ordering), NOT scheduler dict order.
            offset = 0
            for rid in model_runner_output.req_ids:
                routing_offsets[rid] = offset
                offset += num_scheduled_tokens[rid]

        # NOTE(woosuk): As len(num_scheduled_tokens) can be up to 1K or more,
        # the below loop can be a performance bottleneck. We should do our best
        # to avoid expensive operations inside the loop.
        stopped_running_reqs: set[Request] = set()
        stopped_preempted_reqs: set[Request] = set()
        for req_id, num_tokens_scheduled in num_scheduled_tokens.items():
            assert num_tokens_scheduled > 0
            if failed_kv_load_req_ids and req_id in failed_kv_load_req_ids:
                # skip failed or rescheduled requests from KV load failure
                continue
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # The request is already finished. This can happen if the
                # request is aborted while the model is executing it (e.g.,
                # in pipeline parallelism or in async scheduling).
                # NOTE(Kuntai): When delay_free_blocks=True (for async KV
                # cache transfer in KV connector), the aborted request will not
                # be set to None (in order to finish async KV transfer).
                # In this case, we use is_finished() to check.
                continue

            req_index = model_runner_output.req_id_to_index[req_id]
            generated_token_ids = (
                sampled_token_ids[req_index] if sampled_token_ids else []
            )

            scheduled_spec_token_ids = (
                scheduler_output.scheduled_spec_decode_tokens.get(req_id)
            )
            if scheduled_spec_token_ids and generated_token_ids:
                num_draft_tokens = len(scheduled_spec_token_ids)
                num_accepted = len(generated_token_ids) - 1
                num_rejected = num_draft_tokens - num_accepted
                # num_computed_tokens represents the number of tokens
                # processed in the current step, considering scheduled
                # tokens and rejections. If some tokens are rejected,
                # num_computed_tokens is decreased by the number of rejected
                # tokens.
                if request.num_computed_tokens > 0:
                    request.num_computed_tokens -= num_rejected
                # If async scheduling, num_output_placeholders also includes
                # the scheduled spec tokens count and so is similarly adjusted.
                if request.num_output_placeholders > 0:
                    request.num_output_placeholders -= num_rejected
                spec_decoding_stats = self.make_spec_decoding_stats(
                    spec_decoding_stats,
                    num_draft_tokens=num_draft_tokens,
                    num_accepted_tokens=num_accepted,
                    num_invalid_spec_tokens=scheduler_output.num_invalid_spec_tokens,
                    request_id=req_id,
                )

            # Free encoder inputs only after the step has actually executed.
            if request.has_encoder_inputs:
                self._free_encoder_inputs(request)

            stopped = False
            new_logprobs = None
            new_token_ids = generated_token_ids
            pooler_output = pooler_outputs[req_index] if pooler_outputs else None
            kv_transfer_params = None
            status_before_stop = request.status
            num_output_tokens_before = len(request._output_token_ids)

            # Check for stop and update request status.
            if new_token_ids:
                new_token_ids, stopped = self._update_request_with_output(
                    request, new_token_ids
                )
            elif request.pooling_params and pooler_output is not None:
                # Pooling stops as soon as there is output.
                request.status = RequestStatus.FINISHED_STOPPED
                stopped = True

            if new_token_ids and self.structured_output_manager.should_advance(request):
                struct_output_request = request.structured_output_request
                assert struct_output_request is not None
                assert struct_output_request.grammar is not None
                if not struct_output_request.grammar.accept_tokens(  # type: ignore[union-attr]
                    req_id, new_token_ids
                ):
                    logger.error(
                        "Unexpected: grammar rejected tokens %s for request %s. "
                        "Terminating request.",
                        new_token_ids,
                        req_id,
                    )
                    request.status = RequestStatus.FINISHED_ERROR
                    request.resumable = False
                    stopped = True

            if self.preemption_diagnostics is not None:
                self.preemption_diagnostics.completed(
                    scheduler_output, request, bool(new_token_ids)
                )
            routed_experts = None
            if (
                self.enable_return_routed_experts
                and routing_data is not None
                and new_token_ids
            ):
                req_offset = routing_offsets[req_id]
                end = req_offset + num_tokens_scheduled
                block_ids = self._re_block_ids.pop(req_id, [])
                if num_output_tokens_before == 0:
                    # Prefill completed: read full prompt routing from
                    # slot buffer using the block-ID snapshot taken at
                    # schedule time (immune to async preemption).
                    if (
                        request.sampling_params is not None
                        and request.sampling_params.routed_experts_prompt_start
                        is not None
                    ):
                        prompt_start = (
                            request.sampling_params.routed_experts_prompt_start
                        )
                        assert prompt_start < request.num_prompt_tokens
                    else:
                        prompt_start = 0
                    routed_experts = self.routed_experts_mgr.get(
                        block_ids,
                        request.num_prompt_tokens,
                        token_start=prompt_start,
                    )
                else:
                    if scheduled_spec_token_ids:
                        # Spec decode: accepted tokens at the START of
                        # the scheduled range, rejected at the end.
                        routed_experts = routing_data[
                            req_offset : req_offset + len(new_token_ids)
                        ]
                    else:
                        # Normal decode / re-prefill: token(s) at the END.
                        routed_experts = routing_data[end - len(new_token_ids) : end]

            finish_reason = None
            if stopped:
                # Capture finish_reason BEFORE _handle_stopped_request, which may
                # reset the status to WAITING for streaming requests that continue.
                finish_reason = request.get_finished_reason()
                finished = self._handle_stopped_request(request)
                if finished:
                    kv_transfer_params = self._free_request(request)

                if status_before_stop == RequestStatus.RUNNING:
                    stopped_running_reqs.add(request)
                else:
                    stopped_preempted_reqs.add(request)

            # Extract sample logprobs if needed.
            if (
                request.sampling_params is not None
                and request.sampling_params.num_logprobs is not None
                and logprobs
            ):
                new_logprobs = logprobs.slice_request(req_index, len(new_token_ids))

            if num_nans_in_logits is not None and req_id in num_nans_in_logits:
                request.num_nans_in_logits = num_nans_in_logits[req_id]

            # Get prompt logprobs for this request.
            prompt_logprobs_tensors = prompt_logprobs_dict.get(req_id)
            if (
                new_token_ids
                or pooler_output is not None
                or kv_transfer_params
                or stopped
            ):
                # Add EngineCoreOutput for this Request.
                outputs[request.client_index].append(
                    EngineCoreOutput(
                        request_id=req_id,
                        new_token_ids=new_token_ids,
                        finish_reason=finish_reason,
                        new_logprobs=new_logprobs,
                        new_prompt_logprobs_tensors=prompt_logprobs_tensors,
                        pooling_output=pooler_output,
                        stop_reason=request.stop_reason,
                        events=request.take_events(),
                        prefill_stats=request.take_prefill_stats(),
                        kv_transfer_params=kv_transfer_params,
                        trace_headers=request.trace_headers,
                        routed_experts=routed_experts,
                        num_nans_in_logits=request.num_nans_in_logits,
                    )
                )
            else:
                # Invariant: EngineCore returns no partial prefill outputs.
                assert not prompt_logprobs_tensors

        if self.preemption_diagnostics is not None:
            self.preemption_diagnostics.end_output(scheduler_output)

        # Remove the stopped requests from the running and waiting queues.
        if stopped_running_reqs:
            self.running = remove_all(self.running, stopped_running_reqs)
        if stopped_preempted_reqs:
            # This is a rare case and unlikely to impact performance.
            self.waiting.remove_requests(stopped_preempted_reqs)
            for request in stopped_preempted_reqs:
                self._untrack_waiting_request(request)

        if failed_kv_load_req_ids and not self.recompute_kv_load_failures:
            requests = [self.requests[req_id] for req_id in failed_kv_load_req_ids]
            self.finish_requests(failed_kv_load_req_ids, RequestStatus.FINISHED_ERROR)
            for request in requests:
                outputs[request.client_index].append(
                    EngineCoreOutput(
                        request_id=request.request_id,
                        new_token_ids=[],
                        finish_reason=request.get_finished_reason(),
                        events=request.take_events(),
                        trace_headers=request.trace_headers,
                    )
                )

        # KV Connector: update state for finished KV Transfers.
        if kv_connector_output:
            self._update_from_kv_xfer_finished(kv_connector_output)

        # collect KV cache events from KV cache manager
        events = self.kv_cache_manager.take_events()

        # collect KV cache events from connector
        if self.connector is not None:
            connector_events = self.connector.take_events()
            if connector_events:
                if events is None:
                    events = list(connector_events)
                else:
                    events.extend(connector_events)

        # publish collected KV cache events
        if events:
            batch = KVEventBatch(ts=time.time(), events=events)
            self.kv_event_publisher.publish(batch)

        # Create EngineCoreOutputs for all clients that have requests with
        # outputs in this step.
        engine_core_outputs = {
            client_index: EngineCoreOutputs(outputs=outs)
            for client_index, outs in outputs.items()
        }

        finished_req_ids = self.finished_req_ids_dict
        if finished_req_ids:
            # Include ids of requests that finished since last outputs
            # were sent.
            for client_index, finished_set in finished_req_ids.items():
                # Set finished request set in EngineCoreOutputs for this client.
                if (eco := engine_core_outputs.get(client_index)) is not None:
                    eco.finished_requests = finished_set
                else:
                    engine_core_outputs[client_index] = EngineCoreOutputs(
                        finished_requests=finished_set
                    )
            finished_req_ids.clear()

        if (
            stats := self.make_stats(
                spec_decoding_stats, kv_connector_stats, cudagraph_stats, perf_stats
            )
        ) is not None:
            # Return stats to only one of the front-ends.
            if (eco := next(iter(engine_core_outputs.values()), None)) is None:
                # We must return the stats even if there are no request
                # outputs this step.
                engine_core_outputs[0] = eco = EngineCoreOutputs()
            eco.scheduler_stats = stats

        return engine_core_outputs

    @staticmethod
    def _is_blocked_waiting_status(status: RequestStatus) -> bool:
        return status in (
            RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR,
            RequestStatus.WAITING_FOR_REMOTE_KVS,
            RequestStatus.WAITING_FOR_STREAMING_REQ,
        )

    def _enqueue_waiting_request(self, request: Request) -> None:
        if self._is_blocked_waiting_status(request.status):
            self.skipped_waiting.add_request(request)
        else:
            self.waiting.add_request(request)

    def _select_waiting_queue_for_scheduling(self) -> RequestQueue | None:
        if self.policy in (
            SchedulingPolicy.FCFS,
            SchedulingPolicy.LPM,
            SchedulingPolicy.CLPM,
            SchedulingPolicy.SPF,
            SchedulingPolicy.SSPF,
            SchedulingPolicy.SJF,
            SchedulingPolicy.DSPF,
        ):
            return self.skipped_waiting or self.waiting or None

        # PRIORITY mode: compare queue heads when both queues are non-empty.
        if self.waiting and self.skipped_waiting:
            waiting_req = self.waiting.peek_request()
            skipped_req = self.skipped_waiting.peek_request()
            return self.waiting if waiting_req < skipped_req else self.skipped_waiting

        return self.waiting or self.skipped_waiting or None

    def _handle_stopped_request(self, request: Request) -> bool:
        """Return True if finished (can be False for resumable requests)."""
        if not request.resumable:
            return True

        if request.streaming_queue:
            update = request.streaming_queue.popleft()
            if update is None:
                # Streaming request finished.
                return True
            self._update_request_as_session(request, update)
        else:
            request.status = RequestStatus.WAITING_FOR_STREAMING_REQ
            self.num_waiting_for_streaming_input += 1

        self._enqueue_waiting_request(request)
        self._track_waiting_request(request)
        return False

    def _update_request_with_output(
        self, request: Request, new_token_ids: list[int]
    ) -> tuple[list[int], bool]:
        # Append generated tokens and check for stop. Note that if
        # a request is still being prefilled, we expect the model runner
        # to return empty token ids for the request.
        stopped = False
        for num_new, output_token_id in enumerate(new_token_ids, 1):
            request.append_output_token_ids(output_token_id)

            # Check for stop and update request state.
            # This must be called before we make the EngineCoreOutput.
            stopped = check_stop(request, self.max_model_len)
            if stopped:
                del new_token_ids[num_new:]  # Trim new tokens if needed.
                break
        return new_token_ids, stopped

    def _free_encoder_inputs(self, request: Request) -> None:
        cached_encoder_input_ids = self.encoder_cache_manager.get_cached_input_ids(
            request
        )
        # OPTIMIZATION: Avoid list(set) if the set is empty.
        if not cached_encoder_input_ids:
            return

        # Here, we use list(set) to avoid modifying the set while iterating
        # over it.
        for input_id in list(cached_encoder_input_ids):
            mm_feature = request.mm_features[input_id]
            start_pos = mm_feature.mm_position.offset
            num_tokens = mm_feature.mm_position.length
            if self.is_encoder_decoder and request.num_computed_tokens > 0:
                # With Whisper, as soon as we've generated a single token,
                # we know we're done with the encoder input. Cross Attention
                # KVs have been calculated and cached already.
                self.encoder_cache_manager.free_encoder_input(request, input_id)
            elif start_pos + num_tokens <= request.num_computed_tokens:
                # The encoder output is already processed and stored
                # in the decoder's KV cache.
                self.encoder_cache_manager.free_encoder_input(request, input_id)

    def update_draft_token_ids(self, draft_token_ids: DraftTokenIds) -> None:
        for req_id, spec_token_ids in zip(
            draft_token_ids.req_ids,
            draft_token_ids.draft_token_ids,
        ):
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # The request may have been finished. Skip.
                continue

            if request.is_prefill_chunk:
                # Ignore draft tokens for prefill chunks.
                if request.spec_token_ids:
                    request.spec_token_ids = []
                continue

            # Add newly generated spec token ids to the request.
            if self.structured_output_manager.should_advance(request):
                metadata = request.structured_output_request
                spec_token_ids = metadata.grammar.validate_tokens(spec_token_ids)  # type: ignore[union-attr]
            request.spec_token_ids = spec_token_ids

    def update_draft_token_ids_in_output(
        self, draft_token_ids: DraftTokenIds, scheduler_output: SchedulerOutput
    ) -> None:
        num_invalid_spec_tokens: dict[str, int] = {}

        sched_spec_tokens = scheduler_output.scheduled_spec_decode_tokens
        for req_id, spec_token_ids in zip(
            draft_token_ids.req_ids,
            draft_token_ids.draft_token_ids,
        ):
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # The request may have been finished. Skip.
                continue

            placeholder_spec_tokens = sched_spec_tokens.get(req_id)
            if not placeholder_spec_tokens:
                continue

            orig_num_spec_tokens = len(placeholder_spec_tokens)
            # Trim drafts to scheduled number of spec tokens
            # (needed for chunked prefill case for example).
            del spec_token_ids[orig_num_spec_tokens:]
            # Filter out spec tokens which do not adhere to the grammar.
            if self.structured_output_manager.should_advance(request):
                metadata = request.structured_output_request
                assert metadata is not None and metadata.grammar is not None
                spec_token_ids = metadata.grammar.validate_tokens(spec_token_ids)
            # Pad to original number of spec tokens.
            num_invalid_tokens = orig_num_spec_tokens - len(spec_token_ids)
            if num_invalid_tokens:
                spec_token_ids.extend([-1] * num_invalid_tokens)
                num_invalid_spec_tokens[req_id] = num_invalid_tokens

            sched_spec_tokens[req_id] = spec_token_ids

        scheduler_output.num_invalid_spec_tokens = num_invalid_spec_tokens

    def get_request_counts(self) -> tuple[int, int]:
        """Returns (num_running_reqs, num_waiting_reqs)."""
        return len(self.running), len(self.waiting) + len(self.skipped_waiting)

    def add_request(self, request: Request) -> None:
        if self.predictive_admission is not None and (
            request.resumable
            or request.sampling_params is None
            or request.has_encoder_inputs
        ):
            raise ValueError(
                "PKAS predictive admission supports only non-streaming, "
                "decoder-only generation requests"
            )
        existing = self.requests.get(request.request_id)
        if existing is not None:
            update = StreamingUpdate.from_request(request)
            if existing.status != RequestStatus.WAITING_FOR_STREAMING_REQ:
                assert existing.streaming_queue is not None, "duplicate request id"
                # Queue next input chunk (or finished sentinel).
                existing.streaming_queue.append(update)
            elif update is not None:
                # Commence next input chunk.
                self._update_request_as_session(existing, update)
                self._track_waiting_request(existing)
            else:
                # Streaming-input session finished.
                self.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)
        else:
            if (
                self.policy == SchedulingPolicy.SJF
                and request.sampling_params is not None
                and not request.sampling_params.ignore_eos
            ):
                logger.warning_once(
                    "SJF ranks by max_tokens, which is only an upper bound on "
                    "output length for requests without ignore_eos."
                )
            if request.resumable:
                request.streaming_queue = deque()
            self._enqueue_waiting_request(request)
            self._track_waiting_request(request)
            self.requests[request.request_id] = request
            if self.connector is not None:
                self.connector.on_new_request(request)
            if self.preemption_diagnostics is not None:
                self.preemption_diagnostics.add_request(request)
            if self.survival_diagnostics is not None:
                self.survival_diagnostics.request_waiting(request)
            if self.log_stats:
                request.record_event(EngineCoreEventType.QUEUED)

    def finish_requests(
        self, request_ids: str | Iterable[str] | None, finished_status: RequestStatus
    ) -> list[tuple[str, int]]:
        """Handles the finish signal from outside the scheduler.

        For example, the API server can abort a request when the client
        disconnects.

        If request_ids is None, all requests will be finished.

        Returns:
            Tuple of (req_id, client_index) for requests that were aborted. Will not
            include any that were already finished.
        """
        assert RequestStatus.is_finished(finished_status)
        if isinstance(request_ids, str):
            request_ids = (request_ids,)
        elif request_ids is not None:
            request_ids = set(request_ids)
        else:
            request_ids = self.requests.keys()

        running_requests_to_remove = set()
        waiting_requests_to_remove = []
        valid_requests = []

        # First pass: collect requests to remove from queues
        for req_id in request_ids:
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # Invalid request ID.
                continue

            valid_requests.append(request)
            if request.status == RequestStatus.RUNNING:
                running_requests_to_remove.add(request)
            else:
                if request.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
                    self.num_waiting_for_streaming_input -= 1
                waiting_requests_to_remove.append(request)

        # Remove all requests from queues at once for better efficiency
        if running_requests_to_remove:
            self.running = remove_all(self.running, running_requests_to_remove)
        if waiting_requests_to_remove:
            self.waiting.remove_requests(waiting_requests_to_remove)
            self.skipped_waiting.remove_requests(waiting_requests_to_remove)
            for request in waiting_requests_to_remove:
                self._untrack_waiting_request(request)

        # Second pass: set status and free requests
        for request in valid_requests:
            delay_free_blocks = False
            if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                delay_free_blocks = (
                    request.request_id not in self.finished_recving_kv_req_ids
                )
                self.finished_recving_kv_req_ids.discard(request.request_id)
                self.failed_recving_kv_req_ids.discard(request.request_id)

            request.status = finished_status
            self._free_request(request, delay_free_blocks=delay_free_blocks)

        return [(r.request_id, r.client_index) for r in valid_requests]

    def _free_request(
        self, request: Request, delay_free_blocks: bool = False
    ) -> dict[str, Any] | None:
        assert request.is_finished()
        self._pkas_block_ids.pop(request.request_id, None)
        self._pkas_last_gate_target.pop(request.request_id, None)
        self._pkas_reject_cache.pop(request.request_id, None)
        if self._lpm_bypass_counts.pop(request.request_id, None) is not None:
            self._unprotect_skipped_request(request)
        self._lpm_first_skip_hits.pop(request.request_id, None)
        self._lpm_last_skip_hits.pop(request.request_id, None)
        self._defer_counts.pop(request.request_id, None)
        if self._dspf is not None:
            self._dspf.forget(request.request_id)
        self._pkas_epoch += 1
        if self.predictive_admission is not None and request.status in (
            RequestStatus.FINISHED_STOPPED,
            RequestStatus.FINISHED_LENGTH_CAPPED,
        ):
            self.predictive_admission.observe_completion(request.num_output_tokens)
        self._completed_token_high.pop(request.request_id, None)
        self._preemption_recovery_high.pop(request.request_id, None)
        if self.preemption_diagnostics is not None:
            self.preemption_diagnostics.finish(request)
        if self.survival_diagnostics is not None:
            self.survival_diagnostics.finished(request)

        request_logger = RequestEventLogger.get()
        if request_logger is not None:
            request_logger.log_request(request)

        connector_delay_free_blocks, kv_xfer_params = self._connector_finished(request)
        self.encoder_cache_manager.free(request)
        request_id = request.request_id
        self.finished_req_ids.add(request_id)
        if self.finished_req_ids_dict is not None:
            self.finished_req_ids_dict[request.client_index].add(request_id)

        delay_free_blocks |= connector_delay_free_blocks
        if not delay_free_blocks:
            self._free_blocks(request)

        return kv_xfer_params

    def _free_blocks(self, request: Request):
        assert request.is_finished()
        self.kv_cache_manager.free(request)
        del self.requests[request.request_id]

    @property
    def pause_state(self) -> PauseState:
        return self._pause_state

    def set_pause_state(self, pause_state: PauseState) -> None:
        self._pause_state = pause_state

    def get_num_unfinished_requests(self) -> int:
        if self._pause_state == PauseState.PAUSED_ALL:
            return 0
        if self._pause_state == PauseState.PAUSED_NEW:
            return len(self.running)
        num_waiting = (
            len(self.waiting)
            + len(self.skipped_waiting)
            - self.num_waiting_for_streaming_input
        )
        return num_waiting + len(self.running)

    def has_finished_requests(self) -> bool:
        return len(self.finished_req_ids) > 0

    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        """Reset the KV prefix cache.

        If reset_running_requests is True, all the running requests will be
        preempted and moved to the waiting queue.
        Otherwise, this method will only reset the KV prefix cache when there
        is no running requests taking KV cache.
        """
        if reset_running_requests:
            # For logging.
            timestamp = time.monotonic()
            # Invalidate all the current running requests KV's by pushing them to
            # the waiting queue. In this case, we can reduce the ref count of all
            # the kv blocks to 0 and thus we can make sure the reset is successful.
            # Preempt in reverse order so the requests will be added back to the
            # running queue in FIFO order.
            while self.running:
                request = self.running.pop()
                self._preempt_request(request, timestamp, reason="forced_prefix_reset")
                # For async scheduling, any output frames already in flight at
                # preemption time are now stale and must be discarded when they
                # return. num_output_placeholders is exactly that count: 0 if
                # the engine has drained (e.g. pause_generation(keep) waited
                # for idle), 1 for vanilla async mid-step, or 1 + spec/PP frames
                # otherwise.
                request.async_tokens_to_discard = request.num_output_placeholders
                request.num_output_placeholders = 0

            # Clear scheduled request ids cache. Since we are forcing preemption
            # + resumption in the same step, we must act as if these requests were
            # not scheduled in the prior step. They will be flushed from the
            # persistent batch in the model runner.
            self.prev_step_scheduled_req_ids.clear()

        reset_successful = self.kv_cache_manager.reset_prefix_cache()
        if reset_successful:
            self._republish_pending_demand()
            if reset_running_requests and self.predictive_admission is not None:
                # Benchmarks reset between warmup and measurement; completed
                # warmup lengths must not shape the measured forecast. Batch
                # latency curves stay valid and are kept.
                self.predictive_admission.reset_history()
                self._pkas_reject_cache.clear()
                self._pkas_epoch += 1
                logger.info("pkas_estimator_reset: reason=prefix_cache_reset")
        if reset_running_requests and not reset_successful:
            raise RuntimeError(
                "Failed to reset KV cache even when all the running requests are "
                "preempted and moved to the waiting queue. This is likely due to "
                "the presence of running requests waiting for remote KV transfer, "
                "which is not supported yet."
            )

        if reset_connector:
            reset_successful = self.reset_connector_cache() and reset_successful

        if self.preemption_diagnostics is not None:
            self.preemption_diagnostics.cache_reset(reset_successful)
        if self.survival_diagnostics is not None:
            self.survival_diagnostics.cache_reset(reset_successful)
        return reset_successful

    def reset_connector_cache(self) -> bool:
        if self.connector is None:
            logger.warning("reset_connector called but no KV connector is configured.")
            return False

        if self.connector.reset_cache() is False:
            return False

        if self.log_stats:
            assert self.connector_prefix_cache_stats is not None
            self.connector_prefix_cache_stats.reset = True

        return True

    def reset_encoder_cache(self) -> None:
        """Reset the encoder cache to invalidate all cached encoder outputs.

        This should be called when model weights are updated to ensure
        stale vision embeddings are not reused.
        """
        self.encoder_cache_manager.reset()

    def make_stats(
        self,
        spec_decoding_stats: SpecDecodingStats | None = None,
        kv_connector_stats: KVConnectorStats | None = None,
        cudagraph_stats: CUDAGraphStat | None = None,
        perf_stats: PerfStats | None = None,
    ) -> SchedulerStats | None:
        if not self.log_stats:
            return None
        prefix_cache_stats = self.kv_cache_manager.make_prefix_cache_stats()
        assert prefix_cache_stats is not None
        connector_prefix_cache_stats: PrefixCacheStats | None = None
        if self.connector_prefix_cache_stats is not None:
            connector_prefix_cache_stats = self.connector_prefix_cache_stats
            self.connector_prefix_cache_stats = PrefixCacheStats()
        eviction_events = (
            self.kv_metrics_collector.drain_events()
            if self.kv_metrics_collector is not None
            else []
        )
        spec_stats = spec_decoding_stats
        connector_stats_payload = (
            kv_connector_stats.data if kv_connector_stats else None
        )
        return SchedulerStats(
            num_running_reqs=len(self.running),
            num_waiting_reqs=len(self.waiting),
            num_skipped_waiting_reqs=len(self.skipped_waiting),
            kv_cache_usage=self.kv_cache_manager.usage,
            prefix_cache_stats=prefix_cache_stats,
            connector_prefix_cache_stats=connector_prefix_cache_stats,
            kv_cache_eviction_events=eviction_events,
            spec_decoding_stats=spec_stats,
            kv_connector_stats=connector_stats_payload,
            cudagraph_stats=cudagraph_stats,
            perf_stats=perf_stats,
        )

    def make_spec_decoding_stats(
        self,
        spec_decoding_stats: SpecDecodingStats | None,
        num_draft_tokens: int,
        num_accepted_tokens: int,
        num_invalid_spec_tokens: dict[str, int] | None,
        request_id: str,
    ) -> SpecDecodingStats | None:
        if not self.log_stats or not num_draft_tokens:
            return None
        if spec_decoding_stats is None:
            spec_decoding_stats = SpecDecodingStats.new(self.num_spec_tokens)
        if num_invalid_spec_tokens:
            num_draft_tokens -= num_invalid_spec_tokens.get(request_id, 0)
        spec_decoding_stats.observe_draft(
            num_draft_tokens=num_draft_tokens, num_accepted_tokens=num_accepted_tokens
        )
        return spec_decoding_stats

    def shutdown(self) -> None:
        for record, _ in self._pending_batch_logs.values():
            self._log_schedule_batch(record, -1)
        self._pending_batch_logs.clear()
        if self.preemption_diagnostics is not None:
            self.preemption_diagnostics.shutdown()
        if self.survival_diagnostics is not None:
            self.survival_diagnostics.shutdown()
        if self.kv_event_publisher:
            self.kv_event_publisher.shutdown()
        if self.connector is not None:
            self.connector.shutdown()

    ########################################################################
    # KV Connector Related Methods
    ########################################################################

    def get_kv_connector(self) -> KVConnectorBase_V1 | None:
        return self.connector

    def _connector_finished(
        self, request: Request
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Invoke the KV connector request_finished() method if applicable.

        Returns optional kv transfer parameters to be included with the
        request outputs.
        """
        if self.connector is None:
            return False, None

        # Free any out-of-window prefix blocks before we hand the block table to
        # the connector.
        self.kv_cache_manager.remove_skipped_blocks(
            request_id=request.request_id,
            total_computed_tokens=request.num_computed_tokens,
        )

        block_ids = self.kv_cache_manager.get_block_ids(request.request_id)

        if not isinstance(self.connector, SupportsHMA):
            # NOTE(Kuntai): We should deprecate this code path after we enforce
            # all connectors to support HMA.
            # Hybrid memory allocator should be already turned off for this
            # code path, but let's double-check here.
            assert len(self.kv_cache_config.kv_cache_groups) == 1
            return self.connector.request_finished(request, block_ids[0])

        return self.connector.request_finished_all_groups(request, block_ids)

    def _update_waiting_for_remote_kv(self, request: Request) -> None:
        """
        KV Connector: update request state after async recv is finished.

        When the kv transfer is ready, we cache the blocks
        and the request state will be moved back to WAITING from
        WAITING_FOR_REMOTE_KV.
        """
        assert self.connector is not None

        if request.request_id in self.failed_recving_kv_req_ids:
            # Request had KV load failures; num_computed_tokens was already
            # updated in _update_requests_with_invalid_blocks
            if request.num_computed_tokens:
                # Cache any valid computed tokens.
                self.kv_cache_manager.cache_blocks(request, request.num_computed_tokens)
            else:
                # No valid computed tokens, release allocated blocks.
                # There may be a local cache hit on retry.
                self.kv_cache_manager.free(request)

            self.failed_recving_kv_req_ids.remove(request.request_id)
        else:
            # Now that the blocks are ready, actually cache them.
            # This will cache the blocks iff caching is enabled.
            self.kv_cache_manager.cache_blocks(request, request.num_computed_tokens)

            # on a full prompt hit, we need to re-compute the last token
            # in order to be able to sample the next token
            if request.num_computed_tokens == request.num_tokens:
                request.num_computed_tokens = request.num_tokens - 1

        self.finished_recving_kv_req_ids.remove(request.request_id)

    def _try_promote_blocked_waiting_request(self, request: Request) -> bool:
        """
        Try to promote a blocked waiting request back to schedulable states.
        """
        if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
            # finished_recving_kv_req_ids is populated during
            # update_from_output(), based on worker-side connector signals
            # in KVConnectorOutput.finished_recving
            if request.request_id not in self.finished_recving_kv_req_ids:
                return False
            self._update_waiting_for_remote_kv(request)
            if request.num_preemptions:
                request.status = RequestStatus.PREEMPTED
            else:
                request.status = RequestStatus.WAITING
            self._set_waiting_request_ready(request)
            return True

        if request.status == RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR:
            structured_output_req = request.structured_output_request
            if not (structured_output_req and structured_output_req.grammar):
                return False
            request.status = RequestStatus.WAITING
            self._set_waiting_request_ready(request)
            return True

        if request.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
            assert not request.streaming_queue
            return False

        raise AssertionError(
            "Unexpected blocked waiting status in promotion: "
            f"{request.status.name} for request {request.request_id}"
        )

    def _update_from_kv_xfer_finished(self, kv_connector_output: KVConnectorOutput):
        """
        KV Connector: update the scheduler state based on the output.

        The Worker side connectors add finished_recving and
        finished_sending reqs to the output.
        * if finished_sending: free the blocks
        # if finished_recving: add to state so we can
            schedule the request during the next step.
        """

        if self.connector is not None:
            self.connector.update_connector_output(kv_connector_output)

        # KV Connector:: update recv and send status from last step.
        for req_id in kv_connector_output.finished_recving or ():
            logger.debug("Finished recving KV transfer for request %s", req_id)
            assert req_id in self.requests
            req = self.requests[req_id]
            if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                self.finished_recving_kv_req_ids.add(req_id)
            else:
                assert RequestStatus.is_finished(req.status)
                self._free_blocks(self.requests[req_id])
        for req_id in kv_connector_output.finished_sending or ():
            logger.debug("Finished sending KV transfer for request %s", req_id)
            assert req_id in self.requests
            self._free_blocks(self.requests[req_id])

    def _update_requests_with_invalid_blocks(
        self,
        requests: Iterable[Request],
        invalid_block_ids: set[int],
        num_scheduled_tokens: dict[str, int],
        evict_blocks: bool = True,
    ) -> tuple[set[str], int, set[int]]:
        """
        Identify and update requests affected by invalid KV cache blocks.

        This method scans the given requests, detects those with invalid blocks
        and adjusts their `num_computed_tokens` to the longest valid prefix.
        For observability, it also accumulates the total number of tokens that
        will need to be recomputed across all affected requests.

        Args:
            requests: The set of requests to scan for invalid blocks.
            invalid_block_ids: IDs of invalid blocks.
            num_scheduled_tokens: req_id -> number of scheduled tokens.
            evict_blocks: Whether to collect blocks for eviction (False for
                async requests which aren't cached yet).

        Returns:
            tuple:
                - affected_req_ids (set[str]): IDs of requests impacted by
                invalid blocks.
                - total_affected_tokens (int): Total number of tokens that must
                be recomputed across all affected requests.
                - blocks_to_evict (set[int]): Block IDs to evict from cache,
                including invalid blocks and downstream dependent blocks.
        """
        affected_req_ids: set[str] = set()
        total_affected_tokens = 0
        blocks_to_evict: set[int] = set()
        # If a block is invalid and shared by multiple requests in the batch,
        # these requests must be rescheduled, but only the first will recompute
        # it. This set tracks blocks already marked for recomputation.
        marked_invalid_block_ids: set[int] = set()
        for request in requests:
            is_affected = False
            marked_invalid_block = False
            req_id = request.request_id
            # TODO (davidb): add support for hybrid memory allocator
            (req_block_ids,) = self.kv_cache_manager.get_block_ids(req_id)
            # We iterate only over blocks that may contain externally computed
            # tokens
            req_num_computed_tokens = (
                request.num_computed_tokens - num_scheduled_tokens.get(req_id, 0)
            )

            req_num_computed_blocks = (
                req_num_computed_tokens + self.block_size - 1
            ) // self.block_size
            for idx, block_id in zip(range(req_num_computed_blocks), req_block_ids):
                if block_id not in invalid_block_ids:
                    continue

                is_affected = True

                if block_id in marked_invalid_block_ids:
                    # This invalid block is shared with a previous request
                    # and was already marked for recomputation.
                    # This means this request can still consider this block
                    # as computed when rescheduled.
                    # Currently this only applies to sync loading; Async
                    # loading does not yet support block sharing
                    continue

                marked_invalid_block_ids.add(block_id)

                if marked_invalid_block:
                    # This request has already marked an invalid block for
                    # recomputation and updated its num_computed_tokens.
                    continue

                marked_invalid_block = True
                # Truncate the computed tokens at the first failed block
                request.num_computed_tokens = idx * self.block_size
                num_affected_tokens = (
                    req_num_computed_tokens - request.num_computed_tokens
                )
                total_affected_tokens += num_affected_tokens

                # collect invalid block and all downstream dependent blocks
                if evict_blocks:
                    blocks_to_evict.update(req_block_ids[idx:])

            if is_affected:
                if not marked_invalid_block:
                    # All invalid blocks of this request are shared with
                    # previous requests and will be recomputed by them.
                    # Revert to considering only cached tokens as computed.
                    # Currently this only applies to sync loading; Async
                    # loading does not yet support block sharing
                    total_affected_tokens += (
                        request.num_computed_tokens - req_num_computed_tokens
                    )
                    request.num_computed_tokens = req_num_computed_tokens

                affected_req_ids.add(request.request_id)

        return affected_req_ids, total_affected_tokens, blocks_to_evict

    def _handle_invalid_blocks(
        self, invalid_block_ids: set[int], num_scheduled_tokens: dict[str, int]
    ) -> set[str]:
        """
        Handle requests affected by invalid KV cache blocks.

        Returns:
            Set of affected request IDs to skip in update_from_output main loop.
        """
        should_fail = not self.recompute_kv_load_failures

        # handle async KV loads (not cached yet, evict_blocks=False)
        async_load_reqs = (
            req
            for req in self.skipped_waiting
            if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS
        )
        async_failed_req_ids, num_failed_tokens, _ = (
            self._update_requests_with_invalid_blocks(
                async_load_reqs,
                invalid_block_ids,
                num_scheduled_tokens,
                evict_blocks=False,
            )
        )

        total_failed_requests = len(async_failed_req_ids)
        total_failed_tokens = num_failed_tokens

        # handle sync loads (may be cached, collect blocks for eviction)
        sync_failed_req_ids, num_failed_tokens, sync_blocks_to_evict = (
            self._update_requests_with_invalid_blocks(
                self.running, invalid_block_ids, num_scheduled_tokens, evict_blocks=True
            )
        )

        total_failed_requests += len(sync_failed_req_ids)
        total_failed_tokens += num_failed_tokens

        if not total_failed_requests:
            return set()

        # evict invalid blocks and downstream dependent blocks from cache
        # only when not using recompute policy (where blocks will be recomputed
        # and reused by other requests sharing them)
        if sync_blocks_to_evict and not self.recompute_kv_load_failures:
            self.kv_cache_manager.evict_blocks(sync_blocks_to_evict)

        if should_fail:
            all_failed_req_ids = async_failed_req_ids | sync_failed_req_ids
            logger.error(
                "Failing %d request(s) due to KV load failure "
                "(failure_policy=fail, %d tokens affected). Request IDs: %s",
                total_failed_requests,
                total_failed_tokens,
                all_failed_req_ids,
            )
            return all_failed_req_ids

        logger.warning(
            "Recovered from KV load failure: "
            "%d request(s) rescheduled (%d tokens affected).",
            total_failed_requests,
            total_failed_tokens,
        )

        # Mark async requests with KV load failures for retry once loading completes
        self.failed_recving_kv_req_ids |= async_failed_req_ids
        # Return sync affected IDs to skip in update_from_output
        return sync_failed_req_ids
