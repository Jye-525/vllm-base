# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Differential test for the ``Scheduler.schedule()`` refactor.

The reference is ``vllm/v1/core/sched/scheduler.py`` as of ``BASE_COMMIT``
(the last commit before the refactor), loaded from git and executed as a
separate module. Both schedulers are driven with the same deterministic
request stream and fake model output under a fixed fake clock. Every step
must produce the same:

- ``SchedulerOutput`` (scheduled tokens in order, new/cached/resumed/
  preempted/finished requests, block ids, common prefix blocks),
- running, waiting and skipped-waiting queue orders and free KV blocks,
- ``schedule_batch`` records (field order included; ``*_ns`` timing
  fields excluded), and
- INFO log lines (``admitted_prefix_cache``, ``preempted``, ...).

The configurations cover FCFS, PRIORITY, LPM (admission skips with
skip-aware eviction, per-admission re-ranking), the full Peek profile (cLPM
over the whole queue with Peek ``cluster`` eviction), SPF, the SSPF series
(session credit, demand-aware retention, least-done preemption) and PKAS.
"""

from __future__ import annotations

import random
import subprocess
import sys
import time
import types
from pathlib import Path
from typing import Any

import pytest

import vllm.v1.core.sched.scheduler as scheduler_module
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

from .utils import EOS_TOKEN_ID, create_scheduler

# Last vllm-kvreuse commit before the schedule() refactor.
BASE_COMMIT = "cc8b41234a075ef4c8b6109bb5fd788db3f3e669"
SCHEDULER_PATH = "vllm/v1/core/sched/scheduler.py"
REPO_ROOT = Path(__file__).resolve().parents[3]

BS = 16
NUM_BLOCKS = 64
MAX_NUM_SEQS = 8
MAX_BATCHED_TOKENS = 192
MAX_MODEL_LEN = 512
NUM_REQUESTS = 40
MAX_STEPS = 600
STEP_SECONDS = 0.05
CLOCK_START = 1_000_000.0

COMMON = dict(
    enable_prefix_caching=True,
    num_blocks=NUM_BLOCKS,
    block_size=BS,
    max_num_seqs=MAX_NUM_SEQS,
    max_num_batched_tokens=MAX_BATCHED_TOKENS,
    max_model_len=MAX_MODEL_LEN,
)

CONFIGS: dict[str, dict[str, Any]] = {
    "fcfs": dict(scheduling_policy="fcfs"),
    "fcfs_no_prefix_caching": dict(
        scheduling_policy="fcfs", enable_prefix_caching=False
    ),
    "fcfs_long_prefill_threshold": dict(
        scheduling_policy="fcfs", long_prefill_token_threshold=48
    ),
    "priority": dict(scheduling_policy="priority"),
    "lpm": dict(scheduling_policy="lpm", lpm_window_size=0),
    "lpm_window_4": dict(scheduling_policy="lpm", lpm_window_size=4),
    "lpm_skips_skip_eviction": dict(
        scheduling_policy="lpm",
        lpm_window_size=0,
        lpm_admission_skips=3,
        kv_cache_eviction_policy="skip",
    ),
    "lpm_rank_per_admission": dict(
        scheduling_policy="lpm", lpm_window_size=0, rank_per_admission=True
    ),
    "lpm_least_done": dict(
        scheduling_policy="lpm", lpm_window_size=0, preemption_victim="least_done"
    ),
    "peek_full": dict(
        scheduling_policy="clpm",
        lpm_window_size=0,
        lpm_fallback_threshold=0,
        clpm_group_major=True,
        clpm_dynamic_lane=True,
        kv_cache_eviction_policy="peek",
        peek_eviction_mode="cluster",
    ),
    "peek_eviction_fcfs": dict(
        scheduling_policy="fcfs",
        kv_cache_eviction_policy="peek",
        peek_eviction_mode="plain",
    ),
    "spf_aged": dict(
        scheduling_policy="spf",
        lpm_window_size=0,
        spf_age_alpha=50.0,
        rank_preempted_requests=True,
        rank_per_admission=True,
        preemption_victim="least_done",
    ),
    "sspf": dict(scheduling_policy="sspf", lpm_window_size=0),
    "sspf_series": dict(
        scheduling_policy="sspf",
        lpm_window_size=0,
        sspf_session_credit_weight=0.5,
        kv_cache_eviction_policy="demand",
        preemption_victim="least_done",
    ),
    "pkas_paper": dict(scheduling_policy="lpm", admission_policy="pkas"),
    "pkas_repo": dict(
        scheduling_policy="lpm", admission_policy="pkas", pkas_simulation="repo"
    ),
}


@pytest.fixture(autouse=True, scope="module")
def _fixed_none_hash():
    init_none_hash(sha256)


@pytest.fixture(scope="module")
def reference_module() -> types.ModuleType:
    """The pre-refactor scheduler module, executed from git."""
    try:
        proc = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "show", f"{BASE_COMMIT}:{SCHEDULER_PATH}"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:  # git missing
        pytest.skip(f"git unavailable: {exc}")
    if proc.returncode != 0:
        pytest.skip(f"reference scheduler not in git: {proc.stderr.strip()}")
    name = "vllm.v1.core.sched._reference_scheduler"
    module = types.ModuleType(name)
    module.__file__ = f"<{BASE_COMMIT[:10]}>/{SCHEDULER_PATH}"
    sys.modules[name] = module
    try:
        exec(compile(proc.stdout, module.__file__, "exec"), module.__dict__)
        yield module
    finally:
        sys.modules.pop(name, None)


class _Clock:
    def __init__(self) -> None:
        self.now = CLOCK_START

    def time(self) -> float:
        return self.now

    def tick(self) -> None:
        self.now += STEP_SECONDS


class _InfoRecorder:
    """Stands in for a module logger: records INFO lines, forwards the rest."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.lines: list[str] = []

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.lines.append(msg % args if args else msg)

    def isEnabledFor(self, level: int) -> bool:  # noqa: N802
        return True

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _workload(seed: int) -> list[tuple[int, dict[str, Any]]]:
    """(arrival step, request kwargs): prefix families, bursty arrivals."""
    rng = random.Random(seed)
    families = [
        list(range(1 + 500 * f, 1 + 500 * f + rng.randint(32, 96)))
        for f in range(6)
    ]
    specs: list[tuple[int, dict[str, Any]]] = []
    step = 0
    for i in range(NUM_REQUESTS):
        if rng.random() < 0.4:
            step += rng.randint(1, 6)
        family = rng.randrange(len(families))
        prefix = families[family][: rng.randint(16, len(families[family]))]
        suffix = [20000 + 100 * i + j for j in range(rng.randint(1, 60))]
        specs.append(
            (
                step,
                dict(
                    request_id=f"r{i:02d}",
                    tokens=prefix + suffix,
                    max_tokens=rng.randint(1, 20),
                    priority=rng.randint(0, 3),
                    session_id=f"s{family}",
                    arrival_time=CLOCK_START + step * STEP_SECONDS,
                ),
            )
        )
    return specs


def _request(spec: dict[str, Any]) -> Request:
    params = SamplingParams(
        max_tokens=spec["max_tokens"],
        ignore_eos=True,
        extra_args={"session_id": spec["session_id"]},
    )
    params.update_from_generation_config({}, EOS_TOKEN_ID)
    return Request(
        request_id=spec["request_id"],
        prompt_token_ids=list(spec["tokens"]),
        sampling_params=params,
        pooling_params=None,
        arrival_time=spec["arrival_time"],
        priority=spec["priority"],
        block_hasher=get_request_block_hasher(BS, sha256),
    )


def _model_output(scheduler: Scheduler, output) -> ModelRunnerOutput:
    """One sampled token for every request whose prefill has completed."""
    req_ids = list(output.num_scheduled_tokens)
    sampled = []
    for rid in req_ids:
        request = scheduler.requests[rid]
        done = request.num_computed_tokens >= request.num_tokens
        sampled.append([7] if done else [])
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
        sampled_token_ids=sampled,
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


def _block_ids(block_ids) -> list[list[int]] | None:
    return None if block_ids is None else [list(group) for group in block_ids]


def _step_trace(scheduler: Scheduler, output) -> dict[str, Any]:
    cached = output.scheduled_cached_reqs
    return dict(
        num_scheduled_tokens=list(output.num_scheduled_tokens.items()),
        total_num_scheduled_tokens=output.total_num_scheduled_tokens,
        new=[
            (r.req_id, r.num_computed_tokens, _block_ids(r.block_ids))
            for r in output.scheduled_new_reqs
        ],
        cached_req_ids=list(cached.req_ids),
        resumed_req_ids=sorted(cached.resumed_req_ids),
        cached_new_block_ids=[_block_ids(b) for b in cached.new_block_ids],
        cached_num_computed_tokens=list(cached.num_computed_tokens),
        cached_num_output_tokens=list(cached.num_output_tokens),
        spec_decode_tokens=dict(output.scheduled_spec_decode_tokens),
        encoder_inputs=dict(output.scheduled_encoder_inputs),
        common_prefix_blocks=list(output.num_common_prefix_blocks),
        preempted=sorted(output.preempted_req_ids),
        finished=sorted(output.finished_req_ids),
        running=[r.request_id for r in scheduler.running],
        waiting=[r.request_id for r in scheduler.waiting],
        skipped_waiting=[r.request_id for r in scheduler.skipped_waiting],
        free_blocks=scheduler.kv_cache_manager.block_pool.get_num_free_blocks(),
    )


def _deterministic(record: dict[str, Any], recomputed: int) -> list[Any]:
    items: list[Any] = []
    for key, value in record.items():
        if key.endswith("_ns"):
            continue
        if key == "pkas" and value is not None:
            value = [(k, v) for k, v in value.items() if not k.endswith("_ns")]
        items.append((key, value))
    items.append(("recomputed_tokens", recomputed))
    return items


def _run(
    scheduler_cls: type,
    module: types.ModuleType,
    config: dict[str, Any],
    seed: int,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    clock = _Clock()
    with monkeypatch.context() as patch:
        patch.setattr(time, "time", clock.time)
        patch.setattr(time, "monotonic", clock.time)

        template = create_scheduler(**{**COMMON, **config})
        scheduler = scheduler_cls(
            vllm_config=template.vllm_config,
            kv_cache_config=template.kv_cache_config,
            block_size=template.block_size,
            log_stats=True,
            structured_output_manager=StructuredOutputManager(template.vllm_config),
        )
        if scheduler._retention is not None:
            scheduler._retention.clock = clock.time

        records: list[list[Any]] = []

        def capture(record: dict[str, Any], recomputed: int) -> None:
            records.append(_deterministic(record, recomputed))

        patch.setattr(scheduler_cls, "_log_schedule_batch", staticmethod(capture))
        recorder = _InfoRecorder(module.logger)
        patch.setattr(module, "logger", recorder)

        pending = list(_workload(seed))
        steps: list[dict[str, Any]] = []
        for step in range(MAX_STEPS):
            while pending and pending[0][0] <= step:
                scheduler.add_request(_request(pending.pop(0)[1]))
            if not pending and not scheduler.has_unfinished_requests():
                break
            clock.tick()
            output = scheduler.schedule()
            steps.append(_step_trace(scheduler, output))
            scheduler.update_from_output(output, _model_output(scheduler, output))
        # Flush the last pending schedule_batch line.
        scheduler._flush_pending_batch_logs()
        return dict(steps=steps, records=records, info=recorder.lines)


def _first_difference(reference: list[Any], refactored: list[Any]) -> str:
    for i, (a, b) in enumerate(zip(reference, refactored)):
        if a != b:
            return f"index {i}:\n reference  = {a!r}\n refactored = {b!r}"
    return f"lengths differ: reference={len(reference)} refactored={len(refactored)}"


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("config_name", list(CONFIGS))
def test_schedule_matches_pre_refactor_reference(
    reference_module, monkeypatch, config_name, seed
):
    config = CONFIGS[config_name]
    reference = _run(
        reference_module.Scheduler, reference_module, config, seed, monkeypatch
    )
    refactored = _run(Scheduler, scheduler_module, config, seed, monkeypatch)

    assert reference["steps"], "the workload scheduled nothing"
    for part in ("steps", "records", "info"):
        assert refactored[part] == reference[part], (
            f"{config_name} seed={seed}: {part} diverged at "
            + _first_difference(reference[part], refactored[part])
        )
