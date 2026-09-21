# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-owned, observational preemption diagnostics (JSONL schema v1).

This module deliberately has no engine imports; shadow selection is pure Python.
"""

import functools
import json
import logging
import os
import time
import uuid
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)


def simulate(candidates, references, deficit: int, *, tail: bool = False) -> dict:
    """Simulate releases, preserving references outside the candidate set."""
    refs = references.copy()
    remaining = list(candidates)
    victims = []
    released = 0
    while remaining and released < deficit:
        margins = {
            c["request_id"]: sum(refs[b] == n for b, n in Counter(c["blocks"]).items())
            for c in remaining
        }
        sufficient = [
            c for c in remaining
            if margins[c["request_id"]] >= deficit - released
        ]
        if tail:
            chosen = max(remaining, key=lambda c: c["position"])
        elif sufficient:
            chosen = min(
                sufficient, key=lambda c: (margins[c["request_id"]], -c["position"])
            )
        else:
            chosen = max(
                remaining, key=lambda c: (margins[c["request_id"]], c["position"])
            )
        marginal = margins[chosen["request_id"]]
        for block in chosen["blocks"]:
            refs[block] -= 1
        victims.append({"request_id": chosen["request_id"], "released": marginal})
        released += marginal
        remaining.remove(chosen)
    return dict(
        victims=victims,
        victim_count=len(victims),
        released=released,
        excess=max(0, released - deficit),
        remaining_deficit=max(0, deficit - released),
        feasible=released >= deficit,
    )


def observed(method):
    """Fail closed for diagnostics, never for serving; account exclusive work."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        if not self.enabled:
            return None
        start = time.perf_counter_ns()
        other = sum(self.cost.values())
        try:
            return method(self, *args, **kwargs)
        except Exception as exc:
            self.disable(exc)
            return None
        finally:
            self.cost["collection_ns"] += max(
                0, time.perf_counter_ns() - start - (sum(self.cost.values()) - other)
            )
    return wrapper


class PreemptionDiagnostics:
    def __init__(self, directory, configuration, unsupported=()):
        self.instance_id = uuid.uuid4().hex
        self.step = 0
        self.boundary_id = 0
        self.enabled = True
        self.unsupported = list(unsupported)
        self.cost = Counter()
        self.totals = Counter()
        self.requests = {}
        self.episode = None
        self.pending = None
        self.pressure = None
        self.stream = None
        try:
            Path(directory).mkdir(parents=True, exist_ok=True)
            filename = f"preemption-{os.getpid()}-{self.instance_id}.jsonl"
            self.stream = (Path(directory) / filename).open("x")
            self.emit(
                "startup", configuration=configuration, unsupported=self.unsupported
            )
            self.flush()
        except Exception as exc:
            self.disable(exc)

    def disable(self, exc):
        if self.enabled:
            logger.warning("Preemption diagnostics disabled: %s", exc)
        self.enabled = False
        self.requests.clear()
        self.episode = self.pending = None
        if self.stream is not None:
            try:
                self.stream.close()
            except OSError:
                pass
            self.stream = None

    def emit(self, kind, **fields):
        start = time.perf_counter_ns()
        line = json.dumps(dict(
            version=1, kind=kind, instance_id=self.instance_id,
            step_id=self.step, timestamp=time.monotonic(), **fields,
        )) + "\n"
        self.cost["serialization_ns"] += time.perf_counter_ns() - start
        start = time.perf_counter_ns()
        self.stream.write(line)
        self.cost["io_ns"] += time.perf_counter_ns() - start

    def flush(self):
        start = time.perf_counter_ns()
        self.stream.flush()
        self.cost["io_ns"] += time.perf_counter_ns() - start

    @observed
    def cache_reset(self, succeeded):
        self.boundary_id += 1
        self.emit(
            "cache_reset", boundary_id=self.boundary_id, succeeded=succeeded,
            active_requests=len(self.requests), outstanding_batch=self.pending is not None,
            open_episode=self.episode is not None, totals=dict(self.totals),
            overhead=dict(self.cost), unsupported=list(self.unsupported),
        )
        self.flush()

    @observed
    def begin_step(self):
        self.step += 1
        self.step_start = time.perf_counter_ns()
        self.totals["steps"] += 1
        if self.pending is not None:
            self.invalidate("overlapping_batches")

    def invalidate(self, reason):
        self.unsupported.append(reason)
        self.emit("invalidated", reason=reason)
        self.pending = None
        for state in self.requests.values():
            self.close_recovery(state, "invalidated")

    @observed
    def end_step(self):
        self.totals["scheduler_wall_ns"] += time.perf_counter_ns() - self.step_start
        self.emit("step", totals=dict(self.totals), overhead=dict(self.cost))
        self.flush()

    @observed
    def add_request(self, request):
        reasons = []
        if request.resumable:
            reasons.append("streaming_input")
        if request.sampling_params is None:
            reasons.append("non_generation")
        if request.has_encoder_inputs:
            reasons.append("multimodal")
        self.totals["requests"] += 1
        self.requests[request.request_id] = dict(
            request_id=request.request_id, high=0, last_output=None, recovery=None,
            preemptions=0, pressure_preemptions=0, unsupported=reasons,
            added=time.monotonic(), last_recovered=None,
        )
        self.emit("request_start", request_id=request.request_id, unsupported=reasons)

    @observed
    def allocation_failed(self, request_id, required, available):
        self.pressure = dict(
            request_id=request_id, required=required, available=available,
            deficit=required - available,
        )

    @observed
    def admission_failed(self, request_id, required, available):
        self.totals["admission_failures"] += 1

    @observed
    def start_episode(self, scheduler, request, num_tokens, scheduled):
        if self.episode is not None:
            self.totals["allocation_retries_failed"] += 1
            return
        self.totals["episodes"] += 1
        episode = dict(
            episode_id=self.totals["episodes"], **self.pressure,
            policy=scheduler.policy.value, requested_tokens=num_tokens,
            running_count=len(scheduler.running), waiting_count=len(scheduler.waiting),
            actual=[], unsupported=list(self.unsupported),
        )
        self.episode = episode
        if scheduler.policy.value == "priority":
            episode["unsupported"].append("priority_shadow")
        if any(self.requests[r.request_id]["unsupported"] for r in scheduler.running):
            episode["unsupported"].append("unsupported_running_request")
        if episode["unsupported"]:
            return
        start = time.perf_counter_ns()
        order, candidates, refs = [], [], {}
        blocked_position = scheduler.running.index(request)
        now = time.monotonic()
        for position, req in enumerate(scheduler.running):
            groups = scheduler.kv_cache_manager.get_blocks(req.request_id).blocks
            blocks = [b for group in groups for b in group if not b.is_null]
            ids = [b.block_id for b in blocks]
            refs.update((b.block_id, b.ref_cnt) for b in blocks)
            ownership = Counter(ids)
            row = dict(
                request_id=req.request_id, position=position, blocks=ids,
                eligible=(
                    position > blocked_position and req.request_id not in scheduled
                ),
                prompt_tokens=req.num_prompt_tokens,
                generated_tokens=req.num_output_tokens,
                computed_tokens=req.num_computed_tokens,
                age_s=now - self.requests[req.request_id]["added"],
                previous_preemptions=req.num_preemptions,
                owned_blocks=len(ownership),
                shared_blocks=sum(refs[b] > n for b, n in ownership.items()),
                reclaimable_blocks=sum(refs[b] == n for b, n in ownership.items()),
            )
            order.append(row)
            if row["eligible"]:
                candidates.append(row)
        episode.update(running_order=order, references=refs)
        self.totals["candidate_blocks_scanned"] += sum(len(r["blocks"]) for r in order)
        self.cost["candidate_scan_ns"] += time.perf_counter_ns() - start
        start = time.perf_counter_ns()
        episode["tail"] = simulate(candidates, refs, episode["deficit"], tail=True)
        episode["shadow"] = simulate(candidates, refs, episode["deficit"])
        self.cost["simulation_ns"] += time.perf_counter_ns() - start

    @observed
    def end_episode(self, proceeded):
        if self.episode is None:
            return
        episode = self.episode
        episode["proceeded"] = proceeded
        episode["self_preemption"] = any(
            v["request_id"] == episode["request_id"] for v in episode["actual"]
        )
        episode["actual_released"] = sum(v["released"] for v in episode["actual"])
        episode["comparable"] = bool(
            not episode["unsupported"] and proceeded
            and episode["shadow"]["feasible"]
            and episode["actual_released"] >= episode["deficit"]
        )
        if not episode["unsupported"]:
            actual = [
                {"request_id": v["request_id"], "released": v["released"]}
                for v in episode["actual"]
                if v["request_id"] != episode["request_id"]
            ]
            episode["tail_matches_actual"] = actual == episode["tail"]["victims"]
            if not episode["tail_matches_actual"]:
                self.totals["accounting_mismatches"] += 1
                episode["comparable"] = False
        self.emit("episode", **episode)
        self.episode = None

    @observed
    def preempt(self, request, free_before, free_after, reason):
        now = time.monotonic()
        state = self.requests[request.request_id]
        if self.pending is not None and request.request_id in self.pending[1]:
            self.invalidate("preemption_with_outstanding_batch")
        ordinal = request.num_preemptions
        self.totals["preemptions"] += 1
        if reason == "memory_pressure":
            if not state["pressure_preemptions"]:
                self.totals["requests_preempted"] += 1
            state["pressure_preemptions"] += 1
            self.totals["pressure_preemptions"] += 1
        victim = dict(
            request_id=request.request_id, ordinal=ordinal, reason=reason,
            free_before=free_before, free_after=free_after,
            released=free_after - free_before,
        )
        if self.episode is not None and reason == "memory_pressure":
            self.episode["actual"].append(victim)
        self.emit(
            "preemption",
            episode_id=self.episode["episode_id"] if self.episode else None,
            **victim,
        )
        state["preemptions"] += 1
        self.close_recovery(state, "interrupted_recovery")
        if self.unsupported or state["unsupported"]:
            return
        state["recovery"] = dict(
            ordinal=ordinal, reason=reason, preempted_at=now,
            historical_tokens=state["high"],
            resumed_at=None, recovered_at=None, first_output_at=None,
            last_output_at=state["last_output"], prompt_replay=0, output_replay=0,
            since_previous_recovery_s=(
                now - state["last_recovered"]
                if state["last_recovered"] is not None else None
            ),
        )

    @observed
    def resumed(self, request, local_hit_tokens):
        if self.unsupported:
            return
        recovery = self.requests[request.request_id]["recovery"]
        if recovery is not None:
            recovery["resumed_at"] = time.monotonic()
            recovery["local_prefix_hit_tokens"] = local_hit_tokens
            if local_hit_tokens >= recovery["historical_tokens"]:
                recovery["recovered_at"] = recovery["resumed_at"]
                self.requests[request.request_id]["last_recovered"] = (
                    recovery["recovered_at"]
                )

    @observed
    def scheduled(self, output, requests):
        if self.unsupported or not output.num_scheduled_tokens:
            return
        self.pending = (id(output), {
            rid: (
                requests[rid].num_computed_tokens,
                requests[rid].num_computed_tokens + n,
            )
            for rid, n in output.num_scheduled_tokens.items()
            if not self.requests[rid]["unsupported"]
        })

    @observed
    def completed(self, output, request, has_output):
        if self.unsupported or self.pending is None:
            return
        batch_id, intervals = self.pending
        if batch_id != id(output):
            raise ValueError("model output does not match diagnostic batch")
        interval = intervals.pop(request.request_id, None)
        if interval is None:
            return
        start, end = interval
        state = self.requests[request.request_id]
        self.totals["completed_scheduled_tokens"] += end - start
        state["high"] = max(state["high"], end)
        now = time.monotonic()
        recovery = state["recovery"]
        if recovery is not None:
            replay_end = min(end, recovery["historical_tokens"])
            prompt = max(0, min(replay_end, request.num_prompt_tokens) - start)
            generated = max(0, replay_end - max(start, request.num_prompt_tokens))
            recovery["prompt_replay"] += prompt
            recovery["output_replay"] += generated
            if recovery["reason"] == "memory_pressure":
                self.totals["prompt_replay"] += prompt
                self.totals["output_replay"] += generated
            if (
                recovery["recovered_at"] is None
                and end >= recovery["historical_tokens"]
            ):
                recovery["recovered_at"] = now
                state["last_recovered"] = now
            if has_output and recovery["first_output_at"] is None:
                recovery["first_output_at"] = now
        if has_output:
            state["last_output"] = now

    @observed
    def end_output(self, output):
        if self.pending is not None and self.pending[0] == id(output):
            # Cancelled requests may have left the scheduler while the model ran.
            self.pending = None
        self.emit("progress", totals=dict(self.totals))
        self.flush()

    def close_recovery(self, state, outcome):
        recovery = state["recovery"]
        if recovery is None:
            return
        recovered = recovery["recovered_at"] is not None
        if recovered and outcome == "interrupted_recovery":
            outcome = "recovered_before_next_preemption"
        self.emit("recovery", request_id=state["request_id"], outcome=outcome,
                  recovered=recovered, **recovery)
        state["recovery"] = None

    @observed
    def finish(self, request):
        state = self.requests.pop(request.request_id, None)
        if state is None:
            return
        outcome = {
            "FINISHED_ABORTED": "cancelled",
            "FINISHED_ERROR": "error",
            "FINISHED_IGNORED": "ignored",
        }.get(request.status.name, "completed")
        self.close_recovery(state, outcome)
        self.emit(
            "request_end", request_id=request.request_id, outcome=outcome,
            preemptions=state["preemptions"],
            pressure_preemptions=state["pressure_preemptions"],
        )

    @observed
    def shutdown(self):
        for state in self.requests.values():
            self.close_recovery(state, "unfinished_at_shutdown")
        self.emit("shutdown", totals=dict(self.totals), overhead=dict(self.cost),
                  unfinished_requests=len(self.requests), unsupported=self.unsupported)
        self.flush()
        self.stream.close()
        self.requests.clear()
        self.pending = self.episode = None
        self.enabled = False
