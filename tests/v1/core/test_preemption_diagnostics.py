# SPDX-License-Identifier: Apache-2.0
"""CPU checks; can run with --confcutdir=tests/v1/core without model fixtures."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from vllm.v1.core.sched.preemption_diagnostics import PreemptionDiagnostics, simulate


def load_report():
    path = Path(__file__).resolve().parents[3] / "benchmarks/analyze_preemption_diagnostics.py"
    spec = importlib.util.spec_from_file_location("preemption_report", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def candidate(name, position, *blocks):
    return dict(request_id=name, position=position, blocks=list(blocks))


@pytest.mark.parametrize("deficit,expected,released", [(2, ["b"], 2), (3, ["a"], 4), (6, ["a", "b"], 6), (8, ["a", "b", "c"], 7)])
def test_smallest_sufficient_and_largest_fallback(deficit, expected, released):
    candidates = [candidate("a", 1, 1, 2, 3, 4), candidate("b", 2, 5, 6), candidate("c", 3, 7)]
    refs = dict.fromkeys(range(1, 8), 1)
    result = simulate(candidates, refs, deficit)
    assert [v["request_id"] for v in result["victims"]] == expected
    assert result["released"] == released
    assert result["feasible"] == (deficit <= 7)
    assert refs == dict.fromkeys(range(1, 8), 1)
    assert len(candidates) == 3


def test_ties_and_collective_sharers_and_external_references():
    candidates = [candidate("a", 1, 1, 2), candidate("b", 2, 1, 2)]
    result = simulate(candidates, {1: 2, 2: 3}, 1)
    assert result["victims"] == [dict(request_id="b", released=0), dict(request_id="a", released=1)]
    assert not simulate(candidates, {1: 2, 2: 3}, 2)["feasible"]
    assert simulate([candidate("a", 1, 1), candidate("b", 2, 2)], {1: 1, 2: 1}, 1)["victims"][0]["request_id"] == "b"


def request(name="r", prompt=10):
    return NS(request_id=name, resumable=False, sampling_params=object(), has_encoder_inputs=False,
              num_computed_tokens=0, num_preemptions=0, num_prompt_tokens=prompt,
              num_output_tokens=0, status=NS(name="FINISHED_STOPPED"))


def records(directory):
    return [json.loads(line) for path in directory.glob("*.jsonl") for line in path.read_text().splitlines()]


def run_interval(diag, req, start, end, output=False):
    batch = NS(num_scheduled_tokens={req.request_id: end-start})
    req.num_computed_tokens = start
    diag.begin_step()
    diag.scheduled(batch, {req.request_id: req})
    req.num_computed_tokens = end
    diag.end_step()
    diag.completed(batch, req, output)
    diag.end_output(batch)


def test_completed_replay_boundaries_repeated_preemption_and_prefix_hits(tmp_path):
    diag = PreemptionDiagnostics(tmp_path, {})
    req = request()
    diag.add_request(req)
    run_interval(diag, req, 0, 14, output=True)
    req.num_preemptions = 1
    diag.preempt(req, 0, 2, "memory_pressure")
    diag.admission_failed("r", 3, 1)
    assert diag.requests["r"]["recovery"]["resumed_at"] is None
    diag.resumed(req, 4)
    run_interval(diag, req, 4, 8)
    req.num_preemptions = 2
    diag.preempt(req, 0, 2, "memory_pressure")
    assert diag.requests["r"]["recovery"]["historical_tokens"] == 14
    diag.resumed(req, 6)
    run_interval(diag, req, 6, 12)
    run_interval(diag, req, 12, 15, output=True)
    diag.finish(req)
    assert not diag.requests
    diag.shutdown()
    rows = records(tmp_path)
    recovery = [r for r in rows if r["kind"] == "recovery"]
    assert [(r["prompt_replay"], r["output_replay"]) for r in recovery] == [(4, 0), (4, 4)]
    assert recovery[0]["outcome"] == "interrupted_recovery"
    assert recovery[1]["recovered"]
    assert recovery[1]["first_output_at"] >= recovery[1]["resumed_at"]
    summary, *_ = load_report().analyze(rows)
    assert summary["completed_replay_tokens"] == 12
    assert summary["completed_scheduled_tokens"] == 27
    assert summary["fraction_requests_preempted"] == 1
    assert summary["repeated_pressure_preemptions"] == 1
    assert summary["incomplete_replay_tokens"] == 4


def test_cancelled_inflight_not_counted_and_unfinished(tmp_path):
    diag = PreemptionDiagnostics(tmp_path, {})
    req = request()
    diag.add_request(req)
    run_interval(diag, req, 0, 10)
    req.num_preemptions = 1
    diag.preempt(req, 0, 1, "memory_pressure")
    diag.resumed(req, 0)
    diag.begin_step()
    batch = NS(num_scheduled_tokens={"r": 5})
    req.num_computed_tokens = 0
    diag.scheduled(batch, {"r": req})
    req.status.name = "FINISHED_ABORTED"
    diag.finish(req)
    diag.end_output(batch)
    other = request("other")
    diag.add_request(other)
    other.num_preemptions = 1
    diag.preempt(other, 0, 1, "memory_pressure")
    diag.shutdown()
    recovery = [r for r in records(tmp_path) if r["kind"] == "recovery"]
    assert [r["outcome"] for r in recovery] == ["cancelled", "unfinished_at_shutdown"]
    assert all(r["prompt_replay"] == 0 for r in recovery)
    assert not diag.requests and diag.pending is None


@pytest.mark.parametrize("reason", ["forced_prefix_reset", "memory_pressure"])
def test_full_prefix_survives_and_completion_without_output(tmp_path, reason):
    diag = PreemptionDiagnostics(tmp_path, {})
    req = request()
    diag.add_request(req)
    run_interval(diag, req, 0, 10)
    req.num_preemptions = 1
    diag.preempt(req, 0, 1, reason)
    diag.resumed(req, 10)
    diag.finish(req)
    diag.shutdown()
    row = next(r for r in records(tmp_path) if r["kind"] == "recovery")
    assert row["recovered"] and row["first_output_at"] is None
    assert row["prompt_replay"] == 0
    summary, *_ = load_report().analyze(records(tmp_path))
    assert summary["completed_replay_tokens"] == 0
    assert summary["successful_recoveries"] == (reason == "memory_pressure")


def test_episode_snapshot_eligibility_null_and_actual_tail(tmp_path):
    diag = PreemptionDiagnostics(tmp_path, {})
    reqs = [request(str(i)) for i in range(5)]
    for req in reqs:
        diag.add_request(req)
    blocks = {str(i): [NS(block_id=i, ref_cnt=1, is_null=False), NS(block_id=99, ref_cnt=100, is_null=True)] for i in range(5)}
    scheduler = NS(policy=NS(value="fcfs"), running=reqs, waiting=[],
                   kv_cache_manager=NS(get_blocks=lambda rid: NS(blocks=(blocks[rid],))))
    diag.begin_step()
    diag.allocation_failed("1", 3, 1)
    diag.start_episode(scheduler, reqs[1], 20, {"0": 1, "3": 1})
    assert [r["request_id"] for r in diag.episode["running_order"] if r["eligible"]] == ["2", "4"]
    assert "99" not in diag.episode["references"] and 99 not in diag.episode["references"]
    assert diag.totals["candidate_blocks_scanned"] == 5
    diag.start_episode(scheduler, reqs[1], 20, {})
    assert diag.totals["candidate_blocks_scanned"] == 5
    for req in (reqs[4], reqs[2]):
        req.num_preemptions = 1
        diag.preempt(req, 0, 1, "memory_pressure")
    diag.end_episode(True)
    # A second episode in the same step, with an infeasible suffix.
    diag.allocation_failed("4", 5, 0)
    diag.start_episode(scheduler, reqs[4], 10, {})
    reqs[4].num_preemptions += 1
    diag.preempt(reqs[4], 0, 1, "memory_pressure")
    diag.end_episode(False)
    diag.end_step()
    diag.shutdown()
    episodes = [r for r in records(tmp_path) if r["kind"] == "episode"]
    assert episodes[0]["tail_matches_actual"] and episodes[0]["comparable"]
    assert episodes[1]["self_preemption"] and not episodes[1]["shadow"]["feasible"]
    assert episodes[0]["step_id"] == episodes[1]["step_id"]


@pytest.mark.parametrize("unsupported", [["async_scheduling"], ["offload_or_transfer"], ["hybrid_or_non_full_attention"], ["speculative_decoding"]])
def test_unsupported_no_detailed_work(tmp_path, unsupported):
    diag = PreemptionDiagnostics(tmp_path, {}, unsupported)
    req = request()
    diag.add_request(req)
    run_interval(diag, req, 0, 10)
    diag.preempt(req, 0, 1, "memory_pressure")
    assert diag.pending is None and diag.requests["r"]["recovery"] is None
    diag.shutdown()
    assert not any(r["kind"] == "recovery" for r in records(tmp_path))
    summary, *_ = load_report().analyze(records(tmp_path))
    assert summary["unfinished"]["preemptions_without_recovery_record"] == 0
    assert summary["unsupported_coverage"]["preemptions_without_recovery_records"] == 1


@pytest.mark.parametrize("preempt", [False, True])
def test_outstanding_batches_invalidate(tmp_path, preempt):
    diag = PreemptionDiagnostics(tmp_path, {})
    req = request()
    diag.add_request(req)
    batch = NS(num_scheduled_tokens={"r": 10})
    diag.begin_step()
    diag.scheduled(batch, {"r": req})
    if preempt:
        diag.preempt(req, 0, 1, "forced_prefix_reset")
    else:
        diag.begin_step()
    diag.completed(batch, req, True)
    diag.shutdown()
    summary, *_ = load_report().analyze(records(tmp_path))
    assert summary["invalidated_instances"] == [diag.instance_id]
    assert summary["completed_scheduled_tokens"] == 0


def test_io_failure_warns_once_and_discards_state(tmp_path, caplog):
    diag = PreemptionDiagnostics(tmp_path, {})
    diag.add_request(request())
    diag.stream.close()
    diag.begin_step()
    diag.end_step()
    diag.begin_step()
    diag.shutdown()
    assert not diag.enabled and not diag.requests
    assert sum("diagnostics disabled" in r.message for r in caplog.records) == 1


def test_bad_directory(tmp_path, caplog):
    path = tmp_path / "file"
    path.write_text("x")
    diag = PreemptionDiagnostics(path, {})
    assert not diag.enabled


def test_report_truncation_and_malformed_records(tmp_path):
    diag = PreemptionDiagnostics(tmp_path, {})
    diag.begin_step()
    diag.end_step()
    diag.shutdown()
    path = next(tmp_path.glob("*.jsonl"))
    with path.open("a") as stream:
        stream.write('{"version":')
    report = load_report()
    summary = report.write_report([path], tmp_path / "report")
    assert len(summary["parsing_warnings"]) == 1
    assert (tmp_path / "report/report.md").exists()
    with path.open("a") as stream:
        stream.write("\n{}\n")
    with pytest.raises(ValueError, match="malformed JSON"):
        report.read_records([path])


@pytest.mark.parametrize("kind", ["streaming_input", "non_generation", "multimodal"])
def test_unsupported_request_types(tmp_path, kind):
    diag = PreemptionDiagnostics(tmp_path, {})
    req = request()
    if kind == "streaming_input":
        req.resumable = True
    elif kind == "non_generation":
        req.sampling_params = None
    else:
        req.has_encoder_inputs = True
    diag.add_request(req)
    run_interval(diag, req, 0, 10)
    diag.preempt(req, 0, 1, "memory_pressure")
    diag.shutdown()
    rows = records(tmp_path)
    assert next(r for r in rows if r["kind"] == "request_start")["unsupported"] == [kind]
    assert not any(r["kind"] == "recovery" for r in rows)
    assert diag.totals["completed_scheduled_tokens"] == 0


def test_report_excludes_invalidation_and_tail_mismatch(tmp_path):
    report = load_report()
    envelope = dict(version=1, instance_id="test", step_id=1, timestamp=1)
    episode = dict(envelope, kind="episode", episode_id=1, unsupported=[],
                   actual=[dict(request_id="r", released=2)], comparable=False,
                   self_preemption=False, actual_released=2,
                   shadow=dict(feasible=True, victim_count=1, released=2))
    rows = [dict(envelope, kind="startup", unsupported=[]), episode,
            dict(envelope, kind="progress", totals=dict(completed_scheduled_tokens=10, prompt_replay=5))]
    summary, *_ = report.analyze(rows)
    assert summary["comparable_episodes"] == 0
    assert summary["completed_replay_tokens"] == 5
    rows.append(dict(envelope, kind="invalidated", reason="overlapping_batches"))
    summary, *_ = report.analyze(rows)
    assert summary["completed_replay_tokens"] == 0
    assert summary["unfinished"]["instances_without_shutdown"] == 1


def test_measurement_checkpoint_excludes_warmup_work(tmp_path):
    diag = PreemptionDiagnostics(tmp_path / "raw", {})

    def replay(name, hits):
        req = request(name)
        diag.add_request(req)
        run_interval(diag, req, 0, 14, output=True)
        scheduler = NS(
            policy=NS(value="fcfs"), running=[req], waiting=[],
            kv_cache_manager=NS(get_blocks=lambda rid: NS(blocks=(
                [NS(block_id=1, ref_cnt=1, is_null=False)],))),
        )
        diag.begin_step()
        diag.allocation_failed(name, 1, 0)
        diag.start_episode(scheduler, req, 1, {})
        req.num_preemptions = 1
        diag.preempt(req, 0, 1, "memory_pressure")
        diag.end_episode(False)
        diag.end_step()
        diag.resumed(req, hits)
        run_interval(diag, req, hits, 14, output=True)
        diag.finish(req)

    diag.cache_reset(True)
    replay("warmup", 0)
    diag.cache_reset(True)
    baseline = dict(diag.totals)
    marker = dict(
        version=1, scope="measured_workload", warmup_requests=1,
        expected_requests=1, completed=True,
        diagnostic_boundaries=[dict(instance_id=diag.instance_id, boundary_id=2,
                                    baseline_requests=1)],
    )
    replay("measured", 4)
    diag.shutdown()
    report = load_report()
    rows = records(tmp_path / "raw")
    selected, metadata = report.select_measurement_records(rows, marker)
    summary, episodes, requests, recoveries = report.analyze(selected)
    assert summary["totals"]["requests"] == 1
    assert summary["totals"]["steps"] == 3
    assert summary["totals"]["pressure_preemptions"] == 1
    assert summary["completed_scheduled_tokens"] == 24
    assert summary["completed_replay_tokens"] == 10
    assert summary["replay_fraction"] == 10 / 24
    assert summary["accounting_mismatches"] == 0
    assert len(episodes) == 1 and episodes[0]["episode_id"] == 2
    assert [r["request_id"] for r in requests] == ["measured"]
    assert [r["request_id"] for r in recoveries] == ["measured"]
    assert metadata["excluded_warmup_totals"] == baseline
    checkpoint = next(r for r in rows if r["kind"] == "cache_reset" and r["boundary_id"] == 2)
    shutdown = rows[-1]
    assert summary["overhead_ns"] == {
        k: v - checkpoint["overhead"].get(k, 0) for k, v in shutdown["overhead"].items()
    }
    marker_path = tmp_path / "measurement.json"
    marker_path.write_text(json.dumps(marker))
    summary = report.write_report(list((tmp_path / "raw").glob("*.jsonl")),
                                  tmp_path / "report", marker_path)
    assert summary["measurement"]["scope"] == "measured_workload"
    assert "excluded 1 warmup requests" in (tmp_path / "report/report.md").read_text()


@pytest.mark.parametrize("problem", ["busy", "failed", "missing", "later_reset", "new_instance"])
def test_measurement_checkpoint_rejects_invalid_boundaries(tmp_path, problem):
    diag = PreemptionDiagnostics(tmp_path, {})
    if problem == "busy":
        diag.add_request(request())
    diag.cache_reset(problem != "failed")
    marker = dict(version=1, scope="measured_workload", warmup_requests=0,
                  diagnostic_boundaries=[dict(instance_id=diag.instance_id,
                                              boundary_id=99 if problem == "missing" else 1,
                                              baseline_requests=0)])
    if problem == "later_reset":
        diag.cache_reset(True)
    diag.shutdown()
    rows = records(tmp_path)
    if problem == "new_instance":
        rows.append(dict(rows[0], instance_id="new"))
    with pytest.raises(ValueError):
        load_report().select_measurement_records(rows, marker)


def test_measurement_checkpoint_retains_warmup_invalidation(tmp_path):
    diag = PreemptionDiagnostics(tmp_path, {})
    diag.invalidate("overlapping_batches")
    diag.cache_reset(True)
    diag.shutdown()
    marker = dict(version=1, scope="measured_workload", warmup_requests=0,
                  diagnostic_boundaries=[dict(instance_id=diag.instance_id,
                                              boundary_id=1, baseline_requests=0)])
    report = load_report()
    selected, _ = report.select_measurement_records(records(tmp_path), marker)
    summary, *_ = report.analyze(selected)
    assert summary["unsupported_coverage"]["instances_by_reason"]["overlapping_batches"] == 1
