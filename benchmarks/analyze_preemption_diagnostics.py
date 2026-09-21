#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Analyze scheduler-observed preemptions; requires only the Python stdlib."""

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def read_records(paths):
    records, warnings = [], []
    for path in paths:
        lines = Path(path).read_bytes().splitlines(keepends=True)
        for number, line in enumerate(lines, 1):
            try:
                row = json.loads(line)
            except (ValueError, UnicodeError) as exc:
                if number == len(lines) and not line.endswith(b"\n"):
                    warnings.append(f"{path}:{number}: skipped truncated final record")
                    continue
                raise ValueError(f"{path}:{number}: malformed JSON: {exc}") from exc
            if not isinstance(row, dict) or row.get("version") != 1 or not all(
                k in row for k in ("kind", "instance_id", "step_id", "timestamp")
            ):
                raise ValueError(f"{path}:{number}: invalid or unsupported record schema")
            records.append(row)
    return records, warnings


def select_measurement_records(records, marker):
    """Rebase cumulative counters at an idle reset; preserve raw IDs and logs."""
    if marker.get("version") != 1 or marker.get("scope") != "measured_workload":
        raise ValueError("Invalid measurement-boundary schema")
    boundaries = marker.get("diagnostic_boundaries", [])
    by_instance = {b["instance_id"]: b for b in boundaries}
    grouped = defaultdict(list)
    for row in records:
        grouped[row["instance_id"]].append(row)
    if not by_instance or len(by_instance) != len(boundaries) or by_instance.keys() != grouped.keys():
        raise ValueError("Measurement boundaries must match every scheduler instance exactly")
    selected = []
    excluded_totals, excluded_overhead = Counter(), Counter()
    for instance, rows in grouped.items():
        reference = by_instance[instance]
        matches = [i for i, r in enumerate(rows) if r["kind"] == "cache_reset"
                   and r["boundary_id"] == reference["boundary_id"]]
        if len(matches) != 1:
            raise ValueError(f"{instance}: missing or duplicate measurement checkpoint")
        index = matches[0]
        baseline = rows[index]
        if (not baseline["succeeded"] or baseline["active_requests"]
                or baseline["outstanding_batch"] or baseline["open_episode"]):
            raise ValueError(f"{instance}: measurement checkpoint is not a successful idle reset")
        if baseline["totals"].get("requests", 0) != reference["baseline_requests"]:
            raise ValueError(f"{instance}: checkpoint does not match benchmark marker")
        starts = [r for r in rows[:index] if r["kind"] == "startup"]
        if len(starts) != 1:
            raise ValueError(f"{instance}: expected one startup before checkpoint")
        # Runtime invalidation during warmup persists in the scheduler helper.
        selected.append(dict(starts[0], unsupported=baseline["unsupported"]))
        excluded_totals.update(baseline["totals"])
        excluded_overhead.update(baseline["overhead"])
        for row in rows[index+1:]:
            if row["kind"] in ("cache_reset", "startup"):
                raise ValueError(f"{instance}: reset/restart after measurement began")
            row = dict(row)
            for field in ("totals", "overhead"):
                if field not in row:
                    continue
                values = {k: row[field].get(k, 0) - baseline[field].get(k, 0)
                          for k in row[field].keys() | baseline[field].keys()}
                if any(v < 0 for v in values.values()):
                    raise ValueError(f"{instance}: {field} decreased after measurement checkpoint")
                row[field] = values
            selected.append(row)
    if excluded_totals["requests"] != marker["warmup_requests"]:
        raise ValueError("Checkpoint request totals do not match the warmup population")
    metadata = dict(marker, excluded_warmup_totals=dict(excluded_totals),
                    excluded_warmup_overhead_ns=dict(excluded_overhead))
    return selected, metadata


def distribution(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return dict(count=0, median=None, p95=None, maximum=None)
    def percentile(p):
        index = (len(values)-1)*p
        lo, hi = math.floor(index), math.ceil(index)
        return values[lo] + (values[hi]-values[lo])*(index-lo)
    return dict(count=len(values), median=percentile(.5), p95=percentile(.95), maximum=values[-1])


def ratio(n, d):
    return n/d if d else None


def analyze(records, warnings=()):
    instances = defaultdict(list)
    for row in records:
        instances[row["instance_id"]].append(row)
    totals, detailed_totals, overhead, unsupported = Counter(), Counter(), Counter(), Counter()
    unsupported_requests, unsupported_episodes = Counter(), Counter()
    episodes, requests, recoveries = [], [], []
    unfinished = dict(instances_without_shutdown=0, requests_without_end=0,
                      preemptions_without_recovery_record=0)
    invalidated = set()
    accounting_errors = []
    unsupported_missing_recoveries = 0
    for instance, rows in instances.items():
        if any(r["kind"] == "invalidated" for r in rows):
            invalidated.add(instance)
        starts = [r for r in rows if r["kind"] == "startup"]
        reasons = set(starts[0].get("unsupported", [])) if starts else {"missing_startup"}
        reasons.update(r["reason"] for r in rows if r["kind"] == "invalidated")
        unsupported.update(reasons)
        latest, costs = {}, {}
        active, preempted = {}, set()
        request_counts, pressure_counts = Counter(), Counter()
        excluded_requests = set()
        for row in rows:
            latest.update(row.get("totals", {}))
            costs.update(row.get("overhead", {}))
            kind = row["kind"]
            if kind == "episode":
                row = dict(row)
                row["included"] = not reasons and not row["unsupported"]
                episodes.append(row)
                unsupported_episodes.update(row["unsupported"])
            elif kind == "request_start":
                active[row["request_id"]] = row
                unsupported_requests.update(row.get("unsupported", []))
                if row.get("unsupported"):
                    excluded_requests.add(row["request_id"])
            elif kind == "request_end":
                requests.append(row)
                active.pop(row["request_id"], None)
            elif kind == "preemption":
                preempted.add((row["request_id"], row["ordinal"]))
                request_counts[row["request_id"]] += 1
                if row["reason"] == "memory_pressure":
                    pressure_counts[row["request_id"]] += 1
            elif kind == "recovery":
                row = dict(row)
                row["included"] = not reasons and row["reason"] == "memory_pressure"
                recoveries.append(row)
                preempted.discard((row["request_id"], row["ordinal"]))
        totals.update(latest)
        overhead.update(costs)
        if not reasons:
            detailed_totals.update(latest)
        has_shutdown = any(r["kind"] == "shutdown" for r in rows)
        unfinished["instances_without_shutdown"] += not has_shutdown
        if has_shutdown and starts and not reasons:
            pressure_recoveries = [r for r in rows if r["kind"] == "recovery"
                                   and r["reason"] == "memory_pressure"]
            for key in ("prompt_replay", "output_replay"):
                observed = sum(r[key] for r in pressure_recoveries)
                if observed != latest.get(key, 0):
                    accounting_errors.append(
                        f"{instance}: {key} records={observed}, total={latest.get(key, 0)}")
            episode_count = sum(r["kind"] == "episode" for r in rows)
            if episode_count != latest.get("episodes", 0):
                accounting_errors.append(f"{instance}: episode records do not match totals")
        unfinished["requests_without_end"] += len(active)
        excluded = sum(bool(reasons) or rid in excluded_requests for rid, _ in preempted)
        unsupported_missing_recoveries += excluded
        unfinished["preemptions_without_recovery_record"] += len(preempted) - excluded
        for row in active.values():
            requests.append(dict(
                row, outcome="unfinished",
                preemptions=request_counts[row["request_id"]],
                pressure_preemptions=pressure_counts[row["request_id"]],
            ))

    supported = [e for e in episodes if e["included"]]
    comparable = [e for e in supported if e["comparable"]]
    successful = [r for r in recoveries if r["included"] and r["recovered"]
                  and r["outcome"] in ("completed", "recovered_before_next_preemption")]
    incomplete = [r for r in recoveries if r["included"] and r not in successful]
    victims = [v for e in episodes for v in e["actual"]]
    replay = detailed_totals["prompt_replay"] + detailed_totals["output_replay"]
    def elapsed(row, end, start="preempted_at"):
        return row[end]-row[start] if row.get(end) is not None and row.get(start) is not None else None
    times = {label: distribution([elapsed(r, end, start) for r in successful])
             for label, end, start in (
                 ("resumption_wait_s", "resumed_at", "preempted_at"),
                 ("recovery_time_s", "recovered_at", "preempted_at"),
                 ("first_output_wait_s", "first_output_at", "preempted_at"),
                 ("output_gap_s", "first_output_at", "last_output_at"))}
    summary = dict(
        schema_version=1, instances=len(instances), totals=dict(totals),
        episodes_per_1000_steps=ratio(1000*totals["episodes"], totals["steps"]),
        fraction_requests_preempted=ratio(totals["requests_preempted"], totals["requests"]),
        actual_victims_per_episode=distribution([len(e["actual"]) for e in episodes]),
        zero_release_victims=sum(v["released"] == 0 for v in victims),
        released_blocks=distribution([v["released"] for v in victims]),
        self_preemption_episodes=sum(e["self_preemption"] for e in episodes),
        comparable_episodes=len(comparable),
        shadow_fewer_victims=sum(e["shadow"]["victim_count"] < len(e["actual"]) for e in comparable),
        infeasible_shadow_episodes=sum(not e["shadow"]["feasible"] for e in supported),
        comparisons=[dict(instance_id=e["instance_id"], episode_id=e["episode_id"],
                          actual_victims=len(e["actual"]), shadow_victims=e["shadow"]["victim_count"],
                          actual_released=e["actual_released"], shadow_released=e["shadow"]["released"])
                     for e in comparable],
        completed_replay_tokens=replay,
        completed_prompt_replay_tokens=detailed_totals["prompt_replay"],
        completed_output_replay_tokens=detailed_totals["output_replay"],
        completed_scheduled_tokens=detailed_totals["completed_scheduled_tokens"],
        replay_fraction=ratio(replay, detailed_totals["completed_scheduled_tokens"]),
        successful_recoveries=len(successful), incomplete_or_cancelled_recoveries=len(incomplete),
        recovery_outcomes=dict(Counter(r["outcome"] for r in recoveries)),
        incomplete_replay_tokens=sum(r["prompt_replay"]+r["output_replay"] for r in incomplete),
        preemptions_per_request=distribution([r.get("pressure_preemptions") for r in requests]),
        preemptions_per_finished_request=distribution([r.get("pressure_preemptions") for r in requests if r["outcome"] != "unfinished"]),
        repeated_pressure_preemptions=totals["pressure_preemptions"]-totals["requests_preempted"],
        interrupted_recoveries=sum(r["outcome"] == "interrupted_recovery" for r in recoveries if r["included"]),
        time_since_previous_recovery_s=distribution([r.get("since_previous_recovery_s") for r in recoveries if r["included"]]),
        observed_times=times, overhead_ns=dict(overhead),
        diagnostic_to_scheduler_wall_ratio=ratio(sum(overhead.values()), totals["scheduler_wall_ns"]),
        unsupported_coverage=dict(instances_by_reason=dict(unsupported),
                                  requests_by_reason=dict(unsupported_requests),
                                  episodes_by_reason=dict(unsupported_episodes),
                                  preemptions_without_recovery_records=unsupported_missing_recoveries),
        invalidated_instances=sorted(invalidated),
        unfinished=unfinished,
        accounting_mismatches=totals["accounting_mismatches"] + len(accounting_errors),
        record_accounting_errors=accounting_errors,
        parsing_warnings=list(warnings),
    )
    summary["fraction_comparable_shadow_fewer"] = ratio(summary["shadow_fewer_victims"], len(comparable))
    return summary, episodes, requests, recoveries


def write_csv(path, rows):
    fields = sorted({k for row in rows for k in row}) or ["instance_id"]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()})


def write_report(paths, output_dir, measurement_boundary=None):
    records, warnings = read_records(paths)
    measurement = None
    if measurement_boundary is not None:
        marker = json.loads(Path(measurement_boundary).read_text())
        records, measurement = select_measurement_records(records, marker)
    summary, episodes, requests, recoveries = analyze(records, warnings)
    summary["measurement"] = measurement
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for name, rows in (("episodes", episodes), ("requests", requests), ("recoveries", recoveries)):
        write_csv(destination / f"{name}.csv", rows)
    def value(key):
        item = summary[key]
        return f"{item:.6g}" if isinstance(item, float) else str(item)
    lines = ["# Preemption diagnostics", "",
             "Times are scheduler-observed, not client latency or isolated GPU recomputation time.",
             "Shadow savings are simulated victim counts; they do not predict throughput or causal latency gains.", "",
             "| Measurement | Value |", "|---|---:|"]
    if measurement is not None:
        lines[2:2] = [
            f"Scope: measured workload only; excluded {measurement['warmup_requests']} warmup requests.",
            f"Benchmark measurement completed: {measurement['completed']}. "
            "Counters and overhead are differences from the recorded idle-reset checkpoints.",
            "",
        ]
    for key in ("episodes_per_1000_steps", "fraction_requests_preempted", "zero_release_victims",
                "self_preemption_episodes", "comparable_episodes", "fraction_comparable_shadow_fewer",
                "infeasible_shadow_episodes", "completed_scheduled_tokens", "completed_replay_tokens",
                "completed_prompt_replay_tokens", "completed_output_replay_tokens", "replay_fraction",
                "successful_recoveries", "incomplete_or_cancelled_recoveries", "incomplete_replay_tokens",
                "repeated_pressure_preemptions", "interrupted_recoveries", "accounting_mismatches"):
        lines.append(f"| {key} | {value(key)} |")
    lines += ["", "## Distributions", "", "| Measurement | Count | Median | p95 | Maximum |", "|---|---:|---:|---:|---:|"]
    distributions = {k: summary[k] for k in ("actual_victims_per_episode", "released_blocks",
                     "preemptions_per_request", "preemptions_per_finished_request",
                     "time_since_previous_recovery_s")}
    distributions.update(summary["observed_times"])
    for name, dist in distributions.items():
        lines.append(f"| {name} | {dist['count']} | {dist['median']} | {dist['p95']} | {dist['maximum']} |")
    lines += ["", "## Interpretation", "",
              "- Poor reclamation: assess zero/low-release victims alongside the fraction of feasible comparisons using fewer victims.",
              "- Costly recovery: assess completed replay fraction and observed interruptions. Cancelled and incomplete recovery times are excluded from successful distributions; their completed replay is still real work.",
              "- Repeated pressure: assess interrupted recoveries, repeated preemptions, and time from prior recovery to the next preemption.",
              "- Limited relevance: few episodes and little replay limit the case for a new victim policy.",
              "", "## Coverage and accounting", "",
              "Only supported, feasible, tail-validated episodes enter comparisons. Priority preemptions are recorded without shadow selection. Forced resets are excluded from memory-pressure replay and comparisons. Runtime batch overlap excludes the instance's detailed results conservatively.",
              "", "Missing recovery records for explicitly unsupported configurations are counted separately under coverage. Unfinished counts refer to the remaining records.",
              "", "Detailed actual/shadow counts and reclaimed capacity are in episodes.csv and summary.json; individual ordinals and outcomes are in recoveries.csv.",
              "", "Overhead is cumulative instrumented work (including output processing), divided by schedule() wall time; this ratio is not an end-to-end slowdown. The final record's own serialization/flush and timing wrapper are not included in its counters.", ""]
    for key in ("overhead_ns", "diagnostic_to_scheduler_wall_ratio", "unsupported_coverage", "unfinished", "recovery_outcomes", "record_accounting_errors", "parsing_warnings"):
        lines += [f"{key}: `{json.dumps(summary[key], sort_keys=True)}`", ""]
    (destination / "report.md").write_text("\n".join(lines))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--measurement-boundary", type=Path,
        help="Benchmark marker selecting idle-reset checkpoints after warmup.",
    )
    args = parser.parse_args()
    try:
        write_report(args.inputs, args.output_dir, args.measurement_boundary)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
