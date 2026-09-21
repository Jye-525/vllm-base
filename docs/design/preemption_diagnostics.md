# Preemption diagnostics

Set `VLLM_PREEMPTION_DIAGNOSTICS_DIR=/path/to/run/diagnostics` in the serving
process environment before creating the scheduler. Each scheduler writes a
unique `preemption-PID-UUID.jsonl`. Unset the variable to disable diagnostics.
There is no serving API or metrics-schema change. Diagnostic output errors warn
once and disable observation without failing serving. Records are flushed after
schedule steps, output processing, and shutdown; flushing does not imply fsync.

The supported scope is synchronous ordinary text generation with one
full-attention cache group, without speculative decoding, offload/transfer, or
pipeline/context parallelism. Prefix caching and chunked prefill are supported.
FCFS and LPM/cLPM policies receive shadow comparisons. Priority preemptions and
recovery are measured, but priority has no tail-based shadow comparison.
Streaming-input, multimodal, and non-generation requests have no detailed
recovery accounting. Episodes containing unsupported running requests are also
excluded from shadow comparisons. Unsupported configurations still report
pressure/preemption counts and explicit reasons.

## What is measured

A memory-pressure episode starts at the allocator's first insufficient-capacity
check for a running request and ends at successful retry or self-preemption.
The deficit comes directly from required allocation blocks minus free blocks at
that check. Waiting admission failures are counted separately and do not open
episodes. Retries share an episode; multiple episodes can occur in a step.
Forced prefix resets are labelled `forced_prefix_reset` and excluded from
memory-pressure comparisons and replay totals.

At episode creation, diagnostics copy the running order, non-null physical block
ownership and reference counts. Only unscheduled requests strictly after the
blocked request are shadow candidates. Other requests' references remain in the
snapshot. The snapshot includes prompt/generated/computed positions, prior
preemptions, scheduler-residence age, owned/shared/reclaimable blocks, and queue
sizes. No diagnostic prefix lookup, allocator retry, cache touch, eviction,
demand update, or running-order mutation is performed.

Both simulations use the same snapshot. Tail simulation selects the latest
eligible request. The alternative chooses the smallest release covering the
remaining deficit, otherwise the largest release, breaking ties toward the tail.
If all releases are zero it still removes a tail sharer; subsequent marginal
releases may then become positive. Copies of references are decremented after
every simulated victim. Infeasible suffixes report remaining deficit explicitly;
the blocked request is never a simulated victim.

Actual victim releases are changes in free-block count across preemption. These
are **newly allocatable blocks**, not necessarily erased cached KV: zero-reference
prefix-cached blocks may remain reusable until overwritten. Actual tail releases
are cross-checked against the reference simulation. Only supported episodes
where the blocked request proceeds, the shadow is feasible, and tail accounting
matches enter victim-count comparisons. Admission and block eviction are unchanged.

For every supported request, a historical high-water token position is advanced
only on model-output processing. Each preemption gets a separate ordinal. A
successful admission captures its existing local prefix-hit count and resumption
time. Each scheduled interval is retained before the scheduler advances its
computed position, then counted only when its corresponding model output is
processed. Overlap below the historical position is replay, split at the prompt
boundary. This counts partial blocks, chunked replay, and repeated execution of
the same positions correctly; generated tokens replayed during prefill are not
inferred from `schedule_batch`'s decode category.

Recovery completes when completed computation (or a surviving admitted local
prefix) reaches the historical position. Each ordinal retains first subsequent
output time and the gap from the last output before preemption. Outcomes include
completion, cancellation, error/ignored termination, interrupted recovery, recovery before another
preemption, invalidation, and unfinished at shutdown. Request state is released
on departure; retained state is bounded by active requests, one episode snapshot,
and one outstanding batch, plus counters. No completed-request history is held
in the serving process.

Overlapping batches or preemption of a request with outstanding computation
invalidate detailed accounting explicitly. The report conservatively excludes
all detailed measurements from an instance with runtime invalidation, including
measurements preceding detection. Counts remain available. This avoids treating
scheduled-but-unexecuted positions as completed computation.

Times are **scheduler-observed** waits and output gaps, not client latency or
isolated GPU recomputation duration. Recovery timing distributions exclude
cancelled and unfinished recoveries. Completed replay during incomplete recovery
is still real work and remains in replay totals, with a separate subtotal.
Completed-token totals exclude unsupported request types. Forced-reset replay is
excluded from the memory-pressure numerator but remains completed work in the
denominator.

## Analyze

From the repository root, using the same Python environment as serving:

```bash
python vllm/benchmarks/analyze_preemption_diagnostics.py \
  /path/to/run/diagnostics/*.jsonl --output-dir /path/to/run/report
```

The stdlib-only tool produces `report.md`, `summary.json`, `episodes.csv`,
`requests.csv`, and `recoveries.csv`. CSV nested fields contain JSON. Instance ID
plus episode ID identifies an episode; instance ID, request ID, and ordinal
identify an interruption. The schema is version 1 and includes monotonic record
timestamps and scheduler step IDs. Monotonic timestamps should only be subtracted
within an instance. Episode records are emitted on closure, recovery records on
request departure, another preemption, invalidation, or shutdown. Step/progress
records carry cumulative totals, allowing partial-run accounting.

A malformed unterminated final record is reported and skipped. Malformed records
elsewhere, unsupported schemas, and missing record envelopes raise visible
errors. Missing shutdowns, request ends and recovery records are reported;
missing recovery records for explicitly unsupported requests/configurations are
counted separately under coverage. For completed files,
the report checks episode counts and replay-record sums against shutdown totals.
Do not interpret
missing timing samples as zero waits. Raw JSONL remains the source for audits.

Collection, candidate scanning, simulation, serialization and I/O costs are
measured separately with a high-resolution clock, alongside scheduler wall time.
The overhead-to-scheduler ratio includes output-processing instrumentation and
is not an end-to-end slowdown estimate. The last record cannot include its own
serialization, flush, or timing wrapper. Counter snapshots therefore slightly
underestimate the final diagnostic work. Step wall timing ends before that step's
record serialization/flush. Measure overall impact with diagnostics-off/on runs.

## Warmup and measurement boundaries

The scheduler emits a flushed `cache_reset` record from the existing prefix-cache
reset path. It contains success status, a per-instance boundary ID, active-request
and outstanding-batch state, cumulative totals, overhead and unsupported reasons.
It does not clear diagnostic history or change scheduling. A valid measurement
checkpoint requires a successful reset with no active requests, pending model
batch or open preemption episode.

The LooGLE study runs its usual warmup in both off/on arms, waits for completion,
and resets prefixes. Before sending measured requests it writes a local
`*.measurement.json` marker selecting that checkpoint per scheduler instance.
The marker is also written for zero warmup and records whether the measured
phase completed. Kernel execution warmup is preserved; prefix contents are
reset. The raw diagnostics include both phases.

Pass `--measurement-boundary /path/to/run.measurement.json` to the analyzer to
report only the measured phase. It preserves startup coverage information and
raw episode/request IDs, removes records before the selected checkpoints, and
subtracts baseline counters and diagnostic overhead from later snapshots.
Warmup request counts, replay, preemptions, scheduler steps and wall time are
excluded. The selected boundaries and excluded totals are recorded under
`measurement` in `summary.json`. Runtime invalidation during warmup remains
unsupported if it persists in the helper; it cannot be erased by filtering.

This uses checkpoint identities and JSONL ordering, not client/server clock
alignment. Multiple instances must each have exactly one matching checkpoint;
missing checkpoints, active work at reset, subsequent resets/restarts, mismatched
warmup counts, and decreasing counters produce errors. Records through shutdown
remain in the measured report so unfinished work is visible. The LooGLE paired
comparison verifies that the report's marker matches its client result and that
only the expected measured requests were observed. Failed client requests or an
incomplete measurement marker invalidate the comparison.

Counter snapshots exclude the checkpoint record's own serialization/flush and
timing wrapper; their small cost falls on the measured side of the subtraction.
Whole-lifetime reports remain available by omitting the marker argument.

## Experiments (GPU runs require a separate request)

1. Run an identical workload and fixed settings with diagnostics disabled and
   enabled, alternating order and repeating runs. Preserve request arrivals,
   sampling, model, batch/token limits, chunked prefill, prefix caching, admission
   policy, block size, GPU cache capacity, and block eviction policy. Use separate
   output directories for every run; do not append to old results.
2. Cover private prefixes, highly shared prefixes, mixed prompt/output lengths,
   and low/moderate/high memory pressure. Vary one factor per comparison.
   Shared blocks and an earlier missing prefix block can make prompt length a
   poor proxy for immediately reclaimable memory or surviving prefix hits.
3. Use existing serving benchmarks for throughput, TTFT, and client inter-token
   latency, and compare diagnostics-off/on overhead before interpreting costs.
   This instrumentation neither changes admission nor tests a new live victim
   policy. Hold admission and eviction fixed within every pair.
4. Review low/zero-release victims and feasible victim savings as evidence of
   poor reclamation; replay fraction and observed interruptions as evidence of
   costly recovery; interrupted/repeated preemptions as evidence of repeated
   pressure. Few episodes or little replay suggest limited relevance. There is
   no universal threshold: inspect workload-level distributions and counts.
5. Do not convert simulated victim savings into throughput gains or claim causal
   latency improvement. A later experiment changing only the real victim policy
   is required to establish those effects.

## CPU verification

Before testing a non-editable installation, copy the modified package files
(`envs.py`, `v1/core/kv_cache_manager.py`, `v1/core/sched/scheduler.py`, and
`v1/core/sched/preemption_diagnostics.py`) into the active environment's installed
`vllm` package. Confirm the imported paths and contents. Preserve other local
engine changes and use the same environment as LMCache.

```bash
python -m pytest vllm/tests/v1/core/test_preemption_diagnostics.py \
  vllm/tests/v1/core/test_preemption_diagnostics_scheduler.py
python vllm/benchmarks/benchmark_preemption_diagnostics.py --steps 1000
```

The microbenchmark is a CPU synthetic diagnostic-harness measurement, with
no-pressure and repeated-pressure cases and no wall-time assertions. Real
scheduler equivalence tests use existing utilities and a local tiny model
configuration, without model weights. Also run focused existing scheduler
preemption, priority, prefix-cache and async tests in a prepared test environment.

In environments without the shared test-suite dependencies, the focused tests can
use `--confcutdir=vllm/tests/v1/core --import-mode=importlib`. This skips shared
fixtures and is suitable for these local-model diagnostics fixtures. The optional
TVM Torch DLPack extension build can be disabled for CPU checks with
`TVM_FFI_DISABLE_TORCH_C_DLPACK=1`. Existing regressions need a cached model and
tokenizer, or `VLLM_MODEL_REDIRECT_PATH` pointing `facebook/opt-125m` to a local
OPT configuration and tokenizer. These fixtures execute no model weights.

## Compact per-batch preemption costs

The existing `schedule_batch` log includes `preempted_tokens` and
`recomputed_tokens` without enabling JSONL diagnostics. Both are per-step
counts, not cumulative counters. Sum them over the desired measurement phase.
The former `max_scheduled_tokens`, `mean_scheduled_tokens`, `max_prefill_chunk`,
and `mean_prefill_chunk` fields have been removed.

`preempted_tokens` sums the victims' computed-token positions immediately before
memory-pressure preemption resets them. It includes progress supplied by prefix
cache hits and counts repeated preemptions separately; it is not erased KV or
wasted computation. Forced prefix resets are excluded.

`recomputed_tokens` counts completed execution overlapping the historical
completed position during memory-pressure recovery. Surviving prefix hits skip
execution and are not counted. Chunked recovery and repeated replay count when
each batch completes, including replay of generated-token positions.

For supported batches, the existing line is deferred until model output arrives;
its queue sizes, token budget, and other scheduling fields retain their original
scheduling-time snapshot. Thus log timestamps now reflect output processing.
Empty batches are logged immediately. No additional lines are emitted.

Exact accounting supports synchronous, non-overlapping ordinary text generation
with one full-attention cache group, without speculative decoding, transfer,
encoder-decoder models, or pipeline/context parallelism. Unsupported batches use
`-1` rather than zero for unavailable costs. Overlapping batches or scheduling/
preempting an unsupported request disable this accounting for the scheduler
lifetime, so incomplete history cannot later appear as a valid count. A batch still pending at shutdown reports
`recomputed_tokens=-1`; exclude unavailable values from numeric totals and report
incomplete coverage. Logs cannot measure batches lost to an abrupt process exit.
