# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Related-work orderings: Fairway, RAGCache, k-LPM and the SSPF overtaking guard.

The pure-module cases mirror the KVReuse simulator's unit tests
(``simulator/sched-sim/test_{fairway,ragcache,klpm,overtake_guard}.py``);
decision parity with the simulator is tested there (``test_vllm_parity.py``).
"""

import json
import random

import pytest

from vllm.config import SchedulerConfig
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.fairway_prefill_model import (
    ForwardSimChunkTime,
    load_chunk_time,
)
from vllm.v1.core.sched.fairway_ranker import (
    FairwayParams,
    fast_score,
    merge_by_work_charge,
    patience,
    prefill_time_estimator,
    rank_fairway,
)
from vllm.v1.core.sched.klpm_ranker import KLPMCycle, rank_klpm
from vllm.v1.core.sched.overtake_guard import OvertakeGuard, order_with_guard
from vllm.v1.core.sched.ragcache_ranker import (
    BypassWindow,
    rank_ragcache,
    reuse_ratio,
)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus

from .utils import EOS_TOKEN_ID, create_scheduler

BS = 16


@pytest.fixture(autouse=True, scope="module")
def _fixed_none_hash():
    init_none_hash(sha256)


# ---------------------------------------------------------------- Fairway


class TestFairwayMerge:
    def test_fast_lane_places_while_its_charge_is_not_larger(self):
        # A = 4*10 = 40 <= F = 100: x (A=80), y (A=120); then Fair places z.
        work = {"x": 10, "y": 10, "z": 100}
        order = merge_by_work_charge(["x", "y", "z"], ["z", "x", "y"], work, 4)
        assert order == ["x", "y", "z"]

    def test_fair_lane_places_first_when_fast_charge_is_larger(self):
        # A = 4*50 = 200 > F = 100: Fair places z (F=200), then Fast x, y.
        work = {"x": 50, "y": 10, "z": 100}
        order = merge_by_work_charge(["x", "y", "z"], ["z", "x", "y"], work, 4)
        assert order == ["z", "x", "y"]

    def test_every_request_placed_once(self):
        rng = random.Random(1)
        for _ in range(200):
            n = rng.randint(1, 30)
            items = list(range(n))
            work = {i: rng.randint(1, 5000) for i in items}
            fast = rng.sample(items, n)
            order = merge_by_work_charge(fast, items, work, rng.choice([1, 2, 4, 8]))
            assert sorted(order) == items

    def test_lanes_must_match(self):
        with pytest.raises(ValueError):
            merge_by_work_charge(["a"], ["a", "b"], {"a": 1, "b": 1}, 4)


class TestFairwayScore:
    def test_patience_floor_and_multiplier(self):
        params = FairwayParams()
        assert patience(100, lambda n: 0.1, params) == 1.0
        assert patience(100, lambda n: 2.0, params) == pytest.approx(6.0)

    def test_prefill_estimate_sums_chunked_steps(self):
        calls = []

        def chunk_time(start, end):
            calls.append((start, end))
            return 1.0 + end - start

        est = prefill_time_estimator(chunk_time, budget=4)
        assert est(10) == 3 * 1.0 + 10  # chunks [0,4) [4,8) [8,10)
        assert calls == [(0, 4), (4, 8), (8, 10)]
        assert est(0) == 0.0

    @pytest.mark.parametrize(
        "wait,want", [("relative", 2.5), ("absolute", 5.5), ("none", 1.0)]
    )
    def test_wait_modes(self, wait, want):
        est = lambda n: 1.0  # noqa: E731  (patience 3 s)
        score = fast_score(50, 100, 4.5, est, FairwayParams(wait=wait))
        assert score == pytest.approx(want)

    @pytest.mark.parametrize(
        "kwargs", [{"weight": 0}, {"wait": "bogus"}, {"patience_floor_s": -1}]
    )
    def test_invalid_params(self, kwargs):
        with pytest.raises(ValueError):
            FairwayParams(**kwargs)

    def test_without_fair_lane_is_score_order(self):
        rng = random.Random(2)
        est = prefill_time_estimator(lambda s, e: 0.01 + 1e-5 * (e - s), 8192)
        params = FairwayParams(fair_lane=False)
        items = list(range(25))
        num = {i: rng.randint(16, 20000) for i in items}
        hit = {i: rng.randint(0, num[i] - 1) for i in items}
        arr = {i: float(i) for i in items}
        order = rank_fairway(items, hit, num, arr, 30.0, est, params)
        scores = [fast_score(hit[i], num[i], 30.0 - arr[i], est, params) for i in order]
        assert scores == sorted(scores, reverse=True)


FORWARD_MODEL = {
    "units": "seconds",
    "tiles": [16, 32],
    "coef": {
        "t0": 0.01,
        "s16": 0.002,
        "s32": 0.003,
        "tN": 1e-4,
        "tC": 1e-7,
        "tP": 5e-5,
        "tA": 3e-9,
    },
}


class TestForwardSimChunkTime:
    def test_prefill_chunk_is_forward_sim_formula(self):
        t = ForwardSimChunkTime(FORWARD_MODEL)
        c = FORWARD_MODEL["coef"]
        n, x = 100, 300
        assert t(x, x + n) == pytest.approx(
            c["t0"] + c["tP"] * n + c["tA"] * (n * x + n * n / 2)
        )

    def test_one_token_step_is_a_decode(self):
        t = ForwardSimChunkTime(FORWARD_MODEL)
        c = FORWARD_MODEL["coef"]
        # one decoding request: no tile step (N_d = 1 is not above 16)
        assert t(500, 501) == pytest.approx(c["t0"] + c["tN"] + c["tC"] * 500)

    def test_load_from_file(self, tmp_path):
        path = tmp_path / "model.json"
        path.write_text(json.dumps(FORWARD_MODEL))
        assert load_chunk_time(str(path))(0, 64) == ForwardSimChunkTime(FORWARD_MODEL)(
            0, 64
        )

    @pytest.mark.parametrize(
        "bad",
        [
            {**FORWARD_MODEL, "units": "ms"},
            {**FORWARD_MODEL, "coef": {"t0": 1.0}},
            {**FORWARD_MODEL, "tiles": [64]},  # s64 missing
        ],
    )
    def test_invalid_model(self, bad):
        with pytest.raises(ValueError):
            ForwardSimChunkTime(bad)


# ---------------------------------------------------------------- RAGCache


class TestRAGCache:
    def test_ratio(self):
        assert reuse_ratio(48, 64) == 3.0
        assert reuse_ratio(0, 64) == 0.0
        assert reuse_ratio(64, 64) == 64.0  # at least one token computed

    def test_ratio_order_ties_oldest_first(self):
        hit = {"a": 0, "b": 32, "c": 32}
        num = {"a": 64, "b": 64, "c": 64}
        arr = {"a": 0, "b": 2, "c": 1}
        order = rank_ragcache(["b", "a", "c"], hit, num, arr, window=0)
        assert order == ["c", "b", "a"]

    def test_bypass_counts_only_later_arrivals(self):
        w = BypassWindow(3)
        arr = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
        waiting = dict(arr)
        for x in ("B", "C"):
            del waiting[x]
            w.on_admit(x, arr[x], waiting)
        assert w.overtaken == {"A": 2}
        w.on_admit("A", arr["A"], {"D": 3, "E": 4})  # the oldest overtakes nobody
        assert w.overtaken == {}

    def test_due_request_ranks_first(self):
        w = BypassWindow(3)
        arr = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
        waiting = dict(arr)
        for x in ("B", "C", "D"):
            del waiting[x]
            w.on_admit(x, arr[x], waiting)
        hit = {"A": 0, "E": 48}
        num = {"A": 64, "E": 64}
        assert w.is_due("A")
        assert rank_ragcache(["E", "A"], hit, num, arr, 3, "bypass", w) == ["A", "E"]
        fresh = BypassWindow(3)
        assert rank_ragcache(["E", "A"], hit, num, arr, 3, "bypass", fresh) == [
            "E",
            "A",
        ]

    def test_slide_reorders_only_oldest(self):
        items = ["a", "b", "c", "d"]
        hit = {"a": 0, "b": 32, "c": 0, "d": 60}
        num = {x: 64 for x in items}
        arr = {x: i for i, x in enumerate(items)}
        order = rank_ragcache(items, hit, num, arr, 2, "slide")
        assert order == ["b", "a", "c", "d"]

    def test_bypass_mode_needs_state(self):
        with pytest.raises(ValueError):
            rank_ragcache(["a"], {"a": 0}, {"a": 16}, {"a": 0}, 2, "bypass")


# ---------------------------------------------------------------- k-LPM


class TestKLPM:
    # arrival a < b < c < d; cached lengths: d > b > c = a
    ITEMS = ["a", "b", "c", "d"]
    HIT = {"a": 0, "b": 32, "c": 0, "d": 48}
    ARR = {"a": 0, "b": 1, "c": 2, "d": 3}

    def order(self, k, pos=0, items=None):
        return rank_klpm(items or self.ITEMS, self.HIT, self.ARR, k, pos)

    def test_k1_is_arrival_order(self):
        assert self.order(1, items=["d", "c", "b", "a"]) == ["a", "b", "c", "d"]

    def test_k2_alternates_oldest_and_longest_hit(self):
        assert self.order(2) == ["a", "d", "b", "c"]

    def test_pos_shifts_the_oldest_slot(self):
        assert self.order(2, pos=1) == ["d", "a", "b", "c"]
        assert self.order(3, pos=1) == ["d", "b", "a", "c"]

    def test_large_k_is_stable_lpm(self):
        order = self.order(10**9, pos=1, items=["c", "a", "b", "d"])
        assert order == ["d", "b", "c", "a"]

    def test_plan_tail_survives_admitting_its_head(self):
        # The scheduler keeps one plan per step: admitting its head and
        # advancing the cycle must leave the same tail.
        rng = random.Random(4)
        for _ in range(300):
            items = list(range(rng.randint(2, 20)))
            hit = {i: rng.choice([0, 16, 32, 48]) for i in items}
            arr = {i: (rng.randint(0, 5), i) for i in items}
            k = rng.randint(1, 6)
            pos = rng.randrange(k)
            plan = rank_klpm(items, hit, arr, k, pos)
            rest = [i for i in items if i != plan[0]]
            assert rank_klpm(rest, hit, arr, k, (pos + 1) % k) == plan[1:]

    def test_cycle(self):
        c = KLPMCycle(3)
        seen = []
        for _ in range(7):
            seen.append(c.next_is_oldest())
            c.on_admit()
        assert seen == [True, False, False, True, False, False, True]

    @pytest.mark.parametrize("bad", [0, -1, 2.5, "4", True, float("inf")])
    def test_k_validation(self, bad):
        with pytest.raises(ValueError):
            KLPMCycle(bad)

    def test_whole_float_k(self):
        assert KLPMCycle(4.0).k == 4


# ---------------------------------------------------------------- guard


class TestOvertakeGuard:
    def test_only_later_arrivals_count(self):
        g = OvertakeGuard(2)
        g.on_admit("x", 100, 5, {"a": 1, "b": 3, "c": 7})
        assert g.over == {"a": 100, "b": 100}
        g.on_admit("a", 50, 1, {"b": 3, "c": 7})
        assert g.over == {"b": 100}

    def test_due_threshold_and_block_floor(self):
        g = OvertakeGuard(2)
        g.over = {"a": 99, "b": 100, "c": 32}
        assert not g.is_due("a", 50)
        assert g.is_due("b", 50)
        assert g.is_due("c", 0)  # uncached floor: one block (16 tokens)
        assert not g.is_due("d", 0)

    def test_due_go_first_oldest_first(self):
        g = OvertakeGuard(1)
        g.over = {"y": 20, "z": 20, "w": 10}
        order, due = order_with_guard(
            ["x", "y", "z", "w"],
            {"x": 20, "y": 20, "z": 20, "w": 20},
            {"x": 0, "y": 2, "z": 1, "w": 3},
            g,
        )
        assert order == ["z", "y", "x", "w"]
        assert sorted(due) == ["y", "z"]

    @pytest.mark.parametrize("k", [0, -1])
    def test_invalid_k(self, k):
        with pytest.raises(ValueError):
            OvertakeGuard(k)


# ---------------------------------------------------------------- scheduler


def _request(request_id: str, tokens: list[int], max_tokens: int = 1) -> Request:
    params = SamplingParams(max_tokens=max_tokens, ignore_eos=True)
    params.update_from_generation_config({}, EOS_TOKEN_ID)
    return Request(
        request_id=request_id,
        prompt_token_ids=tokens,
        sampling_params=params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(BS, sha256),
    )


def _scheduler(policy: str, **kwargs) -> Scheduler:
    return create_scheduler(
        enable_prefix_caching=True,
        scheduling_policy=policy,
        lpm_window_size=0,
        num_blocks=200,
        **kwargs,
    )


def _drive(scheduler: Scheduler, max_steps: int = 500) -> list[str]:
    """Run until every request finishes; return fresh admissions in order."""
    admits: list[str] = []
    for _ in range(max_steps):
        running, waiting = scheduler.get_request_counts()
        if not running and not waiting:
            return admits
        before = {r.request_id for r in scheduler.running}
        output = scheduler.schedule()
        admits += [rid for rid in output.num_scheduled_tokens if rid not in before]
        req_ids = list(output.num_scheduled_tokens)
        sampled = []
        for rid in req_ids:
            request = scheduler.requests[rid]
            done = request.num_computed_tokens >= request.num_tokens
            sampled.append([1000 + request.num_output_tokens] if done else [])
        scheduler.update_from_output(
            output,
            ModelRunnerOutput(
                req_ids=req_ids,
                req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
                sampled_token_ids=sampled,
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=[],
            ),
        )
    raise AssertionError("scheduler did not finish")


WARM = list(range(1, 49))  # three full blocks


def _mixed() -> list[Request]:
    """A warm prefix W, then misses and hits that all wait at once."""
    reqs = [_request("W", WARM)]
    reqs += [
        _request(f"m{i}", list(range(10_000 + 64 * i, 10_064 + 64 * i)))
        for i in range(4)
    ]
    reqs += [
        _request(f"h{i}", WARM + list(range(20_000 + 16 * i, 20_016 + 16 * i)))
        for i in range(8)
    ]
    return reqs


def _admits(policy: str, **kwargs) -> list[str]:
    scheduler = _scheduler(policy, max_num_seqs=1, **kwargs)
    for request in _mixed():
        scheduler.add_request(request)
    return _drive(scheduler)


class TestSchedulerWiring:
    def test_klpm_cycle_persists_across_steps(self):
        # One admission per step: restarting the cycle per step would be FCFS.
        assert _admits("klpm", klpm_k=2) == [
            "W", "h0", "m0", "h1", "m1", "h2", "m2", "h3", "m3",
            "h4", "h5", "h6", "h7",
        ]  # fmt: skip

    def test_klpm_limits_are_fcfs_and_lpm(self):
        fcfs, lpm = _admits("fcfs"), _admits("lpm")
        assert fcfs != lpm
        assert _admits("klpm", klpm_k=1) == fcfs
        assert _admits("klpm", klpm_k=10**9) == lpm

    def test_ragcache_hit_overtakes_older_miss(self):
        scheduler = _scheduler("ragcache", max_num_seqs=1)
        for request in (
            _request("w", WARM),
            _request("a", list(range(7000, 7048))),
            _request("b", WARM + list(range(8000, 8016))),  # received after a
        ):
            scheduler.add_request(request)
        assert _drive(scheduler) == ["w", "b", "a"]
        assert scheduler._ranking is not None
        assert scheduler._ranking.bypass is not None
        assert scheduler._ranking.bypass.overtaken == {}

    def test_ragcache_bypass_window_bounds_overtaking(self):
        # window 1: once the hit overtook the miss, the miss goes first
        assert _admits("ragcache", ragcache_window=1)[:3] == ["W", "h0", "m0"]

    def test_fairway_runs_with_forward_sim_model(self, tmp_path):
        path = tmp_path / "model.json"
        path.write_text(json.dumps(FORWARD_MODEL))
        admits = _admits("fairway", fairway_prefill_model=str(path))
        assert sorted(admits) == sorted(r.request_id for r in _mixed())

    def test_sspf_guard_off_by_default(self):
        scheduler = _scheduler("sspf")
        assert scheduler._ranking is None

    def test_aborted_waiting_request_is_forgotten(self):
        scheduler = _scheduler("klpm")
        request = _request("x", list(range(100, 140)))
        scheduler.add_request(request)
        assert scheduler._ranking is not None
        assert "x" in scheduler._ranking._seq
        scheduler.finish_requests("x", RequestStatus.FINISHED_ABORTED)
        assert "x" not in scheduler._ranking._seq


class TestConfigValidation:
    def test_fairway_needs_prefill_model(self):
        with pytest.raises(ValueError, match="fairway_prefill_model"):
            SchedulerConfig.default_factory(policy="fairway")

    def test_guard_needs_sspf(self):
        with pytest.raises(ValueError, match="sspf_overtake_guard_k"):
            SchedulerConfig.default_factory(policy="spf", sspf_overtake_guard_k=8)

    def test_klpm_rejects_admission_skips(self):
        with pytest.raises(ValueError, match="lpm_admission_skips"):
            SchedulerConfig.default_factory(policy="klpm", lpm_admission_skips=4)

    @pytest.mark.parametrize("policy", ["fairway", "ragcache", "klpm"])
    def test_related_policies_reject_pkas(self, policy, tmp_path):
        extra = {}
        if policy == "fairway":
            path = tmp_path / "model.json"
            path.write_text(json.dumps(FORWARD_MODEL))
            extra["fairway_prefill_model"] = str(path)
        with pytest.raises(ValueError, match="admission_policy"):
            SchedulerConfig.default_factory(
                policy=policy,
                admission_policy="pkas",
                scheduler_reserve_full_isl=False,
                **extra,
            )
