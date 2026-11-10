# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Demand-aware retention (kv_cache_eviction_policy="demand", SSPF series).

Mirrors the KVReuse simulator's ``eviction="pred"``, ``pe_tier="hold_last"``
under rule R; decision parity is tested in ``simulator/test_vllm_parity.py``.
"""

import math
import random

import pytest

from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.demand_retention import DemandRetention
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    get_request_block_hasher,
    init_none_hash,
    make_block_hash_with_group_id,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus

from .utils import EOS_TOKEN_ID, create_scheduler

BS = 16


class Clock:
    def __init__(self, now: float = 0.0):
        self.now = now

    def __call__(self) -> float:
        return self.now


def _setup(num_blocks: int, demand: dict, depths: dict | None = None, clock=None):
    pool = BlockPool(num_gpu_blocks=num_blocks, enable_caching=True, hash_block_size=BS)
    retention = DemandRetention(
        pool, demand, depths if depths is not None else {}, clock=clock or Clock()
    )
    pool._retention = retention
    return pool, retention


def _cached(pool: BlockPool, hashes: list[bytes]):
    """Allocate one block per hash and register it in the prefix cache."""
    blocks = pool.get_new_blocks(len(hashes))
    for block, h in zip(blocks, hashes):
        key = make_block_hash_with_group_id(BlockHash(h), 0)
        block.block_hash = key
        pool.cached_block_hash_to_block.insert(key, block)
    return blocks


def _ids(blocks) -> list[int]:
    return [b.block_id for b in blocks]


# ---------------------------------------------------------------- order


def test_lru_blocks_go_before_predicted_reuse():
    pool, retention = _setup(6, {b"y": 1}, {b"y": 1})
    x, y = _cached(pool, [b"x", b"y"])  # ids 1, 2
    pool.free_blocks([x, y])
    assert list(retention.side) == [2]
    assert pool.get_num_free_blocks() == 5
    assert _ids(pool.get_new_blocks(5)) == [3, 4, 5, 1, 2]


def test_holds_go_before_demand_and_demand_wins_overlap():
    demand = {b"b": 1, b"c": 1}
    pool, retention = _setup(5, demand, {b"b": 1, b"c": 2})
    blocks = _cached(pool, [b"a", b"b", b"c"])  # ids 1, 2, 3
    retention.hold_turn("s", [BlockHash(b"a"), BlockHash(b"c")])
    pool.free_blocks(blocks)
    assert sorted(retention.side) == [1, 2, 3]
    # a: held only (hold tier). c: held and demanded -> demand tier; deeper
    # than b (depth 1 vs 0) with the same (infinite) rank, so evicted first.
    victims = pool.get_new_blocks(4)
    assert _ids(victims) == [4, 1, 3, 2]
    assert (retention.num_hold_evictions, retention.num_demand_evictions) == (1, 2)


def test_demand_order_worst_rank_then_deepest_then_block_id():
    demand = {b"p": 1, b"q": 1, b"r": 2}
    pool, retention = _setup(5, demand, {b"p": 1, b"q": 1, b"r": 1})
    blocks = _cached(pool, [b"p", b"q", b"r", b"r"])  # ids 1..4, r copied
    retention.set_snapshot([((b"p",), 1), ((b"q",), 1), ((b"r",), 1)])
    pool.free_blocks(blocks)
    assert _ids(retention.pop_victims(4)) == [3, 4, 2, 1]


def test_rank_rules():
    demand = {}
    pool, retention = _setup(4, demand)
    retention.set_snapshot([
        ((b"x",), 1),
        ((b"p0", b"p1", b"gen"), 2),  # a preempted request: 2 prompt blocks
        ((b"p0", b"own"), 2),
    ])
    assert retention._rank(b"x") == 0
    assert retention._rank(b"p0") == retention._rank(b"p1") == 1
    assert retention._rank(b"gen") == 0  # generated blocks rank at the head
    assert retention._rank(b"own") == 2
    assert retention._rank(b"new") == math.inf
    retention.on_preempt([b"own"])
    assert retention._rank(b"own") == 0
    # Demand for p1 ended (not held): infinite until the next snapshot.
    retention.on_demand_change({b"p1": 0}, (b"p0", b"p1"))
    assert retention._rank(b"p1") == math.inf
    retention.set_snapshot([((b"p0", b"p1"), 2)])
    assert retention._rank(b"p1") == 0


@pytest.mark.parametrize("seed", range(20))
def test_rank_map_matches_first_snapshot_path(seed):
    """Rule R on random chained paths (a shared prefix tree): a hash ranks at
    the first snapshot path holding it, or 0 when that path holds it past its
    prompt blocks (a preempted request's generated blocks)."""
    rng = random.Random(seed)
    paths: list[tuple[bytes, ...]] = []
    for _ in range(rng.randint(1, 30)):
        if paths and rng.random() < 0.7:  # share a prefix of an earlier path
            base = rng.choice(paths)
            path = list(base[: rng.randint(0, len(base))])
        else:
            path = []
        parent = path[-1] if path else b"root"
        for _ in range(rng.randint(0 if path else 1, 6)):
            parent = sha256((parent, rng.randrange(4)))
            path.append(parent)
        paths.append(tuple(path))
    snapshot = [(p, rng.randint(0, len(p) + 1)) for p in paths]
    _, retention = _setup(4, {})
    retention.set_snapshot(snapshot)

    def naive(h):
        for pos, (path, num_prompt_blocks) in enumerate(snapshot):
            if h in path:
                return 0 if path.index(h) >= num_prompt_blocks else pos
        return math.inf

    for h in {h for p in paths for h in p} | {b"absent"}:
        assert retention._rank(h) == naive(h)


def test_held_hash_keeps_its_rank_when_demand_ends():
    demand = {b"h": 1}
    pool, retention = _setup(4, demand, {b"h": 1})
    retention.set_snapshot([((b"z",), 1), ((b"h",), 1)])
    retention.hold_turn("s", [BlockHash(b"h")])
    demand[b"h"] = 0
    retention.on_demand_change({b"h": 0}, (b"h",))
    assert retention._rank(b"h") == 1


def test_unwanted_blocks_return_to_lru_shallowest_first():
    path = (b"h1", b"h2", b"h3")
    demand = dict.fromkeys(path, 1)
    pool, retention = _setup(4, demand, {b"h1": 1, b"h2": 2, b"h3": 3})
    blocks = _cached(pool, [b"h3", b"h2", b"h1"])  # ids 1, 2, 3
    pool.free_blocks(blocks)
    assert len(retention.side) == 3
    for h in path:
        demand[h] = 0
    retention.on_demand_change(dict.fromkeys(path, 0), path)
    assert retention.side == {}
    assert _ids(pool.get_new_blocks(3)) == [3, 2, 1]  # h1, h2, h3


def test_new_demand_moves_free_copies_into_the_side_pool():
    demand = {}
    pool, retention = _setup(4, demand, {b"k": 1})
    (block,) = _cached(pool, [b"k"])
    pool.free_blocks([block])
    assert retention.side == {}
    demand[b"k"] = 1
    retention.on_demand_change({b"k": 1}, (b"k",))
    assert list(retention.side) == [block.block_id]
    assert pool.get_num_free_blocks() == 3


def test_without_predictions_allocation_matches_stock_lru():
    def run(with_retention: bool) -> list[int]:
        pool = BlockPool(num_gpu_blocks=8, enable_caching=True, hash_block_size=BS)
        if with_retention:
            pool._retention = DemandRetention(pool, {}, {})
        a = _cached(pool, [b"a", b"b", b"c"])
        b = pool.get_new_blocks(2)
        pool.free_blocks(a[1:])
        pool.free_blocks(b)
        c = _cached(pool, [b"d"])
        pool.free_blocks(a[:1] + c)
        return _ids(pool.get_new_blocks(7))

    assert run(True) == run(False)


# ---------------------------------------------------------------- holds


def test_hold_expires_after_ttl_times_gap():
    clock = Clock(10.0)
    pool, retention = _setup(4, {}, clock=clock)
    (block,) = _cached(pool, [b"a"])
    retention.hold_turn("s", [BlockHash(b"a")])  # gap 5 s initially: until 20
    pool.free_blocks([block])
    assert list(retention.side) == [block.block_id]
    clock.now = 19.9
    retention.expire_holds()
    assert list(retention.side) == [block.block_id]
    clock.now = 20.0
    retention.expire_holds()
    assert retention.side == {}
    assert pool.get_num_free_blocks() == 3


def test_gap_estimates_and_next_turn_drops_the_hold():
    clock = Clock(10.0)
    pool, retention = _setup(4, {}, clock=clock)
    retention.hold_turn("s", [BlockHash(b"a")])
    retention.turn_arrived("s", 13.0)  # observed gap 3 s
    retention.drop_hold("s")
    assert b"a" not in retention._hold
    clock.now = 15.0
    retention.hold_turn("s", [BlockHash(b"a")])
    assert retention._session_hold["s"][0] == 15.0 + 2.0 * 3.0
    retention.turn_arrived("s", 16.0)  # gap 1: session 0.5*3+0.5*1, global 0.9*3+0.1*1
    assert retention._gap_session["s"] == 2.0
    assert retention._gap_all == 0.9 * 3.0 + 0.1 * 1.0
    clock.now = 20.0
    retention.hold_turn("t", [BlockHash(b"b")])  # new session: global gap
    assert retention._session_hold["t"][0] == 20.0 + 2.0 * (0.9 * 3.0 + 0.1 * 1.0)


def test_shared_hold_keeps_latest_expiry_after_one_drop():
    clock = Clock(0.0)
    pool, retention = _setup(4, {}, clock=clock)
    retention.hold_turn("s", [BlockHash(b"a")])  # until 10
    clock.now = 5.0
    retention.hold_turn("t", [BlockHash(b"a")])  # until 15
    retention.drop_hold("t")
    assert retention._hold[b"a"][0] == 1
    assert retention._hold_t[b"a"] == 15.0  # running maximum, as the simulator


# ---------------------------------------------------------------- pool hooks


def test_touch_evict_and_reset_keep_free_counts():
    pool, retention = _setup(5, {b"a": 1, b"b": 1, b"c": 1})
    a, b, c = _cached(pool, [b"a", b"b", b"c"])
    pool.free_blocks([a, b, c])
    assert pool.get_num_free_blocks() == 4
    pool.touch([a])  # a cache hit takes it out of the side pool
    assert a.block_id not in retention.side and a.ref_cnt == 1
    assert pool.get_num_free_blocks() == 3
    pool.evict_blocks({b.block_id})  # no longer cached: reused first
    assert b.block_id not in retention.side and b.block_hash is None
    assert pool.get_num_free_blocks() == 3
    assert _ids(pool.get_new_blocks(1)) == [b.block_id]
    pool.free_blocks([a])
    pool.free_blocks(pool.blocks[b.block_id : b.block_id + 1])
    assert pool.reset_prefix_cache()
    assert retention.side == {} and pool.get_num_free_blocks() == 4


# ---------------------------------------------------------------- scheduler


@pytest.fixture(scope="module")
def _fixed_none_hash():
    init_none_hash(sha256)


def _request(request_id, tokens, session_id=None, max_tokens=1) -> Request:
    extra = None if session_id is None else {"session_id": session_id}
    params = SamplingParams(max_tokens=max_tokens, ignore_eos=True, extra_args=extra)
    params.update_from_generation_config({}, EOS_TOKEN_ID)
    return Request(
        request_id=request_id,
        prompt_token_ids=tokens,
        sampling_params=params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(BS, sha256),
    )


def _step(scheduler) -> None:
    output = scheduler.schedule()
    req_ids = list(output.num_scheduled_tokens)
    sampled = []
    for rid in req_ids:
        request = scheduler.requests[rid]
        done = request.num_computed_tokens >= request.num_tokens
        sampled.append([7] if done else [])
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


def _demand_scheduler():
    return create_scheduler(
        enable_prefix_caching=True,
        scheduling_policy="sspf",
        lpm_window_size=0,
        kv_cache_eviction_policy="demand",
    )


def test_only_demand_policy_creates_retention():
    assert create_scheduler(enable_prefix_caching=True)._retention is None
    scheduler = _demand_scheduler()
    assert scheduler._retention is not None
    assert scheduler.kv_cache_manager.block_pool._retention is scheduler._retention


def test_finished_turn_holds_until_its_session_returns(_fixed_none_hash):
    scheduler = _demand_scheduler()
    retention = scheduler._retention
    scheduler.add_request(_request("t1", list(range(1, 41)), "s"))
    _step(scheduler)  # prefill, one output: finished
    assert "t1" not in scheduler.requests
    assert set(retention._session_hold) == {"s"}
    # Its two full blocks are held free blocks in the side pool.
    assert len(retention.side) == 2
    scheduler.add_request(_request("t2", list(range(1, 41)) + [9] * 16, "s"))
    assert "s" not in retention._session_hold  # replaced by t2's own demand
    assert len(retention.side) == 2  # still wanted: t2 needs them


def test_aborted_turn_never_holds(_fixed_none_hash):
    scheduler = _demand_scheduler()
    scheduler.add_request(_request("t1", list(range(1, 41)), "s", max_tokens=8))
    _step(scheduler)
    scheduler.finish_requests("t1", RequestStatus.FINISHED_ABORTED)
    assert scheduler._retention._session_hold == {}
    assert scheduler._retention.side == {}


def test_preemption_ranks_the_victims_path_first(_fixed_none_hash):
    scheduler = _demand_scheduler()
    scheduler.add_request(_request("a", list(range(1, 41)), max_tokens=8))
    _step(scheduler)
    a = scheduler.requests["a"]
    scheduler.running.remove(a)
    scheduler._preempt_request(a, 0.0)
    retention = scheduler._retention
    assert [retention._rank(h) for h in a.block_hashes] == [0] * len(a.block_hashes)
    # Its cached blocks are queued demand now: retained.
    assert len(retention.side) == 2


def test_reset_with_running_requests_clears_retention(_fixed_none_hash):
    scheduler = _demand_scheduler()
    scheduler.add_request(_request("t1", list(range(1, 41)), "s"))
    _step(scheduler)  # finished: held
    scheduler.add_request(_request("u1", list(range(100, 140)), "u", max_tokens=8))
    _step(scheduler)  # running
    assert scheduler._retention._session_hold
    assert scheduler.reset_prefix_cache(reset_running_requests=True)
    retention = scheduler._retention
    assert retention.side == {} and retention._session_hold == {}


def test_prefix_cache_reset_clears_retention(_fixed_none_hash):
    scheduler = _demand_scheduler()
    scheduler.add_request(_request("t1", list(range(1, 41)), "s"))
    _step(scheduler)
    assert scheduler._retention.side
    assert scheduler.reset_prefix_cache()
    retention = scheduler._retention
    assert retention.side == {} and retention._session_hold == {}
    assert retention._session_end == {} and retention._gap_all is None
