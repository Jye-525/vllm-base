# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The cached-token count reported to the client (usage cached_tokens) is the
request's first admission hit, also when it is preempted mid-prefill."""

import pytest

from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request

from .utils import EOS_TOKEN_ID, create_scheduler

BS = 16


@pytest.fixture(autouse=True, scope="module")
def _fixed_none_hash():
    init_none_hash(sha256)


def _step(scheduler):
    output = scheduler.schedule()
    req_ids = list(output.num_scheduled_tokens)
    sampled = []
    for rid in req_ids:
        request = scheduler.requests[rid]
        done = request.num_computed_tokens >= request.num_tokens
        sampled.append([7] if done else [])
    outputs = scheduler.update_from_output(
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
    return output, [o for eco in outputs.values() for o in eco.outputs]


def test_resume_after_midprefill_preemption_keeps_first_admission_hit():
    scheduler = create_scheduler(
        enable_prefix_caching=True, long_prefill_token_threshold=32
    )
    params = SamplingParams(max_tokens=4, ignore_eos=True)
    params.update_from_generation_config({}, EOS_TOKEN_ID)
    request = Request(
        request_id="a",
        prompt_token_ids=list(range(1, 65)),
        sampling_params=params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(BS, sha256),
    )
    scheduler.add_request(request)

    output, outs = _step(scheduler)  # first chunk: 32 of 64 tokens, no hit
    assert output.num_scheduled_tokens == {"a": 32}
    assert outs == []  # no token yet: prefill stats not reported

    # Preempted mid-prefill. Outside schedule(), so (as reset_prefix_cache does)
    # forget that it ran in the previous step: a resumed request never did.
    scheduler.running.remove(request)
    scheduler._preempt_request(request, 0.0)
    scheduler.prev_step_scheduled_req_ids.discard("a")

    output, outs = _step(scheduler)  # resume: hits its own first 32 tokens
    assert output.num_scheduled_tokens == {"a": 32}
    assert request.num_computed_tokens == 64
    (out,) = outs
    assert out.prefill_stats is not None
    assert out.prefill_stats.num_cached_tokens == 0  # first admission, not 32
    assert out.prefill_stats.num_prompt_tokens == 64
