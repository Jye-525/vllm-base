# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import json

import pytest

from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion
from vllm.entrypoints.openai.engine.protocol import RequestResponseMetadata
from vllm.entrypoints.openai.engine.serving import server_timing_from_output
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.v1.metrics.stats import RequestStateStats


def make_output(tokens=3, metrics=True):
    stats = (
        RequestStateStats(
            num_generation_tokens=tokens,
            queued_ts=1.0,
            scheduled_ts=1.01,
            first_token_ts=1.04,
            last_token_ts=1.10 if tokens > 1 else 1.04,
            first_token_latency=0.04,
            inter_token_latencies=[0.03, 0.03] if tokens > 1 else [],
        )
        if metrics else None
    )
    return RequestOutput(
        request_id="request-0",
        prompt="prompt",
        prompt_token_ids=[1, 2],
        prompt_logprobs=None,
        outputs=[CompletionOutput(
            index=0,
            text="output",
            token_ids=list(range(tokens)),
            cumulative_logprob=None,
            logprobs=None,
            finish_reason="length",
        )],
        finished=True,
        metrics=stats,
    )


def make_serving():
    serving = object.__new__(OpenAIServingCompletion)
    serving.enable_prompt_tokens_details = False
    serving.enable_force_include_usage = False
    return serving


@pytest.mark.parametrize("tokens", [1, 3])
def test_non_stream_completion_server_timing(tokens):
    serving = make_serving()
    request = CompletionRequest(model="model", prompt=[1, 2], max_tokens=tokens)
    metadata = RequestResponseMetadata(request_id="request")
    response = serving.request_output_to_completion_response(
        [make_output(tokens)], request, "request", 0, "model", None, metadata
    )

    timing = response.usage.server_timing
    assert timing is not None
    assert timing.ttft_ms == pytest.approx(40)
    assert timing.tpot_ms == (pytest.approx(30) if tokens > 1 else None)
    assert metadata.final_usage_info is response.usage


def test_stream_completion_final_usage_server_timing():
    serving = make_serving()
    request = CompletionRequest(
        model="model", prompt=[1, 2], max_tokens=3, stream=True,
        stream_options={"include_usage": True},
    )
    metadata = RequestResponseMetadata(request_id="request")

    async def results():
        yield 0, make_output()

    async def collect_events():
        return [
            json.loads(chunk.removeprefix("data: "))
            async for chunk in serving.completion_stream_generator(
                request, [{"prompt_token_ids": [1, 2]}], results(), "request",
                0, "model", 1, None, metadata,
            ) if chunk != "data: [DONE]\n\n"
        ]

    events = asyncio.run(collect_events())
    assert len(events) == 2
    assert events[0]["usage"] is None
    assert events[1]["choices"] == []
    assert events[1]["usage"]["server_timing"]["ttft_ms"] == pytest.approx(40)
    assert events[1]["usage"]["server_timing"]["tpot_ms"] == pytest.approx(30)
    assert metadata.final_usage_info.server_timing is not None


def test_completion_timing_unavailable_for_missing_metrics_or_multiple_prompts():
    serving = make_serving()
    request = CompletionRequest(model="model", prompt=[1, 2], max_tokens=3)
    output = make_output(metrics=False)
    assert server_timing_from_output(output) is None
    response = serving.request_output_to_completion_response(
        [output], request, "request", 0, "model", None,
        RequestResponseMetadata(request_id="request"),
    )
    assert response.usage.server_timing is None

    response = serving.request_output_to_completion_response(
        [make_output(), make_output()], request, "request", 0, "model", None,
        RequestResponseMetadata(request_id="request"),
    )
    assert response.usage.server_timing is None
