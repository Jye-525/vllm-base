# SPDX-License-Identifier: Apache-2.0
"""Work fields of the ``schedule_batch`` log line (step cost model input)."""

from types import SimpleNamespace as NS

import pytest

from vllm.v1.core.sched.scheduler import Scheduler

pytestmark = pytest.mark.cpu_test


def test_execution_work_splits_prefill_replay_and_decode():
    requests = {
        # Prompt chunk: positions 0..7 of a 10-token prompt.
        "prompt": NS(num_prompt_tokens=10, num_tokens=10),
        # Decode: position 11 of 12 tokens.
        "decode": NS(num_prompt_tokens=10, num_tokens=12),
        # Resumed after 8 hit tokens: positions 8..10 replay as prefill,
        # position 11 is the decode token.
        "resumed": NS(num_prompt_tokens=10, num_tokens=12),
    }
    intervals = {"prompt": (0, 8, 0), "decode": (11, 12, 0), "resumed": (8, 12, 12)}
    work = Scheduler._execution_work(NS(requests=requests), intervals)
    # prefill: 8 + 0 + 3; decode: 0 + 1 + 1
    # prefill_attention: (1 + ... + 8) + (9 + 10 + 11) = 36 + 30
    # decode_context: 12 + 12
    assert work == (11, 2, 66, 24)


def test_execution_work_skips_unknown_requests():
    work = Scheduler._execution_work(NS(requests={}), {"gone": (0, 4, 0)})
    assert work == (0, 0, 0, 0)
