# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from collections.abc import AsyncGenerator, Awaitable

from vllm.benchmarks.serve import _execute_request_tasks


def test_execute_request_tasks_in_complete_waves() -> None:
    async def run() -> None:
        completed: set[int] = set()
        active = 0
        peak_active = 0

        async def request(index: int) -> int:
            nonlocal active, peak_active
            active += 1
            peak_active = max(peak_active, active)
            await asyncio.sleep(0)
            active -= 1
            completed.add(index)
            return index

        async def tasks() -> AsyncGenerator[Awaitable[int], None]:
            for index in range(5):
                if index and index % 2 == 0:
                    assert set(range(index)).issubset(completed)
                yield request(index)

        outputs = await _execute_request_tasks(tasks(), wave_size=2)
        assert outputs == [0, 1, 2, 3, 4]
        assert peak_active == 2

    asyncio.run(run())


def test_execute_request_tasks_continuously_by_default() -> None:
    async def run() -> None:
        started: set[int] = set()
        all_started = asyncio.Event()

        async def request(index: int) -> int:
            started.add(index)
            if len(started) == 5:
                all_started.set()
            await all_started.wait()
            return index

        async def tasks() -> AsyncGenerator[Awaitable[int], None]:
            for index in range(5):
                yield request(index)

        outputs = await asyncio.wait_for(
            _execute_request_tasks(tasks(), wave_size=None), timeout=1
        )
        assert outputs == [0, 1, 2, 3, 4]
        assert started == set(range(5))

    asyncio.run(run())
