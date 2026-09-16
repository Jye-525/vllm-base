# SPDX-License-Identifier: Apache-2.0
"""Optional connector-owned prompt-page filtering for full attention.

Allocation and write-slot tables are deliberately not modified. The tail page
must be retained, so removing full interior pages preserves its valid length.
"""

from copy import copy
from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class KVAttentionReadPlan:
    prompt_length: int
    block_size: int
    retained_prompt_blocks: tuple[int, ...]

    def __post_init__(self):
        if self.prompt_length <= 0 or self.block_size <= 0:
            raise ValueError("KV read plan requires positive prompt length and block size")
        count = math.ceil(self.prompt_length / self.block_size)
        kept = self.retained_prompt_blocks
        if (not kept or tuple(sorted(set(kept))) != kept
                or kept[0] < 0 or kept[-1] != count - 1):
            raise ValueError("KV read plan requires sorted unique pages and retained tail")

    @property
    def dropped_tokens(self) -> int:
        return (math.ceil(self.prompt_length / self.block_size)
                - len(self.retained_prompt_blocks)) * self.block_size


def apply_kv_read_plans(common, plans):
    """Return independent attention metadata; never mutate common write state."""
    if not any(plan is not None and plan.dropped_tokens for plan in plans):
        return common
    result = copy(common)
    result.block_table_tensor = common.block_table_tensor.clone()
    result.seq_lens = common.seq_lens.clone()
    result._num_computed_tokens_cache = None
    lengths = common.seq_lens_cpu.clone()
    result._seq_lens_cpu = lengths
    if common._num_computed_tokens_cpu is not None:
        result._num_computed_tokens_cpu = common._num_computed_tokens_cpu.clone()
    if common.seq_lens_cpu_upper_bound is not None:
        result.seq_lens_cpu_upper_bound = common.seq_lens_cpu_upper_bound.clone()
    for row, plan in enumerate(plans):
        if plan is None or not plan.dropped_tokens:
            continue
        logical_length = int(lengths[row])
        if logical_length < plan.prompt_length:
            raise ValueError("pruned attention cannot run before final prompt token")
        # Hierarchical remote admission recomputes exactly one final prompt
        # token; subsequent calls are ordinary single-token decoding.
        query_len = int(common.query_start_loc_cpu[row + 1] - common.query_start_loc_cpu[row])
        if query_len != 1:
            raise ValueError("pruned attention currently requires single-token queries")
        prompt_pages = math.ceil(plan.prompt_length / plan.block_size)
        current_pages = math.ceil(logical_length / plan.block_size)
        logical_pages = plan.retained_prompt_blocks + tuple(range(prompt_pages, current_pages))
        indices = torch.tensor(logical_pages, dtype=torch.long,
                               device=common.block_table_tensor.device)
        read_pages = common.block_table_tensor[row].index_select(0, indices)
        result.block_table_tensor[row].zero_()
        result.block_table_tensor[row, :len(logical_pages)].copy_(read_pages)
        dropped = plan.dropped_tokens
        result.seq_lens[row] -= dropped
        lengths[row] -= dropped
        if result._num_computed_tokens_cpu is not None:
            result._num_computed_tokens_cpu[row] -= dropped
        if result.seq_lens_cpu_upper_bound is not None:
            result.seq_lens_cpu_upper_bound[row] -= dropped
    result.max_seq_len = int(lengths.max())
    return result
