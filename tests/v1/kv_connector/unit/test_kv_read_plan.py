# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention.kv_read_plan import KVAttentionReadPlan, apply_kv_read_plans


def metadata(lengths):
    return SimpleNamespace(
        seq_lens=torch.tensor(lengths), seq_lens_cpu=torch.tensor(lengths),
        _seq_lens_cpu=torch.tensor(lengths), seq_lens_cpu_upper_bound=torch.tensor(lengths),
        _num_computed_tokens_cpu=torch.tensor(lengths) - 1, _num_computed_tokens_cache=None,
        block_table_tensor=torch.tensor([[9, 4, 6, 2, 8, 3], [7, 1, 5, 11, 12, 13]]),
        query_start_loc_cpu=torch.tensor([0, 1, 2]),
        positions=torch.tensor(lengths) - 1, slot_mapping=torch.tensor([33, 17]),
        max_seq_len=max(lengths), is_prefilling=torch.tensor([False, False]),
    )


@pytest.mark.parametrize("length", [49, 64, 65, 81])
def test_read_pages_follow_growth_without_changing_writes(length):
    common = metadata([length, 32])
    original_table = common.block_table_tensor.clone()
    plan = KVAttentionReadPlan(49, 16, (0, 2, 3))
    result = apply_kv_read_plans(common, [plan, None])
    pages = (0, 2, 3) + tuple(range(4, (length + 15) // 16))
    assert result.block_table_tensor[0, :len(pages)].tolist() == original_table[0, list(pages)].tolist()
    assert result.seq_lens.tolist() == [length - 16, 32]
    assert result._seq_lens_cpu.tolist() == [length - 16, 32]
    assert result._num_computed_tokens_cpu.tolist() == [length - 17, 31]
    assert result.slot_mapping is common.slot_mapping
    assert result.positions is common.positions
    assert result.is_prefilling is common.is_prefilling
    assert torch.equal(common.block_table_tensor, original_table)
    assert common.seq_lens.tolist() == [length, 32]


def test_different_requests_share_group_but_have_different_read_views():
    common = metadata([64, 64])
    result = apply_kv_read_plans(common, [KVAttentionReadPlan(64, 16, (0, 2, 3)),
                                         KVAttentionReadPlan(64, 16, (0, 1, 3))])
    assert result.block_table_tensor[0, :3].tolist() == [9, 6, 2]
    assert result.block_table_tensor[1, :3].tolist() == [7, 1, 11]


def test_unpruned_metadata_is_unchanged_and_tail_is_mandatory():
    common = metadata([64, 32])
    assert apply_kv_read_plans(common, [None, None]) is common
    with pytest.raises(ValueError):
        KVAttentionReadPlan(64, 16, (0, 1, 2))
