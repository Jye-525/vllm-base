# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.v1.core.sched.prefill_cost import PrefillCostShape

# config.json values of the benchmark models.
QWEN3_8B = dict(
    hidden_size=4096,
    num_hidden_layers=36,
    intermediate_size=12288,
    num_attention_heads=32,
    num_key_value_heads=8,
    head_dim=128,
    vocab_size=151936,
)
QWEN3_CODER_30B_A3B = dict(
    hidden_size=2048,
    num_hidden_layers=48,
    intermediate_size=6144,
    moe_intermediate_size=768,
    num_experts=128,
    num_experts_per_tok=8,
    num_attention_heads=32,
    num_key_value_heads=4,
    head_dim=128,
    vocab_size=151936,
    decoder_sparse_step=1,
    mlp_only_layers=[],
    shared_expert_intermediate_size=None,
)


def _model(architecture: str, **config) -> SimpleNamespace:
    return SimpleNamespace(
        architectures=[architecture], hf_text_config=SimpleNamespace(**config)
    )


def _attention_weights(config: dict) -> int:
    d, heads, kv_heads, head_dim = (
        config["hidden_size"],
        config["num_attention_heads"],
        config["num_key_value_heads"],
        config["head_dim"],
    )
    return 2 * d * heads * head_dim + 2 * d * kv_heads * head_dim


def test_qwen3_8b_constants() -> None:
    shape = PrefillCostShape.from_model_config(_model("Qwen3ForCausalLM", **QWEN3_8B))

    # 36 x (41,943,040 attention + 150,994,944 MLP) weights per token.
    assert shape.linear_flops_per_token == 2 * 6_945_767_424
    assert shape.attention_flops_per_pair == 589_824
    assert shape.lm_head_flops == 2 * 4096 * 151936


def test_qwen3_moe_counts_router_and_active_experts_only() -> None:
    shape = PrefillCostShape.from_model_config(
        _model("Qwen3MoeForCausalLM", **QWEN3_CODER_30B_A3B)
    )

    moe = 2048 * 128 + 3 * 2048 * 8 * 768
    layer = _attention_weights(QWEN3_CODER_30B_A3B) + moe
    assert shape.linear_flops_per_token == 2 * 48 * layer
    assert shape.attention_flops_per_pair == 4 * 48 * 32 * 128


def test_moe_layer_rule_follows_sparse_step_and_mlp_only_layers() -> None:
    config = dict(
        QWEN3_CODER_30B_A3B,
        num_hidden_layers=4,
        decoder_sparse_step=2,
        mlp_only_layers=[3],
        shared_expert_intermediate_size=512,
    )
    shape = PrefillCostShape.from_model_config(_model("Qwen3MoeForCausalLM", **config))

    # Only layer 1 is sparse: (1 + 1) % 2 == 0, and layer 3 is MLP-only.
    moe = 2048 * 128 + 3 * 2048 * (8 * 768 + 512)
    dense = 3 * 2048 * 6144
    attention = 4 * _attention_weights(config)
    assert shape.linear_flops_per_token == 2 * (attention + moe + 3 * dense)


def test_prefill_flops_attends_to_cached_prefix_and_causally() -> None:
    shape = PrefillCostShape(
        linear_flops_per_token=10, attention_flops_per_pair=2, lm_head_flops=7
    )

    # 3 new tokens: 3 x 5 pairs with the cached prefix, 1 + 2 + 3 causal.
    assert shape.prefill_flops(3, 5) == 10 * 3 + 2 * (15 + 6) + 7
    assert shape.prefill_flops(0, 5) == 0


def test_unsupported_architecture_raises() -> None:
    with pytest.raises(ValueError, match="dspf"):
        PrefillCostShape.from_model_config(_model("LlamaForCausalLM", **QWEN3_8B))
