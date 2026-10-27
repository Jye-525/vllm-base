# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Analytic prefill FLOPs for the DSPF scheduler.

DSPF only sums and compares these costs, so any common FLOP-to-time factor
(GPU peak, utilization, tensor parallelism) cancels. Only Qwen3 dense and
Qwen3-MoE models are supported; other architectures raise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

SUPPORTED_ARCHITECTURES = ("Qwen3ForCausalLM", "Qwen3MoeForCausalLM")


@dataclass(frozen=True, slots=True)
class PrefillCostShape:
    """FLOP coefficients of one prefill.

    ``linear_flops_per_token`` is 2 x the weights a token passes through in
    the decoder layers (attention projections and the active MLP).
    ``attention_flops_per_pair`` is the score and value FLOPs of one causal
    (query, key) pair over all layers. ``lm_head_flops`` is the output layer,
    run once for the last prompt token.
    """

    linear_flops_per_token: int
    attention_flops_per_pair: int
    lm_head_flops: int

    @classmethod
    def from_model_config(cls, model_config: Any) -> PrefillCostShape:
        architectures = list(getattr(model_config, "architectures", None) or [])
        if not any(arch in SUPPORTED_ARCHITECTURES for arch in architectures):
            raise ValueError(
                "The 'dspf' scheduling policy supports only "
                f"{', '.join(SUPPORTED_ARCHITECTURES)}, not {architectures}"
            )
        config = model_config.hf_text_config
        hidden = config.hidden_size
        layers = config.num_hidden_layers
        heads = config.num_attention_heads
        kv_heads = config.num_key_value_heads
        head_dim = getattr(config, "head_dim", None) or hidden // heads
        # q and o projections, then k and v.
        attention_weights = 2 * hidden * heads * head_dim
        attention_weights += 2 * hidden * kv_heads * head_dim
        dense_mlp_weights = 3 * hidden * config.intermediate_size

        num_experts = getattr(config, "num_experts", 0) or 0
        shared = getattr(config, "shared_expert_intermediate_size", 0) or 0
        mlp_weights = 0
        for layer in range(layers):
            if _is_moe_layer(config, layer, num_experts):
                # Router, the active experts, and an optional shared expert
                # (all gated three-projection MLPs), as in qwen3_moe.py.
                experts = config.num_experts_per_tok * config.moe_intermediate_size
                mlp_weights += hidden * num_experts + 3 * hidden * (experts + shared)
            else:
                mlp_weights += dense_mlp_weights
        return cls(
            linear_flops_per_token=2 * (layers * attention_weights + mlp_weights),
            attention_flops_per_pair=4 * layers * heads * head_dim,
            lm_head_flops=2 * hidden * config.vocab_size,
        )

    def prefill_flops(self, num_uncached: int, num_cached: int) -> int:
        """FLOPs to prefill ``num_uncached`` tokens after ``num_cached``
        cached ones: the new tokens attend to the cached prefix and,
        causally, to each other."""
        if num_uncached <= 0:
            return 0
        pairs = num_uncached * num_cached + num_uncached * (num_uncached + 1) // 2
        return (
            self.linear_flops_per_token * num_uncached
            + self.attention_flops_per_pair * pairs
            + self.lm_head_flops
        )


def _is_moe_layer(config: Any, layer: int, num_experts: int) -> bool:
    """Qwen3-MoE's layer rule (qwen3_moe.py): sparse unless listed in
    ``mlp_only_layers``, every ``decoder_sparse_step``-th layer."""
    if num_experts <= 0 or layer in (getattr(config, "mlp_only_layers", None) or []):
        return False
    return (layer + 1) % getattr(config, "decoder_sparse_step", 1) == 0
