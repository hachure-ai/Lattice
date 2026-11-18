"""Lattice ICL encoder using block attention residuals."""

from __future__ import annotations

from typing import Optional, Sequence, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from lattice._model.kv_cache import KVCache, KVCacheEntry
from lattice._model.layers import MultiheadAttentionBlock
from lattice._model.rope import RotaryEmbedding

from .attn_res import AttentionResidual, empty_block_residual


class AttnResBlock(MultiheadAttentionBlock):
    """Lattice transformer layer with residual routing before each sublayer."""

    def __init__(self, *args, use_self_attention_res: bool = True, attn_res_softmax: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        d_model = self.norm1.normalized_shape[0]
        self.self_attention_res = (
            AttentionResidual(d_model, use_softmax=attn_res_softmax) if use_self_attention_res else None
        )
        self.mlp_res = AttentionResidual(d_model, use_softmax=attn_res_softmax)


class AttnResEncoder(nn.Module):
    """A shape-preserving transformer stack with Block AttnRes."""

    def __init__(
        self,
        num_blocks: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.0,
        activation: str | callable = "gelu",
        norm_first: bool = True,
        bias_free_ln: bool = False,
        use_rope: bool = False,
        rope_base: int = 100000,
        rope_interleaved: bool = True,
        ssmax: Union[bool, str] = False,
        zero_init: bool = True,
        recompute: bool = False,
        attn_res_block_size: int = 2,
        attn_res_softmax: bool = True,
        ffn_impl: str = "mlp",
        sandwich_post_norm: bool = False,
        qk_norm: bool = False,
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by nhead ({nhead})"
            )
        if not norm_first:
            raise ValueError("Kimi Attention Residuals require a pre-norm transformer.")
        if attn_res_block_size <= 0:
            raise ValueError("attn_res_block_size must be greater than zero.")
        if num_blocks <= 0:
            raise ValueError("num_blocks must be greater than zero for AttnRes.")

        self.blocks = nn.ModuleList(
            [
                AttnResBlock(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation=activation,
                    norm_first=norm_first,
                    bias_free_ln=bias_free_ln,
                    ssmax=ssmax,
                    zero_init=zero_init,
                    use_self_attention_res=layer_idx > 0,
                    attn_res_softmax=attn_res_softmax,
                    ffn_impl=ffn_impl,
                    sandwich_post_norm=sandwich_post_norm,
                    qk_norm=qk_norm,
                )
                for layer_idx in range(num_blocks)
            ]
        )
        self.output_attn_res = AttentionResidual(d_model, use_softmax=attn_res_softmax)
        self.attn_res_block_size = attn_res_block_size
        self.attn_res_softmax = attn_res_softmax
        self.rope = (
            RotaryEmbedding(
                dim=d_model // nhead, theta=rope_base, interleaved=rope_interleaved
            )
            if use_rope
            else None
        )
        self.recompute = recompute

    @staticmethod
    def _append_completed_block(block_residual: Tensor, prefix_sum: Tensor) -> Tensor:
        return torch.cat((block_residual, prefix_sum.unsqueeze(-2)), dim=-2)

    def _forward_layer(
        self,
        block: AttnResBlock,
        layer_idx: int,
        prefix_sum: Tensor,
        block_residual: Tensor,
        train_size: Optional[int] = None,
        cached_kv: Optional[KVCacheEntry] = None,
        need_kv: bool = False,
        need_weights: bool = False,
    ) -> tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        hidden_states = (
            prefix_sum
            if block.self_attention_res is None
            else block.self_attention_res(prefix_sum, block_residual)
        )

        if layer_idx % self.attn_res_block_size == 0:
            block_residual = self._append_completed_block(block_residual, prefix_sum)
            prefix_sum = None

        hidden_states = block.norm1(hidden_states)
        scores = None
        if cached_kv is not None:
            attention_result = block._attn_block(
                hidden_states,
                cached_kv=cached_kv,
                rope=self.rope,
            )
        else:
            key_value_states = (
                hidden_states
                if train_size is None
                else hidden_states[..., :train_size, :]
            )
            attention_result = block._attn_block(
                hidden_states,
                key_value_states,
                key_value_states,
                rope=self.rope,
                need_kv=need_kv,
                need_weights=need_weights,
            )

        key_projection = value_projection = None
        if need_kv:
            attention_output, key_projection, value_projection = attention_result
        elif need_weights:
            attention_output, scores = attention_result
        else:
            attention_output = attention_result

        # Sandwich post-norm ablation: normalize the sublayer output before
        # the residual add (Identity when the ablation is off).
        attention_output = block.post_attn_norm(attention_output)
        prefix_sum = (
            attention_output if prefix_sum is None else prefix_sum + attention_output
        )
        hidden_states = block.mlp_res(prefix_sum, block_residual)
        mlp_output = block._ff_block(block.norm2(hidden_states))
        prefix_sum = prefix_sum + block.post_ff_norm(mlp_output)
        return prefix_sum, block_residual, key_projection, value_projection, scores

    def forward(
        self,
        src: Tensor,
        train_size: Optional[int] = None,
        idempotency_loop_counts: Optional[Sequence[int]] = None,
        return_last_attn: bool = False,
    ) -> Union[Tensor, tuple[Tensor, Tensor]]:
        """Run the AttnRes stack.

        If ``return_last_attn`` is True and ``train_size`` is given, also
        returns the pre-softmax test-to-train attention scores of the final
        block, of shape (..., num_heads, tgt_len, train_size). Scores require
        dropout to be zero (they are exact weights).
        """
        if idempotency_loop_counts is not None:
            raise ValueError(
                "Layer-wise idempotency loss is not supported by the AttnRes encoder."
            )
        if return_last_attn and train_size is None:
            raise ValueError("return_last_attn requires train_size.")
        if return_last_attn and self.blocks[-1].attn.dropout > 0:
            raise ValueError("return_last_attn requires attention dropout = 0.")

        prefix_sum = src
        block_residual = empty_block_residual(src)
        last_scores = None
        for layer_idx, block in enumerate(self.blocks):
            need_weights = return_last_attn and layer_idx == len(self.blocks) - 1
            if self.recompute:

                def layer_forward(
                    prefix: Tensor,
                    residual: Tensor,
                    current_block: AttnResBlock = block,
                    current_layer_idx: int = layer_idx,
                    current_need_weights: bool = need_weights,
                ) -> tuple[Tensor, Tensor]:
                    next_prefix, next_residual, _, _, scores = self._forward_layer(
                        current_block,
                        current_layer_idx,
                        prefix,
                        residual,
                        train_size=train_size,
                        need_weights=current_need_weights,
                    )
                    if current_need_weights:
                        return next_prefix, next_residual, scores
                    return next_prefix, next_residual

                layer_result = checkpoint(
                    layer_forward,
                    prefix_sum,
                    block_residual,
                    use_reentrant=False,
                )
                if need_weights:
                    prefix_sum, block_residual, last_scores = layer_result
                else:
                    prefix_sum, block_residual = layer_result
            else:
                prefix_sum, block_residual, _, _, scores = self._forward_layer(
                    block,
                    layer_idx,
                    prefix_sum,
                    block_residual,
                    train_size=train_size,
                    need_weights=need_weights,
                )
                if need_weights:
                    last_scores = scores

        output = self.output_attn_res(prefix_sum, block_residual)
        if return_last_attn:
            return output, last_scores
        return output

    def forward_with_cache(
        self,
        src: Tensor,
        icl_cache: KVCache,
        train_size: Optional[int] = None,
        use_cache: bool = False,
        store_cache: bool = True,
        *,
        cache_only: bool = False,
    ) -> Optional[Tensor]:
        if use_cache == store_cache:
            raise ValueError("Exactly one of use_cache or store_cache must be True")
        if store_cache and train_size is None:
            raise ValueError("train_size must be provided when store_cache=True")

        prefix_sum = src
        block_residual = empty_block_residual(src)
        for layer_idx, block in enumerate(self.blocks):
            if cache_only and layer_idx == len(self.blocks) - 1:
                # The final support layer only needs to publish its K/V.
                hidden = (
                    prefix_sum
                    if block.self_attention_res is None
                    else block.self_attention_res(prefix_sum, block_residual)
                )
                query = block.norm1(hidden)
                support = query[..., :train_size, :]
                attn = block.attn
                # Keep the same packed projection as the complete forward.
                _, key, value = F._in_projection_packed(
                    query, support, support, attn.in_proj_weight, attn.in_proj_bias
                )
                shape = (*support.shape[:-1], attn.num_heads, attn.head_dim)
                key = key.view(shape).transpose(-3, -2)
                value = value.view(shape).transpose(-3, -2)
                if self.rope is not None:
                    key = self.rope.rotate_queries_or_keys(key)
                if attn.k_norm is not None:
                    key = attn.k_norm(key)
                icl_cache.kv[layer_idx] = KVCacheEntry(key=key, value=value)
                return None
            cached_kv = icl_cache.kv[layer_idx] if use_cache else None
            prefix_sum, block_residual, key_projection, value_projection, _ = (
                self._forward_layer(
                    block,
                    layer_idx,
                    prefix_sum,
                    block_residual,
                    train_size=train_size if store_cache else None,
                    cached_kv=cached_kv,
                    need_kv=store_cache,
                )
            )
            if store_cache:
                icl_cache.kv[layer_idx] = KVCacheEntry(
                    key=key_projection, value=value_projection
                )

        return self.output_attn_res(prefix_sum, block_residual)
