# Adapted from TabICL v2 (https://github.com/soda-inria/tabicl).
# Copyright (c) 2025 Soda team @ Inria. BSD 3-Clause License; see NOTICE and
# src/lattice/_model/LICENSE.

"""Lattice attention extended with element-wise gating."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from lattice._model.attention import sdpa_with_flattened_batch
from lattice._model.kv_cache import KVCacheEntry
from lattice._model.norms import attach_qk_norm
from lattice._model.rope import RotaryEmbedding
from lattice._model.ssmax import create_ssmax_layer


def gated_multi_head_attention_forward(
    query: Tensor,
    num_heads: int,
    in_proj_weight: Tensor,
    in_proj_bias: Optional[Tensor],
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Optional[Tensor],
    gate_proj_weight: Tensor,
    gate_proj_bias: Optional[Tensor],
    key: Optional[Tensor] = None,
    value: Optional[Tensor] = None,
    cached_kv: Optional[KVCacheEntry] = None,
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    attn_mask: Optional[Tensor] = None,
    rope: Optional[RotaryEmbedding] = None,
    ssmax_layer: Optional[nn.Module] = None,
    q_norm: Optional[nn.Module] = None,
    k_norm: Optional[nn.Module] = None,
    need_kv: bool = False,
    need_weights: bool = False,
) -> Union[Tensor, Tuple[Tensor, ...]]:
    """Apply an element-wise gate to each per-head SDPA output.

    If ``need_weights`` is True, returns ``(gated_output, scores)`` where
    scores are the pre-softmax attention scores of shape
    (..., num_heads, tgt_len, src_len). The gate multiplies the attention
    output only and leaves the scores untouched. ``q_norm``/``k_norm``
    implement the QK-RMSNorm ablation exactly like the plain
    attention path.
    """

    if need_weights and cached_kv is not None:
        raise ValueError("need_weights is not supported together with cached_kv.")
    if need_weights and need_kv:
        raise ValueError("need_weights and need_kv cannot be combined.")

    *batch_shape, target_len, embed_dim = query.shape
    head_dim = embed_dim // num_heads
    if head_dim * num_heads != embed_dim:
        raise ValueError(
            f"embed_dim {embed_dim} is not divisible by num_heads {num_heads}"
        )

    gate = F.linear(query, gate_proj_weight, gate_proj_bias)
    gate = gate.view(*batch_shape, target_len, num_heads, head_dim).transpose(-3, -2)

    if cached_kv is None:
        if key is None or value is None:
            raise ValueError("key and value must be provided when cached_kv is None")
        source_len = key.shape[-2]
        if key.shape != value.shape:
            raise ValueError(
                f"key shape {key.shape} does not match value shape {value.shape}"
            )
        q, k, v = F._in_projection_packed(
            query, key, value, in_proj_weight, in_proj_bias
        )
        q = q.view(*batch_shape, target_len, num_heads, head_dim).transpose(-3, -2)
        k = k.view(*batch_shape, source_len, num_heads, head_dim).transpose(-3, -2)
        v = v.view(*batch_shape, source_len, num_heads, head_dim).transpose(-3, -2)
        if rope is not None:
            q = rope.rotate_queries_or_keys(q)
            k = rope.rotate_queries_or_keys(k)
        if q_norm is not None:
            q = q_norm(q)
        if k_norm is not None:
            k = k_norm(k)
    else:
        k, v = cached_kv.key, cached_kv.value
        source_len = k.shape[-2]
        q_proj_weight = in_proj_weight[:embed_dim]
        q_proj_bias = in_proj_bias[:embed_dim] if in_proj_bias is not None else None
        q = F.linear(query, q_proj_weight, q_proj_bias)
        q = q.view(*batch_shape, target_len, num_heads, head_dim).transpose(-3, -2)
        if rope is not None:
            q = rope.rotate_queries_or_keys(q)
        if q_norm is not None:
            q = q_norm(q)

    if not training:
        dropout_p = 0.0

    expected_2d_shape = (target_len, source_len)
    expected_nd_shape = (*batch_shape, num_heads, target_len, source_len)
    if attn_mask is not None:
        if attn_mask.dim() == 2:
            if attn_mask.shape != expected_2d_shape:
                raise ValueError(
                    f"Expected attn_mask shape {expected_2d_shape}, got {attn_mask.shape}"
                )
            attn_mask = attn_mask.expand(expected_nd_shape)
        elif attn_mask.dim() == len(expected_nd_shape):
            if attn_mask.shape != expected_nd_shape:
                raise ValueError(
                    f"Expected attn_mask shape {expected_nd_shape}, got {attn_mask.shape}"
                )
        else:
            raise ValueError(
                f"attn_mask must be 2D or {len(expected_nd_shape)}D, got {attn_mask.dim()}D"
            )

    if key_padding_mask is not None:
        expected_padding_shape = (*batch_shape, source_len)
        if key_padding_mask.shape != expected_padding_shape:
            raise ValueError(
                f"Expected key_padding_mask shape {expected_padding_shape}, "
                f"got {key_padding_mask.shape}"
            )
        padding_mask = key_padding_mask.view(*batch_shape, 1, 1, source_len).expand(
            expected_nd_shape
        )
        attn_mask = padding_mask if attn_mask is None else attn_mask + padding_mask

    attention_result = sdpa_with_flattened_batch(
        q,
        k,
        v,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        ssmax_layer=ssmax_layer,
        need_weights=need_weights,
    )
    if need_weights:
        attention_output, scores = attention_result
    else:
        attention_output = attention_result
    attention_output = attention_output * torch.sigmoid(gate)
    attention_output = (
        attention_output.transpose(-3, -2)
        .contiguous()
        .view(*batch_shape, target_len, embed_dim)
    )
    attention_output = F.linear(attention_output, out_proj_weight, out_proj_bias)

    if need_kv and cached_kv is None:
        return attention_output, k, v
    if need_weights:
        return attention_output, scores
    return attention_output


class GatedMultiheadAttention(nn.MultiheadAttention):
    """Lattice attention with a query-conditioned element-wise output gate."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        ssmax: Union[bool, str] = False,
        qk_norm: bool = False,
    ) -> None:
        super().__init__(embed_dim, num_heads, dropout=dropout, batch_first=True)
        if isinstance(ssmax, bool):
            ssmax = "qassmax-mlp-elementwise" if ssmax else "none"
        self.ssmax_layer = create_ssmax_layer(
            ssmax_type=ssmax,
            num_heads=num_heads,
            embed_dim=embed_dim,
        )
        self.attn_gate = nn.Linear(embed_dim, embed_dim)
        self.qk_norm = qk_norm
        attach_qk_norm(self, qk_norm, head_dim=embed_dim // num_heads)

    def forward(
        self,
        query: Tensor,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        cached_kv: Optional[KVCacheEntry] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        rope: Optional[RotaryEmbedding] = None,
        need_kv: bool = False,
        need_weights: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, ...]]:
        key_padding_mask = F._canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=F._none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=query.dtype,
        )
        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )

        return gated_multi_head_attention_forward(
            query=query,
            num_heads=self.num_heads,
            in_proj_weight=self.in_proj_weight,
            in_proj_bias=self.in_proj_bias,
            dropout_p=self.dropout,
            out_proj_weight=self.out_proj.weight,
            out_proj_bias=self.out_proj.bias,
            gate_proj_weight=self.attn_gate.weight,
            gate_proj_bias=self.attn_gate.bias,
            key=key,
            value=value,
            cached_kv=cached_kv,
            training=self.training,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            rope=rope,
            ssmax_layer=self.ssmax_layer,
            q_norm=self.q_norm,
            k_norm=self.k_norm,
            need_kv=need_kv,
            need_weights=need_weights,
        )
