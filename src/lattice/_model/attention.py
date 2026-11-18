# Adapted from TabICL v2 (https://github.com/soda-inria/tabicl).
# Copyright (c) 2025 Soda team @ Inria. BSD 3-Clause License; see NOTICE and
# src/lattice/_model/LICENSE.

from __future__ import annotations
from contextlib import contextmanager
from typing import Optional, Tuple, Union
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .rope import RotaryEmbedding
from .kv_cache import KVCacheEntry

try:
    from flash_attn_interface import flash_attn_varlen_func as flash_attn3

    HAS_FLASH_ATTN3 = True
except ImportError:
    HAS_FLASH_ATTN3 = False

_use_flash_attn3 = True

# CUDA SDPA kernels can use a 16-bit grid dimension for the flattened batch.
_SDPA_MAX_BATCH_SIZE = 65_535


@contextmanager
def flash_attn3_toggle(enabled: bool):
    """Context manager to temporarily enable or disable Flash Attention 3.

    Used by ``InferenceManager._run_forward()`` in ``inference.py`` to control
    whether Flash Attention 3 is used during each forward pass based on the
    ``use_fa3`` configuration.
    """
    global _use_flash_attn3
    old = _use_flash_attn3
    _use_flash_attn3 = enabled
    try:
        yield
    finally:
        _use_flash_attn3 = old


def set_flash_attn3_enabled(enabled: bool):
    """Globally enable or disable Flash Attention 3 for this process."""
    global _use_flash_attn3
    _use_flash_attn3 = enabled


def sdpa_with_flattened_batch(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
    ssmax_layer: Optional[nn.Module] = None,
    need_weights: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Applies scaled dot-product attention with flattened batch dimensions.

    This function handles arbitrary batch dimensions by flattening them before
    applying PyTorch's ``scaled_dot_product_attention`` and then reshaping the
    output back to the original shape. This flattening is necessary to properly
    trigger Flash Attention.

    Parameters
    ----------
    q : Tensor
        Query tensor of shape (..., nh, tgt_len, hs) where:

        - ... represents arbitrary batch dimensions
        - nh is the number of attention heads
        - tgt_len is the target sequence length
        - hs is the head size (embedding dimension per head)

    k : Tensor
        Key tensor of shape (..., nh, src_len, hs) with matching batch dimensions.

    v : Tensor
        Value tensor of shape (..., nh, src_len, hs) with matching batch dimensions.

    attn_mask : Optional[Tensor], default=None
        Attention mask of shape (..., nh, tgt_len, src_len).

    dropout_p : float, default=0.0
        Dropout probability applied to attention weights.

    ssmax_layer : Optional[nn.Module], default=None
        If provided, applies scalable softmax (SSMax) scaling to queries before
        attention computation.

    need_weights : bool, default=False
        If True, computes attention explicitly and returns the pre-softmax
        scores of shape (..., nh, tgt_len, src_len) together with the output.
        SSMax query scaling is applied identically to the SDPA path so the
        scores match the internally-used attention. Flash Attention and
        dropout are disabled in this mode.

    Returns
    -------
    Union[Tensor, Tuple[Tensor, Tensor]]
        Attention output tensor of shape (..., nh, tgt_len, hs) preserving
        the original batch dimensions of the input, or a tuple
        ``(output, scores)`` when ``need_weights=True``.
    """
    if need_weights and dropout_p != 0.0:
        raise ValueError("need_weights does not support dropout.")

    q_shape = q.shape
    q = q.reshape(-1, *q.shape[-3:])
    k = k.reshape(-1, *k.shape[-3:])
    v = v.reshape(-1, *v.shape[-3:])
    if attn_mask is not None:
        attn_mask = attn_mask.reshape(-1, *attn_mask.shape[-3:])

    if ssmax_layer is not None:
        src_len = k.size(-2)
        q = ssmax_layer(q, src_len)

    if need_weights:
        head_dim = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                scores = scores.masked_fill(attn_mask, float("-inf"))
            else:
                scores = scores + attn_mask
        weights = scores.softmax(dim=-1)
        out = torch.matmul(weights, v)
        scores = scores.view(*q_shape[:-1], k.shape[-2])
        return out.view(q_shape), scores

    # FlashAttn3 doesn't support dropout and custom attention mask
    if HAS_FLASH_ATTN3 and _use_flash_attn3 and q.is_cuda and attn_mask is None and dropout_p == 0.0:
        # FlashAttention only supports fp16, bf16, and fp8_e4m3
        # Convert to bf16 if needed, then convert back to original dtype
        orig_dtype = q.dtype
        if orig_dtype not in (torch.float16, torch.bfloat16):
            fa_dtype = torch.float16
        else:
            fa_dtype = orig_dtype

        flat_bs, nheads, seqlen_q, headdim = q.shape
        seqlen_k = k.shape[-2]
        q_fa = q.transpose(1, 2).reshape(flat_bs * seqlen_q, nheads, headdim).contiguous().to(fa_dtype)
        k_fa = k.transpose(1, 2).reshape(flat_bs * seqlen_k, nheads, headdim).contiguous().to(fa_dtype)
        v_fa = v.transpose(1, 2).reshape(flat_bs * seqlen_k, nheads, headdim).contiguous().to(fa_dtype)
        cu_seqlens_q = torch.arange(0, (flat_bs + 1) * seqlen_q, seqlen_q, dtype=torch.int32, device=q.device)
        cu_seqlens_k = torch.arange(0, (flat_bs + 1) * seqlen_k, seqlen_k, dtype=torch.int32, device=q.device)
        out = flash_attn3(q_fa, k_fa, v_fa, cu_seqlens_q, cu_seqlens_k, seqlen_q, seqlen_k)
        out = out.view(flat_bs, seqlen_q, nheads, headdim).transpose(1, 2).to(orig_dtype)
    elif q.is_cuda and q.size(0) > _SDPA_MAX_BATCH_SIZE:
        # Row attention flattens (ensemble, rows) into independent batch items.
        # Chunk that axis only: every item keeps its complete key/value context.
        # Keep singleton K/V or mask batches intact for SDPA broadcasting.
        def batch_slice(tensor, start, end):
            if tensor is None or tensor.size(0) == 1:
                return tensor
            return tensor[start:end]

        outputs = []
        for start in range(0, q.size(0), _SDPA_MAX_BATCH_SIZE):
            end = start + _SDPA_MAX_BATCH_SIZE
            outputs.append(F.scaled_dot_product_attention(
                q[start:end], batch_slice(k, start, end), batch_slice(v, start, end),
                batch_slice(attn_mask, start, end), dropout_p,
            ))
        out = torch.cat(outputs, dim=0)
    else:
        out = F.scaled_dot_product_attention(q, k, v, attn_mask, dropout_p)

    return out.view(q_shape)


def multi_head_attention_forward(
    query: Tensor,
    num_heads: int,
    in_proj_weight: Tensor,
    in_proj_bias: Tensor,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Tensor,
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
    """Multi-head attention with support for rotary position embeddings.

    Parameters
    ----------
    query : Tensor
        Query tensor of shape (..., tgt_len, embed_dim).

    num_heads : int
        Number of attention heads.

    in_proj_weight : Tensor
        Combined weight matrix for Q, K, V input projections.

    in_proj_bias : Tensor
        Combined bias vector for input projections.

    dropout_p : float
        Dropout probability applied to attention weights.

    out_proj_weight : Tensor
        Output projection weight matrix.

    out_proj_bias : Tensor
        Output projection bias vector.

    key : Optional[Tensor], default=None
        Key tensor of shape (..., src_len, embed_dim).
        Required when ``cached_kv`` is None.

    value : Optional[Tensor], default=None
        Value tensor of shape (..., src_len, embed_dim).
        Required when ``cached_kv`` is None.

    cached_kv : Optional[KVCacheEntry], default=None
        Pre-computed key and value projections for caching. When provided:

        - key and value parameters are ignored
        - Only query projection is computed
        - cached_kv.key shape: (..., num_heads, src_len, head_dim)
        - cached_kv.value shape: (..., num_heads, src_len, head_dim)
        - RoPE is applied only to queries (keys should already have RoPE applied)

    training : bool, default=True
        Whether the model is in training mode (affects dropout).

    key_padding_mask : Optional[Tensor], default=None
        Mask of shape (..., src_len) that identifies padding elements
        in the key sequence to be ignored:

        - For binary masks: True values indicate positions to ignore.
        - For float masks: Values are directly added to attention scores.

    attn_mask : Optional[Tensor], default=None
        Attention mask of shape (tgt_len, src_len) or
        (..., num_heads, tgt_len, src_len).

    rope : Optional[RotaryEmbedding]
        Rotary positional encoding.

    ssmax_layer : Optional[nn.Module], default=None
        If provided, applies scalable softmax (SSMax) scaling to queries before
        attention computation.

    q_norm : Optional[nn.Module], default=None
        If provided, RMS-normalizes each query head after projection (and
        after RoPE) as part of the QK-norm ablation.

    k_norm : Optional[nn.Module], default=None
        If provided, RMS-normalizes each key head after projection (and after
        RoPE). Applied before K/V caching so cached keys stay normalized.

    need_kv : bool, default=False
        If True and ``cached_kv`` is None, also returns the computed K and V
        projections along with the attention output. Useful for caching K/V for
        subsequent calls.

    need_weights : bool, default=False
        If True, also return the pre-softmax attention scores of shape
        (..., num_heads, tgt_len, src_len). Incompatible with ``cached_kv``
        and dropout. Computed with the same SSMax query scaling as the
        standard path so the scores match the internal attention weights.

    Returns
    -------
    Union[Tensor, Tuple[Tensor, ...]]
        If neither ``need_kv`` nor ``need_weights`` is set, or ``cached_kv``
        is provided:
            Attention output tensor of shape (..., tgt_len, embed_dim).
        If ``need_kv`` is True and ``cached_kv`` is None:
            Tuple of (attn_output, k, v) where:

            - attn_output: shape (..., tgt_len, embed_dim)
            - k: shape (..., num_heads, src_len, head_dim)
            - v: shape (..., num_heads, src_len, head_dim)
        If ``need_weights`` is True (and ``need_kv`` is False):
            Tuple of (attn_output, scores) where scores has shape
            (..., num_heads, tgt_len, src_len).
    """

    if need_weights and cached_kv is not None:
        raise ValueError("need_weights is not supported together with cached_kv.")
    if need_weights and need_kv:
        raise ValueError("need_weights and need_kv cannot be combined.")

    # Extract shape information, supporting arbitrary batch dimensions
    *batch_shape, tgt_len, embed_dim = query.shape
    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"

    if cached_kv is None:
        # Standard: project Q, K, V jointly
        if key is None or value is None:
            raise ValueError("key and value must be provided when cached_kv is None")
        src_len = key.shape[-2]
        assert key.shape == value.shape, f"key shape {key.shape} does not match value shape {value.shape}"
        q, k, v = F._in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
        q = q.view(*batch_shape, tgt_len, num_heads, head_dim).transpose(-3, -2)
        k = k.view(*batch_shape, src_len, num_heads, head_dim).transpose(-3, -2)
        v = v.view(*batch_shape, src_len, num_heads, head_dim).transpose(-3, -2)
        if rope is not None:
            q = rope.rotate_queries_or_keys(q)
            k = rope.rotate_queries_or_keys(k)
        if q_norm is not None:
            q = q_norm(q)
        if k_norm is not None:
            k = k_norm(k)
    else:
        # Use cached K/V, project Q only
        k, v = cached_kv.key, cached_kv.value
        src_len = k.shape[-2]
        q_proj_weight = in_proj_weight[:embed_dim]
        q_proj_bias = in_proj_bias[:embed_dim] if in_proj_bias is not None else None
        q = F.linear(query, q_proj_weight, q_proj_bias)
        q = q.view(*batch_shape, tgt_len, num_heads, head_dim).transpose(-3, -2)
        if rope is not None:
            q = rope.rotate_queries_or_keys(q)
        if q_norm is not None:
            q = q_norm(q)

    # Disable dropout during evaluation
    if not training:
        dropout_p = 0.0

    # Process attention mask
    correct_2d_shape = (tgt_len, src_len)
    correct_nd_shape = (*batch_shape, num_heads, tgt_len, src_len)
    if attn_mask is not None:
        if attn_mask.dim() == 2:
            if attn_mask.shape != correct_2d_shape:
                raise ValueError(f"2D attn_mask should have shape {correct_2d_shape}, but got {attn_mask.shape}")
            attn_mask = attn_mask.expand(*batch_shape, num_heads, tgt_len, src_len)
        elif attn_mask.dim() == len(correct_nd_shape):
            if attn_mask.shape != correct_nd_shape:
                raise ValueError(
                    f"{len(correct_nd_shape)}D attn_mask should have shape {correct_nd_shape}, "
                    f"but got {attn_mask.shape}"
                )
        else:
            raise ValueError(f"attn_mask must be 2D or {len(correct_nd_shape)}D, got {attn_mask.dim()}D")

    # Process key padding mask
    if key_padding_mask is not None:
        if key_padding_mask.shape != (*batch_shape, src_len):
            raise ValueError(
                f"key_padding_mask should have shape {(*batch_shape, src_len)}, but got {key_padding_mask.shape}"
            )
        key_padding_mask = key_padding_mask.view(*batch_shape, 1, 1, src_len).expand(
            *batch_shape, num_heads, tgt_len, src_len
        )

        if attn_mask is None:
            attn_mask = key_padding_mask
        else:
            attn_mask = attn_mask + key_padding_mask

    attn_result = sdpa_with_flattened_batch(
        q, k, v, attn_mask, dropout_p, ssmax_layer=ssmax_layer, need_weights=need_weights
    )  # (..., nh, tgt_len, hs) or (output, scores)

    if need_weights:
        attn_output, scores = attn_result
    else:
        attn_output = attn_result

    # Reshape and project output
    attn_output = attn_output.transpose(-3, -2).contiguous().view(*batch_shape, tgt_len, embed_dim)
    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)  # (batch_shape, tgt_len, E)

    if need_kv and cached_kv is None:
        return attn_output, k, v

    if need_weights:
        return attn_output, scores

    return attn_output
