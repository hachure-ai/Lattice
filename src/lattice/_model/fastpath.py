"""Install the QK and ICL kernels on supported inference models."""

import os
from contextlib import ExitStack
from functools import lru_cache
from types import MethodType

import torch
import torch.nn.functional as F
from triton.errors import TritonError

from ._fast_attnres import fast_encoder_forward, fast_encoder_forward_with_cache, write_slot
from ._fast_cell import fused_norm, make_rope_norm
from ._fast_fused import (
    fused_layer_norm, fused_rms_norm, fused_swiglu, fused_swiglu_forward, install_fused,
    pack_projection_pair, packed_projections, packed_swiglu_forward,
)
from ._fast_norm214 import HAS_FUSED_RMS_NORM, layer_norm_forward
from .layers import MultiheadAttentionBlock, SwiGLU
from .norms import QKRMSNorm
from .ssmax import QASSMaxMLP, _logn


@lru_cache(maxsize=None)
@torch.no_grad()
def _qk_selftest(device):
    generator = torch.Generator(device=device).manual_seed(0)
    norm = QKRMSNorm(16).to(device)
    # A few hundred elements can miss fp16 rounding boundaries even if the
    # reduction order is wrong. Probe enough real-valued gains on the new path.
    if HAS_FUSED_RMS_NORM:
        norm.weight.copy_(torch.randn(16, device=device, generator=generator))
    shape = (2, 257, 8, 16) if HAS_FUSED_RMS_NORM else (2, 3, 8, 16)
    x = torch.randn(shape, device=device, generator=generator)
    x = (x * 0.5).to(torch.float16)
    try:
        return torch.equal(norm(x), fused_norm(norm, x))
    except (TritonError, RuntimeError):
        return False


@lru_cache(maxsize=None)
@torch.no_grad()
def _encoder_selftest(device, width):
    """Check the selected ATen reduction order before installing the encoder."""
    rng = torch.Generator(device=device).manual_seed(103)
    rows = 137
    with torch.cuda.device(device), torch.autocast("cuda", enabled=False):
        x = torch.randn(rows, width, dtype=torch.float32, device=device, generator=rng)
        weight = torch.randn(width, device=device, dtype=x.dtype, generator=rng)
        scales = x.new_empty(rows)
        scores = torch.empty_like(scales)
        try:
            write_slot[(rows,)](x, torch.empty_like(x), scales, 1, 1e-6, 0, width,
                                weight, scores, 1, True, N=width // 128, INV=1 / width)
            native_scales = torch.rsqrt(x.square().mean(-1) + 1e-6)
            native_scores = ((x * native_scales[:, None]) * weight).sum(-1)
            if not torch.equal(scales, native_scales) or not torch.equal(scores, native_scores):
                return False
            if HAS_FUSED_RMS_NORM:
                norm = torch.nn.LayerNorm(width, device=device, dtype=x.dtype)
                norm.weight.copy_(weight)
                norm.bias.copy_(x[0])
                if not torch.equal(layer_norm_forward(norm, x), norm(x)):
                    return False
                for gate in (x, x.half()):
                    up = gate.flip((-1,))
                    fused = fused_swiglu(gate, up)
                    if fused is None or not torch.equal(fused, F.silu(gate) * up):
                        return False
        except (TritonError, RuntimeError):
            return False
    return True


@lru_cache(maxsize=None)
def _fused_selftest(device):
    """Check the fused norm/FFN kernels against the eager fp16 reference."""
    generator = torch.Generator(device=device).manual_seed(0)
    x = torch.randn((64, 512), device=device, generator=generator).half()
    w = torch.randn((512,), device=device, generator=generator)
    b = torch.randn((512,), device=device, generator=generator)
    ref = F.layer_norm(x.float(), (512,), w, b, 1e-5).half()
    out = fused_layer_norm(x, w, b, 1e-5)
    if out is None or (ref.float() - out.float()).abs().max() > 4e-3:
        return False

    q = torch.randn((64, 8, 64), device=device, generator=generator).half()
    qw = torch.ones((64,), device=device)
    eps = torch.finfo(torch.float32).eps
    ref = F.rms_norm(q.float(), (64,), qw, eps).half()
    out = fused_rms_norm(q, qw, eps)
    if out is None or (ref.float() - out.float()).abs().max() > 4e-3:
        return False

    gate = torch.randn((64, 1024), device=device, generator=generator).half()
    up = torch.randn((64, 1024), device=device, generator=generator).half()
    ref = F.silu(gate) * up
    out = fused_swiglu(gate, up)
    return out is not None and (ref.float() - out.float()).abs().max() <= 4e-3


def _supports_cached_qk(row):
    from lattice._model_interleave.row import InterleavedRowInteraction

    if not isinstance(row, InterleavedRowInteraction):
        return False
    rope = row.rope
    if rope is None:
        return True
    # The fused kernel rotates contiguous pairs of half-heads on the token
    # axis. Suppressing eager RoPE requires both query and key to use it.
    return (
        not rope.interleaved and not rope.use_xpos and not rope.learned_freq
        and rope.default_seq_dim == -2 and rope.freqs.numel() == 8
        and all(
            isinstance(norm, QKRMSNorm) and norm.head_dim == 16
            for block in row.row_blocks for norm in (block.attn.q_norm, block.attn.k_norm)
        )
    )


def install(model, *, fuse_qk: bool, allow_kv_cache: bool = False) -> bool:
    """Enable the validated kernels for adaptive inference.

    The installed kernels record what they were chosen for, ``(fuse_qk, kv, device)``.
    A network object can outlive one fit: AutoGluon's ``SharedWeights`` hands the
    same module to every estimator, and a fit that reuses a registered network
    may run with another AMP decision (a small warmup fit installs ``fuse_qk=False``
    where a large one wants ``True``). An install that does not match the request
    is therefore removed and redone instead of being reported as already in place.

    ``allow_kv_cache`` also patches ``forward_with_cache``, the encoder path a
    KV-cached fit builds and serves from. Cached interleaved row blocks use
    the same RoPE dispatch as the uncached path.
    """
    encoder = model.icl_predictor.tf_icl
    blocks = getattr(encoder, "blocks", ())
    if not blocks or not hasattr(encoder, "attn_res_block_size"):
        return False
    width = blocks[0].norm1.normalized_shape[0]
    if width not in (256, 512):
        return False
    device = blocks[0].norm1.weight.device
    if fuse_qk and allow_kv_cache:
        fuse_qk = _supports_cached_qk(model.row_interactor)
    config = (bool(fuse_qk), bool(allow_kv_cache), device.type, device.index)

    if hasattr(model, "_fastpath_restore"):
        if getattr(model, "_fastpath_config", None) == config:
            return True
        uninstall(model)

    if torch.is_grad_enabled() or device.type != "cuda":
        return False
    if not _encoder_selftest(device, width):
        return False

    if fuse_qk:
        row = model.row_interactor
        if not hasattr(row, "row_blocks") or not hasattr(row, "col_blocks"):
            return False
        if not _qk_selftest(device):
            if allow_kv_cache:
                return install(model, fuse_qk=False, allow_kv_cache=True)
            return False

    with ExitStack() as changes:
        def replace(module, name, value):
            if name in vars(module):
                changes.callback(setattr, module, name, vars(module)[name])
            else:
                changes.callback(delattr, module, name)
            setattr(module, name, value)

        row_norms = set()
        if fuse_qk:
            rope = row.rope
            if rope is not None:
                row_norms = {
                    module for block in row.row_blocks for module in block.modules()
                    if isinstance(module, QKRMSNorm)
                }
                rope_forward = make_rope_norm(rope, {})
                # Keep the RoPE module registered so state_dict still includes its weights.
                replace(row, "_use_fused_rope", True)
            replace(row, "recompute", False)
            for module in row.modules():
                if isinstance(module, QKRMSNorm):
                    forward = rope_forward if module in row_norms else fused_norm
                    replace(module, "forward", MethodType(forward, module))
            if not allow_kv_cache:
                for block in row.col_blocks:
                    replace(block, "forward", block.induced_attention)
        if fuse_qk or allow_kv_cache:
            cached_logn = lru_cache(maxsize=64)(_logn)
            for module in model.modules():
                if isinstance(module, QASSMaxMLP):
                    replace(module, "_logn", cached_logn)

        replace(encoder, "forward", MethodType(fast_encoder_forward, encoder))
        if allow_kv_cache and hasattr(type(encoder), "forward_with_cache"):
            replace(encoder, "forward_with_cache",
                    MethodType(fast_encoder_forward_with_cache, encoder))
        fused_requested = os.environ.get("LATTICE_FUSED", "0").lower() not in (
            "0", "off", "false", "disabled", "",
        )
        if fuse_qk and not allow_kv_cache and fused_requested and _fused_selftest(device):
            install_fused(model, encoder, row_norms, replace)
        if HAS_FUSED_RMS_NORM:
            if allow_kv_cache:
                replace(model.row_interactor, "_shared_query_tokens", True)
            # The checkpoints use LayerNorm before attention and MLP, and
            # head-wise RMSNorm on the projected Q/K inside ICL attention.
            for block in model.modules():
                if isinstance(block, MultiheadAttentionBlock):
                    for norm in (block.norm1, block.norm2):
                        if isinstance(norm, torch.nn.LayerNorm):
                            replace(norm, "forward", MethodType(layer_norm_forward, norm))
                elif isinstance(block, SwiGLU):
                    forward = fused_swiglu_forward
                    if allow_kv_cache:
                        replace(block, "_pack_gate_up", lru_cache(maxsize=2)(pack_projection_pair))
                        packed_projections(block, torch.float16 if fuse_qk else block.w_gate.weight.dtype)
                        forward = packed_swiglu_forward
                    replace(block, "forward", MethodType(forward, block))
            if fuse_qk:
                for norm in encoder.modules():
                    if isinstance(norm, QKRMSNorm):
                        replace(norm, "forward", MethodType(fused_norm, norm))
        model._fastpath_config = config
        model._fastpath_restore = changes.pop_all()
    return True


def uninstall(model) -> None:
    """Restore the eager path and release this model's device-local caches."""
    restore = getattr(model, "_fastpath_restore", None)
    if restore is not None:
        del model._fastpath_restore
        if hasattr(model, "_fastpath_config"):
            del model._fastpath_config
        restore.close()
