"""Install the QK and ICL kernels on supported inference models."""

import os
from functools import lru_cache
from types import MethodType

import torch
import torch.nn.functional as F
import triton

from . import ssmax
from ._fast_attnres import fast_encoder_forward
from ._fast_cell import fused_norm, make_rope_norm
from ._fast_fused import fused_layer_norm, fused_rms_norm, fused_swiglu, install_fused
from .norms import QKRMSNorm


@lru_cache(maxsize=None)
def _qk_selftest(device):
    generator = torch.Generator(device=device).manual_seed(0)
    norm = QKRMSNorm(16).to(device)
    x = torch.randn((2, 3, 8, 16), device=device, generator=generator)
    x = (x * 0.5).to(torch.float16)
    return torch.equal(norm(x), fused_norm(norm, x))


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


def install(model, *, fuse_qk: bool) -> bool:
    """Enable the validated kernels for adaptive, uncached inference."""
    if torch.is_grad_enabled():
        return False
    if torch.__version__.split("+")[0] != "2.7.1" or triton.__version__ != "3.3.1":
        return False

    encoder = model.icl_predictor.tf_icl
    blocks = getattr(encoder, "blocks", ())
    if not blocks or not hasattr(encoder, "attn_res_block_size") or blocks[0].norm1.normalized_shape[0] not in (256, 512):
        return False

    row_norms = set()
    if fuse_qk:
        row = model.row_interactor
        if not hasattr(row, "row_blocks") or not hasattr(row, "col_blocks"):
            return False
        device = blocks[0].norm1.weight.device
        if device.type != "cuda" or not _qk_selftest(device):
            return False

        rope = row.rope
        row_norms = {
            module for block in row.row_blocks for module in block.modules()
            if isinstance(module, QKRMSNorm)
        } if rope is not None else set()
        rope_forward = make_rope_norm(rope, {}) if rope is not None else None
        row.rope = None
        row.recompute = False
        for module in row.modules():
            if isinstance(module, QKRMSNorm):
                forward = rope_forward if module in row_norms else fused_norm
                module.forward = MethodType(forward, module)
        for block in row.col_blocks:
            block.forward = block.induced_attention
        if not hasattr(ssmax._logn, "cache_info"):
            ssmax._logn = lru_cache(maxsize=64)(ssmax._logn)

    encoder.forward = MethodType(fast_encoder_forward, encoder)
    fused_requested = os.environ.get("LATTICE_FUSED", "0").lower() not in (
        "0", "off", "false", "disabled", "",
    )
    if fuse_qk and fused_requested and _fused_selftest(blocks[0].norm1.weight.device):
        install_fused(model, encoder, row_norms)
    return True
