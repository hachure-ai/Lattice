"""Fused LayerNorm / RMSNorm / SwiGLU kernels for half-precision CUDA inference.

The kernels mirror the eager reference (fp32 statistics, affine in fp32, output
cast back to fp16) but avoid the autocast round trips that materialise fp32
copies of every normalized tensor.
"""

import os
from types import MethodType

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


def _disabled() -> bool:
    return os.environ.get("LATTICE_FUSED", "0").lower() in ("0", "off", "false", "disabled", "")


def _enabled(kind: str) -> bool:
    value = os.environ.get("LATTICE_FUSED", "0").lower()
    if value in ("0", "off", "false", "disabled", ""):
        return False
    if value in ("1", "on", "true", "yes", "all"):
        return True
    return kind in {item.strip() for item in value.split(",")}


@triton.jit
def _layer_norm_kernel(x, w, b, out, rows, D: tl.constexpr, eps: tl.constexpr,
                       BR: tl.constexpr):
    row = tl.program_id(0) * BR + tl.arange(0, BR)
    cols = tl.arange(0, D)
    valid = row[:, None] < rows
    offset = row[:, None].to(tl.int64) * D + cols[None, :]
    value = tl.load(x + offset, mask=valid, other=0.0).to(tl.float32)
    mean = tl.sum(value, 1) / D
    centered = value - mean[:, None]
    var = tl.sum(centered * centered, 1) / D
    scale = tl.rsqrt(var + eps)[:, None]
    value = centered * scale
    value = value * tl.load(w + cols).to(tl.float32) + tl.load(b + cols).to(tl.float32)
    tl.store(out + offset, value, mask=valid)


@triton.jit
def _rms_norm_kernel(x, w, out, rows, D: tl.constexpr, eps: tl.constexpr,
                     BR: tl.constexpr):
    row = tl.program_id(0) * BR + tl.arange(0, BR)
    cols = tl.arange(0, D)
    valid = row[:, None] < rows
    offset = row[:, None].to(tl.int64) * D + cols[None, :]
    value = tl.load(x + offset, mask=valid, other=0.0).to(tl.float32)
    var = tl.sum(value * value, 1) / D
    value = value * tl.rsqrt(var + eps)[:, None] * tl.load(w + cols).to(tl.float32)
    tl.store(out + offset, value, mask=valid)


@triton.jit
def _swiglu_kernel(gate, up, out, rows, H: tl.constexpr, BH: tl.constexpr,
                   BR: tl.constexpr):
    row = tl.program_id(0) * BR + tl.arange(0, BR)
    cols = tl.arange(0, BH)
    valid = (row[:, None] < rows) & (cols[None, :] < H)
    offset = row[:, None].to(tl.int64) * H + cols[None, :]
    g = tl.load(gate + offset, mask=valid, other=0.0).to(tl.float32)
    u = tl.load(up + offset, mask=valid, other=0.0).to(tl.float32)
    # Match eager fp16: round silu(gate) to fp16, then multiply by up.
    s = (g * tl.sigmoid(g)).to(tl.float16).to(tl.float32)
    tl.store(out + offset, s * u, mask=valid)


def _dims(dim: int) -> bool:
    return 16 <= dim <= 4096 and dim & (dim - 1) == 0


def _warps(dim: int) -> int:
    return 4 if dim <= 256 else 8


def fused_layer_norm(x, weight, bias, eps: float):
    """Return a fused LayerNorm or ``None`` when the kernel cannot run."""
    if _disabled() or weight is None or bias is None:
        return None
    if not x.is_cuda or x.dtype != torch.float16 or not x.is_contiguous():
        return None
    dim = x.shape[-1]
    if not _dims(dim) or x.numel() == 0:
        return None
    if weight.device != x.device or bias.device != x.device:
        return None
    rows = x.numel() // dim
    flat = x.view(rows, dim)
    out = torch.empty_like(flat)
    _layer_norm_kernel[(triton.cdiv(rows, 8),)](
        flat, weight, bias, out, rows, dim, float(eps), 8, num_warps=_warps(dim)
    )
    return out.view_as(x)


def fused_rms_norm(x, weight, eps: float):
    """Return a fused RMSNorm or ``None`` when the kernel cannot run."""
    if _disabled() or weight is None:
        return None
    if not x.is_cuda or x.dtype != torch.float16 or x.numel() == 0:
        return None
    dim = x.shape[-1]
    if not _dims(dim) or weight.device != x.device:
        return None
    contiguous = x if x.is_contiguous() else x.contiguous()
    rows = contiguous.numel() // dim
    flat = contiguous.view(rows, dim)
    out = torch.empty_like(flat)
    _rms_norm_kernel[(triton.cdiv(rows, 8),)](
        flat, weight, out, rows, dim, float(eps), 8, num_warps=_warps(dim)
    )
    return out.view_as(x)


def fused_swiglu(gate, up):
    """Return ``silu(gate) * up`` fused, or ``None`` when unsupported."""
    if _disabled() or gate is None or up is None:
        return None
    if not gate.is_cuda or gate.dtype != torch.float16 or up.dtype != torch.float16:
        return None
    if gate.shape != up.shape or not gate.is_contiguous() or not up.is_contiguous():
        return None
    hidden = gate.shape[-1]
    if not _dims(hidden) or gate.numel() == 0:
        return None
    rows = gate.numel() // hidden
    flat_gate = gate.view(rows, hidden)
    flat_up = up.view(rows, hidden)
    out = torch.empty_like(flat_gate)
    _swiglu_kernel[(triton.cdiv(rows, 8),)](
        flat_gate, flat_up, out, rows, hidden, hidden, 8, num_warps=_warps(hidden)
    )
    return out.view_as(gate)


def fused_layer_norm_forward(self, x):
    out = fused_layer_norm(x, self.weight, self.bias, self.eps)
    if out is not None:
        return out
    return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)


def fused_rms_norm_forward(self, x):
    eps = torch.finfo(torch.float32).eps if self.eps is None else self.eps
    out = fused_rms_norm(x, self.weight, eps)
    if out is not None:
        return out
    return F.rms_norm(
        x.float(),
        (self.head_dim,),
        self.weight.float(),
        eps,
    ).to(x.dtype)


def fused_swiglu_forward(self, x):
    gate = self.w_gate(x)
    up = self.w_up(x)
    out = fused_swiglu(gate, up)
    if out is None:
        out = F.silu(gate) * up
    return self.w_down(self.dropout(out))


def install_fused(model, encoder, row_norms) -> bool:
    """Patch supported modules with the fused kernels.

    ``row_norms`` are the row QK norms already handled by the RoPE/QK fastpath;
    they are left untouched so the bit-exact RoPE kernel keeps its path.
    """
    from .layers import SwiGLU
    from .norms import QKRMSNorm

    if _disabled():
        return False
    device = next(model.parameters()).device
    if device.type != "cuda":
        return False

    norm_modules = []
    use_ln = _enabled("ln")
    use_swiglu = _enabled("swiglu")
    use_rms = _enabled("rms")
    for module in model.modules():
        if isinstance(module, torch.nn.LayerNorm) and use_ln:
            module.forward = MethodType(fused_layer_norm_forward, module)
            norm_modules.append(module)
        elif isinstance(module, SwiGLU) and use_swiglu:
            module.forward = MethodType(fused_swiglu_forward, module)
            norm_modules.append(module)
    for module in encoder.modules():
        if isinstance(module, QKRMSNorm) and module not in row_norms and use_rms:
            module.forward = MethodType(fused_rms_norm_forward, module)
            norm_modules.append(module)
    return bool(norm_modules)
