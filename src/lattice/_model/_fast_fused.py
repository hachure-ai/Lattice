"""Fused LayerNorm / RMSNorm / SwiGLU kernels for CUDA inference.

Normalization computes statistics and affine transforms in fp32. The SwiGLU
kernel rounds the activation to the projection dtype before multiplying.
"""

import os
from types import MethodType

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from ._fast_norm214 import HAS_FUSED_RMS_NORM


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


@triton.jit(do_not_specialize=["elements"])
def _swiglu_kernel(gate, up, out, elements, BLOCK: tl.constexpr, NATIVE_SILU: tl.constexpr,
                   PACKED_WIDTH: tl.constexpr = 0):
    offset = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    valid = offset < elements
    src = offset
    if PACKED_WIDTH:
        # [row, gate | up]: load both halves without making contiguous copies.
        src = offset + (offset // PACKED_WIDTH) * PACKED_WIDTH
    g = tl.load(gate + src, mask=valid, other=0.).to(tl.float32)
    u = tl.load(up + src + PACKED_WIDTH, mask=valid, other=0.).to(tl.float32)
    if NATIVE_SILU:
        # ATen SiLU divides by 1 + expf(-x). Multiplying by sigmoid instead
        # changes which side of a half-precision rounding boundary x lands on.
        activation = tl.div_rn(g, 1.0 + libdevice.exp(-g))
    else:
        activation = g * tl.sigmoid(g)
    rounded = activation.to(gate.dtype.element_ty).to(tl.float32)
    tl.store(out + offset, rounded * u, mask=valid)


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
    if torch.is_grad_enabled() or gate is None or up is None:
        return None
    if not gate.is_cuda or up.device != gate.device or up.dtype != gate.dtype:
        return None
    if not gate.numel() or gate.shape != up.shape or not gate.is_contiguous() or not up.is_contiguous():
        return None
    if HAS_FUSED_RMS_NORM:
        if gate.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            return None
    elif _disabled() or gate.dtype != torch.float16 or gate.ndim == 0 or not _dims(gate.shape[-1]):
        return None

    out = torch.empty_like(gate)
    block = 512
    with torch.cuda.device(gate.device):
        _swiglu_kernel[(triton.cdiv(gate.numel(), block),)](
            gate, up, out, gate.numel(), BLOCK=block, NATIVE_SILU=HAS_FUSED_RMS_NORM, num_warps=4,
        )
    return out


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


def pack_projection_pair(first, second, versions, placement, dtype):
    """Pack immutable inference weights once for the active GEMM precision."""
    return torch.cat((first, second), dim=0).to(dtype=dtype)


def packed_projections(module, dtype):
    gate, up = module.w_gate, module.w_up
    params = (gate.weight, up.weight, gate.bias, up.bias)
    if any(p is None or p.is_inference() or p.dtype == torch.float64 for p in params):
        return None

    def pair(a, b):
        # load_state_dict, parameter replacement and device/dtype moves can
        # each invalidate the packed copy during the lifetime of a shared model.
        return module._pack_gate_up(a, b, (a._version, b._version), (a.device, a.dtype), dtype)

    return pair(gate.weight, up.weight), pair(gate.bias, up.bias)


def packed_swiglu_forward(self, x):
    if self.training or torch.is_grad_enabled() or not x.is_cuda or not x.numel():
        return fused_swiglu_forward(self, x)
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return fused_swiglu_forward(self, x)
    dtype = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else self.w_gate.weight.dtype
    projections = packed_projections(self, dtype)
    if projections is None:
        return fused_swiglu_forward(self, x)
    projected = F.linear(x, *projections)
    width = self.w_gate.out_features
    out = projected.new_empty((*projected.shape[:-1], width))
    with torch.cuda.device(projected.device):
        _swiglu_kernel[(triton.cdiv(out.numel(), 256),)](
            projected, projected, out, out.numel(), BLOCK=256, NATIVE_SILU=True,
            PACKED_WIDTH=width, num_warps=4,
        )
    return self.w_down(self.dropout(out))


def fused_swiglu_forward(self, x):
    gate = self.w_gate(x)
    up = self.w_up(x)
    out = fused_swiglu(gate, up)
    if out is None:
        out = F.silu(gate) * up
    return self.w_down(self.dropout(out))


def install_fused(model, encoder, row_norms, replace) -> bool:
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
    use_swiglu = not HAS_FUSED_RMS_NORM and _enabled("swiglu")
    use_rms = _enabled("rms")
    for module in model.modules():
        if isinstance(module, torch.nn.LayerNorm) and use_ln:
            replace(module, "forward", MethodType(fused_layer_norm_forward, module))
            norm_modules.append(module)
        elif isinstance(module, SwiGLU) and use_swiglu:
            replace(module, "forward", MethodType(fused_swiglu_forward, module))
            norm_modules.append(module)
    for module in encoder.modules():
        if isinstance(module, QKRMSNorm) and module not in row_norms and use_rms:
            replace(module, "forward", MethodType(fused_rms_norm_forward, module))
            norm_modules.append(module)
    return bool(norm_modules)
