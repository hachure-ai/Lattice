"""Fused RoPE and query/key RMS normalization for CUDA inference."""

import torch
import triton
import triton.language as tl

from ._fast_norm214 import HAS_FUSED_RMS_NORM, rms_scale
from .norms import QKRMSNorm


@triton.jit
def _qk_rstd(value, eps: tl.constexpr, width: tl.constexpr, modern: tl.constexpr):
    if modern:
        return rms_scale(value, eps, 16, width)
    else:
        columns = tl.arange(0, width)
        square = value * value
        for shift in tl.static_range(4):
            peer = tl.broadcast_to((columns ^ (1 << shift))[None, :], (16, width))
            square += tl.gather(square, peer, axis=1)
        total = tl.sum(tl.where(columns[None, :] == 0, square, 0.), axis=1)
        return tl.rsqrt(total / width + eps)


# Keep table extents and outer strides dynamic. Fit and predict use different
# row counts but can share the same compiled head-normalization kernels.
@triton.jit
def rms_cast(x, w, out, rows, d:tl.constexpr, n1, n2:tl.constexpr, n3,
             s0,s1,s2,s3,s4:tl.constexpr,
             o0,o1,o2,o3,o4:tl.constexpr,
             eps:tl.constexpr, MODERN:tl.constexpr=False):
    r = tl.program_id(0)*16 + tl.arange(0,16)
    i3 = r % n3
    i2 = r // n3 % n2
    i1 = r // (n3*n2) % n1
    i0 = r // (n3*n2*n1)
    c = tl.arange(0,d)
    a = tl.load(x+(i0*s0+i1*s1+i2*s2+i3*s3)[:,None]+c[None,:]*s4,r[:,None]<rows,0).to(tl.float32)
    inv = _qk_rstd(a, eps, d, MODERN)
    gain = tl.load(w+c).to(tl.float32)
    val = (a*inv[:,None])*gain[None,:]
    tl.store(out+(i0*o0+i1*o1+i2*o2+i3*o3)[:,None]+c[None,:]*o4,val,r[:,None]<rows)


def _can_index_qk(x):
    # The strided kernels address elements using 32-bit offsets.
    return 0 < x.numel() < 2**31 and x.untyped_storage().nbytes() // x.element_size() < 2**31


def fused_norm(self, x):
    dims = (16, 32, 64, 128) if HAS_FUSED_RMS_NORM else (16,)
    if (not x.is_cuda or torch.is_grad_enabled() or not 1 <= x.ndim <= 5
            or x.shape[-1] not in dims or x.shape[-1] != self.head_dim
            or x.dtype != torch.float16 or not _can_index_qk(x) or not self.weight.is_contiguous()
            or self.weight.device != x.device):
        return QKRMSNorm.forward(self,x)
    layout = torch.contiguous_format if HAS_FUSED_RMS_NORM else torch.preserve_format
    out = torch.empty_like(x, memory_format=layout)
    shape = (1,)*(5-x.ndim)+x.shape
    xs = (0,)*(5-x.ndim)+x.stride()
    os = (0,)*(5-x.ndim)+out.stride()
    rows = x.numel()//x.shape[-1]
    eps = torch.finfo(torch.float32).eps if self.eps is None else self.eps
    rms_cast[(triton.cdiv(rows, 16),)](
        x, self.weight, out, rows, x.shape[-1], *shape[1:4], *xs, *os,
        eps, MODERN=HAS_FUSED_RMS_NORM, enable_fp_fusion=False, num_warps=4,
    )
    return out


@triton.jit
def rope_norm(x,w,cos,sin,out,rows,n1,n2:tl.constexpr,n3,
              s0,s1,s2,s3,s4:tl.constexpr,
              o0,o1,o2,o3,o4:tl.constexpr,
              eps:tl.constexpr,DT:tl.constexpr,ROUND_MID:tl.constexpr,MODERN:tl.constexpr=False):
    r = tl.program_id(0)*16+tl.arange(0,16)
    i3 = r % n3
    i2 = r // n3 % n2
    i1 = r // (n3*n2) % n1
    i0 = r // (n3*n2*n1)
    c = tl.arange(0,16)
    pos = (i0*s0+i1*s1+i2*s2+i3*s3)[:,None]
    valid = r[:,None]<rows
    x0 = tl.load(x+pos+c[None,:]*s4,valid,0).to(tl.float32)
    peer = tl.load(x+pos+(c[None,:]^8)*s4,valid,0).to(tl.float32)
    angle = i3[:,None]*8+(c[None,:]%8)
    cval = tl.load(cos+angle)
    sval = tl.load(sin+angle)
    left = x0*cval.to(tl.float32)
    right = tl.where(c[None,:]<8,-peer,peer)*sval.to(tl.float32)
    if ROUND_MID:
        left = left.to(DT).to(tl.float32)
        right = right.to(DT).to(tl.float32)
    a = (left + right).to(DT).to(tl.float32)
    inv = _qk_rstd(a, eps, 16, MODERN)
    val = (a*inv[:,None])*tl.load(w+c)[None,:]
    out_pos = (i0*o0+i1*o1+i2*o2+i3*o3)[:,None]+c[None,:]*o4
    tl.store(out+out_pos,val,valid)


def make_rope_norm(rope,tables):
    if (rope.interleaved or rope.use_xpos or rope.learned_freq
            or rope.default_seq_dim != -2 or rope.freqs.numel() != 8):
        def eager(self, x):
            return QKRMSNorm.forward(self, rope.rotate_queries_or_keys(x))
        return eager

    def forward(self,x):
        if (not x.is_cuda or torch.is_grad_enabled() or not 2 <= x.ndim <= 5
                or x.dtype != torch.float16 or x.shape[-1] != 16 or not _can_index_qk(x)
                or self.head_dim != 16 or not self.weight.is_contiguous() or self.weight.device != x.device):
            return QKRMSNorm.forward(self, rope.rotate_queries_or_keys(x))
        # The fused rotation rounds both products in fp16. Eager RoPE with
        # fp32 frequencies outside autocast instead rounds after the addition.
        autocast_dtype = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled() else None
        if autocast_dtype not in (None, x.dtype) or (
            rope.freqs.dtype != x.dtype and (autocast_dtype is None or rope.freqs.dtype != torch.float32)
        ):
            return QKRMSNorm.forward(self, rope.rotate_queries_or_keys(x))
        key = (x.device,x.dtype,x.shape[-2])
        if key not in tables:
            seq = rope.get_seq_pos(key[-1],device=x.device,dtype=x.dtype)
            freq = torch.einsum("..., f -> ... f",seq.type(rope.freqs.dtype),rope.freqs)
            tables[key] = freq.cos().contiguous(),freq.sin().contiguous()
        cos,sin = tables[key]
        out = torch.empty_like(x,memory_format=torch.contiguous_format)
        shape = (1,)*(5-x.ndim)+x.shape
        xs = (0,)*(5-x.ndim)+x.stride()
        os = (0,)*(5-x.ndim)+out.stride()
        rows = x.numel()//16
        eps = torch.finfo(torch.float32).eps if self.eps is None else self.eps
        rope_norm[(triton.cdiv(rows, 16),)](
            x, self.weight, cos, sin, out, rows, *shape[1:4], *xs, *os, eps,
            DT=tl.float16, ROUND_MID=not HAS_FUSED_RMS_NORM or cos.dtype == x.dtype,
            MODERN=HAS_FUSED_RMS_NORM, enable_fp_fusion=False, num_warps=4,
        )
        return out
    return forward
