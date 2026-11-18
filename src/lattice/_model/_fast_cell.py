"""Fused RoPE and query/key RMS normalization for CUDA inference."""

import torch
import triton
import triton.language as tl

from .norms import QKRMSNorm


@triton.jit
def rms_cast(x, w, out, rows, d:tl.constexpr, n1:tl.constexpr, n2:tl.constexpr, n3:tl.constexpr,
             s0:tl.constexpr,s1:tl.constexpr,s2:tl.constexpr,s3:tl.constexpr,s4:tl.constexpr,
             o0:tl.constexpr,o1:tl.constexpr,o2:tl.constexpr,o3:tl.constexpr,o4:tl.constexpr,
             eps:tl.constexpr):
    r = tl.program_id(0)*16 + tl.arange(0,16)
    i3 = r % n3
    i2 = r // n3 % n2
    i1 = r // (n3*n2) % n1
    i0 = r // (n3*n2*n1)
    c = tl.arange(0,d)
    a = tl.load(x+(i0*s0+i1*s1+i2*s2+i3*s3)[:,None]+c[None,:]*s4,r[:,None]<rows,0).to(tl.float32)
    square = a*a
    for shift in tl.static_range(0,4):
        index = tl.broadcast_to((c ^ (1 << shift))[None,:],(16,d))
        square = square + tl.gather(square,index,axis=1)
    total = tl.sum(tl.where(c[None,:] == 0,square,0),1)
    inv = tl.rsqrt(total/d+eps)
    gain = tl.load(w+c).to(tl.float32)
    val = (a*inv[:,None])*gain[None,:]
    tl.store(out+(i0*o0+i1*o1+i2*o2+i3*o3)[:,None]+c[None,:]*o4,val,r[:,None]<rows)


def fused_norm(self, x):
    if not x.is_cuda or x.ndim > 5 or x.shape[-1] != 16 or x.dtype != torch.float16 or not x.numel():
        return QKRMSNorm.forward(self,x)
    out = torch.empty_like(x)
    shape = (1,)*(5-x.ndim)+x.shape
    xs = (0,)*(5-x.ndim)+x.stride()
    os = (0,)*(5-x.ndim)+out.stride()
    rows = x.numel()//x.shape[-1]
    rms_cast[(triton.cdiv(rows,16),)](x,self.weight,out,rows,x.shape[-1],*shape[1:4],*xs,*os,
             self.eps or torch.finfo(torch.float32).eps,enable_fp_fusion=False,num_warps=4)
    return out


@triton.jit
def rope_norm(x,w,cos,sin,out,rows,n1:tl.constexpr,n2:tl.constexpr,n3:tl.constexpr,
              s0:tl.constexpr,s1:tl.constexpr,s2:tl.constexpr,s3:tl.constexpr,s4:tl.constexpr,
              o0:tl.constexpr,o1:tl.constexpr,o2:tl.constexpr,o3:tl.constexpr,o4:tl.constexpr,
              eps:tl.constexpr,DT:tl.constexpr,ROUND_MID:tl.constexpr):
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
        a = (left+right).to(DT).to(tl.float32)
    else:
        a = left+right
    square = a*a
    for shift in tl.static_range(0,4):
        index = tl.broadcast_to((c^(1<<shift))[None,:],(16,16))
        square += tl.gather(square,index,axis=1)
    total = tl.sum(tl.where(c[None,:]==0,square,0),1)
    inv = tl.rsqrt(total/16+eps)
    val = (a*inv[:,None])*tl.load(w+c)[None,:]
    out_pos = (i0*o0+i1*o1+i2*o2+i3*o3)[:,None]+c[None,:]*o4
    tl.store(out+out_pos,val,valid)


def make_rope_norm(rope,tables):
    def forward(self,x):
        if not x.is_cuda or x.ndim > 5 or x.dtype != torch.float16 or x.shape[-1] != 16 or not x.numel():
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
        rope_norm[(triton.cdiv(rows,16),)](x,self.weight,cos,sin,out,rows,*shape[1:4],*xs,*os,
                 self.eps or torch.finfo(torch.float32).eps,DT=tl.float16,
                 ROUND_MID=True,enable_fp_fusion=False,num_warps=4)
        return out
    return forward
