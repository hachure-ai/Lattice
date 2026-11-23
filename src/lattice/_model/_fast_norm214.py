"""CUDA normalization kernels for ATen's vectorized reductions."""
import torch
import triton
import triton.language as tl

# The operator identifies a candidate reduction layout. The installer compares
# the kernels with native results on the inference device before patching a model.
HAS_FUSED_RMS_NORM = torch.version.cuda is not None and hasattr(torch.ops.aten, "_fused_rms_norm")
TORCH214 = HAS_FUSED_RMS_NORM  # Older benchmark scripts import this name.


@triton.jit
def _four_values(x, R: tl.constexpr, D: tl.constexpr):
    even, odd = tl.split(tl.reshape(x, (R, D // 2, 2)))
    a, c = tl.split(tl.reshape(even, (R, D // 4, 2)))
    b, d = tl.split(tl.reshape(odd, (R, D // 4, 2)))
    return a, b, c, d


@triton.jit
def _peer(value, offset: tl.constexpr, R: tl.constexpr, groups: tl.constexpr):
    index = tl.broadcast_to((tl.arange(0, groups) ^ offset)[None, :], (R, groups))
    return tl.gather(value, index, axis=1)


@triton.jit
def rms_scale(x, eps: tl.constexpr, R: tl.constexpr, D: tl.constexpr):
    # ATen accumulates four consecutive channels per lane, then combines
    # lane totals in descending warp-shuffle order. Here D is at most 128.
    a, b, c, d = _four_values(x, R, D)
    total = tl.fma(d, d, tl.fma(c, c, tl.fma(b, b, a * a)))
    for i in tl.static_range((D // 4).bit_length() - 1):
        total += _peer(total, (D // 8) >> i, R, D // 4)
    total = tl.sum(tl.where(tl.arange(0, D // 4)[None, :] == 0, total, 0.), 1)
    return tl.rsqrt(total / D + eps)


@triton.jit
def _online(value, mean, variance, reciprocal: tl.constexpr):
    delta = value - mean
    mean = tl.fma(delta, reciprocal, mean)
    return mean, tl.fma(delta, value - mean, variance)


@triton.jit
def _merge(mean, variance, offset: tl.constexpr, count: tl.constexpr,
           R: tl.constexpr, groups: tl.constexpr):
    other_mean = _peer(mean, offset, R, groups)
    other_variance = _peer(variance, offset, R, groups)
    delta = mean - other_mean
    mean = tl.fma(0.5, other_mean, 0.5 * mean)
    variance = tl.fma(delta * delta, count * 0.5, other_variance + variance)
    return mean, variance


@triton.jit
def _layer_norm(x, w, bias, out, rows, D: tl.constexpr, eps: tl.constexpr,
                R: tl.constexpr, HAS_BIAS: tl.constexpr):
    row = tl.program_id(0) * R + tl.arange(0, R)
    col = tl.arange(0, D)
    valid = row[:, None] < rows
    offset = row.to(tl.int64)[:, None] * D + col[None, :]
    value = tl.load(x + offset, valid, 0.).to(tl.float32)
    a, b, c, d = _four_values(value, R, D)
    mean, variance = _online(b, a, tl.full((R, D // 4), 0, tl.float32), 0.5)
    mean, variance = _online(c, mean, variance, 1.0 / 3)
    mean, variance = _online(d, mean, variance, 0.25)
    # Each lane processes four channels, each warp merges 32 lanes, then
    # warp leaders merge across warps. D is 128, 256 or 512.
    for i in tl.static_range(5):
        mean, variance = _merge(mean, variance, 16 >> i, 4 << i, R, D // 4)
    for i in tl.static_range((D // 128).bit_length() - 1):
        mean, variance = _merge(mean, variance, (D // 8) >> i, 128 << i, R, D // 4)
    first = tl.arange(0, D // 4)[None, :] == 0
    mean_row = tl.sum(tl.where(first, mean, 0.), 1)
    var_row = tl.sum(tl.where(first, variance, 0.), 1) / D
    norm = (value - mean_row[:, None]) * tl.rsqrt(var_row + eps)[:, None]
    gain = tl.load(w + col)[None, :]
    if HAS_BIAS:
        norm = tl.fma(gain, norm, tl.load(bias + col)[None, :])
    else:
        norm *= gain
    tl.store(out + offset, norm, valid)


def layer_norm_forward(self, x):
    if not x.is_cuda or torch.is_grad_enabled() or len(self.normalized_shape) != 1:
        return torch.nn.LayerNorm.forward(self, x)
    dim = self.normalized_shape[0]
    if dim not in (128, 256, 512) or not x.ndim or x.shape[-1] != dim or not x.numel():
        return torch.nn.LayerNorm.forward(self, x)
    amp_input = x.dtype in (torch.float16, torch.bfloat16) and torch.is_autocast_enabled("cuda")
    if x.dtype != torch.float32 and not amp_input:
        return torch.nn.LayerNorm.forward(self, x)
    if self.weight is None or any(
        t.device != x.device or t.dtype != torch.float32 or not t.is_contiguous()
        or t.storage_offset() % 4 for t in (self.weight, self.bias) if t is not None
    ):
        return torch.nn.LayerNorm.forward(self, x)
    x = x.contiguous()
    if x.dtype == torch.float32 and x.storage_offset() % 4:
        return torch.nn.LayerNorm.forward(self, x)
    out = torch.empty_like(x, dtype=torch.float32)
    rows = x.numel() // dim
    _layer_norm[(triton.cdiv(rows, 4),)](
        x, self.weight, self.bias, out, rows, dim, self.eps, 4,
        self.bias is not None, enable_fp_fusion=False, num_warps=4,
    )
    return out
