"""Preserve NumPy's column summation order for missing-value imputation.

An almost-zero mean can cross an ECDF tie if its summation order changes.
The CUDA kernel accumulates rows in the same order as NumPy's strided sum.
"""

from __future__ import annotations

import numpy as np
import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None


if triton is not None:

    @triton.jit(do_not_specialize=["rows", "width"])
    def _column_sum(X, SUM, COUNT, rows, width, BLOCK: tl.constexpr):
        columns = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        total = tl.full((BLOCK,), 0.0, tl.float64)
        count = tl.full((BLOCK,), 0, tl.int32)
        for row in range(rows):
            values = tl.load(X + row * width + columns, columns < width, other=0.0)
            finite = tl.abs(values) < float("inf")
            total += tl.where(finite, values, 0.0)
            count += finite.to(tl.int32)
        tl.store(SUM + columns, total, columns < width)
        tl.store(COUNT + columns, count, columns < width)


def support_mean(values, *, use_cpu=False):
    """Match the previous NumPy imputation even for cancellation near zero."""
    if not use_cpu and values.is_cuda and values.shape[1] > 1 and triton is not None:
        values = values.contiguous()
        width = values.shape[1]
        total = torch.empty(width, device=values.device, dtype=torch.float64)
        count = torch.empty(width, device=values.device, dtype=torch.int32)
        _column_sum[(triton.cdiv(width, 128),)](
            values,
            total,
            count,
            len(values),
            width,
            128,
            enable_fp_fusion=False,
        )
        return total / count.clamp_min(1)
    # NumPy uses a different (pairwise) reduction for a single contiguous
    # column. Keep that corner case, and installations without Triton, exact.
    array = (values.contiguous() if use_cpu else values).cpu().numpy()
    finite = np.isfinite(array)
    mean = np.where(finite, array, 0).sum(0) / np.maximum(finite.sum(0), 1)
    return torch.as_tensor(mean, device=values.device)
