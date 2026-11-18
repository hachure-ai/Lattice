# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from structured-data-models numerical processors (see NOTICE).
from __future__ import annotations

import math

import torch
from torch import Tensor

_YEOJOHNSON_OPTIMIZATION_STEPS = 44


def _yeojohnson_transform(
    inp: Tensor,
    lambdas: Tensor,
    *,
    magnitude_log: Tensor | None = None,
    positive: Tensor | None = None,
    exponents: Tensor | None = None,
    out: Tensor | None = None,
) -> Tensor:
    eps = torch.finfo(inp.dtype).eps
    two_minus_lambda = 2 - lambdas
    if magnitude_log is None:
        magnitude_log = inp.abs().log1p_()
    if positive is None:
        positive = inp >= 0
    exponents = torch.where(positive, lambdas, two_minus_lambda, out=exponents)
    out = torch.mul(exponents, magnitude_log, out=out)
    out.expm1_().div_(exponents)
    zero_exponent = torch.where(
        positive, lambdas.abs() < eps, two_minus_lambda.abs() < eps
    )
    torch.where(zero_exponent, magnitude_log, out, out=out)
    return out.copysign_(inp)


def _yeojohnson_bounds(inp: Tensor) -> tuple[Tensor, Tensor]:
    missing = inp.isnan()
    max_abs = inp.abs().nan_to_num_(nan=0.0).amax(dim=-2, keepdim=True)
    log1p_max_x = (20 * max_abs).log1p()
    log1p_max_x = torch.where(
        max_abs == 0,
        torch.ones_like(log1p_max_x),
        log1p_max_x,
    )
    finfo = torch.finfo(inp.dtype)
    log_eps = math.log(finfo.eps)
    log_tiny_float = (math.log(finfo.tiny) - log_eps) / 2
    log_max_float = (math.log(finfo.max) + log_eps) / 2

    lower_bound = log_tiny_float / log1p_max_x
    upper_bound = log_max_float / log1p_max_x
    positive_lower = lower_bound
    positive_upper = upper_bound

    negative = inp < 0
    all_negative = (negative | missing).all(dim=-2, keepdim=True)
    any_negative = negative.any(dim=-2, keepdim=True)

    mixed_lower = torch.maximum(2 - positive_upper, positive_lower)
    mixed_upper = torch.minimum(2 - mixed_lower, positive_upper)
    lower_bound = torch.where(any_negative, mixed_lower, positive_lower)
    upper_bound = torch.where(any_negative, mixed_upper, positive_upper)

    negative_lower = 2 - positive_upper
    negative_upper = 2 - positive_lower
    lower_bound = torch.where(all_negative, negative_lower, lower_bound)
    upper_bound = torch.where(all_negative, negative_upper, upper_bound)

    zero = max_abs == 0
    return (
        torch.where(zero, torch.ones_like(lower_bound), lower_bound),
        torch.where(zero, torch.ones_like(upper_bound), upper_bound),
    )


def _yeojohnson_log_likelihood(
    inp: Tensor,
    lambdas: Tensor,
    magnitude_log: Tensor,
    positive: Tensor,
    log_jacobian: Tensor,
    count: Tensor,
    exponents: Tensor,
    transformed: Tensor,
) -> Tensor:
    transformed = _yeojohnson_transform(
        inp=inp,
        lambdas=lambdas,
        magnitude_log=magnitude_log,
        positive=positive,
        exponents=exponents,
        out=transformed,
    )
    mean = transformed.nansum(dim=-2, keepdim=True).div_(count)
    variance = transformed.sub_(mean).square_().nansum(dim=-2, keepdim=True)
    variance /= count
    tiny = torch.finfo(inp.dtype).tiny
    valid = variance.isfinite() & (variance >= tiny)
    loglike = variance.log_().mul_(-count / 2)
    loglike.add_((lambdas - 1) * log_jacobian)
    return loglike.masked_fill_(~valid, -math.inf)


def _optimize_lambdas(
    inp: Tensor,
    constant_features: Tensor,
    *,
    count: Tensor,
) -> Tensor:
    left, right = _yeojohnson_bounds(inp)
    left = left.masked_fill(constant_features, 1.0)
    right = right.masked_fill(constant_features, 1.0)

    # Reuse full-table workspaces throughout the golden-section search.
    magnitude_log = inp.abs().log1p_()
    positive = inp >= 0
    log_jacobian = magnitude_log.copysign(inp).nansum(dim=-2, keepdim=True)
    exponents = torch.empty_like(inp)
    transformed = torch.empty_like(inp)

    invphi = (math.sqrt(5) - 1) / 2
    span = (right - left).mul_(invphi)
    c = right - span
    d = left + span
    fc = _yeojohnson_log_likelihood(
        inp=inp,
        lambdas=c,
        magnitude_log=magnitude_log,
        positive=positive,
        log_jacobian=log_jacobian,
        count=count,
        exponents=exponents,
        transformed=transformed,
    )
    fd = _yeojohnson_log_likelihood(
        inp=inp,
        lambdas=d,
        magnitude_log=magnitude_log,
        positive=positive,
        log_jacobian=log_jacobian,
        count=count,
        exponents=exponents,
        transformed=transformed,
    )
    c_next = torch.empty_like(c)
    d_next = torch.empty_like(d)
    new_point = torch.empty_like(c)
    choose_right = torch.empty_like(c, dtype=torch.bool)

    for _ in range(_YEOJOHNSON_OPTIMIZATION_STEPS):
        choose_right = torch.lt(fc, fd, out=choose_right)
        # Keep the search fully vectorized: each feature independently
        # chooses its next interval without per-column Python branching.
        left = torch.where(choose_right, c, left, out=left)
        right = torch.where(choose_right, right, d, out=right)
        span = torch.sub(right, left, out=span).mul_(invphi)
        c_next = torch.sub(right, span, out=c_next)
        c_next = torch.where(choose_right, d, c_next, out=c_next)
        d_next = torch.add(left, span, out=d_next)
        d_next = torch.where(choose_right, d_next, c, out=d_next)
        new_point = torch.where(choose_right, d_next, c_next, out=new_point)
        new_score = _yeojohnson_log_likelihood(
            inp=inp,
            lambdas=new_point,
            magnitude_log=magnitude_log,
            positive=positive,
            log_jacobian=log_jacobian,
            count=count,
            exponents=exponents,
            transformed=transformed,
        )
        fc = torch.where(choose_right, new_score, fc, out=fc)
        fd = torch.where(choose_right, fd, new_score, out=fd)
        fc, fd = fd, fc
        c, c_next = c_next, c
        d, d_next = d_next, d

    lambdas = (left + right).div_(2)
    return lambdas.masked_fill_(constant_features, 1.0)


def _ndtri(x: Tensor) -> Tensor:
    # MPS does not implement 'torch.special.ndtri', so use 'erfinv' instead.
    if x.device.type == "mps":
        return x.mul(2.0).sub_(1.0).erfinv_().mul_(math.sqrt(2.0))
    return torch.special.ndtri(x)


def _isfinite(x: Tensor) -> Tensor:
    # Equal to 'x.isfinite()', which allocates 'x.abs()' on the way.
    return x.gt(-math.inf).logical_and_(x.lt(math.inf))


def _constant_feature_mask(
    var: Tensor,
    mean: Tensor,
    num_samples: int | Tensor,
) -> Tensor:
    eps = torch.finfo(var.dtype).eps
    upper_bound = num_samples * eps * var + (num_samples * mean * eps) ** 2
    return var <= upper_bound


def _batched_interp(
    values: Tensor,
    boundaries: Tensor,
    references: Tensor,
) -> Tensor:
    # Batched 1D linear interpolation over independent feature grids:
    # values ``[F, N]`` with boundaries/references as ``[Q]`` or ``[F, Q]``.
    n = boundaries.shape[-1]
    if n == 1:
        if references.dim() == 1:
            return references[0].expand_as(values)
        return references[:, :1].expand_as(values)

    idx = torch.searchsorted(boundaries, values, right=True).clamp(1, n - 1)

    if boundaries.dim() == 1:
        x0 = boundaries[idx - 1]
        x1 = boundaries[idx]
        lower_boundary = boundaries[0]
        upper_boundary = boundaries[-1]
    else:
        x0 = boundaries.gather(1, idx - 1)
        x1 = boundaries.gather(1, idx)
        lower_boundary = boundaries[:, :1]
        upper_boundary = boundaries[:, -1:]

    if references.dim() == 1:
        y0 = references[idx - 1]
        y1 = references[idx]
        lower = references[0]
        upper = references[-1]
    else:
        y0 = references.gather(1, idx - 1)
        y1 = references.gather(1, idx)
        lower = references[:, :1]
        upper = references[:, -1:]

    denom = x1 - x0
    weight = torch.where(denom != 0, (values - x0) / denom, 0.0)
    result = torch.lerp(y0, y1, weight)

    result = torch.where(values <= lower_boundary, lower, result)
    return torch.where(values >= upper_boundary, upper, result)


def fit_rank(numerical, max_knots=8192):
    dtype = torch.promote_types(numerical.dtype, torch.float32)
    knots = []
    for chunk in numerical.split(32, dim=-1):
        num_rows = chunk.size(-2)
        max_knots = num_rows if max_knots is None else max_knots
        columns = chunk.to(dtype).movedim(-1, -2)  # [..., C, N]
        finite = _isfinite(columns)
        count = finite.sum(dim=-1, keepdim=True)  # [..., C, 1]
        values = columns.masked_fill(~finite, torch.inf)
        values = values.sort(dim=-1).values.contiguous()
        left = torch.searchsorted(values, values, right=False)
        right = torch.searchsorted(values, values, right=True)
        probabilities = (left + right).to(values.dtype) / (2 * count.clamp_min(1))
        # Rounding to one would give an infinite normal quantile.
        probabilities.clamp_(max=1 - torch.finfo(dtype).eps)
        last = (count - 1).clamp_min(0)
        rows = torch.arange(num_rows, device=values.device)

        if num_rows <= max_knots:
            positions = rows.minimum(last)
        else:
            # Index among distinct column values; a value starts a new
            # one at its first occurrence.
            distinct = (left == rows).cumsum(dim=-1).sub_(1)
            num_distinct = distinct.gather(-1, last) + 1
            # Knots at the first occurrence of each distinct value, padded
            # with the largest one.
            steps = torch.arange(max_knots, device=values.device)
            positions = torch.searchsorted(
                distinct,
                steps.minimum(num_distinct - 1),
            )
            # With more distinct values than knots, select rows whose
            # mid-ranks are closest to normal quantiles spaced evenly
            # between the extremes. Mid-ranks never decrease along rows.
            lower = _ndtri(probabilities[..., :1])
            upper = _ndtri(probabilities.gather(-1, last))
            quantiles = torch.special.ndtr(
                lower.lerp(
                    upper,
                    steps.to(probabilities.dtype) / (max_knots - 1),
                )
            )
            above = torch.searchsorted(probabilities, quantiles).minimum(last)
            below = (above - 1).clamp_(min=0)
            closer = (quantiles - probabilities.gather(-1, below)) < (
                probabilities.gather(-1, above) - quantiles
            )
            spaced = torch.where(closer, below, above)
            spaced[..., :1] = 0
            spaced[..., -1:] = last
            positions = torch.where(
                num_distinct <= max_knots,
                positions,
                spaced,
            )

        missing = count == 0
        knots.append(
            (
                values.gather(-1, positions).masked_fill_(missing, torch.nan),
                probabilities.gather(-1, positions).masked_fill_(
                    missing,
                    torch.nan,
                ),
            )
        )

    values, probabilities = zip(*knots)
    return torch.cat(values, dim=-2), torch.cat(probabilities, dim=-2)
