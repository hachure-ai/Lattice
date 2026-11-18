# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Support-fitted numeric recipe, implemented with device tensors.

Transform kernels adapted from NVIDIA structured-data-models (Apache-2.0;
see NOTICE in this directory)."""

from __future__ import annotations

import numpy as np
import torch

from ._kernels import (
    _batched_interp,
    _constant_feature_mask,
    _ndtri,
    _optimize_lambdas,
    _yeojohnson_transform,
    fit_rank,
)


def moments(x):
    finite = torch.isfinite(x)
    count = finite.sum(0, keepdim=True).clamp_min(1)
    x = x.masked_fill(~finite, torch.nan)
    mean = x.nansum(0, keepdim=True) / count
    var = (x - mean).square().nansum(0, keepdim=True) / count
    return mean, var, count


class LatticeNumericView:
    """Float64 standardization → one distribution transform → log clipping.

    Missing values remain missing until the ensemble fills them from the
    transformed support means. Neither fitting nor column selection sees query.
    """

    def __init__(
        self,
        method,
        device="cpu",
        outlier_threshold=4.0,
        *,
        tensor_output=False,
        kdi_alpha=0.0,
        rank_uniform=False,
        max_knots=0,
    ):
        self.method = method
        self.device_ = torch.device(device)
        self.outlier_threshold = outlier_threshold
        self.tensor_output = tensor_output
        self.kdi_alpha = kdi_alpha
        self.rank_uniform = rank_uniform
        self.max_knots = max_knots

    def to(self, device):
        self.device_ = torch.device(device)
        for name, value in vars(self).copy().items():
            if isinstance(value, torch.Tensor):
                setattr(self, name, value.to(self.device_))
        return self

    def __getstate__(self):
        state = vars(self).copy()
        for name, value in state.items():
            if isinstance(value, torch.Tensor):
                state[name] = value.cpu()
        state["device_"] = torch.device("cpu")
        return state

    def _tensor(self, X):
        if isinstance(X, torch.Tensor):
            return X.to(device=self.device_, dtype=torch.float64)
        return torch.as_tensor(
            np.ascontiguousarray(X), dtype=torch.float64, device=self.device_
        )

    @torch.no_grad()
    def fit(self, X):
        x = self._tensor(X)
        keep = torch.ones(x.shape[1], dtype=torch.bool, device=x.device)
        if len(x) > 1:
            keep = (x != x[:1]).any(0) & ~x.isnan().all(0)
        self.keep_ = keep.cpu().numpy()
        x = x[:, keep]
        self.mean_, var, _ = moments(x)
        self.scale_ = var.sqrt() + 1e-6
        z = ((x - self.mean_) / self.scale_).clamp(-100, 100)
        if z.shape[1]:
            if self.method == "power":
                mean, var, count = moments(z)
                constant = _constant_feature_mask(var, mean, count)
                self.lambdas_ = _optimize_lambdas(z, constant, count=count)
                transformed = _yeojohnson_transform(z, self.lambdas_)
                self.power_mean_, var, count = moments(transformed)
                self.power_scale_ = var.sqrt().masked_fill(
                    _constant_feature_mask(var, self.power_mean_, count), 1.0
                )
            elif self.method == "robust":
                q = z.nanquantile(z.new_tensor([0.25, 0.5, 0.75]), dim=0, keepdim=True)
                self.median_ = q[1]
                self.iqr_ = torch.where(q[0] == q[2], 1.0, q[2] - q[0])
            elif self.method == "rank":
                self.rank_values_, self.rank_probabilities_ = fit_rank(z)
                if self.max_knots and self.rank_values_.shape[1] > self.max_knots:
                    # Coarse quantile grid: cheaper table, smoother CDF.
                    step = (self.rank_values_.shape[1] - 1) / (self.max_knots - 1)
                    idx = torch.round(torch.arange(self.max_knots) * step).long()
                    self.rank_values_ = self.rank_values_[:, idx]
                    self.rank_probabilities_ = self.rank_probabilities_[:, idx]
            elif self.method != "none":
                raise ValueError(f"Unknown Lattice transform: {self.method}")
        transformed = self._distribution(z)
        # ClipSigma uses two passes and sample (rather than population) std.
        mean, var, count = moments(transformed)
        std = (var * count / (count - 1).clamp_min(1)).sqrt().clamp_min(1e-6)
        keep = torch.isfinite(transformed) & (
            (transformed - mean).abs() <= self.outlier_threshold * std
        )
        kept_count = keep.sum(0, keepdim=True)
        kept_mean = transformed.masked_fill(~keep, 0).sum(
            0, keepdim=True
        ) / kept_count.clamp_min(1)
        kept_var = (
            (transformed - kept_mean)
            .masked_fill(~keep, 0)
            .square()
            .sum(0, keepdim=True)
        )
        kept_std = (kept_var / (kept_count - 1).clamp_min(1)).sqrt().clamp_min(1e-6)
        mean = torch.where(kept_count > 0, kept_mean, mean)
        std = torch.where(kept_count > 0, kept_std, std)
        self.lower_ = mean - self.outlier_threshold * std
        self.upper_ = mean + self.outlier_threshold * std
        values = self._clip(transformed)
        self.X_transformed_ = values if self.tensor_output else values.cpu().numpy()
        return self

    def _distribution(self, z):
        if not z.shape[1] or self.method == "none":
            return z
        if self.method == "power":
            z = (
                _yeojohnson_transform(z, self.lambdas_) - self.power_mean_
            ) / self.power_scale_
            bound = torch.finfo(z.dtype).max
            return z.clamp(-bound, bound)
        if self.method == "robust":
            z = (z - self.median_) / self.iqr_
            unit = (z / 3).abs()
            root = (unit.square() + 1).sqrt()
            return torch.where(
                z.isnan(),
                z,
                (unit / root).masked_fill(~root.isfinite(), 1) * 3 * z.sign(),
            )
        probabilities = _batched_interp(
            z.T.contiguous(), self.rank_values_, self.rank_probabilities_
        )
        if self.kdi_alpha:
            # KDI-lite: blend the empirical rank with the normal CDF so tied or
            # discrete columns get a smooth, strictly increasing quantile map.
            parametric = 0.5 * (1.0 + torch.erf(z.T.contiguous() / (2.0 ** 0.5)))
            probabilities = (1.0 - self.kdi_alpha) * probabilities + (
                self.kdi_alpha * parametric
            )
        if self.rank_uniform:
            # Uniform margins: keeps the view bounded instead of blowing up tails.
            return probabilities.T.masked_fill(z.isnan(), torch.nan)
        return _ndtri(probabilities).T.masked_fill(z.isnan(), torch.nan)

    def _clip(self, z):
        delta = z.abs().log1p()
        return torch.minimum(torch.maximum(z, self.lower_ - delta), self.upper_ + delta)

    @torch.no_grad()
    def transform(self, X):
        x = self._tensor(X)[:, self.keep_]
        z = ((x - self.mean_) / self.scale_).clamp(-100, 100)
        values = self._clip(self._distribution(z))
        return values if getattr(self, "tensor_output", False) else values.cpu().numpy()
