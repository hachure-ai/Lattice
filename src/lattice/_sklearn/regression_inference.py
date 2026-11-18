"""Device-side quantile reduction for Lattice-style regression ensembles."""

from __future__ import annotations

import warnings

import numpy as np
import torch

from .adaptive_batching import predict


def aggregate_quantiles(quantiles, signs, scale, mean):
    """Invert each member's target transform, then sort and trim like SDM."""
    quantiles = quantiles.float()
    quantiles = (
        quantiles * torch.as_tensor(signs, device=quantiles.device)[:, None, None]
    )
    quantiles = quantiles * float(scale) + float(mean)
    quantiles = quantiles.sort(dim=-1).values
    cut = int(0.2 * len(quantiles))
    if cut:
        quantiles = quantiles.sort(dim=0).values[cut:-cut]
    return quantiles.mean(dim=0)


def quantile_outputs(model, quantiles, keys, alphas):
    """Keep reduction on device; construct a distribution only for its ICDF."""
    results = {}
    if "raw_quantiles" in keys:
        results["raw_quantiles"] = quantiles
    if "mean" in keys:
        results["mean"] = quantiles.mean(dim=-1)
    if "variance" in keys:
        results["variance"] = quantiles.var(dim=-1)
    if "median" in keys or "quantiles" in keys:
        dist = model.quantile_dist(quantiles)
        if "median" in keys:
            results["median"] = dist.icdf(quantiles.new_tensor(0.5))
        if "quantiles" in keys:
            levels = (
                alphas
                if alphas is not None
                else [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            )
            results["quantiles"] = dist.icdf(quantiles.new_tensor(levels))
    return {key: results[key].float().cpu().numpy() for key in keys}


@torch.inference_mode()
def _predict_chunk(estimator, X, keys, alphas):
    cached = estimator.model_kv_cache_ is not None
    data = estimator._ensemble_data(X, mode="test" if cached else "both")
    for key, values in data:
        quantiles = predict(
            estimator,
            values[0],
            y=None if cached else values[1],
            masks=estimator._ensemble_cat_masks(key),
            cache=estimator.model_kv_cache_[key] if cached else None,
            return_tensor=True,
        )
        quantiles = aggregate_quantiles(
            quantiles,
            estimator.ensemble_generator_.target_signs_,
            estimator.y_scaler_.scale_[0],
            estimator.y_scaler_.mean_[0],
        )
        return quantile_outputs(estimator.model_, quantiles, keys, alphas)
    raise RuntimeError("Lattice ensemble is empty")


def predict_lattice(estimator, X, keys, alphas, feature_mask=None):
    if feature_mask is not None:
        raise ValueError(
            'feature_mask is unsupported by Lattice preprocessing; use preprocessing="legacy"'
        )
    rows = len(X)
    if estimator.device_.type == "cuda":
        free, _ = torch.cuda.mem_get_info(estimator.device_)
        free += torch.cuda.memory_reserved(
            estimator.device_
        ) - torch.cuda.memory_allocated(estimator.device_)
        # Reserve space for all members plus sorting/concatenation temporaries.
        bytes_per_row = estimator.n_estimators * estimator.model_.num_quantiles * 4 * 8
        rows = max(1, min(rows, free // 4 // bytes_per_row))
    results = {key: [] for key in keys}
    start = 0
    while start < len(X):
        try:
            out = _predict_chunk(estimator, X[start : start + rows], keys, alphas)
            for key in keys:
                results[key].append(out[key])
            start += rows
            continue
        except torch.cuda.OutOfMemoryError:
            if rows <= 1:
                raise
            rows = max(1, rows // 2)
            warnings.warn(
                f"Lattice quantile aggregation OOM: query_rows={rows}", RuntimeWarning
            )
        torch.cuda.empty_cache()
    results = {key: np.concatenate(value, axis=0) for key, value in results.items()}
    return results[keys[0]] if len(keys) == 1 else results
