"""Aggregate Lattice-style classification members on the execution device."""

from __future__ import annotations

import warnings

import numpy as np
import torch

from .adaptive_batching import predict


def aggregate_scores(outputs, shuffles, *, average_logits, temperature, return_logits=False):
    """Align all member class permutations and average on the execution device.

    With ``return_logits`` the pre-softmax ensemble average is returned when
    members were averaged as logits; otherwise the returned scores are
    log-probabilities.
    """
    if len(shuffles) != len(outputs):
        raise ValueError("Expected one class permutation per ensemble member")
    classes = outputs.shape[-1]
    indices = np.stack([np.arange(classes) if shuffle is None else shuffle for shuffle in shuffles])
    indices = torch.as_tensor(indices, device=outputs.device)
    aligned = outputs.gather(-1, indices[:, None, :].expand_as(outputs))
    average = aligned.mean(dim=0, dtype=torch.float32)
    if not average_logits:
        average = average / average.sum(dim=-1, keepdim=True)
        if return_logits:
            return torch.log(average.clamp_min(torch.finfo(average.dtype).tiny))
        return average
    if return_logits:
        return average
    average = torch.softmax(average / temperature, dim=-1)
    return average / average.sum(dim=-1, keepdim=True)


@torch.inference_mode()
def _predict_chunk(estimator, X, *, return_logits=False):
    cached = estimator.model_kv_cache_ is not None
    data = estimator._ensemble_data(X, mode="test" if cached else "both")
    for key, values in data:
        outputs = predict(
            estimator,
            values[0],
            y=None if cached else values[1],
            shuffles=None
            if cached
            else estimator.ensemble_generator_.feature_shuffles_[key],
            masks=estimator._ensemble_cat_masks(key),
            cache=estimator.model_kv_cache_[key] if cached else None,
            return_tensor=True,
        )
        scores = aggregate_scores(
            outputs,
            estimator.ensemble_generator_.class_shuffles_[key],
            average_logits=estimator.average_logits,
            temperature=estimator.softmax_temperature_,
            return_logits=return_logits,
        )
        return scores.cpu().numpy()
    raise RuntimeError("Lattice ensemble is empty")


def predict_lattice(estimator, X, feature_mask=None, *, return_logits=False):
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
        # Bound member outputs plus transformed feature views. Internal forward
        # batching separately reserves memory for activations and cache staging.
        bytes_per_row = (
            estimator.n_estimators * estimator.n_classes_ * 4 * 4
            + min(2 * X.shape[1], 500) * (estimator.n_estimators + 8) * 8
        )
        rows = max(1, min(rows, free // 4 // max(1, bytes_per_row)))
    results = []
    start = 0
    while start < len(X):
        try:
            results.append(
                _predict_chunk(estimator, X[start : start + rows], return_logits=return_logits)
            )
            start += rows
            continue
        except torch.cuda.OutOfMemoryError:
            if rows <= 1:
                raise
            rows = max(1, rows // 2)
            warnings.warn(
                f"Lattice probability aggregation OOM: query_rows={rows}", RuntimeWarning
            )
        torch.cuda.empty_cache()
    return results[0] if len(results) == 1 else np.concatenate(results, axis=0)

