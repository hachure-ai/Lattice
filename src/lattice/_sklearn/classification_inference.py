"""Aggregate Lattice-style classification members on the execution device."""

from __future__ import annotations

import warnings

import numpy as np
import torch

from .adaptive_batching import predict


def aggregate_probabilities(outputs, shuffles, *, average_logits, temperature):
    """Undo label permutations and preserve the previous member addition order."""
    average = torch.zeros_like(outputs[0], dtype=torch.float32)
    for output, shuffle in zip(outputs, shuffles, strict=True):
        if shuffle is not None:
            indices = torch.as_tensor(shuffle, device=outputs.device)
            output = output.index_select(-1, indices)
        average.add_(output)
    average.div_(len(outputs))
    if average_logits:
        average = torch.softmax(average / temperature, dim=-1)
    return average / average.sum(dim=-1, keepdim=True)


@torch.inference_mode()
def _predict_chunk(estimator, X):
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
        probabilities = aggregate_probabilities(
            outputs,
            estimator.ensemble_generator_.class_shuffles_[key],
            average_logits=estimator.average_logits,
            temperature=estimator.softmax_temperature_,
        )
        return probabilities.cpu().numpy()
    raise RuntimeError("Lattice ensemble is empty")


def predict_lattice(estimator, X, feature_mask=None):
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
            results.append(_predict_chunk(estimator, X[start : start + rows]))
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
