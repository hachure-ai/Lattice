"""Memory-adaptive member batches for native Lattice inference."""

from __future__ import annotations

import math
import os
import warnings
from collections import OrderedDict

import numpy as np
import torch

from .cache_residency import available_bytes, evict_cache, on_cuda
from .cache_storage import CacheWriter


def memory_terms(model, columns, element_bytes=2):
    """Workspace per row, native KV per row, and fixed column KV per member."""
    mode = model.col_feature_group
    groups = (
        math.ceil(columns / model.col_feature_group_size)
        if mode == "valid"
        else columns
    )
    groups += model.row_num_cls
    cell = model.embed_dim
    icl = cell * model.row_num_cls
    layers = model.icl_num_blocks
    workspace = (
        12 * groups * cell * element_bytes
        + 4 * layers * icl * 4
        + 12 * icl * element_bytes
    )
    workspace += 4 * getattr(model, "num_quantiles", 0)
    cache = 2 * layers * icl * element_bytes + 2 * columns * 4
    fixed = (
        2 * model.col_num_blocks * groups * model.col_num_inds * cell * element_bytes
    )
    return workspace, cache, fixed


def choose_batch(estimator, members, support, query, columns, phase):
    if estimator.device_.type != "cuda":
        return 1, 0
    free = available_bytes(estimator.device_)
    amp, _ = estimator._resolve_amp_fa3()
    work, cache, fixed = memory_terms(estimator.model_, columns, 2 if amp else 4)
    # S3 measurements with 16/128 columns show up to ~1.4x the analytic
    # activation estimate. Reserve 1.5x, independently of persistent KV.
    if phase != "uncached_predict":
        work = math.ceil(work * 1.5)
    if phase == "cache_build":
        per_member = support * (work + 2 * cache) + 2 * fixed
    elif phase == "cached_predict":
        caches = getattr(estimator, "model_kv_cache_", None)
        resident = caches and all(on_cuda(value) for value in caches.values())
        # Resident KV is already subtracted from free memory. CPU prefetch
        # requires two device batches; do not charge for resident KV twice.
        staging = 0 if resident else 2 * (support * cache + fixed)
        rows = query + (
            support if getattr(estimator, "cache_mode_", "kv") == "repr" else 0
        )
        per_member = rows * work + staging
    else:
        # Hierarchical classification can retain multiple support representations.
        class_factor = (
            max(1, math.ceil(estimator.n_classes_ / estimator.model_.max_classes))
            if estimator.model_.max_classes > 0
            else 1
        )
        per_member = (support + query) * work * class_factor + fixed
        # Uncached splitting repeats support processing. The S3 end-to-end
        # comparison found no reliable benefit from the KV shape caps here;
        # retain the original memory-only policy for this phase.
        return max(1, min(members, (free // 2) // max(1, per_member))), free
    limit = max(1, min(members, (free // 2) // max(1, per_member)))
    rows = query if phase == "cached_predict" else support + query
    if phase == "cached_predict" and getattr(estimator, "cache_mode_", "kv") == "repr":
        rows += support
    shape_limit = member_budget(rows, columns, estimator.model_.embed_dim)
    limit = min(limit, shape_limit)
    # Uniform powers of two avoid a large first batch and a tiny tail for the
    # usual 4/8-member ensemble. Explicit batch_size still overrides this path.
    if 1 < limit < members:
        limit = 1 << (int(limit).bit_length() - 1)
    return limit, free


def member_budget(rows, columns, embed_dim):
    """Bound aggregate rows/cells as well as bytes; zero disables either cap."""
    row_budget = int(os.environ.get("LATTICE_MAX_BATCH_ROWS", "49152"))
    cell_budget = int(os.environ.get("LATTICE_MAX_BATCH_CELLS", "8388608"))
    scale = max(1, embed_dim / 128)
    limits = [2**31]
    if row_budget > 0:
        limits.append(max(1, int(row_budget / scale) // max(1, rows)))
    if cell_budget > 0:
        limits.append(max(1, int(cell_budget / scale) // max(1, rows * columns)))
    return min(limits)


def record(estimator, phase, batch, rows, support, columns, available, retries):
    estimator.batching_info_.append(
        {
            "phase": phase,
            "member_batch_size": batch,
            "query_chunk_rows": rows,
            "support_rows": support,
            "transformed_columns": columns,
            "available_bytes": available,
            "oom_retries": retries,
            "cache_prefetch": phase == "cached_predict"
            and estimator.device_.type == "cuda",
        }
    )


def build_cache(estimator):
    """Build into GPU storage within budget or fall back to pinned CPU."""
    estimator.model_kv_cache_ = OrderedDict()
    for key, (X, y) in estimator._ensemble_data(mode="train"):
        masks = estimator._ensemble_cat_masks(key)
        batch, available = choose_batch(
            estimator, len(X), y.shape[1], 0, X.shape[2], "cache_build"
        )
        offset, retries, writer = 0, 0, None
        while offset < len(X):
            end = min(offset + batch, len(X))
            try:
                writer = _cache_call(
                    estimator,
                    X[offset:end],
                    y[offset:end],
                    masks,
                    offset,
                    end,
                    writer,
                    len(X),
                )
                offset = end
                continue
            except torch.cuda.OutOfMemoryError:
                estimator.model_.clear_cache()
                if writer is not None and writer.resident:
                    writer = writer.spill()
                elif batch <= 1:
                    raise  # Never silently turn a KV experiment into uncached inference.
                else:
                    batch = max(1, batch // 2)
                    warnings.warn(
                        f"Lattice KV build OOM: reducing member batch to {batch}",
                        RuntimeWarning,
                    )
                retries += 1
            torch.cuda.empty_cache()
        estimator.model_kv_cache_[key] = writer.finish()
        record(
            estimator,
            "cache_build",
            batch,
            0,
            y.shape[1],
            X.shape[2],
            available,
            retries,
        )


@torch.inference_mode()
def _cache_call(estimator, X, y, masks, start, end, writer, members):
    with estimator._amp_context():
        estimator.model_.forward_with_cache(
            X_train=torch.as_tensor(X, device=estimator.device_, dtype=torch.float32),
            y_train=torch.as_tensor(y, device=estimator.device_, dtype=torch.float32),
            use_cache=False,
            store_cache=True,
            cache_mode=estimator.cache_mode_,
            inference_config=estimator.inference_config_,
            **estimator._cat_kwargs(masks, start, end),
        )
    try:
        source = estimator.model_._cache
        if writer is None:
            writer = CacheWriter.for_fit(estimator, source, members)
        try:
            writer.append(source, start, end)
        except torch.cuda.OutOfMemoryError:
            if not writer.resident:
                raise
            writer = writer.spill() if writer.written else CacheWriter(source, members)
            writer.append(source, start, end)
    finally:
        estimator.model_.clear_cache()
    return writer


@torch.inference_mode()
def _predict_members(
    estimator,
    X,
    y,
    shuffles,
    masks,
    offset,
    end,
    rows,
    cache,
    *,
    staged_cache=None,
    return_tensor=False,
):
    """A function boundary releases device tensors before the OOM retry."""
    support = 0 if y is None else y.shape[1]
    query = X.shape[1] - support
    outputs = []
    staged = staged_cache
    if staged is None and cache is not None:
        staged = cache if offset == 0 and end == cache.train_shape[0] else cache.slice_batch(offset, end)
        cache_device = next(staged.tensors()).device
        target_device = estimator.device_
        if target_device.type == "cuda" and target_device.index is None:
            target_device = torch.device("cuda", torch.cuda.current_device())
        if cache_device != target_device:
            staged = staged.to(estimator.device_)
    target = (
        None
        if y is None
        else torch.as_tensor(y, device=estimator.device_, dtype=torch.float32)
    )
    for start in range(0, query, rows):
        values = X[:, support + start : support + start + rows]
        if cache is None and start == 0 and rows >= query:
            values = X
        elif cache is None:
            blocks = (X[:, :support], values)
            values = (
                torch.cat(blocks, dim=1)
                if isinstance(X, torch.Tensor)
                else np.concatenate(blocks, axis=1)
            )
        features = torch.as_tensor(
            values, device=estimator.device_, dtype=torch.float32
        )
        kwargs = dict(
            inference_config=estimator.inference_config_,
            **estimator._cat_kwargs(masks, offset, end),
        )
        regression = estimator.model_.max_classes == 0
        if not regression:
            kwargs.update(
                return_logits=estimator.average_logits,
                softmax_temperature=estimator.softmax_temperature_,
            )
        with estimator._amp_context():
            if regression:
                if return_tensor and staged is None:
                    output = estimator.model_._inference_forward(
                        X=features, y_train=target, **kwargs
                    )
                elif return_tensor:
                    output = estimator.model_.forward_with_cache(
                        X_test=features,
                        cache=staged,
                        use_cache=True,
                        store_cache=False,
                        **kwargs,
                    )
                elif staged is None:
                    output = estimator.model_.predict_stats(
                        X=features,
                        y_train=target,
                        output_type="raw_quantiles",
                        **kwargs,
                    )
                else:
                    output = estimator.model_.predict_stats_with_cache(
                        X_test=features,
                        cache=staged,
                        use_cache=True,
                        store_cache=False,
                        output_type="raw_quantiles",
                        **kwargs,
                    )
            elif staged is None:
                output = estimator.model_(
                    X=features, y_train=target, feature_shuffles=shuffles, **kwargs
                )
            else:
                output = estimator.model_.forward_with_cache(
                    X_test=features, cache=staged, **kwargs
                )
        outputs.append(
            output.float() if return_tensor else output.float().cpu().numpy()
        )
    if len(outputs) == 1:
        return outputs[0]
    return (
        torch.cat(outputs, dim=1) if return_tensor else np.concatenate(outputs, axis=1)
    )


def predict(
    estimator, X, y=None, shuffles=None, masks=None, cache=None, *, return_tensor=False
):
    support = y.shape[1] if y is not None else cache.train_shape[1]
    query = X.shape[1] - (support if cache is None else 0)
    # Uncached prediction starts with all query rows to avoid recomputing
    # support; query splitting is an OOM fallback. Cached query work is bounded.
    rows = (
        query if cache is None else estimator._test_rows_per_forward(query, X.shape[2])
    )
    phase = "uncached_predict" if cache is None else "cached_predict"
    batch, available = choose_batch(estimator, len(X), support, rows, X.shape[2], phase)
    if estimator.batch_size != "auto":
        batch = min(len(X), estimator.batch_size or len(X))
    offset, retries, outputs = 0, 0, []
    prefetch = (
        cache is not None and estimator.device_.type == "cuda" and not on_cuda(cache)
    )
    if prefetch and not all(tensor.is_pinned() for tensor in cache.tensors()):
        cache = cache.pin_memory()
    while offset < len(X):
        end = min(offset + batch, len(X))
        permutations = (
            None if shuffles is None else np.asarray(shuffles[offset:end]).tolist()
        )
        try:
            if prefetch:
                from .cache_prefetch import CachePrefetcher

                staged = None
                try:
                    with CachePrefetcher(
                        cache, offset, len(X), batch, estimator.device_
                    ) as pending:
                        for start, end, staged in pending:
                            outputs.append(
                                _predict_members(
                                    estimator,
                                    X[start:end],
                                    None,
                                    None,
                                    masks,
                                    start,
                                    end,
                                    rows,
                                    cache,
                                    staged_cache=staged,
                                    return_tensor=return_tensor,
                                )
                            )
                            offset = end
                            staged = None
                finally:
                    staged = None
                continue
            outputs.append(
                _predict_members(
                    estimator,
                    X[offset:end],
                    None if y is None else y[offset:end],
                    permutations,
                    masks,
                    offset,
                    end,
                    rows,
                    cache,
                    return_tensor=return_tensor,
                )
            )
            offset = end
            continue
        except torch.cuda.OutOfMemoryError:
            if cache is not None and on_cuda(cache):
                cache = evict_cache(estimator, cache)
                prefetch = True
            elif batch > 1:
                batch = max(1, batch // 2)
            elif rows > 1:
                rows = max(1, rows // 2)
            else:
                raise
            retries += 1
            warnings.warn(
                f"Lattice prediction OOM: members={batch}, query_rows={rows}",
                RuntimeWarning,
            )
        torch.cuda.empty_cache()
    record(estimator, phase, batch, rows, support, X.shape[2], available, retries)
    estimator.batching_info_[-1]["cache_prefetch"] = prefetch
    estimator.batching_info_[-1]["cache_resident"] = cache is not None and on_cuda(
        cache
    )
    if len(outputs) == 1:
        return outputs[0]
    return (
        torch.cat(outputs, dim=0) if return_tensor else np.concatenate(outputs, axis=0)
    )
