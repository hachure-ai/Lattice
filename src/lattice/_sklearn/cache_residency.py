"""Budget GPU cache residency independently of forward member batching."""

from __future__ import annotations

from collections import OrderedDict

import torch


def available_bytes(device):
    free, _ = torch.cuda.mem_get_info(device)
    return (
        free + torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    )


def on_cuda(cache):
    return next(cache.tensors()).is_cuda


def to_pinned_cpu(cache):
    if all(t.device.type == "cpu" and t.is_pinned() for t in cache.tensors()):
        return cache
    return cache.pin_memory()


def place_caches(estimator):
    """Keep at most a quarter of reclaimable device memory in persistent KV.

    Include already-resident KV in the budget to avoid alternating placement on
    consecutive calls. The remainder is reserved for query data and activations.
    Callers can also restore the fitted cache when preparing a model to serve.
    """
    caches = estimator.model_kv_cache_
    total = sum(cache.size_bytes() for cache in caches.values())
    resident = sum(cache.size_bytes() for cache in caches.values() if on_cuda(cache))
    budget = (available_bytes(estimator.device_) + resident) // 4
    policy = getattr(estimator, "kv_cache_device", "cpu")
    failed = getattr(estimator, "_cache_residency_failed", False)
    use_gpu = policy == "auto" and total <= budget and not failed
    reason = (
        "budget" if use_gpu else "cpu_policy" if policy == "cpu" else "budget_exceeded"
    )
    if failed:
        reason = "previous_oom"
    moved = OrderedDict()
    if use_gpu:
        try:
            for key, cache in caches.items():
                moved[key] = cache.to(estimator.device_)
        except torch.cuda.OutOfMemoryError:
            moved.clear()
            use_gpu = False
            reason = "promotion_oom"
            estimator._cache_residency_failed = True
            torch.cuda.empty_cache()
    if not use_gpu:
        moved = OrderedDict(
            (key, to_pinned_cpu(cache)) for key, cache in caches.items()
        )
    estimator.model_kv_cache_ = moved
    estimator.cache_residency_ = {
        "device": "cuda" if use_gpu else "cpu",
        "bytes": total,
        "budget_bytes": budget,
        "reason": reason,
    }


def evict_cache(estimator, cache):
    """Return pinned storage after a forward OOM without rebuilding support."""
    moved = to_pinned_cpu(cache)
    estimator._cache_residency_failed = True
    for key, existing in estimator.model_kv_cache_.items():
        if existing is cache:
            estimator.model_kv_cache_[key] = moved
    estimator.cache_residency_ = {
        **getattr(estimator, "cache_residency_", {}),
        "device": "cpu",
        "reason": "forward_oom",
    }
    return moved
