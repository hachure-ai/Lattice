"""Write member caches into contiguous GPU or pinned CPU storage."""

from __future__ import annotations

import torch

from lattice._model.kv_cache import KVCache, KVCacheEntry, LatticeCache

from .cache_residency import available_bytes, on_cuda


class CacheWriter:
    """Allocate once and write each member batch to its final slice.

    Copies use the current CUDA stream. One synchronization per batch makes
    the cache ready before exposing it or releasing the source cache.
    """

    def __init__(self, reference: LatticeCache, members: int, *, destination="cpu"):
        self.members = members
        self.written = 0
        self.device = next(reference.tensors()).device
        self.destination = torch.device(destination)
        self.cache = None
        if not self.resident or reference.train_shape[0] != members:
            self.cache = self._make_cache(reference, self._allocate)

    @property
    def resident(self):
        return self.destination.type == "cuda"

    def _allocate(self, value):
        return torch.empty(
            (self.members, *value.shape[1:]),
            dtype=value.dtype,
            device=self.destination,
            pin_memory=not self.resident and self.device.type == "cuda",
        )

    def _make_cache(self, reference, transform):
        def convert(value):
            return None if value is None else transform(value)

        def convert_component(component):
            if component is None:
                return KVCache()
            return KVCache(
                kv={
                    key: KVCacheEntry(convert(entry.key), convert(entry.value))
                    for key, entry in component.kv.items()
                },
                ecdf=None
                if component.ecdf is None
                else {key: convert(value) for key, value in component.ecdf.items()},
            )

        return LatticeCache(
            col_cache=convert_component(reference.col_cache),
            icl_cache=convert_component(reference.icl_cache),
            row_repr=convert(reference.row_repr),
            train_shape=(self.members, *reference.train_shape[1:]),
            num_classes=reference.num_classes,
        )

    @classmethod
    def for_fit(cls, estimator, source, members):
        if estimator.device_.type == "cuda" and estimator.kv_cache_device == "auto":
            device = next(source.tensors()).device
            if device.type == "cuda" and all(t.device == device for t in source.tensors()):
                existing = estimator.model_kv_cache_.values()
                resident = sum(cache.size_bytes() for cache in existing if on_cuda(cache))
                incoming = source.size_bytes() // source.train_shape[0] * members
                budget = (available_bytes(device) + resident) // 4
                if incoming + sum(cache.size_bytes() for cache in existing) <= budget:
                    if source.train_shape[0] == members:
                        return cls(source, members, destination=device)
                    # During a multi-batch build, leave room to repeat the
                    # first support forward alongside the preallocated cache.
                    get_fraction = getattr(torch.cuda.memory, "get_per_process_memory_fraction", None)
                    fraction = get_fraction(device) if get_fraction is not None else 0.7
                    total = torch.cuda.get_device_properties(device).total_memory
                    peak = torch.cuda.max_memory_allocated(device)
                    if incoming + peak * 1.125 <= total * fraction:
                        try:
                            return cls(source, members, destination=device)
                        except torch.cuda.OutOfMemoryError:
                            pass
        return cls(source, members)

    def append(self, source: LatticeCache, start: int, end: int):
        if start != self.written or not start < end <= self.members:
            raise ValueError("Cache batches must cover members once, in order")
        if self.cache is None:
            if source.train_shape[0] != self.members or end != self.members:
                raise ValueError("Direct cache requires all ensemble members")
            self.cache = self._make_cache(source, torch.Tensor.contiguous)
            torch.cuda.current_stream(self.device).synchronize()
            self.written = end
            return
        if source.train_shape != (end - start, *self.cache.train_shape[1:]):
            raise ValueError("Cache batch shape changed while fitting")
        if source.num_classes != self.cache.num_classes:
            raise ValueError("Cache classes changed while fitting")
        target = self.cache.slice_batch(start, end)
        pairs = list(zip(target.tensors(), source.tensors(), strict=True))
        for dst, src in pairs:
            if dst.shape != src.shape or dst.dtype != src.dtype:
                raise ValueError("Cache tensor shape or dtype changed while fitting")
        try:
            for dst, src in pairs:
                dst.copy_(src, non_blocking=self.device.type == "cuda")
        finally:
            if self.device.type == "cuda":
                torch.cuda.current_stream(self.device).synchronize()
        self.written = end

    def spill(self):
        """Move a completed prefix to pinned CPU before retrying a build."""
        writer = CacheWriter(self.cache, self.members)
        if self.written:
            writer.append(self.cache.slice_batch(0, self.written), 0, self.written)
        return writer

    def finish(self):
        if self.written != self.members:
            raise RuntimeError("Cannot expose an incomplete cache")
        return self.cache
