# Adapted from TabICL v2 (https://github.com/soda-inria/tabicl).
# Copyright (c) 2025 Soda team @ Inria. BSD 3-Clause License; see NOTICE and
# src/lattice/_model/LICENSE.

"""
This module provides cache data structures for storing key-value projections
from attention layers, enabling efficient inference by reusing computed values
across test samples.

The caching strategy focuses on:
1. ColEmbedding: Cache K/V of the second attention layer of ISAB blocks
2. ICLearning: Cache K/V from training data at each layer of the ICL transformer
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import torch
from torch import Tensor


@dataclass
class KVCacheEntry:
    """A single key-value cache entry for an attention layer.

    Attributes
    ----------
    key : Optional[Tensor]
        Cached key projections of shape ``(batch, num_heads, seq_len, head_dim)``.

    value : Optional[Tensor]
        Cached value projections of shape ``(batch, num_heads, seq_len, head_dim)``.
    """

    key: Optional[Tensor] = None
    value: Optional[Tensor] = None

    def is_valid(self) -> bool:
        """Check if this cache entry contains valid data."""
        return self.key is not None and self.value is not None

    def __getitem__(self, indices) -> KVCacheEntry:
        """Slice key/value along batch dimensions.

        Returns a new KVCacheEntry with sliced tensors, or an empty entry
        if this entry is not valid.
        """
        if not self.is_valid():
            return KVCacheEntry()
        return KVCacheEntry(key=self.key[indices], value=self.value[indices])

    def __setitem__(self, indices, other: KVCacheEntry):
        """Write a batch slice into this entry."""
        if other.is_valid() and self.is_valid():
            self.key[indices] = other.key
            self.value[indices] = other.value

    def to(self, device, dtype=None, non_blocking=False) -> KVCacheEntry:
        """Move this entry to the given device and optionally cast dtype.

        Returns a new KVCacheEntry.
        """
        if not self.is_valid():
            return KVCacheEntry()
        return KVCacheEntry(
            key=self.key.to(device=device, dtype=dtype, non_blocking=non_blocking),
            value=self.value.to(device=device, dtype=dtype, non_blocking=non_blocking),
        )

    @staticmethod
    def concat(entries: List[KVCacheEntry], dim: int = 0) -> KVCacheEntry:
        """Concatenate multiple KVCacheEntry objects along a dimension.

        Parameters
        ----------
        entries : List[KVCacheEntry]
            Entries to concatenate. All must be valid.

        dim : int, default=0
            Dimension to concatenate along (batch dimension).

        Returns
        -------
        KVCacheEntry
            New entry with concatenated key and value tensors.
        """
        keys = [e.key for e in entries if e.is_valid()]
        values = [e.value for e in entries if e.is_valid()]
        if not keys:
            return KVCacheEntry()
        return KVCacheEntry(key=torch.cat(keys, dim=dim), value=torch.cat(values, dim=dim))


@dataclass
class KVCache:
    """Base class for key-value caches used across model components.

    Provides common structure and operations for caches that store
    ``Dict[int, KVCacheEntry]`` mappings.

    Attributes
    ----------
    kv : Dict[int, KVCacheEntry]
        Maps layer/block index to cached key-value projections.

    ecdf : Optional[Dict[str, Tensor]]
        Support-side ECDF reference statistics (sorted support values and the
        clamped midranks). Kept next to the K/V entries so that query chunks
        can rank their own cells against the same support distribution.
    """

    kv: Dict[int, KVCacheEntry] = field(default_factory=dict)
    ecdf: Optional[Dict[str, Tensor]] = None

    def is_populated(self) -> bool:
        """Check if this cache has valid entries.

        Returns True when the cache contains data (use_cache mode).
        Returns False when the cache is empty (store_cache mode).
        """
        return any(entry.is_valid() for entry in self.kv.values())

    def __getitem__(self, indices) -> KVCache:
        """Slice all entries along batch dimensions.

        Returns a new cache of the same subclass type with sliced entries.
        """
        sliced_kv = {idx: entry[indices] for idx, entry in self.kv.items()}
        sliced_ecdf = None if self.ecdf is None else {name: value[indices] for name, value in self.ecdf.items()}
        return self.__class__(kv=sliced_kv, ecdf=sliced_ecdf)

    def __setitem__(self, indices, other: KVCache):
        """Write batch-sliced entries into this pre-allocated cache."""
        for idx, other_entry in other.kv.items():
            if idx in self.kv:
                target = self.kv[idx]
                assert target.is_valid(), f"Cannot write to cache index {idx} because it is not valid."
                self.kv[idx][indices] = other_entry.to(target.key.device, dtype=target.key.dtype)
        if other.ecdf is not None:
            if self.ecdf is None:
                self.ecdf = {name: value.clone() for name, value in other.ecdf.items()}
            else:
                for name, value in other.ecdf.items():
                    self.ecdf[name][indices] = value.to(self.ecdf[name].device)

    def to(self, device, dtype=None, non_blocking=False) -> KVCache:
        """Move all entries to the given device and optionally cast dtype.

        Returns a new cache of the same subclass type.
        """
        moved_kv = {idx: entry.to(device, dtype=dtype, non_blocking=non_blocking) for idx, entry in self.kv.items()}
        moved_ecdf = None if self.ecdf is None else {
            name: value.to(device=device, non_blocking=non_blocking) for name, value in self.ecdf.items()
        }
        return self.__class__(kv=moved_kv, ecdf=moved_ecdf)


    @staticmethod
    def concat(caches: List[KVCache], dim: int = 0) -> KVCache:
        """Concatenate multiple KVCache objects along a dimension.

        Parameters
        ----------
        caches : List[KVCache]
            Caches to concatenate. All must have the same layer indices.

        dim : int, default=0
            Dimension to concatenate along (batch dimension).

        Returns
        -------
        KVCache
            New cache with concatenated entries at each layer index.
        """
        all_indices = set()
        for cache in caches:
            all_indices.update(cache.kv.keys())
        merged_kv = {}
        for idx in sorted(all_indices):
            entries = [cache.kv[idx] for cache in caches if idx in cache.kv]
            merged_kv[idx] = KVCacheEntry.concat(entries, dim=dim)
        merged_ecdf = None
        if caches and all(cache.ecdf is not None for cache in caches):
            merged_ecdf = {
                name: torch.cat([cache.ecdf[name] for cache in caches], dim=dim)
                for name in caches[0].ecdf
            }
        return KVCache(kv=merged_kv, ecdf=merged_ecdf)

    def preallocate(self, reference: KVCache, batch_shape: tuple, device="cpu", dtype=None):
        """Pre-allocate entries in this cache based on shapes from a reference.

        K/V tensors always have shape ``(*batch, num_heads, seq_len, head_dim)``.
        This method keeps the last three dimensions from the reference entry and
        prepends ``batch_shape`` as the leading dimensions.

        Parameters
        ----------
        reference : KVCache
            A cache from a single batch whose entry shapes are used as a
            template.

        batch_shape : tuple
            The full batch shape to use for the leading dimensions.

        device : str or torch.device
            Device on which to allocate the tensors.

        dtype : torch.dtype or None
            Data type for the allocated tensors. If None, uses the reference
            entry's dtype.
        """
        for idx, ref_entry in reference.kv.items():
            if ref_entry.is_valid():
                target_dtype = dtype if dtype is not None else ref_entry.key.dtype
                key_shape = batch_shape + ref_entry.key.shape[-3:]
                value_shape = batch_shape + ref_entry.value.shape[-3:]
                self.kv[idx] = KVCacheEntry(
                    key=torch.zeros(key_shape, dtype=target_dtype, device=device),
                    value=torch.zeros(value_shape, dtype=target_dtype, device=device),
                )

        if reference.ecdf is not None:
            if self.ecdf is None:
                self.ecdf = {}
            for name, tensor in reference.ecdf.items():
                shape = tuple(batch_shape) + tuple(tensor.shape[1:])
                self.ecdf[name] = torch.zeros(shape, dtype=tensor.dtype, device=device)


@dataclass
class LatticeCache:
    """Top-level cache container for the entire LatticeModel model.

    This aggregates caches for different components of LatticeModel:

    - ColEmbedding cache (for ISAB blocks in column embedding)
    - ICLearning cache (for Encoder layers in the ICL transformer)

    Attributes
    ----------
    col_cache : Optional[KVCache]
        Cache for ColEmbedding ISAB blocks.

    row_repr : Optional[Tensor]
        Cached row representations from the model.

    icl_cache : Optional[KVCache]
        Cache for ICLearning Encoder layers.

    train_shape : Tuple[int, int, int]
        Shape ``(batch_size, train_size, num_features)`` of training data the
        cache was built with.

    num_classes : Optional[int]
        Number of classes in classification tasks (0 for regression).
        Stored when caching to ensure consistent output shape during cache use.
    """

    col_cache: Optional[KVCache] = None
    row_repr: Optional[Tensor] = None
    icl_cache: Optional[KVCache] = None
    train_shape: Tuple[int, int, int] = (0, 0, 0)
    num_classes: Optional[int] = None

    def __post_init__(self):
        """Initialize sub-caches if not provided."""
        if self.col_cache is None:
            self.col_cache = KVCache()
        if self.icl_cache is None:
            self.icl_cache = KVCache()

    @property
    def cache_type(self) -> str:
        """Return the cache type: 'kv', 'repr', or 'empty'."""
        if self.row_repr is not None:
            return "repr"
        col_populated = self.col_cache is not None and self.col_cache.kv
        icl_populated = self.icl_cache is not None and self.icl_cache.kv
        if col_populated or icl_populated:
            return "kv"
        return "empty"

    def cache_size_mb(self) -> int:
        """Return the memory occupied by cached tensors in MB."""
        total = 0
        # Count memory from ColEmbedding
        if self.col_cache:
            for kv in self.col_cache.kv.values():
                if kv.key is not None:
                    total += kv.key.numel() * kv.key.element_size()
                if kv.value is not None:
                    total += kv.value.numel() * kv.value.element_size()
        # Count memory from row representations
        if self.row_repr is not None:
            total += self.row_repr.numel() * self.row_repr.element_size()
        # Count memory from ICLearning
        if self.icl_cache:
            for kv in self.icl_cache.kv.values():
                if kv.key is not None:
                    total += kv.key.numel() * kv.key.element_size()
                if kv.value is not None:
                    total += kv.value.numel() * kv.value.element_size()

        return total // (1024 * 1024)

    def is_empty(self) -> bool:
        """Check if the cache is empty."""
        col_empty = self.col_cache is None or not self.col_cache.kv
        row_empty = self.row_repr is None
        icl_empty = self.icl_cache is None or not self.icl_cache.kv

        return col_empty and row_empty and icl_empty

    def slice_batch(self, start: int, end: int) -> LatticeCache:
        """Slice this cache along the batch dimension (dim 0).

        Parameters
        ----------
        start : int
            Start index of the batch slice.

        end : int
            End index of the batch slice (exclusive).

        Returns
        -------
        LatticeCache
            New cache with sliced tensors (views of the original tensors).
        """
        indices = slice(start, end)
        return LatticeCache(
            col_cache=self.col_cache[indices] if self.col_cache else KVCache(),
            row_repr=self.row_repr[indices] if self.row_repr is not None else None,
            icl_cache=self.icl_cache[indices] if self.icl_cache else KVCache(),
            train_shape=(end - start, self.train_shape[1], self.train_shape[2]),
            num_classes=self.num_classes,
        )

    def tensors(self):
        """Iterate over projections, row representations, and ECDF statistics."""
        for cache in (self.col_cache, self.icl_cache):
            if cache is not None:
                for entry in cache.kv.values():
                    for tensor in (entry.key, entry.value):
                        if tensor is not None:
                            yield tensor
                if cache.ecdf is not None:
                    yield from cache.ecdf.values()
        if self.row_repr is not None:
            yield self.row_repr

    def pin_memory(self) -> LatticeCache:
        """Return a CPU cache with page-locked tensors for asynchronous transfer."""
        def pin(tensor):
            return None if tensor is None else tensor.cpu().pin_memory()

        def pin_cache(cache):
            if cache is None:
                return KVCache()
            return cache.__class__(
                kv={idx: KVCacheEntry(pin(entry.key), pin(entry.value)) for idx, entry in cache.kv.items()},
                ecdf=None if cache.ecdf is None else {name: pin(value) for name, value in cache.ecdf.items()},
            )

        return LatticeCache(
            col_cache=pin_cache(self.col_cache),
            row_repr=pin(self.row_repr),
            icl_cache=pin_cache(self.icl_cache),
            train_shape=self.train_shape,
            num_classes=self.num_classes,
        )

    def record_stream(self, stream):
        """Keep transferred tensors alive while the compute stream uses them."""
        for tensor in self.tensors():
            if tensor.is_cuda:
                tensor.record_stream(stream)

    def size_bytes(self):
        """Logical byte size, including ECDF statistics and row representations."""
        return sum(tensor.numel() * tensor.element_size() for tensor in self.tensors())

    def to(self, device, dtype=None, non_blocking=False) -> LatticeCache:
        """Move all cached tensors to the given device and optionally cast dtype.

        Parameters
        ----------
        device : str or torch.device
            Target device (e.g. ``'cpu'``, ``'cuda:0'``).

        dtype : torch.dtype or None
            Target dtype. If None, preserves the existing dtype.

        Returns
        -------
        LatticeCache
            New cache with all tensors on the target device.
        """
        return LatticeCache(
            col_cache=self.col_cache.to(device, dtype=dtype, non_blocking=non_blocking) if self.col_cache else KVCache(),
            row_repr=self.row_repr.to(device=device, dtype=dtype, non_blocking=non_blocking)
            if self.row_repr is not None else None,
            icl_cache=self.icl_cache.to(device, dtype=dtype, non_blocking=non_blocking) if self.icl_cache else KVCache(),
            train_shape=self.train_shape,
            num_classes=self.num_classes,
        )

    @staticmethod
    def concat(caches: List[LatticeCache], dim: int = 0) -> LatticeCache:
        """Concatenate multiple LatticeCache objects along the batch dimension.

        Parameters
        ----------
        caches : List[LatticeCache]
            Caches to concatenate.

        dim : int, default=0
            Dimension to concatenate along (batch dimension).

        Returns
        -------
        LatticeCache
            New cache with concatenated ``col_cache`` and ``icl_cache`` tensors.
        """
        col_caches = [c.col_cache for c in caches if c.col_cache is not None]
        row_reprs = [c.row_repr for c in caches if c.row_repr is not None]
        icl_caches = [c.icl_cache for c in caches if c.icl_cache is not None]

        total_batch = sum(c.train_shape[0] for c in caches)
        train_size = caches[0].train_shape[1]
        n_features = caches[0].train_shape[2]

        return LatticeCache(
            col_cache=KVCache.concat(col_caches, dim=dim) if col_caches else KVCache(),
            row_repr=torch.cat(row_reprs, dim=dim) if row_reprs else None,
            icl_cache=KVCache.concat(icl_caches, dim=dim) if icl_caches else KVCache(),
            train_shape=(total_batch, train_size, n_features),
            num_classes=caches[0].num_classes,
        )
