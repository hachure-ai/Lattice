"""Two-buffer transfer of native KV member batches to CUDA."""

from __future__ import annotations

import torch


class CachePrefetcher:
    """Stage the next batch while the current batch runs on the compute stream.

    The caller retains the pinned CPU cache. This context owns its transfer
    stream and drains outstanding copies on exit, including OOM retries.
    """

    def __init__(self, cache, offset, members, batch, device):
        self.cache = cache
        self.bounds = iter(
            (start, min(start + batch, members))
            for start in range(offset, members, batch)
        )
        self.device = device
        self.stream = torch.cuda.Stream(device=device)
        self.pending = None

    def _enqueue(self):
        bounds = next(self.bounds, None)
        if bounds is None:
            self.pending = None
            return
        start, end = bounds
        with torch.cuda.stream(self.stream):
            staged = self.cache.slice_batch(start, end).to(
                self.device, non_blocking=True
            )
            ready = self.stream.record_event()
        self.pending = (start, end, staged, ready)

    def __enter__(self):
        # Any producer work on the current stream must finish before copying.
        self.stream.wait_stream(torch.cuda.current_stream(self.device))
        try:
            self._enqueue()
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __iter__(self):
        while self.pending is not None:
            start, end, staged, ready = self.pending
            self.pending = None
            compute = torch.cuda.current_stream(self.device)
            compute.wait_event(ready)
            staged.record_stream(compute)
            self._enqueue()
            yield start, end, staged
            del staged

    def __exit__(self, *_):
        self.stream.synchronize()
        self.pending = None
