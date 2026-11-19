"""Lattice-style interleaved (column block, row block) encoder rounds.

Instead of running all column ISAB blocks and then all row blocks (the serial
``col_embedder -> row_interactor`` pipeline of the baseline), the blocks are
zipped into alternating pairs, mirroring ``SDM``'s ``RowEmbedding``
(structured-data-models, ``sdm/models/lattice/tabular/row_embedding.py``):

``cell embedding -> (col_1 -> row_1) -> ... -> (col_n -> row_n) -> ICL``

Each pair switches between the column view ``(B, G, T, E)`` — where the
column block mixes rows within one feature group — and the row view
``(B, T, C+G, E)`` — where the row block mixes feature groups within one row
(RoPE over the token axis). The ``C`` class tokens follow Lattice's readout
tokens: seeded from the learnable ``cls_tokens`` parameter in the first round,
then split off each round's row-block output so they evolve across the whole
interleaved stack, and finally used as the query of the last row block.

The block objects themselves are lifted from the baseline construction (the
column ISABs of ``col_embedder.tf_col`` and the RoPE blocks of
``RowInteraction.tf_row``), so the parameter set is identical to the serial
baseline — the orchestration is the only variable.
"""

from __future__ import annotations

from functools import partial
from typing import Optional

import torch
from torch import nn, Tensor
from torch.utils.checkpoint import checkpoint

from .._model.inference_config import _default_max_batch_cells
from .._model.kv_cache import KVCache, KVCacheEntry


def _cell_budget(mgr_config=None) -> int:
    """Cell budget for one chunk of a column or row block.

    An explicit ``max_batch_cells`` on the row-stage config wins; otherwise the
    ``LATTICE_INFER_CHUNK_CELLS`` environment variable (or its default) applies.
    """
    if isinstance(mgr_config, dict):
        budget = mgr_config.get("max_batch_cells")
    elif mgr_config is not None:
        budget = getattr(mgr_config, "max_batch_cells", None)
    else:
        budget = None
    return int(budget) if budget is not None else _default_max_batch_cells()


def _chunk_len(total: int, seq_len: int, budget: int, batch: int = 1) -> int:
    """Number of ``total`` items that keep the chunk under ``budget`` cells.

    A chunk holds ``batch * items * seq_len`` cells, so the budget has to be
    shared across the leading batch dimension as well (an ensemble of ``E``
    tables otherwise overshoots it by ``E``). Chunks are kept equally sized, so
    a table that barely exceeds the budget does not degenerate into one full
    chunk plus a nearly empty tail.
    """
    if budget <= 0 or seq_len <= 0 or total <= 0:
        return total
    step = max(1, budget // (seq_len * max(1, batch)))
    if step >= total:
        return total
    n_chunks = -(-total // step)
    return -(-total // n_chunks)


class InterleavedRowInteraction(nn.Module):
    """Alternate column and row attention blocks over shared feature tokens.

    Parameters
    ----------
    col_blocks : nn.ModuleList
        ``InducedSelfAttentionBlock`` stack lifted from
        ``col_embedder.tf_col.blocks``; each block sees ``(B, G, T, E)`` and
        restricts its inducing-point keys/values to ``train_size`` rows.
    row_blocks : nn.ModuleList
        ``MultiheadAttentionBlock`` stack lifted from
        ``RowInteraction.tf_row.blocks``; each block sees ``(B, T, C+G, E)``
        with RoPE over the token axis.
    cls_tokens : nn.Parameter
        Learnable class-token seeds of shape ``(C, E)``.
    out_ln : nn.Module
        Final LayerNorm (or Identity) applied to the class-token outputs.
    rope : Optional[nn.Module]
        Rotary position embedding shared by the row blocks.
    embed_dim : int
        Embedding width ``E``.
    num_cls : int
        Number of class tokens ``C``.
    recompute : bool, default=False
        Wrap each interleaved round in gradient checkpointing.
    """

    def __init__(
        self,
        col_blocks: nn.ModuleList,
        row_blocks: nn.ModuleList,
        cls_tokens: nn.Parameter,
        out_ln: nn.Module,
        rope: Optional[nn.Module],
        embed_dim: int,
        num_cls: int,
        recompute: bool = False,
    ) -> None:
        super().__init__()
        if len(col_blocks) != len(row_blocks):
            raise ValueError(
                "interleave requires equal column and row block counts, got "
                f"{len(col_blocks)} and {len(row_blocks)}."
            )
        if len(col_blocks) < 1:
            raise ValueError("interleave requires at least one block pair.")
        self.col_blocks = col_blocks
        self.row_blocks = row_blocks
        self.cls_tokens = cls_tokens
        self.out_ln = out_ln
        self.rope = rope
        self.embed_dim = embed_dim
        self.num_cls = num_cls
        self.num_blocks = len(col_blocks)
        self.recompute = recompute

    @property
    def _attention_rope(self):
        # The fused QK norms apply the rotation; keep rope registered for saving.
        return None if getattr(self, "_use_fused_rope", False) else self.rope

    def _column_pass(
        self, features: Tensor, train_size: Optional[int], index: int
    ) -> Tensor:
        """Mix rows within each feature group: (B, T, G, E) -> (B, T, G, E)."""
        src = features.transpose(1, 2)  # (B, G, T, E)
        src = self.col_blocks[index](src, train_size)
        return src.transpose(1, 2)

    def _column_pass_chunked(
        self,
        features: Tensor,
        train_size: Optional[int],
        index: int,
        budget: int,
        col_cache: Optional[KVCache] = None,
        use_cache: bool = False,
        store_cache: bool = False,
    ) -> Tensor:
        """Column block over feature-group chunks, written back in place.

        Every feature group owns its own inducing points, so a group's output
        does not depend on which other groups share the batch: splitting ``G``
        is exact. The per-chunk K/V entries of the cached path are re-assembled
        into the single per-block entry the rest of the pipeline expects.
        """
        B, T, G, _ = features.shape
        step = _chunk_len(G, T, budget, B)

        if step >= G:
            src = features.transpose(1, 2)
            if col_cache is None:
                out = self.col_blocks[index](src, train_size)
            else:
                out = self.col_blocks[index].forward_with_cache(
                    src, col_cache, index, train_size, use_cache, store_cache
                )
            return out.transpose(1, 2)

        if col_cache is None:
            for g0 in range(0, G, step):
                chunk = features[:, :, g0 : g0 + step]
                out = self.col_blocks[index](chunk.transpose(1, 2), train_size)
                chunk.copy_(out.transpose(1, 2))
            return features

        if use_cache:
            entry = col_cache.kv[index]
            # Entries keep the full batch shape: (B, G, num_heads, seq, head_dim).
            col_dim = entry.key.dim() - 4
            for g0 in range(0, G, step):
                g1 = min(g0 + step, G)
                shim = KVCache()
                shim.kv[index] = KVCacheEntry(
                    key=entry.key.narrow(col_dim, g0, g1 - g0),
                    value=entry.value.narrow(col_dim, g0, g1 - g0),
                )
                chunk = features[:, :, g0:g1]
                out = self.col_blocks[index].forward_with_cache(
                    chunk.transpose(1, 2), shim, index, train_size, True, False
                )
                chunk.copy_(out.transpose(1, 2))
            return features

        entries = []
        for g0 in range(0, G, step):
            tmp = KVCache()
            chunk = features[:, :, g0 : g0 + step]
            out = self.col_blocks[index].forward_with_cache(
                chunk.transpose(1, 2), tmp, index, train_size, False, True
            )
            chunk.copy_(out.transpose(1, 2))
            entries.append(tmp.kv[index])
        col_cache.kv[index] = KVCacheEntry.concat(entries, dim=entries[0].key.dim() - 4)
        return features

    def _row_round(
        self,
        block: nn.Module,
        cls: Tensor,
        features: Tensor,
        key_mask: Optional[Tensor],
        budget: int,
        last: bool,
    ) -> tuple[Tensor, Tensor]:
        """Row block over row chunks, written back in place.

        The row block only attends across the tokens of a single row, so rows
        are independent and splitting ``T`` is exact.
        """
        B, T, _, _ = cls.shape
        step = _chunk_len(T, features.shape[2] + self.num_cls, budget, B)

        if step >= T:
            tokens = torch.cat((cls, features), dim=2)  # (B, T, C+G, E)
            if last:
                return (
                    block(q=cls, k=tokens, v=tokens, key_padding_mask=key_mask, rope=self._attention_rope),
                    features,
                )
            tokens = block(tokens, key_padding_mask=key_mask, rope=self._attention_rope)
            return tokens[..., : self.num_cls, :], tokens[..., self.num_cls :, :]

        for t0 in range(0, T, step):
            t1 = min(t0 + step, T)
            cls_chunk = cls[:, t0:t1]
            feat_chunk = features[:, t0:t1]
            mask = None if key_mask is None else key_mask[:, t0:t1]
            tokens = torch.cat((cls_chunk, feat_chunk), dim=2)
            if last:
                cls[:, t0:t1] = block(
                    q=cls_chunk, k=tokens, v=tokens, key_padding_mask=mask, rope=self._attention_rope
                )
            else:
                out = block(tokens, key_padding_mask=mask, rope=self._attention_rope)
                cls[:, t0:t1] = out[..., : self.num_cls, :]
                feat_chunk.copy_(out[..., self.num_cls :, :])
                del out
            del tokens
        return cls, features

    def _round(
        self,
        index: int,
        cls: Tensor,
        features: Tensor,
        key_mask: Optional[Tensor],
        train_size: Optional[int],
    ) -> tuple[Tensor, Tensor]:
        """One full interleaved round; the class tokens keep evolving."""
        features = self._column_pass(features, train_size, index)
        x = torch.cat((cls, features), dim=2)  # (B, T, C+G, E)
        x = self.row_blocks[index](x, key_padding_mask=key_mask, rope=self._attention_rope)
        return x[..., : self.num_cls, :], x[..., self.num_cls :, :]

    def _final_round(
        self,
        index: int,
        cls: Tensor,
        features: Tensor,
        key_mask: Optional[Tensor],
        train_size: Optional[int],
    ) -> Tensor:
        """Last round: the row block queries only with the class tokens."""
        features = self._column_pass(features, train_size, index)
        x = torch.cat((cls, features), dim=2)  # (B, T, C+G, E)
        return self.row_blocks[index](
            q=cls, k=x, v=x, key_padding_mask=key_mask, rope=self._attention_rope
        )

    def forward(
        self,
        embeddings: Tensor,
        d: Optional[Tensor] = None,
        mgr_config: Optional[dict] = None,
        train_size: Optional[int] = None,
    ) -> Tensor:
        """Map (B, T, num_cls + G, E) feature tokens to (B, T, num_cls * E).

        ``train_size`` restricts the column blocks' inducing-point
        keys/values to the leading rows, preserving the baseline's
        information-leak guard inside the interleaved schedule.
        """
        # mgr_config is accepted for signature compatibility with RowInteraction;
        # this module uses a single execution path for training and evaluation.
        B, T, HC, E = embeddings.shape

        # Drop the reserved (-100) class-token slots: like the baseline, the
        # class tokens come from the learnable parameter, not the column pass.
        features = embeddings[:, :, self.num_cls :, :]  # (B, T, G, E)
        cls = self.cls_tokens.to(features).expand(B, T, self.num_cls, self.embed_dim)

        # Same padding-mask construction as RowInteraction (inactive when
        # d is None, e.g. the default --ignore_d True training configuration).
        key_mask: Optional[Tensor] = None
        if d is not None:
            d = d + self.num_cls
            indices = torch.arange(HC, device=embeddings.device).view(1, 1, HC)
            key_mask = indices.expand(B, T, HC) >= d.view(B, 1, 1)  # (B, T, HC)

        last = self.num_blocks - 1
        budget = _cell_budget(mgr_config)
        if budget > 0 and not self.training and not torch.is_grad_enabled():
            # Chunked evaluation writes back into the same feature buffer, so the
            # footprint stays flat as the table grows in rows or columns instead
            # of scaling with ``T x G``.
            cls = cls.contiguous()
            for i in range(self.num_blocks):
                features = self._column_pass_chunked(features, train_size, i, budget)
                cls, features = self._row_round(
                    self.row_blocks[i], cls, features, key_mask, budget, i == last
                )
            return self.out_ln(cls).flatten(-2)  # (B, T, C*E)

        for i in range(self.num_blocks):
            if i < last:
                if self.recompute:
                    cls, features = checkpoint(
                        partial(
                            self._round, i, key_mask=key_mask, train_size=train_size
                        ),
                        cls,
                        features,
                        use_reentrant=False,
                    )
                else:
                    cls, features = self._round(i, cls, features, key_mask, train_size)
            else:
                if self.recompute:
                    cls = checkpoint(
                        partial(
                            self._final_round,
                            i,
                            key_mask=key_mask,
                            train_size=train_size,
                        ),
                        cls,
                        features,
                        use_reentrant=False,
                    )
                else:
                    cls = self._final_round(i, cls, features, key_mask, train_size)

        return self.out_ln(cls).flatten(-2)  # (B, T, C*E)

    def forward_with_cache(self, embeddings, col_cache, *, train_size=None, use_cache=False, store_cache=True):
        """Reuse each interleaved column block's support-side inducing states."""
        if use_cache == store_cache or (store_cache and train_size is None):
            raise ValueError("Select one cache operation and provide train_size when storing")
        B, T, _, _ = embeddings.shape
        features = embeddings[:, :, self.num_cls :, :]
        cls = self.cls_tokens.to(features).expand(B, T, self.num_cls, self.embed_dim)
        last = self.num_blocks - 1
        budget = _cell_budget()
        chunked = budget > 0 and not torch.is_grad_enabled()
        if chunked:
            cls = cls.contiguous()
        for i, (col, row) in enumerate(zip(self.col_blocks, self.row_blocks)):
            if chunked:
                features = self._column_pass_chunked(
                    features, train_size, i, budget, col_cache, use_cache, store_cache
                )
                cls, features = self._row_round(row, cls, features, None, budget, i == last)
                continue
            features = col.forward_with_cache(
                features.transpose(1, 2), col_cache=col_cache, block_idx=i,
                train_size=train_size, use_cache=use_cache, store_cache=store_cache,
            ).transpose(1, 2)
            tokens = torch.cat((cls, features), dim=2)
            if i == self.num_blocks - 1:
                cls = row(q=cls, k=tokens, v=tokens, rope=self._attention_rope)
            else:
                tokens = row(tokens, rope=self._attention_rope)
                cls, features = tokens[..., :self.num_cls, :], tokens[..., self.num_cls:, :]
        return self.out_ln(cls).flatten(-2)
