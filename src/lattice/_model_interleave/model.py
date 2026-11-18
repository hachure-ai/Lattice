"""Fourier Hybrid + AttnRes backbone with Lattice-style interleaved col/row rounds."""

from __future__ import annotations

from typing import List, Optional

from torch import nn
from torch import Tensor

from lattice._model_fourier_hybrid_attn_res import LatticeFourierHybridAttnRes

from .row import InterleavedRowInteraction


class LatticeInterleave(LatticeFourierHybridAttnRes):
    """Interleave the column and row encoder blocks in SDM fashion.

    The baseline runs the column ISABs contiguously and then the row blocks
    contiguously (``col_embedder -> row_interactor``); this variant zips them
    into ``(column block -> row block)`` pairs so row- and column-wise
    information exchange happens at every depth (cf. ``SDM``'s
    ``RowEmbedding``). The block objects are lifted verbatim from the baseline
    construction — ``col_embedder.tf_col.blocks`` and the base
    ``RowInteraction``'s ``tf_row.blocks``/``cls_tokens``/``out_ln``/``rope`` —
    so the total depth, parameter set, and layer conventions are identical to
    the serial baseline; only the orchestration changes. Consequently
    ``col_num_blocks`` must equal ``row_num_blocks``.

    The class tokens follow Lattice's readout semantics: seeded from the
    learnable parameter in the first round, split off each round's row-block
    output to evolve across the whole stack, and used as the query of the last
    row block. The Fourier cell embedding, label injection, and ``cat_mask``
    routing inside ``col_embedder`` are untouched (its emptied ``tf_col``
    passes embeddings straight through), as is the AttnRes ICL stack.
    """

    def __init__(self, *args, **kwargs) -> None:
        # The base class passes ``recompute`` straight to its submodules
        # without storing it, so capture it before super().__init__.
        recompute = kwargs.get("recompute", False)
        super().__init__(*args, **kwargs)
        if self.col_num_blocks != self.row_num_blocks:
            raise ValueError(
                "interleave requires col_num_blocks == row_num_blocks, got "
                f"{self.col_num_blocks} and {self.row_num_blocks}."
            )
        # 1) Empty the column ISAB stack; SetTransformer forwards through an
        #    empty block list, so col_embedder becomes a pure cell embedder
        #    (grouping -> Fourier -> label injection).
        col_blocks = self.col_embedder.tf_col.blocks
        self.col_embedder.tf_col.blocks = nn.ModuleList()
        # 2) Replace the row stage with the interleaved orchestrator built from
        #    the baseline row stack's parts. The attribute swap keeps
        #    freeze/checkpoint tooling's handle.
        base = self.row_interactor
        self.row_interactor = InterleavedRowInteraction(
            col_blocks=col_blocks,
            row_blocks=base.tf_row.blocks,
            cls_tokens=base.cls_tokens,
            out_ln=base.out_ln,
            rope=base.tf_row.rope,
            embed_dim=self.embed_dim,
            num_cls=self.row_num_cls,
            recompute=recompute,
        )

    def _train_forward(
        self,
        X: Tensor,
        y_train: Tensor,
        d: Optional[Tensor] = None,
        embed_with_test: bool = False,
        idempotency_loop_counts: Optional[List[int]] = None,
        cat_mask: Optional[Tensor] = None,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Column-wise embedding -> interleaved col/row rounds -> ICL.

        Identical to the baseline orchestration except that ``train_size`` is
        forwarded to the interleaved encoder: the column blocks now live in
        the row stage and must keep their inducing-point keys/values within
        the training rows.
        """
        B, T, H = X.shape
        train_size = y_train.shape[1]
        assert train_size <= T, "Number of training samples exceeds total samples"

        # Check if d is provided and has the same length as the number of features
        if d is not None and len(d.unique()) == 1 and d[0] == H:
            d = None

        # Cell embedding -> interleaved column/row rounds
        representations = self.row_interactor(
            self.col_embedder(
                X,
                y_train=y_train,
                d=d,
                embed_with_test=embed_with_test,
                cat_mask=cat_mask,
            ),
            d=d,
            train_size=None if embed_with_test else train_size,
        )

        # Dataset-wise in-context learning
        return self.icl_predictor(
            representations,
            y_train=y_train,
            idempotency_loop_counts=idempotency_loop_counts,
        )
