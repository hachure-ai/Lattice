"""Combine hybrid Fourier input encoding with the existing AttnRes ICL stack."""

from __future__ import annotations

from lattice._model_attn_res import LatticeAttnRes
from lattice._model_fourier_hybrid.embedding import HybridFourierFeatureEmbedding
from lattice._model_gated.attention import GatedMultiheadAttention


class LatticeFourierHybridAttnRes(LatticeAttnRes):
    """Keep raw categorical slots and jointly encode numerical Fourier slots."""

    def __init__(
        self,
        *args,
        fourier_num_frequencies: int = 32,
        fourier_sigma: float = 1.0,
        fourier_include_raw: bool = True,
        fourier_separate_categorical: bool = True,
        ecdf_rank_metadata: bool = False,
        fourier_embedding_style: str = "hybrid",
        attn_res_block_size: int = 2,
        attn_res_softmax: bool = True,
        icl_gated_attn: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            attn_res_block_size=attn_res_block_size,
            attn_res_softmax=attn_res_softmax,
            **kwargs,
        )
        in_features = self.col_feature_group_size if self.col_feature_group else 1
        self.fourier_num_frequencies = fourier_num_frequencies
        self.fourier_sigma = fourier_sigma
        self.fourier_include_raw = fourier_include_raw
        self.fourier_separate_categorical = fourier_separate_categorical
        self.fourier_embedding_style = fourier_embedding_style
        use_ecdf_rank_metadata = (
            ecdf_rank_metadata or fourier_embedding_style == "typed_sum_ecdf_ln"
        )
        # Let the column embedder compute per-cell train-row ECDF midranks and
        # feed them to the cell embedder's metadata channel.
        self.col_embedder.ecdf_rank_metadata = use_ecdf_rank_metadata
        self.col_embedder.in_linear = HybridFourierFeatureEmbedding(
            in_features=in_features,
            out_features=self.embed_dim,
            num_frequencies=fourier_num_frequencies,
            sigma=fourier_sigma,
            include_raw=fourier_include_raw,
            separate_categorical=fourier_separate_categorical,
            ecdf_rank_metadata=use_ecdf_rank_metadata,
            embedding_style=fourier_embedding_style,
        )
        self.icl_gated_attn = icl_gated_attn
        if icl_gated_attn:
            for block in self.icl_predictor.tf_icl.blocks:
                original = block.attn
                gated = GatedMultiheadAttention(
                    original.embed_dim, original.num_heads,
                    original.dropout, self.icl_ssmax,
                    qk_norm=self.qk_norm,
                )
                # Preserve the common initialization, including QASSMax and
                # the q/k QK-RMSNorm when enabled.
                gated.load_state_dict(original.state_dict(), strict=False)
                block.attn = gated
