"""Lattice backbone with block attention residuals."""

from __future__ import annotations

from lattice._model.base_model import LatticeModel

from .learning import AttnResICLearning


class LatticeAttnRes(LatticeModel):
    """Keep Lattice column/row processing and replace its ICL residual stream."""

    def __init__(self, *args, attn_res_block_size: int = 2, attn_res_softmax: bool = True, **kwargs) -> None:
        recompute = kwargs.get("recompute", False)
        super().__init__(*args, **kwargs)

        icl_dim = self.embed_dim * self.row_num_cls
        out_dim = self.num_quantiles if self.max_classes == 0 else self.max_classes
        self.attn_res_block_size = attn_res_block_size
        self.attn_res_softmax = attn_res_softmax
        self.icl_predictor = AttnResICLearning(
            max_classes=self.max_classes,
            out_dim=out_dim,
            d_model=icl_dim,
            num_blocks=self.icl_num_blocks,
            nhead=self.icl_nhead,
            dim_feedforward=icl_dim * self.ff_factor,
            dropout=self.dropout,
            activation=self.activation,
            norm_first=self.norm_first,
            bias_free_ln=self.bias_free_ln,
            ssmax=self.icl_ssmax,
            zero_init=self.zero_init,
            recompute=recompute,
            attn_res_block_size=attn_res_block_size,
            attn_res_softmax=attn_res_softmax,
            ffn_impl=self.ffn_impl,
            sandwich_post_norm=self.sandwich_post_norm,
            qk_norm=self.qk_norm,
        )
