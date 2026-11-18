"""Dataset-wise ICL using block attention residuals."""

from __future__ import annotations

from typing import Union

from lattice._model.learning import ICLearning

from .encoders import AttnResEncoder


class AttnResICLearning(ICLearning):
    """Replace the homogeneous ICL transformer stack with Block AttnRes."""

    def __init__(
        self,
        max_classes: int,
        out_dim: int,
        d_model: int,
        num_blocks: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.0,
        activation: str | callable = "gelu",
        norm_first: bool = True,
        bias_free_ln: bool = False,
        ssmax: Union[bool, str] = False,
        zero_init: bool = True,
        recompute: bool = False,
        attn_res_block_size: int = 2,
        attn_res_softmax: bool = True,
        ffn_impl: str = "mlp",
        sandwich_post_norm: bool = False,
        qk_norm: bool = False,
    ) -> None:
        super().__init__(
            max_classes=max_classes,
            out_dim=out_dim,
            d_model=d_model,
            num_blocks=num_blocks,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            bias_free_ln=bias_free_ln,
            ssmax=ssmax,
            zero_init=zero_init,
            recompute=recompute,
            ffn_impl=ffn_impl,
            sandwich_post_norm=sandwich_post_norm,
            qk_norm=qk_norm,
        )
        self.tf_icl = AttnResEncoder(
            num_blocks=num_blocks,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            bias_free_ln=bias_free_ln,
            ssmax=ssmax,
            zero_init=zero_init,
            recompute=recompute,
            attn_res_block_size=attn_res_block_size,
            attn_res_softmax=attn_res_softmax,
            ffn_impl=ffn_impl,
            sandwich_post_norm=sandwich_post_norm,
            qk_norm=qk_norm,
        )
