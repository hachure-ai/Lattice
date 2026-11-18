"""Hybrid LatticeModel and Fourier encoding for grouped table features."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class HybridFourierFeatureEmbedding(nn.Module):
    """Embed grouped values with the legacy hybrid or typed-sum Fourier path.

    Each numerical slot is represented by learned sine/cosine features and,
    optionally, its raw value. A categorical slot keeps only its raw value in
    the first channel. Representations for all slots in a feature group are
    concatenated before one joint projection. Setting ``separate_categorical``
    to ``False`` encodes every slot with the numerical path and ignores the
    categorical mask. ``embedding_style='typed_sum_ecdf_ln'`` instead gives
    numerical and categorical slots independent Fourier banks and 65-to-E
    projections, sums their slot embeddings, adds ECDF metadata, and applies
    LayerNorm.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_frequencies: int = 32,
        sigma: float = 1.0,
        include_raw: bool = True,
        separate_categorical: bool = True,
        ecdf_rank_metadata: bool = False,
        embedding_style: str = "hybrid",
        skip_value: float = -100.0,
    ) -> None:
        super().__init__()
        if in_features <= 0:
            raise ValueError("in_features must be positive.")
        if out_features <= 0:
            raise ValueError("out_features must be positive.")
        if num_frequencies <= 0:
            raise ValueError("num_frequencies must be positive.")
        if not math.isfinite(sigma) or sigma <= 0:
            raise ValueError("sigma must be a finite positive value.")

        self.in_features = in_features
        self.out_features = out_features
        self.num_frequencies = num_frequencies
        self.sigma = sigma
        self.include_raw = include_raw
        self.separate_categorical = separate_categorical
        if embedding_style not in {"hybrid", "typed_sum_ecdf_ln"}:
            raise ValueError(f"Unknown Fourier embedding style: {embedding_style!r}.")
        if embedding_style == "typed_sum_ecdf_ln" and not include_raw:
            raise ValueError("typed_sum_ecdf_ln requires include_raw=True.")
        if embedding_style == "typed_sum_ecdf_ln" and not separate_categorical:
            raise ValueError("typed_sum_ecdf_ln requires separate_categorical=True.")
        self.embedding_style = embedding_style
        self.skip_value = skip_value
        self.slot_features = num_frequencies * 2 + int(include_raw)

        # ECDF metadata channel (TabPFN v3.5 port): each per-column midrank is
        # lifted to fixed half-period sin/cos features and projected by a
        # dedicated bias-free linear that is summed onto the value embedding.
        self.ecdf_rank_metadata = (
            ecdf_rank_metadata or embedding_style == "typed_sum_ecdf_ln"
        )
        if self.ecdf_rank_metadata:
            self.ecdf_num_frequencies = 4
            # Half-period phases pi*k*u (k = 1..4) keep u=0 and u=1
            # distinguishable at every frequency parity.
            self.register_buffer(
                "ecdf_frequencies", math.pi * torch.arange(1, 5)
            )
            self.metadata_linear = nn.Linear(
                in_features * self.ecdf_num_frequencies * 2, out_features, bias=False
            )

        self.frequencies = nn.Parameter(torch.empty(in_features, num_frequencies))
        if embedding_style == "typed_sum_ecdf_ln":
            # Lattice-inspired typed paths: every slot retains the original raw
            # value and gets its own learned Fourier frequencies, while the
            # 65->E projections are shared across group positions per type.
            self.categorical_frequencies = nn.Parameter(
                torch.empty(in_features, num_frequencies)
            )
            self.numerical_projection = nn.Linear(self.slot_features, out_features)
            self.categorical_projection = nn.Linear(self.slot_features, out_features)
            self.output_layernorm = nn.LayerNorm(out_features)
        else:
            self.projection = nn.Linear(in_features * self.slot_features, out_features)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.frequencies, std=self.sigma)
        if self.embedding_style == "typed_sum_ecdf_ln":
            nn.init.normal_(self.categorical_frequencies, std=self.sigma)
            for projection in (
                self.numerical_projection,
                self.categorical_projection,
            ):
                nn.init.normal_(
                    projection.weight,
                    std=projection.in_features**-0.5,
                )
                if projection.bias is not None:
                    nn.init.normal_(projection.bias, std=1e-6)
        else:
            nn.init.normal_(
                self.projection.weight,
                std=self.projection.in_features**-0.5,
            )
            if self.projection.bias is not None:
                nn.init.normal_(self.projection.bias, std=1e-6)
        if self.ecdf_rank_metadata:
            nn.init.normal_(
                self.metadata_linear.weight,
                std=self.metadata_linear.in_features**-0.5,
            )

    @staticmethod
    def _fourier_features(src: Tensor, frequencies: Tensor) -> Tensor:
        angles = src.float().unsqueeze(-1) * frequencies.float()
        return torch.cat((angles.sin(), angles.cos()), dim=-1).to(dtype=src.dtype)

    def _numerical_features(
        self, src: Tensor, frequencies: Tensor | None = None
    ) -> Tensor:
        periodic = self._fourier_features(
            src, self.frequencies if frequencies is None else frequencies
        )
        if not self.include_raw:
            return periodic
        return torch.cat((src.unsqueeze(-1), periodic), dim=-1)

    def _categorical_features(self, src: Tensor) -> Tensor:
        # Keeping only x makes the categorical path linear in the raw value,
        # matching LatticeModel while retaining a fixed-width slot representation.
        return F.pad(src.unsqueeze(-1), (0, self.slot_features - 1))

    def _ecdf_fourier_features(self, ranks: Tensor) -> Tensor:
        """Lift per-cell ECDF ranks in [0, 1] to (..., in_features, 2K) sin/cos."""
        phases = ranks.float().unsqueeze(-1) * self.ecdf_frequencies.float()
        return torch.cat((phases.sin(), phases.cos()), dim=-1).to(dtype=ranks.dtype)

    def forward(
        self, src: Tensor, cat_mask: Tensor | None = None, ecdf_rank: Tensor | None = None
    ) -> Tensor:
        if src.ndim < 2:
            raise ValueError(
                "HybridFourierFeatureEmbedding expects at least two input dimensions."
            )
        if src.shape[-1] != self.in_features:
            raise ValueError(
                f"Expected {self.in_features} input features, got {src.shape[-1]}."
            )

        skip_mask = (src == self.skip_value).all(dim=-1)
        numerical = self._numerical_features(src)

        if self.embedding_style == "typed_sum_ecdf_ln":
            if cat_mask is not None and cat_mask.shape[-1] != self.in_features:
                raise ValueError(
                    f"Expected categorical mask width {self.in_features}, "
                    f"got {cat_mask.shape[-1]}."
                )
            categorical = self._numerical_features(
                src, self.categorical_frequencies
            )
            numerical_out = self.numerical_projection(numerical)
            categorical_out = self.categorical_projection(categorical)
            if cat_mask is None:
                selected_out = numerical_out
            else:
                selected_out = torch.where(
                    cat_mask.bool().unsqueeze(-1),
                    categorical_out,
                    numerical_out,
                )
            out = selected_out.sum(dim=-2)
        elif cat_mask is None or not self.separate_categorical:
            selected = numerical
            out = self.projection(selected.flatten(start_dim=-2))
        else:
            if cat_mask.shape[-1] != self.in_features:
                raise ValueError(
                    f"Expected categorical mask width {self.in_features}, "
                    f"got {cat_mask.shape[-1]}."
                )
            categorical = self._categorical_features(src)
            selected = torch.where(
                cat_mask.bool().unsqueeze(-1),
                categorical,
                numerical,
            )
            out = self.projection(selected.flatten(start_dim=-2))
        if self.ecdf_rank_metadata:
            if ecdf_rank is None:
                raise ValueError(
                    "ecdf_rank is required when ecdf_rank_metadata=True."
                )
            if ecdf_rank.shape != src.shape:
                raise ValueError(
                    f"Expected ECDF rank shape {tuple(src.shape)}, "
                    f"got {tuple(ecdf_rank.shape)}."
                )
            metadata = self._ecdf_fourier_features(ecdf_rank).flatten(start_dim=-2)
            out = out + self.metadata_linear(metadata)
        if self.embedding_style == "typed_sum_ecdf_ln":
            out = self.output_layernorm(out)
        return torch.where(
            skip_mask.unsqueeze(-1),
            out.new_tensor(self.skip_value),
            out,
        )
