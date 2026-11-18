"""block attention residuals over a transformer depth axis."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class AttnResRMSNorm(nn.Module):
    """RMSNorm used to turn residual sources into depth-attention keys."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: Tensor) -> Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.square().mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight.float() * hidden_states).to(input_dtype)


class AttentionResidual(nn.Module):
    """Selectively aggregate completed blocks and the current partial block.

    ``use_softmax=True`` mixes sources with a convex softmax over the source
    axis (the AttnRes). ``use_softmax=False`` drops the softmax denominator
    and weights each source by ``exp(score)`` instead; with the zero-init
    scorer every weight starts at 1, so the module initially reproduces the
    plain residual sum of all sources.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6, use_softmax: bool = True) -> None:
        super().__init__()
        self.norm = AttnResRMSNorm(hidden_size, eps=eps)
        self.proj = nn.Linear(hidden_size, 1, bias=False)
        nn.init.zeros_(self.proj.weight)
        self.use_softmax = use_softmax

    def forward(self, prefix_sum: Tensor, block_residual: Tensor) -> Tensor:
        """Apply scalar depth attention over the source dimension."""

        if block_residual.shape[-2] == 0:
            return prefix_sum

        values = torch.cat((block_residual, prefix_sum.unsqueeze(-2)), dim=-2)
        values_float = values.float()
        variance = values_float.square().mean(dim=-1, keepdim=True)
        keys = values_float * torch.rsqrt(variance + self.norm.variance_epsilon)
        score_weight = self.norm.weight.float() * self.proj.weight.squeeze(0).float()
        scores = (keys * score_weight).sum(dim=-1)
        if self.use_softmax:
            probabilities = scores.softmax(dim=-1)
        else:
            probabilities = scores.exp()
        output = (probabilities.unsqueeze(-1) * values_float).sum(dim=-2)
        return output.to(values.dtype)


def empty_block_residual(hidden_states: Tensor) -> Tensor:
    """Create an empty source bank with the token and channel axes preserved."""

    return hidden_states.new_empty(
        *hidden_states.shape[:-1], 0, hidden_states.shape[-1]
    )
