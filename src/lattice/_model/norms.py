"""Normalization primitives for the norm ablations.

``QKRMSNorm`` applies an RMS normalization with a learnable per-channel gain to
the projected queries/keys head-wise (TabPFN-3.5 style QK-norm), so it
can be ablated on top of the Lattice LayerNorm blocks via ``--qk_norm``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class QKRMSNorm(nn.Module):
    """QK-RMSNorm: RMS-normalize each head vector, then apply a learnable gain.

    The mean square is taken over the last (head) dimension and ``weight`` is a
    per-channel gain of shape ``(head_dim,)`` initialized to one, following the
    standard RMSNorm convention (TabPFN-3.5 ``nn.RMSNorm(head_dim)``). The math
    runs through the fused ``F.rms_norm`` kernel: the
    equivalent eager formulation costs several extra fp32 elementwise passes
    over q/k in every attention layer.
    """

    def __init__(self, head_dim: int, eps: float | None = None) -> None:
        super().__init__()
        if head_dim <= 0:
            raise ValueError(f"head_dim must be positive, got {head_dim}.")
        self.head_dim = head_dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(head_dim))

    def forward(self, x: Tensor) -> Tensor:
        if x.size(-1) != self.head_dim:
            raise ValueError(
                f"QKRMSNorm expected head_dim {self.head_dim} on the last axis, got {x.size(-1)}."
            )
        # Default to the fp32 machine epsilon, matching the clamp the unit-RMS
        # variant used; ``F.rms_norm`` adds eps under the root instead, which is
        # numerically equivalent for non-degenerate heads.
        eps = torch.finfo(torch.float32).eps if self.eps is None else self.eps
        if x.dtype == torch.float16:
            # Squaring overflows fp16, so normalize in fp32 and cast back.
            return F.rms_norm(x.float(), (self.head_dim,), self.weight.float(), eps).to(x.dtype)
        # The fused kernel requires the gain to share the input dtype.
        weight = self.weight if self.weight.dtype == x.dtype else self.weight.to(x.dtype)
        return F.rms_norm(x, (self.head_dim,), weight, eps)


def attach_qk_norm(module: nn.Module, qk_norm: bool, head_dim: int) -> None:
    """Create the q/k RMSNorm submodules when the ablation is enabled.

    The submodule names are shared by every attention implementation (plain
    and gated) so that state dicts transfer between them.
    """

    if qk_norm:
        module.q_norm = QKRMSNorm(head_dim)
        module.k_norm = QKRMSNorm(head_dim)
    else:
        module.q_norm = None
        module.k_norm = None
