"""Install the QK and ICL kernels on supported inference models."""

import os
from contextlib import ExitStack
from functools import lru_cache
from types import MethodType

import torch
import torch.nn.functional as F
import triton

from ._fast_attnres import fast_encoder_forward, fast_encoder_forward_with_cache
from ._fast_cell import fused_norm, make_rope_norm
from ._fast_fused import fused_layer_norm, fused_rms_norm, fused_swiglu, install_fused
from .norms import QKRMSNorm
from .ssmax import QASSMaxMLP, _logn


@lru_cache(maxsize=None)
def _qk_selftest(device):
    generator = torch.Generator(device=device).manual_seed(0)
    norm = QKRMSNorm(16).to(device)
    x = torch.randn((2, 3, 8, 16), device=device, generator=generator)
    x = (x * 0.5).to(torch.float16)
    return torch.equal(norm(x), fused_norm(norm, x))


@lru_cache(maxsize=None)
def _fused_selftest(device):
    """Check the fused norm/FFN kernels against the eager fp16 reference."""
    generator = torch.Generator(device=device).manual_seed(0)
    x = torch.randn((64, 512), device=device, generator=generator).half()
    w = torch.randn((512,), device=device, generator=generator)
    b = torch.randn((512,), device=device, generator=generator)
    ref = F.layer_norm(x.float(), (512,), w, b, 1e-5).half()
    out = fused_layer_norm(x, w, b, 1e-5)
    if out is None or (ref.float() - out.float()).abs().max() > 4e-3:
        return False

    q = torch.randn((64, 8, 64), device=device, generator=generator).half()
    qw = torch.ones((64,), device=device)
    eps = torch.finfo(torch.float32).eps
    ref = F.rms_norm(q.float(), (64,), qw, eps).half()
    out = fused_rms_norm(q, qw, eps)
    if out is None or (ref.float() - out.float()).abs().max() > 4e-3:
        return False

    gate = torch.randn((64, 1024), device=device, generator=generator).half()
    up = torch.randn((64, 1024), device=device, generator=generator).half()
    ref = F.silu(gate) * up
    out = fused_swiglu(gate, up)
    return out is not None and (ref.float() - out.float()).abs().max() <= 4e-3


def install(model, *, fuse_qk: bool, allow_kv_cache: bool = False) -> bool:
    """Enable the validated kernels for adaptive inference.

    The installed kernels record what they were chosen for, ``(fuse_qk, kv, device)``.
    A network object can outlive one fit: AutoGluon's ``SharedWeights`` hands the
    same module to every estimator, and a fit that reuses a registered network
    may run with another AMP decision (a small warmup fit installs ``fuse_qk=False``
    where a large one wants ``True``). An install that does not match the request
    is therefore removed and redone instead of being reported as already in place.

    ``allow_kv_cache`` also patches ``forward_with_cache``, the encoder path a
    KV-cached fit builds and serves from. The QK/RoPE rewiring stays off in that
    mode: the cached path passes ``rope`` explicitly, so it must keep the eager
    modules.
    """
    encoder = model.icl_predictor.tf_icl
    blocks = getattr(encoder, "blocks", ())
    if not blocks or not hasattr(encoder, "attn_res_block_size") or blocks[0].norm1.normalized_shape[0] not in (256, 512):
        return False
    device = blocks[0].norm1.weight.device
    config = (bool(fuse_qk), bool(allow_kv_cache), device.type, device.index)

    if hasattr(model, "_fastpath_restore"):
        if getattr(model, "_fastpath_config", None) == config:
            return True
        uninstall(model)

    if torch.is_grad_enabled():
        return False
    if torch.__version__.split("+")[0] != "2.7.1" or triton.__version__ != "3.3.1":
        return False

    if fuse_qk:
        row = model.row_interactor
        if not hasattr(row, "row_blocks") or not hasattr(row, "col_blocks"):
            return False
        if device.type != "cuda" or not _qk_selftest(device):
            return False

    with ExitStack() as changes:
        def replace(module, name, value):
            if name in vars(module):
                changes.callback(setattr, module, name, vars(module)[name])
            else:
                changes.callback(delattr, module, name)
            setattr(module, name, value)

        row_norms = set()
        if fuse_qk:
            rope = row.rope
            if rope is not None:
                row_norms = {
                    module for block in row.row_blocks for module in block.modules()
                    if isinstance(module, QKRMSNorm)
                }
                rope_forward = make_rope_norm(rope, {})
                # Keep the RoPE module registered so state_dict still includes its weights.
                replace(row, "_use_fused_rope", True)
            replace(row, "recompute", False)
            for module in row.modules():
                if isinstance(module, QKRMSNorm):
                    forward = rope_forward if module in row_norms else fused_norm
                    replace(module, "forward", MethodType(forward, module))
            for block in row.col_blocks:
                replace(block, "forward", block.induced_attention)
            cached_logn = lru_cache(maxsize=64)(_logn)
            for module in model.modules():
                if isinstance(module, QASSMaxMLP):
                    replace(module, "_logn", cached_logn)

        replace(encoder, "forward", MethodType(fast_encoder_forward, encoder))
        if allow_kv_cache and hasattr(type(encoder), "forward_with_cache"):
            replace(encoder, "forward_with_cache",
                    MethodType(fast_encoder_forward_with_cache, encoder))
        fused_requested = os.environ.get("LATTICE_FUSED", "0").lower() not in (
            "0", "off", "false", "disabled", "",
        )
        if fuse_qk and fused_requested and _fused_selftest(device):
            install_fused(model, encoder, row_norms, replace)
        model._fastpath_config = config
        model._fastpath_restore = changes.pop_all()
    return True


def uninstall(model) -> None:
    """Restore the eager path and release this model's device-local caches."""
    restore = getattr(model, "_fastpath_restore", None)
    if restore is not None:
        del model._fastpath_restore
        if hasattr(model, "_fastpath_config"):
            del model._fastpath_config
        restore.close()
