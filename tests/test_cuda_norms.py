"""Numerical regressions for the CUDA kernels used by the released checkpoints."""
from functools import lru_cache
from types import SimpleNamespace

import pytest
import torch

triton = pytest.importorskip("triton")
from lattice._model import _fast_attnres as residual, fastpath
from lattice._model._fast_cell import fused_norm, make_rope_norm
from lattice._model._fast_fused import fused_swiglu, fused_swiglu_forward, pack_projection_pair, packed_swiglu_forward
from lattice._model.kv_cache import KVCache
from lattice._model.layers import InducedSelfAttentionBlock, MultiheadAttentionBlock, SwiGLU
from lattice._model_interleave.row import InterleavedRowInteraction
from lattice._model._fast_norm214 import HAS_FUSED_RMS_NORM, layer_norm_forward
from lattice._model.norms import QKRMSNorm
from lattice._model.rope import RotaryEmbedding

pytestmark = pytest.mark.skipif(not HAS_FUSED_RMS_NORM or not torch.cuda.is_available(),
                                reason="requires CUDA and native fused RMSNorm")


def assert_encoder_declines(width):
    # A supported PyTorch build may use a different native reduction order.
    # When a candidate disagrees, the public installation path must leave
    # the encoder's native methods intact.
    device = torch.device("cuda", torch.cuda.current_device())
    assert not fastpath._encoder_selftest(device, width), "installed kernel differs from native PyTorch"
    norm = torch.nn.LayerNorm(width, device=device)
    encoder = SimpleNamespace(blocks=[SimpleNamespace(norm1=norm)], attn_res_block_size=2)
    model = SimpleNamespace(icl_predictor=SimpleNamespace(tf_icl=encoder))
    assert not fastpath.install(model, fuse_qk=False)
    assert not hasattr(encoder, "forward")
    assert not hasattr(model, "_fastpath_restore")


@pytest.mark.parametrize("width", [256, 512])
def test_benchmarked_stack_selects_encoder(width):
    device = torch.device("cuda", torch.cuda.current_device())
    if not (torch.__version__ == "2.14.1+cu130" and triton.__version__ == "3.8.0"
            and "RTX PRO 6000 Blackwell" in torch.cuda.get_device_name(device)):
        pytest.skip("fast-path selection regression for the benchmark GPU and software stack")
    assert fastpath._encoder_selftest(device, width)


@pytest.mark.parametrize("cached", [True, False])
@pytest.mark.parametrize("fuse_qk", [True, False])
@torch.no_grad()
def test_encoder_probe_failure_keeps_eager(monkeypatch, cached, fuse_qk):
    norm = torch.nn.LayerNorm(256).cuda()
    encoder = SimpleNamespace(blocks=[SimpleNamespace(norm1=norm)], attn_res_block_size=2)
    model = SimpleNamespace(icl_predictor=SimpleNamespace(tf_icl=encoder), row_interactor=None)
    monkeypatch.setattr(fastpath, "_supports_cached_qk", lambda row: True)
    monkeypatch.setattr(fastpath, "_encoder_selftest", lambda device, width: False)
    assert not fastpath.install(model, fuse_qk=fuse_qk, allow_kv_cache=cached)
    assert not hasattr(encoder, "forward")
    assert not hasattr(model, "_fastpath_restore")


@pytest.mark.parametrize("dim", [16, 32, 64])
@torch.no_grad()
def test_qk_and_rotary_match_native(dim):
    rng = torch.Generator(device="cuda").manual_seed(213)
    x = torch.randn(4, 257, 8, dim, device="cuda", generator=rng).half().transpose(1, 2)
    norm = QKRMSNorm(dim).cuda()
    norm.weight.normal_(generator=rng)
    with torch.autocast("cuda", dtype=torch.float16):
        assert torch.equal(fused_norm(norm, x), norm(x))
        if dim == 16:
            rope = RotaryEmbedding(dim=dim, interleaved=False).cuda()
            assert torch.equal(make_rope_norm(rope, {})(norm, x), norm(rope.rotate_queries_or_keys(x)))


@pytest.mark.parametrize("width", [128, 256, 512])
@pytest.mark.parametrize("dtype,bias", [(torch.float32, True), (torch.float16, False)])
@torch.no_grad()
def test_block_input_layer_norm(width, dtype, bias):
    rng = torch.Generator(device="cuda").manual_seed(430)
    x = torch.randn(131, 5, width, device="cuda", generator=rng).to(dtype).transpose(0, 1)
    norm = torch.nn.LayerNorm(width, bias=bias).cuda()
    norm.weight.normal_(generator=rng)
    if bias:
        norm.bias.normal_(generator=rng)
    with torch.autocast("cuda", dtype=torch.float16):
        actual, expected = layer_norm_forward(norm, x), norm(x)
    assert actual.dtype == expected.dtype
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("width", [256, 512])
@torch.no_grad()
def test_residual_scale_and_scores(width):
    rng = torch.Generator(device="cuda").manual_seed(103)
    rows, slots = 137, 5
    values = torch.randn(rows, slots, width, device="cuda", generator=rng)
    weight = torch.randn(width, device="cuda", generator=rng)
    bank = torch.empty_like(values)
    scales = torch.empty((rows, slots), device="cuda")
    scores = torch.empty_like(scales)
    for i in range(slots):
        residual.write_slot[(rows,)](values[:, i].contiguous(), bank, scales, slots,
                                    1e-6, i, width, N=width // 128, INV=1 / width)
    group = residual.SCORE_ROWS
    residual.score512[(triton.cdiv(rows, group), slots)](
        bank, scales, weight, scores, slots, slots, rows, width, group, width // 128,
    )
    expected_scale = torch.rsqrt(values.square().mean(-1) + 1e-6)
    expected_scores = ((values * expected_scale.unsqueeze(-1)) * weight).sum(-1)
    if not (torch.equal(scales, expected_scale) and torch.equal(scores, expected_scores)):
        assert_encoder_declines(width)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@torch.no_grad()
def test_swiglu_matches_native(dtype):
    # All 16-bit gate encodings, including subnormals and SiLU rounding ties.
    bits = torch.arange(65536, device="cuda").to(torch.int16)
    if dtype == torch.float32:
        rng = torch.Generator(device="cuda").manual_seed(71)
        gate = torch.cat((bits.view(torch.float16).float(),
                          15 * torch.randn(65536, device="cuda", generator=rng)))
    else:
        gate = bits.view(dtype)
    gate = torch.cat([gate, gate[:1]])  # Partial final Triton block.
    up = torch.linspace(-1.75, 2.125, gate.numel(), dtype=torch.float32, device="cuda").to(dtype)
    expected = torch.nn.functional.silu(gate) * up
    actual = fused_swiglu(gate, up)
    assert actual is not None and actual.dtype == expected.dtype
    assert torch.equal(actual.isnan(), expected.isnan())
    valid = ~expected.isnan()
    if not torch.equal(actual[valid].view(torch.uint8), expected[valid].view(torch.uint8)):
        for width in (256, 512):
            assert_encoder_declines(width)


def test_swiglu_keeps_autograd():
    module = SwiGLU(16, 48).cuda()
    x = torch.randn(3, 16, device="cuda", requires_grad=True)
    ref_grad, = torch.autograd.grad(module(x).sum(), x)
    grad, = torch.autograd.grad(fused_swiglu_forward(module, x).sum(), x)
    torch.testing.assert_close(grad, ref_grad, rtol=0, atol=0)


@torch.no_grad()
def test_packed_swiglu_refreshes_weights():
    module = SwiGLU(128, 256).cuda().eval()
    module._pack_gate_up = lru_cache(maxsize=2)(pack_projection_pair)
    x = torch.randn(5, 33, 128, device="cuda")
    with torch.autocast("cuda", dtype=torch.float16, cache_enabled=False):
        torch.testing.assert_close(packed_swiglu_forward(module, x), module(x))
        module.w_up.weight.zero_()
        module.w_gate.bias.add_(1)
        torch.testing.assert_close(packed_swiglu_forward(module, x), module(x))
        torch.testing.assert_close(packed_swiglu_forward(module, x), module(x))
    assert module._pack_gate_up.cache_info().hits >= 2


@pytest.mark.parametrize("budget", [96, 2000000])
@torch.no_grad()
def test_shared_query_buffer(budget, monkeypatch):
    monkeypatch.setenv("LATTICE_INFER_CHUNK_CELLS", str(budget))
    torch.manual_seed(29)
    dim, readouts, groups = 32, 2, 9
    cols = torch.nn.ModuleList(InducedSelfAttentionBlock(dim, 4, 64, 5, zero_init=False) for _ in range(2))
    rows = torch.nn.ModuleList(MultiheadAttentionBlock(dim, 4, 64, zero_init=False) for _ in range(2))
    model = InterleavedRowInteraction(cols, rows, torch.nn.Parameter(torch.randn(readouts, dim)),
                                     torch.nn.LayerNorm(dim), None, dim, readouts).cuda().eval()
    support = torch.randn(2, groups + readouts, 19, dim, device="cuda").transpose(1, 2)
    query = torch.randn(2, groups + readouts, 17, dim, device="cuda").transpose(1, 2)
    query[0, :, -1].fill_(-100)
    cache = KVCache()
    model.forward_with_cache(support, cache, train_size=19)
    expected = model.forward_with_cache(query.clone(), cache, use_cache=True, store_cache=False)
    saved = query.clone()
    actual = model._buffered_query(query, cache, budget)
    assert torch.equal(actual, expected)
    assert torch.equal(query, saved)
