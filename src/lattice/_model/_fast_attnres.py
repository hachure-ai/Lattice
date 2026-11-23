"""Triton kernels for the inference-time ICL attention residuals."""

import os

import torch
import triton
import triton.language as tl

from ._fast_norm214 import HAS_FUSED_RMS_NORM
from .kv_cache import KVCacheEntry


KEYS_ROWS_THRESHOLD = 2048
SCORE_ROWS = 4
PAIR_ROWS = 4

#: Fraction of the currently allocatable GPU memory the fp32 residual bank may
#: use. Fixed member batches do not go through the adaptive batch-size
#: estimator, so the fastpath has to decline a table that would not fit.
BANK_MEMORY_FRACTION = float(os.environ.get("LATTICE_FASTPATH_BANK_FRACTION", "0.5"))


def _bank_fits(rows: int, slots: int, dim: int, device) -> bool:
    """Whether the fp32 residual bank fits in the allocatable GPU memory."""
    needed = rows * slots * dim * 4
    free = torch.cuda.mem_get_info(device)[0]
    cached = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    return needed <= BANK_MEMORY_FRACTION * (free + max(0, cached))


@triton.jit
def _mul_rn(a, b):
    return tl.inline_asm_elementwise("mul.rn.f32 $0, $1, $2;", "=f,f,f", [a, b],
                                     dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _add_rn(a, b):
    return tl.inline_asm_elementwise("add.rn.f32 $0, $1, $2;", "=f,f,f", [a, b],
                                     dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _tree512(acc, R: tl.constexpr, descending: tl.constexpr = HAS_FUSED_RMS_NORM):
    """ATen's tree for `R` independent 512-lane fp32 reduces.

    Reduce.cuh vectorises to dim0 = 512/4 = 128, block_width = 32, so lane t
    owns elements 4t+i with i in 0..3 and walks stride 128; the four accumulators
    are chained. PyTorch 2.7 shuffles adjacent lanes first, while
    PyTorch 2.14 starts with lanes 16 apart.
    """
    a = tl.reshape(acc, (R, 32, 2, 2))
    p, q = tl.split(a)
    a0, a2 = tl.split(p)
    a1, a3 = tl.split(q)
    v = _add_rn(_add_rn(_add_rn(a0, a1), a2), a3)
    if descending:
        # ATen 2.14 combines opposite halves of the warp first.
        for shift in tl.static_range(5):
            idx = tl.broadcast_to((tl.arange(0, 32) ^ (16 >> shift))[None, :], (R, 32))
            v = _add_rn(v, tl.gather(v, idx, axis=1))
        return tl.sum(tl.where(tl.arange(0, 32)[None, :] == 0, v, 0.), axis=1)
    v = tl.reshape(v, (R, 16, 2))
    lo, hi = tl.split(v)
    v = _add_rn(lo, hi)
    v = tl.reshape(v, (R, 8, 2))
    lo, hi = tl.split(v)
    v = _add_rn(lo, hi)
    v = tl.reshape(v, (R, 4, 2))
    lo, hi = tl.split(v)
    v = _add_rn(lo, hi)
    v = tl.reshape(v, (R, 2, 2))
    lo, hi = tl.split(v)
    v = _add_rn(lo, hi)
    v = tl.reshape(v, (R, 1, 2))
    lo, hi = tl.split(v)
    return tl.reshape(_add_rn(lo, hi), (R,))


@triton.jit
def write_slot(src, bank, rstd, slots, eps, idx, D: tl.constexpr,
               tail_w=None, tail_out=None, tail_count=0, HAS_TAIL: tl.constexpr = False,
               N: tl.constexpr = 4, INV: tl.constexpr = 0.001953125):
    """Store one residual source and its RMS scale in a single kernel.

    Replaces `bank[..., idx] = current` plus `square/mean/add/rsqrt/div` with
    one pass: the variance uses `_tree512` so it matches `mean(-1)` bitwise.

    ATen gives the reduce 128 vector slots and `D/128` iterations, so `N` is 4
    for d_model 512 and 2 for 256; `INV` is `1/D`.

    With `HAS_TAIL` it also computes one pending scorer's score for this very
    slot from the values already in registers, so the paired `mix` does not need
    a second pass over the freshly written slot.
    """
    row = tl.program_id(0)
    lane = tl.arange(0, 128)
    sip = src + row * D + lane
    bip = bank + row * slots * D + idx * D + lane

    acc = tl.zeros((128,), tl.float32)
    for k in tl.static_range(0, N):
        v = tl.load(sip + 128 * k).to(tl.float32)
        tl.store(bip + 128 * k, v)
        acc = _add_rn(acc, _mul_rn(v, v))

    var = _tree512(tl.reshape(acc, (1, 128)), 1) * INV      # mean over D
    inv = tl.rsqrt(var + eps)
    tl.store(rstd + row * slots + idx + tl.arange(0, 1), inv)

    if HAS_TAIL:
        s = tl.zeros((128,), tl.float32)
        for k in tl.static_range(0, N):
            v = tl.load(bip + 128 * k)
            s = _add_rn(s, _mul_rn(_mul_rn(v, inv), tl.load(tail_w + lane + 128 * k)))
        tl.store(tail_out + row * tail_count + idx + tl.arange(0, 1),
                 _tree512(tl.reshape(s, (1, 128)), 1))


@triton.jit
def score512(bank, rstd, w, out, slots, count, rows, D: tl.constexpr, R: tl.constexpr,
             N: tl.constexpr = 4):
    """`(keys * w).sum(-1)` without materialising `keys`.

    keys = bank * rstd per source, so the score is
    `((bank * rstd) * w)` reduced along D with ATen's tree. The multiply chain
    keeps its two separate roundings. `R` rows share one program so the tiny
    per-source reductions do not drown in launch/scheduling overhead.
    """
    k = tl.program_id(1)
    bt = tl.program_id(0) * R + tl.arange(0, R)
    valid = bt < rows
    lane = tl.arange(0, 128)
    bip = bank + (bt * slots * D + k * D)[:, None] + lane[None, :]
    r = tl.load(rstd + bt * slots + k, mask=valid, other=0.0)[:, None]

    m = valid[:, None]
    acc = tl.zeros((R, 128), tl.float32)
    for i in tl.static_range(0, N):
        v = _mul_rn(tl.load(bip + 128 * i, mask=m, other=0.0), r)
        acc = _add_rn(acc, _mul_rn(v, tl.load(w + lane + 128 * i)[None, :]))

    tl.store(out + (bt * count + k)[:, None], _tree512(acc, R)[:, None], mask=m)


@triton.jit
def score512_pair(bank, rstd, w0, w1, out0, out1, slots, count, rows,
                  D: tl.constexpr, R: tl.constexpr, N: tl.constexpr = 4):
    """Two score vectors from one pass over the shared bank slots.

    Each `mix` call reads the whole prefix bank, and the pairs of consecutive
    `mix` calls that share a `count` also share slots 0..count-2: only the last
    slot is rewritten between them. One program therefore computes both
    `(keys*w0).sum(-1)` and `(keys*w1).sum(-1)` for slots 0..count-2 and only
    the first one for slot count-1 (the second one is finished by `write_slot`
    once that slot holds its final value).
    """
    k = tl.program_id(1)
    bt = tl.program_id(0) * R + tl.arange(0, R)
    valid = bt < rows
    lane = tl.arange(0, 128)
    bip = bank + (bt * slots * D + k * D)[:, None] + lane[None, :]
    r = tl.load(rstd + bt * slots + k, mask=valid, other=0.0)[:, None]

    m = valid[:, None]
    acc0 = tl.zeros((R, 128), tl.float32)
    acc1 = tl.zeros((R, 128), tl.float32)
    for i in tl.static_range(0, N):
        v = _mul_rn(tl.load(bip + 128 * i, mask=m, other=0.0), r)
        acc0 = _add_rn(acc0, _mul_rn(v, tl.load(w0 + lane + 128 * i)[None, :]))
        acc1 = _add_rn(acc1, _mul_rn(v, tl.load(w1 + lane + 128 * i)[None, :]))

    tl.store(out0 + (bt * count + k)[:, None], _tree512(acc0, R)[:, None], mask=m)
    tl.store(out1 + (bt * count + k)[:, None], _tree512(acc1, R)[:, None],
             mask=m & (k < count - 1))


@triton.jit
def mix_k(bank, p, out, row_stride, K: tl.constexpr, D: tl.constexpr,
          BLOCK: tl.constexpr):
    """`(p.unsqueeze(-1) * bank).sum(-2)` with ATen's exact k-axis tree.

    ATen lays the reduce axis out as 4 strided lanes (lane t takes k = t, t+4,
    t+8, ...), folds each lane sequentially and chains the four lanes in order.
    `mul.rn`/`add.rn` keep every product separately rounded: plain `a + b * c`
    is contracted into an FMA and drifts by 1 ULP.
    """
    pid = tl.program_id(0)
    NBLK: tl.constexpr = D // BLOCK
    row = pid // NBLK
    cols = (pid % NBLK) * BLOCK + tl.arange(0, BLOCK)
    bptr = bank + row * row_stride + cols
    pptr = p + row * K
    l0 = tl.zeros((BLOCK,), tl.float32)
    l1 = tl.zeros((BLOCK,), tl.float32)
    l2 = tl.zeros((BLOCK,), tl.float32)
    l3 = tl.zeros((BLOCK,), tl.float32)
    s0 = False
    s1 = False
    s2 = False
    s3 = False
    for j in range(4):
        k0 = 4 * j
        if k0 < K:
            pr = _mul_rn(tl.load(pptr + k0), tl.load(bptr + k0 * D).to(tl.float32))
            l0 = _add_rn(l0, pr) if s0 else pr
            s0 = True
        k1 = 4 * j + 1
        if k1 < K:
            pr = _mul_rn(tl.load(pptr + k1), tl.load(bptr + k1 * D).to(tl.float32))
            l1 = _add_rn(l1, pr) if s1 else pr
            s1 = True
        k2 = 4 * j + 2
        if k2 < K:
            pr = _mul_rn(tl.load(pptr + k2), tl.load(bptr + k2 * D).to(tl.float32))
            l2 = _add_rn(l2, pr) if s2 else pr
            s2 = True
        k3 = 4 * j + 3
        if k3 < K:
            pr = _mul_rn(tl.load(pptr + k3), tl.load(bptr + k3 * D).to(tl.float32))
            l3 = _add_rn(l3, pr) if s3 else pr
            s3 = True
    res = l0
    if K > 1:
        res = _add_rn(res, l1)
    if K > 2:
        res = _add_rn(res, l2)
    if K > 3:
        res = _add_rn(res, l3)
    tl.store(out + row * D + cols, res)


def fused_mix(slab, p, count, row_stride, dtype):
    B, T, _, D = slab.shape
    out = torch.empty((B, T, D), device=slab.device, dtype=dtype)
    block = 256
    mix_k[(B * T * (D // block),)](slab, p, out, row_stride, count, D, block)
    return out


def _fused_attn_res_stack(self, src, attention):
    """Shared FP32 attention-residual loop of the fused encoder paths.

    ``attention(layer_idx, block, hidden_states)`` returns one layer's attention
    output. The uncached encoder and the KV-cached one differ only in that call,
    so both run this loop instead of repeating the residual bank.
    """
    blocks = self.blocks
    B, T, D = src.shape
    slots = len(blocks) + 2
    rows = B * T

    N, INV = D // 128, 1.0 / D
    fuse = rows > KEYS_ROWS_THRESHOLD
    pair = self.attn_res_block_size == 1
    bank = torch.empty((B, T, slots, D), device=src.device, dtype=torch.float32)
    rstd = torch.empty((B, T, slots), device=src.device, dtype=torch.float32)
    eps = blocks[0].mlp_res.norm.variance_epsilon
    score_weight = {}
    filled = 0
    staged_slot, staged_value = -1, None
    pending = None
    prefix_sum = src

    def write(idx, current, tail=None):
        current = current.contiguous()
        if tail is None:
            write_slot[(rows,)](current, bank, rstd, slots, eps, idx, D, N=N, INV=INV)
        else:
            tw, tout, tcount = tail
            write_slot[(rows,)](current, bank, rstd, slots, eps, idx, D,
                                tw, tout, tcount, HAS_TAIL=True, N=N, INV=INV)

    def stage(idx, current):
        """Write `current` into slot `idx` unless that slot already holds it."""
        nonlocal staged_slot, staged_value
        if staged_slot == idx and staged_value is current:
            return
        tail = None
        if pending is not None and pending[3] - 1 == idx:
            tail = (pending[1], pending[2], pending[3])
        write(idx, current, tail)
        staged_slot, staged_value = idx, current
        if tail is not None:
            pending[4] = True

    def weight(module):
        w = score_weight.get(id(module))
        if w is None:
            w = module.norm.weight.float() * module.proj.weight.squeeze(0).float()
            score_weight[id(module)] = w
        return w

    def mix(module, count, nxt=None):
        """Scores + softmax + weighted sum for one scorer module."""
        nonlocal pending
        w = weight(module)
        scores = None
        if pending is not None:
            pmod, _, ps, pc, pfull = pending
            pending = None
            if pmod is module and pc == count and pfull:
                scores = ps
        if scores is None:
            if pair and nxt is not None:
                scores = torch.empty((B, T, count), device=src.device, dtype=torch.float32)
                w2 = weight(nxt)
                other = torch.empty_like(scores)
                # CUDA grid dimension 1 is limited to 65,535 blocks. Put the
                # row axis in dimension 0, which supports much larger grids.
                score512_pair[(triton.cdiv(rows, PAIR_ROWS), count)](
                    bank, rstd, w, w2, scores, other, slots, count, rows, D, PAIR_ROWS, N)
                pending = [nxt, w2, other, count, False]
            else:
                scores = torch.empty((B, T, count), device=src.device, dtype=torch.float32)
                score512[(triton.cdiv(rows, SCORE_ROWS), count)](
                    bank, rstd, w, scores, slots, count, rows, D, SCORE_ROWS, N)
        probabilities = scores.softmax(dim=-1) if module.use_softmax else scores.exp()
        if fuse:
            output = fused_mix(bank[:, :, :count], probabilities.reshape(-1, count),
                               count, bank.shape[2] * D, src.dtype).view(B, T, D)
        else:
            output = (probabilities.unsqueeze(-1) * bank[:, :, :count]).sum(dim=-2)
        return output.to(src.dtype)

    def next_scorer(layer_idx):
        """The module `mix` runs on right after `blocks[layer_idx].mlp_res`."""
        if layer_idx + 1 < len(blocks):
            return blocks[layer_idx + 1].self_attention_res
        return self.output_attn_res

    for layer_idx, block in enumerate(blocks):
        if block.self_attention_res is None:
            hidden_states = prefix_sum
        else:
            stage(filled, prefix_sum)
            hidden_states = mix(block.self_attention_res, filled + 1)
        if layer_idx % self.attn_res_block_size == 0:
            stage(filled, prefix_sum)
            filled += 1
            prefix_sum = None
        hidden_states = block.norm1(hidden_states)
        attention_output = attention(layer_idx, block, hidden_states)
        attention_output = block.post_attn_norm(attention_output)
        prefix_sum = attention_output if prefix_sum is None else prefix_sum + attention_output
        stage(filled, prefix_sum)
        hidden_states = mix(block.mlp_res, filled + 1, next_scorer(layer_idx))
        mlp_output = block._ff_block(block.norm2(hidden_states))
        prefix_sum = prefix_sum + block.post_ff_norm(mlp_output)
    stage(filled, prefix_sum)
    return mix(self.output_attn_res, filled + 1)


def fast_encoder_forward(self, src, train_size=None, idempotency_loop_counts=None,
                         return_last_attn=False):
    """Reuse FP32 attention-residual statistics across ICL scorers."""
    blocks = self.blocks
    B, T, D = src.shape
    slots = len(blocks) + 2
    rows = B * T
    if (
        not src.is_cuda or D not in (256, 512) or not rows
        or (HAS_FUSED_RMS_NORM and rows < 16)  # ATen uses a wider reduction block for very few rows.
        or slots > 16  # mix_k holds at most sixteen residual slots.
        or rows * slots * D >= 2**31
        or idempotency_loop_counts is not None or return_last_attn
        or not _bank_fits(rows, slots, D, src.device)
    ):
        return type(self).forward(
            self, src, train_size=train_size,
            idempotency_loop_counts=idempotency_loop_counts,
            return_last_attn=return_last_attn,
        )

    def attention(layer_idx, block, hidden_states):
        key_value_states = hidden_states if train_size is None else hidden_states[..., :train_size, :]
        return block._attn_block(hidden_states, key_value_states, key_value_states, rope=self.rope)

    return _fused_attn_res_stack(self, src, attention)


def fast_encoder_forward_with_cache(self, src, icl_cache, train_size=None,
                                    use_cache=False, store_cache=True, *, cache_only=False):
    """Reuse FP32 attention-residual statistics in the KV-cached encoder paths.

    ``store_cache`` publishes one K/V entry per ICL layer from the same pass that
    computes the residuals; ``use_cache`` attends to the stored entries for the
    query rows. ``cache_only`` (a support pass that only fills the cache) keeps
    the eager implementation.
    """
    if use_cache == store_cache:
        raise ValueError("Exactly one of use_cache or store_cache must be True")
    if store_cache and train_size is None:
        raise ValueError("train_size must be provided when store_cache=True")

    blocks = self.blocks
    B, T, D = src.shape
    slots = len(blocks) + 2
    rows = B * T
    if (
        not src.is_cuda or D not in (256, 512) or not rows
        or (HAS_FUSED_RMS_NORM and rows < 16)  # ATen uses a wider reduction block for very few rows.
        or slots > 16  # mix_k holds at most sixteen residual slots.
        or rows * slots * D >= 2**31
        or cache_only
        or not _bank_fits(rows, slots, D, src.device)
    ):
        return type(self).forward_with_cache(
            self, src, icl_cache, train_size=train_size,
            use_cache=use_cache, store_cache=store_cache, cache_only=cache_only,
        )

    def attention(layer_idx, block, hidden_states):
        if use_cache:
            return block._attn_block(hidden_states, cached_kv=icl_cache.kv[layer_idx], rope=self.rope)
        support = hidden_states[..., :train_size, :]
        output, key, value = block._attn_block(
            hidden_states, support, support, rope=self.rope, need_kv=True,
        )
        icl_cache.kv[layer_idx] = KVCacheEntry(key=key, value=value)
        return output

    return _fused_attn_res_stack(self, src, attention)
