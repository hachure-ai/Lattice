# Lattice

Lattice is a tabular foundation model that makes predictions in context, without
any fine-tuning. A single frozen model handles both classification and
regression; the scikit-learn style estimators below accept NumPy arrays or
pandas DataFrames, including mixed numeric/categorical features and missing
values.

**Checkpoints:**
[hachure-ai/Lattice](https://huggingface.co/hachure-ai/Lattice) (default) &middot;
[hachure-ai/Lattice-tiny](https://huggingface.co/hachure-ai/Lattice-tiny) (tiny)

## Installation

```bash
git clone https://github.com/hachure-ai/Lattice.git
cd Lattice
pip install -e .
```

Python >= 3.10 and PyTorch >= 2.4 are required (`F.rms_norm` needs 2.4). A CUDA
GPU is strongly recommended; CPU inference works but is much slower.

The optional accelerators are stricter and fall back silently when unmet: the
Triton fast path needs exactly `torch==2.7.1` and `triton==3.3.1`; Flash
Attention 3 needs a Hopper GPU (sm90), CUDA >= 12.3 and a
`flash_attn_interface` built from the `hopper/` sources of FlashAttention.

`pip install -e ".[fast]"` installs the validated Triton stack.

Environment variables that control the accelerators (all optional):

| Variable | Default | Effect |
| --- | --- | --- |
| `LATTICE_FASTPATH` | `auto` | `0`/`off` disables the Triton QK + ICL kernels entirely; `auto` installs them when the version gate above holds. |
| `LATTICE_FASTPATH_KV` | `1` | `0`/`off` keeps a KV-cached fit on the eager encoder. By default a cached fit installs the encoder kernel only: the QK/RoPE rewiring belongs to the uncached path. |
| `LATTICE_FUSED` | `0` | `1` additionally fuses LayerNorm, RMSNorm and SwiGLU. Off by default because the fused kernels need a device self-test at install time. |
| `LATTICE_TEST_CHUNK_CELLS` | `2000000` | Cell budget (`rows x features`) of one cached test forward, the main prediction-time memory knob. |

## Quick start

```python
from lattice import LatticeClassifier, LatticeRegressor

clf = LatticeClassifier()        # downloads cls.ckpt from Hugging Face Hub
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
proba = clf.predict_proba(X_test)

reg = LatticeRegressor()         # downloads reg.ckpt from Hugging Face Hub
reg.fit(X_train, y_train)
pred = reg.predict(X_test)
```

The checkpoints are downloaded once and cached by `huggingface_hub`, pinned to
the `v1.0.1` revision. Pass `model_size="tiny"` to use the smaller
`hachure-ai/Lattice-tiny` checkpoints, or `model_path=".../cls.ckpt"` to load a
local file and avoid network access.

Declare categorical columns before fitting if `X` is a NumPy array:

```python
clf.categorical_indices = [2, 7]
```

More examples live in [`examples/`](examples/).

Pass `device="cpu"` to force CPU inference, `n_estimators=...` to change the
number of ensemble members, or `batch_size=...` to trade speed for memory.
`batch_size` defaults to `8`; `"auto"` picks a batch per call and retries
smaller ones on OOM. The choice also decides where a KV cache lives: a fixed
integer keeps it on the device, while `"auto"` follows `kv_cache_device`
(`"auto"` by default, which promotes the cache to the GPU when it fits a
quarter of the reclaimable memory and keeps it in pinned host memory
otherwise). A resident cache is roughly twice as fast for cached prediction
on small test sets; the host fallback is the safety net for tables whose cache
does not fit.
`clf.decision_function(X)` returns pre-softmax class scores (a signed 1-D score
in the binary case) for attribution methods such as Shapley values.

## Checkpoints

| Size | Task | File | URL |
| --- | --- | --- | --- |
| `default` | Classification | `cls.ckpt` | https://huggingface.co/hachure-ai/Lattice/blob/main/cls.ckpt |
| `default` | Regression | `reg.ckpt` | https://huggingface.co/hachure-ai/Lattice/blob/main/reg.ckpt |
| `tiny` | Classification | `cls.ckpt` | https://huggingface.co/hachure-ai/Lattice-tiny/blob/main/cls.ckpt |
| `tiny` | Regression | `reg.ckpt` | https://huggingface.co/hachure-ai/Lattice-tiny/blob/main/reg.ckpt |

Architecture details are stored inside each checkpoint and need not be
specified.

## Licence and acknowledgements

This project is licensed under the Apache License 2.0. See the root `NOTICE`
and `LICENSE` for attribution, including TabICLv2 and NVIDIA
structured-data-models.
