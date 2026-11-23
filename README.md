# Lattice

Developed by **Hachure-AI & Nanjing University**.

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

Lattice requires Python >= 3.10 and PyTorch >= 2.9.0. Both CPU and CUDA
inference are supported. For GPU inference on Linux, install a CUDA build of
PyTorch and run `pip install -e ".[fast]"` to include Triton. The CUDA
benchmarks used PyTorch 2.14.1+cu130 and Triton 3.8.0.

On first use, the CUDA inference fast path compares Triton residual scores and
scales against native PyTorch on the inference GPU. CUDA builds exposing
`aten::_fused_rms_norm` also qualify for accelerated block input LayerNorm,
head-wise Q/K RMSNorm and SwiGLU. The encoder checks representative LayerNorm
and SwiGLU outputs on these builds and falls back to PyTorch if its checks fail.
Q/K fusion has a separate numerical check when requested. Results are cached
in the process, with encoder checks specific to the GPU and model width. On
the RTX PRO 6000 Blackwell, PyTorch 2.9.0 with Triton 3.5.0 uses the native
encoder after these checks.
On the vectorized path, KV-cached models also pack the SwiGLU gate and up
weights for one projection and reuse a row-major buffer for interleaved blocks.
Flash Attention 3 needs a Hopper GPU (sm90), CUDA >= 12.3 and a
`flash_attn_interface` built from the `hopper/` sources of FlashAttention.

The following environment variables control CUDA inference.

| Variable | Default | Effect |
| --- | --- | --- |
| `LATTICE_FASTPATH` | `auto` | `0`/`off` disables the model fast path. `auto` selects kernels for supported CUDA inference models after the numerical checks. |
| `LATTICE_FASTPATH_KV` | `1` | `0`/`off` keeps a KV-cached fit on the eager encoder. Supported cached interleaved models also use fused Q/K and rotary embeddings under AMP. |
| `LATTICE_FUSED` | `0` | `1` requests additional fusions on the uncached AMP path after a device self-test, including opt-in SwiGLU on the legacy path. The vectorized block input, Q/K and SwiGLU kernels use `LATTICE_FASTPATH`. |
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

When fetching a Hub checkpoint, Lattice also requests `config.json` so the Hub
can count the request. These counts include HTTP HEAD requests and are not
unique users or guaranteed completed checkpoint downloads. Older Lattice
versions, direct checkpoint downloads, and local files may not be counted.

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
