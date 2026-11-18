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
Triton fast path needs exactly `torch==2.7.1` and `triton==3.3.1` with the KV
cache off; Flash Attention 3 needs a Hopper GPU (sm90), CUDA >= 12.3 and a
`flash_attn_interface` built from the `hopper/` sources of FlashAttention.

`pip install -e ".[fast]"` installs the validated Triton stack.

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
