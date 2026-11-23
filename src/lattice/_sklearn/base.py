from __future__ import annotations

import os
import sys
import copy
import pickle
import warnings
import contextlib
from importlib import import_module
from pathlib import Path
from typing import Optional
from collections import OrderedDict

import numpy as np
import torch

import sklearn
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from lattice import InferenceConfig


MODEL_IMPLEMENTATIONS = {
    "lattice_fourier_hybrid_attn-res-interleave": ("_model_interleave", "LatticeInterleave"),
}

#: Hugging Face Hub repositories that host the released checkpoints.
CHECKPOINT_REPOS = {
    "default": "hachure-ai/Lattice",
    "tiny": "hachure-ai/Lattice-tiny",
}
#: Released weights revision, pinned for reproducible downloads.
CHECKPOINT_REVISION = "v1.0.1"


def _download_checkpoint(filename: str, model_size: str = "default") -> Path:
    """Fetch a released checkpoint from the Hugging Face Hub (locally cached)."""
    try:
        repo_id = CHECKPOINT_REPOS[model_size]
    except KeyError as exc:
        raise ValueError(
            f"Unknown model_size={model_size!r}; expected one of {sorted(CHECKPOINT_REPOS)}."
        ) from exc
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ValueError(
            "model_path is required when huggingface_hub is not installed. "
            "Install it with `pip install huggingface_hub`, or pass model_path."
        ) from exc
    # The Hub counts requests for config.json. This metadata is optional and
    # must not prevent loading a cached checkpoint when the Hub is unavailable.
    try:
        hf_hub_download(repo_id=repo_id, filename="config.json", revision="main")
    except Exception:
        pass
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=CHECKPOINT_REVISION,
        )
    )


def resolve_model_implementation(model_impl: str) -> tuple[str, str]:
    """Resolve the architecture recorded by a checkpoint."""
    try:
        return MODEL_IMPLEMENTATIONS[model_impl]
    except KeyError as exc:
        # Checkpoints written before the rename carry a vendor prefix on the
        # same architecture tag, so match on the suffix.
        suffix = model_impl.split("_", 1)[-1]
        for key, value in MODEL_IMPLEMENTATIONS.items():
            if key.split("_", 1)[-1] == suffix:
                return value
        supported = ", ".join(sorted(MODEL_IMPLEMENTATIONS))
        raise ValueError(
            f"Unknown checkpoint model_impl={model_impl!r}. Supported inference "
            f"architectures are: {supported}"
        ) from exc


#: Checkpoints already copied to a device, keyed by (path, mtime, device). A
#: single fit moves ~555 tensors over PCIe (~0.07 s); evaluation harnesses fit
#: the same checkpoint once per dataset, so keeping the device copy around turns
#: that cost into a one-off. Bounded to avoid holding many models in VRAM.
_STATE_DICT_CACHE = OrderedDict()
_STATE_DICT_CACHE_MAX = 2

#: Parsed checkpoints, keyed by (path, mtime). Memory-mapped, so entries are
#: cheap to keep; avoids re-deserialising the archive on every fit.
_CKPT_CACHE = OrderedDict()
_CKPT_CACHE_MAX = 2


def _torch_load_checkpoint(path):
    """Read a checkpoint with memory mapping.

    Repeated ``fit`` calls on the same file (one per dataset in an evaluation
    harness) would otherwise re-read and re-deserialise the archive every time
    (~0.1 s for the 150 MB checkpoints used here), so the parsed object is kept
    around. ``mmap=True`` makes the cached object cheap: its tensors point at
    the file instead of holding a private copy of it.
    """
    try:
        key = (str(path), os.path.getmtime(path))
    except (OSError, TypeError):
        key = None
    if key is not None and not os.environ.get("LATTICE_NO_STATE_CACHE"):
        hit = _CKPT_CACHE.get(key)
        if hit is not None:
            _CKPT_CACHE.move_to_end(key)
            return hit
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if key is not None and not os.environ.get("LATTICE_NO_STATE_CACHE"):
        _CKPT_CACHE[key] = checkpoint
        while len(_CKPT_CACHE) > _CKPT_CACHE_MAX:
            _CKPT_CACHE.popitem(last=False)
    return checkpoint


def _device_state_dict(path, state_dict: dict, device: torch.device) -> dict:
    """``state_dict`` on ``device``, reusing a previous copy when possible."""
    if device.type == "cpu":
        return state_dict
    if os.environ.get("LATTICE_NO_STATE_CACHE"):
        return {k: v.to(device, non_blocking=True) for k, v in state_dict.items()}
    try:
        key = (str(path), os.path.getmtime(path), str(device))
    except (OSError, TypeError):
        return {k: v.to(device, non_blocking=True) for k, v in state_dict.items()}

    hit = _STATE_DICT_CACHE.get(key)
    if hit is not None:
        _STATE_DICT_CACHE.move_to_end(key)
        return hit

    moved = {k: v.to(device, non_blocking=True) for k, v in state_dict.items()}
    _STATE_DICT_CACHE[key] = moved
    while len(_STATE_DICT_CACHE) > _STATE_DICT_CACHE_MAX:
        _STATE_DICT_CACHE.popitem(last=False)
    return moved


def _load_state_dict_into(model, state_dict: dict, device: torch.device, path=None):
    """Install a checkpoint into ``model`` with a single host-to-device copy.

    Loading into CPU parameters and then calling ``model.to(device)`` copies the
    whole checkpoint twice, so the tensors are moved once and bound with
    ``assign=True``.
    """
    model.load_state_dict(_device_state_dict(path, state_dict, device), assign=True)
    model.to(device)
    return model


def _check_version_compatibility(metadata: dict) -> None:
    """Warn if saved versions differ from current environment."""
    checks = [
        ("sklearn_version", sklearn.__version__, "scikit-learn"),
        ("torch_version", torch.__version__, "torch"),
        ("numpy_version", np.__version__, "numpy"),
    ]
    for key, current, name in checks:
        saved = metadata.get(key)
        if saved is not None and saved != current:
            warnings.warn(
                f"This file was saved with {name}=={saved} but you are running {name}=={current}. "
                f"This may cause errors or incorrect results.",
                UserWarning,
                stacklevel=3,
            )


class LatticeBaseEstimator(BaseEstimator):
    """Base class for LatticeModel scikit-learn estimators.

    Provides shared functionality for both :class:`LatticeClassifier` and
    :class:`LatticeRegressor`:

    - Device resolution and inference configuration
    - Model persistence (pickle serialization/deserialization)
    - Save/load convenience methods with options for model weights, KV cache,
      and training data

    This class should not be instantiated directly. Use
    :class:`LatticeClassifier` or :class:`LatticeRegressor` instead.
    """

    #: Fitted support caches belong to the estimator, not the shared network.
    supports_external_kv_cache = True

    #: Cell budget (``rows x features``) for one cached test forward pass.
    #: Only used on the KV-cache prediction path, where query rows are
    #: independent and chunking is therefore exact. 2M cells keeps the peak
    #: activation of the col/row/ICL stack around a quarter of what a 40 GB
    #: A100 can hold at bf16. Override via ``LATTICE_TEST_CHUNK_CELLS``.
    _default_test_chunk_cells = 2_000_000

    #: A cached forward batches ``batch_size`` tables of ``n_rows`` rows each;
    #: some kernels index that product with a 16-bit grid dimension, so going
    #: past this raises "CUDA error: invalid configuration argument".
    _max_batched_rows = 65_535

    def _resolve_batch_size(self, n_batches: int, n_rows: int) -> int:
        """Tables per cached forward, capped so the row total stays launchable."""
        batch_size = self.batch_size or n_batches
        if n_rows <= 0:
            return batch_size
        return max(1, min(batch_size, self._max_batched_rows // n_rows))

    def _load_model(self) -> None:
        """Load a model checkpoint from ``model_path``.

        When ``model_path`` is None, the task-specific checkpoint of the
        selected ``model_size`` is downloaded from the Hugging Face Hub and
        cached locally.

        Raises
        ------
        AssertionError
            If the checkpoint doesn't contain the required 'config' or 'state_dict' keys.

        ValueError
            If an explicit ``model_path`` does not point to an existing file.
        """
        if self.model_path is None:
            model_path_ = _download_checkpoint(self._default_checkpoint_file, self.model_size)
        else:
            model_path_ = Path(self.model_path) if isinstance(self.model_path, str) else self.model_path
            if not model_path_.exists():
                raise ValueError(f"Checkpoint not found at '{model_path_}'.")

        checkpoint = _torch_load_checkpoint(model_path_)
        assert "config" in checkpoint, "The checkpoint doesn't contain the model configuration."
        assert "state_dict" in checkpoint, "The checkpoint doesn't contain the model state."

        self.model_path_ = model_path_
        config = dict(checkpoint["config"])
        # Legacy checkpoints may carry a vendor-prefixed table-encoder flag.
        # The release inference path has no such switch, so drop the key.
        for key in [k for k in config if k.endswith("_table_encoder")]:
            config.pop(key)
        self.model_config_ = config
        self.model_impl_ = checkpoint.get("model_impl", "lattice_fourier_hybrid_attn-res-interleave")
        self.model_ = self._make_model()
        _load_state_dict_into(
            self.model_,
            checkpoint["state_dict"],
            getattr(self, "device_", torch.device("cpu")),
            path=model_path_,
        )
        self.model_.eval()

    def _maybe_install_fastpath(self, n_rows: int) -> bool:
        """Select QK + ICL under AMP and ICL alone for FP32.

        The fastpath is independent of how members are batched: ``"auto"`` and
        a fixed integer batch (or ``None``) both use it when the KV cache is
        off. Large tables fall back to the original encoder inside
        :func:`fast_encoder_forward` when the fp32 residual bank would not fit.

        The kernels live on the network, which may be shared with other
        estimators or reused from an earlier fit, so the decision is applied to
        whatever is installed there: a configuration that no longer holds (the
        fit moved device, AMP changed with the table size, or the environment
        switched the fastpath off) is removed instead of being left in place.
        """
        self.fastpath_enabled_ = False
        try:
            from lattice._model import fastpath
        except ImportError:
            return False

        use_amp = getattr(self, "_use_amp_resolved", False)
        # The interleaved row stack honors its fused RoPE flag on both paths.
        # Other cached implementations may pass their rotary embedding directly.
        kv = bool(self.kv_cache) and os.environ.get("LATTICE_FASTPATH_KV", "1").lower() not in {
            "0", "off", "false", "disabled",
        }
        selected = (
            os.environ.get("LATTICE_FASTPATH", "auto").lower() not in {"0", "off", "false", "disabled"}
            and getattr(self, "device_", None) is not None
            and self.device_.type == "cuda"
            # A cached fit has no use for the uncached encoders.
            and (kv or not self.kv_cache)
        )
        if not selected:
            if getattr(self, "model_", None) is not None:
                fastpath.uninstall(self.model_)
            return False

        try:
            with torch.no_grad():
                self.fastpath_enabled_ = fastpath.install(
                    self.model_, fuse_qk=use_amp, allow_kv_cache=kv,
                )
        except RuntimeError:
            return False
        return self.fastpath_enabled_

    def _make_model(self):
        model_impl = getattr(self, "model_impl_", "lattice_fourier_hybrid_attn-res-interleave")
        module, name = resolve_model_implementation(model_impl)
        model_cls = getattr(import_module(f"lattice.{module}"), name)
        # Only this construction uses placeholder tensors. The checkpoint
        # supplies their storage through load_state_dict(assign=True).
        with torch.device("meta"):
            return model_cls(**self.model_config_)

    def _use_lattice_preprocessing(self):
        mode = getattr(self, "preprocessing", "auto")
        if mode not in ("auto", "lattice", "legacy"):
            raise ValueError("preprocessing must be 'auto', 'lattice' or 'legacy'")
        if mode == "lattice" and self.norm_methods is not None:
            raise ValueError("Lattice uses task-specific view schedules. Use preprocessing='legacy' with norm_methods.")
        return mode == "lattice" or (mode == "auto" and self.norm_methods is None and self.ensemble_mode != "legacy")

    def _fit_lattice_ensemble(self, X, y, classification):
        from .lattice_preprocessing.ensemble import LatticeEncoder, LatticeEnsembleGenerator

        if isinstance(self.n_estimators, bool) or not isinstance(self.n_estimators, (int, np.integer)) or self.n_estimators < 1:
            raise ValueError("n_estimators must be a positive integer")
        if self.ensemble_mode != "global":
            raise ValueError("Lattice preprocessing requires ensemble_mode='global'")
        self.preprocessing_ = "lattice"
        self.X_encoder_ = LatticeEncoder(
            categorical_indices=self.categorical_indices,
            # Classification spends the width budget on one-hot and missing
            # indicators; regression keeps the codes-only recipe.
            onehot_width=300 if classification else 0,
            count_card=10 if classification else 50,
            max_missing_indicators=50 if classification else 0,
        )
        X = self.X_encoder_.fit_transform(X)
        self.ensemble_generator_ = LatticeEnsembleGenerator(
            n_estimators=self.n_estimators, classification=classification,
            categorical_mask=self.X_encoder_.categorical_mask_,
            category_route=self.X_encoder_.category_route_, device=self.device_,
            random_state=self.random_state, outlier_threshold=self.outlier_threshold,
            feat_shuffle_method=self.feat_shuffle_method,
            class_shuffle_method=getattr(self, "class_shuffle_method", "shift"),
            tensor_output=True,
            cpu_support_mean=self.kv_cache and self.batch_size == "auto" and self.device_.type == "cuda",
        ).fit(X, y)

    def _fit_ensemble(self, X, y, classification):
        if self.kv_cache_device not in ("auto", "cpu"):
            raise ValueError("kv_cache_device must be 'auto' or 'cpu'.")
        if self._use_lattice_preprocessing():
            return self._fit_lattice_ensemble(X, y, classification)
        self.preprocessing_ = "legacy"
        from .preprocessing import EnsembleGenerator, TransformToNumerical
        from .multiview import DiverseEnsembleGenerator, resolve_norm_methods

        methods = resolve_norm_methods(self.norm_methods)
        if self.ensemble_mode not in ("global", "legacy"):
            raise ValueError("ensemble_mode must be 'global' or 'legacy'.")
        if isinstance(self.n_estimators, bool) or not isinstance(self.n_estimators, (int, np.integer)) or self.n_estimators < 1:
            raise ValueError("n_estimators must be a positive integer.")
        self.X_encoder_ = TransformToNumerical(
            verbose=self.verbose, categorical_indices=getattr(self, "categorical_indices", None)
        )
        X = self.X_encoder_.fit_transform(X)
        options = dict(
            classification=classification, n_estimators=self.n_estimators,
            feat_shuffle_method=self.feat_shuffle_method,
            outlier_threshold=self.outlier_threshold, random_state=self.random_state,
            ensemble_mode=self.ensemble_mode,
        )
        if classification:
            options["class_shuffle_method"] = self.class_shuffle_method
        if not any(method == "asinh" or method.endswith("_svd") for method in methods):
            self.ensemble_generator_ = EnsembleGenerator(
                norm_methods=methods, **options
            )
        else:
            self.ensemble_generator_ = DiverseEnsembleGenerator(
                norm_methods=methods, categorical_mask=self.X_encoder_.categorical_mask_, **options
            )
        self.ensemble_generator_.fit(X, y)

    def _ensemble_data(self, X=None, mode="both", feature_mask=None):
        generator = self.ensemble_generator_
        if getattr(self, "preprocessing_", "legacy") == "lattice":
            generator.to(self.device_)
        if hasattr(generator, "iter_transform"):
            return generator.iter_transform(X, mode, feature_mask)
        return generator.transform(X, mode=mode, feature_mask=feature_mask).items()

    def _ensemble_cat_masks(self, key, feature_mask=None):
        # Base LatticeModel does not consume the mask. Do not change its legacy calls.
        if "fourier" not in getattr(self, "model_impl_", ""):
            return None
        generator = self.ensemble_generator_
        if hasattr(generator, "categorical_masks_"):
            masks = generator.categorical_masks_[key]
            if not masks.any():
                return None
            return masks
        keep = generator.unique_filter_.features_to_keep_
        mask = self.X_encoder_.categorical_mask_[keep]
        permutations = generator.feature_shuffles_[key]
        if feature_mask is not None:
            mask = mask[~np.asarray(feature_mask)[keep]]
            permutations = generator.masked_feature_shuffles_[key]
        return np.stack([mask[permutation] for permutation in permutations])

    def _cat_kwargs(self, masks, start, end):
        if masks is None:
            return {}
        return {"cat_mask": torch.as_tensor(masks[start:end], dtype=torch.bool, device=self.device_)}

    def _test_rows_per_forward(self, n_test: int, n_features: int) -> int:
        """Number of test rows to push through a single cached forward pass.

        With a KV cache the training context lives in the cache, so every test
        row is predicted independently: a query row attends to the cached
        support K/V only (column ISABs treat feature groups as batch
        dimensions, row blocks treat rows as batch dimensions). Splitting the
        query rows therefore leaves the math unchanged (a row's prediction does
        not depend on the values of the other query rows) while keeping the
        peak activation of the col/row/ICL stack proportional to the chunk
        instead of to ``n_test``.

        The outputs are not bit-identical to the unchunked run: matmul
        reduction order depends on the batch shape, which at bf16 shows up as
        ~1e-3 differences in probabilities (measured: 2.7e-3 max over 2k rows
        at ``n_test=1000 vs 2000``); in fp32 it is ~3e-6.

        The chunk size is bounded by a cell budget (``rows x features``), which
        mirrors TabPFN's ``inference_chunk_cells``. Override it with the
        ``LATTICE_TEST_CHUNK_CELLS`` environment variable; set it to ``0`` to
        disable chunking and forward all rows at once.
        """
        budget = os.environ.get("LATTICE_TEST_CHUNK_CELLS")
        try:
            cells = int(budget) if budget else self._default_test_chunk_cells
        except ValueError:
            cells = self._default_test_chunk_cells
        if cells <= 0 or n_features <= 0:
            return max(1, n_test)
        return max(1, min(n_test, cells // n_features))

    def _more_tags(self):
        return dict(non_deterministic=True)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.non_deterministic = True
        return tags

    def _resolve_device(self) -> None:
        """Resolve the target device from the init parameter."""
        if self.device is None:
            self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(self.device, str):
            self.device_ = torch.device(self.device)
        else:
            self.device_ = self.device

    def _resolve_amp_fa3(self) -> tuple:
        """Resolve the ``"auto"`` option for ``use_amp`` and ``use_fa3``.

        Called by ``_build_inference_config`` at ``fit()`` and ``__setstate__``.
        Explicit bool values are returned as-is, while ``"auto"`` triggers a simple
        heuristic based on ``n_samples_in_`` and ``n_features_in_``:

        +--------------------------------------+-------+-------+
        | Regime                               |  AMP  |  FA3  |
        +======================================+=======+=======+
        | Small  (n < 1024 & feat < 60)        |  off  |  off  |
        +--------------------------------------+-------+-------+
        | Medium (above small, n < 10240)      |  on   |  off  |
        +--------------------------------------+-------+-------+
        | Large  (n >= 10240)                  |  on   |  on   |
        +--------------------------------------+-------+-------+

        When ``use_amp=False`` (explicitly disabled by the user) and the data
        is above the small threshold, FA3 is enabled as a fallback accelerator.

        The thresholds are based on preliminary observations and are not rigorously
        tuned. It assumes that the training set is large relative to the test set
        and does not account for KV-cache scenarios. If the auto behaviour is suboptimal
        for your workload, set explicit bool values for ``use_amp`` and ``use_fa3``.

        Returns
        -------
        use_amp : bool
            Whether to enable automatic mixed precision.

        use_fa3 : bool
            Whether to enable Flash Attention 3.
        """
        n_samples = getattr(self, "n_samples_in_", 0)
        n_features = getattr(self, "n_features_in_", 0)
        small_data = n_samples < 1024 and n_features < 60

        # -- AMP --
        if self.use_amp == "auto":
            use_amp = not small_data
        else:
            use_amp = bool(self.use_amp)

        # 环境变量可强制覆盖 auto 决议（评测对账用）：LATTICE_AMP=0/1
        _amp_env = os.environ.get("LATTICE_AMP", "").strip()
        if _amp_env in ("0", "1"):
            use_amp = _amp_env == "1"

        # -- FA3 --
        if self.use_fa3 == "auto":
            if small_data:
                use_fa3 = False
            elif not use_amp:
                # AMP is off and use FA3 as the main accelerator for attention
                use_fa3 = True
            else:
                # AMP is on and FA3 only adds meaningful benefit at large scale
                use_fa3 = n_samples >= 10240
        else:
            use_fa3 = bool(self.use_fa3)

        return use_amp, use_fa3

    def _build_inference_config(self) -> None:
        """Build the inference configuration from init parameters.

        This is called during fit() and also during __setstate__ to reconstruct
        the inference config after loading from a persisted file.
        """
        use_amp, use_fa3 = self._resolve_amp_fa3()
        # 推理时由 estimator 统一包 autocast：部分实现（如 interleave / lattice
        # 行交互）不经过 InferenceManager 的 _run_forward，仅靠 MgrConfig 的
        # use_amp 不会真正生效，这里记录决议结果供 _amp_context 使用。
        self._use_amp_resolved = bool(use_amp) and getattr(self, "device_", None) is not None \
            and self.device_.type == "cuda"

        init_config = {
            "COL_CONFIG": {
                "device": self.device_,
                "use_amp": use_amp,
                "use_fa3": use_fa3,
                "verbose": self.verbose,
                "offload": self.offload_mode,
                "disk_offload_dir": self.disk_offload_dir,
            },
            "ROW_CONFIG": {"device": self.device_, "use_amp": use_amp, "use_fa3": use_fa3, "verbose": self.verbose},
            "ICL_CONFIG": {"device": self.device_, "use_amp": use_amp, "use_fa3": use_fa3, "verbose": self.verbose},
        }
        if self.inference_config is None:
            self.inference_config_ = InferenceConfig()
            self.inference_config_.update_from_dict(init_config)
        elif isinstance(self.inference_config, dict):
            self.inference_config_ = InferenceConfig()
            for key, value in self.inference_config.items():
                if key in init_config:
                    init_config[key].update(value)
            self.inference_config_.update_from_dict(init_config)
        else:
            self.inference_config_ = self.inference_config

    def _amp_context(self):
        """Return the autocast context matching the resolved AMP decision.

        Wrapping the whole model call (instead of relying on each manager's
        ``use_amp`` flag) guarantees mixed precision also applies to inference
        paths implemented outside ``InferenceManager._run_forward``.
        """
        if getattr(self, "_use_amp_resolved", False):
            return torch.autocast("cuda", dtype=torch.float16)
        return contextlib.nullcontext()

    def set_inference_device(self, device: str | torch.device):
        """Move a fitted estimator and refresh AMP without changing cache precision.

        CPU storage during serialization is not CPU inference. Cache conversion
        is deferred until prediction, when the execution device is known.
        """
        self.device = device
        self._cache_residency_failed = False
        self._resolve_device()
        if hasattr(self, "model_"):
            self.model_.to(self.device_)
            self._build_inference_config()
            # The install records the device, so kernels bound to the old one (and
            # their device-local caches) are replaced instead of carried over.
            self._maybe_install_fastpath(getattr(self, "n_samples_in_", 0))
            for name in ("COL_CONFIG", "ROW_CONFIG", "ICL_CONFIG"):
                getattr(self.inference_config_, name).device = self.device_
            self._move_cache_to_device()
        return self

    def _move_cache_to_device(self, *, for_inference: bool = False) -> None:
        """Place cached tensors, preserving dtype during storage and loading.

        Actual CPU/MPS or non-AMP prediction converts the cache to float32.
        Adaptive CUDA prediction chooses resident GPU or pinned CPU storage.
        """
        if not (hasattr(self, "model_kv_cache_") and self.model_kv_cache_ is not None):
            return

        use_amp, _ = self._resolve_amp_fa3()
        needs_upcast = for_inference and (self.device_.type in ("cpu", "mps") or not use_amp)
        upcast_dtype = torch.float32 if needs_upcast else None

        # Warn once if we are actually upcasting reduced-precision tensors
        if upcast_dtype is not None:
            first_cache = next(iter(self.model_kv_cache_.values()))
            cache_dtype = next(iter(first_cache.col_cache.kv.values())).key.dtype
            if cache_dtype != torch.float32:
                if self.device_.type in ("cpu", "mps"):
                    reason = f"{self.device_.type.upper()} inference uses float32 in this estimator"
                else:
                    reason = "AMP is not enabled"
                warnings.warn(
                    f"KV cache contains {cache_dtype} tensors (typically from AMP). "
                    f"Automatically upcasting to float32 because {reason}.",
                    UserWarning,
                    stacklevel=3,
                )

        device_cache = OrderedDict()
        adaptive_cuda = self.batch_size == "auto" and self.device_.type == "cuda"
        for method, cache in self.model_kv_cache_.items():
            cache_device = self.device_
            if adaptive_cuda:
                cache_device = next(cache.tensors()).device if for_inference else "cpu"
            if adaptive_cuda and upcast_dtype is not None and any(
                tensor.is_floating_point() and tensor.dtype != upcast_dtype
                for tensor in cache.tensors()
            ):
                # Reassess residency after conversion; allocating a full FP32
                # copy beside resident FP16 KV can exceed the device budget.
                cache_device = "cpu"
            moved = cache.to(cache_device, dtype=upcast_dtype)
            device_cache[method] = moved
        self.model_kv_cache_ = device_cache
        if for_inference and self.batch_size == "auto" and self.device_.type == "cuda":
            from .cache_residency import place_caches

            place_caches(self)

    def _finish_cache_fit(self):
        self._cache_residency_failed = False
        if self.model_kv_cache_ is None:
            return
        if self.preprocessing_ == "lattice":
            self.ensemble_generator_.release_support()
        if self.batch_size == "auto" and self.device_.type == "cuda":
            from .cache_residency import place_caches

            place_caches(self)

    def __getstate__(self):
        """Customize pickle serialization.

        The fitted state is partitioned into three categories:

        **Always excluded (reconstructed on load by ``__setstate__``):**

        - ``model_`` — The nn.Module is never fine-tuned, so its weights are
          identical to the checkpoint and can be reloaded. Excluding it avoids
          saving 10s-100s MB of redundant data.
        - ``device_`` — Hardware-specific (e.g. ``cuda:0``); meaningless on a
          different machine. Re-derived from the ``device`` init parameter.
        - ``inference_config_`` — Contains ``MgrConfig`` objects with device
          references. Rebuilt from init parameters.
        - ``model_path_`` — Absolute filesystem path to the checkpoint; almost
          certainly invalid on another machine.

        **Optionally included (controlled by ``save()`` flags):**

        - Model weights — When ``save_model_weights=True``, the state dict is
          saved on CPU, making the file self-contained (no checkpoint needed).
        - KV cache — Saved by default (moved to CPU for device portability).
          Excluding it yields a smaller file but requires rebuilding the cache.
        - Training data — Saved by default. Can be excluded when KV cache is
          present, which is useful for data privacy since the KV cache encodes
          the training context and enables prediction without raw data.

        **Always included:**

        - ``ensemble_generator_``, ``X_encoder_``, ``y_encoder_``/``y_scaler_``,
          ``classes_``, ``n_classes_``, ``n_features_in_``, and all init
          parameters — required for prediction and sklearn compatibility.
        - Version metadata — embedded for compatibility checking on load.
        """
        state = self.__dict__.copy()

        # Read save options set by save(), or use defaults for direct pickle
        save_model_weights = state.pop("_save_model_weights", False)
        save_kv_cache = state.pop("_save_kv_cache", True)
        save_training_data = state.pop("_save_training_data", True)

        # Always exclude device-specific and reconstructable attributes
        state.pop("device_", None)
        state.pop("inference_config_", None)
        state.pop("model_path_", None)
        state.pop("_cache_residency_failed", None)

        # Handle model weights
        if save_model_weights and hasattr(self, "model_"):
            # Save state dict (not nn.Module itself)
            state["_model_state_dict"] = {k: v.cpu() for k, v in self.model_.state_dict().items()}
            # model_config_ stays in state
        else:
            state.pop("model_config_", None)
        state.pop("model_", None)

        # Handle KV cache
        if save_kv_cache and state.get("model_kv_cache_") is not None:
            cpu_cache = OrderedDict()
            for method, cache in state["model_kv_cache_"].items():
                cpu_cache[method] = cache.to("cpu")
            state["model_kv_cache_"] = cpu_cache

        # Handle training data
        if not save_training_data and "ensemble_generator_" in state:
            eg = copy.deepcopy(state["ensemble_generator_"])
            eg.X_ = None
            eg.y_ = None
            for prep in eg.preprocessors_.values():
                prep.X_transformed_ = None
                if hasattr(prep, "pipeline_"):
                    prep.pipeline_.X_transformed_ = None
            if hasattr(eg, "numeric_support_"):
                eg.numeric_support_ = {}
            state["ensemble_generator_"] = eg

        # Version metadata
        state["_persistence_metadata"] = {
            "sklearn_version": sklearn.__version__,
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "python_version": sys.version,
            "saved_model_weights": save_model_weights,
            "saved_kv_cache": save_kv_cache,
            "saved_training_data": save_training_data,
        }

        return state

    def __setstate__(self, state):
        """Customize pickle deserialization.

        Reconstructs device-specific attributes and reloads the model.
        """

        metadata = state.pop("_persistence_metadata", None)
        model_state_dict = state.pop("_model_state_dict", None)
        old_mode = state.pop("multiview", "default")
        if old_mode != "default":
            from .multiview import NUMERIC_METHODS

            state["norm_methods"] = [
                method + suffix for method in NUMERIC_METHODS
                for suffix in (("", "_svd") if old_mode == "diverse_svd" else ("",))
            ]
        state.setdefault("ensemble_mode", "legacy")
        state.setdefault("categorical_indices", None)
        state.setdefault("preprocessing", "legacy")
        state.setdefault("kv_cache_device", "cpu")

        # Restore instance state
        self.__dict__.update(state)

        # Check version compatibility
        if metadata:
            _check_version_compatibility(metadata)

        if "n_features_in_" not in state:
            # The serialized estimator was not fitted. There is no device_,
            # model_ or KV cache to restore.
            return

        # Resolve device
        self._resolve_device()

        # Reload or reconstruct model
        if model_state_dict is not None and hasattr(self, "model_config_"):
            # Model weights were saved and reconstruct without checkpoint
            self.model_ = self._make_model()
            _load_state_dict_into(self.model_, model_state_dict, self.device_)
            self.model_.eval()
        else:
            # Reload from checkpoint
            self._load_model()
        self.model_.to(self.device_)

        # Reconstruct inference config
        self._build_inference_config()
        self._maybe_install_fastpath(getattr(self, "n_samples_in_", 0))

        # Move KV cache to device, auto-upcasting if needed
        self._move_cache_to_device()

    def save(
        self,
        path: str | Path,
        save_model_weights: bool = False,
        save_training_data: bool = True,
        save_kv_cache: bool = True,
    ) -> None:
        """Save the fitted estimator to a file.

        The saved file can be loaded back using the class's ``load()`` method,
        ``pickle.load()``, or ``joblib.load()``.

        Parameters
        ----------
        path : str or Path
            File path to save to.

        save_model_weights : bool, default=False
            If True, include the pretrained LatticeModel model weights in the file,
            making it fully self-contained. If False (default), model weights
            are reloaded from the checkpoint on load, resulting in a smaller file.

        save_training_data : bool, default=True
            If True (default), include the training data used for in-context
            learning. If False, training data is excluded for a smaller file
            and better data privacy. This is only allowed when ``save_kv_cache``
            is True and a KV cache exists, since predictions require either
            cached KV projections or the original training data.

        save_kv_cache : bool, default=True
            If True (default) and KV cache exists, include the cached key-value
            projections. If False, the KV cache is excluded, resulting in a
            smaller file but requiring a rebuild for fast inference.

        Raises
        ------
        NotFittedError
            If the estimator has not been fitted.

        ValueError
            If training data is excluded without a KV cache to fall back on.
        """
        check_is_fitted(self)

        has_kv_cache = hasattr(self, "model_kv_cache_") and self.model_kv_cache_ is not None
        if not save_training_data and not (save_kv_cache and has_kv_cache):
            raise ValueError(
                "Cannot exclude training data when KV cache is not available or not being saved. "
                "Either set save_training_data=True, or set kv_cache=True during init and save_kv_cache=True."
            )

        # Set temporary flags for __getstate__
        self._save_model_weights = save_model_weights
        self._save_kv_cache = save_kv_cache
        self._save_training_data = save_training_data
        try:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump(self, f, protocol=5)
        finally:
            del self._save_model_weights
            del self._save_kv_cache
            del self._save_training_data

    @classmethod
    def load(cls, path: str | Path, device: Optional[str | torch.device] = None) -> LatticeBaseEstimator:
        """Load a fitted estimator from a file.

        Parameters
        ----------
        path : str or Path
            File path to load from.

        device : str, torch.device, or None, default=None
            Device to use after loading. If None, the device is determined
            by the estimator's ``device`` init parameter (which defaults to
            auto-detection: CUDA if available, else CPU).

        Returns
        -------
        estimator : LatticeBaseEstimator
            The loaded fitted estimator.
        """
        with open(path, "rb") as f:
            obj = pickle.load(f)

        if device is not None:
            obj.set_inference_device(device)

        return obj

    @staticmethod
    def softmax(x: np.ndarray, axis: int = -1, temperature: float = 0.9) -> np.ndarray:
        """Compute softmax values with temperature scaling using NumPy.

        Computes :math:`\text{softmax}(x / \tau)` where :math:`\tau` is the
        temperature parameter.

        Parameters
        ----------
        x : ndarray
            Input array of logits.

        axis : int, default=-1
            Axis along which to compute softmax.

        temperature : float, default=0.9
            Temperature scaling parameter :math:`\tau`.

        Returns
        -------
        ndarray
            Softmax probabilities along the specified axis, with the same shape
            as the input.
        """
        x = x / temperature
        # Subtract max for numerical stability
        x_max = np.max(x, axis=axis, keepdims=True)
        e_x = np.exp(x - x_max)
        # Compute softmax
        return e_x / np.sum(e_x, axis=axis, keepdims=True)
