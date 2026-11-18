# Adapted from TabICL v2 (https://github.com/soda-inria/tabicl).
# Copyright (c) 2025 Soda team @ Inria. BSD 3-Clause License; see NOTICE and
# src/lattice/_model/LICENSE.

from __future__ import annotations

import os
import warnings
from pathlib import Path
import multiprocessing as mp
from collections import OrderedDict
from typing import Optional, List, Dict

import numpy as np
import torch

from sklearn.base import ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import check_classification_targets
from sklearn.preprocessing import LabelEncoder

from .base import LatticeBaseEstimator
from .sklearn_utils import validate_data, _num_samples

from lattice import InferenceConfig
from lattice._model.kv_cache import LatticeCache


class LatticeClassifier(ClassifierMixin, LatticeBaseEstimator):
    """Tabular In-Context Learning (LatticeModel) Classifier with scikit-learn interface.

    This classifier applies LatticeModel to tabular data classification, using an ensemble
    of transformed dataset views to improve predictions. The ensemble members are
    created by applying different normalization methods, feature permutations,
    and class label shifts.

    Parameters
    ----------
    n_estimators : int, default=8
        Number of estimators for ensemble predictions.

    ensemble_mode : {"global", "legacy"}, default="global"
        Generate permutation pairs globally before assigning preprocessing
        branches. "legacy" restores the previous cross-branch reuse policy.

    categorical_indices : list[int] or None, default=None
        Categorical column positions in the original input. If None, DataFrame
        dtypes determine categories; numerical arrays are treated as numerical.
        Indices are remapped through encoding, filtering and view permutations.

    norm_methods : str or list[str] or None, default=None
        Normalization methods to apply:

        - 'none': Standardization and outlier handling only
        - 'power': Yeo-Johnson power transform
        - 'quantile': Transform features to an approximately normal distribution.
        - 'quantile_rtdl': Quantile transform that adds noise to training data before fitting.
        - 'robust': Scale using median and quantiles
        - 'asinh': Median/IQR scaling followed by inverse hyperbolic sine
        - '<method>_svd': Append support-fitted SVD columns to that method

        Can be a single string or a list of methods to use across ensemble members.
        With preprocessing="auto", None selects the Lattice recipe. Explicit methods
        select the legacy recipe (whose None default is ["none", "power"]). An asinh or SVD
        branch enables type-aware preprocessing for all branches (numeric
        transforms exclude categorical columns); otherwise the original
        encoded-column preprocessing and feature_mask support are retained.

    feat_shuffle_method : str, default='latin'
        Feature permutation strategy:

        - 'none': No shuffling and preserve original feature order
        - 'shift': Circular shifting of feature columns
        - 'random': Random permutation of features
        - 'latin': Latin square patterns for systematic feature permutations

    class_shuffle_method : str, default='shift'
        Class label permutation strategy:

        - 'none': No shuffling and preserve original class labels
        - 'shift': Circular shifting of class labels
        - 'random': Random permutation of class labels
        - 'latin': Latin square patterns for systematic class permutations

    outlier_threshold : float, default=4.0
        Z-score threshold for outlier detection and clipping. Values with
        :math:`|z| > \text{threshold}` are considered outliers.

    softmax_temperature : float or None, default=None
        None uses 1.0 for Lattice preprocessing and 0.9 for the legacy recipe.
        Temperature parameter :math:`\tau` for the softmax function, applied as
        :math:`\text{softmax}(x / \tau)`. Lower values make predictions more
        confident, higher values make them more conservative.

    average_logits : bool, default=True
        Whether to average the logits (True) or probabilities (False) of ensemble members.
        Averaging logits often produces better calibrated probabilities.

    support_many_classes : bool, default=True
        Whether to enable many-class support which performs mixed-radix ensembling during
        column-wise embedding and hierarchical classification during in-context learning.
        Required when the number of classes exceeds the model's max_classes limit.

    batch_size : int, "auto" or None, default=8
        "auto" selects member batches from available device memory, retries
        smaller batches on OOM, and stages native KV from CPU per batch.
        Batch size for inference. If None, all ensemble members are processed in a single batch.
        Adjust this parameter based on available memory. Lower values use less memory but may
        be slower.

    kv_cache : bool or str, default=False
        Controls caching of training data computations to speed up subsequent
        ``predict_proba``/``predict`` calls. The cache is built during ``fit()``.
        Set False for hierarchical classification beyond the model class limit.

        - False: No caching.
        - True or "kv": Cache key-value projections from both column embedding
          and ICL transformer layers. Fast inference but memory-heavy for large
          training sets.
        - "repr": Cache column embedding KV projections and row interaction outputs
          (representations). Uses ~24x less memory than "kv" for the ICL part,
          at the cost of re-running the ICL transformer at predict time.

        The cache retains whatever dtype the model produced during ``fit()``
        (float16 when AMP is active, float32 otherwise). If the cache is later
        loaded or stored on CPU, its dtype is preserved. Actual CPU prediction
        or CUDA prediction without AMP upcasts it to float32 before execution.

    kv_cache_device : {"auto", "cpu"}, default="cpu"
        With ``batch_size="auto"`` on CUDA, keep KV in pinned CPU memory and
        asynchronously stage member batches. Opt into ``"auto"`` to keep KV
        on GPU when it fits the residency budget. Loading defers promotion to predict.

    model_path : str | Path, optional
        Path to a local checkpoint file. If omitted, ``cls.ckpt`` is downloaded
        from the Hugging Face Hub (``hachure-ai/Lattice``) on first use and
        cached locally. Pass a local path for offline use.

    model_size : {"default", "tiny"}, default="default"
        Released checkpoint repository to download when ``model_path`` is None.
        ``"default"`` uses ``hachure-ai/Lattice``; ``"tiny"`` uses the smaller
        ``hachure-ai/Lattice-tiny``. Ignored when ``model_path`` is given.

    device : Optional[str or torch.device], default=None
        Device to use for inference. If None, automatically selects CUDA if
        available, otherwise CPU. Can be specified as a string (``'cuda'``,
        ``'cpu'``, ``'mps'``) or a ``torch.device`` object. MPS (Apple Silicon
        GPU) is supported but must be explicitly requested.

    use_amp : bool or "auto", default="auto"
        Controls automatic mixed precision (AMP) for inference.
        - True / False: force on / off.
        - "auto": Automatically enable AMP based on input data size using the following heuristic:

            +--------------------------------------+-------+-------+
            | Regime                               |  AMP  |  FA3  |
            +======================================+=======+=======+
            | Small  (n < 1024 & feat < 60)        |  off  |  off  |
            +--------------------------------------+-------+-------+
            | Medium (above small, n < 10240)      |  on   |  off  |
            +--------------------------------------+-------+-------+
            | Large  (n >= 10240)                  |  on   |  on   |
            +--------------------------------------+-------+-------+

            The above heuristic is based on the observation that AMP can introduce overhead that outweighs
            its benefits for small inputs. In addition, it assumes that the training set is large relative to
            the test set and does not account for KV-cache scenarios. If it is suboptimal for your workload,
            set it explicitly.

    use_fa3 : bool or "auto", default="auto"
        Whether to use Flash Attention 3 that can speed up inference for large datasets on NVIDIA Hopper
        GPUs like H100. Only effective when FA3 is installed.
        - True / False: force on / off.
        - "auto": Automatically enable FA3 based on input data size using a simple heuristic (see above).

    offload_mode : str or bool, default='auto'
        Controls where column-wise embedding outputs are stored during inference.
        Column-wise embedding produces a large tensor of shape
        (batch_size, n_rows, n_columns, embed_dim) which is the main memory bottleneck.
        Available options:
        - ``'auto'``: Automatically choose based on available memory (default).
        - ``'gpu'`` or ``False``: Keep on GPU. Fastest but limited by VRAM.
        - ``'cpu'`` or ``True``: Offload to CPU memory.
        - ``'disk'``: Offload to memory-mapped files (requires ``disk_offload_dir``).

        It only affects column-wise embedding (COL_CONFIG). For finer-grained control
        over all components, use ``inference_config``.

    disk_offload_dir : Optional[str], default=None
        Directory for memory-mapped files used when ``offload_mode='disk'`` or when
        ``offload_mode='auto'`` falls back to disk offloading.
        It only affects column-wise embedding (COL_CONFIG). For finer-grained control
        over all components, use ``inference_config``.

    random_state : int or None, default=42
        Random seed for reproducibility of ensemble generation, affecting feature
        shuffling and other randomized operations.

    n_jobs : int or None, default=None
        Number of threads to use for PyTorch in case the model is run on CPU.
        None means using the PyTorch default, which is the number of physical CPU cores.
        Negative numbers mean that :math:`\\max(1, n_{\\text{logical\\_cores}} + 1 + \\text{n\\_jobs})`
        threads will be used. In particular, ``n_jobs=-1`` means that all logical cores
        will be used.

    verbose : bool, default=False
        Whether to print detailed information during inference.

    inference_config : Optional[InferenceConfig | Dict[str, Dict[str, Any]]], default=None
        Configuration for inference settings. This parameter provides fine-grained control
        over the three transformers in LatticeModel (column-wise, row-wise, and in-context learning).

        WARNING: This parameter should only be used by advanced users who understand the internal
        architecture of LatticeModel and need precise control over inference.

        When None (default):
            - A new InferenceConfig object is created with default settings
            - The ``device``, ``use_amp``, ``use_fa3``, ``offload_mode``, ``disk_offload_dir``, and ``verbose``
              parameters from the class initialization are applied to the relevant components

        When Dict with allowed top-level keys "COL_CONFIG", "ROW_CONFIG", "ICL_CONFIG":
            - A new InferenceConfig object is created with default settings
            - Any values explicitly specified in the dictionary will override default defaults
            - ``device``, ``use_amp``, ``use_fa3``, ``offload_mode``, ``disk_offload_dir``, and ``verbose``
              from the class initialization are used if they are not specified in the dictionary

        When InferenceConfig:
            - The provided InferenceConfig object is used directly without modification
            - ``device``, ``use_amp``, ``use_fa3``, ``offload_mode``, ``disk_offload_dir``, and ``verbose``
              from the class initialization are ignored
            - All settings must be explicitly defined in the provided InferenceConfig object

    preprocessing : {"auto", "lattice", "legacy"}, default="auto"
        Lattice-style support-fitted feature processing on the model's device.
        "auto" selects Lattice when norm_methods is None and ensemble_mode is
        "global"; explicit norm_methods retain legacy preprocessing. "legacy"
        with norm_methods=None restores the previous none/power default.
        Lattice cycles through none, power, robust, and rank views. Its default
        eight-member Latin shuffle reduces repeated column pairs when possible.
        It uses category count features, numerical sign flips, and at most 500
        features per view.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Class labels known to the classifier.

    n_classes_ : int
        Number of classes in the training data.

    n_features_in_ : int
        Number of features in the training data.

    n_samples_in_ : int
        Number of samples in the training data.

    feature_names_in_ : ndarray of shape ``(n_features_in_,)`` or None
        Feature names seen during ``fit``. Only set when the input ``X`` has
        feature names (e.g., a pandas DataFrame with string column names).

    X_encoder_ : TransformToNumerical
        Encoder for transforming input features to numerical values.

    y_encoder_ : LabelEncoder
        Encoder for transforming class labels to integers and back.

    ensemble_generator_ : EnsembleGenerator
        Fitted ensemble generator that creates multiple dataset views.

    model_ : LatticeModel
        The loaded LatticeModel model used for predictions.

    model_path_ : str
        Path to the loaded checkpoint file.

    model_config_ : dict
        Configuration dictionary from the loaded checkpoint.

    device_ : torch.device
        The device where the model is loaded and computations are performed.

    inference_config_ : InferenceConfig
        The inference configuration.

    cache_mode_ : str or None
        The resolved caching mode, set during ``fit()`` based on the ``kv_cache``
        init parameter. One of ``"kv"``, ``"repr"``, or ``None`` (no caching).

    model_kv_cache_ : OrderedDict[str, LatticeCache] or None
        Pre-computed KV caches for training data, keyed by normalization method.
        Created during ``fit()`` when ``kv_cache`` is enabled. When set,
        ``predict_proba()`` reuses the cached key-value projections instead of
        re-processing training data, enabling faster inference on multiple test sets.
    """

    _default_checkpoint_file = "cls.ckpt"

    def __init__(
        self,
        n_estimators: int = 8,
        norm_methods: Optional[str | List[str]] = None,
        feat_shuffle_method: str = "latin",
        class_shuffle_method: str = "shift",
        outlier_threshold: float = 4.0,
        softmax_temperature: Optional[float] = None,
        average_logits: bool = True,
        support_many_classes: bool = True,
        batch_size: Optional[int | str] = 8,
        kv_cache: bool | str = False,
        model_path: Optional[str | Path] = None,
        model_size: str = "default",
        device: Optional[str | torch.device] = None,
        use_amp: bool | str = "auto",
        use_fa3: bool | str = "auto",
        offload_mode: str | bool = "auto",
        disk_offload_dir: Optional[str] = None,
        random_state: int | None = 42,
        n_jobs: Optional[int] = None,
        verbose: bool = False,
        inference_config: Optional[InferenceConfig | Dict] = None,
        ensemble_mode: str = "global",
        categorical_indices: Optional[List[int]] = None,
        preprocessing: str = "auto",
        kv_cache_device: str = "cpu",
    ):
        self.n_estimators = n_estimators
        self.norm_methods = norm_methods
        self.feat_shuffle_method = feat_shuffle_method
        self.class_shuffle_method = class_shuffle_method
        self.outlier_threshold = outlier_threshold
        self.softmax_temperature = softmax_temperature
        self.average_logits = average_logits
        self.support_many_classes = support_many_classes
        self.batch_size = batch_size
        # LATTICE_KV_CACHE=1 enables the KV cache for harnesses that
        # cannot pass the constructor argument through their estimator cloning.
        self.kv_cache = kv_cache or os.environ.get("LATTICE_KV_CACHE", "0") == "1"
        self.model_path = model_path
        self.model_size = model_size
        self.device = device
        self.use_amp = use_amp
        self.use_fa3 = use_fa3
        self.offload_mode = offload_mode
        self.disk_offload_dir = disk_offload_dir
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.inference_config = inference_config
        self.ensemble_mode = ensemble_mode
        self.categorical_indices = categorical_indices
        self.preprocessing = preprocessing
        self.kv_cache_device = kv_cache_device

    def fit(self, X: np.ndarray, y: np.ndarray) -> LatticeClassifier:
        """Fit the classifier to training data.

        Prepares the model for prediction by:

        1. Encoding class labels using LabelEncoder
        2. Converting input features to numerical values
        3. Fitting the ensemble generator to create transformed dataset views
        4. Loading the pre-trained LatticeModel model
        5. Optionally pre-computing KV caches for training data to speed up inference
           (controlled by the ``kv_cache`` init parameter)

        The model itself is not trained on the data; it uses in-context learning
        at inference time.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input data.

        y : array-like of shape (n_samples,)
            Training target labels.

        Returns
        -------
        self : LatticeClassifier
            Fitted classifier instance.

        Raises
        ------
        ValueError
            If the number of classes exceeds the model's maximum supported classes
            and many-class support is disabled.
        """

        if y is None:
            raise ValueError("This classifier requires y to be passed, but the target y is None.")

        X, y = validate_data(self, X, y, dtype=None, skip_check_array=True)
        check_classification_targets(y)

        # Device setup
        self._resolve_device()

        # Inference configuration
        self.n_samples_in_ = _num_samples(X)
        self._build_inference_config()

        # Load the pre-trained LatticeModel model
        self._load_model()
        self.model_.to(self.device_)
        self._maybe_install_fastpath(self.n_samples_in_)

        # Encode class labels
        self.y_encoder_ = LabelEncoder()
        y = self.y_encoder_.fit_transform(y)
        self.classes_ = self.y_encoder_.classes_
        self.n_classes_ = len(self.y_encoder_.classes_)

        if self.n_classes_ > self.model_.max_classes:
            if self.kv_cache:
                raise ValueError(
                    f"KV caching is not supported when the number of classes ({self.n_classes_}) exceeds the max number "
                    f"of classes ({self.model_.max_classes}) natively supported by the model."
                )

            if not self.support_many_classes:
                raise ValueError(
                    f"The number of classes ({self.n_classes_}) exceeds the max number of classes ({self.model_.max_classes}) "
                    f"natively supported by the model. Consider enabling many-class support which performs mixed-radix "
                    f"ensembling during column-wise embedding and hierarchical classification during in-context learning."
                )

            if self.verbose:
                print(
                    f"The number of classes ({self.n_classes_}) exceeds the max number of classes ({self.model_.max_classes}) "
                    f"natively supported by the model. Therefore, many-class strategy is enabled to perform mixed-radix "
                    f"ensembling during column-wise embedding and hierarchical classification during in-context learning."
                )

        self.softmax_temperature_ = self.softmax_temperature
        if self.softmax_temperature_ is None:
            self.softmax_temperature_ = 1.0 if self._use_lattice_preprocessing() else 0.9
        self.batching_info_ = []
        self._fit_ensemble(X, y, classification=True)

        self.model_kv_cache_ = None
        if self.kv_cache:
            if self.kv_cache is True or self.kv_cache == "kv":
                self.cache_mode_ = "kv"
            elif self.kv_cache == "repr":
                self.cache_mode_ = "repr"
            else:
                raise ValueError(f"Invalid kv_cache value '{self.kv_cache}'. Expected False, True, 'kv', or 'repr'.")
            self._build_kv_cache()
            self._finish_cache_fit()

        return self

    def _build_kv_cache(self) -> None:
        """Pre-compute KV caches for training data across all ensemble batches."""

        if self.batch_size == "auto":
            from .adaptive_batching import build_cache

            return build_cache(self)

        # X=None is required in transform() even though it is the default value
        # because sklearn's _SetOutputMixin wraps transform() with a signature
        # that enforces X as a positional argument.
        train_data = self._ensemble_data(mode="train")
        self.model_kv_cache_ = OrderedDict()

        for norm_method, (Xs, ys) in train_data:
            cat_masks = self._ensemble_cat_masks(norm_method)
            batch_size = self._resolve_batch_size(Xs.shape[0], Xs.shape[1])
            n_batches = int(np.ceil(Xs.shape[0] / batch_size))
            Xs_split = torch.tensor_split(Xs, n_batches) if isinstance(Xs, torch.Tensor) else np.array_split(Xs, n_batches)
            ys_split = torch.tensor_split(ys, n_batches) if isinstance(ys, torch.Tensor) else np.array_split(ys, n_batches)

            caches = []
            offset = 0
            oom = False
            for X_batch, y_batch in zip(Xs_split, ys_split):
                end = offset + X_batch.shape[0]
                cat_kwargs = self._cat_kwargs(cat_masks, offset, end)
                offset = end
                X_batch = torch.as_tensor(X_batch, dtype=torch.float32, device=self.device_)
                y_batch = torch.as_tensor(y_batch, dtype=torch.float32, device=self.device_)
                try:
                    with torch.no_grad(), self._amp_context():
                        self.model_.forward_with_cache(
                            X_train=X_batch,
                            y_train=y_batch,
                            use_cache=False,
                            store_cache=True,
                            cache_mode=self.cache_mode_,
                            inference_config=self.inference_config_,
                            **cat_kwargs,
                        )
                except torch.cuda.OutOfMemoryError:
                    oom = True
                    break
                caches.append(self.model_._cache)
                self.model_.clear_cache()

            if oom:
                # 大表缓存装不下时回退到无缓存推理：结果数值等价（差 ~1e-4），
                # 只是失去 predict 加速，避免整份评测失败。
                self.model_.clear_cache()
                torch.cuda.empty_cache()
                warnings.warn(
                    "KV cache build ran out of GPU memory; falling back to uncached inference.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self.model_kv_cache_ = None
                return

            # Merge all batch caches into a single cache
            self.model_kv_cache_[norm_method] = LatticeCache.concat(caches)

    def _batch_forward(
        self, Xs: np.ndarray, ys: np.ndarray, feature_shuffles: Optional[np.ndarray] = None,
        cat_masks: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Process model forward passes in batches to manage memory efficiently.

        This method handles the batched inference through the LatticeModel model,
        dividing the ensemble members into smaller batches to avoid out-of-memory errors.

        Parameters
        ----------
        Xs : np.ndarray
            Input features of shape ``(n_datasets, n_samples, n_features)``, where
            ``n_datasets`` is the number of ensemble members.

        ys : np.ndarray
            Training labels of shape ``(n_datasets, train_size)``, where ``train_size``
            is the number of samples used for in-context learning.

        feature_shuffles : list or None, default=None
            Lists of feature shuffle patterns to be applied to each ensemble member.
            If None, no feature shuffling is applied.

        Returns
        -------
        np.ndarray
            Model outputs (logits or probabilities) of shape
            ``(n_datasets, test_size, n_classes)`` where
            ``test_size = n_samples - train_size``.
        """

        if self.batch_size == "auto":
            from .adaptive_batching import predict

            return predict(self, Xs, ys, feature_shuffles, cat_masks)

        batch_size = self._resolve_batch_size(Xs.shape[0], Xs.shape[1])
        n_batches = int(np.ceil(Xs.shape[0] / batch_size))
        Xs = torch.tensor_split(Xs, n_batches) if isinstance(Xs, torch.Tensor) else np.array_split(Xs, n_batches)
        ys = torch.tensor_split(ys, n_batches) if isinstance(ys, torch.Tensor) else np.array_split(ys, n_batches)
        if feature_shuffles is None:
            feature_shuffles = [None] * n_batches
        else:
            feature_shuffles = np.array_split(feature_shuffles, n_batches)

        outputs = []
        offset = 0
        for X_batch, y_batch, shuffle_batch in zip(Xs, ys, feature_shuffles):
            end = offset + X_batch.shape[0]
            cat_kwargs = self._cat_kwargs(cat_masks, offset, end)
            offset = end
            X_batch = torch.as_tensor(X_batch, dtype=torch.float32, device=self.device_)
            y_batch = torch.as_tensor(y_batch, dtype=torch.float32, device=self.device_)
            if shuffle_batch is not None:
                shuffle_batch = shuffle_batch.tolist()

            with torch.no_grad(), self._amp_context():
                out = self.model_(
                    X=X_batch,
                    y_train=y_batch,
                    feature_shuffles=shuffle_batch,
                    return_logits=True if self.average_logits else False,
                    softmax_temperature=getattr(self, "softmax_temperature_", self.softmax_temperature),
                    inference_config=self.inference_config_,
                    **cat_kwargs,
                )
            outputs.append(out.float().cpu().numpy())

        return np.concatenate(outputs, axis=0)

    def _batch_forward_with_cache(self, Xs: np.ndarray, kv_cache: LatticeCache, cat_masks=None) -> np.ndarray:
        """Process model forward passes using a pre-computed KV cache.

        The cache is sliced along the batch dimension to match each batch, and
        the test rows of each batch are forwarded in chunks bounded by
        ``_default_test_chunk_cells`` (see ``_test_rows_per_forward``).

        Parameters
        ----------
        Xs : np.ndarray
            Test features of shape ``(n_datasets, test_size, n_features)``.

        kv_cache : LatticeCache
            Single KV cache for all estimators of a normalization method.

        Returns
        -------
        np.ndarray
            Model outputs (logits or probabilities) of shape
            ``(n_datasets, test_size, n_classes)``.
        """
        if self.batch_size == "auto":
            from .adaptive_batching import predict

            return predict(self, Xs, masks=cat_masks, cache=kv_cache)

        n_total = Xs.shape[0]
        batch_size = self._resolve_batch_size(n_total, Xs.shape[1])
        n_batches = int(np.ceil(n_total / batch_size))
        Xs_split = torch.tensor_split(Xs, n_batches) if isinstance(Xs, torch.Tensor) else np.array_split(Xs, n_batches)

        outputs = []
        offset = 0
        for X_batch in Xs_split:
            bs = X_batch.shape[0]
            cache_subset = kv_cache.slice_batch(offset, offset + bs)
            cat_kwargs = self._cat_kwargs(cat_masks, offset, offset + bs)
            offset += bs

            # Query rows are independent once the support side lives in the
            # cache, so chunking them is exact up to matmul reduction order and
            # bounds the peak activation.
            rows_per_chunk = self._test_rows_per_forward(X_batch.shape[1], X_batch.shape[2])
            rows_per_chunk = max(1, min(rows_per_chunk, self._max_batched_rows // bs))
            row_outputs = []
            for start in range(0, X_batch.shape[1], rows_per_chunk):
                X_chunk = torch.as_tensor(
                    X_batch[:, start : start + rows_per_chunk], dtype=torch.float32, device=self.device_
                )
                with torch.no_grad(), self._amp_context():
                    out = self.model_.forward_with_cache(
                        X_test=X_chunk,
                        cache=cache_subset,
                        return_logits=True if self.average_logits else False,
                        softmax_temperature=getattr(self, "softmax_temperature_", self.softmax_temperature),
                        inference_config=self.inference_config_,
                        **cat_kwargs,
                    )
                row_outputs.append(out.float().cpu().numpy())
            outputs.append(np.concatenate(row_outputs, axis=1))
        return np.concatenate(outputs, axis=0)

    def predict_proba(self, X: np.ndarray, *, mask_nan_features: bool = False) -> np.ndarray:
        """Predict class probabilities for test samples.

        Applies the ensemble of LatticeModel models to make predictions, with each ensemble
        member providing predictions that are then averaged. The method:

        1. Transforms input data using the fitted encoders
        2. Applies the ensemble generator to create multiple views
        3. Forwards each view through the model
        4. Corrects for class shuffles
        5. Averages predictions across ensemble members

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples for prediction. Missing values, including entirely
            missing query columns, use the fitted encoder's missing-value handling.

        mask_nan_features : bool, default=False
            Explicitly interpret all-NaN query columns as absent features for
            SHAP explanations. Only supported by the legacy ensemble generator
            (without asinh or SVD branches). Leave False for ordinary prediction.

        Returns
        -------
        np.ndarray of shape (n_samples, n_classes)
            Class probabilities for each test sample.
        """
        check_is_fitted(self)
        self._move_cache_to_device(for_inference=True)
        if isinstance(X, np.ndarray) and len(X.shape) == 1:
            # Reject 1D arrays to maintain sklearn compatibility
            raise ValueError("The provided input X is one-dimensional. Reshape your data.")

        # Check if prediction is possible
        has_kv_cache = hasattr(self, "model_kv_cache_") and self.model_kv_cache_ is not None
        has_training_data = (
            hasattr(self, "ensemble_generator_") and getattr(self.ensemble_generator_, "X_", None) is not None
        )
        if not has_kv_cache and not has_training_data:
            raise RuntimeError(
                "Cannot predict: this estimator was saved without training data and has no KV cache. "
                "Predictions require either cached KV projections or the original training data. "
                "Re-fit the estimator or load from a file saved with save_training_data=True or "
                "save_kv_cache=True."
            )

        if self.n_jobs is not None:
            assert self.n_jobs != 0
            old_n_threads = torch.get_num_threads()
            n_logical_cores = mp.cpu_count()

            if self.n_jobs > 0:
                if self.n_jobs > n_logical_cores:
                    warnings.warn(
                        f"LatticeModel got n_jobs={self.n_jobs} but there are only {n_logical_cores} logical cores available."
                        f" Only {n_logical_cores} threads will be used."
                    )
                n_threads = min(n_logical_cores, self.n_jobs)
            else:
                n_threads = max(1, n_logical_cores + 1 + self.n_jobs)

            torch.set_num_threads(n_threads)

        # Preserve DataFrame structure to retain column names and types for correct feature transformation
        X = validate_data(self, X, reset=False, dtype=None, skip_check_array=True)

        # Natural missingness must not change the feature set with query batching.
        # SHAP's all-NaN convention is opt-in, never inferred in normal prediction.
        feature_mask = None
        if mask_nan_features:
            if hasattr(X, "columns"):  # check for dataframe without importing pandas
                feature_mask = X.isna().all(axis=0).to_numpy()
            else:
                arr = np.asarray(X)
                if np.issubdtype(arr.dtype, np.number):
                    feature_mask = np.isnan(arr).all(axis=0)
                else:
                    # object dtype: v != v is True only for NaN in IEEE 754, safe for strings too
                    feature_mask = np.array([all(v != v for v in arr[:, i]) for i in range(arr.shape[1])])
        if feature_mask is not None and not np.any(feature_mask):
            feature_mask = None

        # Fill explicit masks on a copy; never mutate the caller's query data.
        if feature_mask is not None:
            X = X.copy() if hasattr(X, "columns") else np.array(X, copy=True)
            if hasattr(X, "columns"):  # Proxy way to check whether X is a dataframe
                X.iloc[:, feature_mask] = 0.0
            else:
                X[:, feature_mask] = 0.0

        X = self.X_encoder_.transform(X)

        if self.preprocessing_ == "lattice":
            from .classification_inference import predict_lattice

            try:
                return predict_lattice(self, X, feature_mask)
            finally:
                if self.n_jobs is not None:
                    torch.set_num_threads(old_n_threads)

        # Skip KV cache when features are masked
        has_kv_cache = hasattr(self, "model_kv_cache_") and self.model_kv_cache_ is not None
        use_cache = has_kv_cache and feature_mask is None

        if use_cache:
            # Cache exists: forward only test data and use the pre-computed cache for training data
            test_data = self._ensemble_data(X, mode="test")
            outputs = []
            for norm_method, (Xs_test,) in test_data:
                kv_cache = self.model_kv_cache_[norm_method]
                outputs.append(self._batch_forward_with_cache(
                    Xs_test, kv_cache, self._ensemble_cat_masks(norm_method)
                ))
            outputs = np.concatenate(outputs, axis=0)
        else:
            # No cache or masked features: forward both training and test data
            data = self._ensemble_data(X, mode="both", feature_mask=feature_mask)
            outputs = []
            for norm_method, (Xs, ys) in data:
                if feature_mask is None:
                    feature_shuffles = self.ensemble_generator_.feature_shuffles_[norm_method]
                else:
                    feature_shuffles = self.ensemble_generator_.masked_feature_shuffles_[norm_method]

                outputs.append(self._batch_forward(
                    Xs, ys, feature_shuffles, self._ensemble_cat_masks(norm_method, feature_mask)
                ))
            outputs = np.concatenate(outputs, axis=0)

        # Extract class shuffle patterns from ensemble generator
        class_shuffles = []
        for shuffles in self.ensemble_generator_.class_shuffles_.values():
            class_shuffles.extend(shuffles)

        # Determine actual number of ensemble members
        # May be fewer than requested if dataset has quite limited features and classes
        n_estimators = len(class_shuffles)

        # Aggregate predictions from all ensemble members, correcting for class shuffles
        avg = np.zeros_like(outputs[0])
        for i, shuffle in enumerate(class_shuffles):
            out = outputs[i]
            avg += out[..., shuffle]

        # Calculate ensemble average
        avg /= n_estimators

        # Convert logits to probabilities
        if self.average_logits:
            avg = self.softmax(avg, axis=-1, temperature=getattr(self, "softmax_temperature_", self.softmax_temperature))

        if self.n_jobs is not None:
            torch.set_num_threads(old_n_threads)

        # Normalize probabilities
        return avg / avg.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray, *, mask_nan_features: bool = False) -> np.ndarray:
        """Predict class labels for test samples.

        Uses predict_proba to get class probabilities and returns the class with
        the highest probability for each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples for prediction. Missing values, including entirely
            missing query columns, use the fitted encoder's missing-value handling.

        mask_nan_features : bool, default=False
            Explicitly interpret all-NaN query columns as absent features for
            SHAP explanations. Only supported by the legacy ensemble generator
            (without asinh or SVD branches). Leave False for ordinary prediction.

        Returns
        -------
        array-like of shape (n_samples,)
            Predicted class labels for each test sample.
        """
        proba = self.predict_proba(X, mask_nan_features=mask_nan_features)
        y = np.argmax(proba, axis=1)

        return self.y_encoder_.inverse_transform(y)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags
