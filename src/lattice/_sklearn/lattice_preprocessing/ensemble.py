"""Lattice's task-specific numerical feature recipes for ensemble inference.

Permutation formulas follow NVIDIA SDM (Apache-2.0; see NOTICE).
"""

from __future__ import annotations

from collections import OrderedDict
from math import gcd

import numpy as np
import torch
from sklearn.preprocessing import OrdinalEncoder

from .column_pairs import reduce_pair_reuse
from .numerical import LatticeNumericView
from .support_mean import support_mean


class LatticeEncoder:
    """Support-only typing, one-hot blocks, ordinal codes, counts, missing flags.

    Numeric columns with at most three values (including missing) are inferred
    categorical at 151 or more rows. Explicit indices override that inference.
    Missing/unseen categories share code -1, as in SDM AlignCategories.

    Column layout of ``transform`` (all fitted on support only): numeric,
    numeric-missing indicators, log category counts, ordinal codes, one-hot
    blocks, category-missing indicators. ``onehot_width=0`` disables the
    one-hot blocks and missing indicators, which is the regression recipe
    (codes plus log frequencies only).
    """

    MAX_CELLS = 3000000

    def __init__(
        self,
        categorical_indices=None,
        *,
        max_onehot_card=20,
        onehot_width=300,
        count_card=10,
        max_missing_indicators=50,
    ):
        self.categorical_indices = categorical_indices
        self.max_onehot_card = max_onehot_card
        self.onehot_width = onehot_width
        self.count_card = count_card
        self.max_missing_indicators = max_missing_indicators

    @staticmethod
    def _array(X):
        if hasattr(X, "to_numpy"):
            # Nullable pandas dtypes need explicit conversion of pd.NA.
            try:
                array = X.to_numpy(dtype=object, na_value=np.nan, copy=True)
            except ValueError:
                # Pandas can hand back a read-only buffer for mixed-dtype blocks.
                frame = X.astype(object).where(X.notna(), np.nan)
                array = np.array(frame, dtype=object, copy=True)
        else:
            array = np.asarray(X)
        if array.ndim != 2 or not array.shape[1]:
            raise ValueError(
                "X must be a two-dimensional table with at least one feature"
            )
        if array.dtype.kind == "O":
            array = array.copy()
            array[array == None] = np.nan
        return array

    def fit(self, X, y=None):
        array = self._array(X)
        width = array.shape[1]
        self.n_features_in_ = width
        explicit = self.categorical_indices
        if explicit is not None:
            if len(set(explicit)) != len(explicit) or any(
                isinstance(i, (bool, np.bool_))
                or not isinstance(i, (int, np.integer))
                or i < 0
                or i >= width
                for i in explicit
            ):
                raise ValueError(
                    "categorical_indices must contain unique valid column indices"
                )
            cat = np.array([i in explicit for i in range(width)])
        else:
            cat = []
            for i in range(width):
                dtype = X.dtypes.iloc[i] if hasattr(X, "dtypes") else array[:, i].dtype
                kind = getattr(dtype, "kind", "O")
                categorical = kind not in "iuf"
                # Object NumPy columns containing numbers are numeric; strings
                # remain categorical (including numeric-looking strings).
                if not hasattr(X, "dtypes") and kind == "O":
                    values = array[:, i]
                    categorical = any(isinstance(v, (str, bytes)) for v in values)
                if not categorical and len(array) >= 151:
                    categorical = len(np.unique(array[:, i].astype(float))) <= 3
                cat.append(categorical)
            cat = np.asarray(cat)
        self.numeric_indices_ = np.flatnonzero(~cat)
        self.category_indices_ = np.flatnonzero(cat)
        self.ordinal_ = None
        self.counts_ = []
        self.code_cols_ = []
        self.onehot_ = []
        self.miss_cat_ = []
        self.miss_numeric_ = np.zeros(len(self.numeric_indices_), dtype=bool)
        if len(self.numeric_indices_) and self.max_missing_indicators:
            numeric_support = array[:, self.numeric_indices_].astype(np.float64)
            rate = np.isnan(numeric_support).mean(axis=0)
            candidate = np.flatnonzero((rate >= 0.02) & (rate <= 0.98))
            if len(candidate) > self.max_missing_indicators:
                # Attention is quadratic in width: keep the most-missing columns.
                candidate = candidate[np.argsort(-rate[candidate], kind="stable")]
                candidate = np.sort(candidate[: self.max_missing_indicators])
            self.miss_numeric_[candidate] = True
        if cat.any():
            self.ordinal_ = OrdinalEncoder(
                dtype=np.float64,
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                encoded_missing_value=-1,
            )
            codes = self.ordinal_.fit_transform(array[:, cat]).astype(np.int64)
            for j, categories in enumerate(self.ordinal_.categories_):
                # NaN is not part of AlignCategories' fitted vocabulary.
                n = len(categories) - int(
                    any(
                        isinstance(v, (float, np.floating)) and np.isnan(v)
                        for v in categories
                    )
                )
                if self.max_missing_indicators and (codes[:, j] < 0).any():
                    self.miss_cat_.append(j)
                if n <= self.max_onehot_card:
                    self.onehot_.append((j, n))
                    continue
                self.code_cols_.append(j)
                if n > self.count_card:
                    counts = np.bincount(
                        np.where(codes[:, j] >= 0, codes[:, j], n), minlength=n + 1
                    )
                    self.counts_.append((j, n, np.log1p(counts.astype(np.float64))))
            if sum(n for _, n in self.onehot_) > self.onehot_width:
                # Width budget: keep the cheapest blocks, code the rest.
                keep, used = [], 0
                for j, n in sorted(self.onehot_, key=lambda t: (t[1], t[0])):
                    if used + n <= self.onehot_width:
                        keep.append((j, n))
                        used += n
                    else:
                        self.code_cols_.append(j)
                self.onehot_ = sorted(keep)
                self.code_cols_.sort()
            total_width = (
                len(self.numeric_indices_)
                + int(self.miss_numeric_.sum())
                + len(self.counts_)
                + len(self.code_cols_)
                + len(self.miss_cat_)
                + sum(n for _, n in self.onehot_)
            )
            if len(array) * max(total_width, 1) > self.MAX_CELLS:
                # Tall wide tables cost more compute than the columns return.
                self.code_cols_.extend(j for j, _ in self.onehot_)
                self.code_cols_.sort()
                self.onehot_ = []
                self.miss_numeric_[:] = False
        nn = len(self.numeric_indices_)
        nn_missing = int(self.miss_numeric_.sum())
        nf = len(self.counts_)
        nc = len(self.code_cols_)
        no = sum(n for _, n in self.onehot_)
        nm = len(self.miss_cat_)
        if self.onehot_width:
            head = [False] * (nn + nn_missing)
            tail = [True] * (nf + nc + no + nm)
        else:
            # Codes-only recipe keeps log counts on the numeric (z-scored) route.
            head = [False] * (nn + nn_missing + nf)
            tail = [True] * (nc + no + nm)
        self.categorical_mask_ = np.array(head + tail)
        self.category_route_ = np.array(head + tail)
        return self

    def transform(self, X):
        array = self._array(X)
        if array.shape[1] != self.n_features_in_:
            raise ValueError("Query feature count differs from fit")
        numeric = array[:, self.numeric_indices_].astype(np.float64)
        parts = [numeric]
        if self.miss_numeric_.any():
            parts.append(np.isnan(numeric[:, self.miss_numeric_]).astype(np.float64))
        if self.ordinal_ is None:
            return np.concatenate(parts, axis=1)
        codes = self.ordinal_.transform(array[:, self.category_indices_])
        if self.counts_:
            parts.append(
                np.column_stack(
                    [
                        lookup[np.where(codes[:, j] >= 0, codes[:, j], n).astype(np.int64)]
                        for j, n, lookup in self.counts_
                    ]
                )
            )
        if self.code_cols_:
            parts.append(codes[:, self.code_cols_])
        for j, n in self.onehot_:
            block = np.zeros((len(array), n))
            valid = codes[:, j]
            keep = (valid >= 0) & (valid < n)
            block[np.flatnonzero(keep), valid[keep].astype(np.int64)] = 1.0
            parts.append(block)
        if self.miss_cat_:
            parts.append((codes[:, self.miss_cat_] < 0).astype(np.float64))
        return np.concatenate(parts, axis=1)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


class LatticeEnsembleGenerator:
    """Task-specific view schedules, numeric sign flips, and column permutations.

    The transformed support is shared by members of each branch. The final
    missing-value fill uses that branch's support mean, matching the SDM adaptation.
    Member permutations use a local CPU generator for reproducibility across
    CPU/GPU preprocessing. No SDM installation is needed.
    """

    # Older serialized generators resolve the view schedule from the class.
    methods = ("none", "power", "robust", "rank")
    regression_methods = ("none", "power", "rank", "rank", "none", "power", "rank", "rank")

    def __init__(
        self,
        *,
        n_estimators,
        classification,
        categorical_mask,
        category_route,
        device,
        random_state=42,
        outlier_threshold=4.0,
        feat_shuffle_method="latin",
        class_shuffle_method="shift",
        tensor_output=False,
        cpu_support_mean=False,
    ):
        self.n_estimators = n_estimators
        self.classification = classification
        if classification:
            self.methods = type(self).methods
            # Uniform rank margins with light KDI smoothing (classification).
            self.rank_options_ = dict(kdi_alpha=0.25, rank_uniform=True, max_knots=0)
            self.max_features_ = 500
        else:
            self.methods = self.regression_methods
            # Normal rank margins with a coarse quantile grid (regression).
            self.rank_options_ = dict(
                kdi_alpha=0.75, rank_uniform=False, max_knots=256
            )
            self.max_features_ = 1000
        self.categorical_mask = categorical_mask
        self.category_route = category_route
        self.device = device
        self.random_state = random_state
        self.outlier_threshold = outlier_threshold
        self.feat_shuffle_method = feat_shuffle_method
        self.class_shuffle_method = class_shuffle_method
        self.tensor_output = tensor_output
        self.cpu_support_mean = cpu_support_mean

    def fit(self, X, y):
        self.X_, self.y_ = X.copy(), np.asarray(y).copy()
        if self.tensor_output:
            X = torch.as_tensor(X, dtype=torch.float64, device=self.device)
        self.preprocessors_, self.numeric_support_ = {}, {}
        self.base_masks_, self.sign_masks_, self.fill_ = {}, {}, {}
        for method in self.methods[: min(len(self.methods), self.n_estimators)]:
            blocks, masks, sign_masks = [], [], []
            for categorical in (False, True):
                route = self.category_route == categorical
                prep = LatticeNumericView(
                    method,
                    self.device,
                    self.outlier_threshold,
                    tensor_output=self.tensor_output,
                    **self.rank_options_,
                ).fit(X[:, route])
                self.preprocessors_[method, categorical] = prep
                blocks.append(prep.X_transformed_)
                masks.extend(self.categorical_mask[route][prep.keep_])
                sign_masks.extend([not categorical] * int(prep.keep_.sum()))
            values = (
                torch.cat(blocks, dim=1)
                if self.tensor_output
                else np.concatenate(blocks, axis=1)
            )
            if values.shape[1] == 0:
                # Lattice requires a feature token even on an all-constant table.
                values = (
                    X.new_zeros((len(X), 1))
                    if self.tensor_output
                    else np.zeros((len(X), 1))
                )
                masks, sign_masks = [False], [False]
            self.base_masks_[method] = np.asarray(masks)
            self.sign_masks_[method] = np.asarray(sign_masks)
            self.numeric_support_[method] = values
            if self.tensor_output:
                self.fill_[method] = support_mean(
                    values, use_cpu=getattr(self, "cpu_support_mean", False)
                )
            else:
                valid = np.isfinite(values)
                self.fill_[method] = np.where(valid, values, 0).sum(0) / np.maximum(
                    valid.sum(0), 1
                )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            int(self.random_state)
        ) if self.random_state is not None else generator.seed()
        width = next(iter(self.numeric_support_.values())).shape[1]
        self.signs_ = []
        for member in range(self.n_estimators):
            method = self.methods[member % len(self.methods)]
            mask = self.sign_masks_[method]
            signs = np.ones(width)
            signs[mask] = (
                torch.empty(int(mask.sum()), dtype=torch.float64)
                .bernoulli_(0.5, generator=generator)
                .mul_(-2)
                .add_(1)
                .numpy()
            )
            self.signs_.append(signs)
        base = torch.randperm(width, generator=generator).numpy()
        rows = torch.randperm(width, generator=generator).numpy()
        permutations = []
        strides = ()
        if self.classification and self.feat_shuffle_method == "latin":
            strides = tuple(step for step in range(2, width - 1) if gcd(step, width) == 1)
        for member in range(self.n_estimators):
            if self.feat_shuffle_method == "latin":
                permutation = base[(member - rows) % width]
                if member >= 4 and strides:
                    stride = strides[(member - 4) % len(strides)]
                    permutation = permutation[(np.arange(width) * stride) % width]
            elif self.feat_shuffle_method == "random":
                permutation = torch.randperm(width, generator=generator).numpy()
            elif self.feat_shuffle_method == "shift":
                permutation = np.roll(np.arange(width), member)
            elif self.feat_shuffle_method in (None, "none"):
                permutation = np.arange(width)
            else:
                raise ValueError("Unknown feat_shuffle_method")
            permutations.append(permutation[: self.max_features_])
        if self.classification and self.feat_shuffle_method == "latin":
            permutations = reduce_pair_reuse(permutations, width)
        classes = len(np.unique(y)) if self.classification else 0
        offsets, shuffles = [], []
        for member in range(self.n_estimators):
            if not self.classification:
                shuffle = None
            elif self.class_shuffle_method == "shift":
                if not offsets:
                    offsets = torch.randperm(classes, generator=generator).tolist()
                shuffle = (np.arange(classes) - offsets.pop(0)) % classes
            elif self.class_shuffle_method == "random":
                shuffle = torch.randperm(classes, generator=generator).numpy()
            elif self.class_shuffle_method in (None, "none"):
                shuffle = np.arange(classes)
            else:
                raise ValueError("Unknown class_shuffle_method")
            shuffles.append(shuffle)
        if not self.classification:
            self.target_signs_ = (
                torch.empty(self.n_estimators, dtype=torch.float32)
                .bernoulli_(0.5, generator=generator)
                .mul_(-2)
                .add_(1)
                .numpy()
            )
        self.ensemble_configs_ = OrderedDict(lattice=list(zip(permutations, shuffles)))
        self._device_member_configs = None
        # Sign flips differ per member: column embedding reuse based solely on
        # permutations would be incorrect. Let the backbone process each view.
        self.feature_shuffles_ = OrderedDict(lattice=None)
        self.class_shuffles_ = OrderedDict(lattice=shuffles)
        self.categorical_masks_ = OrderedDict(
            lattice=np.stack(
                [
                    self.base_masks_[self.methods[i % len(self.methods)]][p]
                    for i, p in enumerate(permutations)
                ]
            )
        )
        return self

    def to(self, device):
        self.device = device
        for prep in self.preprocessors_.values():
            prep.to(device)
        for values in (self.numeric_support_, self.fill_):
            for key, value in values.items():
                if isinstance(value, torch.Tensor):
                    values[key] = value.to(device)
        configs = getattr(self, "_device_member_configs", None)
        # Compare resolved tensor devices: "cuda" and "cuda:0" can refer to
        # the same device and must not invalidate these constants every call.
        if configs is not None:
            actual_device = next(iter(self.fill_.values())).device
            if configs[0][0].device != actual_device:
                self._device_member_configs = None
        return self

    def release_support(self):
        """Drop redundant transformed rows after KV construction.

        Keep one CPU input and the fitted statistics for explicit cache rebuilds.
        Query transforms never need the transformed training rows.
        """
        self.numeric_support_.clear()
        for prep in self.preprocessors_.values():
            prep.X_transformed_ = None

    def _restore_support(self):
        if self.numeric_support_:
            return
        if self.X_ is None:
            raise RuntimeError("Training rows were excluded when saving this estimator")
        values = self.X_
        if getattr(self, "tensor_output", False):
            values = torch.as_tensor(values, dtype=torch.float64, device=self.device)
        self.numeric_support_ = {
            method: self._query(values, method) for method in self.base_masks_
        }

    def __getstate__(self):
        state = vars(self).copy()
        state.pop("_device_member_configs", None)
        for name in ("numeric_support_", "fill_"):
            state[name] = {
                key: value.cpu() if isinstance(value, torch.Tensor) else value
                for key, value in state[name].items()
            }
        state["device"] = torch.device("cpu")
        return state

    def _query(self, X, method):
        parts = [
            self.preprocessors_[method, cat].transform(X[:, self.category_route == cat])
            for cat in (False, True)
        ]
        if getattr(self, "tensor_output", False):
            values = torch.cat(parts, dim=1)
            return values if values.shape[1] else values.new_zeros((len(X), 1))
        values = np.concatenate(parts, axis=1)
        return values if values.shape[1] else np.zeros((len(X), 1))

    def iter_transform(self, X=None, mode="both", feature_mask=None):
        if feature_mask is not None:
            raise ValueError(
                'feature_mask is unsupported by Lattice preprocessing; use preprocessing="legacy"'
            )
        if mode not in ("train", "test", "both"):
            raise ValueError("mode must be train, test or both")
        if mode != "train" and X is None:
            raise ValueError("Query data is required")
        if mode != "test":
            self._restore_support()
        tensor_output = getattr(self, "tensor_output", False)
        if tensor_output and X is not None:
            X = torch.as_tensor(X, dtype=torch.float64, device=self.device)
        query = (
            {m: self._query(X, m) for m in self.base_masks_} if mode != "train" else {}
        )
        if tensor_output:
            # Each branch is shared by several members. Concatenate and impute
            # once before applying each member's signs and feature permutation.
            prepared = {}
            for method in self.base_masks_:
                if mode == "train":
                    values = self.numeric_support_[method]
                elif mode == "test":
                    values = query[method]
                else:
                    values = torch.cat((self.numeric_support_[method], query[method]))
                prepared[method] = torch.where(
                    values.isfinite(), values, self.fill_[method]
                )
            sample = next(iter(prepared.values()))
            configs = getattr(self, "_device_member_configs", None)
            if configs is None or configs[0][0].device != sample.device:
                configs = [
                    (
                        torch.as_tensor(signs, device=sample.device),
                        torch.as_tensor(permutation, device=sample.device),
                    )
                    for signs, (permutation, _) in zip(
                        self.signs_, self.ensemble_configs_["lattice"]
                    )
                ]
                self._device_member_configs = configs
            features = torch.empty(
                (self.n_estimators, sample.shape[0], len(configs[0][1])),
                device=sample.device,
                dtype=torch.float32,
            )
        else:
            features = []
        labels = []
        for i, (permutation, shuffle) in enumerate(self.ensemble_configs_["lattice"]):
            method = self.methods[i % len(self.methods)]
            if tensor_output:
                signs, indices = configs[i]
                values = prepared[method]
                features[i].copy_((values * signs).index_select(1, indices))
            elif mode == "train":
                values = self.numeric_support_[method]
            elif mode == "test":
                values = query[method]
            else:
                blocks = (self.numeric_support_[method], query[method])
                values = np.concatenate(blocks)
            if not tensor_output:
                values = np.where(np.isfinite(values), values, self.fill_[method])
                features.append(
                    (values * self.signs_[i])[:, permutation].astype(np.float32)
                )
            if mode != "test":
                target = self.y_ if shuffle is None else shuffle[self.y_]
                if not self.classification:
                    target = target * self.target_signs_[i]
                labels.append(target)
        if tensor_output:
            result = (features,)
            if mode != "test":
                result += (
                    torch.as_tensor(
                        np.stack(labels), device=self.device, dtype=torch.float32
                    ),
                )
            yield "lattice", result
            return
        result = (
            (np.stack(features),)
            if mode == "test"
            else (np.stack(features), np.stack(labels))
        )
        yield "lattice", result
