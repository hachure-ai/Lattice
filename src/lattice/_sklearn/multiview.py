"""Support-fitted, type-aware inference views; the legacy path stays separate."""

from collections import OrderedDict
import warnings

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.extmath import randomized_svd
from sklearn.utils.validation import check_is_fitted

from .preprocessing import OutlierRemover, PreprocessingPipeline, Shuffler, UniqueFeatureFilter, global_shuffle_pairs
from .sklearn_utils import validate_data

NUMERIC_METHODS = ("none", "power", "quantile", "asinh")
SUPPORTED_METHODS = NUMERIC_METHODS + ("robust", "quantile_rtdl")


def resolve_norm_methods(norm_methods):
    """Validate explicit plain/SVD branches without silently falling back."""
    methods = ["none", "power"] if norm_methods is None else norm_methods
    methods = [methods] if isinstance(methods, str) else methods
    if not isinstance(methods, (list, tuple)) or not methods:
        raise ValueError("norm_methods must be a nonempty string or list of methods.")
    valid = SUPPORTED_METHODS + tuple(f"{method}_svd" for method in SUPPORTED_METHODS)
    if any(not isinstance(method, str) or method not in valid for method in methods):
        raise ValueError(f"norm_methods must contain only {valid}.")
    if len(set(methods)) != len(methods):
        raise ValueError("norm_methods must not contain duplicate branches.")
    return list(methods)


class NumericView(TransformerMixin, BaseEstimator):
    """Fit one numerical transform, falling back to the base transform if needed."""

    def __init__(self, method, outlier_threshold=4.0, random_state=42):
        self.method = method
        self.outlier_threshold = outlier_threshold
        self.random_state = random_state

    def fit(self, X, y=None):
        if self.method not in SUPPORTED_METHODS:
            raise ValueError(f"Unknown normalization method: {self.method}")
        self.fallback_ = False
        self.empty_ = X.shape[1] == 0
        if self.empty_:
            self.X_transformed_ = X.copy()
            return self
        try:
            if self.method == "asinh":
                self.center_ = np.median(X, axis=0)
                self.scale_ = np.diff(np.quantile(X, [0.25, 0.75], axis=0), axis=0)[0]
                self.scale_ = np.where(self.scale_ > 1e-12, self.scale_, 1.0)
                values = np.arcsinh((X - self.center_) / self.scale_)
                self.outliers_ = OutlierRemover(threshold=self.outlier_threshold).fit(values)
            else:
                self.pipeline_ = PreprocessingPipeline(self.method, self.outlier_threshold, self.random_state).fit(X)
            self.X_transformed_ = self.transform(X)
            if not np.isfinite(self.X_transformed_).all():
                raise ValueError("non-finite transformed support")
        except (ValueError, FloatingPointError, OverflowError) as exc:
            if self.method == "none":
                raise
            warnings.warn(f"{self.method} preprocessing failed; using none: {exc}", RuntimeWarning)
            self.fallback_ = True
            self.pipeline_ = PreprocessingPipeline("none", self.outlier_threshold, self.random_state).fit(X)
            self.X_transformed_ = self.pipeline_.transform(X)
        return self

    def transform(self, X):
        if self.empty_:
            return X.copy()
        if self.method == "asinh" and not self.fallback_:
            return self.outliers_.transform(np.arcsinh((X - self.center_) / self.scale_))
        return self.pipeline_.transform(X)


class DiverseEnsembleGenerator(TransformerMixin, BaseEstimator):
    """Balanced views with shared numeric fits and optional appended SVD columns.

    Explicit norm_methods branches share a global permutation budget by default.
    ensemble_mode='legacy' reuses permutation slots across branches.
    Category encodings and SVD bases are fitted on support only. Small-rank
    SVD projections are zero-padded so members in a group have equal width.
    """

    def __init__(
        self,
        classification=True,
        n_estimators=32,
        norm_methods=None,
        feat_shuffle_method="latin",
        class_shuffle_method="shift",
        outlier_threshold=4.0,
        random_state=42,
        categorical_mask=None,
        ensemble_mode="global",
    ):
        self.classification = classification
        self.n_estimators = n_estimators
        self.norm_methods = norm_methods
        self.ensemble_mode = ensemble_mode
        self.feat_shuffle_method = feat_shuffle_method
        self.class_shuffle_method = class_shuffle_method
        self.outlier_threshold = outlier_threshold
        self.random_state = random_state
        self.categorical_mask = categorical_mask

    def _make_preprocessor(self, method):
        return NumericView(method, self.outlier_threshold, self.seed_)

    def fit(self, X, y):
        X, y = validate_data(self, X, y, dtype=np.float64)
        methods = resolve_norm_methods(self.norm_methods)
        if self.ensemble_mode not in ("global", "legacy"):
            raise ValueError("ensemble_mode must be 'global' or 'legacy'.")
        if (
            isinstance(self.n_estimators, bool)
            or not isinstance(self.n_estimators, (int, np.integer))
            or self.n_estimators < 1
        ):
            raise ValueError("n_estimators must be a positive integer.")
        cat_mask = (
            np.zeros(X.shape[1], dtype=bool) if self.categorical_mask is None else np.asarray(self.categorical_mask)
        )
        if cat_mask.dtype != bool or cat_mask.shape != (X.shape[1],):
            raise ValueError("categorical_mask must be boolean with one entry per encoded column.")
        self.unique_filter_ = UniqueFeatureFilter()
        self.X_ = self.unique_filter_.fit_transform(X)
        self.y_ = np.array(y, copy=True)
        self.cat_mask_ = cat_mask[self.unique_filter_.features_to_keep_]
        self.n_classes_ = len(np.unique(y)) if self.classification else 0
        if self.X_.shape[1] == 0:
            raise ValueError("No features remain after filtering.")

        rng = np.random.RandomState(self.random_state)
        self.seed_ = int(rng.randint(2**31 - 1))
        branches = [(method.removesuffix("_svd"), method.endswith("_svd")) for method in methods]
        counts = [
            self.n_estimators // len(branches) + int(i < self.n_estimators % len(branches))
            for i in range(len(branches))
        ]
        k_svd = min(16, self.X_.shape[1], self.X_.shape[0] - 1)
        if self.ensemble_mode == "global":
            widths = [
                self.X_.shape[1] + (k_svd if branches[i % len(branches)][1] else 0)
                for i in range(self.n_estimators)
            ]
            global_pairs = global_shuffle_pairs(
                widths, self.n_classes_, self.feat_shuffle_method, self.class_shuffle_method, self.seed_
            )
        self.preprocessors_ = {}
        self.numeric_support_ = {}
        for (method, _), count in zip(branches, counts):
            if not count:
                continue
            if method not in self.preprocessors_:
                prep = self._make_preprocessor(method).fit(self.X_[:, ~self.cat_mask_])
                self.preprocessors_[method] = prep
                self.numeric_support_[method] = prep.X_transformed_

        # One encoding per permutation slot, shared across numerical/SVD branches.
        self.category_encodings_ = []
        encoding_ids = []
        unique_encodings = {}
        for slot in range(max(counts)):
            encoding = []
            for column in np.flatnonzero(self.cat_mask_):
                values = np.unique(self.X_[:, column])
                codes = np.arange(len(values)) if slot % 2 == 0 else rng.permutation(len(values))
                support_codes = codes[np.searchsorted(values, self.X_[:, column])]
                scale = float(support_codes.std())
                scale = scale if scale > 1e-12 else 1.0
                encoding.append((column, values, codes, float(support_codes.mean()), scale))
            self.category_encodings_.append(encoding)
            signature = tuple(tuple(entry[2]) for entry in encoding)
            encoding_ids.append(unique_encodings.setdefault(signature, slot))

        classes = (
            Shuffler(self.n_classes_, self.class_shuffle_method, random_state=self.seed_).shuffle(max(counts))
            if self.classification and self.ensemble_mode == "legacy"
            else [None]
        )
        self.ensemble_configs_ = OrderedDict()
        self.feature_shuffles_ = OrderedDict()
        self.class_shuffles_ = OrderedDict()
        self.categorical_masks_ = OrderedDict()
        self.svd_transforms_ = {}
        self.group_specs_ = {}
        svd_cache = {}
        for branch_index, ((method, svd), count) in enumerate(zip(branches, counts)):
            if not count:
                continue
            key = f"{method}_{'svd' if svd else 'plain'}"
            k = min(16, self.X_.shape[1], self.X_.shape[0] - 1) if svd else 0
            width = self.X_.shape[1] + k
            if self.ensemble_mode == "global":
                configs = global_pairs[branch_index::len(branches)]
            else:
                permutations = Shuffler(width, self.feat_shuffle_method, random_state=self.seed_).shuffle(count)
                configs = [(permutations[i % len(permutations)], classes[i % len(classes)]) for i in range(count)]
            self.group_specs_[key] = (method, k)
            self.ensemble_configs_[key] = configs
            self.feature_shuffles_[key] = [p for p, _ in configs]
            self.class_shuffles_[key] = [p for _, p in configs]
            full_mask = np.concatenate([self.cat_mask_, np.zeros(k, dtype=bool)])
            self.categorical_masks_[key] = np.stack([full_mask[p] for p, _ in configs])
            if k:
                for slot in range(count):
                    fit_key = (method, encoding_ids[slot], k)
                    if fit_key not in svd_cache:
                        support = self._base_view(self.X_, self.numeric_support_[method], slot)
                        svd_cache[fit_key] = self._fit_svd(support, k)
                    self.svd_transforms_[key, slot] = svd_cache[fit_key]
        return self

    def __setstate__(self, state):
        old_mode = state.pop("multiview", None)
        if old_mode is not None:
            state.setdefault("norm_methods", [
                method + suffix for method in NUMERIC_METHODS
                for suffix in (("", "_svd") if old_mode == "diverse_svd" else ("",))
            ])
        state.setdefault("ensemble_mode", "legacy")
        self.__dict__.update(state)

    def _base_view(self, X, numeric, slot):
        result = np.empty_like(X, dtype=np.float64)
        result[:, ~self.cat_mask_] = numeric
        for column, values, codes, center, scale in self.category_encodings_[slot]:
            indices = np.searchsorted(values, X[:, column])
            bounded = np.minimum(indices, len(values) - 1)
            known = (indices < len(values)) & (values[bounded] == X[:, column])
            encoded = np.where(known, codes[bounded], -1)
            result[:, column] = (encoded - center) / scale
        if not np.isfinite(result).all():
            raise ValueError("Non-finite values after multiview preprocessing.")
        return result

    def _fit_svd(self, support, k):
        center = support.mean(axis=0)
        centered = support - center
        try:
            _, singular_values, components = randomized_svd(
                centered, n_components=k, random_state=self.seed_, n_iter=5
            )
            tolerance = np.finfo(np.float64).eps * max(centered.shape) * singular_values[0]
            components[singular_values <= tolerance] = 0.0
            scale = (centered @ components.T).std(axis=0)
            scale = np.where(scale > 1e-12, scale, 1.0)
            if not np.isfinite(components).all() or not np.isfinite(scale).all():
                raise np.linalg.LinAlgError("non-finite SVD")
        except (ValueError, np.linalg.LinAlgError) as exc:
            warnings.warn(f"SVD failed; appending zero components: {exc}", RuntimeWarning)
            components = np.zeros((k, support.shape[1]))
            scale = np.ones(k)
        return center, components, scale

    def iter_transform(self, X=None, mode="both", feature_mask=None):
        """Yield one equal-width group at a time, including masks via group metadata."""
        check_is_fitted(self, "ensemble_configs_")
        if feature_mask is not None:
            raise ValueError("feature_mask/SHAP masking is not supported with asinh or SVD branches.")
        if mode not in ("both", "train", "test"):
            raise ValueError("mode must be both, train, or test.")
        query = None
        if mode != "train":
            if X is None:
                raise ValueError("Query data is required for test/both mode.")
            query = self.unique_filter_.transform(X)
        query_numeric = {}
        for key, configs in self.ensemble_configs_.items():
            method, k = self.group_specs_[key]
            if query is not None and method not in query_numeric:
                query_numeric[method] = self.preprocessors_[method].transform(query[:, ~self.cat_mask_])
            views, labels = [], []
            for slot, (permutation, class_order) in enumerate(configs):
                parts = []
                if mode != "test":
                    parts.append(self._base_view(self.X_, self.numeric_support_[method], slot))
                if query is not None:
                    parts.append(self._base_view(query, query_numeric[method], slot))
                values = np.concatenate(parts, axis=0)
                if k:
                    center, components, scale = self.svd_transforms_[key, slot]
                    projection = ((values - center) @ components.T) / scale
                    values = np.concatenate([values, projection], axis=1)
                if not np.isfinite(values).all():
                    raise ValueError("Non-finite multiview projection.")
                views.append(values[:, permutation])
                if mode != "test":
                    labels.append(np.asarray(class_order)[self.y_.astype(int)] if self.classification else self.y_)
            batch = np.stack(views)
            yield key, ((batch,) if mode == "test" else (batch, np.stack(labels)))

    def transform(self, X=None, mode="both", feature_mask=None):
        return OrderedDict(self.iter_transform(X, mode, feature_mask))
