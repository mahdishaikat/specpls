"""Spectral preprocessing transformers.

Every transformer follows the scikit-learn ``fit``/``transform`` API, so they can
be chained in a :class:`sklearn.pipeline.Pipeline` and are re-fitted inside each
cross-validation fold. That matters for data-dependent methods such as MSC,
whose reference spectrum must come from the training data only.

Spectra are always arranged as ``(n_samples, n_wavenumbers)``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import sparse
from scipy.linalg import solveh_banded
from scipy.signal import savgol_filter
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.utils.validation import check_array, check_is_fitted

__all__ = [
    "SNV",
    "MSC",
    "SavitzkyGolay",
    "Detrend",
    "AsLSBaseline",
    "RegionSelector",
    "EPO",
    "VariableSelector",
    "build_preprocessing",
    "PREPROCESSORS",
]


class _SpectralTransformer(TransformerMixin, BaseEstimator):
    """Shared input validation for all spectral transformers."""

    def _check_fit_input(self, X):
        X = check_array(X, dtype=np.float64)
        self.n_features_in_ = X.shape[1]
        return X

    def _check_transform_input(self, X):
        check_is_fitted(self, "n_features_in_")
        X = check_array(X, dtype=np.float64)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} wavenumbers, but {type(self).__name__} "
                f"was fitted with {self.n_features_in_}."
            )
        return X


class SNV(_SpectralTransformer):
    """Standard Normal Variate: centre and scale each spectrum individually.

    Removes multiplicative scatter and path-length effects. Stateless: fitting
    only records the number of wavenumbers.
    """

    def fit(self, X, y=None):
        self._check_fit_input(X)
        return self

    def transform(self, X):
        X = self._check_transform_input(X)
        mean = X.mean(axis=1, keepdims=True)
        std = X.std(axis=1, ddof=1, keepdims=True)
        std[std == 0] = 1.0
        return (X - mean) / std


class MSC(_SpectralTransformer):
    """Multiplicative Scatter Correction.

    Each spectrum ``x`` is regressed on a reference spectrum ``r``
    (``x = a + b * r``) and corrected as ``(x - a) / b``.

    Parameters
    ----------
    reference : array-like of shape (n_wavenumbers,), optional
        Fixed reference spectrum. If ``None`` (default), the mean spectrum of
        the training data is used.
    """

    def __init__(self, reference=None):
        self.reference = reference

    def fit(self, X, y=None):
        X = self._check_fit_input(X)
        if self.reference is None:
            self.reference_ = X.mean(axis=0)
        else:
            ref = np.asarray(self.reference, dtype=np.float64).ravel()
            if ref.shape[0] != X.shape[1]:
                raise ValueError("reference length must equal the number of wavenumbers.")
            self.reference_ = ref
        return self

    def transform(self, X):
        X = self._check_transform_input(X)
        design = np.column_stack([np.ones_like(self.reference_), self.reference_])
        coef, *_ = np.linalg.lstsq(design, X.T, rcond=None)
        intercept, slope = coef[0], coef[1]
        slope = np.where(slope == 0, 1.0, slope)
        return (X - intercept[:, None]) / slope[:, None]


class SavitzkyGolay(_SpectralTransformer):
    """Savitzky-Golay smoothing or derivative along the wavenumber axis.

    Parameters
    ----------
    window_length : int, default=15
        Number of points in the filter window. Must be odd and greater than
        ``polyorder``.
    polyorder : int, default=2
        Order of the fitted polynomial.
    deriv : int, default=0
        Derivative order (0 = smoothing only, 1 = first derivative, ...).
    """

    def __init__(self, window_length=15, polyorder=2, deriv=0):
        self.window_length = window_length
        self.polyorder = polyorder
        self.deriv = deriv

    def fit(self, X, y=None):
        X = self._check_fit_input(X)
        if self.window_length % 2 == 0:
            raise ValueError("window_length must be odd.")
        if self.polyorder >= self.window_length:
            raise ValueError("polyorder must be less than window_length.")
        if self.deriv > self.polyorder:
            raise ValueError("deriv must be less than or equal to polyorder.")
        if self.window_length > X.shape[1]:
            raise ValueError("window_length cannot exceed the number of wavenumbers.")
        return self

    def transform(self, X):
        X = self._check_transform_input(X)
        return savgol_filter(
            X,
            window_length=self.window_length,
            polyorder=self.polyorder,
            deriv=self.deriv,
            axis=1,
            mode="interp",
        )


class Detrend(_SpectralTransformer):
    """Subtract a low-order polynomial baseline fitted to each spectrum.

    Parameters
    ----------
    order : int, default=1
        Polynomial order (0 = offset, 1 = linear, 2 = quadratic, ...).
    """

    def __init__(self, order=1):
        self.order = order

    def fit(self, X, y=None):
        self._check_fit_input(X)
        if self.order < 0:
            raise ValueError("order must be >= 0.")
        return self

    def transform(self, X):
        X = self._check_transform_input(X)
        axis = np.linspace(-1.0, 1.0, X.shape[1])
        vander = np.vander(axis, self.order + 1)
        coef, *_ = np.linalg.lstsq(vander, X.T, rcond=None)
        return X - (vander @ coef).T


class AsLSBaseline(_SpectralTransformer):
    """Asymmetric Least Squares baseline removal (Eilers & Boelens, 2005).

    Fits a smooth baseline that stays below the Raman peaks, which suits broad
    fluorescence backgrounds.

    Parameters
    ----------
    lam : float, default=1e5
        Smoothness penalty. Larger values give a stiffer baseline.
    p : float, default=0.01
        Asymmetry weight for points above the baseline (0 < p < 1).
    n_iter : int, default=10
        Number of reweighting iterations.
    """

    def __init__(self, lam=1e5, p=0.01, n_iter=10):
        self.lam = lam
        self.p = p
        self.n_iter = n_iter

    def fit(self, X, y=None):
        X = self._check_fit_input(X)
        if not 0 < self.p < 1:
            raise ValueError("p must be between 0 and 1.")
        if X.shape[1] < 3:
            raise ValueError("AsLS needs at least 3 wavenumbers.")
        return self

    def _baseline(self, spectrum, banded_penalty):
        # (W + lam * D'D) is symmetric positive definite with bandwidth 2, so a
        # banded Cholesky solve is much faster than a general sparse solve.
        weights = np.ones_like(spectrum)
        baseline = spectrum
        for _ in range(self.n_iter):
            ab = banded_penalty.copy()
            ab[-1] += weights
            baseline = solveh_banded(ab, weights * spectrum)
            weights = np.where(spectrum > baseline, self.p, 1 - self.p)
        return baseline

    def transform(self, X):
        X = self._check_transform_input(X)
        n = X.shape[1]
        diff = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n))
        penalty = (self.lam * (diff.T @ diff)).todia()
        # Upper banded storage expected by solveh_banded: row 2 = main diagonal.
        banded = np.zeros((3, n))
        for k in range(3):
            banded[2 - k, k:] = penalty.diagonal(k)
        baselines = np.vstack([self._baseline(row, banded) for row in X])
        return X - baselines


class RegionSelector(_SpectralTransformer):
    """Keep only selected wavenumber regions.

    Parameters
    ----------
    wavenumbers : array-like of shape (n_wavenumbers,)
        Wavenumber axis of the incoming spectra (e.g. Raman shift in cm-1).
    regions : sequence of (low, high) tuples
        Inclusive ranges to keep, e.g. ``[(400, 1800)]`` or
        ``[(800, 1200), (1400, 1500)]``.
    """

    def __init__(self, wavenumbers=None, regions=None):
        self.wavenumbers = wavenumbers
        self.regions = regions

    def fit(self, X, y=None):
        X = self._check_fit_input(X)
        if self.wavenumbers is None or self.regions is None:
            raise ValueError("RegionSelector needs both wavenumbers and regions.")
        axis = np.asarray(self.wavenumbers, dtype=np.float64).ravel()
        if axis.shape[0] != X.shape[1]:
            raise ValueError(
                f"wavenumbers has {axis.shape[0]} values but X has {X.shape[1]} columns."
            )
        mask = np.zeros_like(axis, dtype=bool)
        for low, high in _as_regions(self.regions):
            mask |= (axis >= min(low, high)) & (axis <= max(low, high))
        if not mask.any():
            raise ValueError(f"No wavenumbers fall inside regions {self.regions}.")
        self.mask_ = mask
        self.selected_wavenumbers_ = axis[mask]
        return self

    def transform(self, X):
        X = self._check_transform_input(X)
        return X[:, self.mask_]


class EPO(_SpectralTransformer):
    """External Parameter Orthogonalisation (Roger et al., 2003).

    Estimates the directions of unwanted spectral variation ("clutter") and
    projects every spectrum onto the space orthogonal to them:
    ``X_epo = X (I - V V')``, where ``V`` holds the first ``n_components``
    right singular vectors of the clutter matrix ``D``.

    How ``D`` is built:

    * ``clutter`` given: those spectra are used as ``D`` directly, e.g.
      differences between the same sample measured at two temperatures or
      in two vessels.
    * otherwise (default, "y-gradient", as in PLS_Toolbox GLSW/EPO): the
      calibration spectra are sorted by ``y`` and ``D`` contains differences
      between neighbouring samples. Samples with nearly the same analyte
      concentration should have nearly the same spectrum, so what differs
      between them is interference (fluorescence, cell density, matrix).

    ``y`` is required in ``fit`` for the y-gradient mode. Inside
    :class:`~specpls.model.PLSModel` it is fitted on the training folds only.

    Parameters
    ----------
    n_components : int, default=2
        Number of clutter directions removed. Choose by RMSECV; too many
        removes analyte signal.
    clutter : array-like of shape (n_clutter, n_wavenumbers), optional
        Explicit clutter (difference) spectra, already preprocessed the same
        way as ``X``.
    """

    def __init__(self, n_components=2, clutter=None):
        self.n_components = n_components
        self.clutter = clutter

    def fit(self, X, y=None):
        X = self._check_fit_input(X)
        if self.clutter is not None:
            D = check_array(self.clutter, dtype=np.float64)
            if D.shape[1] != X.shape[1]:
                raise ValueError("clutter must have the same number of wavenumbers as X.")
        else:
            if y is None:
                raise ValueError("EPO needs y (y-gradient mode) or an explicit clutter matrix.")
            y = np.asarray(y, dtype=np.float64).ravel()
            if y.shape[0] != X.shape[0]:
                raise ValueError("y must have one value per spectrum.")
            order = np.argsort(y, kind="stable")
            D = np.diff(X[order], axis=0)
        max_k = min(D.shape)
        if not 0 <= self.n_components <= max_k:
            raise ValueError(f"n_components must be between 0 and {max_k}.")
        _, singular_values, vt = np.linalg.svd(D, full_matrices=False)
        self.loadings_ = vt[: self.n_components].T  # (n_wavenumbers, k)
        energy = singular_values**2
        self.explained_clutter_ = energy[: self.n_components] / energy.sum()
        return self

    def transform(self, X):
        X = self._check_transform_input(X)
        V = self.loadings_
        return X - (X @ V) @ V.T


class VariableSelector(_SpectralTransformer):
    """Keep a fixed subset of variables (e.g. intervals chosen by iPLS).

    Unlike :class:`RegionSelector` it works on column positions, so it can be
    placed *after* preprocessing.

    Parameters
    ----------
    mask : array-like of bool or int
        Boolean mask over the variables, or integer column indices.
    """

    def __init__(self, mask=None):
        self.mask = mask

    def fit(self, X, y=None):
        X = self._check_fit_input(X)
        if self.mask is None:
            raise ValueError("VariableSelector needs a mask.")
        mask = np.asarray(self.mask)
        if mask.dtype == bool:
            if mask.shape[0] != X.shape[1]:
                raise ValueError(
                    f"mask has {mask.shape[0]} entries but X has {X.shape[1]} columns."
                )
            self.mask_ = mask.copy()
        else:
            self.mask_ = np.zeros(X.shape[1], dtype=bool)
            self.mask_[mask.astype(int)] = True
        if not self.mask_.any():
            raise ValueError("mask selects no variables.")
        return self

    def transform(self, X):
        X = self._check_transform_input(X)
        return X[:, self.mask_]


def _as_regions(regions) -> list[tuple[float, float]]:
    regions = list(regions)
    if len(regions) == 2 and all(np.isscalar(v) for v in regions):
        regions = [tuple(regions)]
    out = []
    for region in regions:
        if len(region) != 2:
            raise ValueError(f"Each region must be (low, high); got {region!r}.")
        out.append((float(region[0]), float(region[1])))
    return out


PREPROCESSORS = {
    "snv": (SNV, ()),
    "msc": (MSC, ()),
    "savgol": (SavitzkyGolay, ("window_length", "polyorder", "deriv")),
    "sg": (SavitzkyGolay, ("window_length", "polyorder", "deriv")),
    "detrend": (Detrend, ("order",)),
    "asls": (AsLSBaseline, ("lam", "p", "n_iter")),
    "epo": (EPO, ("n_components",)),
}
"""Short names accepted by :func:`build_preprocessing` and the CLI."""


def build_preprocessing(spec: str | Sequence | None) -> Pipeline | None:
    """Build a preprocessing pipeline from a compact text specification.

    Steps are separated by commas and arguments by colons, applied in order.

    Examples
    --------
    >>> build_preprocessing("savgol:15:2:1,snv")   # 1st derivative, then SNV
    Pipeline(steps=[('savitzkygolay', SavitzkyGolay(deriv=1)), ('snv', SNV())])

    Parameters
    ----------
    spec : str, list of transformers, or None
        A string such as ``"asls:1e5:0.01,snv"``, a list of already-built
        transformers, or ``None``/``""`` for no preprocessing.

    Returns
    -------
    Pipeline or None
    """
    if spec is None or (isinstance(spec, str) and not spec.strip()):
        return None
    if not isinstance(spec, str):
        steps = list(spec)
        return make_pipeline(*steps) if steps else None

    steps = []
    for token in spec.split(","):
        name, *args = [part.strip() for part in token.strip().split(":")]
        key = name.lower()
        if key not in PREPROCESSORS:
            known = ", ".join(sorted(PREPROCESSORS))
            raise ValueError(f"Unknown preprocessing step {name!r}. Known: {known}.")
        cls, arg_names = PREPROCESSORS[key]
        if len(args) > len(arg_names):
            raise ValueError(f"{name!r} takes at most {len(arg_names)} arguments.")
        kwargs = {arg: _parse_number(value) for arg, value in zip(arg_names, args, strict=False)}
        steps.append(cls(**kwargs))
    return make_pipeline(*steps)


def _parse_number(text: str) -> int | float:
    value = float(text)
    return int(value) if value.is_integer() and "e" not in text.lower() else value
