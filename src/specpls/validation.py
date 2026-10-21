"""Figures of merit and cross-validation schemes for calibration models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import (
    BaseCrossValidator,
    KFold,
    LeaveOneGroupOut,
    LeaveOneOut,
)

__all__ = [
    "rmse",
    "r2",
    "bias",
    "sep",
    "rpd",
    "regression_metrics",
    "VenetianBlinds",
    "make_cv",
    "select_n_components",
    "LVSelection",
]


# --------------------------------------------------------------------------
# Figures of merit
# --------------------------------------------------------------------------
def _pair(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}.")
    if y_true.size == 0:
        raise ValueError("Empty input.")
    return y_true, y_pred


def rmse(y_true, y_pred) -> float:
    """Root mean square error. Used for RMSEC, RMSECV and RMSEP."""
    y_true, y_pred = _pair(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2(y_true, y_pred) -> float:
    """Coefficient of determination, ``1 - SS_res / SS_tot``."""
    y_true, y_pred = _pair(y_true, y_pred)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / ss_tot)


def bias(y_true, y_pred) -> float:
    """Mean of ``y_pred - y_true``. Positive means the model over-predicts."""
    y_true, y_pred = _pair(y_true, y_pred)
    return float(np.mean(y_pred - y_true))


def sep(y_true, y_pred) -> float:
    """Standard error of prediction, corrected for bias.

    ``RMSEP**2 ~= SEP**2 + bias**2`` (exact up to the n / n-1 factor).
    """
    y_true, y_pred = _pair(y_true, y_pred)
    if y_true.size < 2:
        return float("nan")
    return float(np.std(y_pred - y_true, ddof=1))


def rpd(y_true, y_pred) -> float:
    """Ratio of performance to deviation: ``SD(y_true) / SEP``."""
    y_true, y_pred = _pair(y_true, y_pred)
    error = sep(y_true, y_pred)
    if not error or np.isnan(error):
        return float("nan")
    return float(np.std(y_true, ddof=1) / error)


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """All figures of merit as a dict: rmse, r2, bias, sep, rpd, n."""
    return {
        "rmse": rmse(y_true, y_pred),
        "r2": r2(y_true, y_pred),
        "bias": bias(y_true, y_pred),
        "sep": sep(y_true, y_pred),
        "rpd": rpd(y_true, y_pred),
        "n": int(np.asarray(y_true).size),
    }


# --------------------------------------------------------------------------
# Cross-validation
# --------------------------------------------------------------------------
class VenetianBlinds(BaseCrossValidator):
    """Venetian-blinds cross-validation (sample ``i`` goes to fold ``i % k``).

    This is the default scheme in many chemometrics packages (e.g. "vet" in
    PLS_Toolbox). It assumes samples are ordered, e.g. by time or
    concentration, so that every fold spans the whole range.

    Parameters
    ----------
    n_splits : int, default=10
        Number of folds.
    """

    def __init__(self, n_splits=10):
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2.")
        self.n_splits = n_splits

    def _iter_test_indices(self, X=None, y=None, groups=None):
        n_samples = len(X)
        if self.n_splits > n_samples:
            raise ValueError(f"Cannot have n_splits={self.n_splits} > n_samples={n_samples}.")
        indices = np.arange(n_samples)
        for fold in range(self.n_splits):
            yield indices[fold :: self.n_splits]

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


def make_cv(cv=10, groups=None):
    """Turn a short description into a scikit-learn CV splitter.

    Parameters
    ----------
    cv : int, str or splitter
        * ``int`` - venetian blinds with that many folds
        * ``"venetian:k"`` - venetian blinds, k folds
        * ``"kfold:k"`` - contiguous blocks, k folds (no shuffling)
        * ``"loo"`` - leave-one-out
        * ``"batch"`` / ``"group"`` - leave-one-group-out; requires ``groups``.
          Use this for bioprocess data so that all samples from one batch or
          bioreactor run are held out together.
        * any object with a ``split`` method is returned unchanged.
    groups : array-like, optional
        Group labels (e.g. batch IDs). Required for ``"batch"``.
    """
    if not isinstance(cv, str) and hasattr(cv, "split"):
        return cv
    if isinstance(cv, (bool, np.bool_)):
        raise TypeError(f"Unsupported cv specification: {cv!r}")
    if isinstance(cv, (int, np.integer)):
        return VenetianBlinds(int(cv))
    if not isinstance(cv, str):
        raise TypeError(f"Unsupported cv specification: {cv!r}")

    name, _, arg = cv.lower().partition(":")
    if name in {"venetian", "vet"}:
        return VenetianBlinds(int(arg or 10))
    if name in {"kfold", "con", "contiguous"}:
        return KFold(n_splits=int(arg or 10), shuffle=False)
    if name == "loo":
        return LeaveOneOut()
    if name in {"batch", "group", "logo"}:
        if groups is None:
            raise ValueError("cv='batch' needs `groups` (e.g. batch IDs).")
        return LeaveOneGroupOut()
    raise ValueError(f"Unknown cv specification: {cv!r}")


# --------------------------------------------------------------------------
# Choosing the number of latent variables
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class LVSelection:
    """Result of latent-variable selection.

    ``ratio_ok`` is False when a ``max_ratio`` was requested but no number of
    LVs satisfied it; the LV count with the smallest RMSECV/RMSEC ratio is
    returned in that case.
    """

    n_components: int
    rule: str
    rmsecv: np.ndarray
    ratio_ok: bool = True


def select_n_components(
    rmsecv, rule="parsimonious", tolerance=0.02, rmsec=None, max_ratio=None
) -> LVSelection:
    """Pick the number of latent variables from an RMSECV curve.

    Parameters
    ----------
    rmsecv : array-like
        RMSECV for 1, 2, ..., A latent variables.
    rule : {"parsimonious", "min", "first_min"}
        * ``"min"`` - global minimum. Tends to over-fit on small data sets.
        * ``"first_min"`` - first local minimum of the curve.
        * ``"parsimonious"`` (default) - fewest LVs whose RMSECV is within
          ``tolerance`` (relative) of the global minimum. Trades a negligible
          loss of accuracy for a simpler, more robust model.
    tolerance : float, default=0.02
        Relative tolerance for the parsimonious rule (0.02 = 2 %).
    rmsec : array-like, optional
        RMSEC for the same LVs. Required with ``max_ratio``.
    max_ratio : float, optional
        Over-fitting guard: only LVs with ``RMSECV / RMSEC <= max_ratio`` are
        eligible (e.g. 1.2 means RMSECV may exceed RMSEC by at most 20 %).
        The rule is then applied among the eligible LVs.

    Returns
    -------
    LVSelection
    """
    curve = np.asarray(rmsecv, dtype=np.float64).ravel()
    if curve.size == 0 or not np.all(np.isfinite(curve)):
        raise ValueError("rmsecv must be a non-empty array of finite values.")
    if rule not in {"min", "first_min", "parsimonious"}:
        raise ValueError(f"Unknown rule {rule!r}.")
    if rule == "parsimonious" and tolerance < 0:
        raise ValueError("tolerance must be >= 0.")

    eligible = np.ones(curve.size, dtype=bool)
    if max_ratio is not None:
        if rmsec is None:
            raise ValueError("max_ratio needs the rmsec curve.")
        cal = np.asarray(rmsec, dtype=np.float64).ravel()
        if cal.shape != curve.shape:
            raise ValueError("rmsec and rmsecv must have the same length.")
        ratio = curve / np.maximum(cal, np.finfo(float).tiny)
        eligible = ratio <= max_ratio
        if not eligible.any():
            n = int(np.argmin(ratio)) + 1
            return LVSelection(n_components=n, rule=rule, rmsecv=curve, ratio_ok=False)

    masked = np.where(eligible, curve, np.inf)
    if rule == "min":
        n = int(np.argmin(masked)) + 1
    elif rule == "first_min":
        idx = np.flatnonzero(eligible)
        n = int(idx[-1]) + 1
        for i, j in zip(idx[:-1], idx[1:], strict=True):
            if curve[j] >= curve[i]:
                n = int(i) + 1
                break
    else:
        threshold = masked.min() * (1.0 + tolerance)
        n = int(np.argmax(masked <= threshold)) + 1
    return LVSelection(n_components=n, rule=rule, rmsecv=curve)
