"""Interval PLS (iPLS) variable selection (Norgaard et al., 2000).

The spectrum is split into equal-width intervals. In *forward* mode the
interval that gives the lowest RMSECV is selected first, then the interval
that improves RMSECV most when added to it, and so on, until no interval
improves the model or ``max_intervals`` is reached.

Preprocessing is fitted once per CV fold on the full spectrum (so SNV and
EPO see the whole spectrum) and intervals are selected afterwards.

Note: RMSECV values reported during selection are optimistic, because the
same cross-validation is used to choose the intervals. Judge the final
model on an independent test set (RMSEP).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_array

from .preprocessing import build_preprocessing
from .validation import make_cv, rmse, select_n_components

__all__ = ["ipls", "IPLSResult"]


@dataclass
class IPLSResult:
    """Outcome of :func:`ipls`.

    Attributes
    ----------
    mask : ndarray of bool
        Selected variables (use with :class:`~specpls.preprocessing.VariableSelector`).
    selected : list of int
        Selected interval numbers, in the order they were added.
    intervals : list of (int, int)
        Start and end column (inclusive) of every interval.
    interval_rmsecv : ndarray
        RMSECV of a PLS model on each single interval.
    interval_lvs : ndarray
        LVs chosen for each single-interval model.
    global_rmsecv : float
        RMSECV using all variables.
    global_lvs : int
        LVs of the all-variable model.
    history : list of dict
        One entry per forward step: intervals, RMSECV, LVs.
    n_components : int
        LVs chosen for the final selected-variable model.
    rmsecv : float
        RMSECV of the final selected-variable model (optimistic, see module notes).
    rmsec : float
        RMSEC of the final selected-variable model.
    wavenumbers : ndarray or None
        Axis used for reporting interval ranges.
    """

    mask: np.ndarray
    selected: list
    intervals: list
    interval_rmsecv: np.ndarray
    interval_lvs: np.ndarray
    global_rmsecv: float
    global_lvs: int
    history: list = field(default_factory=list)
    n_components: int = 0
    rmsecv: float = float("nan")
    rmsec: float = float("nan")
    wavenumbers: np.ndarray | None = None
    max_cv_ratio: float | None = None

    @property
    def cv_ratio(self) -> float:
        """RMSECV / RMSEC of the selected model."""
        return self.rmsecv / self.rmsec if self.rmsec > 0 else float("inf")

    def interval_ranges(self) -> list[tuple[float, float]]:
        """(start, end) of the selected intervals on the wavenumber axis."""
        axis = self.wavenumbers
        out = []
        for i in sorted(self.selected):
            lo, hi = self.intervals[i]
            out.append((float(axis[lo]), float(axis[hi])) if axis is not None else (lo, hi))
        return out

    def summary(self) -> str:
        guard = (
            f" (only models with RMSECV/RMSEC <= {self.max_cv_ratio} allowed)"
            if self.max_cv_ratio
            else ""
        )
        global_text = (
            f"RMSECV {self.global_rmsecv:.4g} with {self.global_lvs} LV"
            if np.isfinite(self.global_rmsecv)
            else "no LV meets the ratio limit"
        )
        lines = [
            f"Global model (all variables): {global_text}{guard}",
            f"Selected {len(self.selected)} of {len(self.intervals)} intervals, "
            f"{int(self.mask.sum())} variables: RMSEC {self.rmsec:.4g}, "
            f"RMSECV {self.rmsecv:.4g}, ratio {self.cv_ratio:.2f}, {self.n_components} LV",
        ]
        for step in self.history:
            lines.append(
                f"  step {step['step']}: add interval {step['added']:>3} -> "
                f"RMSECV {step['rmsecv']:.4g} ({step['lvs']} LV)"
            )
        lines.append("Selected ranges: " + ", ".join(
            f"{lo:g}-{hi:g}" for lo, hi in self.interval_ranges()
        ))
        return "\n".join(lines)


def _prepare_folds(X, y, preprocessing, splitter, groups):
    folds = []
    for train, test in splitter.split(X, y, groups=groups):
        if preprocessing is None:
            Xtr, Xte = X[train], X[test]
        else:
            prep = clone(preprocessing)
            Xtr = prep.fit_transform(X[train], y[train])
            Xte = prep.transform(X[test])
        if Xtr.shape[1] != X.shape[1]:
            raise ValueError(
                "iPLS needs preprocessing that keeps the number of variables "
                "(SNV, MSC, Savitzky-Golay, EPO, baselines). Apply region selection "
                "inside the intervals instead."
            )
        folds.append((train, test, Xtr, Xte))
    return folds


def _curves(folds, X_cal, y, columns, max_components, scale):
    """RMSECV and RMSEC for 1..max_lv LVs using only ``columns``."""
    n_vars = int(columns.sum())
    smallest_train = min(len(f[0]) for f in folds)
    max_lv = max(1, min(max_components, n_vars, smallest_train - 1))
    preds = np.empty((max_lv, len(y)))
    for train, test, Xtr, Xte in folds:
        Xtr_c, Xte_c = Xtr[:, columns], Xte[:, columns]
        for a in range(1, max_lv + 1):
            pls = PLSRegression(n_components=a, scale=scale).fit(Xtr_c, y[train])
            preds[a - 1, test] = pls.predict(Xte_c).ravel()
    Xc = X_cal[:, columns]
    rmsec = [
        rmse(y, PLSRegression(n_components=a, scale=scale).fit(Xc, y).predict(Xc))
        for a in range(1, max_lv + 1)
    ]
    return np.array([rmse(y, p) for p in preds]), np.array(rmsec)


def ipls(
    X,
    y,
    *,
    preprocessing=None,
    n_intervals=20,
    max_intervals=4,
    max_components=10,
    cv=10,
    groups=None,
    lv_rule="parsimonious",
    lv_tolerance=0.02,
    min_improvement=0.0,
    max_cv_ratio=None,
    wavenumbers=None,
    scale=False,
) -> IPLSResult:
    """Forward interval PLS.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_wavenumbers)
    y : array-like of shape (n_samples,)
    preprocessing : str, list or Pipeline, optional
        Applied to the full spectrum inside every CV fold before selection,
        e.g. ``"snv,savgol:15:2:0,epo:2"``.
    n_intervals : int, default=20
        Number of equal-width intervals.
    max_intervals : int, default=4
        Maximum number of intervals to combine.
    max_components : int, default=10
        Largest number of LVs tried for each candidate model.
    cv, groups
        Cross-validation, see :func:`specpls.validation.make_cv`.
    lv_rule, lv_tolerance
        LV selection for each candidate, see
        :func:`specpls.validation.select_n_components`.
    min_improvement : float, default=0.0
        Relative RMSECV improvement needed to add another interval
        (e.g. 0.02 = must improve by 2 %).
    max_cv_ratio : float, optional
        Over-fitting guard: a candidate model is only accepted if some number
        of LVs gives ``RMSECV / RMSEC <= max_cv_ratio``; candidates that fail
        are skipped. If no single interval passes, an error is raised.
    wavenumbers : array-like, optional
        Axis for reporting interval ranges.

    Returns
    -------
    IPLSResult
    """
    X = check_array(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    n_vars = X.shape[1]
    if not 2 <= n_intervals <= n_vars:
        raise ValueError(f"n_intervals must be between 2 and {n_vars}.")
    if wavenumbers is not None:
        wavenumbers = np.asarray(wavenumbers, dtype=np.float64).ravel()
    if isinstance(preprocessing, (list, tuple)):
        preprocessing = build_preprocessing([clone(step) for step in preprocessing])
    elif not isinstance(preprocessing, Pipeline):
        preprocessing = build_preprocessing(preprocessing)

    splitter = make_cv(cv, groups)
    folds = _prepare_folds(X, y, preprocessing, splitter, groups)
    X_cal = clone(preprocessing).fit_transform(X, y) if preprocessing is not None else X
    chunks = np.array_split(np.arange(n_vars), n_intervals)
    intervals = [(int(c[0]), int(c[-1])) for c in chunks]

    def evaluate(selected):
        """(RMSECV, LVs, mask, RMSEC); RMSECV is inf if the ratio guard fails."""
        columns = np.zeros(n_vars, dtype=bool)
        for i in selected:
            columns[chunks[i]] = True
        cv_curve, cal_curve = _curves(folds, X_cal, y, columns, max_components, scale)
        choice = select_n_components(
            cv_curve, lv_rule, lv_tolerance, rmsec=cal_curve, max_ratio=max_cv_ratio
        )
        a = choice.n_components
        value = cv_curve[a - 1] if choice.ratio_ok else np.inf
        return value, a, columns, cal_curve[a - 1]

    global_rmsecv, global_lvs, _, _ = evaluate(range(n_intervals))
    single = [evaluate([i]) for i in range(n_intervals)]
    interval_rmsecv = np.array([s[0] for s in single])
    interval_lvs = np.array([s[1] for s in single])
    if not np.isfinite(interval_rmsecv).any():
        raise ValueError(
            f"No single interval gives RMSECV/RMSEC <= {max_cv_ratio}. "
            "Try fewer/wider intervals, different preprocessing or a larger limit."
        )

    best = int(np.argmin(interval_rmsecv))
    selected = [best]
    best_rmsecv, best_lvs, best_mask, best_rmsec = single[best]
    history = [{"step": 1, "added": best, "rmsecv": best_rmsecv, "lvs": best_lvs}]

    while len(selected) < min(max_intervals, n_intervals):
        candidates = [i for i in range(n_intervals) if i not in selected]
        trials = [(evaluate(selected + [i]), i) for i in candidates]
        trial, added = min(trials, key=lambda t: t[0][0])
        if not np.isfinite(trial[0]) or trial[0] >= best_rmsecv * (1 - min_improvement):
            break
        selected.append(added)
        best_rmsecv, best_lvs, best_mask, best_rmsec = trial
        history.append(
            {"step": len(selected), "added": added, "rmsecv": best_rmsecv, "lvs": best_lvs}
        )

    return IPLSResult(
        mask=best_mask,
        selected=selected,
        intervals=intervals,
        interval_rmsecv=interval_rmsecv,
        interval_lvs=interval_lvs,
        global_rmsecv=float(global_rmsecv),
        global_lvs=int(global_lvs),
        history=history,
        n_components=int(best_lvs),
        rmsecv=float(best_rmsecv),
        rmsec=float(best_rmsec),
        wavenumbers=wavenumbers,
        max_cv_ratio=max_cv_ratio,
    )
