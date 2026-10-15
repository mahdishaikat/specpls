"""PLS calibration model with built-in LV selection, diagnostics and persistence."""

from __future__ import annotations

import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import sklearn
from scipy import stats
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_array, check_is_fitted

from .preprocessing import RegionSelector, build_preprocessing
from .validation import make_cv, regression_metrics, rmse, select_n_components

__all__ = ["PLSModel", "load_model"]

_FORMAT_VERSION = 1


class PLSModel(RegressorMixin, BaseEstimator):
    """Single-response PLS calibration for spectroscopic data.

    Wraps preprocessing and :class:`sklearn.cross_decomposition.PLSRegression`
    in one scikit-learn estimator and adds what a chemometric workflow needs:
    automatic choice of latent variables from RMSECV, Hotelling's T2 and
    Q-residual diagnostics, VIP scores, and safe save/load for deployment.

    Preprocessing is fitted *inside* every cross-validation fold, so RMSECV is
    not optimistically biased by data-dependent steps such as MSC.

    Parameters
    ----------
    preprocessing : str, list of transformers, Pipeline or None
        Preprocessing applied before PLS. A string such as
        ``"savgol:15:2:1,snv"`` is parsed with
        :func:`specpls.preprocessing.build_preprocessing`.
    n_components : int or "auto", default="auto"
        Number of latent variables. ``"auto"`` picks it from RMSECV using
        ``lv_rule``.
    max_components : int, default=10
        Largest number of LVs tried when ``n_components="auto"``.
    cv : int, str or splitter, default=10
        Cross-validation scheme, see :func:`specpls.validation.make_cv`.
        Use ``"batch"`` together with ``groups`` in :meth:`fit` to hold out
        whole bioreactor runs.
    lv_rule : {"parsimonious", "min", "first_min"}, default="parsimonious"
        Rule for choosing LVs from the RMSECV curve.
    lv_tolerance : float, default=0.02
        Relative tolerance for the parsimonious rule.
    regions : list of (low, high), optional
        Wavenumber regions to keep, e.g. ``[(400, 1800)]``. Applied before
        preprocessing. Requires ``wavenumbers`` in :meth:`fit`.
    scale : bool, default=False
        Autoscale variables inside PLS. Spectra are normally only
        mean-centred, so the default is ``False`` (unlike scikit-learn).
    max_cv_ratio : float, optional
        Over-fitting guard for automatic LV selection: only LVs with
        ``RMSECV / RMSEC <= max_cv_ratio`` are eligible (1.2 = RMSECV at most
        20 % above RMSEC). A warning is issued if no LV qualifies.

    Attributes
    ----------
    n_components_ : int
        Number of LVs in the final model.
    rmsecv_curve_ : ndarray
        RMSECV for 1..max_components LVs (``None`` if LVs were fixed).
    rmsec_curve_ : ndarray
        RMSEC for the same LVs (``None`` if LVs were fixed).
    ratio_ok_ : bool
        False if ``max_cv_ratio`` was set and no LV satisfied it.
    cv_predictions_ : ndarray
        Cross-validated predictions for the training samples.
    pipeline_ : Pipeline
        The fitted region selection + preprocessing + PLS pipeline.
    calibration_ : dict
        Figures of merit for calibration (RMSEC) and cross-validation (RMSECV).
    wavenumbers_ : ndarray or None
        Wavenumber axis the model was trained on.
    target_name_ : str or None
        Name of the predicted variable, e.g. ``"glucose"``.

    Examples
    --------
    >>> from specpls import PLSModel
    >>> from specpls.datasets import make_synthetic_cell_culture
    >>> data = make_synthetic_cell_culture(random_state=0)
    >>> model = PLSModel(preprocessing="savgol:15:2:1,snv", max_components=8)
    >>> model = model.fit(data.X, data.target("glucose"),
    ...                   wavenumbers=data.wavenumbers, target_name="glucose")
    >>> model.n_components_ > 0
    True
    """

    def __init__(
        self,
        preprocessing=None,
        n_components="auto",
        max_components=10,
        cv=10,
        lv_rule="parsimonious",
        lv_tolerance=0.02,
        regions=None,
        scale=False,
        max_cv_ratio=None,
    ):
        self.preprocessing = preprocessing
        self.n_components = n_components
        self.max_components = max_components
        self.cv = cv
        self.lv_rule = lv_rule
        self.lv_tolerance = lv_tolerance
        self.regions = regions
        self.scale = scale
        self.max_cv_ratio = max_cv_ratio

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def _make_pipeline(self, n_components, wavenumbers):
        steps = []
        if self.regions is not None:
            if wavenumbers is None:
                raise ValueError("`regions` requires `wavenumbers` to be passed to fit().")
            steps.append(("regions", RegionSelector(wavenumbers, self.regions)))
        prep = self.preprocessing
        if isinstance(prep, Pipeline):
            prep = clone(prep)
        elif isinstance(prep, (list, tuple)):
            prep = build_preprocessing([clone(step) for step in prep])
        else:
            prep = build_preprocessing(prep)
        if prep is not None:
            steps.append(("preprocessing", prep))
        steps.append(("pls", PLSRegression(n_components=n_components, scale=self.scale)))
        return Pipeline(steps)

    def fit(self, X, y, *, wavenumbers=None, groups=None, target_name=None):
        """Fit the model, choosing the number of LVs by cross-validation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_wavenumbers)
            Calibration spectra.
        y : array-like of shape (n_samples,)
            Reference values (e.g. glucose in g/L).
        wavenumbers : array-like of shape (n_wavenumbers,), optional
            Wavenumber axis. Stored with the model and checked at prediction
            time; required if ``regions`` is set.
        groups : array-like of shape (n_samples,), optional
            Batch / run labels for group-aware cross-validation.
        target_name : str, optional
            Name of the response, stored for reporting.

        Returns
        -------
        self
        """
        X = check_array(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if y.ndim > 1:
            if y.ndim == 2 and y.shape[1] == 1:
                y = y.ravel()
            else:
                raise ValueError("PLSModel handles one response; fit one model per analyte.")
        if y.shape[0] != X.shape[0]:
            raise ValueError(f"X has {X.shape[0]} samples but y has {y.shape[0]}.")
        if not np.all(np.isfinite(y)):
            raise ValueError("y contains NaN or infinite values.")
        if wavenumbers is not None:
            wavenumbers = np.asarray(wavenumbers, dtype=np.float64).ravel()
            if wavenumbers.shape[0] != X.shape[1]:
                raise ValueError(
                    f"wavenumbers has {wavenumbers.shape[0]} values "
                    f"but X has {X.shape[1]} columns."
                )
        if groups is not None:
            groups = np.asarray(groups)

        splitter = make_cv(self.cv, groups)
        folds = self._preprocess_folds(splitter, X, y, groups, wavenumbers)
        ceiling = max(1, min(min(len(tr) - 1, Xtr.shape[1]) for tr, _, Xtr, _ in folds))

        if self.n_components == "auto":
            max_lv = min(int(self.max_components), ceiling)
            if max_lv < int(self.max_components):
                warnings.warn(
                    f"max_components reduced from {self.max_components} to {max_lv} "
                    "because of the number of samples/variables.",
                    stacklevel=2,
                )
            lv_list = list(range(1, max_lv + 1))
            cv_preds = self._cv_predict(folds, y, lv_list)
            curve = np.array([rmse(y, p) for p in cv_preds])
            cal_curve = self._calibration_curve(X, y, wavenumbers, lv_list)
            choice = select_n_components(
                curve,
                self.lv_rule,
                self.lv_tolerance,
                rmsec=cal_curve,
                max_ratio=self.max_cv_ratio,
            )
            if not choice.ratio_ok:
                warnings.warn(
                    f"No number of LVs gives RMSECV/RMSEC <= {self.max_cv_ratio}; "
                    f"using {choice.n_components} LV (smallest ratio).",
                    stacklevel=2,
                )
            self.n_components_ = choice.n_components
            self.rmsecv_curve_ = curve
            self.rmsec_curve_ = cal_curve
            self.ratio_ok_ = choice.ratio_ok
            self.cv_predictions_ = cv_preds[self.n_components_ - 1]
        else:
            n = int(self.n_components)
            if not 1 <= n <= ceiling:
                raise ValueError(f"n_components must be between 1 and {ceiling}; got {n}.")
            self.n_components_ = n
            self.rmsecv_curve_ = None
            self.rmsec_curve_ = None
            self.cv_predictions_ = self._cv_predict(folds, y, [n])[0]

        self.pipeline_ = self._make_pipeline(self.n_components_, wavenumbers).fit(X, y)
        self.n_features_in_ = X.shape[1]
        self.wavenumbers_ = wavenumbers
        self.target_name_ = target_name

        fitted = self.predict(X)
        rmsec_value = rmse(y, fitted)
        rmsecv_value = rmse(y, self.cv_predictions_)
        self.calibration_ = {
            "rmsec": rmsec_value,
            "r2_cal": regression_metrics(y, fitted)["r2"],
            "rmsecv": rmsecv_value,
            "r2_cv": regression_metrics(y, self.cv_predictions_)["r2"],
            "cv_ratio": rmsecv_value / rmsec_value if rmsec_value > 0 else float("inf"),
            "n_samples": int(X.shape[0]),
        }
        if self.n_components != "auto":
            self.ratio_ok_ = (
                self.max_cv_ratio is None
                or self.calibration_["cv_ratio"] <= self.max_cv_ratio
            )

        # Statistics needed for T2 / Q control limits.
        scores = self._scores_from_processed(self._processed(X))
        self._train_scores = scores
        self._score_var = scores.var(axis=0, ddof=1)
        self._q_cal = self._q_from_processed(self._processed(X))
        self._n_cal = X.shape[0]
        return self

    def _preprocess_folds(self, splitter, X, y, groups, wavenumbers):
        """Fit preprocessing once per CV fold and cache the transformed data.

        Preprocessing does not depend on the number of LVs, so this avoids
        refitting it for every candidate model.
        """
        folds = []
        for train, test in splitter.split(X, y, groups=groups):
            prep = self._make_pipeline(1, wavenumbers)[:-1]
            if len(prep):
                Xtr = prep.fit_transform(X[train], y[train])
                Xte = prep.transform(X[test])
            else:
                Xtr, Xte = X[train], X[test]
            folds.append((train, test, Xtr, Xte))
        if not folds:
            raise ValueError("Cross-validation produced no folds.")
        n_tested = np.bincount(np.concatenate([f[1] for f in folds]), minlength=len(y))
        if not np.all(n_tested == 1):
            raise ValueError(
                "The cross-validation scheme must test every sample exactly once "
                "(e.g. venetian blinds, k-fold, leave-one-out or leave-one-batch-out)."
            )
        return folds

    def _calibration_curve(self, X, y, wavenumbers, lv_list):
        """RMSEC for each number of LVs (preprocessing fitted once on all data)."""
        prep = self._make_pipeline(1, wavenumbers)[:-1]
        Xp = prep.fit_transform(X, y) if len(prep) else X
        return np.array(
            [
                rmse(y, PLSRegression(n_components=a, scale=self.scale).fit(Xp, y).predict(Xp))
                for a in lv_list
            ]
        )

    def _cv_predict(self, folds, y, lv_list):
        """Cross-validated predictions, shape ``(len(lv_list), n_samples)``."""
        preds = np.empty((len(lv_list), len(y)))
        for train, test, Xtr, Xte in folds:
            for i, a in enumerate(lv_list):
                pls = PLSRegression(n_components=a, scale=self.scale).fit(Xtr, y[train])
                preds[i, test] = pls.predict(Xte).ravel()
        return preds

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def _check_X(self, X, wavenumbers=None):
        check_is_fitted(self, "pipeline_")
        X = check_array(X, dtype=np.float64)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} wavenumbers; model expects {self.n_features_in_}."
            )
        if wavenumbers is not None and self.wavenumbers_ is not None:
            axis = np.asarray(wavenumbers, dtype=np.float64).ravel()
            if axis.shape != self.wavenumbers_.shape or not np.allclose(
                axis, self.wavenumbers_, atol=0.5
            ):
                raise ValueError(
                    "Wavenumber axis of new spectra does not match the calibration axis. "
                    "Interpolate the spectra onto `model.wavenumbers_` first."
                )
        return X

    def predict(self, X, *, wavenumbers=None):
        """Predict the response for new spectra.

        If ``wavenumbers`` is given, it is checked against the calibration
        axis, which guards against applying a model to spectra from a
        different instrument set-up.
        """
        X = self._check_X(X, wavenumbers)
        return self.pipeline_.predict(X).ravel()

    def evaluate(self, X, y, *, wavenumbers=None) -> dict[str, float]:
        """Figures of merit on an independent test set (RMSEP, R2, bias, ...)."""
        y_pred = self.predict(X, wavenumbers=wavenumbers)
        metrics = regression_metrics(y, y_pred)
        return {"rmsep" if k == "rmse" else k: v for k, v in metrics.items()}

    def summary(self, X_test=None, y_test=None) -> dict:
        """One-row summary in the style of a publication table.

        Returns LVs, RMSEC, R2 cal, RMSECV, R2 CV and, if a test set is given,
        RMSEP and R2 test.
        """
        check_is_fitted(self, "pipeline_")
        row = {
            "target": self.target_name_,
            "preprocessing": _describe(self.preprocessing),
            "lvs": self.n_components_,
            **{k: v for k, v in self.calibration_.items() if k != "n_samples"},
        }
        if X_test is not None and y_test is not None:
            test = self.evaluate(X_test, y_test)
            row["rmsep"] = test["rmsep"]
            row["r2_test"] = test["r2"]
        return row

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    @property
    def pls_(self) -> PLSRegression:
        """The fitted :class:`~sklearn.cross_decomposition.PLSRegression` step."""
        check_is_fitted(self, "pipeline_")
        return self.pipeline_.named_steps["pls"]

    def _processed(self, X):
        return self.pipeline_[:-1].transform(X) if len(self.pipeline_) > 1 else X

    def _scores_from_processed(self, Xp):
        return self.pls_.transform(Xp)

    def _q_from_processed(self, Xp):
        scores = self.pls_.transform(Xp)
        reconstructed = self.pls_.inverse_transform(scores)
        return np.sum((Xp - reconstructed) ** 2, axis=1)

    def transform(self, X):
        """Preprocessed spectra (input to PLS) for inspection or plotting."""
        return self._processed(self._check_X(X))

    def scores(self, X) -> np.ndarray:
        """Latent-variable scores, shape ``(n_samples, n_components_)``."""
        return self._scores_from_processed(self.transform(X))

    def hotelling_t2(self, X) -> np.ndarray:
        """Hotelling's T2 for each sample (distance within the model plane)."""
        t = self.scores(X)
        return np.sum(t**2 / self._score_var, axis=1)

    def q_residuals(self, X) -> np.ndarray:
        """Q residuals (squared prediction error, SPE) for each sample."""
        return self._q_from_processed(self.transform(X))

    def t2_limit(self, confidence=0.95) -> float:
        """T2 control limit based on the F distribution.

        ``A (n-1) / (n-A) * F(confidence; A, n-A)`` with ``A`` LVs and ``n``
        calibration samples.
        """
        check_is_fitted(self, "pipeline_")
        a, n = self.n_components_, self._n_cal
        if n <= a:
            return float("nan")
        return float(a * (n - 1) / (n - a) * stats.f.ppf(confidence, a, n - a))

    def q_limit(self, confidence=0.95) -> float:
        """Q control limit using Box's scaled chi-square approximation."""
        check_is_fitted(self, "pipeline_")
        q = self._q_cal
        mean, var = q.mean(), q.var(ddof=1)
        if mean <= 0 or var <= 0:
            return float("nan")
        g = var / (2 * mean)
        h = 2 * mean**2 / var
        return float(g * stats.chi2.ppf(confidence, h))

    def outliers(self, X, confidence=0.95) -> dict[str, np.ndarray]:
        """T2, Q and boolean flags for samples outside the control limits."""
        t2 = self.hotelling_t2(X)
        q = self.q_residuals(X)
        t2_lim, q_lim = self.t2_limit(confidence), self.q_limit(confidence)
        return {
            "t2": t2,
            "q": q,
            "t2_limit": t2_lim,
            "q_limit": q_lim,
            "t2_flag": t2 > t2_lim,
            "q_flag": q > q_lim,
        }

    def vip(self) -> np.ndarray:
        """Variable Importance in Projection for each (selected) wavenumber.

        Variables with VIP > 1 are conventionally considered important.
        The mean of VIP**2 over all variables equals 1.
        """
        pls = self.pls_
        w = pls.x_weights_  # (n_vars, A), columns have unit norm
        t = self._train_scores  # calibration scores, (n_samples, A)
        q = pls.y_loadings_[0]  # (A,)
        ss = np.sum(t**2, axis=0) * q**2  # y-variance explained per LV
        w_norm = w / np.linalg.norm(w, axis=0, keepdims=True)
        n_vars = w.shape[0]
        return np.sqrt(n_vars * (w_norm**2 @ ss) / ss.sum())

    @property
    def model_wavenumbers_(self):
        """Wavenumbers that actually enter PLS (after region selection)."""
        check_is_fitted(self, "pipeline_")
        steps = self.pipeline_.named_steps
        axis = self.wavenumbers_
        if "regions" in steps:
            axis = steps["regions"].selected_wavenumbers_
        if axis is not None and "preprocessing" in steps:
            for _, step in steps["preprocessing"].steps:
                if hasattr(step, "mask_") and step.mask_.shape[0] == axis.shape[0]:
                    axis = axis[step.mask_]
        return axis

    @property
    def coef_(self) -> np.ndarray:
        """Regression vector on the preprocessed variables."""
        return self.pls_.coef_.ravel()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path) -> Path:
        """Save the fitted model with metadata to a ``.joblib`` file."""
        check_is_fitted(self, "pipeline_")
        from . import __version__

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": _FORMAT_VERSION,
            "specpls_version": __version__,
            "sklearn_version": sklearn.__version__,
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model": self,
        }
        joblib.dump(payload, path)
        return path


def load_model(path) -> PLSModel:
    """Load a model saved with :meth:`PLSModel.save`.

    Warns if it was created with a different scikit-learn version, because
    pickled estimators are not guaranteed to be compatible across versions.

    Only load files from sources you trust: joblib files can execute code.
    """
    payload = joblib.load(Path(path))
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError(f"{path} is not a specpls model file.")
    if payload.get("format_version", 0) > _FORMAT_VERSION:
        raise ValueError("Model file was written by a newer specpls; please upgrade.")
    if payload.get("sklearn_version") != sklearn.__version__:
        warnings.warn(
            f"Model saved with scikit-learn {payload.get('sklearn_version')}, "
            f"running {sklearn.__version__}. Re-validate before use.",
            stacklevel=2,
        )
    return payload["model"]


def _describe(prep) -> str:
    if prep is None:
        return "none"
    if isinstance(prep, str):
        return prep
    steps = prep.steps if isinstance(prep, Pipeline) else [(None, s) for s in prep]
    return " -> ".join(type(step).__name__ for _, step in steps)
