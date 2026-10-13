"""Standard diagnostic plots for a fitted :class:`~specpls.model.PLSModel`.

Requires matplotlib (``pip install "specpls[plot]"``). Every function accepts
an optional ``ax`` and returns the axes, so plots can be combined in figures.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "plot_spectra",
    "plot_rmsecv",
    "plot_predicted_vs_measured",
    "plot_vip",
    "plot_t2_q",
    "plot_overview",
]


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            'Plotting needs matplotlib: pip install "specpls[plot]"'
        ) from exc
    return plt


def _axes(ax):
    return ax if ax is not None else _plt().subplots(figsize=(6, 4))[1]


def plot_spectra(wavenumbers, X, color_by=None, ax=None, cmap="viridis", label=None):
    """Overlay spectra, optionally coloured by a reference value."""
    plt = _plt()
    ax = _axes(ax)
    X = np.atleast_2d(X)
    if color_by is None:
        ax.plot(wavenumbers, X.T, lw=0.8, alpha=0.7)
    else:
        values = np.asarray(color_by, dtype=float)
        norm = plt.Normalize(values.min(), values.max())
        colormap = plt.get_cmap(cmap)
        for row, value in zip(X, values, strict=True):
            ax.plot(wavenumbers, row, lw=0.8, color=colormap(norm(value)))
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=colormap)
        ax.figure.colorbar(mappable, ax=ax, label=label)
    ax.set_xlabel("Raman shift (cm$^{-1}$)")
    ax.set_ylabel("Intensity")
    return ax


def plot_rmsecv(model, ax=None):
    """RMSECV versus number of latent variables, with the chosen LV marked."""
    ax = _axes(ax)
    if model.rmsecv_curve_ is None:
        raise ValueError("Model was fitted with fixed n_components; no RMSECV curve.")
    lvs = np.arange(1, len(model.rmsecv_curve_) + 1)
    ax.plot(lvs, model.rmsecv_curve_, "o-", color="0.3")
    ax.plot(
        model.n_components_,
        model.rmsecv_curve_[model.n_components_ - 1],
        "o",
        ms=11,
        mfc="none",
        mec="C3",
        mew=2,
        label=f"selected: {model.n_components_} LV",
    )
    ax.set_xlabel("Latent variables")
    ax.set_ylabel("RMSECV")
    ax.set_xticks(lvs)
    ax.legend(frameon=False)
    return ax


def plot_predicted_vs_measured(model, y_cal, X_test=None, y_test=None, ax=None, units=""):
    """Cross-validated (and optional test-set) predictions against reference."""
    ax = _axes(ax)
    y_cal = np.asarray(y_cal, dtype=float)
    cal = model.calibration_
    ax.scatter(
        y_cal,
        model.cv_predictions_,
        s=22,
        alpha=0.75,
        label=f"CV  RMSECV={cal['rmsecv']:.3g}  R²={cal['r2_cv']:.3f}",
    )
    values = [y_cal, model.cv_predictions_]
    if X_test is not None and y_test is not None:
        y_pred = model.predict(X_test)
        test = model.evaluate(X_test, y_test)
        ax.scatter(
            y_test,
            y_pred,
            s=30,
            marker="D",
            color="C3",
            label=f"Test  RMSEP={test['rmsep']:.3g}  R²={test['r2']:.3f}",
        )
        values += [np.asarray(y_test, float), y_pred]
    lo = min(v.min() for v in values)
    hi = max(v.max() for v in values)
    pad = 0.05 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1)
    name = model.target_name_ or "y"
    suffix = f" ({units})" if units else ""
    ax.set_xlabel(f"Measured {name}{suffix}")
    ax.set_ylabel(f"Predicted {name}{suffix}")
    ax.legend(frameon=False, fontsize=8)
    return ax


def plot_vip(model, ax=None, threshold=1.0):
    """VIP scores across the spectrum with the VIP = 1 threshold."""
    ax = _axes(ax)
    vip = model.vip()
    axis = model.model_wavenumbers_
    x = axis if axis is not None else np.arange(vip.size)
    ax.plot(x, vip, lw=1, color="0.25")
    ax.fill_between(x, threshold, vip, where=vip > threshold, color="C1", alpha=0.4)
    ax.axhline(threshold, color="C3", ls="--", lw=1)
    ax.set_xlabel("Raman shift (cm$^{-1}$)" if axis is not None else "Variable")
    ax.set_ylabel("VIP")
    return ax


def plot_t2_q(model, X, ax=None, confidence=0.95, labels=None):
    """Hotelling's T2 versus Q residuals with control limits."""
    ax = _axes(ax)
    out = model.outliers(X, confidence)
    flagged = out["t2_flag"] | out["q_flag"]
    ax.scatter(out["t2"][~flagged], out["q"][~flagged], s=22, alpha=0.75)
    ax.scatter(out["t2"][flagged], out["q"][flagged], s=30, color="C3", label="outside limit")
    ax.axvline(out["t2_limit"], color="C3", ls="--", lw=1)
    ax.axhline(out["q_limit"], color="C3", ls="--", lw=1)
    if labels is not None:
        for i in np.flatnonzero(flagged):
            ax.annotate(str(labels[i]), (out["t2"][i], out["q"][i]), fontsize=7)
    ax.set_xlabel(f"Hotelling's T² ({confidence:.0%} limit)")
    ax.set_ylabel(f"Q residuals ({confidence:.0%} limit)")
    if flagged.any():
        ax.legend(frameon=False)
    return ax


def plot_overview(model, X_cal, y_cal, X_test=None, y_test=None, units=""):
    """Four-panel figure: RMSECV, predicted vs measured, VIP, T2-Q."""
    plt = _plt()
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    if model.rmsecv_curve_ is not None:
        plot_rmsecv(model, ax=axes[0, 0])
    else:
        axes[0, 0].set_axis_off()
    plot_predicted_vs_measured(model, y_cal, X_test, y_test, ax=axes[0, 1], units=units)
    plot_vip(model, ax=axes[1, 0])
    plot_t2_q(model, X_cal, ax=axes[1, 1])
    title = model.target_name_ or "PLS model"
    fig.suptitle(f"{title}: {model.n_components_} LV", fontweight="bold")
    fig.tight_layout()
    return fig
