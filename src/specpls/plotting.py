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
    "plot_ipls",
    "plot_scores",
    "plot_loadings",
]

CAL_STYLE = {"color": "#2a78d6", "marker": "o", "label": "Calibration"}
TEST_STYLE = {"color": "#eb6834", "marker": "D", "label": "Test"}


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError('Plotting needs matplotlib: pip install "specpls[plot]"') from exc
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
        s=36,
        alpha=0.85,
        color=CAL_STYLE["color"],
        marker=CAL_STYLE["marker"],
        edgecolor="white",
        lw=0.6,
        label=f"Calibration (CV)  RMSECV={cal['rmsecv']:.3g}  R²={cal['r2_cv']:.3f}",
    )
    values = [y_cal, model.cv_predictions_]
    if X_test is not None and y_test is not None:
        y_pred = model.predict(X_test)
        test = model.evaluate(X_test, y_test)
        ax.scatter(
            y_test,
            y_pred,
            s=40,
            marker=TEST_STYLE["marker"],
            color=TEST_STYLE["color"],
            edgecolor="white",
            lw=0.6,
            label=f"Test  RMSEP={test['rmsep']:.3g}  R²={test['r2']:.3f}",
        )
        values += [np.asarray(y_test, float), y_pred]
    lo = min(v.min() for v in values)
    hi = max(v.max() for v in values)
    pad = 0.05 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal", adjustable="box")
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


def plot_t2_q(model, X, ax=None, confidence=0.95, labels=None, X_test=None, test_labels=None):
    """Hotelling's T2 versus Q residuals with control limits.

    ``X`` is plotted as calibration (blue circles); an optional ``X_test`` is
    added as test samples (orange diamonds). Samples outside a limit are
    labelled when labels are given.
    """
    ax = _axes(ax)
    sets = [(X, labels, CAL_STYLE)]
    if X_test is not None:
        sets.append((X_test, test_labels, TEST_STYLE))
    for data, names, style in sets:
        out = model.outliers(data, confidence)
        ax.scatter(
            out["t2"],
            out["q"],
            s=36,
            alpha=0.85,
            color=style["color"],
            marker=style["marker"],
            label=style["label"],
            edgecolor="white",
            lw=0.6,
        )
        flagged = out["t2_flag"] | out["q_flag"]
        if names is not None:
            for i in np.flatnonzero(flagged):
                ax.annotate(
                    str(names[i]),
                    (out["t2"][i], out["q"][i]),
                    fontsize=7,
                    xytext=(3, 3),
                    textcoords="offset points",
                )
    ax.axvline(out["t2_limit"], color="0.35", ls="--", lw=1)
    ax.axhline(out["q_limit"], color="0.35", ls="--", lw=1)
    ax.set_xlabel(f"Hotelling's T² (dashed: {confidence:.0%} limit)")
    ax.set_ylabel(f"Q residuals (dashed: {confidence:.0%} limit)")
    if X_test is not None:
        ax.legend(frameon=False)
    return ax


def plot_scores(model, X, y=None, X_test=None, y_test=None, components=(1, 2), ax=None):
    """Score plot.

    With two or more LVs: LV ``components[0]`` versus LV ``components[1]``.
    With a one-LV model: LV1 score for each sample (sample order on x).
    Points are calibration (blue circles) and test (orange diamonds).
    """
    ax = _axes(ax)
    t_cal = model.scores(X)
    t_test = model.scores(X_test) if X_test is not None else None
    pls = model.pls_
    x_var = np.var(model.transform(X), axis=0).sum()
    explained = [
        np.var(np.outer(model._train_scores[:, a], pls.x_loadings_[:, a]), axis=0).sum() / x_var
        for a in range(model.n_components_)
    ]
    if model.n_components_ >= 2:
        i, j = components[0] - 1, components[1] - 1
        ax.scatter(t_cal[:, i], t_cal[:, j], s=36, edgecolor="white", lw=0.6, **CAL_STYLE)
        if t_test is not None:
            ax.scatter(t_test[:, i], t_test[:, j], s=40, edgecolor="white", lw=0.6, **TEST_STYLE)
        ax.set_xlabel(f"LV{i + 1} scores ({explained[i]:.1%} of X variance)")
        ax.set_ylabel(f"LV{j + 1} scores ({explained[j]:.1%} of X variance)")
        ax.axvline(0, color="0.8", lw=0.8)
    else:
        n_cal = t_cal.shape[0]
        ax.scatter(
            np.arange(1, n_cal + 1), t_cal[:, 0], s=36, edgecolor="white", lw=0.6, **CAL_STYLE
        )
        if t_test is not None:
            ax.scatter(
                np.arange(n_cal + 1, n_cal + 1 + t_test.shape[0]),
                t_test[:, 0],
                s=40,
                edgecolor="white",
                lw=0.6,
                **TEST_STYLE,
            )
        ax.set_xlabel("Sample")
        ax.set_ylabel(f"LV1 scores ({explained[0]:.1%} of X variance)")
    ax.axhline(0, color="0.8", lw=0.8)
    ax.legend(frameon=False)
    return ax


def plot_loadings(model, components=None, ax=None):
    """X-loadings of the selected LVs across the spectrum."""
    ax = _axes(ax)
    loadings = model.pls_.x_loadings_
    axis = model.model_wavenumbers_
    x = axis if axis is not None else np.arange(loadings.shape[0])
    comps = components or range(1, model.n_components_ + 1)
    for k, a in enumerate(comps):
        ax.plot(
            x,
            loadings[:, a - 1],
            lw=1.2,
            color=["#2a78d6", "#eb6834", "#1baf7a", "#eda100"][k % 4],
            label=f"LV{a}",
        )
    ax.axhline(0, color="0.8", lw=0.8)
    ax.set_xlabel("Raman shift (cm$^{-1}$)" if axis is not None else "Variable")
    ax.set_ylabel("Loading")
    if len(list(comps)) > 1:
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


def plot_ipls(result, X=None, ax=None):
    """Classic iPLS plot: RMSECV of each interval, selected intervals highlighted.

    The dashed line is the all-variable model; the grey curve (if ``X`` is
    given) is the mean spectrum for orientation.
    """
    ax = _axes(ax)
    axis = result.wavenumbers
    n_vars = result.mask.shape[0]
    x = axis if axis is not None else np.arange(n_vars)
    finite = np.isfinite(result.interval_rmsecv)
    heights = np.where(finite, result.interval_rmsecv, 0.0)
    for i, (lo, hi) in enumerate(result.intervals):
        chosen = i in result.selected
        if not finite[i]:
            ax.text((x[lo] + x[hi]) / 2, 0, "x", ha="center", va="bottom", color="0.5")
            continue
        ax.bar(
            (x[lo] + x[hi]) / 2,
            heights[i],
            width=(x[hi] - x[lo]) * 0.95,
            color="C1" if chosen else "C0",
            alpha=0.85 if chosen else 0.45,
            edgecolor="none",
        )
        ax.text(
            (x[lo] + x[hi]) / 2,
            heights[i],
            str(result.interval_lvs[i]),
            ha="center",
            va="bottom",
            fontsize=7,
        )
    if np.isfinite(result.global_rmsecv):
        ax.axhline(
            result.global_rmsecv,
            color="C3",
            ls="--",
            lw=1,
            label=f"all variables ({result.global_lvs} LV)",
        )
    if X is not None:
        mean = np.asarray(X, dtype=float).mean(axis=0)
        scaled = (mean - mean.min()) / (np.ptp(mean) or 1) * heights.max()
        ax.plot(x, scaled, color="0.4", lw=0.8, label="mean spectrum (scaled)")
    ax.set_xlabel("Raman shift (cm$^{-1}$)" if axis is not None else "Variable")
    ax.set_ylabel("RMSECV (single interval)")
    title = "iPLS: orange = selected; numbers = LVs"
    if result.max_cv_ratio:
        title += f"; x = fails RMSECV/RMSEC <= {result.max_cv_ratio}"
    ax.set_title(title, fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    return ax
