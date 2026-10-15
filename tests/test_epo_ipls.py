import numpy as np
import pytest

from specpls import EPO, SNV, PLSModel, VariableSelector, build_preprocessing, ipls


# ----------------------------------------------------------------- EPO
@pytest.fixture
def clutter_data(rng):
    """Analyte signal plus a strong interferent whose size is unrelated to y."""
    axis = np.linspace(0, 1, 120)
    analyte = np.exp(-((axis - 0.3) ** 2) / 0.002)
    interferent = np.exp(-((axis - 0.6) ** 2) / 0.05)
    y = rng.uniform(1, 5, 40)
    amount = rng.uniform(0, 10, 40)
    X = np.outer(y, analyte) + np.outer(amount, interferent) + rng.normal(0, 1e-3, (40, 120))
    return X, y, interferent


def test_epo_removes_interferent_direction(clutter_data):
    X, y, interferent = clutter_data
    epo = EPO(n_components=1).fit(X, y)
    direction = interferent / np.linalg.norm(interferent)
    before = np.abs(X @ direction).max()
    after = np.abs(epo.transform(X) @ direction).max()
    assert after < 0.05 * before


def test_epo_projection_is_idempotent(clutter_data):
    X, y, _ = clutter_data
    epo = EPO(n_components=2).fit(X, y)
    once = epo.transform(X)
    np.testing.assert_allclose(epo.transform(once), once, atol=1e-10)


def test_epo_zero_components_is_identity(clutter_data):
    X, y, _ = clutter_data
    np.testing.assert_allclose(EPO(n_components=0).fit_transform(X, y), X)


def test_epo_with_explicit_clutter(clutter_data):
    X, y, interferent = clutter_data
    epo = EPO(n_components=1, clutter=np.vstack([interferent, 2 * interferent])).fit(X)
    direction = interferent / np.linalg.norm(interferent)
    np.testing.assert_allclose(np.abs(epo.loadings_[:, 0]), np.abs(direction), atol=1e-10)


def test_epo_needs_y_or_clutter(clutter_data):
    X, _, _ = clutter_data
    with pytest.raises(ValueError, match="needs y"):
        EPO().fit(X)


def test_epo_too_many_components(clutter_data):
    X, y, _ = clutter_data
    with pytest.raises(ValueError, match="n_components"):
        EPO(n_components=500).fit(X, y)


def test_epo_improves_pls_on_clutter(clutter_data):
    X, y, _ = clutter_data
    plain = PLSModel(n_components=1, cv=5).fit(X, y)
    with_epo = PLSModel(preprocessing="epo:1", n_components=1, cv=5).fit(X, y)
    assert with_epo.calibration_["rmsecv"] < 0.5 * plain.calibration_["rmsecv"]


def test_epo_string_spec():
    step = build_preprocessing("snv,epo:3").steps[1][1]
    assert isinstance(step, EPO) and step.n_components == 3


# ----------------------------------------------------------------- VariableSelector
def test_variable_selector_mask_and_indices():
    X = np.arange(20.0).reshape(2, 10)
    mask = np.zeros(10, dtype=bool)
    mask[[1, 4]] = True
    np.testing.assert_array_equal(VariableSelector(mask).fit_transform(X), X[:, [1, 4]])
    np.testing.assert_array_equal(VariableSelector([1, 4]).fit_transform(X), X[:, [1, 4]])


def test_variable_selector_errors():
    X = np.ones((2, 5))
    with pytest.raises(ValueError, match="entries"):
        VariableSelector(np.ones(3, dtype=bool)).fit(X)
    with pytest.raises(ValueError, match="no variables"):
        VariableSelector(np.zeros(5, dtype=bool)).fit(X)


def test_model_wavenumbers_follow_variable_selector(culture):
    mask = np.zeros(culture.X.shape[1], dtype=bool)
    mask[100:200] = True
    model = PLSModel(preprocessing=[SNV(), VariableSelector(mask)], n_components=2, cv=5)
    model.fit(culture.X, culture.target("glucose"), wavenumbers=culture.wavenumbers)
    np.testing.assert_array_equal(model.model_wavenumbers_, culture.wavenumbers[100:200])
    assert model.vip().shape == (100,)


# ----------------------------------------------------------------- iPLS
@pytest.fixture
def one_good_interval(rng):
    """Only variables 40-49 carry the analyte; the rest is noise."""
    y = rng.uniform(0, 10, 50)
    X = rng.normal(0, 1, (50, 100))
    X[:, 40:50] += np.outer(y, np.linspace(1, 2, 10))
    return X, y


def test_ipls_finds_informative_interval(one_good_interval):
    X, y = one_good_interval
    result = ipls(X, y, n_intervals=10, max_intervals=3, cv=5)
    assert result.selected[0] == 4
    assert result.mask[40:50].all()
    assert result.rmsecv < result.global_rmsecv
    assert np.argmin(result.interval_rmsecv) == 4


def test_ipls_history_improves_monotonically(one_good_interval):
    X, y = one_good_interval
    result = ipls(X, y, n_intervals=10, max_intervals=4, cv=5)
    values = [step["rmsecv"] for step in result.history]
    assert values == sorted(values, reverse=True)
    assert len(result.selected) == len(result.history) <= 4


def test_ipls_with_preprocessing_and_axis(one_good_interval):
    X, y = one_good_interval
    axis = np.arange(1000.0, 1100.0)
    result = ipls(X, y, preprocessing="savgol:5:2:0", n_intervals=10, max_intervals=1,
                  cv=5, wavenumbers=axis)
    assert result.interval_ranges() == [(1040.0, 1049.0)]
    assert "Selected ranges" in result.summary()


def test_ipls_rejects_width_changing_preprocessing(one_good_interval, rng):
    X, y = one_good_interval
    with pytest.raises(ValueError, match="keeps the number of variables"):
        ipls(X, y, preprocessing=[VariableSelector(np.arange(50))], n_intervals=5, cv=5)


def test_ipls_invalid_interval_count(one_good_interval):
    X, y = one_good_interval
    with pytest.raises(ValueError, match="n_intervals"):
        ipls(X, y, n_intervals=1)


def test_plot_ipls_runs(one_good_interval):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    from specpls.plotting import plot_ipls

    X, y = one_good_interval
    ax = plot_ipls(ipls(X, y, n_intervals=10, max_intervals=2, cv=5), X)
    assert ax.get_ylabel().startswith("RMSECV")


def test_ipls_ratio_guard(one_good_interval):
    X, y = one_good_interval
    result = ipls(X, y, n_intervals=10, max_intervals=3, cv=5, max_cv_ratio=1.2)
    assert result.cv_ratio <= 1.2 + 1e-9
    assert result.mask[40:50].all()
    assert "ratio" in result.summary()


def test_ipls_ratio_guard_impossible(one_good_interval):
    X, y = one_good_interval
    with pytest.raises(ValueError, match="No single interval"):
        ipls(X, y, n_intervals=10, max_intervals=2, cv=5, max_cv_ratio=0.5)


# ----------------------------------------------------------------- score / loading plots
@pytest.mark.parametrize("n_lv", [1, 3])
def test_score_and_loading_plots(culture, n_lv):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    from specpls.plotting import plot_loadings, plot_scores, plot_t2_q

    X, y = culture.X, culture.target("glucose")
    model = PLSModel(preprocessing="snv", n_components=n_lv, cv=5).fit(
        X[:50], y[:50], wavenumbers=culture.wavenumbers
    )
    ax = plot_scores(model, X[:50], y[:50], X[50:], y[50:])
    assert "LV1" in (ax.get_ylabel() if n_lv == 1 else ax.get_xlabel())
    ax = plot_loadings(model)
    assert len(ax.lines) == n_lv + 1  # one line per LV plus the zero line
    plot_t2_q(model, X[:50], X_test=X[50:], labels=list(range(50)))
