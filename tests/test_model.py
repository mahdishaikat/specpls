import warnings

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import cross_val_predict

from specpls import SNV, PLSModel, SavitzkyGolay, load_model
from specpls.validation import VenetianBlinds


def simpls(X, y, n_components):
    """Reference single-response SIMPLS (de Jong, 1993).

    SIMPLS is the algorithm behind MATLAB's ``plsregress`` and PLS_Toolbox.
    For one response it gives the same model as NIPALS (used by
    scikit-learn), which this test file checks.
    """
    x_mean, y_mean = X.mean(axis=0), y.mean()
    Xc, yc = X - x_mean, y - y_mean
    s = Xc.T @ yc
    R, V = [], []
    for _ in range(n_components):
        r = s.copy()
        t = Xc @ r
        norm = np.linalg.norm(t)
        t, r = t / norm, r / norm
        p = Xc.T @ t
        v = p.copy()
        for v_prev in V:
            v -= v_prev * (v_prev @ p)
        v /= np.linalg.norm(v)
        s -= v * (v @ s)
        R.append(r)
        V.append(v)
    R = np.column_stack(R)
    T = Xc @ R
    q = T.T @ yc
    beta = R @ q
    return beta, y_mean - x_mean @ beta


# ----------------------------------------------------------------- correctness
def test_matches_reference_simpls(culture):
    X, y = culture.X, culture.target("glucose")
    model = PLSModel(n_components=4, cv=5).fit(X, y)
    beta, intercept = simpls(X, y, 4)
    np.testing.assert_allclose(model.predict(X), X @ beta + intercept, rtol=1e-6, atol=1e-8)


def test_matches_plain_sklearn(culture):
    X, y = culture.X, culture.target("lactate")
    model = PLSModel(n_components=3, cv=VenetianBlinds(5)).fit(X, y)
    reference = PLSRegression(n_components=3, scale=False).fit(X, y)
    np.testing.assert_allclose(model.predict(X), reference.predict(X).ravel())
    cv_reference = cross_val_predict(
        PLSRegression(n_components=3, scale=False), X, y, cv=VenetianBlinds(5)
    ).ravel()
    np.testing.assert_allclose(model.cv_predictions_, cv_reference)


def test_auto_lv_selection_and_accuracy(cal_test):
    cal, test = cal_test
    model = PLSModel(preprocessing="snv", max_components=10, cv="batch").fit(
        cal.X, cal.target("glucose"), groups=cal.meta["batch"], target_name="glucose"
    )
    assert 1 <= model.n_components_ <= 10
    assert model.rmsecv_curve_.shape == (10,)
    assert model.calibration_["rmsecv"] == pytest.approx(
        model.rmsecv_curve_[model.n_components_ - 1]
    )
    assert model.calibration_["rmsec"] <= model.calibration_["rmsecv"]
    result = model.evaluate(test.X, test.target("glucose"))
    assert result["r2"] > 0.9
    assert result["rmsep"] < 0.4  # g/L on the simulated data


def test_preprocessing_is_refitted_inside_cv_folds(culture):
    """MSC's reference must come from training folds only (no data leakage)."""
    X, y = culture.X, culture.target("glucose")
    model = PLSModel(preprocessing="msc", n_components=3, cv=5).fit(X, y)
    leaky_reference = model.pipeline_.named_steps["preprocessing"][0].reference_
    folds = model._preprocess_folds(VenetianBlinds(5), X, y, None, None)
    fold_refs = [X[train].mean(axis=0) for train, *_ in folds]
    assert all(not np.allclose(ref, leaky_reference) for ref in fold_refs)


def test_preprocessing_accepts_list_and_does_not_mutate_it(culture):
    steps = [SavitzkyGolay(15, 2, 1), SNV()]
    model = PLSModel(preprocessing=steps, n_components=3, cv=5)
    model.fit(culture.X, culture.target("glucose"))
    assert not hasattr(steps[1], "n_features_in_")


def test_regions_reduce_variables(culture):
    model = PLSModel(regions=[(800, 1200)], n_components=3, cv=5).fit(
        culture.X, culture.target("glucose"), wavenumbers=culture.wavenumbers
    )
    selected = model.model_wavenumbers_
    assert selected.min() >= 800 and selected.max() <= 1200
    assert model.vip().shape == selected.shape


def test_regions_without_wavenumbers_raises(culture):
    with pytest.raises(ValueError, match="wavenumbers"):
        PLSModel(regions=[(800, 1200)]).fit(culture.X, culture.target("glucose"))


def test_is_sklearn_compatible(culture):
    model = PLSModel(preprocessing="snv", n_components=2, cv=3)
    params = model.get_params()
    assert params["preprocessing"] == "snv"
    clone(model).fit(culture.X, culture.target("glucose"))


# ----------------------------------------------------------------- input checks
def test_predict_before_fit():
    with pytest.raises(NotFittedError):
        PLSModel().predict(np.zeros((2, 5)))


def test_two_responses_rejected(culture):
    y = culture.meta[["glucose", "lactate"]].to_numpy()
    with pytest.raises(ValueError, match="one response"):
        PLSModel().fit(culture.X, y)


def test_column_vector_y_accepted(culture):
    y = culture.target("glucose")[:, None]
    PLSModel(n_components=2, cv=3).fit(culture.X, y)


def test_nan_in_y_rejected(culture):
    y = culture.target("glucose").copy()
    y[3] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        PLSModel().fit(culture.X, y)


def test_length_mismatch(culture):
    with pytest.raises(ValueError, match="samples"):
        PLSModel().fit(culture.X, culture.target("glucose")[:-1])


def test_too_many_components(culture):
    with pytest.raises(ValueError, match="n_components"):
        PLSModel(n_components=500, cv=3).fit(culture.X[:20], culture.target("glucose")[:20])


def test_max_components_capped_with_warning(culture):
    X, y = culture.X[:12], culture.target("glucose")[:12]
    with pytest.warns(UserWarning, match="max_components reduced"):
        model = PLSModel(max_components=20, cv=3).fit(X, y)
    assert len(model.rmsecv_curve_) == 7  # 8 training samples per fold - 1


def test_predict_checks_width(culture):
    model = PLSModel(n_components=2, cv=3).fit(culture.X, culture.target("glucose"))
    with pytest.raises(ValueError, match="expects"):
        model.predict(culture.X[:, :-1])


def test_predict_checks_wavenumber_axis(culture):
    model = PLSModel(n_components=2, cv=3).fit(
        culture.X, culture.target("glucose"), wavenumbers=culture.wavenumbers
    )
    model.predict(culture.X, wavenumbers=culture.wavenumbers)  # same axis: fine
    with pytest.raises(ValueError, match="does not match"):
        model.predict(culture.X, wavenumbers=culture.wavenumbers + 5)


# ----------------------------------------------------------------- diagnostics
@pytest.fixture(scope="module")
def fitted(culture):
    return PLSModel(preprocessing="snv", n_components=4, cv=5).fit(
        culture.X, culture.target("glucose"), wavenumbers=culture.wavenumbers
    )


def test_vip_mean_square_is_one(fitted):
    vip = fitted.vip()
    assert vip.shape == (fitted.n_features_in_,)
    assert np.mean(vip**2) == pytest.approx(1.0)


def test_vip_highlights_glucose_band(fitted, culture):
    """The simulated glucose band at 1125 cm-1 should be important."""
    vip = fitted.vip()
    idx = np.argmin(np.abs(culture.wavenumbers - 1125))
    assert vip[idx] > 1


def test_t2_and_q_shapes_and_sign(fitted, culture):
    t2 = fitted.hotelling_t2(culture.X)
    q = fitted.q_residuals(culture.X)
    assert t2.shape == q.shape == (len(culture),)
    assert np.all(t2 >= 0) and np.all(q >= 0)


def test_mean_calibration_t2_equals_lvs(fitted, culture):
    # With sample-variance scaling, mean T2 over the calibration set is A (n-1)/n.
    n = len(culture)
    assert fitted.hotelling_t2(culture.X).mean() == pytest.approx(4 * (n - 1) / n)


def test_limits_increase_with_confidence(fitted):
    assert fitted.t2_limit(0.99) > fitted.t2_limit(0.95)
    assert fitted.q_limit(0.99) > fitted.q_limit(0.95)


def test_abnormal_spectrum_is_flagged(fitted, culture):
    odd = culture.X[:1].copy()
    odd[0, 300:320] += 50 * culture.X.std()  # spike not seen in calibration
    flags = fitted.outliers(odd)
    assert flags["q_flag"][0]


def test_scores_shape(fitted, culture):
    assert fitted.scores(culture.X).shape == (len(culture), 4)


def test_summary_table(fitted, culture):
    row = fitted.summary(culture.X, culture.target("glucose"))
    for key in ["lvs", "rmsec", "r2_cal", "rmsecv", "r2_cv", "rmsep", "r2_test"]:
        assert key in row
    assert row["preprocessing"] == "snv"


# ----------------------------------------------------------------- persistence
def test_save_and_load_round_trip(fitted, culture, tmp_path):
    path = fitted.save(tmp_path / "models" / "glucose.joblib")
    assert path.exists()
    loaded = load_model(path)
    np.testing.assert_array_equal(loaded.predict(culture.X), fitted.predict(culture.X))
    np.testing.assert_array_equal(loaded.wavenumbers_, culture.wavenumbers)


def test_load_rejects_foreign_file(tmp_path):
    import joblib

    joblib.dump({"something": 1}, tmp_path / "x.joblib")
    with pytest.raises(ValueError, match="not a specpls model"):
        load_model(tmp_path / "x.joblib")


def test_load_warns_on_sklearn_version_change(fitted, tmp_path):
    import joblib

    path = fitted.save(tmp_path / "m.joblib")
    payload = joblib.load(path)
    payload["sklearn_version"] = "0.0.1"
    joblib.dump(payload, path)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        load_model(path)
    assert any("scikit-learn" in str(w.message) for w in caught)


# ----------------------------------------------------------------- RMSECV/RMSEC guard
def test_max_cv_ratio_limits_lvs(cal_test):
    cal, _ = cal_test
    y = cal.target("glucose")
    free = PLSModel(max_components=10, cv=5, lv_rule="min").fit(cal.X, y)
    guarded = PLSModel(max_components=10, cv=5, lv_rule="min", max_cv_ratio=1.2).fit(cal.X, y)
    assert guarded.rmsec_curve_.shape == guarded.rmsecv_curve_.shape
    if guarded.ratio_ok_:
        assert guarded.calibration_["cv_ratio"] <= 1.2 + 1e-9
    assert guarded.n_components_ <= free.n_components_


def test_cv_ratio_reported(fitted):
    ratio = fitted.calibration_["cv_ratio"]
    assert ratio == pytest.approx(fitted.calibration_["rmsecv"] / fitted.calibration_["rmsec"])
    assert "cv_ratio" in fitted.summary()


def test_max_cv_ratio_warns_when_impossible(culture):
    with pytest.warns(UserWarning, match="RMSECV/RMSEC"):
        PLSModel(max_components=5, cv=5, max_cv_ratio=0.5).fit(culture.X, culture.target("glucose"))
