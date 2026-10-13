import numpy as np
import pytest
from sklearn.model_selection import KFold, LeaveOneGroupOut, LeaveOneOut

from specpls.validation import (
    VenetianBlinds,
    bias,
    make_cv,
    r2,
    regression_metrics,
    rmse,
    rpd,
    select_n_components,
    sep,
)


# ----------------------------------------------------------------- metrics
def test_rmse_known_value():
    assert rmse([1, 2, 3], [1, 2, 5]) == pytest.approx(np.sqrt(4 / 3))


def test_perfect_prediction():
    y = [1.0, 2.0, 3.0, 4.0]
    assert rmse(y, y) == 0
    assert r2(y, y) == 1
    assert bias(y, y) == 0


def test_bias_sign():
    assert bias([1, 2, 3], [2, 3, 4]) == pytest.approx(1.0)


def test_rmse_decomposes_into_sep_and_bias(rng):
    y = rng.normal(size=200)
    y_pred = y + 0.3 + rng.normal(scale=0.2, size=200)
    n = len(y)
    expected = sep(y, y_pred) ** 2 * (n - 1) / n + bias(y, y_pred) ** 2
    assert rmse(y, y_pred) ** 2 == pytest.approx(expected)


def test_r2_constant_reference_is_nan():
    assert np.isnan(r2([2, 2, 2], [1, 2, 3]))


def test_rpd_definition(rng):
    y = rng.normal(size=50)
    y_pred = y + rng.normal(scale=0.1, size=50)
    assert rpd(y, y_pred) == pytest.approx(np.std(y, ddof=1) / sep(y, y_pred))


def test_metrics_shape_mismatch():
    with pytest.raises(ValueError, match="Shape mismatch"):
        rmse([1, 2], [1, 2, 3])


def test_regression_metrics_keys():
    out = regression_metrics([1, 2, 3], [1.1, 1.9, 3.2])
    assert set(out) == {"rmse", "r2", "bias", "sep", "rpd", "n"}
    assert out["n"] == 3


# ----------------------------------------------------------------- CV splitters
def test_venetian_blinds_assignment():
    folds = [test.tolist() for _, test in VenetianBlinds(3).split(np.zeros((7, 2)))]
    assert folds == [[0, 3, 6], [1, 4], [2, 5]]


def test_venetian_blinds_partition():
    X = np.zeros((23, 1))
    tested = np.concatenate([test for _, test in VenetianBlinds(5).split(X)])
    assert sorted(tested) == list(range(23))


def test_venetian_blinds_too_many_folds():
    with pytest.raises(ValueError, match="n_splits"):
        list(VenetianBlinds(10).split(np.zeros((5, 1))))


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        (5, VenetianBlinds),
        ("venetian:4", VenetianBlinds),
        ("kfold:5", KFold),
        ("loo", LeaveOneOut),
    ],
)
def test_make_cv(spec, expected):
    assert isinstance(make_cv(spec), expected)


def test_make_cv_batch_requires_groups():
    with pytest.raises(ValueError, match="groups"):
        make_cv("batch")
    assert isinstance(make_cv("batch", groups=[1, 1, 2]), LeaveOneGroupOut)


def test_make_cv_passes_splitter_through():
    splitter = KFold(3)
    assert make_cv(splitter) is splitter


@pytest.mark.parametrize("spec", ["nonsense", 2.5, True])
def test_make_cv_rejects_unknown(spec):
    with pytest.raises((ValueError, TypeError)):
        make_cv(spec)


# ----------------------------------------------------------------- LV selection
CURVE = [1.0, 0.5, 0.30, 0.295, 0.31, 0.29, 0.35]


def test_select_min():
    assert select_n_components(CURVE, "min").n_components == 6


def test_select_first_min():
    assert select_n_components(CURVE, "first_min").n_components == 4


def test_select_parsimonious():
    # 0.30 is within 5 % of the minimum 0.29, so 3 LVs are enough.
    assert select_n_components(CURVE, "parsimonious", 0.05).n_components == 3
    # With zero tolerance it falls back to the global minimum.
    assert select_n_components(CURVE, "parsimonious", 0.0).n_components == 6


def test_select_monotonic_curve_first_min_is_last():
    assert select_n_components([3, 2, 1], "first_min").n_components == 3


@pytest.mark.parametrize("bad", [[], [1.0, np.nan]])
def test_select_rejects_bad_curve(bad):
    with pytest.raises(ValueError):
        select_n_components(bad)


def test_select_unknown_rule():
    with pytest.raises(ValueError, match="Unknown rule"):
        select_n_components(CURVE, "magic")
