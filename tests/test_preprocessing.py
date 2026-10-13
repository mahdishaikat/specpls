import numpy as np
import pytest
from scipy.signal import savgol_filter
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from specpls.preprocessing import (
    MSC,
    SNV,
    AsLSBaseline,
    Detrend,
    RegionSelector,
    SavitzkyGolay,
    build_preprocessing,
)

ALL_TRANSFORMERS = [
    SNV(),
    MSC(),
    SavitzkyGolay(window_length=11, polyorder=2, deriv=1),
    Detrend(order=2),
    AsLSBaseline(lam=1e4, p=0.01),
]


@pytest.fixture
def spectra(rng):
    axis = np.linspace(0, 1, 200)
    peaks = np.exp(-((axis - 0.4) ** 2) / 0.001) + 0.5 * np.exp(-((axis - 0.7) ** 2) / 0.002)
    scale = rng.uniform(0.8, 1.2, (15, 1))
    offset = rng.uniform(-1, 1, (15, 1))
    return scale * peaks + offset + rng.normal(0, 0.005, (15, 200))


# ----------------------------------------------------------------- generic API
@pytest.mark.parametrize("transformer", ALL_TRANSFORMERS, ids=lambda t: type(t).__name__)
def test_shape_and_finite(transformer, spectra):
    out = clone(transformer).fit_transform(spectra)
    assert out.shape == spectra.shape
    assert np.all(np.isfinite(out))


@pytest.mark.parametrize("transformer", ALL_TRANSFORMERS, ids=lambda t: type(t).__name__)
def test_wrong_width_raises(transformer, spectra):
    fitted = clone(transformer).fit(spectra)
    with pytest.raises(ValueError, match="wavenumbers"):
        fitted.transform(spectra[:, :-1])


@pytest.mark.parametrize("transformer", ALL_TRANSFORMERS, ids=lambda t: type(t).__name__)
def test_transform_before_fit_raises(transformer, spectra):
    from sklearn.exceptions import NotFittedError

    with pytest.raises(NotFittedError):
        clone(transformer).transform(spectra)


@pytest.mark.parametrize("transformer", ALL_TRANSFORMERS, ids=lambda t: type(t).__name__)
def test_does_not_modify_input(transformer, spectra):
    original = spectra.copy()
    clone(transformer).fit_transform(spectra)
    np.testing.assert_array_equal(spectra, original)


# ----------------------------------------------------------------- SNV
def test_snv_rows_have_zero_mean_unit_std(spectra):
    out = SNV().fit_transform(spectra)
    np.testing.assert_allclose(out.mean(axis=1), 0, atol=1e-12)
    np.testing.assert_allclose(out.std(axis=1, ddof=1), 1, atol=1e-12)


def test_snv_removes_multiplicative_and_additive_effects():
    base = np.sin(np.linspace(0, 6, 100))
    X = np.vstack([base, 3 * base + 5])
    out = SNV().fit_transform(X)
    np.testing.assert_allclose(out[0], out[1], atol=1e-12)


def test_snv_constant_spectrum_does_not_divide_by_zero():
    out = SNV().fit_transform(np.ones((2, 10)))
    np.testing.assert_array_equal(out, 0)


# ----------------------------------------------------------------- MSC
def test_msc_corrects_scatter_back_to_reference():
    ref = np.exp(-((np.linspace(-3, 3, 80)) ** 2))
    X = np.vstack([ref, 2.0 * ref + 0.3, 0.5 * ref - 0.1])
    out = MSC(reference=ref).fit_transform(X)
    np.testing.assert_allclose(out, np.tile(ref, (3, 1)), atol=1e-10)


def test_msc_uses_training_mean_as_reference(spectra):
    msc = MSC().fit(spectra)
    np.testing.assert_allclose(msc.reference_, spectra.mean(axis=0))


def test_msc_reference_length_checked(spectra):
    with pytest.raises(ValueError, match="reference length"):
        MSC(reference=np.ones(3)).fit(spectra)


# ----------------------------------------------------------------- Savitzky-Golay
@pytest.mark.parametrize("deriv", [0, 1, 2])
def test_savgol_matches_scipy(spectra, deriv):
    out = SavitzkyGolay(window_length=15, polyorder=3, deriv=deriv).fit_transform(spectra)
    expected = savgol_filter(spectra, 15, 3, deriv=deriv, axis=1)
    np.testing.assert_allclose(out, expected)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"window_length": 10}, "odd"),
        ({"window_length": 5, "polyorder": 5}, "polyorder"),
        ({"polyorder": 1, "deriv": 2}, "deriv"),
        ({"window_length": 301}, "exceed"),
    ],
)
def test_savgol_invalid_parameters(spectra, kwargs, message):
    with pytest.raises(ValueError, match=message):
        SavitzkyGolay(**kwargs).fit(spectra)


# ----------------------------------------------------------------- baselines
def test_detrend_removes_linear_baseline():
    axis = np.linspace(-1, 1, 50)
    X = np.vstack([2 + 3 * axis, -1 + 0.5 * axis])
    np.testing.assert_allclose(Detrend(order=1).fit_transform(X), 0, atol=1e-10)


def test_asls_removes_broad_background_but_keeps_peak():
    axis = np.linspace(0, 1, 300)
    peak = np.exp(-((axis - 0.5) ** 2) / 0.0005)
    background = 5 + 4 * axis + 3 * axis**2
    corrected = AsLSBaseline(lam=1e3, p=0.01).fit_transform((peak + background)[None, :])[0]
    # Background largely removed away from the peak, peak height preserved.
    assert np.abs(corrected[:100]).max() < 0.05
    assert corrected.max() == pytest.approx(1.0, abs=0.1)


def test_asls_invalid_p(spectra):
    with pytest.raises(ValueError, match="p must be"):
        AsLSBaseline(p=1.5).fit(spectra)


# ----------------------------------------------------------------- RegionSelector
def test_region_selector_keeps_requested_ranges():
    axis = np.arange(400, 1801, 1.0)
    X = np.tile(axis, (2, 1))
    sel = RegionSelector(axis, [(800, 900), (1400, 1500)]).fit(X)
    out = sel.transform(X)
    assert out.shape == (2, 101 + 101)
    assert out.min() == 800 and out.max() == 1500
    np.testing.assert_array_equal(sel.selected_wavenumbers_, out[0])


def test_region_selector_accepts_single_tuple():
    axis = np.arange(10.0)
    sel = RegionSelector(axis, (2, 4)).fit(np.zeros((1, 10)))
    np.testing.assert_array_equal(sel.selected_wavenumbers_, [2, 3, 4])


def test_region_selector_empty_region_raises():
    axis = np.arange(10.0)
    with pytest.raises(ValueError, match="No wavenumbers"):
        RegionSelector(axis, [(100, 200)]).fit(np.zeros((1, 10)))


# ----------------------------------------------------------------- build_preprocessing
def test_build_from_string():
    pipe = build_preprocessing("savgol:15:2:1, snv")
    assert isinstance(pipe, Pipeline)
    sg, snv = (step for _, step in pipe.steps)
    assert isinstance(sg, SavitzkyGolay) and isinstance(snv, SNV)
    assert (sg.window_length, sg.polyorder, sg.deriv) == (15, 2, 1)


def test_build_parses_scientific_notation_as_float():
    pipe = build_preprocessing("asls:1e5:0.01")
    step = pipe.steps[0][1]
    assert step.lam == 1e5 and step.p == 0.01


@pytest.mark.parametrize("spec", [None, "", "   ", []])
def test_build_empty_returns_none(spec):
    assert build_preprocessing(spec) is None


def test_build_unknown_step_lists_known_names():
    with pytest.raises(ValueError, match="Known: .*snv"):
        build_preprocessing("wavelet")


def test_build_too_many_arguments():
    with pytest.raises(ValueError, match="at most"):
        build_preprocessing("snv:3")
