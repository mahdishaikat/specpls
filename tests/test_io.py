import numpy as np
import pandas as pd
import pytest
from scipy.io import savemat

from specpls.io import SpectralData, load_csv, load_mat, split_columns


def test_csv_round_trip(culture, tmp_path):
    path = tmp_path / "data.csv"
    culture.to_frame().to_csv(path, index=False)
    loaded = load_csv(path)
    np.testing.assert_allclose(loaded.X, culture.X)
    np.testing.assert_allclose(loaded.wavenumbers, culture.wavenumbers)
    assert list(loaded.meta.columns) == list(culture.meta.columns)


def test_split_columns_sorts_wavenumbers():
    df = pd.DataFrame({"id": ["a", "b"], "402": [3, 6], "400": [1, 4], "401": [2, 5]})
    data = split_columns(df)
    np.testing.assert_array_equal(data.wavenumbers, [400, 401, 402])
    np.testing.assert_array_equal(data.X, [[1, 2, 3], [4, 5, 6]])
    assert list(data.meta.columns) == ["id"]


def test_split_columns_without_spectra_raises():
    with pytest.raises(ValueError, match="No spectral columns"):
        split_columns(pd.DataFrame({"a": [1], "b": [2]}))


def test_load_mat(tmp_path):
    X = np.arange(12, dtype=float).reshape(3, 4)
    path = tmp_path / "spectra.mat"
    savemat(path, {"X": X, "wavenumbers": np.array([[1, 2, 3, 4]]), "glucose": [[1.0, 2.0, 3.0]]})
    data = load_mat(path, meta_keys=["glucose"])
    np.testing.assert_array_equal(data.X, X)
    np.testing.assert_array_equal(data.target("glucose"), [1, 2, 3])


def test_load_mat_missing_variable(tmp_path):
    path = tmp_path / "spectra.mat"
    savemat(path, {"X": np.ones((2, 2)), "wavenumbers": [1, 2]})
    with pytest.raises(KeyError, match="glucose"):
        load_mat(path, meta_keys=["glucose"])


def test_spectral_data_validation():
    with pytest.raises(ValueError, match="columns"):
        SpectralData(np.ones((2, 3)), np.arange(4))
    with pytest.raises(ValueError, match="one row per spectrum"):
        SpectralData(np.ones((2, 3)), np.arange(3), pd.DataFrame({"a": [1, 2, 3]}))


def test_target_and_subset(culture):
    assert culture.target("glucose").dtype == float
    with pytest.raises(KeyError, match="Available"):
        culture.target("titer")
    first_batch = culture.subset((culture.meta["batch"] == "B1").to_numpy())
    assert len(first_batch) == 10
    assert set(first_batch.meta["batch"]) == {"B1"}


def test_dropna():
    meta = pd.DataFrame({"glucose": [1.0, np.nan, 3.0]})
    data = SpectralData(np.ones((3, 2)), [1, 2], meta).dropna("glucose")
    assert len(data) == 2
