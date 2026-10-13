"""Loading spectra and reference values from CSV, Excel-exported CSV and MATLAB files."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["SpectralData", "load_csv", "load_mat", "split_columns"]


@dataclass
class SpectralData:
    """Spectra plus sample metadata.

    Attributes
    ----------
    X : ndarray of shape (n_samples, n_wavenumbers)
        Spectra, one row per sample.
    wavenumbers : ndarray of shape (n_wavenumbers,)
        Spectral axis (e.g. Raman shift in cm-1).
    meta : pandas.DataFrame
        One row per sample: reference values (glucose, lactate, ...),
        batch IDs, culture day, etc.
    """

    X: np.ndarray
    wavenumbers: np.ndarray
    meta: pd.DataFrame = field(default_factory=pd.DataFrame)

    def __post_init__(self):
        self.X = np.asarray(self.X, dtype=np.float64)
        self.wavenumbers = np.asarray(self.wavenumbers, dtype=np.float64).ravel()
        if self.X.ndim != 2:
            raise ValueError("X must be 2-D (n_samples, n_wavenumbers).")
        if self.X.shape[1] != self.wavenumbers.shape[0]:
            raise ValueError(
                f"X has {self.X.shape[1]} columns but there are "
                f"{self.wavenumbers.shape[0]} wavenumbers."
            )
        if self.meta.empty:
            self.meta = pd.DataFrame(index=range(self.X.shape[0]))
        if len(self.meta) != self.X.shape[0]:
            raise ValueError("meta must have one row per spectrum.")
        self.meta = self.meta.reset_index(drop=True)

    def __len__(self) -> int:
        return self.X.shape[0]

    def target(self, name: str) -> np.ndarray:
        """Reference values for one analyte as a float array."""
        if name not in self.meta.columns:
            raise KeyError(f"{name!r} not found. Available: {list(self.meta.columns)}")
        return self.meta[name].to_numpy(dtype=np.float64)

    def subset(self, mask) -> SpectralData:
        """Rows selected by a boolean mask or integer index."""
        idx = np.asarray(mask)
        return SpectralData(self.X[idx], self.wavenumbers, self.meta.iloc[idx].copy())

    def dropna(self, column: str) -> SpectralData:
        """Drop samples whose ``column`` is missing (e.g. no reference value)."""
        return self.subset(self.meta[column].notna().to_numpy())

    def to_frame(self) -> pd.DataFrame:
        """Wide table: metadata columns followed by one column per wavenumber."""
        spectra = pd.DataFrame(self.X, columns=[_fmt(w) for w in self.wavenumbers])
        return pd.concat([self.meta.reset_index(drop=True), spectra], axis=1)


def _fmt(value: float) -> str:
    return f"{value:g}"


def _is_number(name) -> bool:
    try:
        float(str(name).strip())
    except ValueError:
        return False
    return True


def split_columns(df: pd.DataFrame) -> SpectralData:
    """Split a wide table into spectra and metadata.

    Columns whose header parses as a number (``"400"``, ``"401.5"``) are
    treated as spectral channels; all other columns become metadata.
    """
    spectral = [c for c in df.columns if _is_number(c)]
    if not spectral:
        raise ValueError(
            "No spectral columns found. Spectral column headers must be the "
            "wavenumber values, e.g. 400, 401, 402 ..."
        )
    meta_cols = [c for c in df.columns if c not in spectral]
    wavenumbers = np.array([float(str(c).strip()) for c in spectral])
    order = np.argsort(wavenumbers)
    X = df[spectral].to_numpy(dtype=np.float64)[:, order]
    return SpectralData(X, wavenumbers[order], df[meta_cols].copy())


def load_csv(path, **read_csv_kwargs) -> SpectralData:
    """Load a wide CSV file: metadata columns plus one column per wavenumber.

    Example layout::

        sample_id, batch, day, glucose, lactate, 400, 401, 402, ...
        S01,       B1,    0,   5.78,    0.07,    812, 815, 820, ...

    Extra keyword arguments are passed to :func:`pandas.read_csv`.
    """
    return split_columns(pd.read_csv(Path(path), **read_csv_kwargs))


def load_mat(path, x_key="X", axis_key="wavenumbers", meta_keys=None) -> SpectralData:
    """Load spectra saved from MATLAB with ``save('file.mat', 'X', 'wavenumbers', ...)``.

    Parameters
    ----------
    path : str or Path
        ``.mat`` file (v7 or earlier; for v7.3 save with ``-v7``).
    x_key : str, default="X"
        Variable holding the spectra matrix (samples x wavenumbers).
    axis_key : str, default="wavenumbers"
        Variable holding the spectral axis.
    meta_keys : list of str, optional
        Vector variables (one value per sample) to put in ``meta``,
        e.g. ``["glucose", "lactate"]``.
    """
    from scipy.io import loadmat

    mat = loadmat(Path(path))
    for key in [x_key, axis_key, *(meta_keys or [])]:
        if key not in mat:
            found = [k for k in mat if not k.startswith("__")]
            raise KeyError(f"Variable {key!r} not in {path}. Found: {found}")
    X = np.asarray(mat[x_key], dtype=np.float64)
    axis = np.asarray(mat[axis_key], dtype=np.float64).ravel()
    meta = pd.DataFrame({k: np.asarray(mat[k]).ravel() for k in (meta_keys or [])})
    return SpectralData(X, axis, meta)
