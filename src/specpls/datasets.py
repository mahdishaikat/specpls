"""Synthetic Raman-like cell-culture data for examples, tests and teaching.

The data are *simulated*. They mimic the structure of CHO fed-batch Raman
data (glucose and lactate bands, cell-related bands, a broad fluorescence
background that grows with cell density, scatter and noise) so the package
can be demonstrated without sharing proprietary spectra.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .io import SpectralData

__all__ = ["make_synthetic_cell_culture"]

# (centre cm-1, width cm-1, relative intensity)
_GLUCOSE_BANDS = [
    (422, 10, 0.6),
    (518, 12, 0.7),
    (911, 10, 0.6),
    (1065, 12, 0.8),
    (1125, 10, 1.0),
    (1365, 12, 0.5),
    (1460, 14, 0.5),
]
_LACTATE_BANDS = [
    (545, 10, 0.5),
    (857, 9, 1.0),
    (930, 10, 0.4),
    (1043, 10, 0.6),
    (1418, 12, 0.5),
    (1455, 14, 0.5),
]
_CELL_BANDS = [
    (1003, 6, 0.8),  # phenylalanine ring breathing
    (1250, 20, 0.4),  # amide III
    (1450, 18, 0.9),  # CH2 deformation
    (1660, 22, 1.0),  # amide I
]
_MEDIUM_BANDS = [
    (980, 15, 0.6),  # sulfate / phosphate-like medium band
    (1640, 40, 1.2),  # water bending
]


def _bands(axis, bands):
    out = np.zeros_like(axis)
    for centre, width, height in bands:
        out += height * np.exp(-0.5 * ((axis - centre) / width) ** 2)
    return out


def make_synthetic_cell_culture(
    n_batches=6,
    n_days=10,
    wavenumber_range=(400, 1800),
    step=2.0,
    fluorescence=8.0,
    noise=0.01,
    random_state=None,
) -> SpectralData:
    """Simulate daily Raman spectra from fed-batch CHO cultures.

    Parameters
    ----------
    n_batches : int, default=6
        Number of independent bioreactor runs.
    n_days : int, default=10
        Samples per run (one per day).
    wavenumber_range : (float, float), default=(400, 1800)
        Raman shift range in cm-1.
    step : float, default=2.0
        Spacing of the wavenumber axis.
    fluorescence : float, default=8.0
        Size of the fluorescence background relative to the strongest glucose
        band at 1 g/L. Set to 0 to simulate ideal fluorescence-free spectra.
    noise : float, default=0.01
        Standard deviation of additive noise.
    random_state : int or None
        Seed for reproducibility.

    Returns
    -------
    SpectralData
        ``meta`` holds ``sample_id``, ``batch``, ``day``, ``vcd``
        (1e6 cells/mL), ``glucose`` and ``lactate`` (g/L).
    """
    rng = np.random.default_rng(random_state)
    axis = np.arange(wavenumber_range[0], wavenumber_range[1] + step / 2, step)
    glucose_unit = _bands(axis, _GLUCOSE_BANDS)
    lactate_unit = _bands(axis, _LACTATE_BANDS)
    cell_unit = _bands(axis, _CELL_BANDS)
    medium = _bands(axis, _MEDIUM_BANDS)
    x01 = (axis - axis.min()) / (axis.max() - axis.min())

    spectra, rows = [], []
    for b in range(n_batches):
        growth = rng.uniform(0.45, 0.6)
        vcd_max = rng.uniform(5.0, 7.0)
        glucose_start = rng.uniform(5.0, 6.0)
        lactate_peak = rng.uniform(2.0, 4.5)
        for day in range(n_days):
            vcd = vcd_max / (1 + np.exp(-growth * (day - n_days / 2.5)))
            glucose = glucose_start - 0.45 * day + rng.normal(0, 0.6)
            glucose = float(np.clip(glucose, 0.8, 6.2))
            lactate = lactate_peak * np.exp(-((day - 4.5) ** 2) / 10) + rng.normal(0, 0.3)
            lactate = float(np.clip(lactate, 0.05, 4.8))

            raman = glucose * glucose_unit + lactate * lactate_unit * 0.9
            raman += 0.25 * vcd * cell_unit + medium

            amp = fluorescence * (0.4 + 0.12 * vcd + rng.uniform(0, 0.3))
            shape = 0.6 + 0.8 * np.exp(-((x01 - rng.uniform(0.4, 0.7)) ** 2) / 0.25)
            background = amp * shape * (1 + rng.uniform(-0.15, 0.15) * x01)

            scatter = rng.uniform(0.9, 1.1)
            spectrum = scatter * (raman + background) + rng.normal(0, noise, axis.size)
            spectra.append(spectrum)
            rows.append(
                {
                    "sample_id": f"B{b + 1}-D{day:02d}",
                    "batch": f"B{b + 1}",
                    "day": day,
                    "vcd": round(float(vcd), 3),
                    "glucose": round(glucose, 3),
                    "lactate": round(lactate, 3),
                }
            )
    return SpectralData(np.vstack(spectra), axis, pd.DataFrame(rows))
