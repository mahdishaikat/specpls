"""SNV + smoothing + EPO with an over-fitting guard (RMSECV / RMSEC <= 1.2).

The same workflow used for time-gated Raman glucose models, shown on
simulated data so it can be shared and run by anyone:

    SNV -> Savitzky-Golay smoothing -> EPO -> mean-centring -> PLS

The number of latent variables is chosen automatically, but only among LVs
whose RMSECV is at most 20 % above RMSEC.

Run with:  python examples/04_epo_overfitting_guard.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

from specpls import EPO, SNV, PLSModel, SavitzkyGolay
from specpls.datasets import make_synthetic_cell_culture
from specpls.plotting import plot_overview

MAX_CV_RATIO = 1.2
OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

data = make_synthetic_cell_culture(random_state=7)
is_test = (data.meta["batch"] == "B6").to_numpy()
cal, test = data.subset(~is_test), data.subset(is_test)

model = PLSModel(
    preprocessing=[SNV(), SavitzkyGolay(15, 2, 0), EPO(1)],
    max_components=8,
    cv=10,
    max_cv_ratio=MAX_CV_RATIO,
)
model.fit(cal.X, cal.target("glucose"), wavenumbers=cal.wavenumbers, target_name="glucose")

lv_table = pd.DataFrame(
    {
        "LV": np.arange(1, len(model.rmsecv_curve_) + 1),
        "RMSEC": model.rmsec_curve_,
        "RMSECV": model.rmsecv_curve_,
    }
)
lv_table["ratio"] = lv_table["RMSECV"] / lv_table["RMSEC"]
lv_table["passes"] = lv_table["ratio"] <= MAX_CV_RATIO
print(lv_table.round(3).to_string(index=False))
print(f"Selected {model.n_components_} LV")

for key, value in model.summary(test.X, test.target("glucose")).items():
    print(f"{key:>14}: {value:.3f}" if isinstance(value, float) else f"{key:>14}: {value}")

fig = plot_overview(model, cal.X, cal.target("glucose"), test.X, test.target("glucose"), "g/L")
fig.savefig(OUT / "epo_guard_overview.png", dpi=150)
print(f"Figure written to {OUT}")
