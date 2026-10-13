"""Quick start: build, validate, diagnose and save a glucose PLS model.

Run with:  python examples/01_quickstart.py
Uses simulated data, so it works without any instrument files.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write figures to files; remove to show windows

from specpls import PLSModel, load_model
from specpls.datasets import make_synthetic_cell_culture
from specpls.plotting import plot_overview

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

# 1. Data: 6 simulated fed-batch runs, one Raman spectrum per day.
data = make_synthetic_cell_culture(random_state=1)
is_test = (data.meta["batch"] == "B6").to_numpy()  # hold out one whole run
cal, test = data.subset(~is_test), data.subset(is_test)
print(f"Calibration: {len(cal)} spectra, test: {len(test)} spectra")

# 2. Model: Savitzky-Golay 1st derivative, fingerprint region,
#    LVs chosen by leave-one-batch-out cross-validation.
model = PLSModel(
    preprocessing="savgol:11:2:1",
    regions=[(400, 1800)],
    max_components=10,
    cv="batch",
)
model.fit(
    cal.X,
    cal.target("glucose"),
    wavenumbers=cal.wavenumbers,
    groups=cal.meta["batch"],
    target_name="glucose",
)

# 3. Figures of merit (RMSEC / RMSECV / RMSEP).
for key, value in model.summary(test.X, test.target("glucose")).items():
    print(f"  {key:>14}: {value:.3f}" if isinstance(value, float) else f"  {key:>14}: {value}")

# 4. Diagnostics: are any test spectra outside the model space?
flags = model.outliers(test.X)
n_out = int((flags["t2_flag"] | flags["q_flag"]).sum())
print(f"Test spectra outside 95 % T2/Q limits: {n_out}")

# 5. Figures.
fig = plot_overview(model, cal.X, cal.target("glucose"), test.X, test.target("glucose"), "g/L")
fig.savefig(OUT / "glucose_overview.png", dpi=150)

# 6. Save for deployment, reload, predict.
path = model.save(OUT / "glucose.joblib")
reloaded = load_model(path)
print("Reloaded model predicts:", reloaded.predict(test.X[:3]).round(2))
print(f"Figures and model written to {OUT}")
