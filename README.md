# specpls

[![CI](https://github.com/mahdishaikat/specpls/actions/workflows/ci.yml/badge.svg)](https://github.com/mahdishaikat/specpls/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%E2%80%933.13-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**PLS calibration models with time-gated Raman Spectroscopy in upstream bioprocessing.**

`specpls` turns the usual chemometrics workflow (preprocess spectra, choose
latent variables by cross-validation, check diagnostics, report RMSEC / RMSECV /
RMSEP, deploy) into a tested, reusable Python package. It is built on
scikit-learn, so every piece works with the rest of the Python ML ecosystem.

```python
from specpls import PLSModel, load_csv

data = load_csv("calibration.csv")  # metadata + one column per wavenumber
model = PLSModel(
    preprocessing="savgol:15:2:1,snv",  # SG 1st derivative, then SNV
    regions=[(400, 1800)],  # fingerprint region
    cv="batch",  # leave-one-bioreactor-run-out
).fit(
    data.X,
    data.target("glucose"),
    wavenumbers=data.wavenumbers,
    groups=data.meta["batch"],
    target_name="glucose",
)

model.summary()  # {'lvs': 5, 'rmsec': ..., 'rmsecv': ..., 'r2_cv': ...}
model.vip()  # variable importance per wavenumber
model.outliers(X_new)  # Hotelling's T2 / Q residuals with 95 % limits
model.save("glucose.joblib")
```

## Features

| Area | What is included |
|---|---|
| **Preprocessing** | SNV, MSC, Savitzky-Golay smoothing/derivatives, polynomial detrend, AsLS baseline, wavenumber region selection. All scikit-learn transformers, chainable from a short string like `"asls:1e5:0.01,snv"`. |
| **Model building** | Single-response PLS with automatic LV selection from RMSECV (`parsimonious`, `min`, `first_min` rules). Preprocessing is re-fitted inside every CV fold, so there is no data leakage. |
| **Validation** | Venetian blinds, contiguous blocks, leave-one-out and leave-one-batch-out CV; RMSE, R², bias, SEP and RPD. Optional over-fitting guard: only LVs with RMSECV / RMSEC ≤ a limit (e.g. 1.2) are eligible. |
| **Variable selection** | EPO (External Parameter Orthogonalisation) and forward interval PLS (iPLS). |
| **Diagnostics** | Hotelling's T² and Q residuals with control limits, VIP scores, regression vector, LV scores. |
| **Deployment** | Save/load with version metadata, wavenumber-axis check on new spectra, and a CLI (`specpls train`, `specpls predict`) for colleagues who don't write Python. |
| **I/O** | Wide CSV files and MATLAB `.mat` files. |
| **Plots** | RMSECV curve, predicted vs measured, VIP, T²–Q and a four-panel overview. |

## Installation

```bash
pip install "specpls[plot] @ git+https://github.com/mahdishaikat/specpls"
```

For development:

```bash
git clone https://github.com/mahdishaikat/specpls
cd specpls
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev,docs]"
pre-commit install
```

## Quick start

```bash
python examples/01_quickstart.py            # fit, validate, plot and save a model
python examples/02_compare_preprocessing.py # publication-style comparison table
python examples/04_epo_overfitting_guard.py  # SNV + smoothing + EPO, RMSECV/RMSEC <= 1.2
```

The examples use **simulated** Raman-like CHO culture data
(`specpls.datasets.make_synthetic_cell_culture`), so they run without any
instrument files.

### Command line

```bash
specpls train calibration.csv --target glucose --preprocess "savgol:15:2:1,snv" \
    --region 400:1800 --cv batch --groups batch --out glucose.joblib

specpls predict glucose.joblib new_spectra.csv --id-col sample_id --out predictions.csv
```

`predict` writes the predictions together with T², Q and an `outside_limits`
flag, so results from spectra unlike the calibration set are easy to spot.

## For MATLAB / PLS_Toolbox users

| PLS_Toolbox / MATLAB | specpls |
|---|---|
| `plsregress` / SIMPLS | `PLSModel` (NIPALS; identical predictions for one response, checked in the tests) |
| Preprocessing: SNV, MSC, `savgol`, baseline | `"snv"`, `"msc"`, `"savgol:w:p:d"`, `"asls"` / `"detrend"` |
| CV "vet", "con", "loo" | `cv=10`, `cv="kfold:10"`, `cv="loo"` |
| Leave-batch-out | `cv="batch"` + `groups=` |
| RMSEC, RMSECV, RMSEP | `model.calibration_`, `model.evaluate(X, y)` |
| VIP, T², Q | `model.vip()`, `model.hotelling_t2(X)`, `model.q_residuals(X)` |
| `.mat` files | `specpls.load_mat("file.mat", meta_keys=[...])` |

## Project layout

```
src/specpls/
    preprocessing.py   scikit-learn transformers (SNV, MSC, SG, AsLS, ...)
    validation.py      metrics, CV splitters, LV selection rules
    model.py           PLSModel: fitting, diagnostics, persistence
    io.py              SpectralData container, CSV / MATLAB loaders
    datasets.py        simulated cell-culture spectra
    plotting.py        diagnostic plots (optional matplotlib)
    cli.py             `specpls` command-line tool
tests/                 pytest suite (unit, numerical reference, CLI, examples)
docs/                  MkDocs site (user guide + API reference)
examples/              runnable scripts
```

## Development

```bash
pytest                      # run the tests
pytest --cov                # with coverage
ruff check . && ruff format .   # lint and format
mkdocs serve                # preview documentation at http://127.0.0.1:8000
```

Every push and pull request runs linting, the tests on Python 3.10–3.13
(Linux and Windows) and a docs build on GitHub Actions. See
[CONTRIBUTING.md](CONTRIBUTING.md) for the branch, commit and code-review workflow.

## Background

The defaults reflect work on Raman and time-gated Raman spectroscopy for CHO
cell-culture monitoring:

> Shaikat, M. M., Dhara, V. G., Drennen, J. K., Dong, G., & Anderson, C. A. (2025).
> Enhancing analytical sensitivity in upstream bioprocess using time-gated Raman
> spectroscopy. *Bioprocess and Biosystems Engineering*.
> https://doi.org/10.1007/s00449-025-03261-y

## License

MIT. See [LICENSE](LICENSE).
