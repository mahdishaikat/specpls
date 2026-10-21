# Getting started

## Install

```bash
pip install "specpls[plot] @ git+https://github.com/mahdishaikat/specpls"
```

Python 3.10 or newer is required. The `plot` extra installs matplotlib.

## Prepare your data

`specpls` reads a **wide table**: one row per spectrum, metadata columns
first, then one column per wavenumber. Any column whose header is a number
is treated as a wavenumber.

| sample_id | batch | day | glucose | lactate | 400 | 401 | ... | 1800 |
|---|---|---|---|---|---|---|---|---|
| R1-D00 | R1 | 0 | 5.78 | 0.07 | 812.1 | 815.4 | ... | 402.3 |
| R1-D01 | R1 | 1 | 5.21 | 0.45 | 820.7 | 822.0 | ... | 410.9 |

```python
from specpls import load_csv

data = load_csv("calibration.csv")
data.X  # (n_samples, n_wavenumbers) array
data.wavenumbers  # (n_wavenumbers,) array
data.meta  # pandas DataFrame with sample_id, batch, glucose, ...
```

Coming from MATLAB? Save with
`save('spectra.mat', 'X', 'wavenumbers', 'glucose', 'batch', '-v7')` and load with
`load_mat("spectra.mat", meta_keys=["glucose", "batch"])`.

## Fit a model

```python
from specpls import PLSModel

is_test = (data.meta["batch"] == "R6").to_numpy()  # hold out a whole run
cal, test = data.subset(~is_test), data.subset(is_test)

model = PLSModel(
    preprocessing="savgol:15:2:1,snv",
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

model.summary(test.X, test.target("glucose"))
```

## Plot

```python
from specpls.plotting import plot_overview

fig = plot_overview(
    model, cal.X, cal.target("glucose"), test.X, test.target("glucose"), units="g/L"
)
fig.savefig("glucose.png", dpi=150)
```

## Save and reuse

```python
model.save("glucose.joblib")

from specpls import load_model

model = load_model("glucose.joblib")
model.predict(new_spectra, wavenumbers=new_axis)
```
