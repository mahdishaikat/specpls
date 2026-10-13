# specpls

**PLS calibration models for process spectroscopy (Raman, NIR) in bioprocessing.**

`specpls` packages a standard chemometrics workflow as a tested Python library:

1. **Load** spectra and reference values from CSV or MATLAB files.
2. **Preprocess** with SNV, MSC, Savitzky-Golay, baseline correction and
   region selection.
3. **Calibrate** a PLS model, choosing the number of latent variables by
   cross-validation (including leave-one-batch-out).
4. **Diagnose** with RMSEC / RMSECV / RMSEP, VIP, Hotelling's T² and Q residuals.
5. **Deploy** with versioned model files, axis checks and a command-line tool.

```python
from specpls import PLSModel
from specpls.datasets import make_synthetic_cell_culture

data = make_synthetic_cell_culture(random_state=0)
model = PLSModel(preprocessing="savgol:11:2:1", cv="batch").fit(
    data.X, data.target("glucose"), groups=data.meta["batch"], target_name="glucose"
)
print(model.summary())
```

Start with [Getting started](getting-started.md), or see the
[API reference](api.md).
