# Preprocessing

All preprocessing steps are scikit-learn transformers. Pass them to
`PLSModel(preprocessing=...)` as a short string, a list, or a `Pipeline`.

| Name in string | Class | Purpose | Arguments |
|---|---|---|---|
| `snv` | `SNV` | Per-spectrum centring and scaling; removes multiplicative scatter | none |
| `msc` | `MSC` | Regress each spectrum on a reference (training mean) | none |
| `savgol` / `sg` | `SavitzkyGolay` | Smoothing or derivatives | window, polyorder, deriv |
| `detrend` | `Detrend` | Subtract a polynomial baseline | order |
| `asls` | `AsLSBaseline` | Asymmetric least-squares baseline, good for fluorescence | lam, p, n_iter |

Steps are applied left to right:

```python
PLSModel(preprocessing="savgol:15:2:1,snv")          # SG 1st derivative, then SNV
PLSModel(preprocessing="asls:1e5:0.01,savgol:11:2:0")  # baseline, then smooth
```

or with objects:

```python
from specpls import SNV, SavitzkyGolay
PLSModel(preprocessing=[SavitzkyGolay(15, 2, 1), SNV()])
```

## Region selection

Use `regions` on the model rather than slicing arrays by hand, so the
selection is stored with the model and applied automatically at prediction:

```python
PLSModel(regions=[(400, 1800)])                 # fingerprint region
PLSModel(regions=[(800, 1200), (1400, 1500)])  # several windows
```

!!! note "Why preprocessing lives inside the model"
    Data-dependent steps (MSC's reference spectrum, for example) are fitted on
    the training folds only during cross-validation. Preprocessing the whole
    data set first and then cross-validating leaks information from the
    validation samples and gives an optimistic RMSECV.

## Choosing preprocessing

Compare candidates on **RMSECV**, and keep the test set for a final check:

```bash
python examples/02_compare_preprocessing.py
```
