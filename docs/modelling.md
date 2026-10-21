# Building and validating models

## Choosing the number of latent variables

With `n_components="auto"` (the default), the model is cross-validated for
1 to `max_components` LVs and one is chosen from the RMSECV curve:

| `lv_rule` | Picks |
|---|---|
| `"parsimonious"` (default) | Fewest LVs with RMSECV within `lv_tolerance` (2 %) of the minimum |
| `"min"` | Global minimum of RMSECV |
| `"first_min"` | First local minimum |

`"min"` often adds LVs that improve RMSECV by a negligible amount and make
the model less robust. Set `n_components=5` to fix the number instead.

The full curve is in `model.rmsecv_curve_` and can be plotted with
`specpls.plotting.plot_rmsecv(model)`.

## Cross-validation schemes

| `cv=` | Scheme | When to use |
|---|---|---|
| `10` or `"venetian:10"` | Venetian blinds | Samples ordered by time or concentration |
| `"kfold:5"` | Contiguous blocks | Checking extrapolation along the sample order |
| `"loo"` | Leave-one-out | Very small data sets |
| `"batch"` + `groups=` | Leave-one-batch-out | **Bioprocess data**: samples from the same run are correlated |

!!! warning "Use leave-one-batch-out for bioreactor data"
    Daily samples from the same run share media, cells and fluorescence
    background. Random or venetian CV puts samples from the same run in both
    training and validation, which makes RMSECV look better than performance
    on a new run will be.

## Figures of merit

```python
model.calibration_  # rmsec, r2_cal, rmsecv, r2_cv
model.evaluate(X_test, y_test)  # rmsep, r2, bias, sep, rpd, n
model.summary(X_test, y_test)  # one row for a results table
```

- **RMSEC**: error on the calibration samples (always optimistic).
- **RMSECV**: cross-validation error, used for model selection.
- **RMSEP**: error on an independent test set, for the final performance claim.
- **SEP** and **bias** split RMSEP into random and systematic error.
- **RPD**: SD of reference values / SEP.

## Algorithm

`PLSModel` uses scikit-learn's NIPALS `PLSRegression` with mean-centring only
(`scale=False`). For a single response this gives the same model as SIMPLS
(MATLAB `plsregress`, PLS_Toolbox); the test suite checks this against an
independent SIMPLS implementation.
