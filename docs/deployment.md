# Deployment

## Saving models

```python
model.save("models/glucose_v1.joblib")
```

The file contains the fitted pipeline (region selection, preprocessing, PLS),
the calibration wavenumber axis, the control-limit statistics and metadata:
specpls version, scikit-learn version and creation time.

`load_model` warns if the scikit-learn version differs from the one used for
saving, because pickled estimators are not guaranteed to be compatible across
versions. Re-validate the model in that case.

!!! danger "Only load trusted files"
    joblib/pickle files can execute code when loaded. Load model files only
    from sources you trust.

## Guarding predictions

```python
model.predict(X_new, wavenumbers=new_axis)
```

Passing the wavenumber axis makes the model refuse spectra recorded on a
different axis (for example after an instrument or software change) instead
of silently returning wrong numbers. Combine with `model.outliers()` to flag
spectra outside the calibration space.

## Command line

```bash
specpls train calibration.csv --target glucose --preprocess "savgol:15:2:1,snv" \
    --region 400:1800 --cv batch --groups batch --test-col is_test --out glucose.joblib

specpls predict glucose.joblib new.csv --id-col sample_id --out predictions.csv
```

`train` prints the summary table as JSON. `predict` writes predictions,
T², Q and an `outside_limits` column.
