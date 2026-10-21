# Diagnostics

## VIP scores

```python
vip = model.vip()  # one value per wavenumber in the model
important = model.model_wavenumbers_[vip > 1]
```

Variable Importance in Projection summarises how much each wavenumber
contributes to the explained variance of the response. The mean of VIP² is 1,
so VIP > 1 marks above-average importance. Check that important regions
match known bands of the analyte (for glucose, around 1060–1130 cm⁻¹) rather
than background.

## Hotelling's T² and Q residuals

```python
flags = model.outliers(X_new, confidence=0.95)
flags["t2"], flags["q"]  # per-sample statistics
flags["t2_limit"], flags["q_limit"]  # control limits
flags["t2_flag"] | flags["q_flag"]  # samples to investigate
```

- **T²**: distance from the model centre *within* the LV space. High T² means
  an unusual but model-consistent sample (e.g. an extreme concentration).
- **Q**: squared residual *outside* the LV space. High Q means the spectrum
  contains variation the model has not seen (new component, fluorescence
  change, probe fouling). Predictions for these samples should not be trusted.

Limits:

- T²: `A (n-1)/(n-A) · F(α; A, n-A)`
- Q: Box's scaled χ² approximation from the calibration residuals.

## Plots

```python
from specpls.plotting import plot_t2_q, plot_vip, plot_overview

plot_t2_q(model, X_new, labels=sample_ids)
```
