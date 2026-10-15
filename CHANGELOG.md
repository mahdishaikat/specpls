# Changelog

All notable changes are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com) and the project uses
[Semantic Versioning](https://semver.org).

## [Unreleased]

## [0.2.0] - 2026-10-15

### Added
- `EPO` (External Parameter Orthogonalisation) preprocessing, y-gradient or
  explicit clutter; string spec `"epo:k"`.
- `VariableSelector` for column subsets after preprocessing.
- Forward interval PLS: `specpls.ipls` / `IPLSResult` and `plot_ipls`.
- Over-fitting guard `max_cv_ratio` (RMSECV / RMSEC limit) for LV selection
  and iPLS; `cv_ratio` reported in model summaries.
- Example `04_epo_overfitting_guard.py`.

## [0.1.0] - 2026-10-13

### Added
- Preprocessing transformers: SNV, MSC, Savitzky-Golay, Detrend, AsLS baseline,
  region selection, plus `build_preprocessing` for text specifications.
- `PLSModel` with RMSECV-based latent-variable selection, fold-wise
  preprocessing, Hotelling's T² / Q residuals with limits, VIP, save/load.
- Venetian-blinds, contiguous, leave-one-out and leave-one-batch-out CV.
- CSV and MATLAB loaders, simulated cell-culture data set, plotting helpers.
- `specpls train` / `specpls predict` command-line interface.
- Test suite, GitHub Actions CI, MkDocs documentation.
