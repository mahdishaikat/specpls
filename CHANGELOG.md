# Changelog

All notable changes are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com) and the project uses
[Semantic Versioning](https://semver.org).

## [Unreleased]

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
