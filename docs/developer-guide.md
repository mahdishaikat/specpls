# Developer guide

How the project is engineered, and how to work on it.

## Design principles

- **scikit-learn compatible.** Every transformer and the model implement
  `fit` / `transform` / `predict` and `get_params`, so they work with
  `Pipeline`, `clone`, `GridSearchCV` and other tools.
- **No leakage by construction.** Preprocessing is part of the model and is
  re-fitted in each CV fold.
- **Fail loudly.** Wrong shapes, mismatched wavenumber axes, NaNs in the
  reference values and invalid parameters raise clear errors.
- **Small modules with one job** (preprocessing, validation, model, io,
  plotting, cli). Plotting is optional so servers can install without matplotlib.

## Tooling

| Tool | Purpose | Command |
|---|---|---|
| git + GitHub | Version control, pull requests, review | `git switch -c feat/...` |
| pytest | Unit, reference and integration tests; doctests | `pytest` |
| pytest-cov | Coverage | `pytest --cov` |
| ruff | Linting and formatting | `ruff check . && ruff format .` |
| pre-commit | Runs ruff before every commit | `pre-commit install` |
| GitHub Actions | CI on Linux/Windows, Python 3.10–3.13, docs build | automatic |
| MkDocs + mkdocstrings | This site, API docs generated from docstrings | `mkdocs serve` |

## Test strategy

| Kind | Example |
|---|---|
| Known answer | SNV of `3·x + 5` equals SNV of `x`; MSC recovers the reference |
| Reference implementation | `PLSModel` predictions equal an independent SIMPLS and plain scikit-learn |
| Property | Mean VIP² = 1; mean calibration T² = A(n-1)/n; CV tests each sample once |
| Behaviour | A spectrum with an unseen spike is flagged by Q |
| Error handling | Wrong widths, unknown preprocessing names, bad CV specs |
| Integration | CLI train → predict round trip; example scripts run |

## Workflow

See [CONTRIBUTING.md](https://github.com/YOUR-GITHUB-USERNAME/specpls/blob/main/CONTRIBUTING.md)
for branches, commit messages, pull requests and the review checklist.

## Roadmap

- Locally weighted PLS (LW-PLS) for nonlinear and drifting processes
- Interval PLS / VIP-based variable selection
- Model transfer (piecewise direct standardisation) between instruments
- Multi-response (PLS2) support
