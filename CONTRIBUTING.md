# Contributing to specpls

Thanks for helping. This page describes how changes get from an idea to `main`.

## 1. Set up

```bash
git clone https://github.com/mahdishaikat/specpls
cd specpls
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,docs]"
pre-commit install          # runs ruff on every commit
```

## 2. Branch

`main` is always releasable and protected: nobody pushes to it directly.

```bash
git switch main && git pull
git switch -c feat/lwr-model        # or fix/..., docs/..., test/..., refactor/...
```

## 3. Commit

Small commits, each doing one thing, using [Conventional Commits](https://www.conventionalcommits.org):

```
feat(model): add locally weighted PLS regression
fix(preprocessing): handle constant spectra in SNV
docs: explain leave-one-batch-out CV
test(model): compare predictions against SIMPLS reference
```

## 4. Test

Every change to behaviour comes with a test.

```bash
pytest                       # must pass
pytest --cov                 # check new code is covered
ruff check . && ruff format --check .
```

Testing guidelines for numerical code:

- **Known answers:** test against a hand-calculable case (e.g. SNV of a
  constant spectrum, MSC of an exactly scaled reference).
- **Reference implementations:** compare with an independent algorithm (the
  PLS model is checked against a SIMPLS implementation and plain scikit-learn).
- **Properties:** invariants that must always hold (mean VIP² = 1, every
  sample tested exactly once in CV, T² >= 0).
- **Failure modes:** bad input should raise a clear error, which is also tested.
- Use fixed random seeds so results are reproducible.

## 5. Open a pull request

```bash
git push -u origin feat/lwr-model
gh pr create --fill          # or use the GitHub web page
```

The PR template asks for a summary, how it was tested and a checklist. CI
must be green before review.

## 6. Code review

Reviewers look at, in order:

1. **Correctness:** is the science and the maths right? Is there leakage
   between calibration and validation data?
2. **Tests:** do they cover the new behaviour and the edge cases?
3. **API and naming:** is it consistent with scikit-learn conventions and the
   rest of the package? Would a process scientist understand it?
4. **Documentation:** docstrings (NumPy style), user guide, CHANGELOG entry.
5. **Style:** handled by ruff, so it is not argued about in review.

Authors respond to every comment (change it, or explain why not). Squash-merge
once approved and green.

## 7. Release

1. Update `CHANGELOG.md` and the version in `pyproject.toml`.
2. Merge to `main`, then tag and push the tag:
   `git tag v0.2.1 && git push origin v0.2.1`.
3. The **Release** workflow runs the tests, checks that the tag matches the
   version, builds the wheel and source distribution and publishes a GitHub
   Release with both attached.
