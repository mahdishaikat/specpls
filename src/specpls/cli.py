"""Command-line interface: ``specpls train`` and ``specpls predict``.

Lets colleagues build and apply models without writing Python, e.g.::

    specpls train calibration.csv --target glucose \\
        --preprocess "savgol:15:2:1,snv" --region 400:1800 \\
        --cv batch --groups batch --out glucose.joblib

    specpls predict glucose.joblib new_spectra.csv --out predictions.csv
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

from . import __version__
from .io import load_csv
from .model import PLSModel, load_model


def _parse_region(text: str) -> tuple[float, float]:
    try:
        low, high = (float(v) for v in text.split(":"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Region must look like 400:1800, got {text!r}") from exc
    return low, high


def _parse_cv(text: str):
    return int(text) if text.isdigit() else text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="specpls", description="PLS calibration models for spectroscopic data."
    )
    parser.add_argument("--version", action="version", version=f"specpls {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Fit a PLS model and save it.")
    train.add_argument("data", help="Wide CSV: metadata columns + one column per wavenumber.")
    train.add_argument("--target", required=True, help="Column with reference values.")
    train.add_argument("--preprocess", default="", help='e.g. "savgol:15:2:1,snv"')
    train.add_argument(
        "--region", action="append", type=_parse_region, help="Keep LOW:HIGH; repeatable."
    )
    train.add_argument("--lv", default="auto", help='Number of LVs or "auto" (default).')
    train.add_argument("--max-lv", type=int, default=10)
    train.add_argument("--cv", type=_parse_cv, default=10, help='e.g. 10, "kfold:5", "batch"')
    train.add_argument("--groups", help="Column with batch IDs for --cv batch.")
    train.add_argument(
        "--rule", default="parsimonious", choices=["parsimonious", "min", "first_min"]
    )
    train.add_argument("--test-col", help="Boolean/0-1 column marking test samples.")
    train.add_argument("--out", required=True, help="Output .joblib model file.")

    predict = sub.add_parser("predict", help="Apply a saved model to new spectra.")
    predict.add_argument("model", help=".joblib file from `specpls train`.")
    predict.add_argument("data", help="Wide CSV with the same wavenumber columns.")
    predict.add_argument("--out", help="Write predictions to this CSV (default: stdout).")
    predict.add_argument("--id-col", help="Column to copy into the output (e.g. sample_id).")
    predict.add_argument("--confidence", type=float, default=0.95)
    return parser


def _train(args) -> int:
    data = load_csv(args.data).dropna(args.target)
    test_mask = np.zeros(len(data), dtype=bool)
    if args.test_col:
        test_mask = data.meta[args.test_col].astype(bool).to_numpy()
    cal, test = data.subset(~test_mask), data.subset(test_mask)
    groups = cal.meta[args.groups].to_numpy() if args.groups else None

    model = PLSModel(
        preprocessing=args.preprocess or None,
        n_components="auto" if args.lv == "auto" else int(args.lv),
        max_components=args.max_lv,
        cv=args.cv,
        lv_rule=args.rule,
        regions=args.region,
    )
    model.fit(
        cal.X,
        cal.target(args.target),
        wavenumbers=cal.wavenumbers,
        groups=groups,
        target_name=args.target,
    )
    summary = model.summary(
        test.X if len(test) else None, test.target(args.target) if len(test) else None
    )
    model.save(args.out)
    print(json.dumps(summary, indent=2, default=float))
    print(f"Saved model to {args.out}", file=sys.stderr)
    return 0


def _predict(args) -> int:
    model = load_model(args.model)
    data = load_csv(args.data)
    out = pd.DataFrame()
    if args.id_col:
        out[args.id_col] = data.meta[args.id_col]
    name = model.target_name_ or "prediction"
    out[f"{name}_pred"] = model.predict(data.X, wavenumbers=data.wavenumbers)
    flags = model.outliers(data.X, args.confidence)
    out["t2"], out["q"] = flags["t2"], flags["q"]
    out["outside_limits"] = flags["t2_flag"] | flags["q_flag"]
    if args.out:
        out.to_csv(args.out, index=False)
        print(f"Wrote {len(out)} predictions to {args.out}", file=sys.stderr)
    else:
        out.to_csv(sys.stdout, index=False)
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    handlers = {"train": _train, "predict": _predict}
    return handlers[args.command](args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
