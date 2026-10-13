import json

import pandas as pd
import pytest

from specpls.cli import main


@pytest.fixture
def csv_files(culture, tmp_path):
    frame = culture.to_frame()
    frame["is_test"] = frame["batch"] == "B6"
    cal = tmp_path / "calibration.csv"
    new = tmp_path / "new.csv"
    frame.to_csv(cal, index=False)
    frame[frame["is_test"]].drop(columns=["glucose", "lactate"]).to_csv(new, index=False)
    return cal, new


def test_train_then_predict(csv_files, tmp_path, capsys):
    cal, new = csv_files
    model_path = tmp_path / "glucose.joblib"
    exit_code = main(
        [
            "train",
            str(cal),
            "--target",
            "glucose",
            "--preprocess",
            "snv",
            "--region",
            "400:1800",
            "--cv",
            "batch",
            "--groups",
            "batch",
            "--test-col",
            "is_test",
            "--out",
            str(model_path),
        ]
    )
    assert exit_code == 0
    assert model_path.exists()
    summary = json.loads(capsys.readouterr().out)
    assert summary["target"] == "glucose"
    assert "rmsep" in summary

    out_csv = tmp_path / "pred.csv"
    args = ["predict", str(model_path), str(new), "--out", str(out_csv), "--id-col", "sample_id"]
    assert main(args) == 0
    predictions = pd.read_csv(out_csv)
    assert list(predictions.columns) == ["sample_id", "glucose_pred", "t2", "q", "outside_limits"]
    assert len(predictions) == 10


def test_fixed_lv(csv_files, tmp_path, capsys):
    cal, _ = csv_files
    model_path = tmp_path / "m.joblib"
    main(["train", str(cal), "--target", "lactate", "--lv", "3", "--out", str(model_path)])
    assert json.loads(capsys.readouterr().out)["lvs"] == 3


def test_bad_region_argument(csv_files):
    cal, _ = csv_files
    with pytest.raises(SystemExit):
        main(["train", str(cal), "--target", "glucose", "--region", "abc", "--out", "x"])


def test_version(capsys):
    with pytest.raises(SystemExit):
        main(["--version"])
    assert "specpls" in capsys.readouterr().out
