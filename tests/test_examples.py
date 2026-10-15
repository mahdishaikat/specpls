"""Smoke tests: the example scripts must keep working as the API evolves."""

import runpy
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.mark.slow
@pytest.mark.parametrize(
    "script",
    ["01_quickstart.py", "02_compare_preprocessing.py", "04_epo_overfitting_guard.py"],
)
def test_example_runs(script, monkeypatch, tmp_path):
    pytest.importorskip("matplotlib")
    monkeypatch.chdir(tmp_path)
    runpy.run_path(str(EXAMPLES / script), run_name="__main__")
