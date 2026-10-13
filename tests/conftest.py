"""Shared test fixtures."""

import numpy as np
import pytest

from specpls.datasets import make_synthetic_cell_culture


@pytest.fixture(scope="session")
def culture():
    """Small simulated CHO data set (6 batches x 10 days, 701 wavenumbers)."""
    return make_synthetic_cell_culture(random_state=42)


@pytest.fixture(scope="session")
def cal_test(culture):
    """Calibration batches B1-B5 and an independent test batch B6."""
    is_test = (culture.meta["batch"] == "B6").to_numpy()
    return culture.subset(~is_test), culture.subset(is_test)


@pytest.fixture
def rng():
    return np.random.default_rng(0)
