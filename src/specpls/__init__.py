"""specpls: PLS calibration models for process spectroscopy.

Preprocessing, cross-validated latent-variable selection, model diagnostics
(T2, Q residuals, VIP) and deployment helpers for Raman, NIR and other
spectroscopic PAT data.
"""

from importlib.metadata import PackageNotFoundError, version

from .io import SpectralData, load_csv, load_mat
from .model import PLSModel, load_model
from .preprocessing import (
    EPO,
    MSC,
    SNV,
    AsLSBaseline,
    Detrend,
    RegionSelector,
    SavitzkyGolay,
    VariableSelector,
    build_preprocessing,
)
from .selection import IPLSResult, ipls

try:
    __version__ = version("specpls")
except PackageNotFoundError:  # running from a source checkout without install
    __version__ = "0.0.0+unknown"

__all__ = [
    "__version__",
    "PLSModel",
    "load_model",
    "SpectralData",
    "load_csv",
    "load_mat",
    "SNV",
    "MSC",
    "SavitzkyGolay",
    "Detrend",
    "AsLSBaseline",
    "RegionSelector",
    "EPO",
    "VariableSelector",
    "build_preprocessing",
    "ipls",
    "IPLSResult",
]
