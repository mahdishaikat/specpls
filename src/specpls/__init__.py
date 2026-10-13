"""specpls: PLS calibration models for process spectroscopy.

Preprocessing, cross-validated latent-variable selection, model diagnostics
(T2, Q residuals, VIP) and deployment helpers for Raman, NIR and other
spectroscopic PAT data.
"""

from importlib.metadata import PackageNotFoundError, version



try:
    __version__ = version("specpls")
except PackageNotFoundError:  # running from a source checkout without install
    __version__ = "0.0.0+unknown"

__all__ = [
    "__version__",
]
