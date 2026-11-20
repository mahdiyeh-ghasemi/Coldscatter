"""coldscatter - coupled-channel scattering for ultracold atoms and molecules."""
from . import atomic, units
from .analysis import feshbach_a, fit_feshbach
from .models import single_channel, two_channel_feshbach
from .plotting import export_results_csv, plot_cross_sections, plot_scattering_length
from .potentials import (
                         Constant,
                         Exponential,
                         FromFunction,
                         Gaussian,
                         LennardJones,
                         Potential,
                         mean_scattering_length,
                         vdw_length,
)
from .solver import CoupledChannels, Grid, ScatteringResult
from .long_range import LongRange

try:  # single source of truth: pyproject.toml
    from importlib.metadata import PackageNotFoundError, version as _version
    __version__ = _version("coldscatter")
except PackageNotFoundError:  # running from a source tree without installation
    __version__ = "0.1.1"

__all__ = [
    "Constant",
    "atomic",
    "CoupledChannels",
    "Exponential",
    "FromFunction",
    "Gaussian",
    "Grid",
    "LennardJones",
    "LongRange",
    "Potential",
    "ScatteringResult",
    "feshbach_a",
    "fit_feshbach",
    "mean_scattering_length",
    "export_results_csv",
    "plot_cross_sections",
    "plot_scattering_length",
    "single_channel",
    "two_channel_feshbach",
    "units",
    "vdw_length",
]
