"""coldscatter - coupled-channel scattering for ultracold atoms and molecules."""
from . import units
from .analysis import feshbach_a, fit_feshbach
from .models import single_channel, two_channel_feshbach
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

__version__ = "0.1.0.dev0"

__all__ = [
    "Constant",
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
    "single_channel",
    "two_channel_feshbach",
    "units",
    "vdw_length",
]
