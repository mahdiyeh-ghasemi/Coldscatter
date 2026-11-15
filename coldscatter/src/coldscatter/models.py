"""Ready-made model systems (useful for teaching, tests and prototyping)."""
from __future__ import annotations

import numpy as np

from .potentials import Gaussian, LennardJones
from .solver import CoupledChannels


def single_channel(mu: float, potential, ell: int = 0) -> CoupledChannels:
    """One channel with threshold 0."""
    return CoupledChannels(mu, [0.0], [[potential]], ell=ell, names=["open"])


def two_channel_feshbach(mu: float, c6: float, c12_open: float, c12_closed: float,
                         coupling: float, coupling_r0: float, coupling_sigma: float,
                         delta_mu: float, b_ref: float = 0.0, ell: int = 0) -> CoupledChannels:
    """Open + closed channel model of a magnetic Feshbach resonance.

    Both channels are Lennard-Jones with a common ``C6``; the closed channel's
    threshold moves linearly with the field, ``E_closed(B) = delta_mu * (B - b_ref)``
    (``delta_mu`` in hartree/gauss, i.e. the differential magnetic moment), and a
    Gaussian short-range term of height ``coupling`` (hartree) mixes the channels.
    """
    return CoupledChannels(
        mu,
        lambda B: np.array([0.0, delta_mu * (B - b_ref)]),
        [[LennardJones(c6, c12_open), Gaussian(coupling, coupling_r0, coupling_sigma)],
         [None, LennardJones(c6, c12_closed)]],
        ell=ell,
        names=["open", "closed"],
    )
