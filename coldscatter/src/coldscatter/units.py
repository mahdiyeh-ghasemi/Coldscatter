"""Atomic units and conversion helpers.

Internally, *everything* in ``coldscatter`` is in Hartree atomic units:

=========  ==========================================
length     bohr (a0)
energy     hartree (Eh)
mass       electron masses (m_e)
=========  ==========================================

Magnetic field is the one exception at the user-facing level: it is given in
gauss and converted where needed (see :data:`MU_B_HARTREE_PER_GAUSS`).
Constants come from SciPy's CODATA tables.
"""
from __future__ import annotations

import numpy as np
from scipy import constants as _c

#: 1 hartree expressed in kelvin
HARTREE_IN_K: float = _c.physical_constants["hartree-kelvin relationship"][0]
#: 1 atomic mass unit expressed in electron masses
AMU_IN_ME: float = _c.physical_constants["atomic mass constant"][0] / _c.m_e
#: 1 bohr in metres
BOHR_IN_M: float = _c.physical_constants["Bohr radius"][0]
#: Bohr magneton times 1 gauss, in hartree
MU_B_HARTREE_PER_GAUSS: float = (
    _c.physical_constants["Bohr magneton in K/T"][0] * 1e-4 / HARTREE_IN_K
)


def from_kelvin(energy_k):
    """Energy in kelvin (k_B * T) -> hartree."""
    return np.asarray(energy_k, dtype=float) / HARTREE_IN_K


def to_kelvin(energy_hartree):
    """Energy in hartree -> kelvin."""
    return np.asarray(energy_hartree, dtype=float) * HARTREE_IN_K


def amu(mass_amu):
    """Mass in atomic mass units -> electron masses."""
    return np.asarray(mass_amu, dtype=float) * AMU_IN_ME


def reduced_mass(m1_amu: float, m2_amu: float) -> float:
    """Reduced mass of two particles (given in amu), returned in atomic units."""
    return float(amu(m1_amu * m2_amu / (m1_amu + m2_amu)))


def zeeman_shift(magnetic_moment_mu_b: float, b_gauss):
    """Linear Zeeman shift ``mu * B`` in hartree (moment in units of mu_B)."""
    return magnetic_moment_mu_b * MU_B_HARTREE_PER_GAUSS * np.asarray(b_gauss, float)
