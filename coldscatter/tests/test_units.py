import numpy as np
from scipy import constants as c

from coldscatter import units as u


def test_hartree_kelvin():
    assert abs(u.HARTREE_IN_K - 315775.0248) < 1e-3


def test_round_trip_kelvin():
    assert np.isclose(u.to_kelvin(u.from_kelvin(1e-6)), 1e-6)


def test_bohr_magneton_in_mhz_per_gauss():
    hartree_hz = c.physical_constants["hartree-hertz relationship"][0]
    mhz_per_gauss = u.MU_B_HARTREE_PER_GAUSS * hartree_hz / 1e6
    assert abs(mhz_per_gauss - 1.39962449) < 1e-6


def test_reduced_mass_identical_atoms():
    assert np.isclose(u.reduced_mass(87.0, 87.0), 0.5 * 87.0 * u.AMU_IN_ME)
