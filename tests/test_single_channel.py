import warnings

import numpy as np
from helpers import C6, MU, lj_for_rm, ode_reference_scalar

import coldscatter as cs
from coldscatter import units as u

E_COLD = u.from_kelvin(1e-7)


def test_square_well_exact():
    """a = R0 [1 - tan(K R0)/(K R0)] for a square well with a hard wall at the origin."""
    v0, r0 = 2e-6, 50.0
    k0 = np.sqrt(2 * MU * v0)
    prob = cs.single_channel(MU, cs.FromFunction(lambda r: np.where(r < r0, -v0, 0.0)))
    exact = r0 * (1 - np.tan(k0 * r0) / (k0 * r0))
    for h in (0.05, 0.025):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = prob.solve(u.from_kelvin(1e-9), cs.Grid(h, 3000.0, h))
        assert abs(res.scattering_length().real - exact) / abs(exact) < 1e-4


def test_lennard_jones_against_independent_ode():
    pot = lj_for_rm(11.0)
    grid = cs.Grid(7.0, 600.0, 0.01)
    a_num = cs.single_channel(MU, pot).solve(E_COLD, grid).scattering_length().real
    a_ref = ode_reference_scalar(pot, MU, E_COLD, grid)
    assert abs(a_num - a_ref) / abs(a_ref) < 1e-4


def test_fourth_order_convergence_in_step():
    prob = cs.single_channel(MU, lj_for_rm(11.0))
    a = [prob.solve(E_COLD, cs.Grid(4.0, 1000.0, h)).scattering_length().real
         for h in (0.04, 0.02, 0.01)]
    ratio = abs(a[0] - a[1]) / abs(a[1] - a[2])
    assert ratio > 8.0          # ideal Numerov: 16


def test_rmax_convergence_is_cubic_tail_error():
    """Neglecting -C6/R^6 beyond r_max leaves an error that falls roughly like r_max^-3."""
    prob = cs.single_channel(MU, lj_for_rm(11.0))
    a = [prob.solve(E_COLD, cs.Grid(4.0, rmax, 0.02)).scattering_length().real
         for rmax in (500, 1000, 2000)]
    assert abs(a[1] - a[2]) < abs(a[0] - a[1]) / 4.0


def test_van_der_waals_universality():
    """Median a over many short-range phases ~ Gribakin-Flambaum mean scattering length."""
    a_bar = cs.mean_scattering_length(MU, C6)
    vals = [cs.single_channel(MU, lj_for_rm(rm))
            .solve(u.from_kelvin(1e-9), cs.Grid(4.0, 3000.0, 0.02)).scattering_length().real
            for rm in np.linspace(9.0, 12.0, 120)]
    assert abs(np.median(vals) - a_bar) / a_bar < 0.10


def test_elastic_cross_section_low_energy_limit():
    res = cs.single_channel(MU, lj_for_rm(11.0)).solve(u.from_kelvin(1e-9),
                                                       cs.Grid(7.0, 4000.0, 0.02))
    sigma = res.cross_sections()["elastic"]
    assert abs(sigma - 4 * np.pi * abs(res.scattering_length()) ** 2) / sigma < 1e-3
    assert abs(res.cross_sections()["inelastic"]) < 1e-9


def test_p_wave_threshold_law():
    """tan(delta_1) ~ k^3: quadrupling the energy multiplies it by 2^3."""
    prob = cs.single_channel(MU, lj_for_rm(11.0), ell=1)
    k = [prob.solve(u.from_kelvin(e), cs.Grid(7.0, 3000.0, 0.02)).K[0, 0] for e in (1e-8, 4e-8)]
    assert abs(k[1] / k[0] - 8.0) < 0.8
