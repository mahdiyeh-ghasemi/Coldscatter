import numpy as np

import coldscatter as cs
from coldscatter import units as u
from helpers import MU, C6, DMU, feshbach_model, lj_for_rm

E_COLD = u.from_kelvin(1e-7)
GRID = cs.Grid(7.0, 800.0, 0.02)


def test_decoupled_limit_equals_single_channel():
    two = cs.two_channel_feshbach(MU, C6, C6 * 11.0**6 / 2, C6 * 10.4**6 / 2, coupling=0.0,
                                  coupling_r0=11.0, coupling_sigma=1.5, delta_mu=DMU)
    one = cs.single_channel(MU, lj_for_rm(11.0))
    a2 = two.solve(E_COLD, GRID, B=10.0).scattering_length()
    a1 = one.solve(E_COLD, GRID).scattering_length()
    assert abs(a2 - a1) < 1e-8 * abs(a1)


def test_s_matrix_unitary_and_symmetric_with_two_open_channels():
    res = feshbach_model().solve(E_COLD, GRID, B=-10.0)    # B < 0: both channels open
    assert list(res.open_channels) == [0, 1]
    S = res.S
    assert np.abs(S @ S.conj().T - np.eye(2)).max() < 1e-10
    assert np.abs(S - S.T).max() < 1e-10
    assert res.k_asymmetry < 1e-6


def test_optical_theorem_and_flux_conservation():
    res = feshbach_model().solve(E_COLD, GRID, B=-10.0)
    sig = res.cross_sections()
    pref = np.pi / res.k_entrance**2
    assert sig["inelastic"] > 0.0
    assert abs(sig["loss"]) < 1e-9 * sig["total"]                # unitary model: no loss
    assert np.isclose(sig["inelastic"], sum(sig["to_channel"].values()), rtol=1e-12)
    assert np.isclose(sig["total"], 2 * pref * (1 - res.s(0, 0).real), rtol=1e-8)


def test_resonance_pole_and_fit():
    prob = feshbach_model()
    B = np.linspace(22.0, 44.0, 221)
    a = prob.scan_field(B, E_COLD, GRID)
    assert np.abs(a.imag).max() < 1e-6 * np.abs(a.real).max()        # lossless
    ar = a.real
    jump = int(np.argmax(np.abs(np.diff(ar))))
    assert ar[jump] > 0 > ar[jump + 1]                               # a -> +inf, then -inf
    a_bg, B0, delta = cs.fit_feshbach(B, ar)
    assert abs(B0 - 0.5 * (B[jump] + B[jump + 1])) < 0.15


def test_scan_matches_single_solves():
    prob = feshbach_model()
    B = np.array([5.0, 20.0, 30.0])
    batched = prob.scan_field(B, E_COLD, GRID)
    single = np.array([prob.solve(E_COLD, GRID, B=b).scattering_length() for b in B])
    assert np.allclose(batched, single, rtol=1e-9)


def test_fit_recovers_synthetic_resonance():
    B = np.linspace(-5, 5, 401)
    a = cs.feshbach_a(B, 100.0, 0.013, 1.7)
    a_bg, B0, delta = cs.fit_feshbach(B, a)
    assert np.allclose([a_bg, B0, delta], [100.0, 0.013, 1.7], rtol=1e-5, atol=1e-5)
