"""Regression tests for the fixes in 0.1.1 (see docs/FIXES_0.1.1.md)."""
import re
import warnings
from pathlib import Path

import numpy as np
import pytest

import coldscatter as cs
from coldscatter.asymptotics import (
    _free_closed_logderiv,
    _free_closed_ratio,
    _long_closed_logderiv,
    _long_closed_ratio,
)
from coldscatter.atomic import ALKALI_GROUND_STATES, alkali_hamiltonian, hyperfine_hamiltonian
from coldscatter.propagators import (
    log_derivative,
    log_derivative_batch,
)

HARTREE_HZ = 6.579683920502e15


# --- 1-3: log-derivative propagator, single-radius matching, two sectors ---

def _exact_free_y(k, a, b):
    return k / np.tan(k * (b - a))


@pytest.mark.parametrize("N", [1, 2])
def test_log_derivative_is_fourth_order(N):
    k, a, b = 3.0, 1.0, 2.3
    errors = []
    for m in (50, 100, 200):
        h = (b - a) / (2 * m)
        W = np.broadcast_to(-k * k * np.eye(N), (2 * m + 1, N, N)).copy()
        errors.append(abs(log_derivative(W, h)[0, 0] - _exact_free_y(k, a, b)))
    assert errors[0] / errors[1] > 14 and errors[1] / errors[2] > 14


def test_log_derivative_needs_even_number_of_intervals():
    with pytest.raises(ValueError, match="even number of intervals"):
        log_derivative(np.zeros((4, 1, 1)), 0.1)


def test_log_derivative_sectors_can_be_chained():
    k, a, b = 3.0, 1.0, 2.3
    h = (b - a) / 200
    W = np.full((201, 1, 1), -k * k)
    whole = log_derivative(W, h)
    first = log_derivative(W[:101], h)
    chained = log_derivative(W[100:], h, first)
    assert np.allclose(whole, chained, rtol=1e-12, atol=0)


def test_log_derivative_batch_matches_single():
    rng = np.random.default_rng(1)
    n, N = 201, 3
    r = np.linspace(2.0, 10.0, n)
    A = rng.normal(size=(N, N))
    W0 = (A + A.T)[None] * np.exp(-r)[:, None, None] + np.diag([5.0, 6.0, 7.0])[None] / r[:, None, None] ** 2
    D = np.array([[-1.0, 0.5, 2.0], [-2.0, -0.1, 1.0]])
    h = r[1] - r[0]
    batch = log_derivative_batch(W0, D, h)
    for b in range(2):
        W = W0.copy()
        W[:, np.arange(N), np.arange(N)] += D[b]
        assert np.allclose(batch[b], log_derivative(W, h), rtol=1e-12, atol=1e-12)


def _lj_problem():
    mu = cs.units.reduced_mass(40.0, 40.0)
    pot = cs.LennardJones.from_depth(depth=1e-3, r_min=8.0)
    return cs.single_channel(mu, pot)


def test_default_and_legacy_propagators_agree():
    prob = _lj_problem()
    grid = cs.Grid(5.5, 60.0, 0.002)
    e = cs.units.from_kelvin(1e-3)
    a_ld = prob.solve(e, grid).scattering_length()
    a_nu = prob.solve(e, grid, propagator="numerov").scattering_length()
    assert abs(a_ld - a_nu) < 1e-4 * abs(a_ld)


def test_unknown_propagator_rejected():
    with pytest.raises(ValueError, match="propagator"):
        _lj_problem().solve(1e-9, cs.Grid(5.5, 60.0, 0.01), propagator="rk4")


def test_two_sector_grid_needs_log_derivative():
    grid = cs.Grid(5.5, 200.0, 0.002, r_mid=40.0, step_long=0.05)
    with pytest.raises(ValueError, match="log-derivative"):
        _lj_problem().solve(1e-9, grid, propagator="numerov")


def test_two_sector_grid_matches_uniform_and_ends_at_r_max():
    prob = _lj_problem()
    e = cs.units.from_kelvin(1e-6)
    uniform = cs.Grid(5.5, 400.0, 0.002)
    split = cs.Grid(5.5, 400.0, 0.002, r_mid=40.0, step_long=0.05)
    assert split.r[-1] == pytest.approx(400.0)
    assert split.n_points < uniform.n_points / 5
    a_u = prob.solve(e, uniform).scattering_length()
    a_s = prob.solve(e, split).scattering_length()
    assert abs(a_u - a_s) < 1e-6 * abs(a_u)


def test_grid_rejects_half_specified_second_sector():
    with pytest.raises(ValueError, match="both r_mid and step_long"):
        cs.Grid(5.0, 100.0, 0.01, r_mid=20.0)
    with pytest.raises(ValueError, match="r_min < r_mid < r_max"):
        cs.Grid(5.0, 100.0, 0.01, r_mid=200.0, step_long=0.1)


# --- closed-channel long-range reference used 2 mu kappa^2 instead of kappa^2 ---

@pytest.mark.parametrize("mu", [1.0, 1000.0, 50000.0])
def test_zero_tail_closed_reference_equals_free(mu):
    zero = cs.LongRange({6: 0.0}, r_far=400.0)
    kappa, r1, r2 = 0.05, 100.0, 100.5
    assert _long_closed_ratio(zero, mu, 0, kappa, r1, r2) == pytest.approx(
        _free_closed_ratio(0, kappa, r1, r2), rel=1e-8)
    assert _long_closed_logderiv(zero, mu, 1, kappa, r1) == pytest.approx(
        _free_closed_logderiv(1, kappa, r1), rel=1e-8)


# --- 4-5: boundary warnings ---

def test_non_decaying_coupling_warns():
    mu = cs.units.reduced_mass(40.0, 40.0)
    lj = cs.LennardJones.from_depth(1e-3, 8.0)
    prob = cs.CoupledChannels(mu, [0.0, 1e-6], [[lj, 1e-7], [None, lj]])
    with pytest.warns(UserWarning, match=r"coupling \[0\]\[1\] does not decay"):
        prob.solve(cs.units.from_kelvin(1e-6), cs.Grid(5.5, 100.0, 0.01))


def test_decaying_potentials_do_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _lj_problem().solve(cs.units.from_kelvin(1e-6), cs.Grid(5.5, 100.0, 0.01))


def test_scan_warns_when_r_min_is_classically_allowed():
    model = cs.two_channel_feshbach(
        mu=cs.units.reduced_mass(40.0, 40.0), c6=3000.0, c12_open=3.0e8, c12_closed=2.8e8,
        coupling=2e-5, coupling_r0=9.0, coupling_sigma=1.0, delta_mu=1e-9)
    with pytest.warns(UserWarning, match="classically allowed at r_min"):
        model.scan([0.0, 1.0], cs.units.from_kelvin(1e-6), cs.Grid(9.0, 100.0, 0.01))


# --- 6: atomic structure ---

@pytest.mark.parametrize("atom,splitting_hz", [
    ("6Li", 228.2052610e6), ("23Na", 1771.6261288e6), ("85Rb", 3035.7324390e6),
    ("87Rb", 6834.682610904e6), ("133Cs", 9192.631770e6), ("40K", 1285.7886e6),
])
def test_alkali_presets_reproduce_hyperfine_splitting(atom, splitting_hz):
    e = np.linalg.eigvalsh(alkali_hamiltonian(atom, 0.0)) * HARTREE_HZ
    assert e.max() - e.min() == pytest.approx(splitting_hz, rel=1e-8)


def test_alkali_preset_unknown_atom():
    with pytest.raises(ValueError, match="unknown atom"):
        alkali_hamiltonian("7Xx")


def test_quadrupole_term_requires_i_and_j_at_least_one():
    with pytest.raises(ValueError, match="I >= 1 and J >= 1"):
        hyperfine_hamiltonian(I=1.5, J=0.5, A_hf=1e-7, B_hf=1e-8)


def test_quadrupole_term_is_traceless_and_hermitian():
    H0 = hyperfine_hamiltonian(I=1.5, J=1.5, A_hf=0.0)
    HQ = hyperfine_hamiltonian(I=1.5, J=1.5, A_hf=0.0, B_hf=1e-8)
    Q = HQ - H0
    assert np.allclose(Q, Q.conj().T)
    assert abs(np.trace(Q)) < 1e-20


def test_atomic_module_is_exported():
    assert cs.atomic.ALKALI_GROUND_STATES is ALKALI_GROUND_STATES


# --- 7: identical particles ---

def test_identical_particle_cross_sections():
    prob = _lj_problem()
    res = prob.solve(cs.units.from_kelvin(1e-6), cs.Grid(5.5, 100.0, 0.01))
    dist = res.cross_sections()
    bos = res.cross_sections(identical="bosons")
    fer = res.cross_sections(identical="fermions")
    assert bos["elastic"] == pytest.approx(2 * dist["elastic"])
    assert fer["elastic"] == 0.0
    a = res.scattering_length().real
    assert bos["elastic"] == pytest.approx(8 * np.pi * a**2, rel=1e-3)
    with pytest.raises(ValueError):
        res.cross_sections(identical="anyons")


# --- 8: version ---

def test_version_matches_pyproject():
    root = Path(__file__).resolve().parents[1]
    text = (root / "pyproject.toml").read_text()
    version = re.search(r'^version\s*=\s*"([^"]+)"', text, re.M).group(1)  # no tomllib on 3.10
    assert cs.__version__ == version


# --- 10: resonance fit ---

def test_fit_feshbach_with_zero_background_guess_and_validation():
    B = np.linspace(-5.0, 5.0, 81) + 0.0625
    a = cs.feshbach_a(B, a_bg=100.0, B0=0.3, delta=0.5)
    a_bg, B0, delta = cs.fit_feshbach(B, a)
    assert (a_bg, B0, delta) == pytest.approx((100.0, 0.3, 0.5), rel=1e-6)
    with pytest.raises(ValueError):
        cs.fit_feshbach([1.0, 2.0], [1.0, 2.0])
