"""Independent validation of the single-channel d-wave threshold law."""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import spherical_jn, spherical_yn

import coldscatter as cs
from coldscatter import units as u
from helpers import MU


def _independent_d_wave_k(potential, energy, mu, r_match=40.0):
    """Compute tan(delta_2) using an independent adaptive ODE integration."""
    k = np.sqrt(2.0 * mu * energy)
    r0 = 1e-4

    def rhs(r, y):
        q = 6.0 / r**2 + 2.0 * mu * (potential(r) - energy)
        return [y[1], q * y[0]]

    solution = solve_ivp(
        rhs,
        (r0, r_match),
        [1.0, 3.0 / r0],
        method="DOP853",
        rtol=1e-11,
        atol=1e-12,
        max_step=0.05,
    )
    assert solution.success, solution.message

    value, derivative = solution.y[:, -1]
    log_derivative = derivative / value
    x = k * r_match

    f = x * spherical_jn(2, x)
    g = -x * spherical_yn(2, x)
    fp = k * (
        spherical_jn(2, x)
        + x * spherical_jn(2, x, derivative=True)
    )
    gp = -k * (
        spherical_yn(2, x)
        + x * spherical_yn(2, x, derivative=True)
    )

    return (log_derivative * f - fp) / (gp - log_derivative * g)


def test_d_wave_threshold_law_and_independent_ode():
    """Check K_2 ~ E^(5/2) and agreement with an independent ODE solution."""
    potential = cs.Gaussian(-1e-6, 10.0, 2.0)
    problem = cs.single_channel(MU, potential, ell=2)
    energies = (1e-6, 4e-6)

    grid = cs.Grid(0.5, 40.0, 0.025)
    numerical = []
    reference = []

    for energy in energies:
        result = problem.solve(u.from_kelvin(energy), grid)
        numerical.append(result.K[0, 0])
        reference.append(
            _independent_d_wave_k(
                potential, u.from_kelvin(energy), MU
            )
        )

    numerical = np.asarray(numerical)
    reference = np.asarray(reference)

    assert np.all(np.isfinite(numerical))
    assert np.all(numerical < 0.0)
    assert np.allclose(numerical, reference, rtol=3e-5, atol=0.0)

    ratio = numerical[1] / numerical[0]
    assert abs(ratio / 32.0 - 1.0) < 2e-3


def test_d_wave_threshold_ratio_is_step_stable():
    """Check that the threshold ratio is stable under Numerov step refinement."""
    potential = cs.Gaussian(-1e-6, 10.0, 2.0)
    problem = cs.single_channel(MU, potential, ell=2)
    energies = (1e-6, 4e-6)

    ratios = []
    for h in (0.05, 0.025, 0.0125):
        grid = cs.Grid(0.5, 40.0, h)
        values = [
            problem.solve(u.from_kelvin(energy), grid).K[0, 0]
            for energy in energies
        ]
        ratios.append(values[1] / values[0])

    assert np.all(np.isfinite(ratios))
    assert max(ratios) - min(ratios) < 2e-5
    assert abs(ratios[-1] / 32.0 - 1.0) < 2e-3
