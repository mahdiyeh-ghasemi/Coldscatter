"""Independent derivative-matching reference for one open and one closed channel."""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import spherical_jn, spherical_yn

from coldscatter.asymptotics import match_k_matrix
from coldscatter.potentials import Gaussian, LennardJones
from coldscatter.propagators import renormalized_numerov
from coldscatter.solver import CoupledChannels, Grid


def _derivative_match_k(psi, dpsi, r, k2, ell):
    """Extract K from Psi and Psi' using an independent derivative match."""
    open_idx = np.flatnonzero(k2 > 0)
    closed_idx = np.flatnonzero(k2 < 0)
    if len(open_idx) != 1 or len(closed_idx) != 1:
        raise ValueError("reference expects exactly one open and one closed channel")

    k = np.sqrt(k2[open_idx[0]])
    kappa = np.sqrt(-k2[closed_idx[0]])
    l_open = int(ell[open_idx[0]])
    l_closed = int(ell[closed_idx[0]])

    if l_closed != 0:
        raise NotImplementedError("reference currently supports s-wave closed channels")

    x = k * r
    scale = 1.0 / np.sqrt(k)

    F = scale * x * spherical_jn(l_open, x)
    G = -scale * x * spherical_yn(l_open, x)

    Fp = scale * k * (
        spherical_jn(l_open, x)
        + x * spherical_jn(l_open, x, derivative=True)
    )
    Gp = -scale * k * (
        spherical_yn(l_open, x)
        + x * spherical_yn(l_open, x, derivative=True)
    )

    # Physical decaying solution for an s-wave closed channel.
    D = np.exp(-kappa * r)
    Dp = -kappa * D

    f = np.array([F, 0.0])
    g = np.array([G, 0.0])
    d = np.array([0.0, D])

    fp = np.array([Fp, 0.0])
    gp = np.array([Gp, 0.0])
    dp = np.array([0.0, Dp])

    # Unknowns: two numerical solution coefficients, K, and closed amplitude.
    matrix = np.block([
        [psi, -g[:, None], -d[:, None]],
        [dpsi, -gp[:, None], -dp[:, None]],
    ])
    rhs = np.concatenate((f, fp))

    solution = np.linalg.solve(matrix, rhs)
    return float(solution[-2])


def _ode_fundamental(problem, energy, wall, r_match):
    """Integrate the coupled fundamental matrix and its derivative."""
    n = problem.n
    eye = np.eye(n)
    y0 = np.concatenate((np.zeros((n, n)).ravel(), eye.ravel()))

    def rhs(r, y):
        psi = y[: n * n].reshape(n, n)
        dpsi = y[n * n :].reshape(n, n)
        W = problem.w_matrix(np.array([r]), energy)[0]
        return np.concatenate((dpsi.ravel(), (W @ psi).ravel()))

    sol = solve_ivp(
        rhs,
        (wall, r_match),
        y0,
        method="DOP853",
        t_eval=[r_match],
        rtol=2e-12,
        atol=2e-13,
    )
    if not sol.success:
        raise RuntimeError(sol.message)

    psi = sol.y[: n * n, -1].reshape(n, n)
    dpsi = sol.y[n * n :, -1].reshape(n, n)
    return psi, dpsi



def test_open_closed_coupled_matching_converges_against_derivative_reference():
    """Check fourth-order convergence of open-channel K for an open-closed pair."""
    mu = 1.0
    energy = 0.08
    wall = 0.5
    r_max = 24.0

    problem = CoupledChannels(
        mu=mu,
        thresholds=[0.0, 0.20],
        potentials=[
            [Gaussian(0.30, 1.5, 0.45), Gaussian(0.12, 3.0, 0.7)],
            [None, Gaussian(0.25, 1.5, 0.45)],
        ],
        ell=0,
    )

    k2 = 2.0 * mu * (energy - problem.thresholds(0.0))

    # Independent reference, computed once.
    psi, dpsi = _ode_fundamental(problem, energy, wall, r_max)
    k_reference = _derivative_match_k(
        psi, dpsi, r_max, k2, problem.ell
    )

    errors = []

    for h in (0.1, 0.05, 0.025):
        grid = Grid(wall + h, r_max, h)
        W = problem.w_matrix(grid.r, energy)
        M = renormalized_numerov(W, h)

        K_matrix, open_idx, _, asymmetry = match_k_matrix(
            M, grid.r[-2], grid.r[-1], k2, problem.ell
        )

        assert np.array_equal(open_idx, [0])
        assert np.isfinite(K_matrix[0, 0])
        assert asymmetry < 1e-8

        errors.append(abs(K_matrix[0, 0] - k_reference))

    assert np.all(np.isfinite(errors))
    assert errors[0] > errors[1] > errors[2]

    ratios = [
        errors[i] / errors[i + 1]
        for i in range(len(errors) - 1)
    ]

    assert all(ratio > 12.0 for ratio in ratios)
    assert errors[-1] < 2e-9
def test_open_channel_s_matrix_is_unitary_with_closed_channel():
    """A closed channel must not cause loss of open-channel flux."""
    lj = LennardJones.from_depth(depth=0.02, r_min=2.0)

    problem = CoupledChannels(
        mu=1.0,
        thresholds=[0.0, 0.20],
        potentials=[
            [lj, Gaussian(0.001, 3.0, 0.7)],
            [None, lj],
        ],
        ell=0,
    )

    grid = Grid(0.6, 24.0, 0.025)
    result = problem.solve(0.08, grid)

    assert np.array_equal(result.open_channels, [0])
    assert result.S.shape == (1, 1)

    unitarity_error = np.max(
        np.abs(result.S @ result.S.conj().T - np.eye(1))
    )
    assert unitarity_error < 1e-10

    assert np.isclose(abs(result.s(0, 0)), 1.0, atol=1e-10)

    cross_sections = result.cross_sections()
    assert abs(cross_sections["loss"]) < 1e-9 * cross_sections["total"]
    assert cross_sections["inelastic"] == 0.0