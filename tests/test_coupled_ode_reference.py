"""Independent DOP853 reference for two-channel Numerov propagation."""

import numpy as np
from scipy.integrate import solve_ivp

from coldscatter.asymptotics import match_k_matrix
from coldscatter.potentials import Gaussian
from coldscatter.propagators import renormalized_numerov
from coldscatter.solver import CoupledChannels, Grid


def ode_reference_matrix(problem, energy, grid, wall):
    """Return the boundary-to-boundary matrix from an independent ODE solve."""
    r = grid.r
    n_channels = problem.n
    identity = np.eye(n_channels)

    def rhs(x, y):
        psi = y[:n_channels**2].reshape(n_channels, n_channels)
        dpsi = y[n_channels**2:].reshape(n_channels, n_channels)

        W = problem.w_matrix(np.array([x]), energy)[0]

        return np.concatenate((dpsi.ravel(), (W @ psi).ravel()))

    # Fundamental matrix: Psi(wall) = 0, Psi'(wall) = I.
    y0 = np.concatenate((
        np.zeros((n_channels, n_channels)).ravel(),
        identity.ravel(),
    ))

    sol = solve_ivp(
        rhs,
        (wall, r[-1]),
        y0,
        method="DOP853",
        t_eval=[r[-2], r[-1]],
        rtol=1e-12,
        atol=1e-13,
    )

    if not sol.success:
        raise RuntimeError(f"DOP853 reference failed: {sol.message}")

    psi_previous = sol.y[:n_channels**2, 0].reshape(n_channels, n_channels)
    psi_last = sol.y[:n_channels**2, 1].reshape(n_channels, n_channels)

    # M satisfies Psi(r[-2]) = M @ Psi(r[-1]).
    return np.linalg.solve(psi_last.T, psi_previous.T).T


def test_two_open_channel_numerov_against_dop853():
    mu = 1.0
    energy = 0.08
    wall = 0.5
    r_max = 30.0

    problem = CoupledChannels(
        mu=mu,
        thresholds=[0.0, 0.0],
        potentials=[
            [
                Gaussian(0.80, 0.5, 0.35),
                Gaussian(0.08, 4.5, 1.0),
            ],
            [
                None,
                Gaussian(0.65, 0.5, 0.35),
            ],
        ],
        ell=0,
    )

    k2 = np.full(2, 2.0 * mu * energy)
    ell = np.zeros(2, dtype=int)

    errors = []

    for h in (0.1, 0.05, 0.025):
        grid = Grid(wall + h, r_max, h)
        W = problem.w_matrix(grid.r, energy)

        M_numerov = renormalized_numerov(W, h)
        M_ode = ode_reference_matrix(problem, energy, grid, wall)

        K_numerov, *_ = match_k_matrix(
            M_numerov, grid.r[-2], grid.r[-1], k2, ell
        )
        K_ode, *_ = match_k_matrix(
            M_ode, grid.r[-2], grid.r[-1], k2, ell
        )

        errors.append(np.linalg.norm(K_numerov - K_ode))

    assert np.all(np.isfinite(errors))
    assert errors[0] > errors[1] > errors[2], errors
    assert errors[2] < 1e-5, errors
    assert errors[0] / errors[1] > 10.0, errors
    assert errors[1] / errors[2] > 10.0, errors


def test_coupled_propagation_matrix_converges_against_dop853():
    """Check convergence of the coupled Numerov propagation matrix."""
    mu = 1.0
    energy = 0.08
    wall = 0.5
    r_max = 30.0

    problem = CoupledChannels(
        mu=mu,
        thresholds=[0.0, 0.0],
        potentials=[
            [
                Gaussian(0.80, 0.5, 0.35),
                Gaussian(0.08, 4.5, 1.0),
            ],
            [
                None,
                Gaussian(0.65, 0.5, 0.35),
            ],
        ],
        ell=0,
    )

    errors = []

    for h in (0.1, 0.05, 0.025):
        grid = Grid(wall + h, r_max, h)
        W = problem.w_matrix(grid.r, energy)

        M_numerov = renormalized_numerov(W, h)
        M_ode = ode_reference_matrix(problem, energy, grid, wall)

        errors.append(np.linalg.norm(M_numerov - M_ode))

    assert np.all(np.isfinite(errors))
    assert errors[0] / errors[1] > 20.0, errors
    assert errors[1] / errors[2] > 20.0, errors
