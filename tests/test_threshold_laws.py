"""Numerical regression tests for coupled s-wave threshold laws."""

import numpy as np

from coldscatter.asymptotics import match_k_matrix
from coldscatter.potentials import Gaussian
from coldscatter.propagators import renormalized_numerov
from coldscatter.solver import CoupledChannels, Grid


def test_coupled_s_wave_threshold_scaling():
    """Check Wigner threshold scaling on the open side of a coupled threshold."""
    problem = CoupledChannels(
        mu=1.0,
        thresholds=[0.0, 0.20],
        potentials=[
            [Gaussian(0.30, 1.5, 0.45), Gaussian(0.12, 3.0, 0.7)],
            [None, Gaussian(0.25, 1.5, 0.45)],
        ],
        ell=0,
    )

    h = 0.0125
    grid = Grid(0.5 + h, 24.0, h)

    def calculate(delta):
        energy = 0.20 + delta
        k2 = 2.0 * (energy - problem.thresholds(0.0))
        matrix = renormalized_numerov(
            problem.w_matrix(grid.r, energy), h
        )
        return match_k_matrix(
            matrix, grid.r[-2], grid.r[-1], k2, problem.ell
        )

    K_threshold, opened_threshold, _, _ = calculate(0.0)
    assert np.array_equal(opened_threshold, [0])

    scaled_k12 = []
    scaled_k22 = []

    for delta in [1e-10, 1e-8, 1e-6]:
        K, opened, _, asymmetry = calculate(delta)

        assert np.array_equal(opened, [0, 1])
        assert np.all(np.isfinite(K))
        assert asymmetry < 1e-8
        assert np.allclose(K, K.T, rtol=0.0, atol=1e-8)

        assert abs(K[0, 0] - K_threshold[0, 0]) < 1e-5

        scaled_k12.append(K[0, 1] / delta**0.25)
        scaled_k22.append(K[1, 1] / delta**0.5)

    assert np.allclose(scaled_k12, scaled_k12[0], rtol=2e-5, atol=0.0)
    assert np.allclose(scaled_k22, scaled_k22[0], rtol=2e-5, atol=0.0)
