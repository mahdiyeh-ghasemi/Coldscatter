import numpy as np

from coldscatter.propagators import (
    renormalized_numerov,
    renormalized_numerov_batch,
)


def test_batch_propagator_matches_individual_propagations():
    h = 0.05
    n_points = 300
    n_channels = 2

    r = np.arange(n_points) * h

    W0 = np.zeros((n_points, n_channels, n_channels))
    W0[:, 0, 0] = 0.02 + 0.0001 * r**2
    W0[:, 1, 1] = 0.04 + 0.0002 * r**2
    W0[:, 0, 1] = 0.005 * np.exp(-r / 3.0)
    W0[:, 1, 0] = W0[:, 0, 1]

    D = np.array([
        [-0.01, 0.02],
        [0.00, 0.03],
        [0.02, -0.01],
    ])

    batch_results = renormalized_numerov_batch(W0, D, h)

    for i in range(len(D)):
        W = W0.copy()
        W[:, np.arange(n_channels), np.arange(n_channels)] += D[i]

        single_result = renormalized_numerov(W, h)

        assert np.allclose(
            batch_results[i],
            single_result,
            rtol=1e-11,
            atol=1e-12,
        )

def test_free_f_wave_propagation_matches_exact_solution():
    """Check Numerov propagation against the exact free f-wave solution."""
    from scipy.special import spherical_jn, spherical_yn

    ell = 3
    k = 0.3
    r_min = 1.0
    r_max = 20.0
    errors = []

    for h in (0.1, 0.05, 0.025):
        n_points = round((r_max - r_min) / h) + 1
        r = r_min + h * np.arange(n_points)

        W = (ell * (ell + 1) / r**2 - k**2)[:, None, None]
        M_numerov = renormalized_numerov(W, h)[0, 0]

        r_wall = r_min - h
        x_wall = k * r_wall

        def f(x):
            return x * spherical_jn(ell, x)

        def g(x):
            return x * spherical_yn(ell, x)

        def exact_solution(x):
            return f(x) * g(x_wall) - g(x) * f(x_wall)

        M_exact = exact_solution(k * r[-2]) / exact_solution(k * r[-1])
        errors.append(abs(M_numerov - M_exact))

    assert errors[0] / errors[1] > 25.0
    assert errors[1] / errors[2] > 25.0
    assert errors[-1] < 1e-12


def test_matrix_propagator_matches_independent_scalar_channels():
    """Check uncoupled matrix propagation against scalar propagation."""
    h = 0.0125
    n_points = 200
    r = np.arange(n_points) * h

    W = np.zeros((n_points, 2, 2))
    W[:, 0, 0] = 0.02 + 0.0001 * r**2
    W[:, 1, 1] = 0.04 + 0.0002 * r**2

    M_matrix = renormalized_numerov(W, h)

    for channel in range(2):
        W_scalar = W[:, channel, channel][:, None, None]
        M_scalar = renormalized_numerov(W_scalar, h)

        assert np.allclose(
            M_matrix[channel, channel],
            M_scalar[0, 0],
            rtol=1e-13,
            atol=1e-14,
        )

    assert abs(M_matrix[0, 1]) < 1e-14
    assert abs(M_matrix[1, 0]) < 1e-14
