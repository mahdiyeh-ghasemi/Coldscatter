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