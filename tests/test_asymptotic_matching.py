"""Independent analytical tests for asymptotic K-matrix matching."""

import numpy as np

from coldscatter.asymptotics import match_k_matrix


def test_free_s_wave_hard_wall_matches_exact_k_matrix():
    """For a free s-wave outside a hard wall, K = -tan(k * R_wall)."""
    k = 0.37
    r_wall = 0.83
    r1, r2 = 12.3, 12.4

    u1 = np.sin(k * (r1 - r_wall))
    u2 = np.sin(k * (r2 - r_wall))
    M = np.array([[u1 / u2]])

    K, open_idx, k_open, asymmetry = match_k_matrix(
        M=M, r1=r1, r2=r2, k2=np.array([k**2]), ell=0
    )

    exact_K = -np.tan(k * r_wall)

    assert np.array_equal(open_idx, np.array([0]))
    assert np.allclose(k_open, [k], rtol=1e-14, atol=0.0)
    assert np.isclose(K[0, 0], exact_K, rtol=1e-12, atol=1e-12)
    assert asymmetry < 1e-14


def test_uncoupled_closed_channel_does_not_change_open_channel_k():
    """An uncoupled closed channel must not affect open-channel matching."""
    k = 0.37
    kappa = 0.61
    r_wall = 0.83
    r1, r2 = 12.3, 12.4

    u1 = np.sin(k * (r1 - r_wall))
    u2 = np.sin(k * (r2 - r_wall))

    uc1 = np.sinh(kappa * (r1 - r_wall))
    uc2 = np.sinh(kappa * (r2 - r_wall))

    M = np.diag([u1 / u2, uc1 / uc2])

    K, open_idx, k_open, asymmetry = match_k_matrix(
        M=M,
        r1=r1,
        r2=r2,
        k2=np.array([k**2, -kappa**2]),
        ell=0,
    )

    exact_K = -np.tan(k * r_wall)

    assert np.array_equal(open_idx, np.array([0]))
    assert np.allclose(k_open, [k], rtol=1e-14, atol=0.0)
    assert K.shape == (1, 1)
    assert np.isclose(K[0, 0], exact_K, rtol=1e-12, atol=1e-12)
    assert asymmetry < 1e-14


def test_zero_energy_closed_channel_matches_analytic_limit():
    """A threshold channel must use the non-growing zero-energy solution."""
    k = 0.37
    r_wall = 0.83
    r1, r2 = 12.3, 12.4

    u1 = np.sin(k * (r1 - r_wall))
    u2 = np.sin(k * (r2 - r_wall))
    exact_K = -np.tan(k * r_wall)

    for ell in range(4):
        regular_ratio = (r1 / r2) ** (ell + 1)
        M = np.diag([u1 / u2, regular_ratio])

        K, open_idx, k_open, asymmetry = match_k_matrix(
            M=M,
            r1=r1,
            r2=r2,
            k2=np.array([k**2, 0.0]),
            ell=[0, ell],
        )

        assert np.array_equal(open_idx, [0])
        assert np.allclose(k_open, [k], rtol=1e-14, atol=0.0)
        assert K.shape == (1, 1)
        assert np.isclose(K[0, 0], exact_K, rtol=1e-12, atol=1e-12)
        assert asymmetry < 1e-14
