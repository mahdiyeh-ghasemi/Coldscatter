"""Asymptotic matching: from the propagated solution to K and S matrices.

At large R every channel ``i`` with ``k_i^2 = 2 mu (E - E_i) > 0`` is open and
behaves like a free (Riccati-Bessel) wave; closed channels must decay.  We write

    psi(R) = F a + G b + D d ,

with ``F_i = k_i^{-1/2} (k_i R) j_l(k_i R)``, ``G_i = -k_i^{-1/2} (k_i R) y_l(k_i R)``
for open channels and ``D_i`` the decaying modified Bessel solution for closed
channels.  Demanding ``psi(R1) = M psi(R2)`` at the last two grid points fixes
``b`` and ``d`` for ``a = 1``; then ``K = b`` and

    S = (1 + iK)(1 - iK)^{-1}.

For a single channel ``K = tan(delta)``, ``S = exp(2 i delta)``.
The ``k^{-1/2}`` normalisation makes K symmetric and S unitary.
"""
from __future__ import annotations

import numpy as np
from scipy.special import kve, spherical_jn, spherical_yn


def _ric_f(ell: int, x):
    return x * spherical_jn(ell, x)


def _ric_g(ell: int, x):
    return -x * spherical_yn(ell, x)


def _decaying_scaled(ell: int, x):
    """``sqrt(pi x / 2) K_{l+1/2}(x) * exp(x)`` (scaled to avoid underflow)."""
    return np.sqrt(np.pi * x / 2.0) * kve(ell + 0.5, x)


def match_k_matrix(M: np.ndarray, r1: float, r2: float, k2: np.ndarray, ell):
    """Return ``(K, open_idx, k_open, asymmetry)``.

    Parameters
    ----------
    M : (N, N)
        ``psi(r1) = M psi(r2)`` with ``r1 < r2`` (from the propagator).
    k2 : (N,)
        ``2 mu (E - threshold_i)`` for every channel.
    ell : (N,) ints
        Partial wave of each channel.
    """
    N = len(k2)
    ell = np.broadcast_to(np.asarray(ell, dtype=int), (N,))
    open_idx = np.flatnonzero(k2 > 0)
    closed_idx = np.flatnonzero(k2 <= 0)
    n_open, n_closed = len(open_idx), len(closed_idx)
    if n_open == 0:
        raise ValueError("no open channel at this energy")

    k_open = np.sqrt(k2[open_idx])
    Fo1 = np.zeros((N, n_open))
    Fo2 = np.zeros((N, n_open))
    Go1 = np.zeros((N, n_open))
    Go2 = np.zeros((N, n_open))
    for c, i in enumerate(open_idx):
        k, l = k_open[c], int(ell[i])
        s = 1.0 / np.sqrt(k)
        Fo1[i, c], Fo2[i, c] = s * _ric_f(l, k * r1), s * _ric_f(l, k * r2)
        Go1[i, c], Go2[i, c] = s * _ric_g(l, k * r1), s * _ric_g(l, k * r2)

    Dc1 = np.zeros((N, n_closed))
    Dc2 = np.zeros((N, n_closed))
    for c, i in enumerate(closed_idx):
        kappa, l = np.sqrt(-k2[i]), int(ell[i])
        Dc2[i, c] = 1.0
        if kappa == 0.0:
            Dc1[i, c] = (r2 / r1) ** l
        else:
            x1, x2 = kappa * r1, kappa * r2
            d1 = _decaying_scaled(l, x1)
            d2 = _decaying_scaled(l, x2)
            Dc1[i, c] = (d1 / d2) * np.exp(x2 - x1)

    lhs = np.hstack([M @ Go2 - Go1, M @ Dc2 - Dc1])
    rhs = Fo1 - M @ Fo2
    sol = np.linalg.solve(lhs, rhs)
    K = sol[:n_open, :]
    asym = float(np.max(np.abs(K - K.T)) / max(np.max(np.abs(K)), 1e-300))
    return 0.5 * (K + K.T), open_idx, k_open, asym


def s_from_k(K: np.ndarray) -> np.ndarray:
    eye = np.eye(K.shape[0])
    return (eye + 1j * K) @ np.linalg.inv(eye - 1j * K)


def scattering_length_from_s(s_ii, k: float) -> complex:
    """Energy-dependent s-wave scattering length, free of branch cuts.

    a(k) = (1 - S) / (i k (1 + S)).  For one open channel S = exp(2 i delta)
    this equals -tan(delta)/k and diverges correctly at a resonance
    (delta = pi/2), unlike i log(S)/(2k), whose principal branch saturates at
    |a| = pi/(2k).  For k a -> 0 it reduces to a = alpha - i beta.
    """
    s = np.complex128(s_ii)
    return complex(np.divide(1.0 - s, 1j * k * (1.0 + s)))
