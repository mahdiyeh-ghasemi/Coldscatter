"""Matrix renormalized-Numerov propagator (Johnson 1977).

We integrate the radial coupled equations

    psi''(R) = W(R) psi(R)

outwards on a *uniform* grid.  With ``phi_n = (1 - h^2 W_n / 12) psi_n`` the
Numerov recursion becomes ``phi_{n+1} = T_n phi_n - phi_{n-1}`` with
``T_n = 2 + h^2 W_n (1 - h^2 W_n / 12)^{-1}``.  Propagating the *ratio*
``G_n = phi_{n-1} phi_n^{-1}`` (instead of ``phi`` itself)

    G_{n+1} = (T_n - G_n)^{-1}

stores only bounded ratios, so closed (exponentially growing) channels cannot
overflow.  This is the same idea behind the log-derivative propagators used
in MOLSCAT.

Reference: B. R. Johnson, J. Chem. Phys. 67, 4086 (1977).
"""
from __future__ import annotations

import numpy as np


def _prepare_W(W, h):
    """Validate and convert the coupling array (shared by both propagators)."""
    if np.iscomplexobj(W):
        raise ValueError("complex (absorbing) coupling matrices are not supported")
    W = np.asarray(W, dtype=float)
    if W.ndim != 3 or W.shape[1] != W.shape[2]:
        raise ValueError("W must have shape (n, N, N)")
    if W.shape[0] < 3:
        raise ValueError("need at least 3 grid points")
    if not (np.isfinite(h) and h > 0):
        raise ValueError("step h must be finite and > 0")
    if not np.all(np.isfinite(W)):
        raise ValueError("W contains NaN or inf")
    return W


def renormalized_numerov(W: np.ndarray, h: float) -> np.ndarray:
    """Propagate from the first to the last grid point.

    Parameters
    ----------
    W : (n, N, N) array
        Symmetric coupling matrix ``W(R_i)`` on a uniform grid (n >= 3).
    h : float
        Grid step.

    Returns
    -------
    M : (N, N) array
        Matrix such that ``psi_{n-2} = M @ psi_{n-1}`` for *every* regular
        solution (the last two grid points are used for asymptotic matching).

    Notes
    -----
    The boundary condition is ``psi = 0`` one step before the first grid point,
    i.e. a hard wall; place ``R_min`` deep inside the classically forbidden
    region so that the result is independent of it.
    """
    W = _prepare_W(W, h)
    n, N, _ = W.shape
    h2 = h * h

    if N == 1:  # fast scalar path
        w = W[:, 0, 0]
        a = 1.0 - h2 * w / 12.0
        t = 2.0 + h2 * w / a
        g = 0.0
        for k in range(n - 1):
            g = 1.0 / (t[k] - g)
        return np.array([[g * a[n - 1] / a[n - 2]]])

    eye = np.eye(N)
    A = eye - (h2 / 12.0) * W
    T = 2.0 * eye + h2 * np.linalg.solve(A, W)
    G = np.zeros((N, N))
    for k in range(n - 1):
        G = np.linalg.solve(T[k] - G, eye)
    return np.linalg.solve(A[n - 2], G @ A[n - 1])


def renormalized_numerov_batch(W0: np.ndarray, D: np.ndarray, h: float) -> np.ndarray:
    """Propagate many fields at once.

    ``W(R, b) = W0(R) + diag(D[b])`` - the field only shifts the diagonal
    (thresholds), so one pass over the grid serves every field value and the
    per-step linear algebra is batched with NumPy.

    Parameters
    ----------
    W0 : (n, N, N)   field-independent part of W
    D  : (nB, N)     diagonal offsets, one row per field
    Returns
    -------
    M : (nB, N, N)   same meaning as in :func:`renormalized_numerov`.
    """
    W0 = _prepare_W(W0, h)
    if np.iscomplexobj(D):
        raise ValueError("complex field offsets are not supported")
    D = np.asarray(D, dtype=float)
    n, N, _ = W0.shape
    if (D.ndim != 2 or D.shape[0] < 1 or D.shape[1] != N
            or not np.all(np.isfinite(D))):
        raise ValueError("D must be a non-empty, finite array of shape (nB, N)")
    nB = D.shape[0]
    h2 = h * h
    eye = np.eye(N)
    idx = np.arange(N)

    def mats(k):
        W = np.broadcast_to(W0[k], (nB, N, N)).copy()
        W[:, idx, idx] += D
        A = eye - (h2 / 12.0) * W
        return A, 2.0 * eye + h2 * np.linalg.solve(A, W)

    G = np.zeros((nB, N, N))
    for k in range(n - 1):
        G = np.linalg.solve(
            mats(k)[1] - G,
            np.broadcast_to(eye, (nB, N, N))
        )

    A2, _ = mats(n - 2)
    A1, _ = mats(n - 1)
    return np.linalg.solve(A2, G @ A1)
