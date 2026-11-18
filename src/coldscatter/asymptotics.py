"""Asymptotic matching to free or explicitly specified long-range references."""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import kve, spherical_jn, spherical_yn

from .long_range import LongRange


def _ric_f(ell: int, x):
    return x * spherical_jn(ell, x)


def _ric_g(ell: int, x):
    return -x * spherical_yn(ell, x)


def _decaying_scaled(ell: int, x):
    """Scaled decaying Riccati-modified spherical Bessel function."""
    return np.sqrt(np.pi * x / 2.0) * kve(ell + 0.5, x)


def _free_open_values(ell, k, r):
    x = k * r
    f = _ric_f(ell, x) / np.sqrt(k)
    g = _ric_g(ell, x) / np.sqrt(k)
    fp = np.sqrt(k) * (
        spherical_jn(ell, x) + x * spherical_jn(ell, x, derivative=True)
    )
    gp = -np.sqrt(k) * (
        spherical_yn(ell, x) + x * spherical_yn(ell, x, derivative=True)
    )
    return f, fp, g, gp


def _long_open_values(tail, mu, ell, k, r1, r2):
    """Integrate a pair of open reference solutions inward from r_far."""
    rf = float(tail.r_far)

    if not (
        np.isfinite(r1) and np.isfinite(r2)
        and 0.0 < r1 < r2 < rf
    ):
        raise ValueError(
            "open-channel matching radii must satisfy "
            "0 < r1 < r2 < LongRange.r_far"
        )
    if not (np.isfinite(mu) and mu > 0.0):
        raise ValueError("mu must be finite and > 0")
    if not (np.isfinite(k) and k > 0.0):
        raise ValueError("open-channel wave number k must be finite and > 0")
    if isinstance(ell, bool) or not isinstance(ell, (int, np.integer)) or ell < 0:
        raise ValueError("ell must be a non-negative integer")

    y0 = np.asarray(_free_open_values(ell, k, rf), dtype=float)

    def rhs(r, y):
        v = tail.potential(r)
        q = ell * (ell + 1) / r**2 + 2.0 * mu * (v - k**2 / (2.0 * mu))
        return (y[1], q * y[0], y[3], q * y[2])

    sol = solve_ivp(
        rhs, (rf, r1), y0, method="DOP853",
        t_eval=(r2, r1), rtol=tail.rtol, atol=tail.atol,
    )
    if not sol.success:
        raise RuntimeError(
            f"open long-range reference integration failed: {sol.message}"
        )
    if sol.y.shape != (4, 2) or not np.all(np.isfinite(sol.y)):
        raise RuntimeError(
            "open long-range reference integration returned invalid values"
        )

    at2, at1 = sol.y[:, 0], sol.y[:, 1]
    return at1[0], at1[1], at1[2], at1[3], at2[0], at2[1], at2[2], at2[3]


def _free_closed_ratio(ell, kappa, r1, r2):
    if kappa == 0.0:
        return (r2 / r1) ** ell
    x1, x2 = kappa * r1, kappa * r2
    d1 = _decaying_scaled(ell, x1)
    d2 = _decaying_scaled(ell, x2)
    return (d1 / d2) * np.exp(x2 - x1)


def _long_closed_ratio(tail, mu, ell, kappa, r1, r2):
    """Return the signed ratio D(r1)/D(r2).

    The closed-channel reference wavefunction is propagated inward as
    a two-component linear system. Piecewise rescaling controls the
    dynamic range without introducing logarithmic-derivative poles
    at nodes of D(r).

    The accumulated rescaling factors are retained so that the returned
    ratio has the correct magnitude and sign.
    """
    rf = float(tail.r_far)
    r1, r2 = float(r1), float(r2)

    if isinstance(ell, (bool, np.bool_)) or not isinstance(ell, (int, np.integer)) or ell < 0:
        raise ValueError("ell must be a non-negative integer")
    if not (np.isfinite(r1) and np.isfinite(r2) and 0.0 < r1 < r2 < rf):
        raise ValueError(
            "closed-channel matching radii must satisfy "
            "0 < r1 < r2 < LongRange.r_far"
        )
    if not (np.isfinite(mu) and mu > 0.0):
        raise ValueError("mu must be finite and > 0")
    if not (np.isfinite(kappa) and kappa >= 0.0):
        raise ValueError("kappa must be finite and >= 0")

    # Asymptotic logarithmic derivative of the decaying free solution.
    if kappa == 0.0:
        y0 = -float(ell) / rf
    else:
        x = kappa * rf
        nu = ell + 0.5
        km = kve(abs(nu - 1.0), x)
        kp = kve(nu + 1.0, x)
        kc = kve(nu, x)
        y0 = kappa * (0.5 / x - 0.5 * (km + kp) / kc)

    if not np.isfinite(y0):
        raise RuntimeError("invalid asymptotic closed-channel boundary condition")

    def rhs(r, state):
        d, dp = state
        q = (
            ell * (ell + 1) / r**2
            + 2.0 * mu * (kappa**2 + tail.potential(r))
        )
        return (dp, q * d)

    # Geometric segmentation limits the dynamic range accumulated
    # during any one integration segment.
    n_segments = max(
        1,
        int(np.ceil(np.log(rf / r1) / np.log(1.25))),
    )
    radii = np.concatenate((
        np.geomspace(rf, r1, n_segments + 1),
        np.array([r2]),
    ))
    radii = np.array(sorted(set(radii.tolist()), reverse=True))

    state = np.array([1.0, y0], dtype=float)
    d_at_r2 = None
    log_scale_after_r2 = 0.0

    for r_start, r_stop in zip(radii[:-1], radii[1:]):
        sol = solve_ivp(
            rhs,
            (float(r_start), float(r_stop)),
            state,
            method="DOP853",
            rtol=tail.rtol,
            atol=tail.atol,
        )
        if not sol.success:
            raise RuntimeError(
                "closed long-range wavefunction integration failed: "
                f"{sol.message}"
            )

        state = sol.y[:, -1]
        scale = float(np.max(np.abs(state)))

        if not np.isfinite(scale) or scale == 0.0:
            raise RuntimeError(
                "closed long-range wavefunction lost numerical scale"
            )

        state /= scale

        if r_stop == r2:
            d_at_r2 = float(state[0])
            log_scale_after_r2 = 0.0
        elif d_at_r2 is not None:
            log_scale_after_r2 += float(np.log(scale))

    if d_at_r2 is None:
        raise RuntimeError("internal error: matching radius r2 was not reached")

    d_at_r1 = float(state[0])

    if d_at_r2 == 0.0:
        raise RuntimeError(
            "closed-channel reference has a node at r2; "
            "D(r1)/D(r2) is singular"
        )

    if d_at_r1 == 0.0:
        return 0.0

    log_abs_ratio = (
        np.log(abs(d_at_r1))
        - np.log(abs(d_at_r2))
        + log_scale_after_r2
    )

    max_log = np.log(np.finfo(float).max)
    min_log = np.log(np.nextafter(0.0, 1.0))

    if not np.isfinite(log_abs_ratio) or log_abs_ratio > max_log:
        raise RuntimeError(
            "closed long-range reference ratio exceeds floating-point range"
        )
    if log_abs_ratio < min_log:
        return 0.0

    sign = np.sign(d_at_r1) * np.sign(d_at_r2)
    return float(sign * np.exp(log_abs_ratio))


def match_k_matrix(M, r1, r2, k2, ell, mu=1.0, long_range=None):
    """Return ``(K, open_idx, k_open, asymmetry)``.

    ``long_range`` contains one optional :class:`LongRange` reference per
    channel. ``None`` preserves the original free-reference matching.
    """
    N = len(k2)
    raw_ell = np.asarray(ell, dtype=object)
    if any(isinstance(x, (bool, np.bool_)) for x in raw_ell.flat):
        raise ValueError("ell must contain non-negative integers, not booleans")
    try:
        ell_values = np.asarray(ell, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("ell must contain non-negative integers") from None
    if (not np.all(np.isfinite(ell_values)) or np.any(ell_values < 0)
            or np.any(ell_values != np.round(ell_values))):
        raise ValueError("ell must contain non-negative integers")
    try:
        ell = np.broadcast_to(ell_values.astype(int), (N,))
    except ValueError:
        raise ValueError("ell must be a scalar or have one entry per channel") from None
    if long_range is None:
        tails = (None,) * N
    elif isinstance(long_range, LongRange):
        tails = (long_range,) * N
    else:
        tails = tuple(long_range)
        if len(tails) != N:
            raise ValueError("long_range must have one entry per channel")
        if any(t is not None and not isinstance(t, LongRange) for t in tails):
            raise TypeError("each long_range entry must be LongRange or None")

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
        k, l, tail = k_open[c], int(ell[i]), tails[i]
        if tail is None:
            f1, _, g1, _ = _free_open_values(l, k, r1)
            f2, _, g2, _ = _free_open_values(l, k, r2)
        else:
            f1, _, g1, _, f2, _, g2, _ = _long_open_values(
                tail, mu, l, k, r1, r2
            )
        Fo1[i, c], Fo2[i, c] = f1, f2
        Go1[i, c], Go2[i, c] = g1, g2

    Dc1 = np.zeros((N, n_closed))
    Dc2 = np.zeros((N, n_closed))
    for c, i in enumerate(closed_idx):
        kappa, l, tail = np.sqrt(-k2[i]), int(ell[i]), tails[i]
        Dc2[i, c] = 1.0
        if tail is None:
            Dc1[i, c] = _free_closed_ratio(l, kappa, r1, r2)
        else:
            Dc1[i, c] = _long_closed_ratio(tail, mu, l, kappa, r1, r2)

    lhs = np.hstack([M @ Go2 - Go1, M @ Dc2 - Dc1])
    rhs = Fo1 - M @ Fo2
    try:
        sol = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError(
            "asymptotic matching matrix is singular; "
            "check the grid, the energy and the channel setup"
        ) from exc
    K = sol[:n_open, :]
    asym = float(np.max(np.abs(K - K.T)) / max(np.max(np.abs(K)), 1e-300))
    return 0.5 * (K + K.T), open_idx, k_open, asym


def s_from_k(K):
    eye = np.eye(K.shape[0])
    return (eye + 1j * K) @ np.linalg.inv(eye - 1j * K)


def scattering_length_from_s(s_ii, k: float) -> complex:
    """Energy-dependent s-wave scattering length, free of branch cuts."""
    s = np.complex128(s_ii)
    return complex(np.divide(1.0 - s, 1j * k * (1.0 + s)))
