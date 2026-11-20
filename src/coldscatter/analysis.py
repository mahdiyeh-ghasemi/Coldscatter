"""Post-processing: Feshbach-resonance line-shape model and fit."""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares


def feshbach_a(B, a_bg: float, B0: float, delta: float):
    r"""Lossless resonance: ``a(B) = a_bg (1 - Delta / (B - B0))`` (Chin et al., RMP 82, 1225)."""
    B = np.asarray(B, dtype=float)
    return a_bg * (1.0 - delta / (B - B0))


def fit_feshbach(B, a, p0=None):
    """Fit :func:`feshbach_a` to real scattering lengths ``a(B)`` around ONE resonance.

    Pass only the window around the resonance (a few widths on each side): the
    model assumes a constant background, which is not true over a wide range.
    The initial guess is taken from the pole (largest jump in ``a``) and the
    residuals are weighted so that points next to the pole do not dominate.

    Returns ``(a_bg, B0, delta)``.
    """
    B, a = np.asarray(B, float), np.asarray(a, float)
    if B.shape != a.shape or B.ndim != 1 or B.size < 4:
        raise ValueError("B and a must be 1-D arrays of equal length (at least 4 points)")
    if not (np.all(np.isfinite(B)) and np.all(np.isfinite(a))):
        raise ValueError("B and a must be finite")
    order = np.argsort(B)
    B, a = B[order], a[order]
    span = float(np.ptp(B))
    if span <= 0:
        raise ValueError("B must contain at least two different fields")
    if p0 is None:
        # Pole: the largest jump between neighbours *with a sign change* (a
        # lossless resonance flips sign there); fall back to the largest jump.
        jumps = np.abs(np.diff(a))
        flips = np.sign(a[:-1]) != np.sign(a[1:])
        i = int(np.argmax(np.where(flips, jumps, -np.inf))) if np.any(flips) else int(np.argmax(jumps))
        B0 = 0.5 * (B[i] + B[i + 1])
        far = np.abs(B - B0) > 0.25 * span
        a_bg = float(np.median(a[far])) if np.any(far) else float(np.median(a))
        if abs(a_bg) < 1e-12 * max(1.0, float(np.max(np.abs(a)))):
            # a_bg ~ 0 would make the width guess divide by zero
            a_bg = float(np.median(np.abs(a))) or 1.0
        j = int(np.argmin(np.abs(np.abs(B - B0) - 0.25 * span)))
        delta = (1.0 - a[j] / a_bg) * (B[j] - B0)
        if not np.isfinite(delta) or delta == 0.0:
            delta = 0.01 * span
        p0 = (a_bg, B0, delta)

    eps = 1e-12 * span

    def resid(p):
        a_bg, B0, d = p
        x = B - B0
        x = np.where(np.abs(x) < eps, np.where(x < 0, -eps, eps), x)
        model = a_bg * (1.0 - d / x)
        return (model - a) / (np.abs(a) + abs(a_bg) + 1e-300)

    sol = least_squares(resid, p0, loss="soft_l1", f_scale=0.1)
    return tuple(float(x) for x in sol.x)
