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
    order = np.argsort(B)
    B, a = B[order], a[order]
    if p0 is None:
        i = int(np.argmax(np.abs(np.diff(a))))
        B0 = 0.5 * (B[i] + B[i + 1])
        far = np.abs(B - B0) > 0.5 * np.ptp(B) * 0.5
        a_bg = float(np.median(a[far])) if np.any(far) else float(np.median(a))
        j = int(np.argmin(np.abs(np.abs(B - B0) - 0.25 * np.ptp(B))))
        delta = (1.0 - a[j] / a_bg) * (B[j] - B0)
        p0 = (a_bg, B0, delta)

    def resid(p):
        a_bg, B0, d = p
        model = a_bg * (1.0 - d / np.where(B == B0, 1e-12, B - B0))
        return (model - a) / (np.abs(a) + abs(a_bg))

    sol = least_squares(resid, p0, loss="soft_l1", f_scale=0.1)
    return tuple(float(x) for x in sol.x)
