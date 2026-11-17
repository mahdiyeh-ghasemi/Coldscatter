"""Radial potentials (hartree) as functions of the separation R (bohr)."""
from __future__ import annotations

from collections.abc import Callable
from math import gamma, pi

import numpy as np


def _require_finite(name, value):
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _require_positive(name, value):
    value = _require_finite(name, value)
    if value <= 0:
        raise ValueError(f"{name} must be > 0")
    return value


class Potential:
    """Base class. Subclasses implement ``__call__(r) -> ndarray`` (hartree)."""

    def __call__(self, r):  # pragma: no cover - interface
        raise NotImplementedError

    def __add__(self, other):
        return _Sum(self, as_potential(other))

    __radd__ = __add__

    def __mul__(self, scale: float):
        return _Scaled(self, float(scale))

    __rmul__ = __mul__


class _Sum(Potential):
    def __init__(self, a, b):
        self.a, self.b = a, b

    def __call__(self, r):
        return self.a(r) + self.b(r)


class _Scaled(Potential):
    def __init__(self, p, s):
        self.p, self.s = p, s

    def __call__(self, r):
        return self.s * self.p(r)


class Constant(Potential):
    def __init__(self, value: float):
        self.value = _require_finite("value", value)

    def __call__(self, r):
        return np.full_like(np.asarray(r, dtype=float), self.value)


class FromFunction(Potential):
    """Wrap any vectorised callable ``f(r)`` (e.g. an ab initio spline)."""

    def __init__(self, func: Callable):
        if not callable(func):
            raise TypeError("func must be callable")
        self.func = func

    def __call__(self, r):
        return np.asarray(self.func(np.asarray(r, dtype=float)), dtype=float)


class LennardJones(Potential):
    r"""``V(R) = C12 / R**12 - C6 / R**6``."""

    def __init__(self, c6: float, c12: float):
        self.c6 = _require_finite("c6", c6)
        self.c12 = _require_finite("c12", c12)

    @classmethod
    def from_depth(cls, depth: float, r_min: float) -> LennardJones:
        """Build from well depth ``D`` and equilibrium distance ``r_min``."""
        depth = _require_positive("depth", depth)
        r_min = _require_positive("r_min", r_min)
        return cls(c6=2.0 * depth * r_min**6, c12=depth * r_min**12)

    def __call__(self, r):
        r = np.asarray(r, dtype=float)
        if np.any(~np.isfinite(r)) or np.any(r <= 0):
            raise ValueError("r must contain finite, positive separations")
        x = 1.0 / r**6
        return self.c12 * x * x - self.c6 * x


class Gaussian(Potential):
    r"""``V(R) = amplitude * exp(-(R - r0)**2 / (2 sigma**2))`` - a handy short-range coupling."""

    def __init__(self, amplitude: float, r0: float, sigma: float):
        self.amplitude = _require_finite("amplitude", amplitude)
        self.r0 = _require_finite("r0", r0)
        self.sigma = _require_positive("sigma", sigma)

    def __call__(self, r):
        r = np.asarray(r, dtype=float)
        return self.amplitude * np.exp(-0.5 * ((r - self.r0) / self.sigma) ** 2)


class Exponential(Potential):
    r"""``V(R) = amplitude * exp(-alpha R)``."""

    def __init__(self, amplitude: float, alpha: float):
        self.amplitude = _require_finite("amplitude", amplitude)
        self.alpha = _require_finite("alpha", alpha)

    def __call__(self, r):
        return self.amplitude * np.exp(-self.alpha * np.asarray(r, dtype=float))


def as_potential(obj) -> Potential:
    if isinstance(obj, Potential):
        return obj
    if callable(obj):
        return FromFunction(obj)
    return Constant(float(obj))


# --- van der Waals universality (Gribakin & Flambaum 1993; Chin et al. 2010) ---

def vdw_length(mu: float, c6: float) -> float:
    r"""``R_vdW = (1/2) (2 mu C6)**(1/4)`` in bohr (atomic units, hbar = 1)."""
    mu = _require_positive("mu", mu)
    c6 = _require_positive("c6", c6)
    return 0.5 * (2.0 * mu * c6) ** 0.25


def mean_scattering_length(mu: float, c6: float) -> float:
    r"""Gribakin-Flambaum mean scattering length ``a_bar = 4 pi / Gamma(1/4)^2 * R_vdW``."""
    return 4.0 * pi / gamma(0.25) ** 2 * vdw_length(mu, c6)
