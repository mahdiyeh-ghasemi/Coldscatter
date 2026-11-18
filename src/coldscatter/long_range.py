"""Diagonal long-range reference potentials used for asymptotic matching."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Real
from types import MappingProxyType

import numpy as np


@dataclass(frozen=True)
class LongRange:
    r"""Diagonal reference tail

    .. math:: V_{\\mathrm{ref}}(R) = -\\sum_n C_n/R^n.

    Coefficients are in atomic units: ``C_n`` has units
    hartree * bohr**n. This object defines the asymptotic reference
    used during matching. It is NOT added to the physical potential.

    The coefficient mapping is copied and made read-only at construction.
    """
    coefficients: Mapping[int, float]
    r_far: float = 5000.0
    rtol: float = 2e-11
    atol: float = 2e-13
    _terms: tuple = field(init=False, repr=False)

    def __post_init__(self):
        if not isinstance(self.coefficients, Mapping):
            raise TypeError(
                "coefficients must be a mapping such as {6: C6, 8: C8}"
            )

        terms = []
        for power, coefficient in self.coefficients.items():
            if isinstance(power, bool) or not isinstance(power, (int, np.integer)):
                raise TypeError("long-range powers must be integers")
            if power < 3:
                raise ValueError("long-range powers must be >= 3")
            if isinstance(coefficient, bool) or not isinstance(coefficient, Real):
                raise TypeError("long-range coefficients must be real numbers")
            value = float(coefficient)
            if not np.isfinite(value):
                raise ValueError("long-range coefficients must be finite")
            terms.append((int(power), value))

        for name, value, condition in (
            ("r_far", self.r_far, lambda x: x > 0),
            ("rtol", self.rtol, lambda x: 0 < x < 1),
            ("atol", self.atol, lambda x: x > 0),
        ):
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a real number")
            value = float(value)
            if not np.isfinite(value) or not condition(value):
                if name == "r_far":
                    raise ValueError("r_far must be finite and > 0 (bohr)")
                if name == "rtol":
                    raise ValueError("rtol must be finite and between 0 and 1")
                raise ValueError("atol must be finite and > 0")
            object.__setattr__(self, name, value)

        immutable_coefficients = MappingProxyType(
            {power: coefficient for power, coefficient in sorted(terms)}
        )
        object.__setattr__(self, "coefficients", immutable_coefficients)
        object.__setattr__(self, "_terms", tuple(sorted(terms)))

    @property
    def terms(self):
        """Immutable tuple of ``(power, coefficient)`` pairs."""
        return self._terms

    def potential(self, r):
        """Evaluate the reference potential (hartree) at radii in bohr."""
        r = np.asarray(r, dtype=float)
        if np.any(r <= 0) or not np.all(np.isfinite(r)):
            raise ValueError("r must contain finite positive radii")

        value = np.zeros_like(r, dtype=float)
        with np.errstate(over="ignore", under="ignore", divide="ignore",
                         invalid="ignore"):
            for power, coefficient in self._terms:
                value -= coefficient / np.power(r, power)

        if not np.all(np.isfinite(value)):
            raise FloatingPointError(
                "long-range potential is outside the floating-point range "
                "for the supplied radius"
            )
        return value
