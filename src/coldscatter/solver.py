"""High-level API: define coupled channels, solve, scan, extract observables."""
from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from .asymptotics import match_k_matrix, s_from_k, scattering_length_from_s
from .potentials import as_potential
from .long_range import LongRange
from .propagators import renormalized_numerov, renormalized_numerov_batch


@dataclass(frozen=True)
class Grid:
    """Uniform radial grid in bohr.

    ``r_min`` must lie deep inside the repulsive wall (hard-wall boundary
    condition) and ``r_max`` far outside the range of the interaction.
    """

    r_min: float
    r_max: float
    step: float

    def __post_init__(self):
        if not all(np.isfinite(v) for v in (self.r_min, self.r_max, self.step)):
            raise ValueError("r_min, r_max and step must be finite numbers")
        if self.r_min <= 0:
            raise ValueError("r_min must be > 0 (bohr)")
        if self.step <= 0 or self.r_max <= self.r_min:
            raise ValueError("need step > 0 and r_max > r_min")
        if self.n_points < 4:
            raise ValueError("grid too coarse (fewer than 4 points)")
        n_exact = (self.r_max - self.r_min) / self.step
        if abs(n_exact - round(n_exact)) > 1e-6:
            warnings.warn("(r_max - r_min) is not a multiple of step: the grid ends at "
                          f"r = {self.r[-1]:.6g}, not at r_max.", UserWarning, stacklevel=3)

    @property
    def n_points(self) -> int:
        return round((self.r_max - self.r_min) / self.step) + 1

    @property
    def r(self) -> np.ndarray:
        return self.r_min + self.step * np.arange(self.n_points)


@dataclass
class ScatteringResult:
    """Outcome of one coupled-channel calculation at fixed (E, B)."""

    mu: float
    energy: float            # total energy (hartree)
    field: float             # gauss
    entrance: int            # global index of the incident channel
    k2: np.ndarray           # 2 mu (E - threshold_i), all channels
    ell: np.ndarray          # partial wave per channel
    open_channels: np.ndarray
    k_open: np.ndarray
    K: np.ndarray            # K matrix, open channels only
    S: np.ndarray            # S matrix, open channels only
    k_asymmetry: float       # diagnostic: relative asymmetry of raw K (~ numerical error)

    def _o(self, channel: int) -> int:
        hit = np.flatnonzero(self.open_channels == channel)
        if len(hit) == 0:
            raise ValueError(f"channel {channel} is closed at this energy")
        return int(hit[0])

    @property
    def k_entrance(self) -> float:
        return float(self.k_open[self._o(self.entrance)])

    def s(self, i: int, j: int) -> complex:
        """S-matrix element between *global* channel indices."""
        return complex(self.S[self._o(i), self._o(j)])

    def scattering_length(self) -> complex:
        r"""Complex s-wave scattering length ``a = alpha - i beta`` from
        ``S_ii = exp(-2 i k a)`` (so ``beta >= 0`` signals inelastic loss).

        Only meaningful for an s-wave entrance channel, at energies low enough
        that ``|k a| << 1``.
        """
        if self.ell[self.entrance] != 0:
            raise ValueError("scattering length is defined for s-wave entrance channels")
        k = self.k_entrance
        return scattering_length_from_s(self.s(self.entrance, self.entrance), k)

    def cross_sections(self) -> dict:
        r"""Partial-wave cross sections in bohr^2 (distinguishable particles), with
        ``pref = (2l+1) pi / k^2`` of the entrance channel:

        * ``elastic``    = ``pref |1 - S_ii|^2``
        * ``inelastic``  = ``pref sum_{j != i, open} |S_ji|^2``  (state-changing collisions)
        * ``loss``       = ``pref (1 - sum_j |S_ji|^2)``  (flux leaving the modelled channels;
          zero for a unitary S matrix, non-zero only with absorbing/complex models)
        * ``total``      = ``elastic + inelastic + loss = 2 pref (1 - Re S_ii)`` (optical theorem)
        """
        i = self._o(self.entrance)
        k = self.k_open[i]
        pref = (2 * int(self.ell[self.entrance]) + 1) * np.pi / k**2
        row = self.S[:, i]
        to = {int(c): float(pref * abs(row[o]) ** 2)
              for o, c in enumerate(self.open_channels) if c != self.entrance}
        elastic = float(pref * abs(1.0 - self.S[i, i]) ** 2)
        inelastic = float(sum(to.values()))
        loss = float(pref * (1.0 - np.sum(np.abs(row) ** 2)))
        return {"elastic": elastic, "inelastic": inelastic, "loss": loss,
                "total": elastic + inelastic + loss, "to_channel": to}


class CoupledChannels:
    r"""Radial coupled-channel problem

    .. math:: \psi_i'' = \sum_j \Big[\tfrac{\ell_i(\ell_i+1)}{R^2}\delta_{ij}
              + 2\mu\big(V_{ij}(R) + \delta_{ij}\,E_i(B) - \delta_{ij} E\big)\Big]\psi_j

    Parameters
    ----------
    mu : reduced mass (atomic units, see :func:`coldscatter.units.reduced_mass`).
    thresholds : sequence of floats, *or* a callable ``B_gauss -> sequence`` (hartree).
    potentials : ``N x N`` nested list; entries are :class:`Potential`, callables,
        numbers or ``None``.  Only the upper triangle (``i <= j``) is read; the
        matrix is symmetrised automatically.
    ell : partial wave per channel (int or sequence); default 0.
    long_range : LongRange, sequence of LongRange/None, or None.
        Optional diagonal asymptotic reference tail for matching. A single
        LongRange object is broadcast to all channels. A sequence must have
        one entry per channel. None retains free-reference matching.
        The reference tail is not added to the physical potential.
    """

    def __init__(self, mu: float, thresholds, potentials: Sequence[Sequence], ell=0,
                 names: Sequence[str] | None = None, long_range=None):
        self.mu = float(mu)
        if not (np.isfinite(self.mu) and self.mu > 0):
            raise ValueError("mu must be a finite number > 0 (atomic units)")
        self.n = len(potentials)
        if self.n < 1:
            raise ValueError("need at least one channel")
        for row in potentials:
            if len(row) != self.n:
                raise ValueError("potentials must be a square N x N nested list")
        for i in range(self.n):
            for j in range(i):
                if potentials[i][j] is not None:
                    raise ValueError(
                        f"potentials[{i}][{j}] is in the lower triangle. "
                        "Only the upper triangle is accepted; move this coupling "
                        f"to potentials[{j}][{i}].")
        self._pots = [[None if potentials[i][j] is None or j < i else as_potential(potentials[i][j])
                       for j in range(self.n)] for i in range(self.n)]
        if callable(thresholds):
            self._thr: Callable = thresholds
        else:
            t = np.asarray(thresholds, dtype=float)
            self._thr = lambda B, _t=t: _t
        raw_ell = np.asarray(ell, dtype=object)
        if any(isinstance(value, (bool, np.bool_)) for value in raw_ell.flat):
            raise ValueError("ell must contain non-negative integers, not booleans")
        ell_arr = np.asarray(ell, dtype=float)
        if (not np.all(np.isfinite(ell_arr)) or np.any(ell_arr < 0)
                or np.any(ell_arr != np.round(ell_arr))):
            raise ValueError("ell must be non-negative integers")
        try:
            self.ell = np.broadcast_to(ell_arr.astype(int), (self.n,)).copy()
        except ValueError:
            raise ValueError("ell must be a scalar or have one entry per channel") from None
        if long_range is None:
            self.long_range = (None,) * self.n
        elif isinstance(long_range, LongRange):
            self.long_range = (long_range,) * self.n
        else:
            self.long_range = tuple(long_range)
            if len(self.long_range) != self.n:
                raise ValueError("long_range must have one entry per channel")
            if any(t is not None and not isinstance(t, LongRange)
                   for t in self.long_range):
                raise TypeError("long_range entries must be LongRange or None")
        self.names = list(names) if names is not None else [f"ch{i}" for i in range(self.n)]
        if len(self.names) != self.n:
            raise ValueError("names must have one entry per channel")
        self.thresholds(0.0)  # validates shape and finiteness

    def thresholds(self, B: float = 0.0) -> np.ndarray:
        thr = np.asarray(self._thr(B), dtype=float)
        if thr.shape != (self.n,) or not np.all(np.isfinite(thr)):
            raise ValueError("thresholds must be a finite array with one entry per channel")
        return thr

    def potential_matrix(self, r: np.ndarray) -> np.ndarray:
        V = np.zeros((len(r), self.n, self.n))
        for i in range(self.n):
            for j in range(i, self.n):
                p = self._pots[i][j]
                if p is not None:
                    v = p(r)
                    V[:, i, j] = v
                    V[:, j, i] = v
        return V

    def w_matrix(self, r: np.ndarray, energy: float, B: float = 0.0) -> np.ndarray:
        thr = self.thresholds(B)
        W = 2.0 * self.mu * self.potential_matrix(r)
        idx = np.arange(self.n)
        W[:, idx, idx] += 2.0 * self.mu * (thr - energy) + self.ell * (self.ell + 1) / r[:, None] ** 2
        return W

    def _result(self, M, E, thr, B, entrance, r) -> ScatteringResult:
        k2 = 2.0 * self.mu * (E - thr)
        K, open_idx, k_open, asym = match_k_matrix(
            M, r[-2], r[-1], k2, self.ell, mu=self.mu,
            long_range=self.long_range
        )
        return ScatteringResult(self.mu, float(E), float(B), int(entrance), k2, self.ell.copy(),
                                open_idx, k_open, K, s_from_k(K), asym)

    def _check_run(self, collision_energy, entrance):
        if not (np.isfinite(collision_energy) and collision_energy > 0):
            raise ValueError("collision_energy must be finite and > 0 (hartree)")
        ok = isinstance(entrance, (int, np.integer)) and not isinstance(entrance, bool)
        if not (ok and 0 <= entrance < self.n):
            raise ValueError(f"entrance must be an integer in [0, {self.n - 1}]")

    def solve(self, collision_energy: float, grid: Grid, B: float = 0.0,
              entrance: int = 0) -> ScatteringResult:
        """Solve at fixed field ``B`` (gauss).

        ``collision_energy`` (hartree) is measured from the threshold of the
        ``entrance`` channel.
        """
        self._check_run(collision_energy, entrance)
        if not np.isfinite(B):
            raise ValueError("B must be finite (gauss)")
        thr = self.thresholds(B)
        E = thr[entrance] + collision_energy
        r = grid.r
        W = self.w_matrix(r, E, B)
        if np.min(np.linalg.eigvalsh(W[0])) <= 0:
            warnings.warn("a channel is classically allowed at r_min: the hard-wall boundary "
                          "condition affects the result; decrease r_min.", stacklevel=2)
        M = renormalized_numerov(W, grid.step)
        return self._result(M, E, thr, B, entrance, r)

    def scan(self, fields, collision_energy: float, grid: Grid,
             entrance: int = 0) -> list[ScatteringResult]:
        """Solve for many fields (gauss) in one batched sweep over the grid."""
        self._check_run(collision_energy, entrance)
        fields = np.atleast_1d(np.asarray(fields, dtype=float))
        if fields.ndim != 1 or fields.size == 0 or not np.all(np.isfinite(fields)):
            raise ValueError("fields must be a non-empty 1-D array of finite values (gauss)")
        r = grid.r
        thr = np.array([self.thresholds(B) for B in fields])            # (nB, N)
        E = thr[:, entrance] + collision_energy                         # (nB,)
        W0 = 2.0 * self.mu * self.potential_matrix(r)
        idx = np.arange(self.n)
        W0[:, idx, idx] += self.ell * (self.ell + 1) / r[:, None] ** 2
        D = 2.0 * self.mu * (thr - E[:, None])
        M = renormalized_numerov_batch(W0, D, grid.step)
        return [self._result(M[b], E[b], thr[b], fields[b], entrance, r)
                for b in range(len(fields))]

    def scan_field(self, fields, collision_energy: float, grid: Grid, entrance: int = 0):
        """Complex s-wave scattering length ``a(B)`` in bohr for each field (gauss)."""
        return np.array([res.scattering_length() for res in
                         self.scan(fields, collision_energy, grid, entrance)])
