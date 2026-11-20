"""High-level API: define coupled channels, solve, scan, extract observables."""
from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from .asymptotics import (match_k_matrix, match_k_matrix_logderiv, s_from_k,
                          scattering_length_from_s)
from .potentials import as_potential
from .long_range import LongRange
from .propagators import (log_derivative, log_derivative_batch, renormalized_numerov,
                          renormalized_numerov_batch)

PROPAGATORS = ("log-derivative", "numerov")


def _check_propagator(name):
    if name not in PROPAGATORS:
        raise ValueError(f"propagator must be one of {PROPAGATORS}, got {name!r}")


@dataclass(frozen=True)
class Grid:
    """Radial grid in bohr: uniform, or two uniform sectors.

    ``r_min`` must lie deep inside the repulsive wall (hard-wall boundary
    condition) and ``r_max`` far outside the range of the interaction.

    Optional ``r_mid`` and ``step_long`` give a second, coarser sector from
    ``r_mid`` to ``r_max`` (log-derivative propagator only).  Use a step that
    resolves the short-range well up to ``r_mid`` and a larger one for the
    slowly varying long range - the usual MOLSCAT RMID / DRL strategy.
    """

    r_min: float
    r_max: float
    step: float
    r_mid: float | None = None
    step_long: float | None = None

    def __post_init__(self):
        if not all(np.isfinite(v) for v in (self.r_min, self.r_max, self.step)):
            raise ValueError("r_min, r_max and step must be finite numbers")
        if self.r_min <= 0:
            raise ValueError("r_min must be > 0 (bohr)")
        if self.step <= 0 or self.r_max <= self.r_min:
            raise ValueError("need step > 0 and r_max > r_min")
        if (self.r_mid is None) != (self.step_long is None):
            raise ValueError("give both r_mid and step_long, or neither")
        if self.r_mid is not None:
            if not (np.isfinite(self.r_mid) and np.isfinite(self.step_long)):
                raise ValueError("r_mid and step_long must be finite numbers")
            if not (self.r_min < self.r_mid < self.r_max) or self.step_long <= 0:
                raise ValueError("need r_min < r_mid < r_max and step_long > 0")
        if self.n_points < 4:
            raise ValueError("grid too coarse (fewer than 4 points)")
        for a, b, h, name in self._segments():
            n_exact = (b - a) / h
            if abs(n_exact - round(n_exact)) > 1e-6:
                warnings.warn(f"{name} is not a multiple of step: the grid ends at "
                              f"r = {self.r[-1]:.6g}, not at r_max.", UserWarning, stacklevel=3)

    def _segments(self):
        if self.r_mid is None:
            return [(self.r_min, self.r_max, self.step, "(r_max - r_min)")]
        return [(self.r_min, self.r_mid, self.step, "(r_mid - r_min)"),
                (self.r_mid, self.r_max, self.step_long, "(r_max - r_mid)")]

    @property
    def sectors(self) -> list[tuple[float, float, int]]:
        """``(r_start, step, n_intervals)`` for each uniform sector."""
        out, start = [], self.r_min
        for a, b, h, _ in self._segments():
            m = max(1, round((b - a) / h))
            out.append((start, h, m))
            start = start + m * h
        return out

    @property
    def n_points(self) -> int:
        return sum(m for _, _, m in self.sectors) + 1

    @property
    def r(self) -> np.ndarray:
        parts = [r0 + h * np.arange(m + (i == len(self.sectors) - 1))
                 for i, (r0, h, m) in enumerate(self.sectors)]
        return np.concatenate(parts)


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

    def cross_sections(self, identical: str | None = None) -> dict:
        r"""Partial-wave cross sections in bohr^2, with
        ``pref = g (2l+1) pi / k^2`` of the entrance channel:

        * ``elastic``    = ``pref |1 - S_ii|^2``
        * ``inelastic``  = ``pref sum_{j != i, open} |S_ji|^2``  (state-changing collisions)
        * ``loss``       = ``pref (1 - sum_j |S_ji|^2)``  (flux leaving the modelled channels;
          zero for a unitary S matrix, non-zero only with absorbing/complex models)
        * ``total``      = ``elastic + inelastic + loss = 2 pref (1 - Re S_ii)`` (optical theorem)

        ``identical``: ``None`` (distinguishable, ``g = 1``), ``"bosons"`` or
        ``"fermions"`` for two identical particles in the *same* internal state
        in the entrance channel.  Then ``g = 2`` for partial waves allowed by
        exchange symmetry (even ``l`` for bosons, odd ``l`` for fermions) and
        ``g = 0`` for forbidden ones.
        """
        if identical not in (None, "bosons", "fermions"):
            raise ValueError("identical must be None, 'bosons' or 'fermions'")
        i = self._o(self.entrance)
        k = self.k_open[i]
        l_in = int(self.ell[self.entrance])
        if identical is None:
            g = 1.0
        else:
            allowed = (l_in % 2 == 0) if identical == "bosons" else (l_in % 2 == 1)
            g = 2.0 if allowed else 0.0
        pref = g * (2 * l_in + 1) * np.pi / k**2
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

    def _result(self, prop, E, thr, B, entrance, r, propagator) -> ScatteringResult:
        """``prop`` is Y at r[-1] (log-derivative) or M (Numerov, legacy)."""
        k2 = 2.0 * self.mu * (E - thr)
        if propagator == "log-derivative":
            K, open_idx, k_open, asym = match_k_matrix_logderiv(
                prop, r[-1], k2, self.ell, mu=self.mu, long_range=self.long_range)
        else:
            K, open_idx, k_open, asym = match_k_matrix(
                prop, r[-2], r[-1], k2, self.ell, mu=self.mu, long_range=self.long_range)
        return ScatteringResult(self.mu, float(E), float(B), int(entrance), k2, self.ell.copy(),
                                open_idx, k_open, K, s_from_k(K), asym)

    @staticmethod
    def _sectors(grid: Grid, propagator: str):
        """List of ``(points, step)`` at which W is evaluated and propagated.

        The log-derivative method needs an even number of intervals per
        sector, so each sector uses its grid points plus mid-points (step
        h/2).  The last point is always ``grid.r[-1]``.
        """
        if propagator == "numerov":
            if grid.r_mid is not None:
                raise ValueError("two-sector grids (r_mid, step_long) need "
                                 "propagator='log-derivative'")
            return [(grid.r, grid.step)]
        return [(r0 + 0.5 * h * np.arange(2 * m + 1), 0.5 * h) for r0, h, m in grid.sectors]

    def _check_boundaries(self, W_first, r):
        if np.min(np.linalg.eigvalsh(np.atleast_3d(W_first).reshape(-1, self.n, self.n)), initial=np.inf) <= 0:
            warnings.warn("a channel is classically allowed at r_min: the hard-wall boundary "
                          "condition affects the result; decrease r_min.", stacklevel=3)
        r_max = float(r[-1])
        r_in = np.array([max(0.8 * r_max, float(r[0]))])
        r_out = np.array([r_max])
        for i in range(self.n):
            for j in range(i, self.n):
                p = self._pots[i][j]
                if p is None:
                    continue
                v_out, v_in = abs(float(p(r_out)[0])), abs(float(p(r_in)[0]))
                if v_out > 1e-14 and v_out > 0.5 * v_in:
                    kind = "coupling" if i != j else "potential"
                    warnings.warn(
                        f"{kind} [{i}][{j}] does not decay at r_max = {r_max:.6g} bohr "
                        f"(|V(r_max)| = {v_out:.3g} Eh, decays slower than ~R^-3). Matching "
                        "assumes it is negligible beyond r_max: put constant offsets in "
                        "`thresholds`, use a short-range form for couplings, or a LongRange "
                        "reference for slow tails.", stacklevel=3)

    def _check_run(self, collision_energy, entrance):
        if not (np.isfinite(collision_energy) and collision_energy > 0):
            raise ValueError("collision_energy must be finite and > 0 (hartree)")
        ok = isinstance(entrance, (int, np.integer)) and not isinstance(entrance, bool)
        if not (ok and 0 <= entrance < self.n):
            raise ValueError(f"entrance must be an integer in [0, {self.n - 1}]")

    def solve(self, collision_energy: float, grid: Grid, B: float = 0.0,
              entrance: int = 0, propagator: str = "log-derivative") -> ScatteringResult:
        """Solve at fixed field ``B`` (gauss).

        ``collision_energy`` (hartree) is measured from the threshold of the
        ``entrance`` channel.  ``propagator`` is ``"log-derivative"`` (default:
        Johnson propagator on the grid plus mid-points, matched at ``r_max``)
        or ``"numerov"`` (legacy renormalized Numerov with two-point matching).
        """
        self._check_run(collision_energy, entrance)
        _check_propagator(propagator)
        if not np.isfinite(B):
            raise ValueError("B must be finite (gauss)")
        thr = self.thresholds(B)
        E = thr[entrance] + collision_energy
        sectors = self._sectors(grid, propagator)
        prop = None
        for s_idx, (rp, h) in enumerate(sectors):
            W = self.w_matrix(rp, E, B)
            if s_idx == 0:
                self._check_boundaries(W[0], grid.r)
            prop = (log_derivative(W, h, prop) if propagator == "log-derivative"
                    else renormalized_numerov(W, h))
        return self._result(prop, E, thr, B, entrance, sectors[-1][0], propagator)

    def scan(self, fields, collision_energy: float, grid: Grid,
             entrance: int = 0, propagator: str = "log-derivative") -> list[ScatteringResult]:
        """Solve for many fields (gauss) in one batched sweep over the grid."""
        self._check_run(collision_energy, entrance)
        _check_propagator(propagator)
        fields = np.atleast_1d(np.asarray(fields, dtype=float))
        if fields.ndim != 1 or fields.size == 0 or not np.all(np.isfinite(fields)):
            raise ValueError("fields must be a non-empty 1-D array of finite values (gauss)")
        sectors = self._sectors(grid, propagator)
        thr = np.array([self.thresholds(B) for B in fields])            # (nB, N)
        E = thr[:, entrance] + collision_energy                         # (nB,)
        D = 2.0 * self.mu * (thr - E[:, None])
        idx = np.arange(self.n)
        prop = None
        for s_idx, (rp, h) in enumerate(sectors):
            W0 = 2.0 * self.mu * self.potential_matrix(rp)
            W0[:, idx, idx] += self.ell * (self.ell + 1) / rp[:, None] ** 2
            if s_idx == 0:
                first = np.broadcast_to(W0[0], (len(fields), self.n, self.n)).copy()
                first[:, idx, idx] += D
                self._check_boundaries(first, grid.r)
            prop = (log_derivative_batch(W0, D, h, prop) if propagator == "log-derivative"
                    else renormalized_numerov_batch(W0, D, h))
        r_end = sectors[-1][0]
        return [self._result(prop[b], E[b], thr[b], fields[b], entrance, r_end, propagator)
                for b in range(len(fields))]

    def scan_field(self, fields, collision_energy: float, grid: Grid, entrance: int = 0,
                   propagator: str = "log-derivative"):
        """Complex s-wave scattering length ``a(B)`` in bohr for each field (gauss)."""
        return np.array([res.scattering_length() for res in
                         self.scan(fields, collision_energy, grid, entrance, propagator)])
