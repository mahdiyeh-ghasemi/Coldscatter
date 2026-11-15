"""Shared reference calculations for the test-suite."""
import numpy as np
from scipy.integrate import solve_ivp

import coldscatter as cs
from coldscatter import units as u
from coldscatter.asymptotics import match_k_matrix, s_from_k

MU = u.reduced_mass(87.0, 87.0)   # Rb-87-like
C6 = 4700.0                       # a.u.
DMU = 2.0 * u.MU_B_HARTREE_PER_GAUSS


def lj_for_rm(rm, c6=C6):
    """Lennard-Jones with fixed C6 and equilibrium distance ``rm``."""
    return cs.LennardJones(c6, c6 * rm**6 / 2.0)


def ode_reference_scalar(potential, mu, energy, grid):
    """Single-channel s-wave scattering length from an independent adaptive ODE solve."""
    r = grid.r

    def rhs(x, y):
        return [y[1], 2.0 * mu * (potential(x) - energy) * y[0]]

    y0 = 1e-30
    sol = solve_ivp(rhs, (r[0], r[-1]),
                    [y0, y0 * np.sqrt(2.0 * mu * (potential(r[0]) - energy))],
                    t_eval=[r[-2], r[-1]], method="DOP853", rtol=1e-12, atol=1e-200)
    M = np.array([[sol.y[0, 0] / sol.y[0, 1]]])
    K, _, k, _ = match_k_matrix(M, r[-2], r[-1], np.array([2.0 * mu * energy]), [0])
    return (1j * np.log(s_from_k(K)[0, 0]) / (2.0 * k[0])).real


def feshbach_model(coupling=2e-4):
    return cs.two_channel_feshbach(
        MU, C6, c12_open=C6 * 11.0**6 / 2, c12_closed=C6 * 10.4**6 / 2,
        coupling=coupling, coupling_r0=11.0, coupling_sigma=1.5, delta_mu=DMU)
