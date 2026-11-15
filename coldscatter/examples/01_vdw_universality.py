"""Scattering length of a Lennard-Jones potential vs. its short-range phase.

Gribakin-Flambaum theory predicts a = a_bar [1 - tan(Phi - pi/8)], so scanning
the short-range parameter sweeps a through poles, with median a_bar.
"""
import numpy as np
import matplotlib.pyplot as plt

import coldscatter as cs
from coldscatter import units as u

mu, c6 = u.reduced_mass(87.0, 87.0), 4700.0
a_bar = cs.mean_scattering_length(mu, c6)
grid = cs.Grid(4.0, 3000.0, 0.02)
r_m = np.linspace(9.0, 12.0, 400)          # equilibrium distance sets C12 = C6 r_m^6 / 2
a = np.array([cs.single_channel(mu, cs.LennardJones(c6, c6 * rm**6 / 2))
              .solve(u.from_kelvin(1e-9), grid).scattering_length().real for rm in r_m])

fig, ax = plt.subplots(figsize=(6.4, 3.8))
ax.plot(r_m, np.where(np.abs(a) < 600, a, np.nan), lw=1.4)
ax.axhline(a_bar, color="k", ls="--", lw=1, label=r"$\bar a$ (Gribakin-Flambaum)")
ax.axhline(np.median(a), color="C3", ls=":", lw=1.2, label="median of computed $a$")
ax.set(xlabel=r"equilibrium distance $r_m$ (bohr)", ylabel="a (bohr)", ylim=(-600, 600))
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig("docs/img/vdw_universality.png", dpi=160)
print(f"a_bar = {a_bar:.1f} a0, median(a) = {np.median(a):.1f} a0")
