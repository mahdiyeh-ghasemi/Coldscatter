"""Magnetic Feshbach resonance in a two-channel model, with a line-shape fit."""
import numpy as np
import matplotlib.pyplot as plt

import coldscatter as cs
from coldscatter import units as u

mu, c6 = u.reduced_mass(87.0, 87.0), 4700.0
model = cs.two_channel_feshbach(
    mu, c6, c12_open=c6 * 11.0**6 / 2, c12_closed=c6 * 10.4**6 / 2,
    coupling=2e-4, coupling_r0=11.0, coupling_sigma=1.5,
    delta_mu=2 * u.MU_B_HARTREE_PER_GAUSS)            # delta_mu = 2 mu_B

grid = cs.Grid(7.0, 800.0, 0.02)
B = np.linspace(22.0, 44.0, 441)                      # gauss
a = model.scan_field(B, u.from_kelvin(1e-7), grid).real
a_bg, B0, delta = cs.fit_feshbach(B, a)
print(f"fit: a_bg = {a_bg:.1f} a0, B0 = {B0:.3f} G, Delta = {delta:.2f} G")

fig, ax = plt.subplots(figsize=(6.4, 3.8))
ax.plot(B, np.where(np.abs(a) < 1500, a, np.nan), lw=1.6, label="coupled-channel")
ax.plot(B, np.where(np.abs(a) < 1500, cs.feshbach_a(B, a_bg, B0, delta), np.nan),
        "k--", lw=1, label=r"$a_{bg}\,[1-\Delta/(B-B_0)]$ fit")
ax.axvline(B0, color="C3", ls=":", lw=1)
ax.set(xlabel="magnetic field (G)", ylabel="a (bohr)", ylim=(-1500, 1500))
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig("docs/img/feshbach_scan.png", dpi=160)
