"""Comparison with values printed in the literature (87Rb van der Waals scales).

Sources:
- C. Chin, R. Grimm, P. Julienne, E. Tiesinga, Rev. Mod. Phys. 82, 1225 (2010), Table I:
  87Rb, m = 86.909187 u, C6 = 4698 a.u.  ->  R_vdW = 82.58 a0, E_vdW = 0.2922 mK.
- P. S. Julienne's table of van der Waals scales: 87Rb, C6 = 4691 a.u.  ->  a_bar = 78.92 a0.
"""
import coldscatter as cs
from coldscatter import units as u

MU_RB87 = u.reduced_mass(86.909187, 86.909187)


def test_rb87_vdw_length_matches_chin_2010():
    assert abs(cs.vdw_length(MU_RB87, 4698.0) - 82.58) < 0.02


def test_rb87_vdw_energy_matches_chin_2010():
    r = cs.vdw_length(MU_RB87, 4698.0)
    e_mk = float(u.to_kelvin(1.0 / (2.0 * MU_RB87 * r**2))) * 1e3
    assert abs(e_mk - 0.2922) < 1e-4


def test_rb87_mean_scattering_length_matches_literature():
    assert abs(cs.mean_scattering_length(MU_RB87, 4691.0) - 78.92) < 0.01