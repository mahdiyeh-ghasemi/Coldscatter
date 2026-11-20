#!/usr/bin/env python3
"""Compare Coldscatter with converged MOLSCAT 2025.0 reference values.

    python benchmarks/molscat/compare.py [--step 5e-4]

The model (two s-wave channels, Lennard-Jones 12-6 diagonal potentials and
coupling) is defined identically in both programs; see
reference_molscat_2025.0.json.  MOLSCAT does not need to be installed.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from scipy import constants as sc

import coldscatter as cs

REF = json.loads((Path(__file__).with_name("reference_molscat_2025.0.json")).read_text())

ANGSTROM = 1e-10 / sc.physical_constants["Bohr radius"][0]          # bohr per angstrom
CM = 100.0 / sc.physical_constants["hartree-inverse meter relationship"][0]  # hartree per cm^-1


def _lj(term):
    c12, c6 = term["C12"], term["C6"]
    return lambda r: (c12 * (r / ANGSTROM) ** -12 + c6 * (r / ANGSTROM) ** -6) * CM


def build_model() -> cs.CoupledChannels:
    m = REF["model"]
    mu = cs.units.amu(m["reduced_mass_amu"])
    return cs.CoupledChannels(
        float(mu),
        np.asarray(m["thresholds_cm"]) * CM,
        [[_lj(m["V11"]), _lj(m["V12"])], [None, _lj(m["V22"])]],
        ell=0,
    )


def solve(model, energy_cm, step_A, rmax_A, rmin_A=0.65):
    """Grid ends exactly at rmax_A (the MOLSCAT RMAX)."""
    n = round((rmax_A - rmin_A) / step_A)
    r0 = rmax_A - n * step_A
    grid = cs.Grid(r0 * ANGSTROM, rmax_A * ANGSTROM, step_A * ANGSTROM)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.solve(energy_cm * CM, grid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=float, default=5e-4, help="grid step in angstrom (cases A, B)")
    ap.add_argument("--step-long", type=float, default=2e-3, help="grid step in angstrom (case C)")
    args = ap.parse_args()
    model = build_model()

    A = REF["cases"]["A_two_open_channels"]
    res = solve(model, A["energy_cm"], args.step, A["rmax_A"])
    print(f"Case A  E = {A['energy_cm']} cm-1, both channels open, step = {args.step} A")
    for key, (i, j) in {"11": (0, 0), "21": (1, 0), "22": (1, 1)}.items():
        ref = complex(*A["S"][key])
        print(f"  S{key}: MOLSCAT {ref:.10f}  Coldscatter {res.S[i, j]:.10f}  |diff| {abs(res.S[i, j] - ref):.1e}")

    B = REF["cases"]["B_closed_channel_resonance"]
    print(f"\nCase B  closed-channel resonance, S11, step = {args.step} A")
    for p in B["points"]:
        ref = complex(*p["S11"])
        s = solve(model, p["energy_cm"], args.step, B["rmax_A"]).S[0, 0]
        print(f"  E = {p['energy_cm']:<8} MOLSCAT {ref:.8f}  Coldscatter {s:.8f}  |diff| {abs(s - ref):.1e}")

    C = REF["cases"]["C_near_threshold_scattering_length"]
    print(f"\nCase C  near-threshold scattering length, R_max = {C['rmax_A']} A, step = {args.step_long} A")
    for p in C["points"]:
        a = solve(model, p["energy_cm"], args.step_long, C["rmax_A"]).scattering_length().real / ANGSTROM
        print(f"  E = {p['energy_cm']:<8} a: MOLSCAT {p['a_A']:.10f} A  Coldscatter {a:.10f} A  "
              f"rel.diff {abs(a - p['a_A']) / p['a_A']:.1e}")


if __name__ == "__main__":
    main()
