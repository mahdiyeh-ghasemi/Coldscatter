#!/usr/bin/env python3
"""Write the MOLSCAT 2025.0 input files used for reference_molscat_2025.0.json.

Usage:
    python write_molscat_inputs.py            # writes case_A.input, case_B.input, case_C.input
    molscat-basic < case_A.input > case_A.output

Build ``molscat-basic`` from https://github.com/molscat/molscat
(``cd source_code && make molscat-basic``; needs gfortran + LAPACK/BLAS).

Interaction: ITYPE = 2 (vibrating diatom + atom) is used only as a container
for two s-wave channels with thresholds 0 and 100 cm^-1; the "vibrational"
labels 1 and 2 index the channels, and the three LAMBDA entries give
V11, V22 and V12 as Lennard-Jones 12-6 terms.
"""
from pathlib import Path

POTL = """ &BASIS
    ITYPE  = 2,
    NLEVEL = 2,  JLEVEL = 0, 1,   0, 2,
                 ELEVEL = 0.D0,   100.D0,
 /
 &POTL
    RM     = 1.0,   EPSIL = 1.0,
    MXLAM  = 3,
    LAMBDA = 0, 1, 1,    0, 2, 2,    0, 1, 2,
    NTERM  = 2,          2,          2,
    NPOWER = -12, -6,    -12, -6,    -12, -6,
    A      = 400., -400.,   400., -400.,   40., -40.,
 /
"""


def block(label, energies, rmax, long_prop, extra=""):
    elist = ", ".join(f"{e:.10g}" for e in energies)
    return f""" &INPUT
    LABEL  = '{label}',
    URED   = 20.,
    IPROPS = 7,       IPROPL = {long_prop},     IPRINT = 12,   ISCRU = 0,
    RMIN   = 0.8,     RMAX   = {rmax},          RMID   = 1.5,
    DRS    = 0.00002, DRL    = 0.00002,         IRMSET = 8, {extra}
    JTOTL  = 0,       JTOTU  = 0,               IBFIX  = 2,
    NNRG   = {len(energies)},  ENERGY = {elist},  EUNITS = 1,
    LASTIN = 1,
 /
""" + POTL


CASES = {
    "case_A.input": block("2-channel LJ, both open", [200.0], "6.0", 6),
    "case_B.input": block("2-channel LJ, closed-channel resonance",
                          [30.0, 42.8380, 42.8385, 42.8388, 42.8390,
                           42.8392, 42.8395, 42.8400, 60.0], "6.0", 6),
    "case_C.input": block("2-channel LJ, near threshold", [1.0e-4, 1.0e-6],
                          "100.0", 9, extra="TOLHIL = 1.0E-10,"),
}

if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    for name, text in CASES.items():
        (here / name).write_text(text)
        print("wrote", here / name)
