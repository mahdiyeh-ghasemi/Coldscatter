"""Equivalence test against converged MOLSCAT 2025.0 output.

Same model in both programs (two s-wave channels, LJ 12-6 diagonal potentials
and LJ 12-6 coupling, mu = 20 amu, same R_max).  Reference values and the
MOLSCAT settings used to converge them are in
benchmarks/molscat/reference_molscat_2025.0.json; MOLSCAT is not needed here.
"""
import importlib.util
from pathlib import Path

import pytest

_path = Path(__file__).resolve().parents[1] / "benchmarks" / "molscat" / "compare.py"
_spec = importlib.util.spec_from_file_location("molscat_compare", _path)
mc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mc)

MODEL = mc.build_model()
CASES = mc.REF["cases"]


def test_two_open_channels_s_matrix():
    case = CASES["A_two_open_channels"]
    S = mc.solve(MODEL, case["energy_cm"], 5e-4, case["rmax_A"]).S
    for key, (i, j) in {"11": (0, 0), "21": (1, 0), "22": (1, 1)}.items():
        assert abs(S[i, j] - complex(*case["S"][key])) < 1e-6, key


@pytest.mark.parametrize("point", CASES["B_closed_channel_resonance"]["points"],
                         ids=lambda p: f"E={p['energy_cm']}")
def test_closed_channel_resonance(point):
    case = CASES["B_closed_channel_resonance"]
    s = mc.solve(MODEL, point["energy_cm"], 5e-4, case["rmax_A"]).S[0, 0]
    near_pole = min(abs(point["energy_cm"] - e) for e in case["pole_energies_cm"]) < 5e-4
    # Within ~1e-4 cm^-1 of the pole S11 turns through 2 pi, so a ~1e-8 cm^-1
    # offset in the resonance position appears as |dS| ~ 1e-4 there.
    tol = 2e-4 if near_pole else 2e-6
    assert abs(s - complex(*point["S11"])) < tol


@pytest.mark.parametrize("point", CASES["C_near_threshold_scattering_length"]["points"],
                         ids=lambda p: f"E={p['energy_cm']}")
def test_near_threshold_scattering_length(point):
    case = CASES["C_near_threshold_scattering_length"]
    a = mc.solve(MODEL, point["energy_cm"], 2e-3, case["rmax_A"]).scattering_length().real / mc.ANGSTROM
    assert abs(a - point["a_A"]) / point["a_A"] < 1e-7


def test_near_threshold_does_not_degrade_with_finer_step():
    """Near threshold the result must stay converged when the step is refined."""
    case = CASES["C_near_threshold_scattering_length"]
    point = case["points"][1]
    a = mc.solve(MODEL, point["energy_cm"], 2.5e-4, case["rmax_A"]).scattering_length().real / mc.ANGSTROM
    assert abs(a - point["a_A"]) / point["a_A"] < 1e-7


def test_s_matrix_converged_at_coarse_step():
    """The default propagator is converged to MOLSCAT already at h = 1e-3 A."""
    case = CASES["A_two_open_channels"]
    S = mc.solve(MODEL, case["energy_cm"], 1e-3, case["rmax_A"]).S
    assert abs(S[1, 0] - complex(*case["S"]["21"])) < 2e-9
    assert abs(S[0, 0] - complex(*case["S"]["11"])) < 1e-7


def test_two_sector_grid_matches_molscat():
    import coldscatter as cs
    case = CASES["C_near_threshold_scattering_length"]
    A = mc.ANGSTROM
    grid = cs.Grid(0.65 * A, 100.0 * A, 2e-3 * A, r_mid=6.0 * A, step_long=5e-2 * A)
    for point in case["points"]:
        a = MODEL.solve(point["energy_cm"] * mc.CM, grid).scattering_length().real / A
        assert abs(a - point["a_A"]) / point["a_A"] < 1e-7
