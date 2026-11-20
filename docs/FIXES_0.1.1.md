# Review and fixes for Coldscatter 0.1.1

This document lists the problems found in a review of 0.1.0, how each was
confirmed, how it was fixed, and the test that guards it. Numbers come from
`benchmarks/molscat/compare.py` and the test suite.

## Summary

| # | problem | severity | status |
|---|---|---|---|
| 1 | Two-point matching inside a non-zero potential: step error first order in h | numerical, small | fixed |
| 2 | Near threshold the error **grew** when the step was refined (round-off) | numerical, important for ultracold work | fixed |
| 3 | Uniform grid only: long ranges need ~10x more points than necessary | performance | fixed (two-sector grid) |
| 4 | No check that potentials/couplings vanish at r_max | silent wrong results | fixed (warning) |
| 5 | `scan()` skipped the "classically allowed at r_min" warning | inconsistency | fixed |
| 6 | `atomic.py`: 87Rb g-factors as silent defaults, no quadrupole term, not exported | correctness risk | fixed; pair basis still open |
| 7 | Cross sections only for distinguishable particles | missing physics | fixed |
| 8 | `__version__` (0.1.0.dev0) differed from pyproject/CITATION (0.1.0) | metadata | fixed |
| 9 | Work log at repository root | tidiness | moved |
| 10 | Fragile initial guess in `fit_feshbach` | robustness | fixed |
| 11 | Closed-channel long-range reference used `2 mu (kappa^2 + V)` | **bug** | fixed |
| 12 | `tests/run_all.py` failed 26 tests (parametrized/fixture tests called without arguments) | test tooling | fixed |

Two points from the first review were withdrawn after checking:
`docs/MANUAL.md` is an intentional pointer to the root manual (not a
conflicting duplicate), and `tools/monitor_usage.py` is a documented
maintainer tool outside the package.

## MOLSCAT validation (context for 1-3)

MOLSCAT 2025.0 was built from source and run on exactly the same model
(two s-wave channels, LJ 12-6 diagonal potentials **and** coupling, mu = 20
amu, same R_max). The ~1e-3 difference recorded for 0.1.0 is not a
Coldscatter error: MOLSCAT's own `molscat-2chan-LJ` example is not converged
with its shipped step sizes. Converged MOLSCAT and Coldscatter agree; see
`benchmarks/molscat/README.md`.

## 1. First-order step error from two-point matching

**Symptom.** Against converged MOLSCAT, |dS21| halved exactly when the step
halved (1.4e-7, 7.0e-8, 3.5e-8, 1.8e-8): first order, although Numerov is
fourth order.

**Diagnosis.** The propagator alone is fine (error ~h^5 on an exact test). The
error is independent of r_min. Switching the potential smoothly to zero before
r_max restored fourth order (factor 16 per halving). So the error comes from
matching: `psi(r[-2]) = M psi(r[-1])` was matched to *free* Riccati functions,
which do not satisfy the equation on [r[-2], r[-1]] when V(r_max) != 0, giving
an error ~ h V(r_max).

**Fix.** New default propagator: Johnson's log-derivative method
(`propagators.log_derivative`, J. Comput. Phys. 13, 445 (1973), the method
behind MOLSCAT's LDPROP). It carries Y = psi' psi^-1 directly, so matching uses
one radius (`asymptotics.match_k_matrix_logderiv`):
`[Y G - G', Y D - D'] X = F' - Y F`. The solver propagates on the user grid plus
mid-points (the method needs an even number of intervals).

**Result.** |dS21| = 7e-10 already at h = 1e-3 angstrom; S within 6e-8 of MOLSCAT.
Tests: `test_log_derivative_is_fourth_order`, `test_s_matrix_converged_at_coarse_step`.

## 2. Loss of precision near threshold

**Symptom.** a(E = 1e-6 cm^-1), R_max = 100 angstrom: relative error 1e-8 at h = 2e-3,
2.7e-6 at 5e-4, 5.5e-6 at 2.5e-4 angstrom. The same happens in the square-well
test (1.4e-6 -> 1.1e-4 as h goes 0.05 -> 0.0125 bohr); the old test only passed
because it stopped at h = 0.025.

**Diagnosis.** At small k h the two values psi(r[-2]) and psi(r[-1]) are nearly
equal, so the information about the derivative sits in their small difference,
while the Numerov ratio carries accumulated round-off ~ n eps. The error scales
like 1/h^2.

**Fix.** Same as 1: the log-derivative is propagated and matched directly, no
difference of neighbouring values is formed.

**Result.** 3.6e-9 relative error at every step from 4e-3 to 2.5e-4 angstrom.
Tests: `test_near_threshold_does_not_degrade_with_finer_step`, `test_square_well_exact`.

## 3. Uniform grid only

**Fix.** `Grid(r_min, r_max, step, r_mid=..., step_long=...)` defines a fine inner
and a coarse outer sector; the outer log-derivative propagation starts from the
inner result (MOLSCAT's RMID/DRL idea). `Grid.sectors` exposes the layout.

**Result.** Case C: 99 351 points, 2.4 s -> 4 556 points, 0.13 s, same accuracy
(7e-9 vs MOLSCAT). Tests: `test_two_sector_grid_matches_molscat`,
`test_two_sector_grid_matches_uniform_and_ends_at_r_max`.

## 4. Non-decaying potentials at r_max

`CoupledChannels(..., [[lj, 1e-7], ...])` accepted a constant coupling and
matched as if it vanished. The solver now warns when any element has
|V(r_max)| > 1e-14 Eh and |V(r_max)| > 0.5 |V(0.8 r_max)| (decays slower than ~R^-3),
pointing to `thresholds` for constant offsets and `LongRange` for slow tails.
Tests: `test_non_decaying_coupling_warns`, `test_decaying_potentials_do_not_warn`.

## 5. `scan()` warning

`scan()` now runs the same boundary checks as `solve()`.
Test: `test_scan_warns_when_r_min_is_classically_allowed`.

## 6. Atomic structure

* `ALKALI_GROUND_STATES` presets (6Li, 23Na, 40K, 85Rb, 87Rb, 133Cs; Arimondo et
  al., RMP 49, 31 (1977); Steck data) and `alkali_hamiltonian(atom, B)`; every
  preset reproduces the ground-state hyperfine splitting to 1e-8.
* `hyperfine_hamiltonian` documents that its defaults are 87Rb values and gains
  the electric-quadrupole term `B_hf` (I, J >= 1).
* `coldscatter.atomic` is exported.
* **Still open:** a two-atom hyperfine/Zeeman channel basis feeding the solver
  (roadmap v0.3). Feshbach results are therefore for model potentials only.

## 7. Identical particles

`cross_sections(identical="bosons" | "fermions")`: factor 2 for allowed partial
waves, 0 for forbidden ones (same internal state). Check: sigma -> 8 pi a^2 for
bosons at low energy. Test: `test_identical_particle_cross_sections`.

## 8. Version

`__version__` is read from the installed package metadata (pyproject.toml); all
three places say 0.1.1. Test: `test_version_matches_pyproject`.

## 9. Work log

`LONG_RANGE_VALIDATION_RECORD.txt` moved to `docs/validation_logs/`.

## 10. `fit_feshbach`

Input validation; the pole is the largest jump *with a sign change*; a guard
against a_bg ~ 0 (division by zero in the width guess); symmetric handling of
B = B0. Test: `test_fit_feshbach_with_zero_background_guess_and_validation`.

## 11. Bug: closed-channel long-range reference

`_long_closed_ratio` integrated `D'' = [l(l+1)/r^2 + 2 mu (kappa^2 + V)] D`. Since
`kappa^2 = -k2` already contains 2 mu, the correct term is `kappa^2 + 2 mu V`. With
a zero tail the ratio must equal the free one; it did not (1.036 vs 1.025 at
mu = 1, 3.06 vs 1.025 at mu = 1000). The existing test did not catch it because
its "direct" reference copied the same equation. Both are fixed.
**Results from 0.1.0 that used `LongRange` tails in closed channels should be
recomputed.** Test: `test_zero_tail_closed_reference_equals_free`.

## 12. `tests/run_all.py`

The "dependency-free" runner called every test with no arguments, so all
parametrized tests and fixture tests (e.g. `tmp_path`) failed (26 on this tree),
and most test modules import pytest anyway. It now delegates to pytest when
installed; without pytest it runs argument-free tests and reports the rest as
SKIP (0 failures, 7 skipped in a pytest-free virtualenv).

## Behaviour changes to be aware of

* Default propagator is now `"log-derivative"`; `propagator="numerov"` gives the
  old behaviour.
* The hard wall (psi = 0) is **at** r_min with the default and one step before
  r_min with Numerov. Irrelevant when r_min is deep in the repulsive wall (as
  recommended); it matters only for models with a wall exactly at r_min, such as
  the square-well test, which now uses the wall at r_min.
* The default costs about two potential evaluations per user grid point
  (mid-points); use a two-sector grid for long ranges.
