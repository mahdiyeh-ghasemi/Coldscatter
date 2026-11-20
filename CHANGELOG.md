# Changelog

## 0.1.1 (2026-11-20)

Details, diagnosis and tests for every item: [docs/FIXES_0.1.1.md](docs/FIXES_0.1.1.md).

### Fixed
- Closed-channel long-range reference used `2 mu (kappa^2 + V)` instead of
  `kappa^2 + 2 mu V` (recompute 0.1.0 results with closed-channel `LongRange` tails).
- First-order step error from two-point matching inside a non-zero potential.
- Loss of precision near threshold when the step is refined.
- `scan()` now gives the same r_min warning as `solve()`.
- `__version__` consistent with pyproject.toml and CITATION.cff.
- More robust initial guess and input validation in `fit_feshbach`.
- `tests/run_all.py` delegates to pytest, and skips (instead of failing) tests that need it.

### Added
- Johnson log-derivative propagator (`log_derivative`, `log_derivative_batch`)
  with single-radius matching (`match_k_matrix_logderiv`); default for
  `solve`, `scan`, `scan_field` (`propagator="numerov"` keeps the old path).
- Two-sector grids: `Grid(..., r_mid=..., step_long=...)`, `Grid.sectors`.
- Warning when a potential or coupling does not decay at r_max.
- `cross_sections(identical="bosons" | "fermions")`.
- `atomic.ALKALI_GROUND_STATES`, `atomic.alkali_hamiltonian`, quadrupole term
  `B_hf`; `coldscatter.atomic` exported.
- MOLSCAT 2025.0 equivalence benchmark (`benchmarks/molscat`, `tests/test_molscat_benchmark.py`).

### Changed
- Hard wall at r_min (default propagator) instead of one step before it.

## 0.1.0 (2026-11-18)
- First development release.
