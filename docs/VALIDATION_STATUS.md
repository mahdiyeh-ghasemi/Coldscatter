# Validation status (v0.1.1)

## Verified
- Internal test suite passes (see `pytest`).
- Single-channel and open-channel Wronskian checks.
- Closed-channel long-range reference checks against direct integration.
- Matching-radius stability.
- Independent solve_ivp/DOP853 reference for a limited one-open/one-closed problem.
- S-matrix symmetry and unitarity in the tested cases.
- Fourth-order step convergence of the default log-derivative propagator, and no
  loss of precision when the step is refined near threshold.
- **Equivalence test against MOLSCAT 2025.0** on an identical two-channel
  Lennard-Jones model (LJ coupling, same R_max), see `benchmarks/molscat/`.
  With the default propagator (step 5e-4 angstrom):
  - two open channels: S matrix within 6e-8;
  - one open + one closed channel, resonance near 42.839 cm^-1: S11 within
    1.2e-7 away from the pole; within 1e-4 cm^-1 of the pole the difference
    (up to 9e-5) corresponds to a ~1e-8 cm^-1 offset of the resonance position;
  - near threshold (E = 1e-4, 1e-6 cm^-1, R_max = 100 angstrom): scattering
    length within 7e-9 relative, also on a two-sector grid.
  Note: MOLSCAT's own `molscat-2chan-LJ` example is not converged with its
  shipped step sizes; the converged settings are recorded in the benchmark.

## Not yet validated
- Multichannel configurations with more than one closed channel.
- Realistic atom-pair systems (hyperfine/Zeeman pair basis not built in yet):
  Feshbach resonance positions and widths against MOLSCAT / literature values.
- Off-diagonal long-range couplings (not implemented; only diagonal reference matching).

## Scope
Results should be used only for the tested configurations above until the
benchmarks in the roadmap are passed.
