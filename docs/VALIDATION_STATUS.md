# Validation status (v0.1.0, pre-alpha)

## Verified
- Internal test suite passes (see `pytest`).
- Single-channel and open-channel Wronskian checks.
- Closed-channel reference-ratio checks.
- Matching-radius stability.
- Independent solve_ivp/DOP853 reference for a limited one-open/one-closed problem.
- S-matrix symmetry and unitarity in the tested cases.

## Not yet validated
- Comparison against MOLSCAT 2025.0 (`molscat-2chan-LJ` example): the S-matrix
  differs by about 1e-3 and the source is not yet identified. The current comparison
  does not use the identical coupling model, so it is not an equivalence test.
- Multichannel configurations with closed channels in the coupled-channel solver.
- Feshbach resonance positions and widths against literature values.
- Off-diagonal long-range couplings (not implemented; only diagonal reference matching).

## Scope
Results should be used only for the tested configurations above until the
benchmarks in the roadmap are passed.
