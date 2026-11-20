# Roadmap

**v0.1 (this release)** - internally tested single/multi-channel core, batched field scans, and resonance fitting. Analytical and independent numerical checks are included; external realistic-system benchmarking remains open.

**v0.2 - correctness and reach**
- First MOLSCAT equivalence test done in 0.1.1 (`benchmarks/molscat`). Cross-check complete scattering observables against MOLSCAT and published Rb/Cs/Na-K benchmarks; document model definitions, units, convergence settings, and comparison tables.
- ~~Log-derivative propagator~~ (done in 0.1.1: Johnson, two-sector grid). Next: fully adaptive steps / Airy propagator (Manolopoulos-Alexander) for very long ranges.
- Long-range matching to the `-C6/R^6` solutions (removes the `R_max^-3` error).
- Bound states and resonance positions (BOUND/FIELD-like); resonance width and `a_bg` extraction.

**v0.3 - physics builders**
- Atom-pair basis: hyperfine + Zeeman, `|f m_f>` states, `M_F` conservation, singlet/triplet potentials.
- Molecule-molecule and atom-molecule bases: rigid rotor, microwave dressing, sticky-collision/loss models.
- Identical-particle symmetrisation, thermal averaging, rate coefficients.

**v0.4 - performance and differentiability**
- Optional JAX/Numba backend; gradients of `a(B)` with respect to potential parameters (fit to data).

**v1.0 - community**
- Documentation site, tutorials, benchmark suite, JOSS paper, PyPI release.
