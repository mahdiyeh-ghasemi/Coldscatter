# Roadmap

**v0.1 (this release)** - validated single/multi-channel core, batched field scans, fit.

**v0.2 - correctness and reach**
- Cross-check against MOLSCAT on published Rb/Cs/Na-K benchmarks (publish the comparison tables).
- Adaptive-step log-derivative propagator (Johnson / Manolopoulos) to cut cost on long ranges.
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
