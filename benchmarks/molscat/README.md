# MOLSCAT 2025.0 benchmark

Equivalence test of Coldscatter against MOLSCAT 2025.0 on an identical model:
two s-wave channels (thresholds 0 and 100 cm⁻¹), reduced mass 20 amu, and
Lennard-Jones 12-6 terms for V₁₁, V₂₂ **and** the coupling V₁₂
(`V = C12 R⁻¹² + C6 R⁻⁶`, cm⁻¹ with R in Å). Both programs match to free
Riccati–Bessel functions at the same R_max.

| file | purpose |
|---|---|
| `write_molscat_inputs.py` | writes the three MOLSCAT inputs (written for this benchmark) |
| `reference_molscat_2025.0.json` | converged MOLSCAT results and the settings used |
| `compare.py` | runs Coldscatter on the same model and prints the comparison |
| `../../tests/test_molscat_benchmark.py` | CI test using the stored references (MOLSCAT not required) |

## Results (Coldscatter step 5×10⁻⁴ Å for A/B, 2×10⁻³ Å for C)

| case | quantity | 0.1.1 default (log-derivative) | 0.1.0 (Numerov, two-point matching) |
|---|---|---|---|
| A: E = 200 cm⁻¹, both channels open, R_max = 6 Å | S₁₁, S₂₁, S₂₂ | 5.5×10⁻⁸ | 3.8×10⁻⁷ |
| B: closed-channel resonance near 42.839 cm⁻¹, R_max = 6 Å | S₁₁ away from the pole | 1.2×10⁻⁷ | 6.5×10⁻⁷ |
| B: same, within 10⁻⁴ cm⁻¹ of the pole | S₁₁ | 8.6×10⁻⁵ (≈10⁻⁸ cm⁻¹ shift in resonance position) | 8.6×10⁻⁵ |
| C: E = 10⁻⁴ and 10⁻⁶ cm⁻¹, R_max = 100 Å | scattering length a(k) | 7×10⁻⁹ relative, at any step | 1×10⁻⁸ at h = 2×10⁻³ Å, 5.5×10⁻⁶ at h = 2.5×10⁻⁴ Å |

Step-size behaviour (case A, |ΔS₂₁|): Numerov 1.4×10⁻⁷ → 7.0×10⁻⁸ → 3.5×10⁻⁸ →
1.8×10⁻⁸ for h = 2, 1, 0.5, 0.25 ×10⁻³ Å (first order); log-derivative is
converged to 7×10⁻¹⁰ already at h = 10⁻³ Å.

## Convergence notes for MOLSCAT (important)

* The **shipped MOLSCAT example `molscat-2chan-LJ.input` is not converged**
  (AIPROP with DRS = DRL = 10⁻³): its S₂₁ is off by 1.5×10⁻³ relative. This
  is the source of the "~10⁻³ unexplained difference" previously recorded in
  `docs/VALIDATION_STATUS.md`. With DRS = DRL ≤ 10⁻⁴ (MAPROP + MDPROP)
  MOLSCAT changes by < 10⁻¹⁰ and agrees with Coldscatter.
* At long range use AIPROP with a tight `TOLHIL`. Fixed-step MDPROP out to
  100 Å accumulates round-off: DRL = 2×10⁻⁴ vs 5×10⁻⁵ changes a by 1 %.

## Reproduce the MOLSCAT side

```bash
git clone https://github.com/molscat/molscat
cd molscat/source_code && make molscat-basic        # gfortran + LAPACK/BLAS
cd -; python benchmarks/molscat/write_molscat_inputs.py
molscat/source_code/bin/molscat-basic < benchmarks/molscat/case_A.input > case_A.output
```

MOLSCAT is GPL-3.0 and is not redistributed here; this folder contains only
input parameters and numerical results. Cite J. M. Hutson & C. R. Le Sueur,
Comput. Phys. Commun. 241, 9 (2019) and MOLSCAT version 2025.0.
