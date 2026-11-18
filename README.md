# Coldscatter

**Coupled-channel scattering for ultracold atoms and molecules, implemented in a readable, tested Python package.**

`coldscatter` computes S/K matrices, scattering lengths, cross sections and Feshbach
resonances for two colliding particles, from a set of radial channels you define.
The goal is a community reference tool in the spirit of QuTiP and Pulser:
a small, well-documented API on top of a numerically validated core.

> **Status: v0.1 (pre-alpha).** The core solver has passed internal tests against
> analytical solutions and independent numerical integration references (see
> [Validation](#validation)). It has **not yet** been benchmarked against MOLSCAT
> or experimental scattering data for realistic atomic/molecular systems. External
> benchmarking is the first item on the [roadmap](docs/ROADMAP.md).
## Quick start

```bash
git clone https://github.com/mahdiyeh-ghasemi/Coldscatter.git
cd Coldscatter
python -m pip install -e ".[examples]"
```

```python
import coldscatter as cs
from coldscatter import units as u

mu = u.reduced_mass(87.0, 87.0)                       # atomic units
c6 = 4700.0
prob = cs.single_channel(mu, cs.LennardJones(c6, c6 * 11.0**6 / 2))
res = prob.solve(u.from_kelvin(1e-7), cs.Grid(r_min=4.0, r_max=1500.0, step=0.01))
print(res.scattering_length())      # complex, in bohr
print(res.cross_sections())
```

Magnetic Feshbach resonance, batched over the field (one sweep of the grid serves all fields):

```python
import numpy as np
import coldscatter as cs
from coldscatter import units as u
mu = u.reduced_mass(87.0, 87.0)
c6 = 4700.0

model = cs.two_channel_feshbach(mu, c6, c12_open=c6 * 11.0**6 / 2, c12_closed=c6 * 10.4**6 / 2,
                                coupling=2e-4, coupling_r0=11.0, coupling_sigma=1.5,
                                delta_mu=2 * u.MU_B_HARTREE_PER_GAUSS)
B_gauss = np.linspace(20.0, 45.0, 251)
a = model.scan_field(B_gauss, u.from_kelvin(1e-7), cs.Grid(7.0, 800.0, 0.02))
print(a)  # complex scattering length in bohr

# Optional: fit one isolated resonance using a narrow field window.
# B_window = ...  # choose a narrow interval around the resonance
# a_window = model.scan_field(B_window, u.from_kelvin(1e-7), cs.Grid(7.0, 800.0, 0.02))
# a_bg, B0, delta = cs.fit_feshbach(B_window, a_window.real)
```

![Feshbach resonance](docs/img/feshbach_scan.png)
![van der Waals universality](docs/img/vdw_universality.png)

## What is inside

| module | role |
|---|---|
| `CoupledChannels` | N-channel radial problem: thresholds `E_i(B)`, potential matrix `V_ij(R)`, partial waves |
| `renormalized_numerov` | stable matrix propagator (Johnson 1977), fast scalar path, batched-over-field path |
| `asymptotics` | free or explicit diagonal long-range reference matching, open and closed channels |
| `ScatteringResult` | `K`, `S`, complex scattering length, elastic/inelastic/total cross sections |
| `potentials`, `units` | Lennard-Jones, Gaussian, ..., vdW length and `a_bar`; atomic units, K, gauss |

Conventions and equations: [docs/PHYSICS.md](docs/PHYSICS.md). Full installation and usage guide: [User manual](MANUAL.md).

## Validation

Run `pytest` (or `python tests/run_all.py` without pytest). The internal test suite checks:

- square well against the analytic scattering length (relative error ~1e-6)
- Lennard-Jones against an independent adaptive ODE solver (< 1e-4)
- Numerov fourth-order convergence with step size; an `R_max^-3` matching-tail error in the specific free-reference test model (not a general error law for explicit long-range matching)
- van der Waals universality: median `a` over short-range phases ~ Gribakin-Flambaum `a_bar`
- `sigma_el -> 4 pi |a|^2`; p-wave Wigner threshold law `tan(delta_1) ~ k^3`
- decoupled two-channel limit reproduces the single channel exactly
- coupled, open-closed, and propagation-matrix results against independent DOP853 ODE integration
- open- and closed-channel long-range reference matching, including Wronskian preservation and direct closed-channel integration checks
- d- and f-wave threshold behaviour against independent numerical references
- S-matrix unitarity (1e-16), symmetry, optical theorem, batched scan = single solves
- Feshbach pole location and line-shape fit
- input validation for grids, potentials, channel definitions, and propagators

The Rb-87 literature tests currently check van der Waals length, energy, and
mean-scattering-length scales derived from published constants. They do not
constitute a full scattering-observable benchmark against experiment or MOLSCAT.
The DOP853 tests provide independent integration references for selected model
problems, not external validation of a realistic atomic or molecular system.

## Known limitations (v0.1)

- Uniform radial grid; the step is set by the shortest local wavelength.
- With `long_range=None`, matching uses free asymptotic reference functions; a residual
  long-range `-C6/R^6` contribution can therefore affect the result. Supplying a
  `LongRange` reference accounts for the specified diagonal tail in the matching
  functions, but does not remove the need to check convergence with `R_max` and
  `r_far`. Off-diagonal long-range couplings are not currently supported.
- You supply the channel basis by hand: `coldscatter.atomic` builds the single-atom hyperfine + Zeeman Hamiltonian only; there is no two-atom basis, singlet/triplet projector or rotor builder yet. `scattering_length()` returns the energy-dependent a(k) = -tan(delta)/k, so use a collision energy with k|a| << 1 for the zero-energy value.
- The Feshbach fit assumes a constant background and a single resonance.

## Contributing

Issues and PRs are welcome, especially validation cases against published results and
MOLSCAT output. See [docs/ROADMAP.md](docs/ROADMAP.md).

## Authorship and AI assistance

Developed and maintained by Mahdiyeh Ghasemi, with the assistance of Claude (Anthropic) as an AI coding support tool.

The scientific direction, implementation decisions, physics review, and evaluation of numerical methods are led by the project maintainer. Independent verification and validation against analytical solutions, established numerical methods, and published physical results are ongoing.

## License

MIT.


## Long-range reference matching

For a known diagonal tail, specify an explicit reference without adding it
again to the physical potential:

```python
import coldscatter as cs
from coldscatter import units as u

mu = u.reduced_mass(87.0, 87.0)
c6 = 4700.0
potential = cs.LennardJones(c6, c6 * 11.0**6 / 2)

tail = cs.LongRange({6: c6}, r_far=5000.0)
model = cs.CoupledChannels(mu, [0.0], [[potential]], long_range=tail)
```

The coefficients are in atomic units. See the
[User manual](MANUAL.md) for channel-specific tails, numerical convergence,
units, limitations, and runnable examples.


## Plotting and data export

Install optional plotting dependencies:

    pip install -e ".[examples]"

Assuming `results` contains the output of a scan:

    import coldscatter as cs
    cs.plot_scattering_length(results, x="field", save="scattering_length.png")
    cs.plot_cross_sections(results, x="field", save="cross_sections.png")
    cs.export_results_csv(results, "scan_results.csv")

See [MANUAL.md](MANUAL.md) for details.

## Citation and attribution

If you use Coldscatter in research, cite the software version and the
underlying physical methods and data sources. See
[docs/CITATION_AND_COPYRIGHT.md](docs/CITATION_AND_COPYRIGHT.md) and
[CITATION.cff](CITATION.cff).

## Publication monitoring

Run the optional monitoring script:

    python tools/monitor_usage.py

Search matches are leads for manual inspection, not verified records of
software use. See the monitoring section in [MANUAL.md](MANUAL.md).
