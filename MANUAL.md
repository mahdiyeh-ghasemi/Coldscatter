# Coldscatter User Manual

**Version:** 0.1.1
**Units:** atomic units unless explicitly stated

## 1. Scope

Coldscatter solves radial coupled-channel scattering problems and extracts
open-channel K and S matrices, energy-dependent scattering lengths, and
partial-wave cross sections. It includes a Johnson log-derivative propagator
(default), a renormalized Numerov propagator (legacy), and asymptotic matching.

This manual describes the current public API and the optional long-range
reference matching feature. The package is under active development;
internal numerical verification is not a substitute for comparison with
experimental data or independent established scattering packages.

## 2. Installation

From a clone of the repository:

```bash
python -m pip install -e ".[examples]"
```

For the core package, NumPy and SciPy are required. Matplotlib is used by
the plotting examples. Run the test suite with:

```bash
python -m pytest -q
```

## 3. Units and conventions

The solver uses atomic units:

| Quantity | Unit |
|---|---|
| Length and radial grid | bohr |
| Energy and thresholds | hartree |
| Reduced mass | electron masses |
| Magnetic field | gauss |
| Potential | hartree |

Use `coldscatter.units` for conversions and reduced masses. In a
long-range expansion `-C_n/R^n`, the coefficient `C_n` has units
hartree * bohr**n.

## 4. Define a single-channel problem

```python
import coldscatter as cs
from coldscatter import units as u

mu = u.reduced_mass(87.0, 87.0)
c6 = 4700.0
potential = cs.LennardJones(c6, c6 * 11.0**6 / 2)

model = cs.CoupledChannels(
    mu=mu,
    thresholds=[0.0],
    potentials=[[potential]],
)

grid = cs.Grid(r_min=4.0, r_max=1500.0, step=0.02)
result = model.solve(u.from_kelvin(1e-7), grid)

print(result.K)
print(result.S)
print(result.scattering_length())
print(result.cross_sections())
```

The collision energy is measured from the threshold of the entrance
channel. The radial grid must resolve the shortest relevant wavelength
and extend sufficiently far into the asymptotic region.

### 4.1 Propagator and grid (0.1.1)

`solve`, `scan` and `scan_field` take `propagator="log-derivative"`
(default) or `propagator="numerov"` (legacy). The default carries the
log-derivative matrix and matches at the single radius `r_max`; it is
fourth-order in the step and does not lose precision near threshold.
The hard wall (`psi = 0`) sits **at** `r_min` with the default and one step
before `r_min` with `"numerov"`; place `r_min` deep in the repulsive wall
so that this does not matter.

For long ranges use a two-sector grid: a fine step through the
short-range well and a coarse step outside it (like MOLSCAT's RMID/DRL):

```python
grid = cs.Grid(r_min=4.0, r_max=1500.0, step=0.02, r_mid=40.0, step_long=0.5)
```

`Grid.r` returns all points; `Grid.sectors` lists `(r_start, step,
n_intervals)`. Two-sector grids need the log-derivative propagator.

A warning is raised when a potential or coupling has not decayed at
`r_max` (decays slower than about `R^-3`, e.g. a constant coupling):
free-reference matching assumes it is negligible beyond `r_max`.

### 4.2 Identical particles

`result.cross_sections(identical="bosons")` (or `"fermions"`) applies the
symmetrisation factor for two identical particles in the same internal
state: 2 for allowed partial waves (even `l` for bosons, odd for fermions)
and 0 for forbidden ones. The default `None` means distinguishable.

### 4.3 Atomic structure

`coldscatter.atomic.alkali_hamiltonian("87Rb", B_gauss)` builds the
hyperfine + Zeeman Hamiltonian from `ALKALI_GROUND_STATES` (6Li, 23Na,
40K, 85Rb, 87Rb, 133Cs). `hyperfine_hamiltonian(...)` defaults to 87Rb
g-factors, so pass `g_J`, `g_I` explicitly for other atoms; `B_hf` adds the
electric-quadrupole term for `I, J >= 1`. A two-atom hyperfine/Zeeman
channel basis for the scattering solver is not yet built in (roadmap v0.3).

## 5. Long-range reference matching

### 5.1 Define the tail

The `LongRange` class represents

    V_ref(R) = -C_6/R^6 - C_8/R^8 - ...

For example:

```python
tail = cs.LongRange(
    {6: 4700.0, 8: 1.0e6},
    r_far=5000.0,
)
```

The coefficient dictionary maps integer powers to their coefficients.
Powers must be integers greater than or equal to 3. Coefficients must
be finite real numbers. `r_far` is the outer starting radius in bohr.
The numerical integration tolerances can be adjusted with `rtol` and
`atol`.

### 5.2 Attach the reference to a model

```python
model = cs.CoupledChannels(
    mu=mu,
    thresholds=[0.0],
    potentials=[[potential]],
    long_range=cs.LongRange({6: c6}, r_far=5000.0),
)
```

A single `LongRange` object is applied to all channels. To assign
different tails to different channels, supply one entry per channel:

```python
model = cs.CoupledChannels(
    mu=mu,
    thresholds=[0.0, 0.01],
    potentials=[
        [potential_open, coupling],
        [None, potential_closed],
    ],
    long_range=[
        cs.LongRange({6: c6_open}),
        cs.LongRange({6: c6_closed}),
    ],
)
```

An entry of `None` retains free-reference matching in that channel.
Omitting `long_range` entirely preserves the original behavior.

### 5.3 Reference tail is not an extra physical potential

The reference tail is used only to construct the asymptotic matching
functions. It is not added to the potential matrix. If the physical
Lennard-Jones potential already contains `-C6/R^6`, use the same `C6`
in the reference object; do not add another `-C6/R^6` term to the
physical potential.

### 5.4 Numerical convergence

The reference functions are initialized at `r_far` and integrated inward
with DOP853. Open-channel reference solutions start from the free
Riccati-Bessel pair. Closed-channel reference wavefunctions are propagated
as the second-order pair `(D, D')`. Piecewise rescaling controls their
dynamic range while retaining the scale needed to recover the signed
ratio `D(r1)/D(r2)` (legacy path) or the log-derivative `D'/D` at `r_max`
(default path). The matching radius must not coincide with a node of `D`.
Versions <= 0.1.0 used `2 mu (kappa^2 + V)` instead of `kappa^2 + 2 mu V`
in this closed-channel integration; results with closed-channel
`LongRange` tails from those versions should be recomputed.

With the default propagator the matching uses the single radius `r_max`
(the legacy Numerov path uses the last two grid points). Check the
stability of observables by varying both:

- `Grid.r_max`, which changes the matching location;
- `LongRange.r_far`, which changes the initialization point of the
  reference solution.

A practical convergence study should compare K or S matrix elements
and the observable of interest at multiple values of both radii. A
stable result for one model and energy does not establish convergence
for every potential, partial wave, or threshold configuration. Ratios
that underflow below floating-point range may be returned as zero;
ratios that exceed the representable range raise an error.

## 6. Coupled-channel models and field scans

`CoupledChannels` accepts a square upper-triangular potential definition.
The lower triangle must contain `None`; the potential matrix is
symmetrized internally. Thresholds may be a fixed sequence or a
callable of magnetic field in gauss.

```python
result = model.solve(
    collision_energy=energy,
    grid=grid,
    B=field_gauss,
    entrance=0,
)

results = model.scan(
    fields,
    collision_energy=energy,
    grid=grid,
    entrance=0,
)
```

`scan_field` returns the complex s-wave entrance-channel scattering
length over the supplied magnetic-field array.

## 7. Result objects

A `ScatteringResult` contains:

- `K`: real open-channel K matrix;
- `S`: open-channel S matrix;
- `open_channels`: global indices of open channels;
- `k_open`: wave numbers of open channels;
- `k_asymmetry`: relative asymmetry of the unsymmetrized K matrix;
- `scattering_length()`: energy-dependent complex s-wave scattering
  length for the entrance channel;
- `cross_sections(identical=None)`: elastic, inelastic, loss, and total
  cross sections (see 4.2 for identical particles).

The scattering length is energy-dependent. For a reliable zero-energy
interpretation, use a sufficiently low collision energy and check the
condition `|k a| << 1`.

## 8. Examples

The repository examples include:

| File | Purpose |
|---|---|
| `examples/01_vdw_universality.py` | van der Waals universality and short-range phase |
| `examples/02_feshbach_scan.py` | two-channel magnetic-field scan and resonance fit |
| `examples/03_long_range_matching.py` | comparison of free and C6 reference matching |

Run an example from the repository root, for example:

```bash
python examples/03_long_range_matching.py
```

## 9. Current scope and limitations

- The first long-range implementation supports diagonal channel tails.
  Off-diagonal long-range couplings are not supported.
- The tail is an explicitly supplied reference model; coefficients are
  not inferred automatically from arbitrary potential callables.
- The finite `r_far` initialization requires convergence checks.
- The grid is uniform or two uniform sectors (`r_mid`, `step_long`);
  fully adaptive step control is not implemented.
- The channel basis and physical interaction matrix are supplied by
  the user; automatic construction of general atomic and molecular
  channel bases is outside this API.
- Identical-particle symmetrization factors are not automated.
- Feshbach line-shape fitting assumes a constant background and a
  single resonance.

## 10. Validation

Run:

```bash
python -m pytest -q
```

The test suite includes analytical limits, independent numerical
integration references, matching checks, threshold behavior, and
unitarity checks. The long-range tests additionally check zero-tail
consistency, open-reference Wronskian preservation, closed-channel
ratios against direct second-order integration (including propagation
across a wavefunction node), closed-channel stability against changes
in `r_far`, channel-specific configuration, and matching-radius
stability for a representative single-channel model.

These checks are numerical verification for selected models. They do
not establish agreement with experimental scattering data or constitute
a complete external benchmark against MOLSCAT.

## 11. Troubleshooting

**`r_far` must exceed the matching radius**

Increase `LongRange.r_far` beyond `Grid.r_max`. The reference is
integrated inward from that outer radius.

**Matching matrix is singular**

Check the energy, open-channel definition, grid, channel thresholds,
and potential setup. Try a convergence study with a different matching
radius and inspect the raw propagation behavior.

**Results change with `r_max` or `r_far`**

The asymptotic region may not yet be reached, or the chosen reference
tail may not adequately represent the physical long-range interaction.
Check the tail coefficients and repeat the calculation at larger radii.

**A channel is unexpectedly closed**

A channel is open only when `2 * mu * (E - threshold)` is positive.
At exact threshold it is treated as closed.

## 12. Citation and development

Please cite the software version used in computational work and report
the model, units, grid, channel thresholds, potential parameters,
long-range coefficients, and convergence checks. The repository
contains the current license and development roadmap. A versioned
release and archival DOI should be used when available.


## 13. Plotting and CSV export

Plotting requires the optional Matplotlib dependency. Install it with:

    pip install -e ".[examples]"

Example, assuming results contains the output of a scan:

    import coldscatter as cs
    cs.plot_scattering_length(results, x="field", save="scattering_length.png")
    cs.plot_cross_sections(results, x="field", save="cross_sections.png")
    cs.export_results_csv(results, "scan_results.csv")

Complex matrices in CSV output are represented as JSON objects with separate real and imaginary arrays. Matplotlib is imported only when plotting is used.

## 14. Citation, copyright, and third-party software

Coldscatter is distributed under the MIT License. If you use it in research, cite the software version using CITATION.cff and cite the underlying physical methods, published data, and external sources independently.

Report the model, units, grid, channel thresholds, potential parameters, long-range coefficients, and convergence checks. Preserve applicable third-party notices when redistributing the software.

See docs/CITATION_AND_COPYRIGHT.md and THIRD_PARTY_NOTICES.md. Use a versioned release and archival DOI when available. Do not assume a DOI exists before one has been assigned.

## 15. Publication and repository activity monitoring

Run the optional monitoring script from the repository root:

    python tools/monitor_usage.py

The first run establishes a baseline. Existing search matches are recorded without generating new-publication alerts. Later runs report newly appearing search matches.

Optional environment variables: COLDSCATTER_GITHUB_REPO, COLDSCATTER_GITHUB_TOKEN, GITHUB_TOKEN, COLDSCATTER_STATE, COLDSCATTER_SMTP_HOST, COLDSCATTER_SMTP_PORT, COLDSCATTER_SMTP_USER, COLDSCATTER_SMTP_PASSWORD, COLDSCATTER_ALERT_FROM, and COLDSCATTER_ALERT_TO.

Keep credentials in environment variables or a secure secret store. Never commit passwords, tokens, or the local monitoring state file.

Publication search matches are leads, not proof that a paper used Coldscatter. They cannot establish omitted citation, infringement, or misconduct. GitHub traffic statistics are aggregate and do not identify individual users. The script does not run automatically on a schedule.
