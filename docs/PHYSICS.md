# Physics and conventions

Atomic units throughout (`hbar = m_e = e = a0 = 1`): lengths in bohr, energies in hartree,
masses in electron masses. Magnetic field is entered in gauss.

## Radial problem

For fixed total energy `E` and partial waves `l_i`, the radial wave functions `psi_i(R)` obey

    psi_i'' = sum_j W_ij(R) psi_j,
    W_ij = [ l_i(l_i+1)/R^2 + 2 mu (E_i(B) - E) ] delta_ij + 2 mu V_ij(R),

where `E_i(B)` is the channel threshold and `V_ij` the (symmetric) interaction matrix.
Channel `i` is open if `k_i^2 = 2 mu (E - E_i) > 0`.

## Propagation

**Default: Johnson log-derivative method** (B. R. Johnson, J. Comput. Phys.
13, 445 (1973)), which carries `Y = psi' psi^{-1}` directly. On a sector with an even number
of intervals of width `h` (Coldscatter uses the user grid plus its mid-points, so the
propagation step is `step/2`):

    Y_k = (1 + h Y_{k-1})^{-1} Y_{k-1} + (h/3) w_k U_k,      w_k = 4, 2, 4, ..., 4, 1
    U_k = W_k (k even),   U_k = (1 - h^2 W_k / 6)^{-1} W_k (k odd),
    Y_0 = Y_start + (h/3) W_0.

The global error is `O(h^4)`. The hard-wall condition `psi = 0` is imposed **at** `R_min`
(`Y_start -> infinity`). A second, coarser sector (`Grid(r_mid=..., step_long=...)`) starts
from the `Y` at the end of the first one.

**Alternative (`propagator="numerov"`): renormalized Numerov** (B. R. Johnson, J. Chem. Phys. 67,
4086 (1977)) propagates `G_n = phi_{n-1} phi_n^{-1}` with `phi_n = (1 - h^2 W_n/12) psi_n`:

    G_{n+1} = ( 2 + h^2 W_n (1 - h^2 W_n/12)^{-1} - G_n )^{-1}.

The hard wall is one step before `R_min`.

## Matching

**Default: single radius.** With `psi' = Y psi` at `R = r_max` and the asymptotic form below,

    [Y G - G',  Y D - D'] X = F' - Y F,       K = X (open-open block),

where the closed-channel functions are normalised to `D(r_max) = 1`, so `D'` is their
logarithmic derivative.

**Numerov: two points.** With Numerov, `psi(R_1) = M psi(R_2)` at the last two grid points.
Asymptotically

    psi = F a + G b + D d,

with open-channel Riccati-Bessel functions `F_i = k_i^{-1/2} (k_i R) j_l(k_i R)`,
`G_i = -k_i^{-1/2} (k_i R) y_l(k_i R)` and decaying closed-channel functions `D_i`.
Solving for `b` (and the closed-channel amplitudes `d`) with `a = 1` gives `K = b`,
`S = (1 + iK)(1 - iK)^{-1}`. For one channel `K = tan(delta)`.

## Observables

- Energy-dependent s-wave scattering length, free of branch cuts:
  `a(k) = (1 - S_ii) / (i k (1 + S_ii))`, which equals `-tan(delta)/k` for one open channel and
  diverges at a resonance (the older `i log(S)/(2k)` saturates at `|a| = pi/(2k)`).
  For `k|a| << 1` it reduces to the zero-energy `a = alpha - i beta` of `S_ii = exp(-2 i k a)`;
  `beta >= 0` means inelastic loss (Hutson, New J. Phys. 9, 152 (2007)).
- Cross sections (distinguishable particles, one partial wave, `pref = (2l+1) pi / k^2`):
  `sigma_el = pref |1 - S_ii|^2`, `sigma_inel = pref sum_{j != i} |S_ji|^2`,
  `sigma_tot = 2 pref (1 - Re S_ii)` for a unitary S.
  Identical bosons/fermions need the usual symmetrisation factors (not yet automated).
- van der Waals length `R_vdW = (2 mu C6)^{1/4} / 2`; mean scattering length
  `a_bar = 4 pi / Gamma(1/4)^2 R_vdW` (Gribakin and Flambaum, PRA 48, 546 (1993)).
- Resonance line shape `a(B) = a_bg (1 - Delta/(B - B0))` (Chin et al., RMP 82, 1225 (2010)).

## Validation scope

The current test suite checks selected analytical limits, numerical convergence,
and independent integration references. In particular, coupled-channel propagation
is compared with DOP853 solutions for model potentials, while separate tests check
asymptotic matching and open/closed-channel matching.

The Rb-87 literature checks verify van der Waals scales calculated from published
long-range constants. They do not yet compare a complete scattering matrix,
scattering length, or resonance position with experimental data or MOLSCAT.
The model-potential tests should therefore be understood as internal numerical
verification, not as an external benchmark of realistic collision physics.

## References

- B. R. Johnson, J. Chem. Phys. 67, 4086 (1977); J. Comput. Phys. 13, 445 (1973).
- J. M. Hutson and C. R. Le Sueur, Comput. Phys. Commun. 241, 9 (2019) (MOLSCAT).
- C. Chin, R. Grimm, P. Julienne, E. Tiesinga, Rev. Mod. Phys. 82, 1225 (2010).

## Long-range reference matching

By default, matching uses free Riccati-Bessel functions in open channels
and decaying modified spherical Bessel functions in closed channels.
For interactions with a known diagonal asymptotic tail, a channel may
instead use the reference potential

    V_ref(R) = -sum_n C_n / R^n.

The public ``LongRange`` class represents this tail. For example,

    tail = coldscatter.LongRange({6: 4700.0}, r_far=5000.0)

uses a van der Waals ``-C6/R^6`` reference. Coefficients are in atomic
units: ``C_n`` is in hartree * bohr**n, radii are in bohr, and energies
are in hartree.

The reference radial equation is integrated inward from ``r_far`` using
DOP853. Open-channel reference solutions are initialized with the
free Riccati-Bessel pair at ``r_far`` and propagated as a pair of
independent solutions. Their Wronskian normalization is verified in
regression tests against the free-wave value ``W(F,G) = -1``.

For a closed channel, the reference wavefunction ``D(R)`` satisfies

\[
D''(R) =
\left[
\frac{\ell(\ell+1)}{R^2}
+ 2\mu\left(\kappa^2 + V_{\mathrm{ref}}(R)\right)
\right]D(R),
\]

where ``kappa`` is the positive inverse decay length. At ``r_far``, the
initial logarithmic derivative is taken from the decaying free modified
spherical-Bessel reference for ``kappa > 0``. At the zero-energy
threshold, its limiting value is ``-ell/r_far``.

The closed-channel implementation propagates the pair ``(D, D')``
inward and rescales it piecewise to control its numerical dynamic range.
The accumulated scale factors are retained when evaluating the signed
ratio ``D(r1)/D(r2)``. This permits propagation across nodes without
the logarithmic-derivative poles that occur when ``D(R) = 0``.
The denominator radius ``r2`` must not itself be a node.

The boundary condition is initialized at a finite ``r_far``. Its
adequacy must therefore be checked by increasing ``r_far`` and verifying
stability of the matching quantities and final observables.

Pass ``long_range`` to ``CoupledChannels``. A single ``LongRange`` object
is broadcast to all channels; alternatively, pass a sequence containing
one ``LongRange`` or ``None`` per channel. ``None`` selects the original
free reference for that channel. The default ``long_range=None`` preserves
the original matching behavior.

**Important:** ``LongRange`` defines a reference function only. It is not
added to the physical potential supplied to ``CoupledChannels``. The
physical potential should already contain the interaction being modeled.
For a Lennard-Jones potential, the asymptotic reference is commonly
``LongRange({6: potential.c6})``; the repulsive ``C12/R^12`` term is not
part of that leading asymptotic reference.

This implementation supports diagonal channel tails. Off-diagonal
long-range couplings are not included. The finite starting radius
``r_far`` and matching radius must be checked for convergence in
representative calculations. Increasing ``r_far`` and ``r_max`` should
not materially change the observables at the requested precision.
