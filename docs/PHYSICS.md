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

Johnson's renormalized Numerov method (J. Chem. Phys. 67, 4086 (1977)) propagates the
ratio `G_n = phi_{n-1} phi_n^{-1}` with `phi_n = (1 - h^2 W_n/12) psi_n`:

    G_{n+1} = ( 2 + h^2 W_n (1 - h^2 W_n/12)^{-1} - G_n )^{-1}.

It is fourth-order accurate in the step `h`, and stable for closed channels because only
bounded ratios are stored. The hard-wall boundary condition `psi = 0` is imposed one step
before `R_min`.

## Matching

At the last two grid points, `psi(R_1) = M psi(R_2)`. Asymptotically

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
