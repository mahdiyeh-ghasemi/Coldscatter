
"""Atomic internal structure for alkali-metal atoms."""

import numpy as np

from . import units


def spin_matrices(j):
    """Return dimensionless angular-momentum matrices Jx, Jy, Jz, J+, J-."""
    if j < 0 or not np.isclose(2 * j, round(2 * j)):
        raise ValueError("j must be a non-negative integer or half-integer.")

    m_values = np.arange(j, -j - 1, -1, dtype=float)
    dim = len(m_values)

    Jz = np.diag(m_values)
    Jp = np.zeros((dim, dim), dtype=complex)

    for col, m in enumerate(m_values):
        target = m + 1
        rows = np.where(np.isclose(m_values, target))[0]
        if len(rows):
            row = rows[0]
            Jp[row, col] = np.sqrt(j * (j + 1) - m * (m + 1))

    Jm = Jp.conj().T
    Jx = 0.5 * (Jp + Jm)
    Jy = (Jp - Jm) / (2.0j)

    return Jx, Jy, Jz, Jp, Jm



def hyperfine_basis_blocks(I, J):
    """Group uncoupled basis states by conserved total projection M_F.

    Basis ordering follows |m_J, m_I>, with m_J as the outer index.
    Returns a dictionary mapping each M_F to its basis indices.
    """
    _, _, Jz, _, _ = spin_matrices(J)
    _, _, Iz, _, _ = spin_matrices(I)

    m_J_values = np.diag(Jz)
    m_I_values = np.diag(Iz)

    blocks = {}

    index = 0
    for m_J in m_J_values:
        for m_I in m_I_values:
            M_F = float(m_J + m_I)
            blocks.setdefault(M_F, []).append(index)
            index += 1

    return blocks

def extract_hamiltonian_blocks(H, blocks):
    """Extract independent matrix blocks using basis-index groups.

    Parameters
    ----------
    H : array_like
        Square Hamiltonian matrix.
    blocks : dict
        Mapping from block labels (e.g. M_F) to basis indices.

    Returns
    -------
    dict
        Mapping from block labels to corresponding square submatrices.
    """
    H = np.asarray(H)

    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError("H must be a square matrix.")

    extracted = {}

    for label, indices in blocks.items():
        indices = np.asarray(indices, dtype=int)

        if np.any(indices < 0) or np.any(indices >= H.shape[0]):
            raise ValueError("Block indices are outside the Hamiltonian dimensions.")

        extracted[label] = H[np.ix_(indices, indices)]

    return extracted

def hyperfine_hamiltonian(
    I,
    J,
    A_hf,
    B_gauss=0.0,
    g_J=2.002331070,
    g_I=-0.0009951414,
    B_hf=0.0,
):
    r"""Construct the hyperfine + Zeeman Hamiltonian.

    .. math::
        H = A\,\mathbf{I}\cdot\mathbf{J}
            + B_{hf}\frac{3(\mathbf{I}\cdot\mathbf{J})^2 + \tfrac32\mathbf{I}\cdot\mathbf{J}
              - I(I+1)J(J+1)}{2I(2I-1)J(2J-1)}
            + \mu_B B\,(g_J J_z + g_I I_z)

    Basis: |m_J, m_I> (m_J outer).  ``A_hf`` and ``B_hf``: hartree;
    ``B_gauss``: gauss; ``g_J``, ``g_I``: dimensionless, Bohr-magneton
    convention (Arimondo et al., Rev. Mod. Phys. 49, 31 (1977)).

    .. warning::
       The default ``g_J`` and ``g_I`` are the **87Rb** ground-state values.
       For other atoms pass them explicitly, or use :func:`alkali_hamiltonian`
       with an entry of :data:`ALKALI_GROUND_STATES`.

    The electric-quadrupole term ``B_hf`` only exists for ``I >= 1`` and
    ``J >= 1`` (it is zero for alkali ground states, J = 1/2).
    Returns: Hamiltonian in hartree.
    """
    Ix, Iy, Iz, _, _ = spin_matrices(I)
    Jx, Jy, Jz, _, _ = spin_matrices(J)

    IJ = np.kron(Jx, Ix) + np.kron(Jy, Iy) + np.kron(Jz, Iz)
    H_hf = A_hf * IJ

    if B_hf != 0.0:
        if I < 1 or J < 1:
            raise ValueError("the quadrupole term B_hf needs I >= 1 and J >= 1")
        dim = IJ.shape[0]
        H_hf = H_hf + B_hf * (
            3.0 * IJ @ IJ + 1.5 * IJ - I * (I + 1) * J * (J + 1) * np.eye(dim)
        ) / (2.0 * I * (2 * I - 1) * J * (2 * J - 1))

    identity_I = np.eye(Ix.shape[0])
    identity_J = np.eye(Jx.shape[0])

    H_zeeman = units.MU_B_HARTREE_PER_GAUSS * B_gauss * (
        g_J * np.kron(Jz, identity_I)
        + g_I * np.kron(identity_J, Iz)
    )

    return H_hf + H_zeeman


_MHZ_IN_HARTREE = 1e6 / 6.579683920502e15

#: Ground-state (J = 1/2) parameters of common alkali atoms.  ``A_hf`` in
#: hartree, g-factors in the Bohr-magneton convention.  Sources: E. Arimondo,
#: M. Inguscio, P. Violino, Rev. Mod. Phys. 49, 31 (1977); D. A. Steck,
#: "Rubidium 87 / Rubidium 85 / Cesium / Sodium D Line Data".
ALKALI_GROUND_STATES = {
    "6Li":   dict(I=1.0, J=0.5, A_hf=152.1368407 * _MHZ_IN_HARTREE,
                  g_J=2.0023010, g_I=-0.0004476540),
    "23Na":  dict(I=1.5, J=0.5, A_hf=885.81306440 * _MHZ_IN_HARTREE,
                  g_J=2.00229600, g_I=-0.00080461080),
    "40K":   dict(I=4.0, J=0.5, A_hf=-285.7308 * _MHZ_IN_HARTREE,
                  g_J=2.00229421, g_I=0.000176490),
    "85Rb":  dict(I=2.5, J=0.5, A_hf=1011.910813 * _MHZ_IN_HARTREE,
                  g_J=2.00233113, g_I=-0.00029364000),
    "87Rb":  dict(I=1.5, J=0.5, A_hf=3417.341305452145 * _MHZ_IN_HARTREE,
                  g_J=2.00233113, g_I=-0.0009951414),
    "133Cs": dict(I=3.5, J=0.5, A_hf=2298.1579425 * _MHZ_IN_HARTREE,
                  g_J=2.00254032, g_I=-0.00039885395),
}


def alkali_hamiltonian(atom: str, B_gauss=0.0):
    """Hyperfine + Zeeman Hamiltonian of a ground-state alkali atom by name.

    ``atom`` is a key of :data:`ALKALI_GROUND_STATES` (e.g. ``"87Rb"``).
    """
    try:
        p = ALKALI_GROUND_STATES[atom]
    except KeyError:
        raise ValueError(f"unknown atom {atom!r}; known: {sorted(ALKALI_GROUND_STATES)}") from None
    return hyperfine_hamiltonian(B_gauss=B_gauss, **p)
