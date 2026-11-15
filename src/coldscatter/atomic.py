
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
):
    """Construct the hyperfine + Zeeman Hamiltonian.

    Basis: |m_J, m_I>
    A_hf: Hartree
    B_gauss: Gauss
    g_J and g_I: dimensionless, in Bohr-magneton convention.
    Returns: Hamiltonian in Hartree.
    """
    Ix, Iy, Iz, _, _ = spin_matrices(I)
    Jx, Jy, Jz, _, _ = spin_matrices(J)

    H_hf = A_hf * (
        np.kron(Jx, Ix)
        + np.kron(Jy, Iy)
        + np.kron(Jz, Iz)
    )

    identity_I = np.eye(Ix.shape[0])
    identity_J = np.eye(Jx.shape[0])

    H_zeeman = units.MU_B_HARTREE_PER_GAUSS * B_gauss * (
        g_J * np.kron(Jz, identity_I)
        + g_I * np.kron(identity_J, Iz)
    )

    return H_hf + H_zeeman
