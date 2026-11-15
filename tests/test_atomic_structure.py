
import numpy as np

from coldscatter import units as u


def test_rb87_hyperfine_basis_dimension():
    """The Rb-87 5S1/2 uncoupled basis has eight states."""
    S = 0.5
    I = 1.5

    basis = [
        (mS, mI)
        for mS in (S, -S)
        for mI in (I, I - 1, -I + 1, -I)
    ]

    assert len(basis) == 8
    assert len(set(basis)) == 8


def test_rb87_hyperfine_zero_field_eigenvalues():
    """Check the analytic hyperfine eigenvalues for I=3/2, S=1/2."""
    nu_hfs = 6.834682610904324e9  # Hz
    hartree_hz = 6.579683920502e15  # Hz per Hartree

    A_hf = (nu_hfs / 2.0) / hartree_hz

    energy_F1 = -5.0 * A_hf / 4.0
    energy_F2 = 3.0 * A_hf / 4.0

    eigenvalues = np.array(
        [energy_F1] * 3 + [energy_F2] * 5
    )

    assert eigenvalues.shape == (8,)
    assert np.count_nonzero(np.isclose(eigenvalues, energy_F1)) == 3
    assert np.count_nonzero(np.isclose(eigenvalues, energy_F2)) == 5
    assert np.isclose(energy_F2 - energy_F1, 2.0 * A_hf)
    assert np.isclose(np.sum(eigenvalues), 0.0, atol=1e-18)

def test_rb87_hyperfine_hamiltonian_spectrum():
    """Numerical Hamiltonian spectrum must match analytic hyperfine levels."""
    from coldscatter.atomic import hyperfine_hamiltonian

    nu_hfs = 6.834682610904324e9  # Hz
    hartree_hz = 6.579683920502e15  # Hz per Hartree
    A_hf = (nu_hfs / 2.0) / hartree_hz

    H = hyperfine_hamiltonian(I=1.5, J=0.5, A_hf=A_hf)
    numerical = np.linalg.eigvalsh(H)

    expected = np.sort(
        np.array([-5.0 * A_hf / 4.0] * 3 + [3.0 * A_hf / 4.0] * 5)
    )

    assert H.shape == (8, 8)
    assert np.allclose(H, H.conj().T)
    assert np.allclose(numerical, expected, rtol=1e-12, atol=1e-18)
    assert np.isclose(np.trace(H), 0.0, atol=1e-18)

def test_spin_matrices_commutation_relations():
    """Check angular-momentum commutation relations for spin 1/2 and 3/2."""
    from coldscatter.atomic import spin_matrices

    for j in (0.5, 1.5):
        Jx, Jy, Jz, Jp, Jm = spin_matrices(j)

        assert np.allclose(
            Jx @ Jy - Jy @ Jx,
            1j * Jz
        )

        assert np.allclose(
            Jp @ Jm - Jm @ Jp,
            2.0 * Jz
        )
def test_rb87_zeeman_hamiltonian():
    from coldscatter.atomic import hyperfine_hamiltonian

    nu_hfs = 6.834682610904324e9
    hartree_hz = 6.579683920502e15
    A_hf = (nu_hfs / 2.0) / hartree_hz

    H0 = hyperfine_hamiltonian(I=1.5, J=0.5, A_hf=A_hf)
    H10 = hyperfine_hamiltonian(
        I=1.5, J=0.5, A_hf=A_hf, B_gauss=10.0
    )

    assert H10.shape == (8, 8)
    assert np.allclose(H10, H10.conj().T)
    assert np.allclose(H0, H0.conj().T)
    zeeman_difference = H10 - H0
    assert np.max(np.abs(zeeman_difference)) > 1e-12
    # The Zeeman term must be linear in magnetic field.
    H20 = hyperfine_hamiltonian(
        I=1.5, J=0.5, A_hf=A_hf, B_gauss=20.0
    )

    assert np.allclose(H20 - H0, 2.0 * (H10 - H0))
    assert np.allclose(
    H20 - H0,
    2.0 * (H10 - H0),
    rtol=1e-12,
    atol=1e-20,
)


def test_rb87_breit_rabi_spectrum():
    """Compare the numerical Zeeman spectrum with the Breit-Rabi formula."""
    from coldscatter.atomic import hyperfine_hamiltonian

    I = 1.5
    J = 0.5
    B = 100.0  # Gauss

    nu_hfs = 6.834682610904324e9  # Hz
    hartree_hz = 6.579683920502e15  # Hz per Hartree
    A_hf = (nu_hfs / 2.0) / hartree_hz
    delta_E = 2.0 * A_hf

    g_J = 2.002331070
    g_I = -0.0009951414
    mu_B = u.MU_B_HARTREE_PER_GAUSS

    H = hyperfine_hamiltonian(
        I=I,
        J=J,
        A_hf=A_hf,
        B_gauss=B,
        g_J=g_J,
        g_I=g_I,
    )

    numerical = np.linalg.eigvalsh(H)

    x = (g_J - g_I) * mu_B * B / delta_E

    expected = []

    for mF in (-2, -1, 0, 1, 2):
        common = (
            -delta_E / (2.0 * (2.0 * I + 1.0))
            + g_I * mu_B * mF * B
        )

        radical = np.sqrt(
            1.0
            + (4.0 * mF / (2.0 * I + 1.0)) * x
            + x**2
        )

        upper = common + 0.5 * delta_E * radical
        expected.append(upper)

        if abs(mF) < I + J:
            lower = common - 0.5 * delta_E * radical
            expected.append(lower)

    expected = np.sort(np.array(expected))

    assert numerical.shape == (8,)
    assert expected.shape == (8,)
    assert np.allclose(
        numerical,
        expected,
        rtol=1e-11,
        atol=1e-18,
    )

def test_rb87_total_projection_conservation():
    """A z-directed magnetic field conserves total projection M_F."""
    from coldscatter.atomic import hyperfine_hamiltonian

    I = 1.5
    J = 0.5
    nu_hfs = 6.834682610904324e9
    hartree_hz = 6.579683920502e15
    A_hf = (nu_hfs / 2.0) / hartree_hz

    basis = [
        (mJ, mI)
        for mJ in (J, -J)
        for mI in (I, I - 1, -I + 1, -I)
    ]

    M_F = np.array([mJ + mI for mJ, mI in basis])

    H = hyperfine_hamiltonian(
        I=I,
        J=J,
        A_hf=A_hf,
        B_gauss=100.0,
    )

    for i in range(len(basis)):
        for j in range(len(basis)):
            if not np.isclose(M_F[i], M_F[j]):
                assert np.isclose(H[i, j], 0.0, atol=1e-18)
def test_rb87_hyperfine_basis_blocks():
    """Check the M_F block structure of the Rb-87 ground-state basis."""
    from coldscatter.atomic import hyperfine_basis_blocks

    blocks = hyperfine_basis_blocks(I=1.5, J=0.5)

    expected_MF = [-2.0, -1.0, 0.0, 1.0, 2.0]

    assert sorted(blocks) == expected_MF
    assert [len(blocks[mf]) for mf in expected_MF] == [1, 2, 2, 2, 1]

    all_indices = sorted(
        index
        for indices in blocks.values()
        for index in indices
    )

    assert all_indices == list(range(8))

def test_rb87_hamiltonian_block_extraction():
    """Check extraction and spectra of independent M_F Hamiltonian blocks."""
    from coldscatter.atomic import (
        hyperfine_hamiltonian,
        hyperfine_basis_blocks,
        extract_hamiltonian_blocks,
    )

    I = 1.5
    J = 0.5

    nu_hfs = 6.834682610904324e9
    hartree_hz = 6.579683920502e15
    A_hf = (nu_hfs / 2.0) / hartree_hz

    H = hyperfine_hamiltonian(
        I=I,
        J=J,
        A_hf=A_hf,
        B_gauss=100.0,
    )

    basis_blocks = hyperfine_basis_blocks(I=I, J=J)
    matrix_blocks = extract_hamiltonian_blocks(H, basis_blocks)

    assert sorted(matrix_blocks) == [-2.0, -1.0, 0.0, 1.0, 2.0]
    assert [matrix_blocks[mf].shape[0] for mf in sorted(matrix_blocks)] == [
        1, 2, 2, 2, 1
    ]

    block_eigenvalues = np.sort(
        np.concatenate([
            np.linalg.eigvalsh(block)
            for block in matrix_blocks.values()
        ])
    )

    full_eigenvalues = np.linalg.eigvalsh(H)

    assert np.allclose(
        block_eigenvalues,
        full_eigenvalues,
        rtol=1e-12,
        atol=1e-18,
    )