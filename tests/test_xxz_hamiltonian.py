# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for Xxz Hamiltonian
"""Tests for XXZ Kuramoto Hamiltonian (Kouchekian-Teodorescu S² embedding)."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.bridge.knm_hamiltonian import (
    OMEGA_N_16,
    knm_to_hamiltonian,
    knm_to_xxz_hamiltonian,
)


def _ring(n: int) -> np.ndarray:
    T = np.zeros((n, n))
    for i in range(n):
        j = (i + 1) % n
        T[i, j] = T[j, i] = 1.0
    return T


class TestXXZHamiltonian:
    def test_delta_zero_equals_xy(self):
        """Δ=0 should reproduce the standard XY Hamiltonian exactly."""
        K = 2.0 * _ring(3)
        omega = OMEGA_N_16[:3]
        H_xy = knm_to_hamiltonian(K, omega).to_matrix()
        H_xxz = knm_to_xxz_hamiltonian(K, omega, delta=0.0).to_matrix()
        assert np.allclose(H_xy, H_xxz, atol=1e-12)

    def test_delta_one_is_heisenberg(self):
        """Δ=1 should add ZZ terms (Heisenberg model)."""
        K = 1.5 * _ring(2)
        omega = OMEGA_N_16[:2]
        H_xy = knm_to_hamiltonian(K, omega).to_matrix()
        H_xxz = knm_to_xxz_hamiltonian(K, omega, delta=1.0).to_matrix()
        # Should differ by the ZZ term
        diff = H_xxz - H_xy
        assert np.linalg.norm(diff) > 0.1

    def test_heisenberg_has_su2_symmetry(self):
        """Δ=1 with uniform ω should commute with total S² (SU(2) symmetric)."""
        K = _ring(2)
        omega = np.zeros(2)  # uniform frequencies → no symmetry breaking
        H = knm_to_xxz_hamiltonian(K, omega, delta=1.0).to_matrix()
        # Total S² = (X₁+X₂)² + (Y₁+Y₂)² + (Z₁+Z₂)² = 2I + 2(XX+YY+ZZ)
        # For Heisenberg with uniform coupling, [H, S²] = 0
        from qiskit.quantum_info import SparsePauliOp

        S2 = SparsePauliOp.from_list(
            [
                ("II", 2.0),
                ("XX", 2.0),
                ("YY", 2.0),
                ("ZZ", 2.0),
            ]
        ).to_matrix()
        comm = H @ S2 - S2 @ H
        assert np.allclose(comm, 0, atol=1e-10)

    def test_spectrum_changes_with_delta(self):
        """Different Δ values should give different spectra."""
        K = 2.0 * _ring(3)
        omega = OMEGA_N_16[:3]
        E_xy = np.linalg.eigvalsh(knm_to_xxz_hamiltonian(K, omega, 0.0).to_matrix())
        E_heis = np.linalg.eigvalsh(knm_to_xxz_hamiltonian(K, omega, 1.0).to_matrix())
        assert not np.allclose(E_xy, E_heis)

    def test_gap_varies_with_delta(self):
        K = 2.0 * _ring(3)
        omega = OMEGA_N_16[:3]
        E0 = np.linalg.eigvalsh(knm_to_xxz_hamiltonian(K, omega, 0.0).to_matrix())
        E1 = np.linalg.eigvalsh(knm_to_xxz_hamiltonian(K, omega, 1.0).to_matrix())
        gap_xy = E0[1] - E0[0]
        gap_heis = E1[1] - E1[0]
        assert gap_xy != gap_heis

    def test_negative_delta_frustrated(self):
        """Negative Δ (antiferromagnetic ZZ) should still produce valid Hamiltonian."""
        K = _ring(2)
        omega = OMEGA_N_16[:2]
        H = knm_to_xxz_hamiltonian(K, omega, delta=-0.5)
        assert H.to_matrix().shape == (4, 4)

    def test_4qubit(self):
        K = _ring(4)
        omega = OMEGA_N_16[:4]
        H = knm_to_xxz_hamiltonian(K, omega, delta=0.5)
        assert H.to_matrix().shape == (16, 16)


def test_xxz_hermitian():
    K = _ring(3)
    omega = OMEGA_N_16[:3]
    H = knm_to_xxz_hamiltonian(K, omega, delta=0.5)
    mat = H.to_matrix()
    if hasattr(mat, "toarray"):
        mat = mat.toarray()
    np.testing.assert_allclose(mat, mat.conj().T, atol=1e-12)


def test_xxz_traceless():
    K = _ring(3)
    omega = OMEGA_N_16[:3]
    H = knm_to_xxz_hamiltonian(K, omega, delta=0.0)
    mat = H.to_matrix()
    if hasattr(mat, "toarray"):
        mat = mat.toarray()
    assert abs(np.trace(mat)) < 1e-8


def test_xxz_2q_spectrum():
    K = _ring(2)
    omega = OMEGA_N_16[:2]
    H = knm_to_xxz_hamiltonian(K, omega, delta=0.5)
    eigvals = np.linalg.eigvalsh(H.to_matrix())
    assert np.all(np.isfinite(eigvals))


@pytest.mark.parametrize("delta", [0.0, 0.5, 1.0, 2.0])
def test_xxz_various_deltas(delta):
    K = _ring(3)
    omega = OMEGA_N_16[:3]
    H = knm_to_xxz_hamiltonian(K, omega, delta=delta)
    assert H.num_qubits == 3
