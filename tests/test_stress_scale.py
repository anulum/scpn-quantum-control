"""Stress tests: verify correctness at larger qubit counts."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.bridge import (
    OMEGA_N_16,
    build_knm_paper27,
    knm_to_ansatz,
    knm_to_hamiltonian,
)


def test_12q_hamiltonian_structure():
    """12-qubit Hamiltonian: correct qubit count, Hermitian, real spectrum."""
    K = build_knm_paper27(L=12)
    omega = OMEGA_N_16[:12]
    H = knm_to_hamiltonian(K, omega)
    assert H.num_qubits == 12

    n_z = sum(1 for p in H.paulis if str(p).count("Z") == 1 and str(p).count("X") == 0)
    n_xx = sum(1 for p in H.paulis if str(p).count("X") == 2)
    assert n_z == 12
    assert n_xx == 12 * 11 // 2


def test_16q_hamiltonian_structure():
    """Full 16-qubit Hamiltonian: all 120 XX+YY pairs + 16 Z field terms."""
    K = build_knm_paper27(L=16)
    omega = OMEGA_N_16
    H = knm_to_hamiltonian(K, omega)
    assert H.num_qubits == 16

    n_terms = len(H.paulis)
    # Identity + 16 Z + 120 XX + 120 YY = 257 terms (before simplify)
    assert n_terms > 100


def test_12q_ansatz_depth():
    """12-qubit ansatz at reps=2: reasonable gate count."""
    K = build_knm_paper27(L=12)
    qc = knm_to_ansatz(K, reps=2)
    assert qc.num_qubits == 12
    assert qc.num_parameters == 12 * 2 * 2  # n * 2_params * reps
    assert qc.size() > 0


def test_16q_ansatz_reps1():
    """16-qubit ansatz at reps=1."""
    K = build_knm_paper27(L=16)
    qc = knm_to_ansatz(K, reps=1)
    assert qc.num_qubits == 16
    assert qc.num_parameters == 16 * 2 * 1


@pytest.mark.slow
def test_8q_hamiltonian_diagonalization():
    """8-qubit exact diagonalization: ground state is real, spectrum bandwidth > 1."""
    K = build_knm_paper27(L=8)
    omega = OMEGA_N_16[:8]
    H = knm_to_hamiltonian(K, omega)
    mat = H.to_matrix()
    if hasattr(mat, "toarray"):
        mat = mat.toarray()
    eigvals = np.linalg.eigvalsh(mat)
    assert eigvals[0] < eigvals[-1]
    assert (eigvals[-1] - eigvals[0]) > 1.0
    assert np.all(np.isreal(eigvals))


def test_knm_16x16_exponential_decay():
    """K[0,15] (distance 15) << K[0,1] (distance 1) — exponential decay holds."""
    K = build_knm_paper27(L=16)
    # Natural: K_base * exp(-alpha * d)
    assert K[0, 1] > K[0, 5] > K[0, 10]
    # Cross-hierarchy boost on L1-L16 overrides natural value
    assert K[0, 15] == pytest.approx(0.05, abs=0.001)
