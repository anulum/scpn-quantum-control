"""Cross-module consistency: verify pipeline paths produce compatible results."""

from __future__ import annotations

import numpy as np
from qiskit.quantum_info import Statevector

from scpn_quantum_control.bridge import OMEGA_N_16, build_knm_paper27, knm_to_hamiltonian
from scpn_quantum_control.hardware.classical import (
    classical_exact_diag,
    classical_kuramoto_reference,
)
from scpn_quantum_control.phase.xy_kuramoto import QuantumKuramotoSolver


def test_kuramoto_solver_vs_classical_ground_energy():
    """QuantumKuramotoSolver Hamiltonian matches classical_exact_diag ground energy."""
    K = build_knm_paper27(L=4)
    omega = OMEGA_N_16[:4]

    solver = QuantumKuramotoSolver(4, K, omega)
    H_solver = solver.build_hamiltonian()

    H_bridge = knm_to_hamiltonian(K, omega)

    # Both should produce identical Hamiltonians
    mat_solver = H_solver.to_matrix()
    mat_bridge = H_bridge.to_matrix()
    if hasattr(mat_solver, "toarray"):
        mat_solver = mat_solver.toarray()
    if hasattr(mat_bridge, "toarray"):
        mat_bridge = mat_bridge.toarray()

    assert np.allclose(mat_solver, mat_bridge, atol=1e-10)


def test_classical_diag_vs_numpy_eigvalsh():
    """classical_exact_diag matches direct numpy eigvalsh."""
    K = build_knm_paper27(L=4)
    omega = OMEGA_N_16[:4]

    result = classical_exact_diag(n_osc=4, K=K, omega=omega)
    E0_classical = result["ground_energy"]

    H = knm_to_hamiltonian(K, omega)
    mat = H.to_matrix()
    if hasattr(mat, "toarray"):
        mat = mat.toarray()
    E0_numpy = np.linalg.eigvalsh(mat)[0]

    assert abs(E0_classical - E0_numpy) < 1e-10


def test_classical_kuramoto_R_positive():
    """Classical Kuramoto integration produces R in [0, 1]."""
    result = classical_kuramoto_reference(n_osc=4, t_max=1.0, dt=0.01)
    R_final = result["R"][-1]
    assert 0.0 <= R_final <= 1.0 + 1e-10


def test_energy_expectation_ground_state():
    """Ground state of H has energy matching the lowest eigenvalue."""
    K = build_knm_paper27(L=4)
    omega = OMEGA_N_16[:4]

    H = knm_to_hamiltonian(K, omega)
    mat = H.to_matrix()
    if hasattr(mat, "toarray"):
        mat = mat.toarray()

    eigvals, eigvecs = np.linalg.eigh(mat)
    ground_sv = Statevector(eigvecs[:, 0].copy())

    solver = QuantumKuramotoSolver(4, K, omega)
    E = solver.energy_expectation(ground_sv)
    assert abs(E - eigvals[0]) < 1e-8


def test_hamiltonian_commutes_with_total_z_parity():
    """XY Hamiltonian conserves total Z parity (XX+YY preserves excitation number mod 2)."""
    K = build_knm_paper27(L=4)
    omega = OMEGA_N_16[:4]

    H = knm_to_hamiltonian(K, omega)
    mat = H.to_matrix()
    if hasattr(mat, "toarray"):
        mat = mat.toarray()

    # Build parity operator: tensor product of Z's
    from functools import reduce

    Z = np.array([[1, 0], [0, -1]])
    parity = reduce(np.kron, [Z] * 4)

    commutator = mat @ parity - parity @ mat
    assert np.allclose(commutator, 0, atol=1e-10), "H should commute with Z-parity"
