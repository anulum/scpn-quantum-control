"""Integration tests: full pipeline from Knm matrix to quantum observables."""

from __future__ import annotations

import numpy as np
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

from scpn_quantum_control.bridge import (
    OMEGA_N_16,
    build_knm_paper27,
    knm_to_ansatz,
    knm_to_hamiltonian,
)
from scpn_quantum_control.phase.xy_kuramoto import QuantumKuramotoSolver


def test_knm_to_vqe_ground_state_4q():
    """Full pipeline: Knm → Hamiltonian → VQE ansatz → exact ground state check."""
    K = build_knm_paper27(L=4)
    omega = OMEGA_N_16[:4]

    H = knm_to_hamiltonian(K, omega)
    ansatz = knm_to_ansatz(K, reps=2)

    mat = H.to_matrix()
    if hasattr(mat, "toarray"):
        mat = mat.toarray()
    eigvals = np.linalg.eigvalsh(mat)
    exact_ground = eigvals[0]

    assert exact_ground < 0, "Ground state energy should be negative"
    assert ansatz.num_qubits == 4
    assert ansatz.num_parameters == 16  # 4 qubits * 2 params * 2 reps


def test_knm_to_trotter_evolution_R():
    """Full pipeline: Knm → Kuramoto solver → Trotter circuit → energy expectation."""
    from qiskit import transpile

    K = build_knm_paper27(L=4)
    omega = OMEGA_N_16[:4]

    solver = QuantumKuramotoSolver(4, K, omega)
    qc = solver.evolve(time=0.5, trotter_steps=3)

    # Transpile to decompose PauliEvolution before Aer
    qc_t = transpile(qc, basis_gates=["cx", "u3", "u2", "u1", "id"], optimization_level=0)
    qc_t.save_statevector()
    sim = AerSimulator(method="statevector")
    sv = Statevector(sim.run(qc_t).result().get_statevector())

    E = solver.energy_expectation(sv)
    assert np.isfinite(E)
    assert qc.num_qubits == 4


def test_8q_pipeline_hamiltonian_spectrum():
    """8-qubit pipeline: Knm → Hamiltonian → spectrum properties."""
    K = build_knm_paper27(L=8)
    omega = OMEGA_N_16[:8]
    H = knm_to_hamiltonian(K, omega)

    mat = H.to_matrix()
    if hasattr(mat, "toarray"):
        mat = mat.toarray()
    eigvals = np.linalg.eigvalsh(mat)

    assert eigvals[0] < eigvals[-1], "Non-degenerate spectrum"
    assert np.all(np.diff(eigvals) >= -1e-12), "Eigenvalues sorted"
    # Bandwidth scales with coupling + field strength
    bandwidth = eigvals[-1] - eigvals[0]
    assert bandwidth > 1.0


def test_full_16_layer_hamiltonian():
    """Full 16-layer SCPN: Knm → Hamiltonian construction (no diagonalization at 2^16)."""
    K = build_knm_paper27(L=16)
    omega = OMEGA_N_16

    H = knm_to_hamiltonian(K, omega)
    assert H.num_qubits == 16

    # SparsePauliOp should have the right structure:
    # N field terms (Z_i) + N*(N-1)/2 coupling terms (XX_ij + YY_ij)
    n_z = sum(1 for p in H.paulis if str(p).count("Z") == 1 and str(p).count("X") == 0)
    n_xx = sum(1 for p in H.paulis if str(p).count("X") == 2)
    assert n_z == 16
    assert n_xx == 16 * 15 // 2  # all pairs coupled
