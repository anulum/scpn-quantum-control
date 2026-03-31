# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for Integration Pipeline
"""Integration tests: full pipeline from Knm matrix to quantum observables — elite coverage."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

from scpn_quantum_control.bridge import (
    OMEGA_N_16,
    build_knm_paper27,
    knm_to_ansatz,
    knm_to_hamiltonian,
)
from scpn_quantum_control.phase.xy_kuramoto import QuantumKuramotoSolver

# ---------------------------------------------------------------------------
# Knm → Hamiltonian → VQE ansatz pipeline
# ---------------------------------------------------------------------------


class TestKnmToVQEPipeline:
    def test_4q_ground_state_negative(self):
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        H = knm_to_hamiltonian(K, omega)
        mat = H.to_matrix()
        if hasattr(mat, "toarray"):
            mat = mat.toarray()
        eigvals = np.linalg.eigvalsh(mat)
        assert eigvals[0] < 0

    def test_4q_ansatz_dimensions(self):
        K = build_knm_paper27(L=4)
        ansatz = knm_to_ansatz(K, reps=2)
        assert ansatz.num_qubits == 4
        assert ansatz.num_parameters == 16

    def test_hamiltonian_is_hermitian(self):
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        H = knm_to_hamiltonian(K, omega)
        mat = H.to_matrix()
        if hasattr(mat, "toarray"):
            mat = mat.toarray()
        np.testing.assert_allclose(mat, mat.conj().T, atol=1e-12)

    @pytest.mark.parametrize("L", [2, 3, 4])
    def test_various_sizes(self, L):
        K = build_knm_paper27(L=L)
        omega = OMEGA_N_16[:L]
        H = knm_to_hamiltonian(K, omega)
        assert H.num_qubits == L
        ansatz = knm_to_ansatz(K, reps=1)
        assert ansatz.num_qubits == L


# ---------------------------------------------------------------------------
# Knm → Trotter → energy
# ---------------------------------------------------------------------------


class TestKnmToTrotterPipeline:
    def test_4q_trotter_energy_finite(self):
        from qiskit import transpile

        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        solver = QuantumKuramotoSolver(4, K, omega)
        qc = solver.evolve(time=0.5, trotter_steps=3)

        qc_t = transpile(qc, basis_gates=["cx", "u3", "u2", "u1", "id"], optimization_level=0)
        qc_t.save_statevector()
        sim = AerSimulator(method="statevector")
        sv = Statevector(sim.run(qc_t).result().get_statevector())

        E = solver.energy_expectation(sv)
        assert np.isfinite(E)
        assert qc.num_qubits == 4

    def test_2q_trotter(self):
        from qiskit import transpile

        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        solver = QuantumKuramotoSolver(2, K, omega)
        qc = solver.evolve(time=0.3, trotter_steps=2)

        qc_t = transpile(qc, basis_gates=["cx", "u3", "u2", "u1", "id"], optimization_level=0)
        qc_t.save_statevector()
        sim = AerSimulator(method="statevector")
        sv = Statevector(sim.run(qc_t).result().get_statevector())

        E = solver.energy_expectation(sv)
        assert np.isfinite(E)


# ---------------------------------------------------------------------------
# Hamiltonian spectrum
# ---------------------------------------------------------------------------


class TestHamiltonianSpectrum:
    def test_8q_spectrum_sorted(self):
        K = build_knm_paper27(L=8)
        omega = OMEGA_N_16[:8]
        H = knm_to_hamiltonian(K, omega)
        mat = H.to_matrix()
        if hasattr(mat, "toarray"):
            mat = mat.toarray()
        eigvals = np.linalg.eigvalsh(mat)
        assert np.all(np.diff(eigvals) >= -1e-12)

    def test_8q_bandwidth(self):
        K = build_knm_paper27(L=8)
        omega = OMEGA_N_16[:8]
        H = knm_to_hamiltonian(K, omega)
        mat = H.to_matrix()
        if hasattr(mat, "toarray"):
            mat = mat.toarray()
        eigvals = np.linalg.eigvalsh(mat)
        bandwidth = eigvals[-1] - eigvals[0]
        assert bandwidth > 1.0

    def test_4q_spectrum_all_real(self):
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        H = knm_to_hamiltonian(K, omega)
        mat = H.to_matrix()
        if hasattr(mat, "toarray"):
            mat = mat.toarray()
        eigvals = np.linalg.eigvalsh(mat)
        assert np.all(np.isreal(eigvals))
        assert np.all(np.isfinite(eigvals))


# ---------------------------------------------------------------------------
# 16-layer Hamiltonian structure
# ---------------------------------------------------------------------------


class TestFull16Layer:
    def test_hamiltonian_construction(self):
        K = build_knm_paper27(L=16)
        omega = OMEGA_N_16
        H = knm_to_hamiltonian(K, omega)
        assert H.num_qubits == 16

    def test_pauli_term_counts(self):
        K = build_knm_paper27(L=16)
        omega = OMEGA_N_16
        H = knm_to_hamiltonian(K, omega)
        n_z = sum(1 for p in H.paulis if str(p).count("Z") == 1 and str(p).count("X") == 0)
        n_xx = sum(1 for p in H.paulis if str(p).count("X") == 2)
        assert n_z == 16
        assert n_xx == 16 * 15 // 2
