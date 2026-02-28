"""Determinism tests: verify reproducible outputs given the same inputs."""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from scpn_quantum_control.bridge import (
    OMEGA_N_16,
    build_knm_paper27,
    knm_to_ansatz,
    knm_to_hamiltonian,
)
from scpn_quantum_control.control.q_disruption import QuantumDisruptionClassifier
from scpn_quantum_control.phase.xy_kuramoto import QuantumKuramotoSolver
from scpn_quantum_control.qec.control_qec import ControlQEC


def test_knm_deterministic():
    """build_knm_paper27 returns identical matrices on repeated calls."""
    K1 = build_knm_paper27(L=16)
    K2 = build_knm_paper27(L=16)
    assert np.array_equal(K1, K2)


def test_hamiltonian_deterministic():
    """Same K + omega → identical SparsePauliOp."""
    K = build_knm_paper27(L=4)
    omega = OMEGA_N_16[:4]
    H1 = knm_to_hamiltonian(K, omega)
    H2 = knm_to_hamiltonian(K, omega)
    assert H1 == H2


def test_ansatz_deterministic():
    """Same K + reps → identical circuit structure."""
    K = build_knm_paper27(L=4)
    qc1 = knm_to_ansatz(K, reps=2)
    qc2 = knm_to_ansatz(K, reps=2)
    assert qc1.num_qubits == qc2.num_qubits
    assert qc1.num_parameters == qc2.num_parameters
    assert qc1.size() == qc2.size()


def test_kuramoto_evolve_deterministic():
    """Same params → identical Trotter circuit."""
    K = build_knm_paper27(L=4)
    omega = OMEGA_N_16[:4]
    s1 = QuantumKuramotoSolver(4, K, omega)
    s2 = QuantumKuramotoSolver(4, K, omega)
    qc1 = s1.evolve(time=0.5, trotter_steps=2)
    qc2 = s2.evolve(time=0.5, trotter_steps=2)
    assert qc1.num_qubits == qc2.num_qubits
    assert qc1.size() == qc2.size()


def test_disruption_seed_reproducibility():
    """Same seed → identical classifier parameters."""
    c1 = QuantumDisruptionClassifier(n_features=5, n_layers=2, seed=42)
    c2 = QuantumDisruptionClassifier(n_features=5, n_layers=2, seed=42)
    assert np.array_equal(c1.params, c2.params)


def test_qec_errors_seeded():
    """Same RNG seed → identical error patterns."""
    qec = ControlQEC(distance=3)
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)
    err_x1, err_z1 = qec.simulate_errors(p_error=0.1, rng=rng1)
    err_x2, err_z2 = qec.simulate_errors(p_error=0.1, rng=rng2)
    assert np.array_equal(err_x1, err_x2)
    assert np.array_equal(err_z1, err_z2)


def test_statevector_simulation_deterministic():
    """Same circuit → identical statevector results."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.save_statevector()

    sim = AerSimulator(method="statevector")
    sv1 = sim.run(qc).result().get_statevector()
    sv2 = sim.run(qc).result().get_statevector()
    assert np.allclose(sv1, sv2)
