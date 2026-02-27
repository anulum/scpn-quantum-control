"""Tests for phase/xy_kuramoto.py."""
import numpy as np
import pytest

from scpn_quantum_control.phase.xy_kuramoto import QuantumKuramotoSolver


@pytest.fixture
def small_solver():
    K = np.array([[0, 0.5], [0.5, 0]])
    omega = np.array([1.0, 1.0])
    return QuantumKuramotoSolver(2, K, omega)


def test_build_hamiltonian(small_solver):
    H = small_solver.build_hamiltonian()
    assert H.num_qubits == 2


def test_evolve_circuit(small_solver):
    small_solver.build_hamiltonian()
    qc = small_solver.evolve(1.0, trotter_steps=5)
    assert qc.num_qubits == 2


def test_order_parameter_range(small_solver):
    small_solver.build_hamiltonian()
    from qiskit.quantum_info import Statevector
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(2)
    sv = Statevector.from_instruction(qc)
    R, psi = small_solver.measure_order_parameter(sv)
    assert 0.0 <= R <= 1.0


def test_run_returns_trajectory(small_solver):
    result = small_solver.run(t_max=0.5, dt=0.1)
    assert "R" in result
    assert "times" in result
    assert len(result["R"]) == len(result["times"])


def test_strong_coupling_increases_R():
    """Strong coupling (K >> delta_omega) should maintain or increase R."""
    K = np.array([[0, 5.0], [5.0, 0]])
    omega = np.array([1.0, 1.0])
    solver = QuantumKuramotoSolver(2, K, omega)
    result = solver.run(t_max=0.5, dt=0.1)
    # R should not collapse for identical frequencies with strong coupling
    assert result["R"][-1] > 0.1
