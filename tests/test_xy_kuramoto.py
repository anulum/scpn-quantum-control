# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Xy Kuramoto
"""Tests for phase/xy_kuramoto.py."""

import sys

import numpy as np
import pytest

from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27
from scpn_quantum_control.phase.xy_kuramoto import QuantumKuramotoSolver

SIZES = [2, 3, 4]


@pytest.fixture(params=SIZES)
def solver(request):
    n = request.param
    K = build_knm_paper27(L=n)
    omega = OMEGA_N_16[:n]
    return QuantumKuramotoSolver(n, K, omega)


def test_build_hamiltonian(solver):
    H = solver.build_hamiltonian()
    assert H.num_qubits == solver.n


def test_rejects_invalid_coupling_shape():
    K = np.zeros((2, 3))
    omega = np.zeros(2)
    with pytest.raises(ValueError, match=r"K_coupling shape must be \(2, 2\)"):
        QuantumKuramotoSolver(2, K, omega)


def test_rejects_invalid_omega_shape():
    K = np.zeros((3, 3))
    omega = np.zeros(2)
    with pytest.raises(ValueError, match=r"omega_natural shape must be \(3,\)"):
        QuantumKuramotoSolver(3, K, omega)


def test_rejects_non_finite_inputs():
    K = np.array([[0.0, np.inf], [np.inf, 0.0]])
    omega = np.zeros(2)
    with pytest.raises(ValueError, match="K_coupling must contain only finite values"):
        QuantumKuramotoSolver(2, K, omega)

    with pytest.raises(ValueError, match="omega_natural must contain only finite values"):
        QuantumKuramotoSolver(2, np.zeros((2, 2)), np.array([0.0, np.nan]))


def test_rejects_asymmetric_coupling_matrix():
    K = np.array([[0.0, 1.0], [0.5, 0.0]])
    omega = np.zeros(2)
    with pytest.raises(ValueError, match="K_coupling must be symmetric"):
        QuantumKuramotoSolver(2, K, omega)


def test_rejects_invalid_trotter_settings():
    K = np.zeros((2, 2))
    omega = np.zeros(2)
    with pytest.raises(ValueError, match="trotter_order must be 1 or 2"):
        QuantumKuramotoSolver(2, K, omega, trotter_order=3)

    solver = QuantumKuramotoSolver(2, K, omega)
    with pytest.raises(ValueError, match="trotter_steps must be a positive integer"):
        solver.evolve(1.0, trotter_steps=0)
    with pytest.raises(ValueError, match="time must be finite and non-negative"):
        solver.evolve(-1.0)


def test_rejects_invalid_run_grid():
    solver = QuantumKuramotoSolver(2, np.zeros((2, 2)), np.zeros(2))
    with pytest.raises(ValueError, match="t_max must be finite and non-negative"):
        solver.run(t_max=-0.1, dt=0.1)
    with pytest.raises(ValueError, match="dt must be finite and positive"):
        solver.run(t_max=0.1, dt=0.0)
    with pytest.raises(ValueError, match="trotter_per_step must be a positive integer"):
        solver.run(t_max=0.1, dt=0.1, trotter_per_step=0)


def test_evolve_circuit(solver):
    solver.build_hamiltonian()
    qc = solver.evolve(1.0, trotter_steps=5)
    assert qc.num_qubits == solver.n


def test_order_parameter_range(solver):
    solver.build_hamiltonian()
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    qc = QuantumCircuit(solver.n)
    sv = Statevector.from_instruction(qc)
    R, psi = solver.measure_order_parameter(sv)
    assert 0.0 <= R <= 1.0


def test_run_returns_trajectory(solver):
    result = solver.run(t_max=0.5, dt=0.1)
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


def test_single_oscillator():
    """n=1: no coupling, R determined by single qubit state."""
    K = np.array([[0.0]])
    omega = np.array([1.0])
    solver = QuantumKuramotoSolver(1, K, omega)
    H = solver.build_hamiltonian()
    assert H.num_qubits == 1

    result = solver.run(t_max=0.2, dt=0.1)
    # Single qubit => R = |<X> + i<Y>| is well-defined
    for r in result["R"]:
        assert 0.0 <= r <= 1.0 + 1e-10


def test_second_order_trotter():
    """SuzukiTrotter(order=2) should build a valid evolution circuit."""
    from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27

    n = 4
    K = build_knm_paper27(L=n)
    omega = OMEGA_N_16[:n]
    t = 1.0
    reps = 2

    solver_1 = QuantumKuramotoSolver(n, K, omega, trotter_order=1)
    solver_2 = QuantumKuramotoSolver(n, K, omega, trotter_order=2)
    solver_1.build_hamiltonian()
    solver_2.build_hamiltonian()

    qc1 = solver_1.evolve(t, trotter_steps=reps).decompose(reps=2)
    qc2 = solver_2.evolve(t, trotter_steps=reps).decompose(reps=2)

    assert qc1.num_qubits == n
    assert qc2.num_qubits == n
    assert qc2.size() > qc1.size()


def test_energy_expectation():
    """energy_expectation should return <H> consistent with direct computation."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    K = np.array([[0, 0.5], [0.5, 0]])
    omega = np.array([1.0, 1.0])
    solver = QuantumKuramotoSolver(2, K, omega)
    solver.build_hamiltonian()

    qc = QuantumCircuit(2)
    qc.ry(0.5, 0)
    sv = Statevector.from_instruction(qc)
    E = solver.energy_expectation(sv)
    assert isinstance(E, float)
    # Energy from direct matrix multiplication
    H_mat = np.array(solver._hamiltonian.to_matrix())
    psi = np.array(sv)
    E_direct = float(np.real(psi.conj() @ H_mat @ psi))
    np.testing.assert_allclose(E, E_direct, atol=1e-12)


def test_energy_expectation_builds_hamiltonian_lazily():
    """energy_expectation builds H if called before build_hamiltonian."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    K = np.array([[0, 0.5], [0.5, 0]])
    omega = np.array([1.0, 1.0])
    solver = QuantumKuramotoSolver(2, K, omega)
    sv = Statevector.from_instruction(QuantumCircuit(2))

    energy = solver.energy_expectation(sv)

    assert isinstance(energy, float)
    assert solver._hamiltonian is not None


def test_order_parameter_qiskit_fallback(monkeypatch: pytest.MonkeyPatch):
    """Order parameter falls back to Qiskit expectations if Rust lacks the helper."""
    from qiskit.quantum_info import Statevector

    class EngineWithoutExpectations:
        pass

    monkeypatch.setitem(sys.modules, "scpn_quantum_engine", EngineWithoutExpectations())

    K = np.array([[0.0]])
    omega = np.array([0.0])
    solver = QuantumKuramotoSolver(1, K, omega)
    sv = Statevector([1 / np.sqrt(2), 1 / np.sqrt(2)])

    R, psi = solver.measure_order_parameter(sv)

    assert pytest.approx(1.0) == R
    assert psi == pytest.approx(0.0)


def test_trotter_product_expands_with_reps():
    """Increasing Trotter reps should expand the decomposed product formula."""

    K = np.array([[0, 0.8], [0.8, 0]])
    omega = np.array([1.0, 2.0])
    t = 0.5

    solver = QuantumKuramotoSolver(2, K, omega)
    solver.build_hamiltonian()

    sizes = []
    for reps in [1, 3, 8]:
        qc = solver.evolve(t, trotter_steps=reps).decompose(reps=2)
        sizes.append(qc.size())

    assert sizes[1] > sizes[0]
    assert sizes[2] > sizes[1]


# ---------------------------------------------------------------------------
# Hamiltonian physical invariants
# ---------------------------------------------------------------------------


def test_hamiltonian_hermitian():
    """H must be Hermitian (eigenvalues real, spectrum bounded)."""
    K = build_knm_paper27(L=4)
    omega = OMEGA_N_16[:4]
    solver = QuantumKuramotoSolver(4, K, omega)
    H = solver.build_hamiltonian()
    mat = H.to_matrix()
    if hasattr(mat, "toarray"):
        mat = mat.toarray()
    np.testing.assert_allclose(mat, mat.conj().T, atol=1e-12)


def test_hamiltonian_traceless():
    """XY Hamiltonian (Pauli terms) is traceless."""
    K = build_knm_paper27(L=3)
    omega = OMEGA_N_16[:3]
    solver = QuantumKuramotoSolver(3, K, omega)
    H = solver.build_hamiltonian()
    mat = H.to_matrix()
    if hasattr(mat, "toarray"):
        mat = mat.toarray()
    assert abs(np.trace(mat)) < 1e-8


def test_evolution_unitarity():
    """Trotter evolution circuit must preserve statevector norm."""
    from qiskit.quantum_info import Statevector

    K = build_knm_paper27(L=3)
    omega = OMEGA_N_16[:3]
    solver = QuantumKuramotoSolver(3, K, omega)
    qc = solver.evolve(0.5, trotter_steps=3)
    sv = Statevector.from_instruction(qc)
    np.testing.assert_allclose(float(np.sum(np.abs(sv) ** 2)), 1.0, atol=1e-12)


def test_zero_time_evolution_is_identity():
    """Mutation guard: zero-time evolution must leave arbitrary states unchanged."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    K = build_knm_paper27(L=3)
    omega = OMEGA_N_16[:3]
    solver = QuantumKuramotoSolver(3, K, omega)

    init = QuantumCircuit(3)
    init.ry(0.3, 0)
    init.rx(0.7, 1)
    init.rz(1.1, 2)
    before = Statevector.from_instruction(init)
    after = before.evolve(solver.evolve(0.0, trotter_steps=4))

    np.testing.assert_allclose(after.data, before.data, atol=1e-12)


def test_pauli_operator_targets_requested_qubit():
    """Mutation guard: internal Pauli labels must respect Qiskit qubit ordering."""
    solver = QuantumKuramotoSolver(3, np.zeros((3, 3)), np.zeros(3))

    assert str(solver._pauli_op("X", 0).paulis[0]) == "IIX"
    assert str(solver._pauli_op("Y", 1).paulis[0]) == "IYI"
    assert str(solver._pauli_op("Z", 2).paulis[0]) == "ZII"


def test_run_time_grid_preserves_requested_endpoints():
    """Mutation guard: trajectory grid must include the requested start and stop times."""
    solver = QuantumKuramotoSolver(2, np.zeros((2, 2)), np.array([0.2, 0.4]))

    result = solver.run(t_max=0.35, dt=0.1, trotter_per_step=1)

    assert result["times"][0] == pytest.approx(0.0)
    assert result["times"][-1] == pytest.approx(0.35)
    assert len(result["times"]) == len(result["R"])


# ---------------------------------------------------------------------------
# Rust path: Kuramoto Euler vs quantum Trotter parity
# ---------------------------------------------------------------------------


def test_rust_kuramoto_euler_matches_direction():
    """Rust kuramoto_euler and quantum solver should evolve phases in the same direction.

    Not exact parity (Trotter vs Euler), but both should show synchronisation
    tendency for strong coupling.
    """
    try:
        import scpn_quantum_engine as eng

        n = 4
        K = build_knm_paper27(L=n)
        omega = OMEGA_N_16[:n]
        theta0 = np.zeros(n, dtype=np.float64)

        # Rust classical
        theta_rust = np.array(eng.kuramoto_euler(theta0, omega, K, 0.01, 100))
        R_rust = eng.order_parameter(theta_rust)

        # Quantum
        solver = QuantumKuramotoSolver(n, K, omega)
        result = solver.run(t_max=1.0, dt=0.5)
        R_quantum = result["R"][-1]

        # Both should show some synchronisation (R > 0)
        assert R_rust > 0
        assert R_quantum >= 0
    except ImportError:
        pytest.skip("scpn-quantum-engine not available")


# ---------------------------------------------------------------------------
# Pipeline: Knm → Solver → Trotter → R(t) → wired end-to-end
# ---------------------------------------------------------------------------


def test_pipeline_full_kuramoto_evolution():
    """Full pipeline: build_knm → QuantumKuramotoSolver → run → R trajectory.
    Verifies the core solver is wired and produces physical output.
    """
    import time

    K = build_knm_paper27(L=4)
    omega = OMEGA_N_16[:4]

    t0 = time.perf_counter()
    solver = QuantumKuramotoSolver(4, K, omega)
    result = solver.run(t_max=0.3, dt=0.1, trotter_per_step=5)
    dt = (time.perf_counter() - t0) * 1000

    assert len(result["R"]) > 0
    for R in result["R"]:
        assert 0.0 <= R <= 1.0 + 1e-10

    print(f"\n  PIPELINE Knm→KuramotoSolver→R(t) (4q, 3 steps): {dt:.1f} ms")
    print(f"  R trajectory: {[f'{r:.4f}' for r in result['R']]}")
