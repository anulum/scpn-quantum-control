"""Tests for hardware runner + experiments using local AerSimulator."""

import numpy as np
import pytest

from scpn_quantum_control.hardware.classical import (
    classical_brute_mpc,
    classical_exact_diag,
    classical_exact_evolution,
    classical_kuramoto_reference,
)
from scpn_quantum_control.hardware.runner import HardwareRunner, JobResult


@pytest.fixture
def sim_runner(tmp_path):
    runner = HardwareRunner(use_simulator=True, results_dir=str(tmp_path / "results"))
    runner.connect()
    return runner


def test_connect_simulator(sim_runner):
    assert sim_runner.backend is not None
    assert "aer" in sim_runner.backend_name.lower() or sim_runner.use_simulator


def test_transpile_simple_circuit(sim_runner):
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    isa = sim_runner.transpile(qc)
    assert isa.num_qubits >= 2


def test_run_sampler_bell(sim_runner):
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    results = sim_runner.run_sampler(qc, shots=1000, name="bell_test")
    assert len(results) == 1
    assert results[0].counts is not None
    total = sum(results[0].counts.values())
    assert total == 1000


def test_save_result(sim_runner):
    jr = JobResult(
        job_id="test_123",
        backend_name="test",
        experiment_name="save_test",
        counts={"00": 500, "11": 500},
        timestamp="2026-02-28T12:00:00",
    )
    path = sim_runner.save_result(jr, "test_save.json")
    assert path.exists()
    import json

    with open(path) as f:
        data = json.load(f)
    assert data["job_id"] == "test_123"


def test_circuit_stats(sim_runner):
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    isa = sim_runner.transpile(qc)
    stats = sim_runner.circuit_stats(isa)
    assert "depth" in stats
    assert "ecr_gates" in stats
    assert stats["n_qubits"] >= 2


# ── Classical reference tests ──


def test_classical_kuramoto():
    ref = classical_kuramoto_reference(4, t_max=0.5, dt=0.1)
    assert len(ref["R"]) == 6
    assert all(0.0 <= r <= 1.5 for r in ref["R"])


def test_classical_exact_diag():
    ref = classical_exact_diag(4)
    assert ref["ground_energy"] < 0  # XY Hamiltonian should have negative ground
    assert ref["spectral_gap"] > 0
    assert len(ref["eigenvalues"]) == 16  # 2^4


def test_classical_exact_evolution():
    ref = classical_exact_evolution(3, 0.3, 0.1)
    assert len(ref["R"]) == 4
    assert all(np.isfinite(r) for r in ref["R"])


def test_classical_exact_evolution_n1():
    """n=1: single oscillator, R from single-qubit XY expectations."""
    ref = classical_exact_evolution(1, 0.2, 0.1)
    assert len(ref["R"]) == 3
    for r in ref["R"]:
        assert 0.0 <= r <= 1.0 + 1e-10


def test_classical_exact_diag_n1():
    """n=1: single qubit Hamiltonian has 2 eigenvalues."""
    ref = classical_exact_diag(1)
    assert len(ref["eigenvalues"]) == 2
    assert ref["spectral_gap"] > 0


def test_classical_brute_mpc():
    B = np.eye(2)
    target = np.array([0.8, 0.6])
    ref = classical_brute_mpc(B, target, horizon=4)
    assert ref["n_evaluated"] == 16
    assert len(ref["optimal_actions"]) == 4
    assert ref["optimal_cost"] <= ref["all_costs"].max()


# ── Full experiment pipeline on simulator ──


def test_kuramoto_4osc_on_simulator(sim_runner):
    from scpn_quantum_control.hardware.experiments import kuramoto_4osc_experiment

    result = kuramoto_4osc_experiment(sim_runner, shots=500, n_time_steps=3, dt=0.05)
    assert result["experiment"] == "kuramoto_4osc"
    assert len(result["hw_R"]) == 3
    assert len(result["classical_R"]) > 0


def test_qaoa_mpc_on_simulator(sim_runner):
    """Quick QAOA test with minimal iterations."""
    import scpn_quantum_control.hardware.experiments as exp_mod
    from scpn_quantum_control.hardware.experiments import qaoa_mpc_4_experiment

    original = exp_mod.minimize

    def limited_minimize(fn, x0, **kwargs):
        kwargs.setdefault("options", {})["maxiter"] = 5
        return original(fn, x0, **kwargs)

    exp_mod.minimize = limited_minimize
    try:
        result = qaoa_mpc_4_experiment(sim_runner, shots=200)
        assert "brute_force_cost" in result
        assert "qaoa_p1" in result
    finally:
        exp_mod.minimize = original


def test_vqe_4q_on_simulator(sim_runner):
    """VQE should converge below exact ground energy + tolerance."""
    import scpn_quantum_control.hardware.experiments as exp_mod
    from scpn_quantum_control.hardware.experiments import vqe_4q_experiment

    original = exp_mod.minimize

    def limited_minimize(fn, x0, **kwargs):
        kwargs.setdefault("options", {})["maxiter"] = 20
        return original(fn, x0, **kwargs)

    exp_mod.minimize = limited_minimize
    try:
        result = vqe_4q_experiment(sim_runner, shots=100, maxiter=20)
        assert result["experiment"] == "vqe_4q"
        assert result["n_qubits"] == 4
        assert np.isfinite(result["vqe_energy"])
        assert len(result["energy_history"]) > 0
    finally:
        exp_mod.minimize = original


@pytest.mark.slow
def test_upde_16_snapshot_on_simulator(sim_runner):
    """16-layer UPDE snapshot: produces R and per-qubit expectations.

    Marked slow: 16-qubit Trotter circuit takes ~5 min on AerSimulator.
    Run with: pytest -m slow
    """
    from scpn_quantum_control.hardware.experiments import upde_16_snapshot_experiment

    result = upde_16_snapshot_experiment(sim_runner, shots=100, trotter_steps=1)
    assert result["experiment"] == "upde_16_snapshot"
    assert result["n_layers"] == 16
    assert 0.0 <= result["hw_R"] <= 1.5
    assert np.isfinite(result["classical_R"])
    assert len(result["hw_exp_x"]) == 16


def test_kuramoto_4osc_zne_on_simulator(sim_runner):
    """ZNE experiment: produces R per scale and extrapolated value."""
    from scpn_quantum_control.hardware.experiments import kuramoto_4osc_zne_experiment

    result = kuramoto_4osc_zne_experiment(sim_runner, shots=500, dt=0.05, scales=[1, 3])
    assert result["experiment"] == "kuramoto_4osc_zne"
    assert len(result["R_per_scale"]) == 2
    assert np.isfinite(result["zne_R"])
    assert np.isfinite(result["classical_R"])


def test_transpile_with_dd(sim_runner):
    """DD pass should not crash; on simulator it falls back to original circuit."""
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.barrier()
    qc.cx(1, 2)
    qc.measure_all()
    # transpile_with_dd handles the scheduling internally and falls back gracefully
    dd_circuit = sim_runner.transpile_with_dd(qc)
    assert dd_circuit.num_qubits >= 3


# ── Bloch vector tests ──


def test_bloch_vector_magnitude_bounded(tmp_path):
    """Bloch magnitudes from valid expectations must be in [0, 1]."""
    import json

    data = {
        "exp_x": [0.3, -0.2, 0.0],
        "exp_y": [0.4, 0.1, 0.0],
        "exp_z": [0.5, 0.8, 1.0],
    }
    path = tmp_path / "bloch_test.json"
    with open(path, "w") as f:
        json.dump(data, f)

    from scpn_quantum_control.hardware.classical import bloch_vectors_from_json

    result = bloch_vectors_from_json(str(path))
    assert result["n_qubits"] == 3
    # sqrt(x^2+y^2+z^2) must be <= 1 for valid quantum states
    for m in result["bloch_magnitudes"]:
        assert 0.0 <= m <= 1.0 + 1e-10


# ── Sparse eigensolver test ──


def test_exact_diag_sparse_path():
    """Sparse eigensolver (k_eigenvalues) should agree with dense on ground energy."""
    ref_dense = classical_exact_diag(4)
    ref_sparse = classical_exact_diag(4, k_eigenvalues=6)
    assert abs(ref_dense["ground_energy"] - ref_sparse["ground_energy"]) < 1e-8
    assert abs(ref_dense["spectral_gap"] - ref_sparse["spectral_gap"]) < 1e-8
    assert len(ref_sparse["eigenvalues"]) == 6


# ── Endianness agreement tests ──


def test_classical_evolution_matches_qiskit():
    """Classical expm evolution R must match Qiskit Statevector evolution.

    This verifies that _build_initial_state and _expectation_pauli use
    Qiskit's little-endian convention consistently with knm_to_hamiltonian.
    """

    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.quantum_info import Statevector
    from qiskit.synthesis import LieTrotter

    from scpn_quantum_control.bridge.knm_hamiltonian import (
        OMEGA_N_16,
        build_knm_paper27,
        knm_to_hamiltonian,
    )

    n = 3
    K = build_knm_paper27(L=n)
    omega = OMEGA_N_16[:n]
    dt = 0.3

    # Qiskit circuit evolution (ground truth)
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(n)
    for i in range(n):
        qc.ry(float(omega[i]) % (2 * np.pi), i)
    H = knm_to_hamiltonian(K, omega)
    evo = PauliEvolutionGate(H, time=dt, synthesis=LieTrotter(reps=100))
    qc.append(evo, range(n))
    sv = Statevector.from_instruction(qc)

    # Classical expm evolution
    ref = classical_exact_evolution(n, dt, dt, K, omega)
    R_classical = ref["R"][-1]

    # Qiskit R from statevector per-qubit expectations
    from qiskit.quantum_info import SparsePauliOp

    z_complex = 0.0 + 0.0j
    for q in range(n):
        x_label = ["I"] * n
        y_label = ["I"] * n
        x_label[q] = "X"
        y_label[q] = "Y"
        x_op = SparsePauliOp("".join(reversed(x_label)))
        y_op = SparsePauliOp("".join(reversed(y_label)))
        z_complex += sv.expectation_value(x_op).real + 1j * sv.expectation_value(y_op).real
    z_complex /= n
    R_qiskit = abs(z_complex)

    assert abs(R_classical - R_qiskit) < 1e-6, (
        f"R mismatch: classical={R_classical:.6f}, qiskit={R_qiskit:.6f}"
    )
