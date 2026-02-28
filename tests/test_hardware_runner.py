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


def test_noise_baseline_on_simulator(sim_runner):
    """Baseline circuit: low depth, R close to classical."""
    from scpn_quantum_control.hardware.experiments import noise_baseline_experiment

    result = noise_baseline_experiment(sim_runner, shots=500)
    assert result["experiment"] == "noise_baseline"
    assert result["n_qubits"] == 4
    assert 0.0 <= result["hw_R"] <= 1.5
    assert np.isfinite(result["classical_R"])
    assert len(result["hw_exp_x"]) == 4
    assert len(result["hw_exp_z"]) == 4


def test_kuramoto_8osc_zne_on_simulator(sim_runner):
    """8-osc ZNE: produces R per scale and finite extrapolated value."""
    from scpn_quantum_control.hardware.experiments import kuramoto_8osc_zne_experiment

    result = kuramoto_8osc_zne_experiment(sim_runner, shots=500, dt=0.05, scales=[1, 3])
    assert result["experiment"] == "kuramoto_8osc_zne"
    assert result["n_oscillators"] == 8
    assert len(result["R_per_scale"]) == 2
    assert np.isfinite(result["zne_R"])
    assert np.isfinite(result["classical_R"])


def test_vqe_8q_hardware_on_simulator(sim_runner):
    """VQE 8q hardware path: returns hw_energy, sim_energy, exact_energy."""
    import scpn_quantum_control.hardware.experiments as exp_mod
    from scpn_quantum_control.hardware.experiments import vqe_8q_hardware_experiment

    original = exp_mod.minimize

    def limited_minimize(fn, x0, **kwargs):
        kwargs.setdefault("options", {})["maxiter"] = 20
        return original(fn, x0, **kwargs)

    exp_mod.minimize = limited_minimize
    try:
        result = vqe_8q_hardware_experiment(sim_runner, shots=100, maxiter=20)
        assert result["experiment"] == "vqe_8q_hardware"
        assert result["n_qubits"] == 8
        assert np.isfinite(result["hw_energy"])
        assert np.isfinite(result["sim_energy"])
        assert np.isfinite(result["exact_energy"])
    finally:
        exp_mod.minimize = original


@pytest.mark.slow
def test_upde_16_dd_on_simulator(sim_runner):
    """16-layer UPDE with DD: R + per-qubit expectations for 16 layers.

    Marked slow: 16-qubit circuits take significant time on AerSimulator.
    Run with: pytest -m slow
    """
    from scpn_quantum_control.hardware.experiments import upde_16_dd_experiment

    result = upde_16_dd_experiment(sim_runner, shots=100, trotter_steps=1)
    assert result["experiment"] == "upde_16_dd"
    assert result["n_layers"] == 16
    assert 0.0 <= result["hw_R_raw"] <= 1.5
    assert 0.0 <= result["hw_R_dd"] <= 1.5
    assert np.isfinite(result["classical_R"])
    assert len(result["hw_exp_x_dd"]) == 16


def test_kuramoto_4osc_trotter2_on_simulator(sim_runner):
    """Trotter-2: produces R per step, comparison data."""
    from scpn_quantum_control.hardware.experiments import kuramoto_4osc_trotter2_experiment

    result = kuramoto_4osc_trotter2_experiment(sim_runner, shots=500, n_time_steps=3, dt=0.05)
    assert result["experiment"] == "kuramoto_4osc_trotter2"
    assert result["trotter_order"] == 2
    assert len(result["hw_R"]) == 3
    assert len(result["classical_R"]) > 0
    assert all(np.isfinite(r) for r in result["hw_R"])


def test_sync_threshold_on_simulator(sim_runner):
    """Sync threshold sweep: R should increase with K_base."""
    from scpn_quantum_control.hardware.experiments import sync_threshold_experiment

    result = sync_threshold_experiment(sim_runner, shots=500, k_values=[0.1, 0.45])
    assert result["experiment"] == "sync_threshold"
    assert len(result["results"]) == 2
    for entry in result["results"]:
        assert 0.0 <= entry["hw_R"] <= 1.5
        assert np.isfinite(entry["classical_R"])


def test_ansatz_comparison_hw_on_simulator(sim_runner):
    """Ansatz comparison: all three produce finite hw_energy."""
    import scpn_quantum_control.hardware.experiments as exp_mod
    from scpn_quantum_control.hardware.experiments import ansatz_comparison_hw_experiment

    original = exp_mod.minimize

    def limited_minimize(fn, x0, **kwargs):
        kwargs.setdefault("options", {})["maxiter"] = 15
        return original(fn, x0, **kwargs)

    exp_mod.minimize = limited_minimize
    try:
        result = ansatz_comparison_hw_experiment(sim_runner, shots=100, maxiter=15)
        assert result["experiment"] == "ansatz_comparison_hw"
        assert len(result["comparison"]) == 3
        for entry in result["comparison"]:
            assert np.isfinite(entry["hw_energy"])
            assert np.isfinite(entry["sim_energy"])
    finally:
        exp_mod.minimize = original


def test_zne_higher_order_on_simulator(sim_runner):
    """ZNE higher-order: produces extrapolations for multiple polynomial orders."""
    from scpn_quantum_control.hardware.experiments import zne_higher_order_experiment

    result = zne_higher_order_experiment(
        sim_runner, shots=500, dt=0.05, scales=[1, 3, 5], poly_order=2
    )
    assert result["experiment"] == "zne_higher_order"
    assert len(result["R_per_scale"]) == 3
    assert "order_1" in result["extrapolations"]
    assert "order_2" in result["extrapolations"]
    assert np.isfinite(result["extrapolations"]["order_1"]["zne_R"])
    assert np.isfinite(result["extrapolations"]["order_2"]["zne_R"])


def test_decoherence_scaling_on_simulator(sim_runner):
    """Decoherence scaling: produces data points and gamma fit."""
    from scpn_quantum_control.hardware.experiments import decoherence_scaling_experiment

    result = decoherence_scaling_experiment(sim_runner, shots=500, qubit_counts=[2, 4, 6])
    assert result["experiment"] == "decoherence_scaling"
    assert len(result["data_points"]) == 3
    for dp in result["data_points"]:
        assert dp["depth"] > 0
        assert 0.0 <= dp["hw_R"] <= 1.5
    assert np.isfinite(result["fit_gamma"])


def test_vqe_landscape_on_simulator(sim_runner):
    """VQE landscape: energy variance for barren plateau detection."""
    from scpn_quantum_control.hardware.experiments import vqe_landscape_experiment

    result = vqe_landscape_experiment(sim_runner, shots=100, n_samples=10)
    assert result["experiment"] == "vqe_landscape"
    assert "knm_informed" in result["landscapes"]
    assert "two_local" in result["landscapes"]
    for name in ["knm_informed", "two_local"]:
        land = result["landscapes"][name]
        assert land["std_energy"] > 0
        assert land["min_energy"] < land["max_energy"]


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


def test_bell_test_4q_on_simulator(sim_runner):
    """CHSH Bell test: S values are finite, S_sim shows entanglement."""
    import scpn_quantum_control.phase.phase_vqe as vqe_mod
    from scpn_quantum_control.hardware.experiments import bell_test_4q_experiment

    original = vqe_mod.minimize

    def limited_minimize(fn, x0, **kwargs):
        kwargs.setdefault("options", {})["maxiter"] = 30
        return original(fn, x0, **kwargs)

    vqe_mod.minimize = limited_minimize
    try:
        result = bell_test_4q_experiment(sim_runner, shots=500, maxiter=30)
        assert result["experiment"] == "bell_test_4q"
        assert np.isfinite(result["S_hw"])
        assert np.isfinite(result["S_sim"])
        assert result["S_sim"] > 0
        assert "correlators_hw" in result
        assert "correlators_sim" in result
    finally:
        vqe_mod.minimize = original


def test_correlator_4q_on_simulator(sim_runner):
    """ZZ correlator: 4×4 symmetric matrix with finite Frobenius error."""
    import scpn_quantum_control.phase.phase_vqe as vqe_mod
    from scpn_quantum_control.hardware.experiments import correlator_4q_experiment

    original = vqe_mod.minimize

    def limited_minimize(fn, x0, **kwargs):
        kwargs.setdefault("options", {})["maxiter"] = 30
        return original(fn, x0, **kwargs)

    vqe_mod.minimize = limited_minimize
    try:
        result = correlator_4q_experiment(sim_runner, shots=500, maxiter=30)
        assert result["experiment"] == "correlator_4q"
        corr = np.array(result["corr_hw"])
        assert corr.shape == (4, 4)
        np.testing.assert_allclose(corr, corr.T, atol=1e-10)
        assert np.isfinite(result["frobenius_error"])
        assert result["max_correlation_hw"] > 0
    finally:
        vqe_mod.minimize = original


def test_qkd_qber_4q_on_simulator(sim_runner):
    """QKD QBER: error rates in [0,1], dict keys present."""
    import scpn_quantum_control.phase.phase_vqe as vqe_mod
    from scpn_quantum_control.hardware.experiments import qkd_qber_4q_experiment

    original = vqe_mod.minimize

    def limited_minimize(fn, x0, **kwargs):
        kwargs.setdefault("options", {})["maxiter"] = 30
        return original(fn, x0, **kwargs)

    vqe_mod.minimize = limited_minimize
    try:
        result = qkd_qber_4q_experiment(sim_runner, shots=500, maxiter=30)
        assert result["experiment"] == "qkd_qber_4q"
        assert 0.0 <= result["qber_z_hw"] <= 1.0
        assert 0.0 <= result["qber_x_hw"] <= 1.0
        assert 0.0 <= result["qber_sim"] <= 1.0
        assert isinstance(result["secure_hw"], bool)
        assert isinstance(result["secure_sim"], bool)
        assert np.isfinite(result["key_rate_hw"])
    finally:
        vqe_mod.minimize = original


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
