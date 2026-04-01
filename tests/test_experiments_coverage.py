# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Hardware Experiments Coverage Tests
"""Coverage tests for hardware.experiments — all 20 experiments on AerSimulator."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit.quantum_info import SparsePauliOp

from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27
from scpn_quantum_control.hardware.experiments import (
    ALL_EXPERIMENTS,
    _build_evo_base,
    _build_xyz_circuits,
    _correlator_from_counts,
    _expectation_per_qubit,
    _qaoa_cost_from_counts,
    _R_from_xyz,
    _run_vqe,
    ansatz_comparison_hw_experiment,
    bell_test_4q_experiment,
    correlator_4q_experiment,
    decoherence_scaling_experiment,
    kuramoto_4osc_experiment,
    kuramoto_4osc_trotter2_experiment,
    kuramoto_4osc_zne_experiment,
    kuramoto_8osc_experiment,
    kuramoto_8osc_zne_experiment,
    noise_baseline_experiment,
    qaoa_mpc_4_experiment,
    qkd_qber_4q_experiment,
    sync_threshold_experiment,
    upde_16_dd_experiment,
    upde_16_snapshot_experiment,
    vqe_4q_experiment,
    vqe_8q_experiment,
    vqe_8q_hardware_experiment,
    vqe_landscape_experiment,
    zne_higher_order_experiment,
)
from scpn_quantum_control.hardware.runner import HardwareRunner, JobResult


@pytest.fixture(scope="module")
def sim_runner(tmp_path_factory):
    """Shared AerSimulator runner for all experiment tests."""
    results_dir = tmp_path_factory.mktemp("results")
    runner = HardwareRunner(
        use_simulator=True,
        optimization_level=0,
        results_dir=str(results_dir),
    )
    runner.connect()
    return runner


class _MockRunner:
    """Lightweight mock runner for 16-qubit experiments where AerSimulator is too slow.

    Returns random counts so experiment logic (circuit build, result parse,
    classical comparison) executes fully without 2^16 state simulation.
    """

    def __init__(self, tmp_dir):
        from pathlib import Path

        self.results_dir = Path(tmp_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._pm = True  # sentinel so transpile_with_dd doesn't raise

    def run_sampler(self, circuits, shots=100, name="mock"):
        from qiskit import QuantumCircuit

        if isinstance(circuits, QuantumCircuit):
            circuits = [circuits]
        results = []
        rng = np.random.default_rng(42)
        for i, qc in enumerate(circuits):
            n = qc.num_qubits
            # random counts over a few bitstrings
            bitstrings = [format(int(rng.integers(0, 2**n)), f"0{n}b") for _ in range(8)]
            remaining = shots
            counts = {}
            for bs in bitstrings[:-1]:
                c = int(rng.integers(1, max(remaining // 4, 2)))
                counts[bs] = c
                remaining -= c
            counts[bitstrings[-1]] = int(max(remaining, 1))
            results.append(
                JobResult(
                    job_id=f"mock_{i}",
                    backend_name="mock",
                    experiment_name=f"{name}_{i}",
                    counts=counts,
                    wall_time_s=0.01,
                    timestamp="2026-03-26T00:00:00",
                    metadata={"depth": 50, "n_qubits": n, "ecr_gates": 20, "total_gates": 100},
                )
            )
        return results

    def run_estimator(self, circuit, observables, name="mock", parameter_values=None):
        n_obs = len(observables)
        evs = np.random.default_rng(42).uniform(-1, 1, n_obs)
        return JobResult(
            job_id="mock_est",
            backend_name="mock",
            experiment_name=name,
            expectation_values=evs,
            wall_time_s=0.01,
            timestamp="2026-03-26T00:00:00",
            metadata={
                "depth": 50,
                "n_qubits": circuit.num_qubits,
                "ecr_gates": 20,
                "total_gates": 100,
            },
        )

    def transpile(self, circuit):
        return circuit

    def transpile_with_dd(self, circuit, dd_sequence=None):
        return circuit

    def save_result(self, result, filename=None):
        import json

        data = result.to_dict() if isinstance(result, JobResult) else [r.to_dict() for r in result]
        path = self.results_dir / (filename or "mock_result.json")
        with open(path, "w") as f:
            json.dump(data, f)
        return path


@pytest.fixture(scope="module")
def mock_runner(tmp_path_factory):
    """Mock runner for heavy (16-qubit) experiments."""
    return _MockRunner(str(tmp_path_factory.mktemp("mock_results")))


# --- Helper function tests ---


class TestBuildEvoBase:
    def test_returns_circuit(self):
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        qc = _build_evo_base(2, K, omega, t=0.1, trotter_reps=1)
        assert qc.num_qubits == 2

    def test_trotter_order_2(self):
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        qc = _build_evo_base(2, K, omega, t=0.1, trotter_reps=1, trotter_order=2)
        assert qc.num_qubits == 2

    def test_4_qubit(self):
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        qc = _build_evo_base(4, K, omega, t=0.2, trotter_reps=2)
        assert qc.num_qubits == 4


class TestBuildXYZCircuits:
    def test_returns_three_circuits(self):
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        base = _build_evo_base(2, K, omega, t=0.1, trotter_reps=1)
        z_qc, x_qc, y_qc = _build_xyz_circuits(base, 2)
        assert z_qc.num_qubits == 2
        assert x_qc.num_qubits == 2
        assert y_qc.num_qubits == 2


class TestExpectationPerQubit:
    def test_all_zeros(self):
        counts = {"00": 1000}
        exp, std = _expectation_per_qubit(counts, 2)
        np.testing.assert_allclose(exp, [1.0, 1.0], atol=0.01)

    def test_all_ones(self):
        counts = {"11": 1000}
        exp, std = _expectation_per_qubit(counts, 2)
        np.testing.assert_allclose(exp, [-1.0, -1.0], atol=0.01)

    def test_mixed(self):
        counts = {"00": 500, "11": 500}
        exp, std = _expectation_per_qubit(counts, 2)
        np.testing.assert_allclose(exp, [0.0, 0.0], atol=0.01)

    def test_std_nonzero(self):
        counts = {"00": 500, "11": 500}
        _, std = _expectation_per_qubit(counts, 2)
        assert all(s > 0 for s in std)


class TestRFromXYZ:
    def test_coherent_state(self):
        z = {"00": 1000}
        x = {"00": 1000}
        y = {"00": 1000}
        R, R_std, exp_x, exp_y, exp_z, std_x, std_y, std_z = _R_from_xyz(z, x, y, 2)
        assert R > 0.5
        assert isinstance(R_std, float)

    def test_returns_all_fields(self):
        z = {"00": 500, "11": 500}
        x = {"00": 500, "11": 500}
        y = {"00": 500, "11": 500}
        result = _R_from_xyz(z, x, y, 2)
        assert len(result) == 8
        R, R_std, exp_x, exp_y, exp_z, std_x, std_y, std_z = result
        assert 0 <= R <= 1.0 + 1e-6
        assert isinstance(R_std, float)


class TestQAOACost:
    def test_returns_float(self):
        ham = SparsePauliOp.from_list([("ZZ", 1.0), ("IZ", 0.5), ("ZI", -0.3)])
        counts = {"00": 400, "01": 200, "10": 200, "11": 200}
        cost = _qaoa_cost_from_counts(counts, ham, 2)
        assert isinstance(cost, float)

    def test_identity_term(self):
        ham = SparsePauliOp.from_list([("II", 1.0)])
        counts = {"00": 1000}
        cost = _qaoa_cost_from_counts(counts, ham, 2)
        assert abs(cost - 1.0) < 0.01

    def test_x_pauli_zeroes(self):
        ham = SparsePauliOp.from_list([("XI", 1.0)])
        counts = {"00": 500, "01": 500}
        cost = _qaoa_cost_from_counts(counts, ham, 2)
        assert abs(cost) < 0.01


class TestCorrelatorFromCounts:
    def test_perfect_correlation(self):
        counts = {"00": 500, "11": 500}
        c = _correlator_from_counts(counts, 0, 1)
        assert abs(c - 1.0) < 0.01

    def test_anti_correlation(self):
        counts = {"01": 500, "10": 500}
        c = _correlator_from_counts(counts, 0, 1)
        assert abs(c - (-1.0)) < 0.01

    def test_empty_counts(self):
        c = _correlator_from_counts({}, 0, 1)
        assert c == 0.0


class TestRunVQE:
    def test_returns_result(self):
        result = _run_vqe(2, maxiter=30)
        assert "vqe_energy" in result
        assert "exact_ground_energy" in result
        assert "energy_gap" in result
        assert "energy_history" in result
        assert len(result["energy_history"]) > 0


# --- Experiment function tests (all 20, minimal params) ---


class TestKuramoto4Osc:
    def test_runs(self, sim_runner):
        result = kuramoto_4osc_experiment(sim_runner, shots=100, n_time_steps=2, dt=0.05)
        assert result["experiment"] == "kuramoto_4osc"
        assert result["n_oscillators"] == 4
        assert len(result["hw_R"]) == 2
        assert len(result["hw_R_std"]) == 2
        assert len(result["classical_R"]) > 0
        assert len(result["hw_expectations"]) == 2


class TestKuramoto8Osc:
    def test_runs(self, mock_runner):
        result = kuramoto_8osc_experiment(mock_runner, shots=100, n_time_steps=2, dt=0.05)
        assert result["experiment"] == "kuramoto_8osc"
        assert result["n_oscillators"] == 8
        assert len(result["hw_R"]) == 2


class TestVQE4Q:
    def test_runs(self, sim_runner):
        result = vqe_4q_experiment(sim_runner, shots=100, maxiter=10)
        assert result["experiment"] == "vqe_4q"
        assert "vqe_energy" in result
        assert "exact_ground_energy" in result


class TestVQE8Q:
    def test_runs(self, mock_runner):
        result = vqe_8q_experiment(mock_runner, shots=100, maxiter=10)
        assert result["experiment"] == "vqe_8q"
        assert "vqe_energy" in result


class TestQAOAMPC4:
    def test_runs(self, sim_runner):
        result = qaoa_mpc_4_experiment(sim_runner, shots=100)
        assert result["experiment"] == "qaoa_mpc_4"
        assert "brute_force_cost" in result
        assert "qaoa_p1" in result
        assert "qaoa_p2" in result


class TestUPDE16Snapshot:
    def test_runs(self, mock_runner):
        from unittest.mock import patch

        fake_classical = {
            "times": np.array([0.05]),
            "R": np.array([0.8]),
            "exp_x": np.zeros((1, 16)),
            "exp_y": np.zeros((1, 16)),
        }
        with patch(
            "scpn_quantum_control.hardware.experiments.classical_exact_evolution",
            return_value=fake_classical,
        ):
            result = upde_16_snapshot_experiment(mock_runner, shots=100, trotter_steps=1)
        assert result["experiment"] == "upde_16_snapshot"
        assert result["n_layers"] == 16
        assert "hw_R" in result
        assert "classical_R" in result
        assert len(result["hw_exp_x"]) == 16


class TestKuramoto4OscZNE:
    def test_runs(self, sim_runner):
        result = kuramoto_4osc_zne_experiment(sim_runner, shots=100, dt=0.05, scales=[1, 3])
        assert result["experiment"] == "kuramoto_4osc_zne"
        assert len(result["R_per_scale"]) == 2
        assert "zne_R" in result
        assert "classical_R" in result


class TestNoiseBaseline:
    def test_runs(self, sim_runner):
        result = noise_baseline_experiment(sim_runner, shots=100)
        assert result["experiment"] == "noise_baseline"
        assert result["n_qubits"] == 4
        assert "hw_R" in result
        assert len(result["hw_exp_x"]) == 4


class TestKuramoto8OscZNE:
    def test_runs(self, mock_runner):
        result = kuramoto_8osc_zne_experiment(mock_runner, shots=100, dt=0.05, scales=[1, 3])
        assert result["experiment"] == "kuramoto_8osc_zne"
        assert result["n_oscillators"] == 8
        assert "zne_R" in result


class TestVQE8QHardware:
    def test_runs(self, mock_runner):
        result = vqe_8q_hardware_experiment(mock_runner, shots=100, maxiter=10)
        assert result["experiment"] == "vqe_8q_hardware"
        assert "sim_energy" in result
        assert "hw_energy" in result
        assert "exact_energy" in result


class TestUPDE16DD:
    def test_runs(self, mock_runner):
        from unittest.mock import patch

        fake_classical = {
            "times": np.array([0.05]),
            "R": np.array([0.8]),
            "exp_x": np.zeros((1, 16)),
            "exp_y": np.zeros((1, 16)),
        }
        with patch(
            "scpn_quantum_control.hardware.experiments.classical_exact_evolution",
            return_value=fake_classical,
        ):
            result = upde_16_dd_experiment(mock_runner, shots=100, trotter_steps=1)
        assert result["experiment"] == "upde_16_dd"
        assert "hw_R_raw" in result
        assert "hw_R_dd" in result
        assert "classical_R" in result


class TestKuramoto4OscTrotter2:
    def test_runs(self, sim_runner):
        result = kuramoto_4osc_trotter2_experiment(sim_runner, shots=100, n_time_steps=2, dt=0.05)
        assert result["experiment"] == "kuramoto_4osc_trotter2"
        assert result["trotter_order"] == 2
        assert len(result["hw_R"]) == 2


class TestSyncThreshold:
    def test_runs(self, sim_runner):
        result = sync_threshold_experiment(sim_runner, shots=100, k_values=[0.1, 0.5])
        assert result["experiment"] == "sync_threshold"
        assert len(result["results"]) == 2
        for entry in result["results"]:
            assert "K_base" in entry
            assert "hw_R" in entry
            assert "classical_R" in entry


class TestAnsatzComparisonHW:
    def test_runs(self, sim_runner):
        result = ansatz_comparison_hw_experiment(sim_runner, shots=100, maxiter=10)
        assert result["experiment"] == "ansatz_comparison_hw"
        assert len(result["comparison"]) == 3
        names = {e["ansatz"] for e in result["comparison"]}
        assert "knm_informed" in names
        assert "two_local" in names
        assert "efficient_su2" in names


class TestZNEHigherOrder:
    def test_runs(self, sim_runner):
        result = zne_higher_order_experiment(
            sim_runner, shots=100, dt=0.05, scales=[1, 3, 5], poly_order=2
        )
        assert result["experiment"] == "zne_higher_order"
        assert "order_1" in result["extrapolations"]
        assert "order_2" in result["extrapolations"]
        assert "classical_R" in result


class TestDecoherenceScaling:
    def test_runs(self, sim_runner):
        result = decoherence_scaling_experiment(sim_runner, shots=100, qubit_counts=[2, 4])
        assert result["experiment"] == "decoherence_scaling"
        assert len(result["data_points"]) == 2
        assert "fit_gamma" in result
        assert "fit_r_squared" in result


class TestVQELandscape:
    def test_runs(self, sim_runner):
        result = vqe_landscape_experiment(sim_runner, shots=100, n_samples=5)
        assert result["experiment"] == "vqe_landscape"
        assert "knm_informed" in result["landscapes"]
        assert "two_local" in result["landscapes"]
        for _name, landscape in result["landscapes"].items():
            assert "std_energy" in landscape
            assert "mean_energy" in landscape


class TestBellTest4Q:
    def test_runs(self, sim_runner):
        result = bell_test_4q_experiment(sim_runner, shots=100, maxiter=10)
        assert result["experiment"] == "bell_test_4q"
        assert "S_hw" in result
        assert "S_sim" in result
        assert isinstance(result["violates_classical_hw"], bool)


class TestCorrelator4Q:
    def test_runs(self, sim_runner):
        result = correlator_4q_experiment(sim_runner, shots=100, maxiter=10)
        assert result["experiment"] == "correlator_4q"
        assert len(result["corr_hw"]) == 4
        assert "frobenius_error" in result


class TestQKDQBER4Q:
    def test_runs(self, sim_runner):
        result = qkd_qber_4q_experiment(sim_runner, shots=100, maxiter=10)
        assert result["experiment"] == "qkd_qber_4q"
        assert "qber_z_hw" in result
        assert "qber_x_hw" in result
        assert isinstance(result["secure_hw"], bool)
        assert "key_rate_hw" in result


# --- ALL_EXPERIMENTS registry ---


class TestAllExperimentsRegistry:
    def test_has_20_entries(self):
        assert len(ALL_EXPERIMENTS) == 20

    def test_all_callables(self):
        for name, fn in ALL_EXPERIMENTS.items():
            assert callable(fn), f"{name} is not callable"
