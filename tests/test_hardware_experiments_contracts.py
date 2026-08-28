# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Hardware experiments contract tests
"""Contract tests for hardware experiment builders, result parsers, simulator paths, and experiment registry entries."""

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


class TestBuildEvoBase:
    """Verify base evolution-circuit construction."""

    def test_returns_circuit(self):
        """Return a circuit with the requested oscillator width."""
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        qc = _build_evo_base(2, K, omega, t=0.1, trotter_reps=1)
        assert qc.num_qubits == 2

    def test_trotter_order_2(self):
        """Support second-order Trotter synthesis."""
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        qc = _build_evo_base(2, K, omega, t=0.1, trotter_reps=1, trotter_order=2)
        assert qc.num_qubits == 2

    def test_4_qubit(self):
        """Construct a four-qubit evolution circuit."""
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        qc = _build_evo_base(4, K, omega, t=0.2, trotter_reps=2)
        assert qc.num_qubits == 4


class TestBuildXYZCircuits:
    """Verify measurement-basis circuit expansion."""

    def test_returns_three_circuits(self):
        """Return one circuit for each Cartesian measurement basis."""
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        base = _build_evo_base(2, K, omega, t=0.1, trotter_reps=1)
        z_qc, x_qc, y_qc = _build_xyz_circuits(base, 2)
        assert z_qc.num_qubits == 2
        assert x_qc.num_qubits == 2
        assert y_qc.num_qubits == 2


class TestExpectationPerQubit:
    """Verify per-qubit expectation and uncertainty extraction."""

    def test_all_zeros(self):
        """Map all-zero counts to positive unit expectations."""
        counts = {"00": 1000}
        exp, std = _expectation_per_qubit(counts, 2)
        np.testing.assert_allclose(exp, [1.0, 1.0], atol=0.01)

    def test_all_ones(self):
        """Map all-one counts to negative unit expectations."""
        counts = {"11": 1000}
        exp, std = _expectation_per_qubit(counts, 2)
        np.testing.assert_allclose(exp, [-1.0, -1.0], atol=0.01)

    def test_mixed(self):
        """Map balanced opposite outcomes to zero expectations."""
        counts = {"00": 500, "11": 500}
        exp, std = _expectation_per_qubit(counts, 2)
        np.testing.assert_allclose(exp, [0.0, 0.0], atol=0.01)

    def test_std_nonzero(self):
        """Report nonzero uncertainty for balanced opposite outcomes."""
        counts = {"00": 500, "11": 500}
        _, std = _expectation_per_qubit(counts, 2)
        assert all(s > 0 for s in std)


class TestRFromXYZ:
    """Verify order-parameter reconstruction from XYZ counts."""

    def test_coherent_state(self):
        """Produce a coherent order parameter and scalar uncertainty."""
        z = {"00": 1000}
        x = {"00": 1000}
        y = {"00": 1000}
        R, R_std, exp_x, exp_y, exp_z, std_x, std_y, std_z = _R_from_xyz(z, x, y, 2)
        assert R > 0.5
        assert isinstance(R_std, float)

    def test_returns_all_fields(self):
        """Return all reconstructed means and standard deviations."""
        z = {"00": 500, "11": 500}
        x = {"00": 500, "11": 500}
        y = {"00": 500, "11": 500}
        result = _R_from_xyz(z, x, y, 2)
        assert len(result) == 8
        R, R_std, exp_x, exp_y, exp_z, std_x, std_y, std_z = result
        assert 0 <= R <= 1.0 + 1e-6
        assert isinstance(R_std, float)


class TestQAOACost:
    """Verify count-based QAOA Hamiltonian evaluation."""

    def test_returns_float(self):
        """Return a scalar cost for diagonal Pauli terms."""
        ham = SparsePauliOp.from_list([("ZZ", 1.0), ("IZ", 0.5), ("ZI", -0.3)])
        counts = {"00": 400, "01": 200, "10": 200, "11": 200}
        cost = _qaoa_cost_from_counts(counts, ham, 2)
        assert isinstance(cost, float)

    def test_identity_term(self):
        """Evaluate an identity term to its coefficient."""
        ham = SparsePauliOp.from_list([("II", 1.0)])
        counts = {"00": 1000}
        cost = _qaoa_cost_from_counts(counts, ham, 2)
        assert abs(cost - 1.0) < 0.01

    def test_x_pauli_zeroes(self):
        """Ignore non-diagonal Pauli terms in count-based evaluation."""
        ham = SparsePauliOp.from_list([("XI", 1.0)])
        counts = {"00": 500, "01": 500}
        cost = _qaoa_cost_from_counts(counts, ham, 2)
        assert abs(cost) < 0.01


class TestCorrelatorFromCounts:
    """Verify pair-correlator extraction from sampled counts."""

    def test_perfect_correlation(self):
        """Return positive unity for perfectly correlated outcomes."""
        counts = {"00": 500, "11": 500}
        c = _correlator_from_counts(counts, 0, 1)
        assert abs(c - 1.0) < 0.01

    def test_anti_correlation(self):
        """Return negative unity for perfectly anticorrelated outcomes."""
        counts = {"01": 500, "10": 500}
        c = _correlator_from_counts(counts, 0, 1)
        assert abs(c - (-1.0)) < 0.01

    def test_empty_counts(self):
        """Return zero when no observations are available."""
        c = _correlator_from_counts({}, 0, 1)
        assert c == 0.0


class TestRunVQE:
    """Verify the shared statevector VQE execution path."""

    def test_returns_result(self):
        """Return energy, gap, and optimization-history fields."""
        result = _run_vqe(2, maxiter=30)
        assert "vqe_energy" in result
        assert "exact_ground_energy" in result
        assert "energy_gap" in result
        assert "energy_history" in result
        assert len(result["energy_history"]) > 0


class TestKuramoto4Osc:
    """Verify the four-oscillator simulator experiment."""

    def test_runs(self, sim_runner):
        """Return hardware and classical trajectories for four oscillators."""
        result = kuramoto_4osc_experiment(sim_runner, shots=100, n_time_steps=2, dt=0.05)
        assert result["experiment"] == "kuramoto_4osc"
        assert result["n_oscillators"] == 4
        assert len(result["hw_R"]) == 2
        assert len(result["hw_R_std"]) == 2
        assert len(result["classical_R"]) > 0
        assert len(result["hw_expectations"]) == 2


class TestKuramoto8Osc:
    """Verify the eight-oscillator injected-runner experiment."""

    def test_runs(self, mock_runner):
        """Return a two-step eight-oscillator trajectory."""
        result = kuramoto_8osc_experiment(mock_runner, shots=100, n_time_steps=2, dt=0.05)
        assert result["experiment"] == "kuramoto_8osc"
        assert result["n_oscillators"] == 8
        assert len(result["hw_R"]) == 2


class TestVQE4Q:
    """Verify the four-qubit statevector VQE experiment."""

    def test_runs(self, sim_runner):
        """Return variational and exact energies for four qubits."""
        result = vqe_4q_experiment(sim_runner, shots=100, maxiter=10)
        assert result["experiment"] == "vqe_4q"
        assert "vqe_energy" in result
        assert "exact_ground_energy" in result


class TestVQE8Q:
    """Verify the eight-qubit statevector VQE experiment."""

    def test_runs(self, mock_runner):
        """Return variational energy fields for eight qubits."""
        result = vqe_8q_experiment(mock_runner, shots=100, maxiter=10)
        assert result["experiment"] == "vqe_8q"
        assert "vqe_energy" in result


class TestQAOAMPC4:
    """Verify the four-qubit QAOA model-predictive controller."""

    def test_runs(self, sim_runner):
        """Return brute-force and both QAOA-depth results."""
        result = qaoa_mpc_4_experiment(sim_runner, shots=100)
        assert result["experiment"] == "qaoa_mpc_4"
        assert "brute_force_cost" in result
        assert "qaoa_p1" in result
        assert "qaoa_p2" in result


class TestUPDE16Snapshot:
    """Verify the injected-runner 16-layer UPDE snapshot."""

    def test_runs(self, mock_runner):
        """Return quantum and patched classical snapshot fields."""
        from unittest.mock import patch

        fake_classical = {
            "times": np.array([0.05]),
            "R": np.array([0.8]),
            "exp_x": np.zeros((1, 16)),
            "exp_y": np.zeros((1, 16)),
        }
        with patch(
            "scpn_quantum_control.hardware.experiment_control.classical_exact_evolution",
            return_value=fake_classical,
        ):
            result = upde_16_snapshot_experiment(mock_runner, shots=100, trotter_steps=1)
        assert result["experiment"] == "upde_16_snapshot"
        assert result["n_layers"] == 16
        assert "hw_R" in result
        assert "classical_R" in result
        assert len(result["hw_exp_x"]) == 16


class TestKuramoto4OscZNE:
    """Verify four-oscillator zero-noise extrapolation."""

    def test_runs(self, sim_runner):
        """Return results at both scales and an extrapolated value."""
        result = kuramoto_4osc_zne_experiment(sim_runner, shots=100, dt=0.05, scales=[1, 3])
        assert result["experiment"] == "kuramoto_4osc_zne"
        assert len(result["R_per_scale"]) == 2
        assert "zne_R" in result
        assert "classical_R" in result


class TestNoiseBaseline:
    """Verify the local noise-baseline experiment."""

    def test_runs(self, sim_runner):
        """Return bounded four-qubit baseline observables."""
        result = noise_baseline_experiment(sim_runner, shots=100)
        assert result["experiment"] == "noise_baseline"
        assert result["n_qubits"] == 4
        assert "hw_R" in result
        assert len(result["hw_exp_x"]) == 4


class TestKuramoto8OscZNE:
    """Verify eight-oscillator zero-noise extrapolation."""

    def test_runs(self, mock_runner):
        """Return an eight-oscillator extrapolated synchronization value."""
        result = kuramoto_8osc_zne_experiment(mock_runner, shots=100, dt=0.05, scales=[1, 3])
        assert result["experiment"] == "kuramoto_8osc_zne"
        assert result["n_oscillators"] == 8
        assert "zne_R" in result


class TestVQE8QHardware:
    """Verify the injected-runner eight-qubit VQE boundary."""

    def test_runs(self, mock_runner):
        """Return simulated, injected-runner, and exact energies."""
        result = vqe_8q_hardware_experiment(mock_runner, shots=100, maxiter=10)
        assert result["experiment"] == "vqe_8q_hardware"
        assert "sim_energy" in result
        assert "hw_energy" in result
        assert "exact_energy" in result


class TestUPDE16DD:
    """Verify the injected-runner dynamical-decoupling experiment."""

    def test_runs(self, mock_runner):
        """Return raw, decoupled, and patched classical observables."""
        from unittest.mock import patch

        fake_classical = {
            "times": np.array([0.05]),
            "R": np.array([0.8]),
            "exp_x": np.zeros((1, 16)),
            "exp_y": np.zeros((1, 16)),
        }
        with patch(
            "scpn_quantum_control.hardware.experiment_mitigation.classical_exact_evolution",
            return_value=fake_classical,
        ):
            result = upde_16_dd_experiment(mock_runner, shots=100, trotter_steps=1)
        assert result["experiment"] == "upde_16_dd"
        assert "hw_R_raw" in result
        assert "hw_R_dd" in result
        assert "classical_R" in result


class TestKuramoto4OscTrotter2:
    """Verify second-order four-oscillator Trotter evolution."""

    def test_runs(self, sim_runner):
        """Return the requested second-order trajectory."""
        result = kuramoto_4osc_trotter2_experiment(sim_runner, shots=100, n_time_steps=2, dt=0.05)
        assert result["experiment"] == "kuramoto_4osc_trotter2"
        assert result["trotter_order"] == 2
        assert len(result["hw_R"]) == 2


class TestSyncThreshold:
    """Verify the synchronization-threshold sweep."""

    def test_runs(self, sim_runner):
        """Return quantum and classical values at each coupling."""
        result = sync_threshold_experiment(sim_runner, shots=100, k_values=[0.1, 0.5])
        assert result["experiment"] == "sync_threshold"
        assert len(result["results"]) == 2
        for entry in result["results"]:
            assert "K_base" in entry
            assert "hw_R" in entry
            assert "classical_R" in entry


class TestAnsatzComparisonHW:
    """Verify injected-runner VQE ansatz comparison."""

    def test_runs(self, sim_runner):
        """Compare all three configured ansatz families."""
        result = ansatz_comparison_hw_experiment(sim_runner, shots=100, maxiter=10)
        assert result["experiment"] == "ansatz_comparison_hw"
        assert len(result["comparison"]) == 3
        names = {e["ansatz"] for e in result["comparison"]}
        assert "knm_informed" in names
        assert "two_local" in names
        assert "efficient_su2" in names


class TestZNEHigherOrder:
    """Verify higher-order zero-noise extrapolation."""

    def test_runs(self, sim_runner):
        """Return linear and quadratic extrapolations."""
        result = zne_higher_order_experiment(
            sim_runner, shots=100, dt=0.05, scales=[1, 3, 5], poly_order=2
        )
        assert result["experiment"] == "zne_higher_order"
        assert "order_1" in result["extrapolations"]
        assert "order_2" in result["extrapolations"]
        assert "classical_R" in result


class TestDecoherenceScaling:
    """Verify local decoherence-scaling estimation."""

    def test_runs(self, sim_runner):
        """Return two width samples and fitted decay statistics."""
        result = decoherence_scaling_experiment(sim_runner, shots=100, qubit_counts=[2, 4])
        assert result["experiment"] == "decoherence_scaling"
        assert len(result["data_points"]) == 2
        assert "fit_gamma" in result
        assert "fit_r_squared" in result


class TestVQELandscape:
    """Verify sampled VQE landscape statistics."""

    def test_runs(self, sim_runner):
        """Return statistics for informed and generic ansatz landscapes."""
        result = vqe_landscape_experiment(sim_runner, shots=100, n_samples=5)
        assert result["experiment"] == "vqe_landscape"
        assert "knm_informed" in result["landscapes"]
        assert "two_local" in result["landscapes"]
        for _name, landscape in result["landscapes"].items():
            assert "std_energy" in landscape
            assert "mean_energy" in landscape


class TestBellTest4Q:
    """Verify the four-qubit Bell experiment."""

    def test_runs(self, sim_runner):
        """Return simulated and sampled Bell statistics."""
        result = bell_test_4q_experiment(sim_runner, shots=100, maxiter=10)
        assert result["experiment"] == "bell_test_4q"
        assert "S_hw" in result
        assert "S_sim" in result
        assert isinstance(result["violates_classical_hw"], bool)


class TestCorrelator4Q:
    """Verify the four-qubit correlator experiment."""

    def test_runs(self, sim_runner):
        """Return all four correlators and their aggregate error."""
        result = correlator_4q_experiment(sim_runner, shots=100, maxiter=10)
        assert result["experiment"] == "correlator_4q"
        assert len(result["corr_hw"]) == 4
        assert "frobenius_error" in result


class TestQKDQBER4Q:
    """Verify the four-qubit QKD error-rate experiment."""

    def test_runs(self, sim_runner):
        """Return both basis error rates and security fields."""
        result = qkd_qber_4q_experiment(sim_runner, shots=100, maxiter=10)
        assert result["experiment"] == "qkd_qber_4q"
        assert "qber_z_hw" in result
        assert "qber_x_hw" in result
        assert isinstance(result["secure_hw"], bool)
        assert "key_rate_hw" in result


class TestAllExperimentsRegistry:
    """Verify completeness of the hardware experiment registry."""

    def test_has_20_entries(self):
        """Retain all twenty registered experiment entry points."""
        assert len(ALL_EXPERIMENTS) == 20

    def test_all_callables(self):
        """Expose every registry entry as a callable."""
        for name, fn in ALL_EXPERIMENTS.items():
            assert callable(fn), f"{name} is not callable"
