# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Hardware Runner Coverage Tests
"""Coverage tests for hardware.runner — simulator paths, all methods."""

from __future__ import annotations

import json
from unittest.mock import patch

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from scpn_quantum_control.hardware.runner import HardwareRunner, JobResult

# --- JobResult ---


class TestJobResult:
    def test_to_dict_with_counts(self):
        jr = JobResult(
            job_id="j1",
            backend_name="aer",
            experiment_name="exp1",
            counts={"00": 500, "11": 500},
            wall_time_s=1.5,
            timestamp="2026-03-26T12:00:00",
            metadata={"depth": 10},
        )
        d = jr.to_dict()
        assert d["job_id"] == "j1"
        assert d["backend"] == "aer"
        assert d["experiment"] == "exp1"
        assert d["counts"] == {"00": 500, "11": 500}
        assert d["wall_time_s"] == 1.5
        assert "expectation_values" not in d

    def test_to_dict_with_expectation_values(self):
        evs = np.array([0.5, -0.3])
        jr = JobResult(
            job_id="j2",
            backend_name="aer",
            experiment_name="exp2",
            expectation_values=evs,
            wall_time_s=2.0,
            timestamp="2026-03-26T12:00:00",
        )
        d = jr.to_dict()
        assert d["expectation_values"] == [0.5, -0.3]
        assert "counts" not in d

    def test_to_dict_minimal(self):
        jr = JobResult(job_id="j3", backend_name="aer", experiment_name="exp3")
        d = jr.to_dict()
        assert d["job_id"] == "j3"
        assert "counts" not in d
        assert "expectation_values" not in d


# --- HardwareRunner init ---


class TestRunnerInit:
    def test_simulator_init(self, tmp_path):
        runner = HardwareRunner(use_simulator=True, results_dir=str(tmp_path / "res"))
        assert runner.use_simulator is True
        assert runner.results_dir.exists()

    def test_default_resilience_level(self, tmp_path):
        runner = HardwareRunner(use_simulator=True, results_dir=str(tmp_path / "res"))
        assert runner.resilience_level == 2

    def test_fractional_gates_default(self, tmp_path):
        runner = HardwareRunner(use_simulator=True, results_dir=str(tmp_path / "res"))
        assert runner.use_fractional_gates is True

    def test_optimization_level(self, tmp_path):
        runner = HardwareRunner(
            use_simulator=True,
            optimization_level=3,
            results_dir=str(tmp_path / "res"),
        )
        assert runner.optimization_level == 3

    def test_results_dir_fallback(self, tmp_path):
        runner = HardwareRunner(
            use_simulator=True,
            results_dir=str(tmp_path / "no_such_dir" / "deep" / "path"),
        )
        assert runner.results_dir.exists()

    def test_custom_noise_model(self, tmp_path):
        runner = HardwareRunner(
            use_simulator=True,
            noise_model="placeholder",
            results_dir=str(tmp_path / "res"),
        )
        assert runner._noise_model == "placeholder"

    def test_custom_token_channel_instance(self, tmp_path):
        runner = HardwareRunner(
            token="fake_token",
            channel="ibm_quantum",
            instance="my/inst/ance",
            results_dir=str(tmp_path / "res"),
        )
        assert runner.token == "fake_token"
        assert runner.channel == "ibm_quantum"
        assert runner.instance == "my/inst/ance"


# --- Properties ---


class TestRunnerProperties:
    def test_backend_none_before_connect(self, tmp_path):
        runner = HardwareRunner(use_simulator=True, results_dir=str(tmp_path / "res"))
        assert runner.backend is None

    def test_backend_name_before_connect(self, tmp_path):
        runner = HardwareRunner(use_simulator=True, results_dir=str(tmp_path / "res"))
        assert runner.backend_name == "not_connected"

    def test_backend_after_connect(self, tmp_path):
        runner = HardwareRunner(use_simulator=True, results_dir=str(tmp_path / "res"))
        runner.connect()
        assert runner.backend is not None

    def test_backend_name_after_connect(self, tmp_path):
        runner = HardwareRunner(use_simulator=True, results_dir=str(tmp_path / "res"))
        runner.connect()
        assert "aer" in runner.backend_name.lower() or runner.backend_name != "not_connected"


# --- Connect ---


class TestRunnerConnect:
    def test_connect_simulator(self, tmp_path):
        runner = HardwareRunner(use_simulator=True, results_dir=str(tmp_path / "res"))
        runner.connect()
        assert runner._backend is not None
        assert runner._pm is not None

    def test_connect_simulator_with_noise_model(self, tmp_path):
        from qiskit_aer.noise import NoiseModel

        nm = NoiseModel()
        runner = HardwareRunner(
            use_simulator=True,
            noise_model=nm,
            results_dir=str(tmp_path / "res"),
        )
        runner.connect()
        assert runner._backend is not None

    def test_connect_simulator_no_fractional_gates(self, tmp_path):
        runner = HardwareRunner(
            use_simulator=True,
            use_fractional_gates=False,
            results_dir=str(tmp_path / "res"),
        )
        runner.connect()
        assert runner._backend is not None


# --- Transpile ---


class TestRunnerTranspile:
    def test_transpile_raises_before_connect(self, tmp_path):
        runner = HardwareRunner(use_simulator=True, results_dir=str(tmp_path / "res"))
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        with pytest.raises(RuntimeError, match="connect"):
            runner.transpile(qc)

    def test_transpile_returns_circuit(self, tmp_path):
        runner = HardwareRunner(use_simulator=True, results_dir=str(tmp_path / "res"))
        runner.connect()
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        isa = runner.transpile(qc)
        assert isinstance(isa, QuantumCircuit)

    def test_transpile_observable(self, tmp_path):
        runner = HardwareRunner(use_simulator=True, results_dir=str(tmp_path / "res"))
        runner.connect()
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        isa = runner.transpile(qc)
        obs = SparsePauliOp.from_list([("ZZ", 1.0)])
        mapped = runner.transpile_observable(obs, isa)
        assert isinstance(mapped, SparsePauliOp)

    def test_circuit_stats(self, tmp_path):
        runner = HardwareRunner(use_simulator=True, results_dir=str(tmp_path / "res"))
        runner.connect()
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        isa = runner.transpile(qc)
        stats = runner.circuit_stats(isa)
        assert "depth" in stats
        assert "n_qubits" in stats
        assert "ecr_gates" in stats
        assert "total_gates" in stats
        assert stats["n_qubits"] == 2


# --- Run Sampler (simulator) ---


class TestRunnerSampler:
    def test_run_sampler_single_circuit(self, tmp_path):
        runner = HardwareRunner(use_simulator=True, results_dir=str(tmp_path / "res"))
        runner.connect()
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        results = runner.run_sampler(qc, shots=1000, name="test_sampler")
        assert len(results) == 1
        assert results[0].counts is not None
        assert sum(results[0].counts.values()) == 1000

    def test_run_sampler_multiple_circuits(self, tmp_path):
        runner = HardwareRunner(use_simulator=True, results_dir=str(tmp_path / "res"))
        runner.connect()
        circuits = []
        for _ in range(3):
            qc = QuantumCircuit(2)
            qc.h(0)
            qc.cx(0, 1)
            qc.measure_all()
            circuits.append(qc)
        results = runner.run_sampler(circuits, shots=500, name="test_batch")
        assert len(results) == 3
        for r in results:
            assert r.counts is not None
            assert r.backend_name == "aer_simulator"

    def test_run_sampler_fields(self, tmp_path):
        runner = HardwareRunner(use_simulator=True, results_dir=str(tmp_path / "res"))
        runner.connect()
        qc = QuantumCircuit(1)
        qc.x(0)
        qc.measure_all()
        results = runner.run_sampler(qc, shots=100, name="field_test")
        r = results[0]
        assert r.job_id.startswith("sim_")
        assert r.wall_time_s >= 0
        assert r.timestamp != ""
        assert "depth" in r.metadata


# --- Run Estimator (simulator) ---


class TestRunnerEstimator:
    def test_run_estimator_basic(self, tmp_path):
        runner = HardwareRunner(use_simulator=True, results_dir=str(tmp_path / "res"))
        runner.connect()
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        obs = [SparsePauliOp.from_list([("ZZ", 1.0)])]
        result = runner.run_estimator(qc, obs, name="test_est")
        assert result.expectation_values is not None
        assert abs(result.expectation_values[0] - 1.0) < 0.01

    def test_run_estimator_with_parameters(self, tmp_path):
        from qiskit.circuit import Parameter

        runner = HardwareRunner(use_simulator=True, results_dir=str(tmp_path / "res"))
        runner.connect()
        theta = Parameter("theta")
        qc = QuantumCircuit(1)
        qc.ry(theta, 0)
        obs = [SparsePauliOp.from_list([("Z", 1.0)])]
        result = runner.run_estimator(
            qc, obs, parameter_values=[[0.0], [3.14159]], name="param_test"
        )
        assert result.expectation_values is not None
        assert result.expectation_values.shape[0] == 2

    def test_run_estimator_fields(self, tmp_path):
        runner = HardwareRunner(use_simulator=True, results_dir=str(tmp_path / "res"))
        runner.connect()
        qc = QuantumCircuit(1)
        qc.x(0)
        obs = [SparsePauliOp.from_list([("Z", 1.0)])]
        r = runner.run_estimator(qc, obs, name="est_fields")
        assert r.job_id == "sim_estimator"
        assert r.backend_name == "aer_simulator"
        assert r.wall_time_s >= 0


# --- ZNE ---


class TestRunnerZNE:
    def test_run_estimator_zne(self, tmp_path):
        runner = HardwareRunner(use_simulator=True, results_dir=str(tmp_path / "res"))
        runner.connect()
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        obs = [SparsePauliOp.from_list([("ZZ", 1.0)])]
        zne_result = runner.run_estimator_zne(qc, obs, scales=[1, 3], order=1, name="zne_test")
        assert hasattr(zne_result, "zero_noise_estimate")
        assert hasattr(zne_result, "fit_residual")


# --- Dynamical Decoupling ---


class TestRunnerDD:
    def test_transpile_with_dd_default(self, tmp_path):
        runner = HardwareRunner(use_simulator=True, results_dir=str(tmp_path / "res"))
        runner.connect()
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        dd_circuit = runner.transpile_with_dd(qc)
        assert isinstance(dd_circuit, QuantumCircuit)

    def test_transpile_with_dd_custom_sequence(self, tmp_path):
        runner = HardwareRunner(use_simulator=True, results_dir=str(tmp_path / "res"))
        runner.connect()
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        dd_circuit = runner.transpile_with_dd(qc, dd_sequence=["x", "y", "x", "y"])
        assert isinstance(dd_circuit, QuantumCircuit)

    def test_transpile_with_dd_raises_before_connect(self, tmp_path):
        runner = HardwareRunner(use_simulator=True, results_dir=str(tmp_path / "res"))
        qc = QuantumCircuit(2)
        qc.h(0)
        with pytest.raises(RuntimeError, match="connect"):
            runner.transpile_with_dd(qc)


# --- Log and Save ---


class TestRunnerLogSave:
    def test_log_job_creates_file(self, tmp_path):
        runner = HardwareRunner(use_simulator=True, results_dir=str(tmp_path / "res"))
        runner.connect()
        runner._log_job("job_123", "my_experiment")
        jobs_path = runner.results_dir / "jobs.json"
        assert jobs_path.exists()
        with open(jobs_path) as f:
            entries = json.load(f)
        assert len(entries) == 1
        assert entries[0]["job_id"] == "job_123"

    def test_log_job_appends(self, tmp_path):
        runner = HardwareRunner(use_simulator=True, results_dir=str(tmp_path / "res"))
        runner.connect()
        runner._log_job("job_1", "exp1")
        runner._log_job("job_2", "exp2")
        with open(runner.results_dir / "jobs.json") as f:
            entries = json.load(f)
        assert len(entries) == 2

    def test_save_result_single(self, tmp_path):
        runner = HardwareRunner(use_simulator=True, results_dir=str(tmp_path / "res"))
        runner.connect()
        jr = JobResult(
            job_id="j1",
            backend_name="aer",
            experiment_name="my_exp",
            counts={"00": 100},
        )
        path = runner.save_result(jr)
        assert path.exists()
        with open(path) as f:
            data = json.load(f)
        assert data["job_id"] == "j1"

    def test_save_result_list(self, tmp_path):
        runner = HardwareRunner(use_simulator=True, results_dir=str(tmp_path / "res"))
        runner.connect()
        results = [
            JobResult(job_id="j1", backend_name="aer", experiment_name="exp1"),
            JobResult(job_id="j2", backend_name="aer", experiment_name="exp2"),
        ]
        path = runner.save_result(results, filename="batch.json")
        assert path.exists()
        with open(path) as f:
            data = json.load(f)
        assert len(data) == 2

    def test_save_result_custom_filename(self, tmp_path):
        runner = HardwareRunner(use_simulator=True, results_dir=str(tmp_path / "res"))
        runner.connect()
        jr = JobResult(job_id="j1", backend_name="aer", experiment_name="exp")
        path = runner.save_result(jr, filename="custom.json")
        assert path.name == "custom.json"


# --- Retrieve Job (error path) ---


class TestRunnerRetrieveJob:
    def test_retrieve_job_raises_without_service(self, tmp_path):
        runner = HardwareRunner(use_simulator=True, results_dir=str(tmp_path / "res"))
        runner.connect()
        with pytest.raises(RuntimeError, match="connect.*hardware"):
            runner.retrieve_job("fake_job_id")


# --- Save Token (mocked) ---


class TestRunnerSaveToken:
    def test_save_token_calls_qiskit(self):
        try:
            import qiskit_ibm_runtime  # noqa: F401
        except ImportError:
            pytest.skip("qiskit-ibm-runtime not installed")
        with patch("qiskit_ibm_runtime.QiskitRuntimeService") as mock_runtime:
            HardwareRunner.save_token("test_token", instance="ibm-q/open/main")
            mock_runtime.save_account.assert_called_once()
