# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Hardware Runner Mock Tests
"""Mock-based tests for IBM Quantum hardware runner paths."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from qiskit import QuantumCircuit

from scpn_quantum_control.hardware.runner import HardwareRunner, JobResult


@pytest.fixture()
def tmp_results(tmp_path):
    return str(tmp_path / "results")


class TestJobResult:
    def test_to_dict_full(self):
        jr = JobResult(
            job_id="abc123",
            backend_name="ibm_fez",
            experiment_name="test",
            counts={"00": 500, "11": 500},
            expectation_values=np.array([0.5, -0.3]),
            wall_time_s=1.5,
            timestamp="2026-03-28T12:00:00",
            metadata={"depth": 10},
        )
        d = jr.to_dict()
        assert d["job_id"] == "abc123"
        assert d["counts"] == {"00": 500, "11": 500}
        assert d["expectation_values"] == [0.5, -0.3]

    def test_to_dict_minimal(self):
        jr = JobResult(
            job_id="x",
            backend_name="sim",
            experiment_name="e",
        )
        d = jr.to_dict()
        assert "counts" not in d
        assert "expectation_values" not in d


class TestHardwareRunnerInit:
    def test_results_dir_fallback(self, monkeypatch):
        """If results_dir creation fails, falls back to tempdir."""
        bad_path = "/nonexistent_root_1234567890/results"
        runner = HardwareRunner(results_dir=bad_path)
        assert runner.results_dir.exists()

    def test_backend_name_not_connected(self, tmp_results):
        runner = HardwareRunner(results_dir=tmp_results)
        assert runner.backend_name == "not_connected"


class TestHardwareConnect:
    def test_connect_ibm_with_token(self, tmp_results):
        mock_service = MagicMock()
        mock_backend = MagicMock()
        mock_backend.name = "ibm_fez"
        mock_backend.num_qubits = 156
        mock_service.backend.return_value = mock_backend
        mock_service_cls = MagicMock(return_value=mock_service)

        runner = HardwareRunner(
            token="fake_token",
            backend_name="ibm_fez",
            results_dir=tmp_results,
        )
        with (
            patch.dict(
                "sys.modules",
                {
                    "qiskit_ibm_runtime": MagicMock(QiskitRuntimeService=mock_service_cls),
                },
            ),
            patch(
                "scpn_quantum_control.hardware.runner.generate_preset_pass_manager",
                return_value=MagicMock(),
            ),
        ):
            runner.connect()

        assert runner._backend is mock_backend

    def test_connect_ibm_least_busy(self, tmp_results):
        mock_service = MagicMock()
        mock_backend = MagicMock()
        mock_backend.name = "ibm_brisbane"
        mock_backend.num_qubits = 127
        mock_service.return_value = mock_service
        mock_service.least_busy.return_value = mock_backend
        mock_service_cls = MagicMock(return_value=mock_service)

        runner = HardwareRunner(results_dir=tmp_results)
        with (
            patch.dict(
                "sys.modules",
                {
                    "qiskit_ibm_runtime": MagicMock(QiskitRuntimeService=mock_service_cls),
                },
            ),
            patch(
                "scpn_quantum_control.hardware.runner.generate_preset_pass_manager",
                return_value=MagicMock(),
            ),
        ):
            runner.connect()

        assert runner._backend is mock_backend

    def test_retrieve_job_requires_connect(self, tmp_results):
        runner = HardwareRunner(results_dir=tmp_results)
        with pytest.raises(RuntimeError, match="connect"):
            runner.retrieve_job("job_123")

    def test_retrieve_job(self, tmp_results):
        runner = HardwareRunner(results_dir=tmp_results)
        runner._service = MagicMock()
        runner._service.job.return_value = MagicMock(name="job_obj")
        runner.retrieve_job("job_123")
        runner._service.job.assert_called_once_with("job_123")


class TestHardwareRunSampler:
    def test_run_sampler_hardware(self, tmp_results):
        runner = HardwareRunner(results_dir=tmp_results)
        runner.use_simulator = False
        runner._backend = MagicMock()
        runner._backend.name = "ibm_fez"
        runner._pm = MagicMock()
        runner._pm.run.side_effect = lambda qc: qc
        runner.results_dir = Path(tmp_results)
        runner.results_dir.mkdir(parents=True, exist_ok=True)

        mock_job = MagicMock()
        mock_job.job_id.return_value = "hw_job_001"
        pub_result = MagicMock()
        pub_result.data.meas.get_counts.return_value = {"00": 500, "11": 500}
        mock_job.result.return_value = [pub_result]

        mock_sampler_cls = MagicMock()
        mock_sampler_inst = MagicMock()
        mock_sampler_inst.run.return_value = mock_job
        mock_sampler_cls.return_value = mock_sampler_inst

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()

        with patch.dict(
            "sys.modules",
            {
                "qiskit_ibm_runtime": MagicMock(SamplerV2=mock_sampler_cls),
            },
        ):
            results = runner.run_sampler(qc, shots=1000, name="test_hw")

        assert len(results) == 1
        assert results[0].job_id == "hw_job_001"
        assert results[0].counts == {"00": 500, "11": 500}


class TestHardwareRunEstimator:
    def test_run_estimator_hardware(self, tmp_results):
        from qiskit.quantum_info import SparsePauliOp

        runner = HardwareRunner(results_dir=tmp_results)
        runner.use_simulator = False
        runner._backend = MagicMock()
        runner._backend.name = "ibm_fez"
        runner._pm = MagicMock()
        runner._pm.run.side_effect = lambda qc: qc
        runner.results_dir = Path(tmp_results)
        runner.results_dir.mkdir(parents=True, exist_ok=True)

        mock_job = MagicMock()
        mock_job.job_id.return_value = "hw_est_001"
        pub_result = MagicMock()
        pub_result.data.evs = np.array([0.5])
        mock_job.result.return_value = [pub_result]

        mock_estimator_cls = MagicMock()
        mock_estimator_inst = MagicMock()
        mock_estimator_inst.run.return_value = mock_job
        mock_estimator_cls.return_value = mock_estimator_inst

        qc = QuantumCircuit(2)
        qc.h(0)
        obs = [SparsePauliOp.from_list([("ZZ", 1.0)])]

        with patch.dict(
            "sys.modules",
            {
                "qiskit_ibm_runtime": MagicMock(EstimatorV2=mock_estimator_cls),
            },
        ):
            result = runner.run_estimator(qc, obs, name="test_est")

        assert result.job_id == "hw_est_001"

    def test_run_estimator_hardware_with_params(self, tmp_results):
        from qiskit.quantum_info import SparsePauliOp

        runner = HardwareRunner(results_dir=tmp_results)
        runner.use_simulator = False
        runner._backend = MagicMock()
        runner._backend.name = "ibm_fez"
        runner._pm = MagicMock()
        runner._pm.run.side_effect = lambda qc: qc
        runner.results_dir = Path(tmp_results)
        runner.results_dir.mkdir(parents=True, exist_ok=True)

        mock_job = MagicMock()
        mock_job.job_id.return_value = "hw_est_002"
        pub_result = MagicMock()
        pub_result.data.evs = np.array([0.3])
        mock_job.result.return_value = [pub_result]

        mock_estimator_cls = MagicMock()
        mock_estimator_inst = MagicMock()
        mock_estimator_inst.run.return_value = mock_job
        mock_estimator_cls.return_value = mock_estimator_inst

        qc = QuantumCircuit(2)
        obs = [SparsePauliOp.from_list([("ZZ", 1.0)])]

        with patch.dict(
            "sys.modules",
            {
                "qiskit_ibm_runtime": MagicMock(EstimatorV2=mock_estimator_cls),
            },
        ):
            result = runner.run_estimator(
                qc, obs, parameter_values=[[0.1, 0.2]], name="test_params"
            )

        assert result.job_id == "hw_est_002"


class TestSaveResult:
    def test_save_single(self, tmp_results):
        runner = HardwareRunner(results_dir=tmp_results)
        jr = JobResult(
            job_id="j1",
            backend_name="sim",
            experiment_name="exp1",
            counts={"0": 100},
        )
        path = runner.save_result(jr, filename="test.json")
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["job_id"] == "j1"

    def test_save_batch(self, tmp_results):
        runner = HardwareRunner(results_dir=tmp_results)
        batch = [
            JobResult(job_id="j1", backend_name="sim", experiment_name="e1"),
            JobResult(job_id="j2", backend_name="sim", experiment_name="e2"),
        ]
        path = runner.save_result(batch)
        assert path.exists()

    def test_save_token(self):
        mock_service_cls = MagicMock()
        with patch.dict(
            "sys.modules",
            {
                "qiskit_ibm_runtime": MagicMock(QiskitRuntimeService=mock_service_cls),
            },
        ):
            HardwareRunner.save_token("fake_token", instance="inst/grp/proj")
        mock_service_cls.save_account.assert_called_once()


class TestLogJob:
    def test_log_job_creates_file(self, tmp_results):
        runner = HardwareRunner(results_dir=tmp_results)
        runner._log_job("job123", "my_exp")
        jobs_path = runner.results_dir / "jobs.json"
        assert jobs_path.exists()
        entries = json.loads(jobs_path.read_text())
        assert len(entries) == 1
        assert entries[0]["job_id"] == "job123"

    def test_log_job_appends(self, tmp_results):
        runner = HardwareRunner(results_dir=tmp_results)
        runner._log_job("job1", "exp1")
        runner._log_job("job2", "exp2")
        entries = json.loads((runner.results_dir / "jobs.json").read_text())
        assert len(entries) == 2
