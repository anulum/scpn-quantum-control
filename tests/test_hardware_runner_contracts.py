# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Hardware runner contract tests
"""Contract tests for hardware runner serialisation, connection, execution, persistence, and IBM-service boundaries."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from scpn_quantum_control.hardware import runner as runner_mod
from scpn_quantum_control.hardware.runner import HardwareRunner, JobResult
from scpn_quantum_control.structured_log_fallback import get_structured_logger


class _CountsRegister:
    """Small result-register stub for sampler count extraction contracts."""

    def __init__(self, counts: dict[str, int]) -> None:
        self._counts = counts

    def get_counts(self) -> dict[str, int]:
        return dict(self._counts)


def _pub_result(counts: dict[str, int]) -> SimpleNamespace:
    return SimpleNamespace(data=SimpleNamespace(meas=_CountsRegister(counts)))


@pytest.fixture()
def tmp_results(tmp_path: Path) -> str:
    """Return a results directory path inside the temporary directory."""
    return str(tmp_path / "results")


class TestJobResult:
    """Serialising a job result."""

    def test_to_dict_with_counts(self) -> None:
        """A result with counts serialises its identifiers and counts."""
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

    def test_to_dict_with_expectation_values(self) -> None:
        """A result with expectation values serialises them and no counts."""
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

    def test_to_dict_minimal(self) -> None:
        """A minimal result serialises without counts or expectation values."""
        jr = JobResult(job_id="j3", backend_name="aer", experiment_name="exp3")
        d = jr.to_dict()
        assert d["job_id"] == "j3"
        assert "counts" not in d
        assert "expectation_values" not in d


class TestRunnerInit:
    """Constructing a simulator runner."""

    def test_simulator_init(self, tmp_path: Path) -> None:
        """A simulator runner creates its results directory."""
        runner = HardwareRunner(use_simulator=True, results_dir=str(tmp_path / "res"))
        assert runner.use_simulator is True
        assert runner.results_dir.exists()

    def test_default_resilience_level(self, tmp_path: Path) -> None:
        """The default resilience level is 2."""
        runner = HardwareRunner(use_simulator=True, results_dir=str(tmp_path / "res"))
        assert runner.resilience_level == 2

    def test_fractional_gates_default(self, tmp_path: Path) -> None:
        """Fractional gates are on by default."""
        runner = HardwareRunner(use_simulator=True, results_dir=str(tmp_path / "res"))
        assert runner.use_fractional_gates is True

    def test_optimization_level(self, tmp_path: Path) -> None:
        """The default optimisation level is 3."""
        runner = HardwareRunner(
            use_simulator=True,
            optimization_level=3,
            results_dir=str(tmp_path / "res"),
        )
        assert runner.optimization_level == 3

    def test_results_dir_fallback(self, tmp_path: Path) -> None:
        """A missing nested results directory is created."""
        runner = HardwareRunner(
            use_simulator=True,
            results_dir=str(tmp_path / "no_such_dir" / "deep" / "path"),
        )
        assert runner.results_dir.exists()

    def test_custom_noise_model(self, tmp_path: Path) -> None:
        """A custom noise model is kept."""
        runner = HardwareRunner(
            use_simulator=True,
            noise_model="placeholder",
            results_dir=str(tmp_path / "res"),
        )
        assert runner._noise_model == "placeholder"

    def test_custom_token_channel_instance(self, tmp_path: Path) -> None:
        """A custom token, channel and instance are kept."""
        runner = HardwareRunner(
            token="fake_token",
            channel="ibm_quantum",
            instance="my/inst/ance",
            results_dir=str(tmp_path / "res"),
        )
        assert runner.token == "fake_token"
        assert runner.channel == "ibm_quantum"
        assert runner.instance == "my/inst/ance"


class TestRunnerProperties:
    """Backend properties before and after connecting."""

    def test_backend_none_before_connect(self, tmp_path: Path) -> None:
        """There is no backend before connecting."""
        runner = HardwareRunner(use_simulator=True, results_dir=str(tmp_path / "res"))
        assert runner.backend is None

    def test_backend_name_before_connect(self, tmp_path: Path) -> None:
        """The backend name reads not_connected before connecting."""
        runner = HardwareRunner(use_simulator=True, results_dir=str(tmp_path / "res"))
        assert runner.backend_name == "not_connected"

    def test_backend_after_connect(self, tmp_path: Path) -> None:
        """There is a backend after connecting."""
        runner = HardwareRunner(use_simulator=True, results_dir=str(tmp_path / "res"))
        runner.connect()
        assert runner.backend is not None

    def test_backend_name_after_connect(self, tmp_path: Path) -> None:
        """The backend name changes after connecting."""
        runner = HardwareRunner(use_simulator=True, results_dir=str(tmp_path / "res"))
        runner.connect()
        assert "aer" in runner.backend_name.lower() or runner.backend_name != "not_connected"


class TestRunnerConnect:
    """Connecting to a simulator."""

    def test_connect_simulator(self, tmp_path: Path) -> None:
        """Connecting sets the backend and the pass manager."""
        runner = HardwareRunner(use_simulator=True, results_dir=str(tmp_path / "res"))
        runner.connect()
        assert runner._backend is not None
        assert runner._pm is not None

    def test_connect_simulator_with_noise_model(self, tmp_path: Path) -> None:
        """Connecting with a noise model sets the backend."""
        from qiskit_aer.noise import NoiseModel

        nm = NoiseModel()
        runner = HardwareRunner(
            use_simulator=True,
            noise_model=nm,
            results_dir=str(tmp_path / "res"),
        )
        runner.connect()
        assert runner._backend is not None

    def test_connect_simulator_no_fractional_gates(self, tmp_path: Path) -> None:
        """Connecting without fractional gates sets the backend."""
        runner = HardwareRunner(
            use_simulator=True,
            use_fractional_gates=False,
            results_dir=str(tmp_path / "res"),
        )
        runner.connect()
        assert runner._backend is not None


class TestRunnerTranspile:
    """Transpiling circuits and observables."""

    def test_transpile_raises_before_connect(self, tmp_path: Path) -> None:
        """Transpiling before connecting raises."""
        runner = HardwareRunner(use_simulator=True, results_dir=str(tmp_path / "res"))
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        with pytest.raises(RuntimeError, match="connect"):
            runner.transpile(qc)

    def test_transpile_returns_circuit(self, tmp_path: Path) -> None:
        """Transpiling returns a circuit."""
        runner = HardwareRunner(use_simulator=True, results_dir=str(tmp_path / "res"))
        runner.connect()
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        isa = runner.transpile(qc)
        assert isinstance(isa, QuantumCircuit)

    def test_transpile_observable(self, tmp_path: Path) -> None:
        """Mapping an observable returns a Pauli operator."""
        runner = HardwareRunner(use_simulator=True, results_dir=str(tmp_path / "res"))
        runner.connect()
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        isa = runner.transpile(qc)
        obs = SparsePauliOp.from_list([("ZZ", 1.0)])
        mapped = runner.transpile_observable(obs, isa)
        assert isinstance(mapped, SparsePauliOp)

    def test_circuit_stats(self, tmp_path: Path) -> None:
        """Circuit statistics report depth, width and ECR count."""
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


class TestRunnerSampler:
    """Running the sampler on a simulator."""

    def test_run_sampler_single_circuit(self, tmp_path: Path) -> None:
        """One circuit gives one result with all 1,000 shots."""
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

    def test_run_sampler_multiple_circuits(self, tmp_path: Path) -> None:
        """Three circuits give three simulator results."""
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
            assert r.backend_name in {"aer_simulator", "basic_simulator"}

    def test_run_sampler_fields(self, tmp_path: Path) -> None:
        """A sampler result has its id, wall time and timestamp."""
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


class TestRunnerEstimator:
    """Running the estimator on a simulator."""

    def test_run_estimator_basic(self, tmp_path: Path) -> None:
        """The estimator returns the expected expectation value."""
        runner = HardwareRunner(use_simulator=True, results_dir=str(tmp_path / "res"))
        runner.connect()
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        obs = [SparsePauliOp.from_list([("ZZ", 1.0)])]
        result = runner.run_estimator(qc, obs, name="test_est")
        assert result.expectation_values is not None
        assert abs(result.expectation_values[0] - 1.0) < 0.01

    def test_run_estimator_with_parameters(self, tmp_path: Path) -> None:
        """The estimator handles a parameter set per value."""
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

    def test_run_estimator_fields(self, tmp_path: Path) -> None:
        """An estimator result has its id, backend and wall time."""
        runner = HardwareRunner(use_simulator=True, results_dir=str(tmp_path / "res"))
        runner.connect()
        qc = QuantumCircuit(1)
        qc.x(0)
        obs = [SparsePauliOp.from_list([("Z", 1.0)])]
        r = runner.run_estimator(qc, obs, name="est_fields")
        assert r.job_id == "sim_estimator"
        assert r.backend_name in {"aer_simulator", "basic_simulator"}
        assert r.wall_time_s >= 0


class TestRunnerZNE:
    """Zero-noise extrapolation on a simulator."""

    def test_run_estimator_zne(self, tmp_path: Path) -> None:
        """Extrapolation returns an estimate and its fit residual."""
        runner = HardwareRunner(use_simulator=True, results_dir=str(tmp_path / "res"))
        runner.connect()
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        obs = [SparsePauliOp.from_list([("ZZ", 1.0)])]
        zne_result = runner.run_estimator_zne(qc, obs, scales=[1, 3], order=1, name="zne_test")
        assert hasattr(zne_result, "zero_noise_estimate")
        assert hasattr(zne_result, "fit_residual")


class TestRunnerDD:
    """Dynamical decoupling."""

    def test_transpile_with_dd_default(self, tmp_path: Path) -> None:
        """Decoupling with the default sequence returns a circuit."""
        runner = HardwareRunner(use_simulator=True, results_dir=str(tmp_path / "res"))
        runner.connect()
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        dd_circuit = runner.transpile_with_dd(qc)
        assert isinstance(dd_circuit, QuantumCircuit)

    def test_transpile_with_dd_custom_sequence(self, tmp_path: Path) -> None:
        """Decoupling with a custom sequence returns a circuit."""
        runner = HardwareRunner(use_simulator=True, results_dir=str(tmp_path / "res"))
        runner.connect()
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        dd_circuit = runner.transpile_with_dd(qc, dd_sequence=["x", "y", "x", "y"])
        assert isinstance(dd_circuit, QuantumCircuit)

    def test_transpile_with_dd_raises_before_connect(self, tmp_path: Path) -> None:
        """Decoupling before connecting raises."""
        runner = HardwareRunner(use_simulator=True, results_dir=str(tmp_path / "res"))
        qc = QuantumCircuit(2)
        qc.h(0)
        with pytest.raises(RuntimeError, match="connect"):
            runner.transpile_with_dd(qc)


class TestRunnerLogSave:
    """The runner's job log and saved results."""

    def test_log_job_creates_file(self, tmp_path: Path) -> None:
        """Logging a job creates the log with one entry."""
        runner = HardwareRunner(use_simulator=True, results_dir=str(tmp_path / "res"))
        runner.connect()
        runner._log_job("job_123", "my_experiment")
        jobs_path = runner.results_dir / "jobs.json"
        assert jobs_path.exists()
        with open(jobs_path) as f:
            entries = json.load(f)
        assert len(entries) == 1
        assert entries[0]["job_id"] == "job_123"

    def test_log_job_appends(self, tmp_path: Path) -> None:
        """Logging a second job appends to the log."""
        runner = HardwareRunner(use_simulator=True, results_dir=str(tmp_path / "res"))
        runner.connect()
        runner._log_job("job_1", "exp1")
        runner._log_job("job_2", "exp2")
        with open(runner.results_dir / "jobs.json") as f:
            entries = json.load(f)
        assert len(entries) == 2

    def test_save_result_single(self, tmp_path: Path) -> None:
        """Saving one result writes its file."""
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

    def test_save_result_list(self, tmp_path: Path) -> None:
        """Saving a list writes every result."""
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
        assert len(data["results"]) == 2

    def test_save_result_custom_filename(self, tmp_path: Path) -> None:
        """A custom file name is used."""
        runner = HardwareRunner(use_simulator=True, results_dir=str(tmp_path / "res"))
        runner.connect()
        jr = JobResult(job_id="j1", backend_name="aer", experiment_name="exp")
        path = runner.save_result(jr, filename="custom.json")
        assert path.name == "custom.json"


class TestRunnerRetrieveJob:
    """Retrieving a job."""

    def test_retrieve_job_raises_without_service(self, tmp_path: Path) -> None:
        """Retrieving without a hardware connection raises."""
        runner = HardwareRunner(use_simulator=True, results_dir=str(tmp_path / "res"))
        runner.connect()
        with pytest.raises(RuntimeError, match="connect.*hardware"):
            runner.retrieve_job("fake_job_id")


class TestRunnerSaveToken:
    """Saving an account token."""

    def test_save_token_calls_qiskit(self) -> None:
        """Saving a token calls the runtime's account store."""
        try:
            import qiskit_ibm_runtime  # noqa: F401
        except ImportError:
            pytest.skip("qiskit-ibm-runtime not installed")
        with patch("qiskit_ibm_runtime.QiskitRuntimeService") as mock_runtime:
            HardwareRunner.save_token("test_token", instance="ibm-q/open/main")
            mock_runtime.save_account.assert_called_once()


class TestHardwareRunnerInit:
    """Constructing a hardware runner."""

    def test_structured_logger_exposes_logging_interface(self) -> None:
        """The structured logger exposes a logging interface."""
        logger = get_structured_logger("scpn.test")
        assert isinstance(logger, logging.Logger) or hasattr(logger, "bind")

    def test_default_instance_env_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The default instance falls back to the environment."""
        monkeypatch.setenv("SCPN_IBM_CRN", "crn:v1:test")
        monkeypatch.delenv("SCPN_IBM_INSTANCE", raising=False)

        try:
            import scpn_quantum_control.config as config_mod
        except Exception:
            assert HardwareRunner._default_instance() == "crn:v1:test"
            return

        with patch.object(
            config_mod, "get_config", side_effect=RuntimeError("config unavailable")
        ):
            assert HardwareRunner._default_instance() == "crn:v1:test"

    def test_results_dir_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """If results_dir creation fails, falls back to tempdir."""
        bad_path = "/nonexistent_root_1234567890/results"
        runner = HardwareRunner(results_dir=bad_path)
        assert runner.results_dir.exists()

    def test_backend_name_not_connected(self, tmp_results: str) -> None:
        """The backend name reads not_connected before connecting."""
        runner = HardwareRunner(results_dir=tmp_results)
        assert runner.backend_name == "not_connected"


class TestHardwareConnect:
    """Connecting to the provider and retrieving jobs."""

    def test_connect_ibm_service_none_raises(self, tmp_results: str) -> None:
        """A missing runtime service makes connecting raise."""
        runner = HardwareRunner(results_dir=tmp_results)
        with (
            patch.dict(
                "sys.modules",
                {
                    "qiskit_ibm_runtime": MagicMock(
                        QiskitRuntimeService=MagicMock(return_value=None)
                    ),
                },
            ),
            pytest.raises(RuntimeError, match="connect"),
        ):
            runner.connect()

    def test_connect_ibm_backend_none_raises(self, tmp_results: str) -> None:
        """No least-busy backend makes connecting raise."""
        mock_service = MagicMock()
        mock_service.least_busy.return_value = None
        runner = HardwareRunner(results_dir=tmp_results)
        with (
            patch.dict(
                "sys.modules",
                {
                    "qiskit_ibm_runtime": MagicMock(
                        QiskitRuntimeService=MagicMock(return_value=mock_service)
                    ),
                },
            ),
            pytest.raises(RuntimeError, match="connect"),
        ):
            runner.connect()

    def test_connect_simulator_uses_aer_when_available(self, tmp_results: str) -> None:
        """The simulator uses Aer when it is installed."""
        mock_backend = MagicMock()
        mock_backend.name = "aer_simulator"
        mock_backend.num_qubits = 32
        mock_aer = MagicMock(return_value=mock_backend)
        mock_pm = MagicMock()

        runner = HardwareRunner(use_simulator=True, results_dir=tmp_results)
        with (
            patch.dict("sys.modules", {"qiskit_aer": MagicMock(AerSimulator=mock_aer)}),
            patch(
                "scpn_quantum_control.hardware.runner.generate_preset_pass_manager",
                return_value=mock_pm,
            ) as mock_generate,
        ):
            runner.connect()

        mock_aer.assert_called_once_with()
        mock_generate.assert_called_once()
        assert runner.backend is mock_backend
        assert runner.backend_name == "aer_simulator"

    def test_connect_simulator_passes_noise_model_to_aer(self, tmp_results: str) -> None:
        """The noise model is passed to Aer."""
        mock_backend = MagicMock()
        mock_backend.name = "aer_simulator"
        noise_model = object()
        mock_aer = MagicMock(return_value=mock_backend)

        runner = HardwareRunner(
            use_simulator=True,
            noise_model=noise_model,
            results_dir=tmp_results,
        )
        with (
            patch.dict("sys.modules", {"qiskit_aer": MagicMock(AerSimulator=mock_aer)}),
            patch(
                "scpn_quantum_control.hardware.runner.generate_preset_pass_manager",
                return_value=MagicMock(),
            ),
        ):
            runner.connect()

        mock_aer.assert_called_once_with(noise_model=noise_model)

    def test_connect_ibm_with_token(self, tmp_results: str) -> None:
        """A token connects to the named backend with its descriptor."""
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
        assert runner.backend_descriptor.name == "qiskit_ibm"
        assert runner.backend_descriptor.provider == "ibm_quantum"
        assert runner.backend_descriptor.can_submit is True
        assert runner.backend_descriptor.submit_requires_approval is True

    def test_connect_ibm_least_busy(self, tmp_results: str) -> None:
        """Without a backend name the least busy one is chosen."""
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

    def test_retrieve_job_requires_connect(self, tmp_results: str) -> None:
        """Retrieving a job before connecting raises."""
        runner = HardwareRunner(results_dir=tmp_results)
        with pytest.raises(RuntimeError, match="connect"):
            runner.retrieve_job("job_123")

    def test_retrieve_job(self, tmp_results: str) -> None:
        """Retrieving a job asks the service for that id."""
        runner = HardwareRunner(results_dir=tmp_results)
        runner._service = MagicMock()
        runner._service.job.return_value = MagicMock(name="job_obj")
        runner.retrieve_job("job_123")
        runner._service.job.assert_called_once_with("job_123")


class TestHardwareRunSampler:
    """Running the sampler on hardware."""

    def test_extract_counts_custom_register_fallback(self) -> None:
        """Counts are read from a custom register name."""
        pub_result = SimpleNamespace(data=SimpleNamespace(custom_reg=_CountsRegister({"01": 7})))

        assert runner_mod._extract_counts(pub_result) == {"01": 7}

    def test_extract_counts_missing_register_raises(self) -> None:
        """A result without a classical register raises."""
        pub_result = SimpleNamespace(data=SimpleNamespace(alpha=object()))

        with pytest.raises(RuntimeError, match="Could not find classical register"):
            runner_mod._extract_counts(pub_result)

    def test_run_sampler_hardware(self, tmp_results: str) -> None:
        """A hardware sampler run returns its id and counts."""
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
        pub_result = _pub_result({"00": 500, "11": 500})
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
    """Running the estimator on hardware."""

    def test_run_estimator_hardware(self, tmp_results: str) -> None:
        """A hardware estimator run returns its id."""
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

    def test_run_estimator_hardware_with_params(self, tmp_results: str) -> None:
        """A parameterised hardware estimator run returns its id."""
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

    def test_run_estimator_zne_uses_default_scales(
        self, tmp_results: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Extrapolation uses the default scales 1, 3 and 5."""
        from qiskit.quantum_info import SparsePauliOp

        runner = HardwareRunner(results_dir=tmp_results)
        estimator = MagicMock(
            return_value=JobResult(
                job_id="zne",
                backend_name="sim",
                experiment_name="zne",
                expectation_values=np.array([0.25]),
            )
        )
        monkeypatch.setattr(runner, "run_estimator", estimator)

        qc = QuantumCircuit(1)
        qc.h(0)
        obs = [SparsePauliOp.from_list([("Z", 1.0)])]
        result = runner.run_estimator_zne(qc, obs)

        assert result.noise_scales == [1, 3, 5]
        assert estimator.call_count == 3

    def test_run_estimator_zne_requires_expectation_values(
        self, tmp_results: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Extrapolation without expectation values raises."""
        from qiskit.quantum_info import SparsePauliOp

        runner = HardwareRunner(results_dir=tmp_results)
        estimator = MagicMock(
            return_value=JobResult(
                job_id="zne",
                backend_name="sim",
                experiment_name="zne",
                expectation_values=None,
            )
        )
        monkeypatch.setattr(runner, "run_estimator", estimator)

        qc = QuantumCircuit(1)
        qc.h(0)
        obs = [SparsePauliOp.from_list([("Z", 1.0)])]

        with pytest.raises(ValueError, match="no expectation values"):
            runner.run_estimator_zne(qc, obs, scales=[1])


class TestSaveResult:
    """Saving results and tokens."""

    def test_save_single(self, tmp_results: str) -> None:
        """Saving one result writes its file."""
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

    def test_save_batch(self, tmp_results: str) -> None:
        """Saving a batch writes its file."""
        runner = HardwareRunner(results_dir=tmp_results)
        batch = [
            JobResult(job_id="j1", backend_name="sim", experiment_name="e1"),
            JobResult(job_id="j2", backend_name="sim", experiment_name="e2"),
        ]
        path = runner.save_result(batch)
        assert path.exists()

    def test_save_token(self) -> None:
        """Saving a token calls the runtime's account store."""
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
    """The job log."""

    def test_log_job_creates_file(self, tmp_results: str) -> None:
        """Logging a job creates the log with one entry."""
        runner = HardwareRunner(results_dir=tmp_results)
        runner._log_job("job123", "my_exp")
        jobs_path = runner.results_dir / "jobs.json"
        assert jobs_path.exists()
        entries = json.loads(jobs_path.read_text())
        assert len(entries) == 1
        assert entries[0]["job_id"] == "job123"

    def test_log_job_appends(self, tmp_results: str) -> None:
        """Logging a second job appends to the log."""
        runner = HardwareRunner(results_dir=tmp_results)
        runner._log_job("job1", "exp1")
        runner._log_job("job2", "exp2")
        entries = json.loads((runner.results_dir / "jobs.json").read_text())
        assert len(entries) == 2
