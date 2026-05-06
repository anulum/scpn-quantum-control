# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — AsyncHardwareRunner tests
"""Tests for the asyncio IBM-job runner (audit C13).

All tests run against a mocked IBM Runtime. No network, no token, no
real sampler — only the async orchestration logic is exercised.
"""

from __future__ import annotations

import asyncio
import sys
import threading
import time
import types
from typing import Any
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# qiskit_ibm_runtime stubs — install before the async_runner import
# ---------------------------------------------------------------------------


class _StubSampler:
    """Mimics SamplerV2: sampler.run(circuits) returns a job."""

    def __init__(self, mode: Any) -> None:
        self.mode = mode
        self.options = MagicMock()
        self._last_circuits: list[Any] | None = None

    def run(self, circuits: list[Any]) -> MagicMock:
        self._last_circuits = circuits
        job = MagicMock()
        job.job_id.return_value = f"job_{id(circuits)}"
        # pub_result with a register that has `get_counts`.
        pub = MagicMock()
        pub.data.meas.get_counts.return_value = {"0000": 1024}
        job.result.return_value = [pub] * len(circuits)
        return job


@pytest.fixture(autouse=True)
def _stub_qiskit_ibm_runtime(monkeypatch: pytest.MonkeyPatch):
    """Install a tiny ``qiskit_ibm_runtime`` module with a SamplerV2 stub."""
    mod = types.ModuleType("qiskit_ibm_runtime")
    mod.SamplerV2 = _StubSampler  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "qiskit_ibm_runtime", mod)
    yield


# Import AFTER the stub fixture is declared so the module resolves the
# stubbed SamplerV2 when it does its lazy import.
from scpn_quantum_control.hardware import async_runner as ar  # noqa: E402

# ---------------------------------------------------------------------------
# Runner stubs
# ---------------------------------------------------------------------------


class _StubRunner:
    """Quacks like HardwareRunner for the fields async_runner touches."""

    def __init__(self, backend_name: str = "ibm_mock") -> None:
        self.backend_name = backend_name
        self._backend = MagicMock(name=f"backend({backend_name})")
        self.transpiled: list[Any] = []
        self.jobs_logged: list[tuple[str, str]] = []

    def transpile(self, circuit: Any) -> Any:
        self.transpiled.append(circuit)
        return circuit

    def _log_job(self, job_id: str, name: str) -> None:
        self.jobs_logged.append((job_id, name))


def _fake_circuits(n: int = 2) -> list[Any]:
    return [MagicMock(name=f"circ_{i}") for i in range(n)]


# ---------------------------------------------------------------------------
# Constructor contract
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_logger_falls_back_when_structured_logger_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import builtins
        import logging

        real_import = builtins.__import__

        def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name.endswith("logging_setup"):
                raise RuntimeError("structured logger unavailable")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", guarded_import)

        logger = ar._get_logger("fallback_logger")

        assert isinstance(logger, logging.Logger)
        assert logger.name == "fallback_logger"

    def test_accepts_single_runner(self) -> None:
        r = _StubRunner()
        a = ar.AsyncHardwareRunner(r)  # type: ignore[arg-type]
        assert a.n_runners == 1
        assert a.max_concurrent == 1

    def test_accepts_list_of_runners(self) -> None:
        a = ar.AsyncHardwareRunner([_StubRunner(), _StubRunner()])  # type: ignore[list-item]
        assert a.n_runners == 2
        assert a.max_concurrent == 2

    def test_max_concurrent_override(self) -> None:
        a = ar.AsyncHardwareRunner(
            [_StubRunner(), _StubRunner(), _StubRunner()],  # type: ignore[list-item]
            max_concurrent=1,
        )
        assert a.max_concurrent == 1

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            ar.AsyncHardwareRunner([])

    def test_rejects_zero_concurrent(self) -> None:
        with pytest.raises(ValueError, match="max_concurrent"):
            ar.AsyncHardwareRunner(_StubRunner(), max_concurrent=0)  # type: ignore[arg-type]

    def test_wraps_non_list_iterable_as_single_runner(self) -> None:
        runner_set = {_StubRunner("set_runner")}

        a = ar.AsyncHardwareRunner(runner_set)  # type: ignore[arg-type]

        assert a.n_runners == 1
        assert a._runners[0] is runner_set  # type: ignore[attr-defined]

    def test_next_runner_advances_round_robin_index(self) -> None:
        r1, r2 = _StubRunner("ibm_a"), _StubRunner("ibm_b")
        a = ar.AsyncHardwareRunner([r1, r2])  # type: ignore[list-item]

        assert a._next_runner() is r1
        assert a._next_runner() is r2
        assert a._next_runner() is r1
        assert a._rr_index == 3  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# submit_one_async
# ---------------------------------------------------------------------------


class TestSubmitOne:
    def test_submit_one_returns_handle(self) -> None:
        r = _StubRunner("ibm_test")
        a = ar.AsyncHardwareRunner(r)  # type: ignore[arg-type]
        before = time.time()
        handle = asyncio.run(
            a.submit_one_async(_fake_circuits(3), shots=1024, name="phase1_even"),
        )
        after = time.time()
        assert handle.job_id.startswith("job_")
        assert handle.runner is r
        assert handle.experiment == "phase1_even"
        assert before <= handle.submitted_at <= after
        assert handle._job is not None

    def test_submit_one_transpiles_all_circuits(self) -> None:
        r = _StubRunner()
        a = ar.AsyncHardwareRunner(r)  # type: ignore[arg-type]
        circs = _fake_circuits(4)
        asyncio.run(a.submit_one_async(circs))
        assert r.transpiled == circs

    def test_submit_one_logs_job(self) -> None:
        r = _StubRunner()
        a = ar.AsyncHardwareRunner(r)  # type: ignore[arg-type]
        asyncio.run(a.submit_one_async(_fake_circuits(), name="my_exp"))
        assert len(r.jobs_logged) == 1
        job_id, name = r.jobs_logged[0]
        assert name == "my_exp"
        assert job_id.startswith("job_")

    def test_submit_one_honours_explicit_runner(self) -> None:
        r1, r2 = _StubRunner("ibm_a"), _StubRunner("ibm_b")
        a = ar.AsyncHardwareRunner([r1, r2])  # type: ignore[list-item]
        handle = asyncio.run(
            a.submit_one_async(_fake_circuits(), runner=r2),  # type: ignore[arg-type]
        )
        assert handle.runner is r2


# ---------------------------------------------------------------------------
# submit_batch_async — fan-out + semaphore
# ---------------------------------------------------------------------------


class TestSubmitBatch:
    def test_batch_dispatches_round_robin(self) -> None:
        r1, r2 = _StubRunner("ibm_a"), _StubRunner("ibm_b")
        a = ar.AsyncHardwareRunner([r1, r2])  # type: ignore[list-item]
        batches = [_fake_circuits(1) for _ in range(4)]
        handles = asyncio.run(a.submit_batch_async(batches, name="dla"))
        assert [h.runner.backend_name for h in handles] == [
            "ibm_a",
            "ibm_b",
            "ibm_a",
            "ibm_b",
        ]

    def test_batch_assigns_unique_experiment_names(self) -> None:
        r = _StubRunner()
        a = ar.AsyncHardwareRunner(r)  # type: ignore[arg-type]
        batches = [_fake_circuits(1) for _ in range(3)]
        handles = asyncio.run(a.submit_batch_async(batches, name="depth7"))
        assert [h.experiment for h in handles] == ["depth7_0", "depth7_1", "depth7_2"]

    def test_batch_respects_semaphore(self) -> None:
        """With max_concurrent=1, the semaphore must serialise submissions."""
        r = _StubRunner()
        a = ar.AsyncHardwareRunner(r, max_concurrent=1)  # type: ignore[arg-type]

        submit_times: list[float] = []

        real_submit = a._submit_blocking  # type: ignore[attr-defined]

        def _slow_submit(*args: Any, **kwargs: Any) -> ar.AsyncJobHandle:
            submit_times.append(time.time())
            time.sleep(0.05)
            return real_submit(*args, **kwargs)

        a._submit_blocking = _slow_submit  # type: ignore[method-assign]

        batches = [_fake_circuits(1) for _ in range(3)]
        asyncio.run(a.submit_batch_async(batches))

        # With serialisation the spans must not overlap significantly.
        diffs = [submit_times[i + 1] - submit_times[i] for i in range(len(submit_times) - 1)]
        assert all(d >= 0.04 for d in diffs), (
            f"semaphore did not serialise; inter-submit diffs: {diffs}"
        )

    def test_batch_concurrency_beats_serial(self) -> None:
        """With max_concurrent=3, submissions must overlap semantically."""
        r = _StubRunner()
        a = ar.AsyncHardwareRunner(r, max_concurrent=3)  # type: ignore[arg-type]

        real_submit = a._submit_blocking  # type: ignore[attr-defined]
        lock = threading.Lock()
        active = 0
        max_active = 0

        def _slow_submit(*args: Any, **kwargs: Any) -> ar.AsyncJobHandle:
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
            try:
                time.sleep(0.10)
                return real_submit(*args, **kwargs)
            finally:
                with lock:
                    active -= 1

        a._submit_blocking = _slow_submit  # type: ignore[method-assign]

        batches = [_fake_circuits(1) for _ in range(3)]
        asyncio.run(a.submit_batch_async(batches))

        assert max_active == 3, f"fan-out did not parallelise; max_active={max_active}"


# ---------------------------------------------------------------------------
# wait_for_job_async + wait_all_async
# ---------------------------------------------------------------------------


class TestWaitForResults:
    def test_wait_returns_job_results(self) -> None:
        r = _StubRunner()
        a = ar.AsyncHardwareRunner(r)  # type: ignore[arg-type]
        handle = asyncio.run(a.submit_one_async(_fake_circuits(2), name="exp"))
        results = asyncio.run(a.wait_for_job_async(handle, timeout_s=123.0))
        assert len(results) == 2
        assert results[0].job_id == handle.job_id
        assert results[0].counts == {"0000": 1024}
        assert results[0].backend_name == r.backend_name
        assert results[0].experiment_name == "exp_0"
        handle._job.result.assert_called_once_with(timeout=123.0)

    def test_wait_raises_without_job(self) -> None:
        r = _StubRunner()
        a = ar.AsyncHardwareRunner(r)  # type: ignore[arg-type]
        handle = ar.AsyncJobHandle(job_id="orphan", runner=r, experiment="x")  # type: ignore[arg-type]
        with pytest.raises(RuntimeError, match="no underlying job"):
            asyncio.run(a.wait_for_job_async(handle))

    def test_wait_all_gathers_every_handle(self) -> None:
        r = _StubRunner()
        a = ar.AsyncHardwareRunner(r)  # type: ignore[arg-type]
        batches = [_fake_circuits(1) for _ in range(4)]
        handles = asyncio.run(a.submit_batch_async(batches))
        results = asyncio.run(a.wait_all_async(handles))
        assert len(results) == 4
        assert all(len(r) == 1 for r in results)


# ---------------------------------------------------------------------------
# Pipeline smoke
# ---------------------------------------------------------------------------


class TestAsyncPipelineSmoke:
    def test_pipeline_submit_batch_then_wait_all(self) -> None:
        r1, r2 = _StubRunner("ibm_a"), _StubRunner("ibm_b")
        a = ar.AsyncHardwareRunner([r1, r2])  # type: ignore[list-item]
        batches = [_fake_circuits(2) for _ in range(4)]

        async def _pipeline() -> list[list[Any]]:
            handles = await a.submit_batch_async(batches, shots=2048, name="dla")
            return await a.wait_all_async(handles)

        all_results = asyncio.run(_pipeline())
        assert len(all_results) == 4
        flat = [jr for sub in all_results for jr in sub]
        assert {jr.backend_name for jr in flat} == {"ibm_a", "ibm_b"}
        assert all(jr.counts == {"0000": 1024} for jr in flat)


class TestSubmitCircuitBatchProvenance:
    def test_zne_records_all_ibm_job_ids(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ZNE scale runs and the final counts run must all keep real IBM job ids."""

        class _FakeCircuit:
            num_clbits = 4

            def measure_all(self) -> None:
                return None

        class _FakeAnsatz:
            def build_circuit(self) -> _FakeCircuit:
                return _FakeCircuit()

        class _FakeTarget:
            def durations(self) -> list[Any]:
                return []

        class _FakeBackend:
            name = "ibm_fez"
            target = _FakeTarget()

        service_kwargs_seen: dict[str, Any] = {}

        class _FakeService:
            def __init__(self, **kwargs: Any) -> None:
                self.kwargs = kwargs
                service_kwargs_seen.update(kwargs)

            def backend(self, target: str) -> _FakeBackend:
                assert target == "ibm_fez"
                return _FakeBackend()

        class _FakePassManager:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                return None

            def run(self, circuit: Any) -> Any:
                return circuit

        class _FakeJob:
            def __init__(self, job_id: str) -> None:
                self._job_id = job_id

            def job_id(self) -> str:
                return self._job_id

            def result(self, *args: Any, **kwargs: Any) -> list[Any]:
                pub = MagicMock()
                pub.data.meas.get_counts.return_value = {"0000": 256}
                return [pub]

        class _FakeSampler:
            counter = 0

            def __init__(self, mode: Any) -> None:
                self.mode = mode
                self.options = MagicMock()

            def run(self, circuits: list[Any]) -> _FakeJob:
                type(self).counter += 1
                return _FakeJob(f"ibm_job_{type(self).counter}")

        class _FakeSyncOrderParameter:
            def __call__(self, **kwargs: Any) -> dict[str, float]:
                return {"sync_order": 0.25}

        class _FakeRichardsonFactory:
            def __init__(self, scale_factors: list[int]) -> None:
                self.scale_factors = scale_factors

        def _execute_with_zne(circuit: Any, executor: Any, **kwargs: Any) -> float:
            return sum(executor(circuit) for _ in range(3)) / 3.0

        qiskit_ibm = types.ModuleType("qiskit_ibm_runtime")
        qiskit_ibm.QiskitRuntimeService = _FakeService  # type: ignore[attr-defined]
        qiskit_ibm.SamplerV2 = _FakeSampler  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "qiskit_ibm_runtime", qiskit_ibm)

        preset = types.ModuleType("qiskit.transpiler.preset_passmanagers")
        preset.generate_preset_pass_manager = (  # type: ignore[attr-defined]
            lambda *args, **kwargs: _FakePassManager()
        )
        monkeypatch.setitem(sys.modules, "qiskit.transpiler.preset_passmanagers", preset)

        passes = types.ModuleType("qiskit.transpiler.passes")
        passes.ALAPScheduleAnalysis = _FakePassManager  # type: ignore[attr-defined]
        passes.PadDynamicalDecoupling = _FakePassManager  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "qiskit.transpiler.passes", passes)

        transpiler = types.ModuleType("qiskit.transpiler")
        transpiler.PassManager = _FakePassManager  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "qiskit.transpiler", transpiler)

        circuit_lib = types.ModuleType("qiskit.circuit.library")
        circuit_lib.XGate = object  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "qiskit.circuit.library", circuit_lib)

        mitiq = types.ModuleType("mitiq")
        mitiq_zne = types.ModuleType("mitiq.zne")
        mitiq_zne.execute_with_zne = _execute_with_zne  # type: ignore[attr-defined]
        mitiq.zne = mitiq_zne  # type: ignore[attr-defined]
        mitiq_inference = types.ModuleType("mitiq.zne.inference")
        mitiq_inference.RichardsonFactory = _FakeRichardsonFactory  # type: ignore[attr-defined]
        mitiq_scaling = types.ModuleType("mitiq.zne.scaling")
        mitiq_scaling.fold_global = lambda circuit, scale_factor: circuit  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "mitiq", mitiq)
        monkeypatch.setitem(sys.modules, "mitiq.zne", mitiq_zne)
        monkeypatch.setitem(sys.modules, "mitiq.zne.inference", mitiq_inference)
        monkeypatch.setitem(sys.modules, "mitiq.zne.scaling", mitiq_scaling)

        import scpn_quantum_control.analysis as analysis

        monkeypatch.setattr(analysis, "SyncOrderParameter", _FakeSyncOrderParameter)
        monkeypatch.setenv("SCPN_IBM_TOKEN", "test-token")
        monkeypatch.setenv("SCPN_IBM_INSTANCE", "legacy-instance")

        runner = ar.AsyncHardwareRunner(backend="ibm_fez", shots=256)
        job = runner.submit_circuit_batch(
            _FakeAnsatz(),
            lambda **kwargs: {"observable_seen": 1.0},
            enable_zne=True,
        )

        result = asyncio.run(job.result())

        assert result["job_id"] == "ibm_job_4"
        assert result["zne_job_ids"] == ["ibm_job_1", "ibm_job_2", "ibm_job_3"]
        assert result["job_ids"] == [
            "ibm_job_1",
            "ibm_job_2",
            "ibm_job_3",
            "ibm_job_4",
        ]
        assert result["zne_applied"] is True
        assert result["status"] == "DONE"
        assert service_kwargs_seen["instance"] == "legacy-instance"

    def test_submit_circuit_batch_prefers_crn_over_legacy_instance(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CRN env var is preferred; legacy instance remains fallback-only."""
        import sys
        import types

        from scpn_quantum_control.hardware import async_runner as ar

        class _FakeCircuit:
            num_clbits = 1

            def measure_all(self) -> None:
                return None

        class _FakeAnsatz:
            def build_circuit(self) -> _FakeCircuit:
                return _FakeCircuit()

        class _FakeBackend:
            name = "ibm_fez"

        service_kwargs_seen: dict[str, Any] = {}

        class _FakeService:
            def __init__(self, **kwargs: Any) -> None:
                service_kwargs_seen.update(kwargs)

            def backend(self, target: str) -> _FakeBackend:
                return _FakeBackend()

        class _FakePassManager:
            def run(self, circuit: Any) -> Any:
                return circuit

        class _FakeJob:
            def job_id(self) -> str:
                return "job_1"

            def result(self, *args: Any, **kwargs: Any) -> list[Any]:
                pub = MagicMock()
                pub.data.meas.get_counts.return_value = {"0": 1}
                return [pub]

        class _FakeSampler:
            def __init__(self, mode: Any) -> None:
                self.options = MagicMock()

            def run(self, circuits: list[Any]) -> _FakeJob:
                return _FakeJob()

        qiskit_ibm = types.ModuleType("qiskit_ibm_runtime")
        qiskit_ibm.QiskitRuntimeService = _FakeService  # type: ignore[attr-defined]
        qiskit_ibm.SamplerV2 = _FakeSampler  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "qiskit_ibm_runtime", qiskit_ibm)

        preset = types.ModuleType("qiskit.transpiler.preset_passmanagers")
        preset.generate_preset_pass_manager = (  # type: ignore[attr-defined]
            lambda *args, **kwargs: _FakePassManager()
        )
        monkeypatch.setitem(sys.modules, "qiskit.transpiler.preset_passmanagers", preset)

        passes = types.ModuleType("qiskit.transpiler.passes")
        passes.ALAPScheduleAnalysis = object  # type: ignore[attr-defined]
        passes.PadDynamicalDecoupling = object  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "qiskit.transpiler.passes", passes)

        monkeypatch.setenv("SCPN_IBM_TOKEN", "test-token")
        monkeypatch.setenv("SCPN_IBM_CRN", "preferred-crn")
        monkeypatch.setenv("SCPN_IBM_INSTANCE", "legacy-instance")

        runner = ar.AsyncHardwareRunner(backend="ibm_fez", shots=1)
        job = runner.submit_circuit_batch(
            _FakeAnsatz(),
            lambda **kwargs: {"observable_seen": 1.0},
        )
        result = asyncio.run(job.result())

        assert result["status"] == "DONE"
        assert service_kwargs_seen["instance"] == "preferred-crn"

    def test_submit_circuit_batch_without_token_returns_no_counts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No IBM token must not trigger implicit local simulation or observables."""

        class _FakeCircuit:
            num_clbits = 1

            def measure_all(self) -> None:
                return None

        class _FakeAnsatz:
            def build_circuit(self) -> _FakeCircuit:
                return _FakeCircuit()

        qiskit_ibm = types.ModuleType("qiskit_ibm_runtime")
        qiskit_ibm.QiskitRuntimeService = object  # type: ignore[attr-defined]
        qiskit_ibm.SamplerV2 = object  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "qiskit_ibm_runtime", qiskit_ibm)
        monkeypatch.delenv("SCPN_IBM_TOKEN", raising=False)

        observable = MagicMock(return_value={"observable_seen": 1.0})
        runner = ar.AsyncHardwareRunner(backend="ibm_fez", shots=1)
        job = runner.submit_circuit_batch(_FakeAnsatz(), observable)

        result = asyncio.run(job.result())

        assert result["status"] == "NO_IBM_TOKEN"
        assert result["counts_available"] is False
        assert result["job_id"] is None
        observable.assert_not_called()

    def test_submit_circuit_batch_local_simulation_requires_explicit_opt_in(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Explicit local simulation is labelled and still evaluates observables."""

        class _FakeCircuit:
            num_clbits = 1

            def measure_all(self) -> None:
                return None

        class _FakeAnsatz:
            def build_circuit(self) -> _FakeCircuit:
                return _FakeCircuit()

        class _FakeLocalJob:
            def result(self) -> list[Any]:
                pub = MagicMock()
                pub.data.meas.get_counts.return_value = {"0": 3, "1": 1}
                return [pub]

        class _FakeStatevectorSampler:
            def run(self, circuits: list[Any], shots: int) -> _FakeLocalJob:
                assert shots == 4
                return _FakeLocalJob()

        qiskit_ibm = types.ModuleType("qiskit_ibm_runtime")
        qiskit_ibm.QiskitRuntimeService = object  # type: ignore[attr-defined]
        qiskit_ibm.SamplerV2 = object  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "qiskit_ibm_runtime", qiskit_ibm)

        primitives = types.ModuleType("qiskit.primitives")
        primitives.StatevectorSampler = _FakeStatevectorSampler  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "qiskit.primitives", primitives)
        monkeypatch.delenv("SCPN_IBM_TOKEN", raising=False)

        runner = ar.AsyncHardwareRunner(backend="ibm_fez", shots=4)
        job = runner.submit_circuit_batch(
            _FakeAnsatz(),
            lambda **kwargs: {"observable_seen": float(sum(kwargs["counts"].values()))},
            allow_local_simulation=True,
        )

        result = asyncio.run(job.result())

        assert result["status"] == "DONE_LOCAL_SIMULATION"
        assert result["job_id"] == "local_simulated"
        assert result["observable_seen"] == 4.0

    def test_submit_circuit_batch_backend_fallback_zne_skip_and_timeout(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Backend fallback, ZNE skip, measurement insertion, and queue timeout stay auditable."""

        class _FakeCircuit:
            num_clbits = 0

            def __init__(self) -> None:
                self.measured = False

            def measure_all(self) -> None:
                self.measured = True

        class _FakeAnsatz:
            def __init__(self) -> None:
                self.circuit = _FakeCircuit()

            def build_circuit(self) -> _FakeCircuit:
                return self.circuit

        class _FakeBackend:
            name = "least_busy_backend"

        class _FakeService:
            def __init__(self, **kwargs: Any) -> None:
                return None

            def backend(self, target: str) -> _FakeBackend:
                raise RuntimeError("backend unavailable")

            def least_busy(self, **kwargs: Any) -> _FakeBackend:
                return _FakeBackend()

        class _FakePassManager:
            def run(self, circuit: Any) -> Any:
                return circuit

        class _FakeJob:
            def job_id(self) -> str:
                return "queued_job"

            def result(self, *args: Any, **kwargs: Any) -> list[Any]:
                raise TimeoutError("still queued")

        class _FakeSampler:
            def __init__(self, mode: Any) -> None:
                self.options = MagicMock()

            def run(self, circuits: list[Any]) -> _FakeJob:
                return _FakeJob()

        class _FakeFactory:
            def __init__(self, scale_factors: list[int]) -> None:
                self.scale_factors = scale_factors

        qiskit_ibm = types.ModuleType("qiskit_ibm_runtime")
        qiskit_ibm.QiskitRuntimeService = _FakeService  # type: ignore[attr-defined]
        qiskit_ibm.SamplerV2 = _FakeSampler  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "qiskit_ibm_runtime", qiskit_ibm)

        preset = types.ModuleType("qiskit.transpiler.preset_passmanagers")
        preset.generate_preset_pass_manager = (  # type: ignore[attr-defined]
            lambda *args, **kwargs: _FakePassManager()
        )
        monkeypatch.setitem(sys.modules, "qiskit.transpiler.preset_passmanagers", preset)

        passes = types.ModuleType("qiskit.transpiler.passes")
        passes.ALAPScheduleAnalysis = object  # type: ignore[attr-defined]
        passes.PadDynamicalDecoupling = object  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "qiskit.transpiler.passes", passes)

        mitiq = types.ModuleType("mitiq")
        mitiq_zne = types.ModuleType("mitiq.zne")

        def _raise_zne(*args: Any, **kwargs: Any) -> float:
            raise RuntimeError("zne unavailable")

        mitiq_zne.execute_with_zne = _raise_zne  # type: ignore[attr-defined]
        mitiq.zne = mitiq_zne  # type: ignore[attr-defined]
        mitiq_inference = types.ModuleType("mitiq.zne.inference")
        mitiq_inference.RichardsonFactory = _FakeFactory  # type: ignore[attr-defined]
        mitiq_scaling = types.ModuleType("mitiq.zne.scaling")
        mitiq_scaling.fold_global = lambda circuit, scale_factor: circuit  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "mitiq", mitiq)
        monkeypatch.setitem(sys.modules, "mitiq.zne", mitiq_zne)
        monkeypatch.setitem(sys.modules, "mitiq.zne.inference", mitiq_inference)
        monkeypatch.setitem(sys.modules, "mitiq.zne.scaling", mitiq_scaling)
        monkeypatch.setenv("SCPN_IBM_TOKEN", "test-token")

        ansatz = _FakeAnsatz()
        runner = ar.AsyncHardwareRunner(backend="missing_backend", shots=8)
        job = runner.submit_circuit_batch(
            ansatz,
            lambda **kwargs: {"should_not_run": 1.0},
            enable_zne=True,
        )

        result = asyncio.run(job.result())

        assert ansatz.circuit.measured is True
        assert result["job_id"] == "queued_job"
        assert result["job_ids"] == ["queued_job"]
        assert result["status"] == "QUEUED_ON_IBM"
        assert result["counts_available"] is False
        assert "should_not_run" not in result

    def test_submit_circuit_batch_submission_error_without_local_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _FakeCircuit:
            num_clbits = 1

            def measure_all(self) -> None:
                return None

        class _FakeAnsatz:
            def build_circuit(self) -> _FakeCircuit:
                return _FakeCircuit()

        class _FailingService:
            def __init__(self, **kwargs: Any) -> None:
                raise RuntimeError("service unavailable")

        qiskit_ibm = types.ModuleType("qiskit_ibm_runtime")
        qiskit_ibm.QiskitRuntimeService = _FailingService  # type: ignore[attr-defined]
        qiskit_ibm.SamplerV2 = object  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "qiskit_ibm_runtime", qiskit_ibm)
        monkeypatch.setenv("SCPN_IBM_TOKEN", "test-token")

        runner = ar.AsyncHardwareRunner(backend="ibm_fez", shots=1)
        job = runner.submit_circuit_batch(
            _FakeAnsatz(),
            lambda **kwargs: {"should_not_run": 1.0},
        )

        result = asyncio.run(job.result())

        assert result["status"] == "IBM_SUBMISSION_ERROR"
        assert result["counts_available"] is False
        assert result["job_id"] is None

    def test_submit_circuit_batch_submission_error_can_use_labelled_local_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _FakeCircuit:
            num_clbits = 1

            def measure_all(self) -> None:
                return None

        class _FakeAnsatz:
            def build_circuit(self) -> _FakeCircuit:
                return _FakeCircuit()

        class _FailingService:
            def __init__(self, **kwargs: Any) -> None:
                raise RuntimeError("service unavailable")

        class _FakeLocalJob:
            def result(self) -> list[Any]:
                pub = MagicMock()
                pub.data.meas.get_counts.return_value = {"0": 2}
                return [pub]

        class _FakeStatevectorSampler:
            def run(self, circuits: list[Any], shots: int) -> _FakeLocalJob:
                assert shots == 2
                return _FakeLocalJob()

        qiskit_ibm = types.ModuleType("qiskit_ibm_runtime")
        qiskit_ibm.QiskitRuntimeService = _FailingService  # type: ignore[attr-defined]
        qiskit_ibm.SamplerV2 = object  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "qiskit_ibm_runtime", qiskit_ibm)

        primitives = types.ModuleType("qiskit.primitives")
        primitives.StatevectorSampler = _FakeStatevectorSampler  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "qiskit.primitives", primitives)
        monkeypatch.setenv("SCPN_IBM_TOKEN", "test-token")

        runner = ar.AsyncHardwareRunner(backend="ibm_fez", shots=2)
        job = runner.submit_circuit_batch(
            _FakeAnsatz(),
            lambda **kwargs: {"observable_seen": float(sum(kwargs["counts"].values()))},
            allow_local_simulation=True,
        )

        result = asyncio.run(job.result())

        assert result["status"] == "DONE_LOCAL_SIMULATION"
        assert result["job_id"] == "local_simulated"
        assert result["observable_seen"] == 2.0
