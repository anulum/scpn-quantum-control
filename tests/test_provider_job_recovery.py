# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — durable asynchronous job recovery tests
"""Exercise persisted submission custody through the original async facade."""

import asyncio
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest
from qiskit import QuantumCircuit
from qiskit.primitives.containers import SamplerPub
from qiskit_ibm_runtime import RuntimeDecoder

from scpn_quantum_control.hardware.async_runner import AsyncHardwareRunner, AsyncJobHandle
from scpn_quantum_control.hardware.provider_job_journal import (
    ProviderJobJournal,
    SubmissionUnknownError,
)
from scpn_quantum_control.hardware.runner import HardwareRunner


class NativeJobTransport:
    """Observe real local Runtime jobs and inject only negative transport outcomes."""

    def __init__(self, job: Any, backend: Any) -> None:
        self.native = job
        self.inputs: dict[str, Any] = job.inputs
        self.target = backend
        self.cancel_calls = 0

    def job_id(self) -> str:
        """Return the actual native job identifier."""
        return str(self.native.job_id())

    def backend(self) -> Any:
        """Retain the actual original runner target."""
        return self.target

    def result(self, timeout: float | None = None) -> Any:
        """Delegate native production sampling without invented counts."""
        return self.native.result()

    def status(self) -> Any:
        """Read actual native job status."""
        return self.native.status()

    def cancel(self) -> bool:
        """Inject only cancellation acknowledgement, never confirmation."""
        self.cancel_calls += 1
        return True


@pytest.fixture
def native_transport(monkeypatch: pytest.MonkeyPatch) -> dict[str, NativeJobTransport]:
    """Adapt the network boundary to a real local SDK sampler for positive proof."""
    import qiskit_ibm_runtime

    jobs: dict[str, NativeJobTransport] = {}
    original_sampler = qiskit_ibm_runtime.SamplerV2

    class LocalSamplerTransport:
        """Retain original inputs and execute the installed native SDK."""

        def __init__(self, mode: Any) -> None:
            self.backend = mode
            self.delegate = original_sampler(mode=mode)
            self.options = self.delegate.options

        def run(self, circuits: list[Any], *, shots: int | None = None) -> NativeJobTransport:
            """Run genuine sampling and preserve its original handle."""
            job = self.delegate.run(circuits, shots=shots)
            wrapped = NativeJobTransport(job, self.backend)
            jobs[wrapped.job_id()] = wrapped
            return wrapped

    monkeypatch.setattr(qiskit_ibm_runtime, "SamplerV2", LocalSamplerTransport)
    return jobs


def connected_runner(tmp_path: Path) -> HardwareRunner:
    """Construct and connect the real original local Aer runner and transpiler."""
    runner = HardwareRunner(use_simulator=True, results_dir=str(tmp_path / "results"))
    runner.connect()
    return runner


def measured_one() -> QuantumCircuit:
    """Use the independent deterministic basis-state oracle |1>."""
    circuit = QuantumCircuit(1)
    circuit.x(0)
    circuit.measure_all()
    return circuit


def test_restart_refuses_unknown_submission(tmp_path: Path) -> None:
    """Cold reopen cannot turn a committed uncertain effect into another submit."""
    attempt_id = str(uuid4())
    journal = ProviderJobJournal(tmp_path / "journal")
    journal.prepare(
        attempt_id,
        target="ibm_exact",
        payload=b"native-qpy",
        shots=16,
        experiment="recovery",
        circuits=1,
    )
    journal.begin_submit(attempt_id)
    restored = ProviderJobJournal(tmp_path / "journal")
    assert restored.snapshot(attempt_id)["state"] == "submission_unknown"
    with pytest.raises(SubmissionUnknownError):
        restored.begin_submit(attempt_id)
    runner = AsyncHardwareRunner()
    with pytest.raises(SubmissionUnknownError):
        runner.recover_job(restored, attempt_id)


def test_provider_job_recovery_01(
    tmp_path: Path, native_transport: dict[str, NativeJobTransport]
) -> None:
    """Repeated original retrieval preserves one actual SDK submission and counts."""
    runner = connected_runner(tmp_path)
    async_runner = AsyncHardwareRunner(runner)
    journal = ProviderJobJournal(tmp_path / "journal")
    attempt_id = str(uuid4())

    async def execute() -> None:
        handle = await async_runner.submit_one_async(
            [measured_one()], shots=32, journal=journal, attempt_id=attempt_id
        )
        first = await async_runner.wait_for_job_async(handle)
        assert first[0].counts is not None
        first[0].counts["1"] = 1
        second = await async_runner.wait_for_job_async(handle)
        assert second[0].counts == {"1": 32}
        assert len(native_transport) == 1
        cold = async_runner.recover_job(ProviderJobJournal(tmp_path / "journal"), attempt_id)
        third = await async_runner.wait_for_job_async(cold)
        assert third[0].counts == {"1": 32}
        assert third[0].job_id == handle.job_id
        assert len(native_transport) == 1
        assert journal.snapshot(attempt_id)["billing"] == "unknown"
        events = journal.snapshot(attempt_id)["events"]
        assert isinstance(events, list)
        encoded = next(
            event["observation"]["native_runtime_json"]
            for event in events
            if "native_runtime_json" in event["observation"]
        )
        raw = json.loads(encoded, cls=RuntimeDecoder)
        assert raw[0].data.meas.get_counts() == {"1": 32}

    asyncio.run(execute())


def test_provider_job_recovery_03(
    tmp_path: Path, native_transport: dict[str, NativeJobTransport]
) -> None:
    """Acknowledged cancellation does not defeat genuine completed output."""
    runner = connected_runner(tmp_path)
    async_runner = AsyncHardwareRunner(runner)
    journal = ProviderJobJournal(tmp_path / "journal")
    attempt_id = str(uuid4())

    async def execute() -> None:
        handle = await async_runner.submit_one_async(
            [measured_one()], shots=16, journal=journal, attempt_id=attempt_id
        )
        native_transport[handle.job_id].result()
        cancelled = await async_runner.cancel_job_async(handle)
        assert cancelled["state"] == "cancellation_requested"
        assert cancelled["billing"] == "unknown"
        output = await async_runner.wait_for_job_async(handle)
        assert output[0].counts == {"1": 16}
        terminal = await async_runner.cancel_job_async(handle)
        assert terminal["state"] == "completed"
        assert native_transport[handle.job_id].cancel_calls == 1
        events = journal.snapshot(attempt_id)["events"]
        assert isinstance(events, list)
        assert sum(event["state"] == "completed" for event in events) == 1

    asyncio.run(execute())


def test_provider_job_recovery_02(tmp_path: Path) -> None:
    """Actual child sampling followed by response loss leaves cold restart uncertain."""
    attempt_id = str(uuid4())
    root = tmp_path / "journal"
    script = """
import asyncio,json,os,sys
from pathlib import Path
from types import SimpleNamespace
from typing import NoReturn
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler
import qiskit_ibm_runtime
from qiskit_ibm_runtime import RuntimeEncoder
from scpn_quantum_control.hardware.async_runner import AsyncHardwareRunner
from scpn_quantum_control.hardware.provider_job_journal import ProviderJobJournal
from scpn_quantum_control.hardware.runner import HardwareRunner
root=Path(sys.argv[1])
def lose_native_response(circuits: list[QuantumCircuit], *, shots: int) -> NoReturn:
    'Perform actual local SDK sampling, then lose only the return transport.'
    job=StatevectorSampler(seed=47).run(circuits,shots=shots)
    result=job.result()
    (root.parent/'provider_observation.json').write_text(json.dumps({
        'provider_job_id':job.job_id(), 'producer':'StatevectorSampler',
        'native_result':json.dumps(result,cls=RuntimeEncoder),
    }))
    os._exit(23)
def native_transport(*,mode: object) -> SimpleNamespace:
    'Adapt only the transport boundary to the installed local SDK.'
    return SimpleNamespace(options=SimpleNamespace(default_shots=16),run=lose_native_response)
qiskit_ibm_runtime.SamplerV2=native_transport
runner=HardwareRunner(use_simulator=True,results_dir=str(root.parent/'results'))
runner.connect()
circuit=QuantumCircuit(1)
circuit.x(0)
circuit.measure_all()
asyncio.run(AsyncHardwareRunner(runner).submit_one_async(
    [circuit],shots=16,journal=ProviderJobJournal(root),attempt_id=sys.argv[2],
))
"""
    child = subprocess.run(
        [sys.executable, "-c", script, str(root), attempt_id],
        env=dict(os.environ),
        check=False,
        timeout=15,
    )
    assert child.returncode == 23
    evidence = json.loads((tmp_path / "provider_observation.json").read_text())
    assert evidence["producer"] == "StatevectorSampler"
    assert evidence["provider_job_id"]
    raw = json.loads(evidence["native_result"], cls=RuntimeDecoder)
    assert raw[0].data.meas.get_counts() == {"1": 16}
    cold = ProviderJobJournal(root)
    assert cold.snapshot(attempt_id)["state"] == "submission_unknown"
    assert cold.snapshot(attempt_id)["provider_job_id"] is None
    assert cold.snapshot(attempt_id)["billing"] == "unknown"
    with pytest.raises(SubmissionUnknownError):
        cold.begin_submit(attempt_id)
    with pytest.raises(SubmissionUnknownError):
        AsyncHardwareRunner().recover_job(cold, attempt_id)


def test_lost_response_reconciles_exact_original_native_job(
    tmp_path: Path,
    native_transport: dict[str, NativeJobTransport],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An actual native effect followed by response loss recovers without submit."""
    import qiskit_ibm_runtime

    original_sampler = qiskit_ibm_runtime.SamplerV2

    class LostResponseSampler:
        """Inject response loss strictly after original SDK job creation."""

        def __init__(self, mode: Any) -> None:
            """Retain the actual local SDK boundary and its unchanged options."""
            self.delegate = original_sampler(mode=mode)
            self.options = self.delegate.options

        def run(self, circuits: list[Any], *, shots: int | None = None) -> NativeJobTransport:
            """Execute genuine sampling, then drop only its return transport."""
            self.delegate.run(circuits, shots=shots)
            raise ConnectionError("response lost after acceptance")

    monkeypatch.setattr(qiskit_ibm_runtime, "SamplerV2", LostResponseSampler)
    runner = connected_runner(tmp_path)
    async_runner = AsyncHardwareRunner(runner)
    journal = ProviderJobJournal(tmp_path / "journal")
    attempt_id = str(uuid4())

    async def submit() -> None:
        with pytest.raises(ConnectionError):
            await async_runner.submit_one_async(
                [measured_one()], shots=32, journal=journal, attempt_id=attempt_id
            )

    asyncio.run(submit())
    assert len(native_transport) == 1
    assert journal.snapshot(attempt_id)["state"] == "submission_unknown"
    job_id, native = next(iter(native_transport.items()))
    monkeypatch.setattr(runner, "retrieve_job", lambda identity: native_transport[identity])
    recovered = async_runner.recover_job(journal, attempt_id, provider_job_id=job_id)
    result = asyncio.run(async_runner.wait_for_job_async(recovered))
    assert result[0].counts == {"1": 32}
    assert len(native_transport) == 1
    assert journal.snapshot(attempt_id)["billing"] == "unknown"
    assert native.job_id() == recovered.job_id


def test_recovery_refuses_changed_native_shots(
    tmp_path: Path,
    native_transport: dict[str, NativeJobTransport],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real retained handle with altered public input cannot be rebound."""
    runner = connected_runner(tmp_path)
    async_runner = AsyncHardwareRunner(runner)
    journal = ProviderJobJournal(tmp_path / "journal")
    attempt_id = str(uuid4())
    handle = asyncio.run(
        async_runner.submit_one_async(
            [measured_one()], shots=16, journal=journal, attempt_id=attempt_id
        )
    )
    native = native_transport[handle.job_id]
    native.inputs["options"] = {"default_shots": 32}
    monkeypatch.setattr(runner, "retrieve_job", lambda identity: native_transport[identity])
    before = journal.snapshot(attempt_id)
    with pytest.raises(ValueError, match="shots differ"):
        async_runner.recover_job(journal, attempt_id)
    assert journal.snapshot(attempt_id) == before
    assert len(native_transport) == 1


@pytest.mark.parametrize("mutation", ["target", "handle", "payload", "bindings", "pub_shots"])
def test_recovery_checks_each_original_native_identity(
    tmp_path: Path,
    native_transport: dict[str, NativeJobTransport],
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    """A mismatched native provider envelope leaves durable custody unchanged."""
    runner = connected_runner(tmp_path)
    async_runner = AsyncHardwareRunner(runner)
    journal = ProviderJobJournal(tmp_path / "journal")
    identity = str(uuid4())
    handle = asyncio.run(
        async_runner.submit_one_async(
            [measured_one()], shots=16, journal=journal, attempt_id=identity
        )
    )
    native = native_transport[handle.job_id]
    monkeypatch.setattr(runner, "retrieve_job", lambda _: native)
    source_circuit = SamplerPub.coerce(native.inputs["pubs"][0]).circuit
    if mutation == "target":
        native.target = SimpleNamespace(name="other_target")
    elif mutation == "handle":
        monkeypatch.setattr(native, "job_id", lambda: "other_handle")
    elif mutation == "payload":
        altered = QuantumCircuit(1)
        altered.measure_all()
        native.inputs["pubs"] = [SamplerPub.coerce((altered, {}, 16))]
    elif mutation == "bindings":
        from qiskit.circuit import Parameter

        parameterized = QuantumCircuit(1)
        parameterized.ry(Parameter("theta"), 0)
        parameterized.measure_all()
        native.inputs["pubs"] = [SamplerPub.coerce((parameterized, [[0.5]], 16))]
    else:
        native.inputs["pubs"] = [(source_circuit, {}, 32)]
    before = journal.snapshot(identity)
    with pytest.raises(ValueError):
        async_runner.recover_job(journal, identity)
    assert journal.snapshot(identity) == before
    assert len(native_transport) == 1


def test_expired_provider_session_preserves_original_handle(
    tmp_path: Path,
    native_transport: dict[str, NativeJobTransport],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed read-only lookup cannot reset custody or cause another submit."""
    runner = connected_runner(tmp_path)
    async_runner = AsyncHardwareRunner(runner)
    journal = ProviderJobJournal(tmp_path / "journal")
    identity = str(uuid4())
    handle = asyncio.run(
        async_runner.submit_one_async(
            [measured_one()], shots=16, journal=journal, attempt_id=identity
        )
    )

    def expired(_: str) -> Any:
        raise PermissionError("expired provider session")

    monkeypatch.setattr(runner, "retrieve_job", expired)
    before = journal.snapshot(identity)
    with pytest.raises(PermissionError):
        async_runner.recover_job(journal, identity)
    assert journal.snapshot(identity) == before
    assert before["provider_job_id"] == handle.job_id
    assert before["billing"] == "unknown"
    assert len(native_transport) == 1


def test_native_tuple_publication_recovers_exact_request(
    tmp_path: Path,
    native_transport: dict[str, NativeJobTransport],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Native PUB tuple representation retains exact compiled QPY and shots."""
    runner = connected_runner(tmp_path)
    async_runner = AsyncHardwareRunner(runner)
    journal = ProviderJobJournal(tmp_path / "journal")
    identity = str(uuid4())
    handle = asyncio.run(
        async_runner.submit_one_async(
            [measured_one()], shots=16, journal=journal, attempt_id=identity
        )
    )
    native = native_transport[handle.job_id]
    native.inputs["pubs"] = [(SamplerPub.coerce(native.inputs["pubs"][0]).circuit, {}, 16)]
    monkeypatch.setattr(runner, "retrieve_job", lambda _: native)
    cold = async_runner.recover_job(journal, identity)
    results = asyncio.run(async_runner.wait_for_job_async(cold))
    assert results[0].counts == {"1": 16}
    assert len(native_transport) == 1


def test_cancelled_awaiter_retains_dispatch_capacity_and_durable_handle(
    tmp_path: Path,
    native_transport: dict[str, NativeJobTransport],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancelling an awaiter cannot orphan or duplicate the running native effect."""
    import qiskit_ibm_runtime

    original = qiskit_ibm_runtime.SamplerV2
    entered = threading.Event()
    release = threading.Event()

    class PausedTransport:
        """Pause only the transport boundary while actual SDK execution remains original."""

        def __init__(self, mode: Any) -> None:
            self.delegate = original(mode=mode)
            self.options = self.delegate.options

        def run(self, circuits: list[Any], *, shots: int | None = None) -> NativeJobTransport:
            """Expose an in-flight effect then execute original local sampling."""
            entered.set()
            assert release.wait(timeout=5)
            result = self.delegate.run(circuits, shots=shots)
            assert isinstance(result, NativeJobTransport)
            return result

    monkeypatch.setattr(qiskit_ibm_runtime, "SamplerV2", PausedTransport)
    runner = connected_runner(tmp_path)
    async_runner = AsyncHardwareRunner(runner, max_concurrent=1)
    journal = ProviderJobJournal(tmp_path / "journal")
    identity = str(uuid4())
    original_circuit = measured_one()

    async def execute() -> None:
        task = asyncio.create_task(
            async_runner.submit_one_async(
                [original_circuit], shots=16, journal=journal, attempt_id=identity
            )
        )
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        assert journal.snapshot(identity)["state"] == "submission_unknown"
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert len(native_transport) == 1
        stored = journal.snapshot(identity)
        assert stored["provider_job_id"] in native_transport
        assert stored["state"] == "submitted"
        monkeypatch.setattr(runner, "retrieve_job", lambda job_id: native_transport[job_id])
        recovered = async_runner.recover_job(journal, identity)
        output = await async_runner.wait_for_job_async(recovered)
        assert output[0].counts == {"1": 16}
        with pytest.raises(SubmissionUnknownError):
            await async_runner.submit_one_async(
                [original_circuit], shots=16, journal=journal, attempt_id=identity
            )
        assert len(native_transport) == 1

    asyncio.run(execute())


def test_public_durable_methods_refuse_unbound_or_changed_handles(
    tmp_path: Path,
    native_transport: dict[str, NativeJobTransport],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No observer, cancel or result call may act through a changed durable handle."""
    runner = connected_runner(tmp_path)
    async_runner = AsyncHardwareRunner(runner)
    journal = ProviderJobJournal(tmp_path / "journal")
    identity = str(uuid4())

    async def execute() -> None:
        with pytest.raises(ValueError, match="both journal"):
            await async_runner.submit_one_async([measured_one()], journal=journal)
        unbound = AsyncJobHandle(job_id="unbound", runner=runner, experiment="unbound")
        with pytest.raises(ValueError, match="durable handle"):
            await async_runner.observe_job_async(unbound)
        with pytest.raises(ValueError, match="durable handle"):
            await async_runner.cancel_job_async(unbound)
        handle = await async_runner.submit_one_async(
            [measured_one()], shots=16, journal=journal, attempt_id=identity
        )
        observed = await async_runner.observe_job_async(handle)
        assert observed["state"] == "submitted"
        original_id = handle.job_id
        handle.job_id = "foreign"
        with pytest.raises(ValueError, match="handle differs"):
            await async_runner.wait_for_job_async(handle)
        handle.job_id = original_id
        monkeypatch.setattr(native_transport[original_id], "job_id", lambda: "foreign")
        with pytest.raises(ValueError, match="native result handle"):
            await async_runner.wait_for_job_async(handle)

    asyncio.run(execute())


def test_completed_recovery_checks_route_and_explicit_handle(
    tmp_path: Path, native_transport: dict[str, NativeJobTransport]
) -> None:
    """Cached completion still binds its original target and provider identity."""
    runner = connected_runner(tmp_path)
    async_runner = AsyncHardwareRunner(runner)
    journal = ProviderJobJournal(tmp_path / "journal")
    identity = str(uuid4())
    handle = asyncio.run(
        async_runner.submit_one_async(
            [measured_one()], shots=16, journal=journal, attempt_id=identity
        )
    )
    asyncio.run(async_runner.wait_for_job_async(handle))
    other = HardwareRunner(results_dir=str(tmp_path / "other"))
    with pytest.raises(ValueError, match="target differs"):
        async_runner.recover_job(journal, identity, runner=other)
    with pytest.raises(ValueError, match="handle differs"):
        async_runner.recover_job(journal, identity, provider_job_id="foreign")
    terminal = asyncio.run(async_runner.observe_job_async(handle))
    assert terminal["state"] == "completed"
    assert len(native_transport) == 1


@pytest.mark.parametrize("size", [0, 257])
def test_durable_batch_budget_refuses_before_native_work(
    tmp_path: Path,
    native_transport: dict[str, NativeJobTransport],
    size: int,
) -> None:
    """An inadmissible batch needs neither connected transpiler nor provider effect."""
    runner = HardwareRunner(results_dir=str(tmp_path / "results"))
    async_runner = AsyncHardwareRunner(runner)
    journal = ProviderJobJournal(tmp_path / "journal")
    identity = str(uuid4())
    with pytest.raises(ValueError, match="between one and256"):
        asyncio.run(
            async_runner.submit_one_async(
                [measured_one()] * size, journal=journal, attempt_id=identity
            )
        )
    assert not native_transport
    with pytest.raises(KeyError):
        journal.snapshot(identity)


def test_late_native_completion_retains_confirmed_cancellation_observation(
    tmp_path: Path,
    native_transport: dict[str, NativeJobTransport],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A racing cancellation status cannot discard genuine returned SDK samples."""
    runner = connected_runner(tmp_path)
    async_runner = AsyncHardwareRunner(runner)
    journal = ProviderJobJournal(tmp_path / "journal")
    identity = str(uuid4())
    handle = asyncio.run(
        async_runner.submit_one_async(
            [measured_one()], shots=16, journal=journal, attempt_id=identity
        )
    )
    native = native_transport[handle.job_id]
    native.result()
    monkeypatch.setattr(native, "status", lambda: "CANCELLED")
    cancelled = asyncio.run(async_runner.cancel_job_async(handle))
    assert cancelled["state"] == "cancelled"
    output = asyncio.run(async_runner.wait_for_job_async(handle))
    assert output[0].counts == {"1": 16}
    final = journal.snapshot(identity)
    assert final["state"] == "completed"
    assert final["billing"] == "unknown"
    events = final["events"]
    assert isinstance(events, list)
    assert any(event["state"] == "cancelled" for event in events)
    assert any("native_runtime_json" in event["observation"] for event in events)
    assert len(native_transport) == 1


def test_original_sampler_executes_local_durable_batch_without_transport_replacement(
    tmp_path: Path,
) -> None:
    """The unchanged installed SamplerV2 and original runner own positive execution."""
    runner = connected_runner(tmp_path)
    async_runner = AsyncHardwareRunner(runner)
    journal = ProviderJobJournal(tmp_path / "journal")
    identity = str(uuid4())

    async def execute() -> None:
        handle = await async_runner.submit_one_async(
            [measured_one()], shots=16, journal=journal, attempt_id=identity
        )
        results = await async_runner.wait_for_job_async(handle)
        assert results[0].counts == {"1": 16}
        cached = await async_runner.wait_for_job_async(handle)
        assert cached[0].job_id == handle.job_id
        assert cached[0].counts == {"1": 16}
        assert journal.snapshot(identity)["state"] == "completed"
        assert journal.snapshot(identity)["billing"] == "unknown"

    asyncio.run(execute())


def test_native_sdk_wire_roundtrip_keeps_original_inputs_recoverable(
    tmp_path: Path,
    native_transport: dict[str, NativeJobTransport],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The original Runtime wire codec preserves actual PUB and BindingsArray values."""
    from qiskit_ibm_runtime import RuntimeEncoder

    runner = connected_runner(tmp_path)
    async_runner = AsyncHardwareRunner(runner)
    journal = ProviderJobJournal(tmp_path / "journal")
    identity = str(uuid4())
    handle = asyncio.run(
        async_runner.submit_one_async(
            [measured_one()], shots=16, journal=journal, attempt_id=identity
        )
    )
    native = native_transport[handle.job_id]
    encoded = json.dumps(native.inputs, cls=RuntimeEncoder)
    native.inputs = json.loads(encoded, cls=RuntimeDecoder)
    monkeypatch.setattr(runner, "retrieve_job", lambda _: native)
    recovered = async_runner.recover_job(journal, identity)
    output = asyncio.run(async_runner.wait_for_job_async(recovered))
    assert output[0].counts == {"1": 16}
    assert recovered.job_id == handle.job_id
    assert len(native_transport) == 1
