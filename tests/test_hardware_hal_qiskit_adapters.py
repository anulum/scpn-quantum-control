# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — hardware HAL Qiskit adapters tests
# scpn-quantum-control -- Qiskit HAL adapter tests
"""Tests for concrete Qiskit adapters behind the provider-neutral HAL."""

from __future__ import annotations

import base64
import io
from collections.abc import Iterator, Sequence
from dataclasses import replace
from types import SimpleNamespace
from typing import BinaryIO, cast

import pytest
from qiskit import QuantumCircuit, qpy
from qiskit.circuit import Parameter
from qiskit.primitives.containers import BitArray, DataBin, SamplerPubResult
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import SamplerV2

from scpn_quantum_control.hardware.hal import HardwareAbstractionLayer, QuantumWorkload
from scpn_quantum_control.hardware.hal_qiskit import (
    QiskitAerHALAdapter,
    QiskitRuntimeHALAdapter,
    qiskit_circuit_to_qasm3_workload,
    qiskit_circuit_to_workload,
)
from scpn_quantum_control.hardware.provider_semantics import (
    GateModelObservation,
    WorkloadSemantics,
)


def _bell_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc


def test_qiskit_qpy_workload_round_trips_through_local_aer_hal() -> None:
    """A real Qiskit circuit should execute through HAL and Aer."""
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    hal.register_backend(QiskitAerHALAdapter(hal.profile("local_qiskit_aer")))
    workload = qiskit_circuit_to_workload(
        _bell_circuit(),
        workload_id="bell",
        shots=128,
        metadata={"purpose": "hal_aer_round_trip"},
    )

    job = hal.submit("local_qiskit_aer", workload)
    result = hal.result(job)

    assert job.status == "completed"
    assert result.status == "completed"
    assert result.shots == 128
    assert sum(result.counts.values()) == 128
    assert set(result.counts).issubset({"00", "11"})
    assert result.metadata["execution_mode"] == "qiskit_aer"
    assert result.metadata["ir_format"] == "qiskit_qpy"


def test_local_aer_late_cancel_preserves_completed_evidence_and_identity() -> None:
    """Keep a real local Aer result terminal and refuse misassociated handles."""
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    hal.register_backend(QiskitAerHALAdapter(hal.profile("local_qiskit_aer")))
    workload = qiskit_circuit_to_workload(_bell_circuit(), workload_id="aer_late_cancel", shots=32)
    job = hal.submit("local_qiskit_aer", workload)
    recovered = replace(job, status="submitted")
    first = hal.result(recovered)
    assert hal.result(recovered) is first
    assert hal.cancel(recovered) is job
    assert hal.status(recovered) == "completed"
    assert hal.result(recovered) is first
    assert first.shots == sum(first.counts.values()) == 32
    foreign = replace(recovered, workload_id="another-workload")
    for operation in (hal.status, hal.result, hal.cancel):
        with pytest.raises(ValueError, match="workload_id"):
            operation(foreign)
    assert hal.status(job) == "completed"


def test_qiskit_runtime_adapter_uses_injected_sampler_and_approval_gate() -> None:
    """IBM Runtime adapter should be injectable and still approval-gated."""

    class FakeBackend:
        name = "ibm_fake"
        num_qubits = 127

    class FakeRegister:
        def get_counts(self) -> dict[str, int]:
            return {"0": 3, "1": 5}

    class FakeData:
        c = FakeRegister()

    class FakePubResult:
        data = FakeData()

    class FakeRuntimeResult:
        def __iter__(self) -> Iterator[FakePubResult]:
            return iter((FakePubResult(),))

    class FakeRuntimeJob:
        def job_id(self) -> str:
            return "runtime-job-1"

        def status(self) -> str:
            return "DONE"

        def result(self, timeout: float | None = None) -> FakeRuntimeResult:
            assert timeout == 600.0
            return FakeRuntimeResult()

        def cancel(self) -> None:
            self.cancelled = True

    class FakeSampler:
        def __init__(self, mode: object) -> None:
            assert mode is FakeBackend
            self.options = type("Options", (), {})()

        def run(self, circuits: Sequence[QuantumCircuit]) -> FakeRuntimeJob:
            assert len(circuits) == 1
            return FakeRuntimeJob()

    hal = HardwareAbstractionLayer.with_builtin_profiles()
    hal.register_backend(
        QiskitRuntimeHALAdapter(
            hal.profile("ibm_quantum"),
            backend=FakeBackend,
            sampler_factory=FakeSampler,
        )
    )
    workload = qiskit_circuit_to_workload(_bell_circuit(), workload_id="runtime", shots=8)

    job = hal.submit("ibm_quantum", workload, approval_id="approved-runtime")
    result = hal.result(job)

    assert job.job_id == "runtime-job-1"
    assert job.status == "submitted"
    assert hal.status(job) == "completed"
    assert result.status == "completed"
    assert result.counts == {"0": 3, "1": 5}
    assert result.metadata["execution_mode"] == "qiskit_runtime_sampler"
    assert result.metadata["approval_id"] == "approved-runtime"


@pytest.mark.parametrize(
    "scenario,expected,cancel_calls",
    [
        ("already_done", "completed", 0),
        ("race_done", "completed", 1),
        ("cached_during_cancel", "completed", 1),
        ("accepted", "cancelled", 1),
        ("pending", "running", 1),
    ],
)
def test_runtime_cancel_reports_observed_provider_outcome(
    scenario: str, expected: str, cancel_calls: int
) -> None:
    """Do not label an IBM job cancelled when its provider finished or remains running."""

    class ProviderJob:
        state = "DONE" if scenario == "already_done" else "RUNNING"
        cancellations = 0

        def job_id(self) -> str:
            return "runtime-cancel-race"

        def status(self) -> str:
            return self.state

        def cancel(self) -> None:
            self.cancellations += 1
            if scenario == "race_done":
                self.state = "DONE"
            elif scenario == "cached_during_cancel":
                self.state = "DONE"
                assert hal.result(job).counts == {"0": 4}
                self.state = "CANCELLED"
            elif scenario == "accepted":
                self.state = "CANCELLED"

        def result(self, timeout: float | None = None) -> object:
            assert timeout == 600.0
            assert self.state == "DONE"
            register = type("Register", (), {"get_counts": lambda self: {"0": 4}})()
            data = type("Data", (), {"c": register})()
            return [type("Pub", (), {"data": data})()]

    provider_job = ProviderJob()

    class Sampler:
        def __init__(self, mode: object) -> None:
            self.options = type("Options", (), {})()

        def run(self, circuits: Sequence[QuantumCircuit]) -> ProviderJob:
            assert len(circuits) == 1
            return provider_job

    hal = HardwareAbstractionLayer.with_builtin_profiles()
    hal.register_backend(
        QiskitRuntimeHALAdapter(
            hal.profile("ibm_quantum"), backend=object(), sampler_factory=Sampler
        )
    )
    workload = qiskit_circuit_to_workload(_bell_circuit(), workload_id=f"race_{scenario}", shots=4)
    job = hal.submit("ibm_quantum", workload, approval_id="offline-fault-injection")
    outcome = hal.cancel(replace(job, status="running"))
    assert outcome.status == expected
    assert outcome.metadata == job.metadata
    assert provider_job.cancellations == cancel_calls
    assert hal.status(job) == expected
    if expected == "completed":
        first = hal.result(job)
        assert first.counts == {"0": 4}
        assert first.shots == 4
        assert hal.result(job) is first
        assert hal.cancel(job).status == "completed"
        assert provider_job.cancellations == cancel_calls


def test_qiskit_runtime_adapter_rejects_shot_mismatch() -> None:
    """Runtime result decoding must fail closed on count/shot mismatch."""

    class FakeBackend:
        name = "ibm_fake"
        num_qubits = 127

    class FakeRegister:
        def get_counts(self) -> dict[str, int]:
            return {"0": 3, "1": 4}

    class FakeData:
        c = FakeRegister()

    class FakePubResult:
        data = FakeData()

    class FakeRuntimeResult:
        def __iter__(self) -> Iterator[FakePubResult]:
            return iter((FakePubResult(),))

    class FakeRuntimeJob:
        def job_id(self) -> str:
            return "runtime-job-shot-mismatch"

        def status(self) -> str:
            return "DONE"

        def result(self, timeout: float | None = None) -> FakeRuntimeResult:
            assert timeout == 600.0
            return FakeRuntimeResult()

        def cancel(self) -> None:
            self.cancelled = True

    class FakeSampler:
        def __init__(self, mode: object) -> None:
            assert mode is FakeBackend
            self.options = type("Options", (), {})()

        def run(self, circuits: Sequence[QuantumCircuit]) -> FakeRuntimeJob:
            assert len(circuits) == 1
            return FakeRuntimeJob()

    hal = HardwareAbstractionLayer.with_builtin_profiles()
    hal.register_backend(
        QiskitRuntimeHALAdapter(
            hal.profile("ibm_quantum"),
            backend=FakeBackend,
            sampler_factory=FakeSampler,
        )
    )
    workload = qiskit_circuit_to_workload(
        _bell_circuit(), workload_id="runtime_bad_shots", shots=8
    )
    job = hal.submit("ibm_quantum", workload, approval_id="approved-runtime")

    with pytest.raises(ValueError, match="shot count mismatch"):
        hal.result(job)


def test_qiskit_runtime_adapter_sums_overlapping_pub_results() -> None:
    """Counts from multiple PUB results should be accumulated, not overwritten."""

    class FakeBackend:
        name = "ibm_fake"
        num_qubits = 127

    class FakeRegisterFirst:
        def get_counts(self) -> dict[str, int]:
            return {"0": 2, "1": 1}

    class FakeRegisterSecond:
        def get_counts(self) -> dict[str, int]:
            return {"0": 5, "11": 3}

    class FakeDataFirst:
        c = FakeRegisterFirst()

    class FakeDataSecond:
        c = FakeRegisterSecond()

    class FakePubResultFirst:
        data = FakeDataFirst()

    class FakePubResultSecond:
        data = FakeDataSecond()

    class FakeRuntimeResult:
        def __iter__(self) -> Iterator[object]:
            return iter((FakePubResultFirst(), FakePubResultSecond()))

    class FakeRuntimeJob:
        def job_id(self) -> str:
            return "runtime-job-merge"

        def status(self) -> str:
            return "DONE"

        def result(self, timeout: float | None = None) -> FakeRuntimeResult:
            assert timeout == 600.0
            return FakeRuntimeResult()

        def cancel(self) -> None:
            self.cancelled = True

    class FakeSampler:
        def __init__(self, mode: object) -> None:
            assert mode is FakeBackend
            self.options = type("Options", (), {})()

        def run(self, circuits: Sequence[QuantumCircuit]) -> FakeRuntimeJob:
            assert len(circuits) == 1
            return FakeRuntimeJob()

    hal = HardwareAbstractionLayer.with_builtin_profiles()
    hal.register_backend(
        QiskitRuntimeHALAdapter(
            hal.profile("ibm_quantum"),
            backend=FakeBackend,
            sampler_factory=FakeSampler,
        )
    )
    workload = qiskit_circuit_to_workload(_bell_circuit(), workload_id="runtime_merge", shots=11)
    job = hal.submit("ibm_quantum", workload, approval_id="approved-runtime")
    result = hal.result(job)

    assert result.counts == {"0": 7, "1": 1, "11": 3}
    assert result.shots == 11


def test_qiskit_qasm3_workload_round_trips_when_importer_is_installed() -> None:
    """OpenQASM 3 payloads should execute when qiskit-qasm3-import is present."""
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    hal.register_backend(QiskitAerHALAdapter(hal.profile("local_qiskit_aer")))
    workload = qiskit_circuit_to_qasm3_workload(
        _bell_circuit(),
        workload_id="bell_qasm3",
        shots=64,
        metadata={"purpose": "hal_qasm3_round_trip"},
    )

    job = hal.submit("local_qiskit_aer", workload)
    result = hal.result(job)

    assert result.status == "completed"
    assert result.shots == 64
    assert sum(result.counts.values()) == 64
    assert set(result.counts).issubset({"00", "11"})
    assert result.metadata["ir_format"] == "openqasm3"


def test_qiskit_adapter_rejects_non_qiskit_workload_payload() -> None:
    """Concrete Qiskit adapters should not pretend to execute MLIR strings."""
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    hal.register_backend(QiskitAerHALAdapter(hal.profile("local_qiskit_aer")))

    from scpn_quantum_control.hardware.hal import QuantumWorkload

    workload = QuantumWorkload(
        workload_id="bad_payload",
        ir_format="mlir",
        program="module {}",
        n_qubits=1,
        shots=1,
    )

    with pytest.raises(ValueError, match="qiskit_qpy|OpenQASM"):
        hal.submit("local_qiskit_aer", workload)


def test_qiskit_runtime_status_normalisation_maps_provider_tokens() -> None:
    """Qiskit Runtime status values should map to canonical HAL status values."""
    from scpn_quantum_control.hardware import hal_qiskit as qiskit_mod

    assert qiskit_mod._normalise_status("DONE") == "completed"
    assert qiskit_mod._normalise_status("CANCELED") == "cancelled"
    assert qiskit_mod._normalise_status("IN-PROGRESS") == "running"
    assert qiskit_mod._normalise_status("INPROGRESS") == "running"
    assert qiskit_mod._normalise_status("INITIALIZING") == "submitted"
    assert qiskit_mod._normalise_status("STARTING") == "submitted"
    assert qiskit_mod._normalise_status("CREATING") == "submitted"
    assert qiskit_mod._normalise_status("ABORTING") == "cancelled"
    assert qiskit_mod._normalise_status("CANCELLING") == "cancelled"


def test_qiskit_provider_job_id_extraction_requires_identifier() -> None:
    """Qiskit provider job id extraction should fail closed when id is unavailable."""
    from scpn_quantum_control.hardware import hal_qiskit as qiskit_mod

    class MissingJobId:
        def job_id(self) -> str:
            return "   "

    with pytest.raises(ValueError, match="provider job id"):
        qiskit_mod._provider_job_id(MissingJobId(), provider_name="qiskit_runtime")


def test_qiskit_provider_job_id_rejects_control_characters() -> None:
    """Qiskit provider job ids must reject control-character payloads."""
    from scpn_quantum_control.hardware import hal_qiskit as qiskit_mod

    class BadJobId:
        def job_id(self) -> str:
            return "runtime-job-\n1"

    with pytest.raises(ValueError, match="provider job id"):
        qiskit_mod._provider_job_id(BadJobId(), provider_name="qiskit_runtime")


def test_qiskit_provider_job_id_trims_padding() -> None:
    """Qiskit provider job ids should be trimmed to canonical form."""
    from scpn_quantum_control.hardware import hal_qiskit as qiskit_mod

    class PaddedJobId:
        def job_id(self) -> str:
            return "  runtime-job-42  "

    assert (
        qiskit_mod._provider_job_id(PaddedJobId(), provider_name="qiskit_runtime")
        == "runtime-job-42"
    )


def test_qiskit_backend_name_rejects_control_characters() -> None:
    """Qiskit backend names must reject control-character payloads."""
    from scpn_quantum_control.hardware import hal_qiskit as qiskit_mod

    class BadBackend:
        name = "ibm_\nbackend"

    with pytest.raises(ValueError, match="backend name"):
        qiskit_mod._backend_name(BadBackend())


def test_qiskit_backend_name_trims_padding() -> None:
    """Qiskit backend names should be canonicalised by trimming padding."""
    from scpn_quantum_control.hardware import hal_qiskit as qiskit_mod

    class PaddedBackend:
        name = "  ibm_backend  "

    assert qiskit_mod._backend_name(PaddedBackend()) == "ibm_backend"


def test_aer_falsey_injected_backend_keeps_native_target_and_transport() -> None:
    """A falsey native-compatible backend cannot cause implicit Aer replacement."""

    class FalseyBackend:
        name = "owner-pinned-native-aer"

        def __init__(self) -> None:
            self.native = AerSimulator()
            self.calls = 0

        def __bool__(self) -> bool:
            return False

        def __getattr__(self, name: str) -> object:
            return getattr(self.native, name)

        def run(self, circuit: QuantumCircuit, shots: int) -> object:
            self.calls += 1
            return self.native.run(circuit, shots=shots)

    backend = FalseyBackend()
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    hal.register_backend(QiskitAerHALAdapter(hal.profile("local_qiskit_aer"), backend=backend))
    circuit = QuantumCircuit(1, 1)
    circuit.x(0)
    circuit.measure(0, 0)
    workload = qiskit_circuit_to_workload(
        circuit,
        workload_id="falsey_aer",
        shots=8,
        capture_semantics=True,
        requested_target=backend.name,
    )
    job = hal.submit("local_qiskit_aer", workload)
    assert hal.result(job).counts == {"1": 8} and backend.calls == 1
    assert job.submission is not None and job.submission.target_name == backend.name


def test_runtime_falsey_native_sampler_factory_is_not_replaced() -> None:
    """Explicit native sampler factories remain selected independently of truthiness."""
    backend = GenericBackendV2(1, basis_gates=["x"], noise_info=False)

    class FalseyFactory:
        def __init__(self) -> None:
            self.calls = 0

        def __bool__(self) -> bool:
            return False

        def __call__(self, *, mode: object) -> SamplerV2:
            assert mode is backend
            self.calls += 1
            return SamplerV2(mode=backend)

    factory = FalseyFactory()
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    hal.register_backend(
        QiskitRuntimeHALAdapter(
            hal.profile("ibm_quantum"),
            backend=backend,
            sampler_factory=factory,
        )
    )
    circuit = QuantumCircuit(1, 1)
    circuit.x(0)
    circuit.measure(0, 0)
    job = hal.submit(
        "ibm_quantum",
        qiskit_circuit_to_workload(
            circuit,
            workload_id="falsey_sampler_factory",
            shots=8,
            capture_semantics=True,
        ),
        approval_id="software-only",
    )
    assert hal.result(job).counts == {"1": 8} and factory.calls == 1


@pytest.mark.parametrize("backend_id", ["local_qiskit_aer", "ibm_quantum"])
def test_bound_qasm3_native_companion_preserves_partial_permutation(backend_id: str) -> None:
    """Actual OpenQASM parser and native simulators retain the bound static readout map."""
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    if backend_id == "local_qiskit_aer":
        backend = AerSimulator()
        hal.register_backend(QiskitAerHALAdapter(hal.profile(backend_id), backend=backend))
    else:
        backend = GenericBackendV2(3, basis_gates=["x"], noise_info=False)
        hal.register_backend(QiskitRuntimeHALAdapter(hal.profile(backend_id), backend=backend))
    circuit = QuantumCircuit(3, 2)
    circuit.x(2)
    circuit.measure([2, 0], [0, 1])
    workload = qiskit_circuit_to_qasm3_workload(
        circuit,
        workload_id="bound_qasm3_native",
        shots=8,
        capture_semantics=True,
        requested_target=backend.name,
    )
    assert isinstance(workload.semantics, WorkloadSemantics)
    assert workload.semantics.measurement_map == ((2, 0), (0, 1))
    assert workload.semantics.parameters == ()
    job = hal.submit(backend_id, workload, approval_id="software-only")
    result = hal.result(job)
    assert result.counts == {"01": 8} and isinstance(
        result.provider_observation, GateModelObservation
    )
    assert job.submission is not None and job.submission.original_program == workload.program
    assert job.submission.compilation == "targeted" and job.submission.compiled_program_sha256


def test_qasm3_capture_refuses_loss_of_original_shared_parameter_uuid() -> None:
    """OpenQASM cannot impersonate QPY's original native shared-parameter identity."""
    shared = Parameter("shared")
    circuit = QuantumCircuit(2, 2)
    circuit.ry(shared, 0)
    circuit.ry(shared, 1)
    circuit.measure([0, 1], [0, 1])
    with pytest.raises(ValueError, match="parameter UUID.*QPY"):
        qiskit_circuit_to_qasm3_workload(
            circuit,
            workload_id="qasm_shared_uuid",
            shots=8,
            capture_semantics=True,
        )
    native = qiskit_circuit_to_workload(
        circuit,
        workload_id="qpy_shared_uuid",
        shots=8,
        capture_semantics=True,
        parameter_bindings={shared: 0.0},
    )
    assert isinstance(native.semantics, WorkloadSemantics)
    assert native.semantics.parameters == (("shared", str(shared.uuid), ((0, 0), (1, 0))),)
    assert circuit.parameters == {shared}


@pytest.mark.parametrize("case", ["empty", "multiple", "null_tail", "opaque"])
def test_runtime_native_single_pub_admission_precedes_count_decoding(case: str) -> None:
    """Captured single-circuit requests refuse ambiguous provider results before decoding."""
    pub = SamplerPubResult(DataBin(c=BitArray.from_samples(["00"] * 4, num_bits=2)))

    class FaultJob:
        def __init__(self) -> None:
            self.pubs: list[object] = {
                "empty": [],
                "multiple": [pub, pub],
                "null_tail": [pub, None],
                "opaque": [object()],
            }[case]
            self.result_calls = 0

        def job_id(self) -> str:
            return "native-runtime-pub-admission"

        def result(self, timeout: float | None = None) -> list[object]:
            assert timeout == 600.0
            self.result_calls += 1
            return self.pubs

    provider_job = FaultJob()
    backend = GenericBackendV2(2, basis_gates=["h", "cx"], noise_info=False)

    class FaultSampler:
        def __init__(self, *, mode: object) -> None:
            assert mode is backend
            self.options = SimpleNamespace(default_shots=0)

        def run(self, circuits: Sequence[QuantumCircuit]) -> FaultJob:
            assert len(circuits) == 1 and self.options.default_shots == 4
            return provider_job

    hal = HardwareAbstractionLayer.with_builtin_profiles()
    hal.register_backend(
        QiskitRuntimeHALAdapter(
            hal.profile("ibm_quantum"),
            backend=backend,
            sampler_factory=FaultSampler,
        )
    )
    workload = qiskit_circuit_to_workload(
        _bell_circuit(),
        workload_id="native_pub_admission",
        shots=4,
        capture_semantics=True,
    )
    job = hal.submit("ibm_quantum", workload, approval_id="fault-contract-only")
    with pytest.raises(ValueError, match="exactly one PUB|actual SamplerPubResult"):
        hal.result(job)
    provider_job.pubs = [pub]
    result = hal.result(job)
    assert result.counts == {"00": 4} and provider_job.result_calls == 2
    assert hal.result(job) is result and provider_job.result_calls == 2


@pytest.mark.parametrize("builder_name", ["qpy", "qasm3"])
def test_qiskit_native_builder_refuses_foreign_source_and_undeclared_target(
    builder_name: str,
) -> None:
    """Both native builders reject foreign objects and target pins without capture."""
    builder = (
        qiskit_circuit_to_workload if builder_name == "qpy" else qiskit_circuit_to_qasm3_workload
    )
    with pytest.raises(TypeError, match="qiskit.QuantumCircuit"):
        builder(cast(QuantumCircuit, object()), workload_id="foreign_source", shots=4)
    with pytest.raises(ValueError, match="requires native semantics capture"):
        builder(_bell_circuit(), workload_id="undeclared_target", shots=4, requested_target="pin")
    if builder_name == "qpy":
        with pytest.raises(ValueError, match="parameter_bindings requires"):
            qiskit_circuit_to_workload(
                _bell_circuit(),
                workload_id="undeclared_bindings",
                shots=4,
                parameter_bindings={},
            )


@pytest.mark.parametrize(
    "case", ["corrupt_qpy", "zero", "two", "foreign_sdk_object", "qasm3", "ir"]
)
def test_qiskit_source_decoder_faults_precede_backend_construction(
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    """Native codec shape and dependency faults cannot trigger backend construction or run."""
    import qiskit_aer

    factory_calls = 0

    def forbidden_factory() -> object:
        nonlocal factory_calls
        factory_calls += 1
        raise AssertionError("source refusal must precede backend construction")

    monkeypatch.setattr(qiskit_aer, "AerSimulator", forbidden_factory)
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    profile = hal.profile("local_qiskit_aer")
    adapter = QiskitAerHALAdapter(profile)
    program = "invalid-base64!"
    ir = "qiskit_qpy"
    if case in {"zero", "two", "foreign_sdk_object"}:
        buffer = io.BytesIO()
        qpy.dump([] if case == "zero" else [_bell_circuit()] * (2 if case == "two" else 1), buffer)
        program = base64.b64encode(buffer.getvalue()).decode("ascii")
        if case == "foreign_sdk_object":

            def broken_codec(buffer: BinaryIO) -> list[object]:
                assert buffer.read(6) == b"QISKIT"
                return [object()]

            monkeypatch.setattr(qpy, "load", broken_codec)
    elif case == "qasm3":
        ir = "openqasm3"
        program = "not valid qasm"
    elif case == "ir":
        ir = "mlir"
        profile = replace(profile, ir_formats=(*profile.ir_formats, "mlir"))
        adapter = QiskitAerHALAdapter(profile)
    workload = QuantumWorkload(
        workload_id="bad_native_source", ir_format=ir, program=program, n_qubits=2, shots=4
    )
    with pytest.raises((ValueError, TypeError)):
        adapter.submit(workload)
    assert factory_calls == 0


def test_qiskit_profile_and_runtime_approval_refusals_precede_sampler_factory() -> None:
    """Wrong profiles and absent cloud approval never construct a transport."""
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    with pytest.raises(ValueError, match="local_qiskit_aer profile"):
        QiskitAerHALAdapter(hal.profile("local_braket_sv"))
    with pytest.raises(ValueError, match="ibm_quantum profile"):
        QiskitRuntimeHALAdapter(hal.profile("local_braket_sv"), backend=object())
    calls = 0

    def factory(*, mode: object) -> object:
        nonlocal calls
        calls += 1
        raise AssertionError("approval refusal must precede sampler construction")

    adapter = QiskitRuntimeHALAdapter(
        hal.profile("ibm_quantum"), backend=object(), sampler_factory=factory
    )
    with pytest.raises(PermissionError, match="approval_id"):
        adapter.submit(qiskit_circuit_to_workload(_bell_circuit(), workload_id="no_go", shots=4))
    assert calls == 0


def test_runtime_actual_native_legacy_pub_keeps_existing_count_contract() -> None:
    """Actual local Runtime output still joins native registers without an opt-in companion."""
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    backend = GenericBackendV2(1, basis_gates=["x"], noise_info=False)
    hal.register_backend(QiskitRuntimeHALAdapter(hal.profile("ibm_quantum"), backend=backend))
    circuit = QuantumCircuit(1, 1)
    circuit.x(0)
    circuit.measure(0, 0)
    job = hal.submit(
        "ibm_quantum",
        qiskit_circuit_to_workload(
            circuit,
            workload_id="native_legacy_pub",
            shots=8,
        ),
        approval_id="software-only",
    )
    result = hal.result(job)
    assert result.counts == {"1": 8} and result.provider_observation is None
    assert job.submission is None and hal.result(job) is result


def test_aer_native_empty_count_fault_keeps_prior_completed_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A native job's damaged count channel refuses before changing prior retained evidence."""

    class FaultBackend:
        def __init__(self) -> None:
            self.native = AerSimulator()
            self.empty = False

        def __getattr__(self, name: str) -> object:
            return getattr(self.native, name)

        def run(self, circuit: QuantumCircuit, shots: int) -> object:
            job = self.native.run(circuit, shots=shots)
            result = job.result()
            if self.empty:
                monkeypatch.setattr(result, "get_counts", lambda: {})
                monkeypatch.setattr(job, "result", lambda: result)
            return job

    backend = FaultBackend()
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    hal.register_backend(QiskitAerHALAdapter(hal.profile("local_qiskit_aer"), backend=backend))
    workload = qiskit_circuit_to_workload(
        _bell_circuit(), workload_id="prior_native_counts", shots=8
    )
    prior = hal.submit("local_qiskit_aer", workload)
    saved = hal.result(prior)
    backend.empty = True
    with pytest.raises(ValueError, match="did not contain any counts"):
        hal.submit("local_qiskit_aer", replace(workload, workload_id="bad_native_counts"))
    assert hal.result(prior) is saved and saved.shots == 8


def test_runtime_missing_retained_transport_restores_without_evidence_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A controlled missing-transport fault refuses through HAL and keeps unrelated native output."""
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    backend = GenericBackendV2(1, basis_gates=["x"], noise_info=False)
    adapter = QiskitRuntimeHALAdapter(hal.profile("ibm_quantum"), backend=backend)
    hal.register_backend(adapter)
    source = QuantumCircuit(1, 1)
    source.x(0)
    source.measure(0, 0)
    prior = hal.submit(
        "ibm_quantum",
        qiskit_circuit_to_workload(
            source,
            workload_id="prior_native_runtime",
            shots=8,
            capture_semantics=True,
        ),
        approval_id="software-only",
    )
    saved = hal.result(prior)
    other_source = QuantumCircuit(1, 1)
    other_source.measure(0, 0)
    other = hal.submit(
        "ibm_quantum",
        qiskit_circuit_to_workload(
            other_source,
            workload_id="other_native_runtime",
            shots=8,
            capture_semantics=True,
        ),
        approval_id="software-only",
    )
    with monkeypatch.context() as injected:
        injected.delitem(adapter._provider_jobs, other.job_id)
        for operation in (hal.status, hal.result, hal.cancel):
            with pytest.raises(KeyError, match="unknown job_id"):
                operation(other)
        assert hal.result(prior) is saved and saved.counts == {"1": 8}
    restored = hal.result(other)
    assert restored.counts == {"0": 8} and hal.result(other) is restored
    assert hal.result(prior) is saved


def test_runtime_callable_native_backend_name_preserves_selected_target() -> None:
    """Callable native-compatible names retain the exact underlying compiler and sampler target."""
    native = GenericBackendV2(1, basis_gates=["x"], noise_info=False)

    class NativeBackendView:
        def name(self) -> str:
            name = native.name
            assert isinstance(name, str)
            return name

        def __getattr__(self, name: str) -> object:
            return getattr(native, name)

    view = NativeBackendView()
    calls: list[object] = []

    def factory(*, mode: object) -> SamplerV2:
        assert mode is view
        calls.append(mode)
        return SamplerV2(mode=native)

    hal = HardwareAbstractionLayer.with_builtin_profiles()
    hal.register_backend(
        QiskitRuntimeHALAdapter(
            hal.profile("ibm_quantum"),
            backend=view,
            sampler_factory=factory,
        )
    )
    circuit = QuantumCircuit(1, 1)
    circuit.x(0)
    circuit.measure(0, 0)
    job = hal.submit(
        "ibm_quantum",
        qiskit_circuit_to_workload(
            circuit,
            workload_id="callable_native_name",
            shots=8,
            capture_semantics=True,
            requested_target=native.name,
        ),
        approval_id="software-only",
    )
    assert job.submission is not None and job.submission.target_name == native.name
    assert hal.result(job).counts == {"1": 8} and calls == [view]


@pytest.mark.parametrize("route", ["aer", "runtime"])
def test_qiskit_native_loader_faults_keep_original_cause_and_refuse_execution(
    monkeypatch: pytest.MonkeyPatch,
    route: str,
) -> None:
    """Native simulator construction and Runtime import failures cannot choose a substitute."""
    import qiskit_aer
    import qiskit_ibm_runtime

    original = ImportError("controlled native SDK loader fault")
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    workload = qiskit_circuit_to_workload(_bell_circuit(), workload_id="sdk_loader_fault", shots=8)
    if route == "aer":

        def failing_aer() -> object:
            raise original

        monkeypatch.setattr(qiskit_aer, "AerSimulator", failing_aer)
        hal.register_backend(QiskitAerHALAdapter(hal.profile("local_qiskit_aer")))
        backend_id = "local_qiskit_aer"
    else:

        def failing_import(name: str) -> object:
            if name == "SamplerV2":
                raise original
            raise AttributeError(name)

        monkeypatch.delattr(qiskit_ibm_runtime, "SamplerV2")
        monkeypatch.setattr(qiskit_ibm_runtime, "__getattr__", failing_import, raising=False)
        hal.register_backend(QiskitRuntimeHALAdapter(hal.profile("ibm_quantum"), backend=object()))
        backend_id = "ibm_quantum"
    with pytest.raises(RuntimeError, match="required for") as error:
        hal.submit(backend_id, workload, approval_id="fault-contract-only")
    assert error.value.__cause__ is original
