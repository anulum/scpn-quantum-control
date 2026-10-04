# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — hardware HAL braket adapters tests
# scpn-quantum-control -- Braket HAL adapter tests
"""Tests for Braket adapters behind the provider-neutral HAL."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import cast

import pytest
from braket.circuits import Circuit, FreeParameter
from braket.devices import LocalSimulator

from scpn_quantum_control.hardware.hal import HardwareAbstractionLayer, QuantumWorkload
from scpn_quantum_control.hardware.hal_braket import (
    BraketAwsHALAdapter,
    BraketLocalHALAdapter,
    braket_circuit_to_workload,
)
from scpn_quantum_control.hardware.provider_capability_core import ProviderCapabilitySnapshot
from scpn_quantum_control.hardware.provider_modalities import ModalitySemantics
from scpn_quantum_control.hardware.provider_semantics import (
    GateModelObservation,
    WorkloadSemantics,
)


def _bell_circuit() -> Circuit:
    return Circuit().h(0).cnot(0, 1)


@pytest.mark.parametrize("value", [10**400, float("nan"), float("inf"), True, "0.5"])
def test_native_braket_binding_refuses_nonfinite_or_coerced_values(value: object) -> None:
    """Actual native source binding refuses malformed numeric values before execution."""
    circuit = Circuit().ry(0, FreeParameter("theta")).measure(0)
    with pytest.raises(ValueError, match="finite|binding"):
        braket_circuit_to_workload(
            circuit,
            workload_id="invalid_binding",
            shots=4,
            capture_semantics=True,
            parameter_bindings={"theta": cast(float, value)},
        )


def test_braket_local_simulator_round_trips_through_hal() -> None:
    """A real Braket circuit should execute through a local Braket simulator."""
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    hal.register_backend(BraketLocalHALAdapter(hal.profile("local_braket_sv")))
    workload = braket_circuit_to_workload(
        _bell_circuit(),
        workload_id="braket_bell",
        shots=128,
        metadata={"purpose": "hal_braket_round_trip"},
    )

    job = hal.submit("local_braket_sv", workload)
    result = hal.result(job)

    assert job.status == "completed"
    assert result.status == "completed"
    assert result.shots == 128
    assert sum(result.counts.values()) == 128
    assert set(result.counts).issubset({"00", "11"})
    assert result.metadata["execution_mode"] == "braket_local"
    assert result.metadata["ir_format"] == "openqasm3"


@pytest.mark.parametrize("backend_id", ["local_braket_sv", "local_braket_dm"])
def test_braket_local_late_cancel_preserves_completed_evidence_and_identity(
    backend_id: str,
) -> None:
    """Keep real local Braket results terminal and refuse foreign handles."""
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    hal.register_backend(BraketLocalHALAdapter(hal.profile(backend_id)))
    workload = braket_circuit_to_workload(
        _bell_circuit(), workload_id="braket_late_cancel", shots=32
    )
    job = hal.submit(backend_id, workload)
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


def test_braket_aws_adapter_uses_injected_device_and_approval_gate() -> None:
    """AWS Braket adapter should be injectable and approval-gated."""

    class FakeTaskResult:
        measurement_counts = {"0": 4, "1": 2}

    class FakeTask:
        id = "arn:aws:braket:task/fake-task"

        def state(self) -> str:
            return "COMPLETED"

        def result(self) -> FakeTaskResult:
            return FakeTaskResult()

        def cancel(self) -> None:
            self.cancelled = True

    class FakeDevice:
        name = "fake-braket-device"

        def run(self, circuit: Circuit, shots: int) -> FakeTask:
            assert shots == 6
            assert isinstance(circuit, Circuit)
            return FakeTask()

    hal = HardwareAbstractionLayer.with_builtin_profiles()
    hal.register_backend(
        BraketAwsHALAdapter(
            hal.profile("aws_braket_ionq"),
            device=FakeDevice(),
        )
    )
    workload = braket_circuit_to_workload(_bell_circuit(), workload_id="aws_bell", shots=6)

    job = hal.submit("aws_braket_ionq", workload, approval_id="approved-braket")
    result = hal.result(job)

    assert job.status == "submitted"
    assert result.status == "completed"
    assert result.counts == {"00": 4, "01": 2}
    assert result.metadata["execution_mode"] == "braket_aws"
    assert result.metadata["approval_id"] == "approved-braket"
    assert hal.status(job) == "completed"


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
def test_braket_aws_cancel_reports_observed_provider_outcome(
    scenario: str, expected: str, cancel_calls: int
) -> None:
    """Do not label an AWS task cancelled unless its provider confirms it."""

    class ProviderTask:
        id = "arn:aws:braket:task/offline-cancel-race"
        current = "COMPLETED" if scenario == "already_done" else "RUNNING"
        cancellations = 0

        def state(self) -> str:
            return self.current

        def cancel(self) -> None:
            self.cancellations += 1
            if scenario == "race_done":
                self.current = "COMPLETED"
            elif scenario == "cached_during_cancel":
                self.current = "COMPLETED"
                assert hal.result(job).counts == {"00": 4}
                self.current = "CANCELLED"
            elif scenario == "accepted":
                self.current = "CANCELLED"

        def result(self) -> object:
            assert self.current == "COMPLETED"
            return type("Result", (), {"measurement_counts": {"00": 4}})()

    task = ProviderTask()

    class Device:
        name = "offline-device"

        def run(self, circuit: Circuit, shots: int) -> ProviderTask:
            assert isinstance(circuit, Circuit)
            assert shots == 4
            return task

    hal = HardwareAbstractionLayer.with_builtin_profiles()
    hal.register_backend(BraketAwsHALAdapter(hal.profile("aws_braket_ionq"), device=Device()))
    workload = braket_circuit_to_workload(_bell_circuit(), workload_id=f"race_{scenario}", shots=4)
    job = hal.submit("aws_braket_ionq", workload, approval_id="offline-fault-injection")
    outcome = hal.cancel(replace(job, status="running"))
    assert outcome.status == expected
    assert outcome.metadata == job.metadata
    assert task.cancellations == cancel_calls
    assert hal.status(job) == expected
    if expected == "completed":
        first = hal.result(job)
        assert first.counts == {"00": 4}
        assert first.shots == 4
        assert hal.result(job) is first
        assert hal.cancel(job).status == "completed"
        assert task.cancellations == cancel_calls


def test_braket_aws_adapter_rejects_task_without_id() -> None:
    """AWS Braket adapter should fail closed when provider task id is missing."""

    class FakeTask:
        def state(self) -> str:
            return "COMPLETED"

        def result(self) -> object:
            return type("FakeTaskResult", (), {"measurement_counts": {"0": 1}})()

        def cancel(self) -> None:
            self.cancelled = True

    class FakeDevice:
        name = "fake-braket-device"

        def run(self, circuit: Circuit, shots: int) -> FakeTask:
            del circuit, shots
            return FakeTask()

    hal = HardwareAbstractionLayer.with_builtin_profiles()
    hal.register_backend(
        BraketAwsHALAdapter(
            hal.profile("aws_braket_ionq"),
            device=FakeDevice(),
        )
    )
    workload = braket_circuit_to_workload(_bell_circuit(), workload_id="aws_missing_id", shots=1)

    with pytest.raises(ValueError, match="task id"):
        hal.submit("aws_braket_ionq", workload, approval_id="approved-braket")


def test_braket_provider_task_id_rejects_control_characters() -> None:
    """Braket provider task identifiers must reject control-character payloads."""
    from scpn_quantum_control.hardware import hal_braket as braket_mod

    class BadTask:
        id = "arn:aws:braket:task/fake-\njob"

    with pytest.raises(ValueError, match="provider task id"):
        braket_mod._task_id(BadTask())


def test_braket_provider_task_id_trims_padding() -> None:
    """Braket provider task identifiers should be canonicalised by trimming padding."""
    from scpn_quantum_control.hardware import hal_braket as braket_mod

    class PaddedTask:
        id = "  arn:aws:braket:task/fake-task  "

    assert braket_mod._task_id(PaddedTask()) == "arn:aws:braket:task/fake-task"


def test_braket_device_name_rejects_control_characters() -> None:
    """Braket device names must reject control-character payloads."""
    from scpn_quantum_control.hardware import hal_braket as braket_mod

    class BadDevice:
        name = "fake-\ndevice"

    with pytest.raises(ValueError, match="device name"):
        braket_mod._device_name(BadDevice())


def test_braket_device_name_trims_padding() -> None:
    """Braket device names should be canonicalised by trimming padding."""
    from scpn_quantum_control.hardware import hal_braket as braket_mod

    class PaddedDevice:
        name = "  fake-braket-device  "

    assert braket_mod._device_name(PaddedDevice()) == "fake-braket-device"


def test_braket_status_normalisation_maps_provider_tokens() -> None:
    """Braket status values should map to canonical HAL status values."""
    from scpn_quantum_control.hardware import hal_braket as braket_mod

    assert braket_mod._normalise_status("FINISHED") == "completed"
    assert braket_mod._normalise_status("CANCELED") == "cancelled"
    assert braket_mod._normalise_status("IN-PROGRESS") == "running"
    assert braket_mod._normalise_status("INPROGRESS") == "running"
    assert braket_mod._normalise_status("INITIALIZING") == "submitted"
    assert braket_mod._normalise_status("STARTING") == "submitted"
    assert braket_mod._normalise_status("CREATING") == "submitted"
    assert braket_mod._normalise_status("ABORTING") == "cancelled"
    assert braket_mod._normalise_status("CANCELLING") == "cancelled"


def test_braket_aws_adapter_rejects_shot_mismatch() -> None:
    """Braket adapter must fail closed when decoded counts diverge from requested shots."""

    class FakeTaskResult:
        measurement_counts = {"0": 4, "1": 2}

    class FakeTask:
        id = "arn:aws:braket:task/fake-mismatch-task"

        def state(self) -> str:
            return "COMPLETED"

        def result(self) -> FakeTaskResult:
            return FakeTaskResult()

        def cancel(self) -> None:
            self.cancelled = True

    class FakeDevice:
        name = "fake-braket-device"

        def run(self, circuit: Circuit, shots: int) -> FakeTask:
            del circuit
            assert shots == 7
            return FakeTask()

    hal = HardwareAbstractionLayer.with_builtin_profiles()
    hal.register_backend(
        BraketAwsHALAdapter(
            hal.profile("aws_braket_ionq"),
            device=FakeDevice(),
        )
    )
    workload = braket_circuit_to_workload(
        _bell_circuit(), workload_id="aws_shot_mismatch", shots=7
    )

    job = hal.submit("aws_braket_ionq", workload, approval_id="approved-braket")
    with pytest.raises(ValueError, match="shot count mismatch"):
        hal.result(job)


def test_braket_aws_device_arn_rejects_control_characters() -> None:
    """Reject control characters before a Braket device ARN is stored."""
    profile = HardwareAbstractionLayer.with_builtin_profiles().profile("aws_braket_ionq")
    with pytest.raises(ValueError, match="Braket device ARN"):
        BraketAwsHALAdapter(
            profile,
            device_arn="arn:aws:braket:us-east-1::device/qpu/ionq/\naria-1",
            device_factory=lambda arn: arn,
        )


def test_braket_aws_device_arn_trims_padding() -> None:
    """Store the canonical ARN after removing surrounding whitespace."""
    profile = HardwareAbstractionLayer.with_builtin_profiles().profile("aws_braket_ionq")
    adapter = BraketAwsHALAdapter(
        profile,
        device_arn="  arn:aws:braket:us-east-1::device/qpu/ionq/aria-1  ",
        device_factory=lambda arn: arn,
    )
    assert adapter._device_arn == "arn:aws:braket:us-east-1::device/qpu/ionq/aria-1"


class _FaultTask:
    """Expose controlled provider failures after real native circuit decoding."""

    def __init__(self, native_result: object) -> None:
        self.id: object = "offline-braket-fault-task"
        self.native_result = native_result
        self.state_value = "COMPLETED"
        self.result_calls = 0

    def state(self) -> str:
        """Return the injected lifecycle without submitting to a provider."""
        return self.state_value

    def result(self) -> object:
        """Expose the original result object without count coercion."""
        self.result_calls += 1
        return self.native_result


class _FaultDevice:
    """Count native run calls while controlling only the provider result channel."""

    def __init__(self, task: _FaultTask) -> None:
        self.name: object = "offline-braket-fault-device"
        self.task = task
        self.run_calls = 0

    def run(self, circuit: Circuit, shots: int) -> _FaultTask:
        """Receive an actual decoded Braket circuit and exact positive shots."""
        assert isinstance(circuit, Circuit) and shots > 0
        self.run_calls += 1
        return self.task


@pytest.mark.parametrize("backend_id", ["local_braket_sv", "aws_braket_ionq"])
def test_falsey_native_device_is_retained_without_replacement(backend_id: str) -> None:
    """Native transport truthiness must not select an implicit replacement device."""

    class FalseySimulator:
        name = "owner-supplied-local-simulator"

        def __init__(self) -> None:
            self.native = LocalSimulator()
            self.calls = 0

        def __bool__(self) -> bool:
            return False

        def run(self, circuit: Circuit, shots: int) -> object:
            self.calls += 1
            return self.native.run(circuit, shots=shots)

    hal = HardwareAbstractionLayer.with_builtin_profiles()
    device = FalseySimulator()
    profile = hal.profile(backend_id)
    adapter = (
        BraketLocalHALAdapter(profile, device=device)
        if backend_id == "local_braket_sv"
        else BraketAwsHALAdapter(profile, device=device)
    )
    hal.register_backend(adapter)
    workload = braket_circuit_to_workload(
        Circuit().x(0).measure(0),
        workload_id="falsey_transport",
        shots=8,
        capture_semantics=True,
        requested_target=device.name,
    )
    job = hal.submit(backend_id, workload, approval_id="software-only")
    assert hal.result(job).counts == {"1": 8}
    assert device.calls == 1
    assert job.submission is not None and job.submission.target_name == device.name


@pytest.mark.parametrize("target,bindings", [("pin", None), (None, {})])
def test_braket_builder_requires_explicit_capture_for_native_companion(
    target: str | None,
    bindings: Mapping[str, float] | None,
) -> None:
    """Legacy payload construction cannot silently admit new target or binding settings."""
    with pytest.raises(ValueError, match="require native semantics capture"):
        braket_circuit_to_workload(
            _bell_circuit(),
            workload_id="undeclared_companion",
            shots=4,
            requested_target=target,
            parameter_bindings=bindings,
        )


def test_braket_builder_rejects_foreign_circuit_and_nonstatic_measurement_order() -> None:
    """Wrong SDK objects and gates after the declared final readout refuse locally."""
    with pytest.raises(TypeError, match="braket.circuits.Circuit"):
        braket_circuit_to_workload(object(), workload_id="foreign", shots=4)
    with pytest.raises(ValueError, match="follows final measurement"):
        braket_circuit_to_workload(
            Circuit().measure(0).x(1), workload_id="late_gate", shots=4, capture_semantics=True
        )


@pytest.mark.parametrize(
    "case", ["unbound", "missing_bindings", "unknown_binding", "wrong_map", "modality"]
)
def test_braket_native_request_refusal_leaves_transport_and_prior_evidence_unchanged(
    case: str,
) -> None:
    """Source wiring, complete shared bindings and gate modality qualify before run."""
    hal = HardwareAbstractionLayer.with_builtin_profiles()

    class CountingSimulator:
        name = "counted-native-simulator"

        def __init__(self) -> None:
            self.native = LocalSimulator()
            self.calls = 0

        def run(self, circuit: Circuit, shots: int) -> object:
            self.calls += 1
            return self.native.run(circuit, shots=shots)

    native = CountingSimulator()
    adapter = BraketLocalHALAdapter(hal.profile("local_braket_sv"), device=native)
    hal.register_backend(adapter)
    good = braket_circuit_to_workload(
        Circuit().x(0).measure(0), workload_id="prior_braket", shots=4, capture_semantics=True
    )
    old = hal.submit("local_braket_sv", good)
    saved = hal.result(old)
    source = (
        Circuit().ry(0, FreeParameter("shared")).ry(1, FreeParameter("shared")).measure([0, 1])
    )
    if case == "unknown_binding":
        with pytest.raises(ValueError, match="original native parameters"):
            braket_circuit_to_workload(
                source,
                workload_id="bad_symbols",
                shots=4,
                capture_semantics=True,
                parameter_bindings={"foreign": 0.0},
            )
    else:
        bad = braket_circuit_to_workload(
            source,
            workload_id="bad_request",
            shots=4,
            capture_semantics=case != "unbound",
        )
        if case == "wrong_map":
            assert isinstance(bad.semantics, WorkloadSemantics)
            bad = replace(bad, semantics=replace(bad.semantics, measurement_map=((1, 0), (0, 1))))
        elif case == "modality":
            assert isinstance(bad.semantics, WorkloadSemantics)
            bad = replace(
                bad,
                semantics=ModalitySemantics(
                    program_sha256=bad.semantics.program_sha256,
                    modality="analog",
                    native_axes=(0, 1),
                ),
            )
        with pytest.raises(ValueError, match="bindings|wiring|gate-model"):
            hal.submit("local_braket_sv", bad)
    assert hal.result(old) is saved and saved.counts == {"1": 4} and native.calls == 1


@pytest.mark.parametrize("backend_id", ["local_braket_sv", "aws_braket_ionq"])
@pytest.mark.parametrize("case", ["missing", "empty", "short", "negative", "fractional", "shots"])
def test_braket_result_count_fault_does_not_replace_prior_result(
    backend_id: str,
    case: str,
) -> None:
    """Provider count faults refuse at the public result boundary and permit recovery."""
    raw = SimpleNamespace(measurement_counts={"00": 4})
    task = _FaultTask(raw)
    device = _FaultDevice(task)
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    profile = hal.profile(backend_id)
    adapter = (
        BraketLocalHALAdapter(profile, device=device)
        if backend_id == "local_braket_sv"
        else BraketAwsHALAdapter(profile, device=device)
    )
    hal.register_backend(adapter)
    workload = braket_circuit_to_workload(_bell_circuit(), workload_id="count_fault", shots=4)
    old = hal.submit(backend_id, workload, approval_id="fault-contract-only")
    saved = hal.result(old)
    task.id = "offline-braket-fault-next"
    bad_counts: object = {
        "missing": None,
        "empty": {},
        "short": {"000": 4},
        "negative": {"00": -4},
        "fractional": {"00": 1.5},
        "shots": {"00": 3},
    }[case]
    task.native_result = SimpleNamespace(measurement_counts=bad_counts)
    if backend_id == "local_braket_sv":
        with pytest.raises(ValueError):
            hal.submit(backend_id, replace(workload, workload_id="bad_counts"))
    else:
        bad_job = hal.submit(
            backend_id,
            replace(workload, workload_id="bad_counts"),
            approval_id="fault-contract-only",
        )
        with pytest.raises(ValueError):
            hal.result(bad_job)
        task.native_result = raw
        assert hal.result(bad_job).counts == {"00": 4}
    assert hal.result(old) is saved and saved.counts == {"00": 4}


@pytest.mark.parametrize("case", ["order", "counts", "strings"])
def test_braket_native_observation_keeps_request_and_raw_provider_counts(case: str) -> None:
    """Captured Braket output requires exact measured-qubit order and native integral counts."""
    raw = SimpleNamespace(measured_qubits=[0, 1], measurement_counts={"00": 4})
    task = _FaultTask(raw)
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    adapter = BraketAwsHALAdapter(hal.profile("aws_braket_ionq"), device=_FaultDevice(task))
    hal.register_backend(adapter)
    workload = braket_circuit_to_workload(
        _bell_circuit(),
        workload_id="raw_observation",
        shots=4,
        capture_semantics=True,
    )
    job = hal.submit("aws_braket_ionq", workload, approval_id="fault-contract-only")
    if case == "order":
        task.native_result = SimpleNamespace(measured_qubits=[1, 0], measurement_counts={"00": 4})
    elif case == "counts":
        task.native_result = SimpleNamespace(measured_qubits=[0, 1], measurement_counts=["00"])
    else:
        task.native_result = SimpleNamespace(
            measured_qubits=[0, 1], measurement_counts={"00": "4"}
        )
    with pytest.raises(ValueError):
        hal.result(job)
    task.native_result = raw
    result = hal.result(job)
    assert result.counts == {"00": 4}
    assert isinstance(result.provider_observation, GateModelObservation)
    raw.measurement_counts["00"] = 99
    assert result.counts == {"00": 4} and result.provider_observation.raw_counts == {"00": 4}


@pytest.mark.parametrize("backend_id", ["local_braket_sv", "aws_braket_ionq"])
def test_braket_profile_and_ir_failures_precede_device_construction(backend_id: str) -> None:
    """Invalid profiles, missing cloud approval and undecodable source have no transport effect."""
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    invalid_profile = hal.profile("local_qiskit_aer")
    factory_calls: list[str] = []

    def factory(arn: str) -> _FaultDevice:
        factory_calls.append(arn)
        return _FaultDevice(_FaultTask(SimpleNamespace(measurement_counts={"00": 4})))

    with pytest.raises(ValueError, match="profile"):
        BraketLocalHALAdapter(invalid_profile)
    with pytest.raises(ValueError, match="profile"):
        BraketAwsHALAdapter(invalid_profile, device=object())
    profile = hal.profile(backend_id)
    adapter = (
        BraketLocalHALAdapter(profile)
        if backend_id == "local_braket_sv"
        else BraketAwsHALAdapter(profile, device_arn="offline-arn", device_factory=factory)
    )
    hal.register_backend(adapter)
    corrupt = QuantumWorkload(
        workload_id="corrupt_qasm",
        ir_format="openqasm3",
        program="not valid qasm",
        n_qubits=2,
        shots=4,
    )
    with pytest.raises(ValueError, match="could not be decoded") as error:
        hal.submit(backend_id, corrupt, approval_id="fault-contract-only")
    assert error.value.__cause__ is not None and factory_calls == []
    if backend_id == "aws_braket_ionq":
        with pytest.raises(PermissionError, match="approval_id"):
            adapter.submit(
                braket_circuit_to_workload(_bell_circuit(), workload_id="no_go", shots=4)
            )
        with pytest.raises(ValueError, match="device or device_arn"):
            BraketAwsHALAdapter(profile)
        for probe, age in ((None, 1.0), (lambda: cast(ProviderCapabilitySnapshot, None), None)):
            with pytest.raises(ValueError, match="must be paired"):
                BraketAwsHALAdapter(
                    profile,
                    device=object(),
                    capability_probe=probe,
                    max_calibration_age_seconds=age,
                )


@pytest.mark.parametrize("default_loader", [False, True])
def test_braket_cloud_factory_executes_actual_local_native_sdk(
    monkeypatch: pytest.MonkeyPatch,
    default_loader: bool,
) -> None:
    """Configured AWS construction paths are exercised with an actual local simulator substitute."""
    from braket import aws

    calls: list[str] = []

    def factory(arn: str) -> LocalSimulator:
        calls.append(arn)
        return LocalSimulator()

    monkeypatch.setattr(aws, "AwsDevice", factory)
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    adapter = BraketAwsHALAdapter(
        hal.profile("aws_braket_ionq"),
        device_arn="software-only-arn",
        device_factory=None if default_loader else factory,
    )
    hal.register_backend(adapter)
    workload = braket_circuit_to_workload(
        Circuit().x(0).measure(0),
        workload_id="factory_native",
        shots=8,
        capture_semantics=True,
    )
    job = hal.submit("aws_braket_ionq", workload, approval_id="software-only")
    assert hal.result(job).counts == {"1": 8} and calls == ["software-only-arn"]


@pytest.mark.parametrize("case", ["fresh", "stale", "target", "shots", "simulator"])
def test_braket_submission_metadata_admits_only_exact_fresh_target(case: str) -> None:
    """Capability refusals precede transport; fresh metadata remains in the retained handle."""
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    profile = hal.profile("aws_braket_ionq")
    device = _FaultDevice(_FaultTask(SimpleNamespace(measurement_counts={"00": 4})))
    timestamp = (datetime.now(UTC) - timedelta(hours=1 if case == "stale" else 0)).isoformat()
    snapshot = ProviderCapabilitySnapshot(
        route_id="offline-braket-capability",
        aggregator=profile.broker,
        provider=profile.provider,
        backend_id=profile.backend_id,
        target_name="changed" if case == "target" else str(device.name),
        n_qubits=2,
        supported_ir_formats=("openqasm3",),
        online=True,
        simulator=case == "simulator",
        max_shots=3 if case == "shots" else 4,
        calibration_timestamp=timestamp,
    )
    hal.register_backend(
        BraketAwsHALAdapter(
            profile,
            device=device,
            capability_probe=lambda: snapshot,
            max_calibration_age_seconds=60,
        )
    )
    workload = braket_circuit_to_workload(_bell_circuit(), workload_id="metadata_gate", shots=4)
    if case == "fresh":
        job = hal.submit(profile.backend_id, workload, approval_id="fault-contract-only")
        assert job.metadata["calibration_timestamp"] == timestamp
        assert job.metadata["calibration_max_age_seconds"] == 60
        assert hal.result(job).counts == {"00": 4} and device.run_calls == 1
    else:
        with pytest.raises(ValueError, match="submit-time capability refused"):
            hal.submit(profile.backend_id, workload, approval_id="fault-contract-only")
        assert device.run_calls == 0


@pytest.mark.parametrize("name_kind", ["callable", "class"])
def test_braket_selected_native_device_identity_is_not_coerced(name_kind: str) -> None:
    """Native callable names and unnamed transport identities qualify exact target pins."""
    task = _FaultTask(SimpleNamespace(measured_qubits=[0], measurement_counts={"1": 4}))
    device = _FaultDevice(task)
    if name_kind == "callable":
        device.name = lambda: "native-callable-name"
        expected = "native-callable-name"
    else:
        del device.name
        expected = "_FaultDevice"
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    hal.register_backend(BraketLocalHALAdapter(hal.profile("local_braket_sv"), device=device))
    workload = braket_circuit_to_workload(
        Circuit().x(0).measure(0),
        workload_id="native_device_identity",
        shots=4,
        capture_semantics=True,
        requested_target=expected,
    )
    job = hal.submit("local_braket_sv", workload)
    assert job.submission is not None and job.submission.target_name == expected
    assert hal.result(job).counts == {"1": 4} and device.run_calls == 1


def test_braket_default_native_loader_failure_preserves_original_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Local SDK construction failures remain observable without a substitute transport."""
    from braket import devices

    original = RuntimeError("actual native simulator loader failure")
    calls: list[str] = []

    def failing_loader(backend: str) -> object:
        calls.append(backend)
        raise original

    monkeypatch.setattr(devices, "LocalSimulator", failing_loader)
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    hal.register_backend(BraketLocalHALAdapter(hal.profile("local_braket_sv")))
    workload = braket_circuit_to_workload(_bell_circuit(), workload_id="failed_sdk_load", shots=4)
    with pytest.raises(RuntimeError, match="amazon-braket-sdk") as error:
        hal.submit("local_braket_sv", workload)
    assert error.value.__cause__ is original and calls == ["braket_sv"]


def test_braket_decoder_rejects_incorrectly_advertised_ir_before_native_loading() -> None:
    """A custom profile cannot make the native Braket decoder accept a foreign IR."""
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    original = hal.profile("local_braket_sv")
    profile = replace(original, ir_formats=("openqasm3", "qiskit_qpy"))
    device = _FaultDevice(_FaultTask(object()))
    adapter = BraketLocalHALAdapter(profile, device=device)
    workload = QuantumWorkload(
        workload_id="foreign_ir",
        ir_format="qiskit_qpy",
        program="opaque-original-source",
        n_qubits=2,
        shots=4,
    )
    with pytest.raises(ValueError, match="require OpenQASM 3"):
        adapter.submit(workload)
    assert device.run_calls == 0


def test_braket_fixed_numeric_rotation_has_no_phantom_free_parameter_identity() -> None:
    """Actual native numeric gate values stay numeric through capture and execution."""
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    hal.register_backend(BraketLocalHALAdapter(hal.profile("local_braket_sv")))
    workload = braket_circuit_to_workload(
        Circuit().ry(0, 0.0).measure(0),
        workload_id="fixed_native_rotation",
        shots=8,
        capture_semantics=True,
    )
    assert (
        isinstance(workload.semantics, WorkloadSemantics) and workload.semantics.parameters == ()
    )
    assert hal.result(hal.submit("local_braket_sv", workload)).counts == {"0": 8}


@pytest.mark.parametrize("fault", ["selector", "transport", "stored_width"])
def test_braket_incomplete_retention_refuses_and_restores_without_evidence_loss(
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    """Controlled retention faults refuse through HAL while an unrelated native result stays intact."""
    factory_calls: list[str] = []

    def factory(arn: str) -> LocalSimulator:
        factory_calls.append(arn)
        return LocalSimulator()

    hal = HardwareAbstractionLayer.with_builtin_profiles()
    adapter = BraketAwsHALAdapter(
        hal.profile("aws_braket_ionq"),
        device_arn="software-only-retention-arn",
        device_factory=factory,
    )
    hal.register_backend(adapter)
    prior_workload = braket_circuit_to_workload(
        Circuit().x(0).measure(0),
        workload_id="prior_native_retention",
        shots=8,
    )
    prior = hal.submit("aws_braket_ionq", prior_workload, approval_id="software-only")
    saved = hal.result(prior)
    other_workload = braket_circuit_to_workload(
        Circuit().i(0).measure(0),
        workload_id="other_native_retention",
        shots=8,
    )
    other = hal.submit("aws_braket_ionq", other_workload, approval_id="software-only")
    with monkeypatch.context() as injected:
        if fault == "selector":
            injected.setattr(adapter, "_device_arn", None)
            with pytest.raises(ValueError, match="device_arn is required"):
                hal.submit("aws_braket_ionq", other_workload, approval_id="software-only")
        elif fault == "transport":
            injected.delitem(adapter._tasks, other.job_id)
            for operation in (hal.status, hal.result, hal.cancel):
                with pytest.raises(KeyError, match="unknown job_id"):
                    operation(other)
        else:
            damaged = replace(other, metadata={**other.metadata, "n_qubits": 0})
            injected.setitem(adapter._jobs, other.job_id, damaged)
            with pytest.raises(ValueError, match="positive n_qubits"):
                hal.result(other)
        assert hal.result(prior) is saved and saved.counts == {"1": 8}
        assert factory_calls == ["software-only-retention-arn"] * 2
    restored = hal.result(other)
    assert restored.counts == {"0": 8} and hal.result(other) is restored
    assert hal.result(prior) is saved and factory_calls == ["software-only-retention-arn"] * 2
