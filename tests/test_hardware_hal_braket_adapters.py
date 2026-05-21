# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Braket HAL adapter tests
"""Tests for Braket adapters behind the provider-neutral HAL."""

from __future__ import annotations

import pytest
from braket.circuits import Circuit

from scpn_quantum_control.hardware.hal import HardwareAbstractionLayer
from scpn_quantum_control.hardware.hal_braket import (
    BraketAwsHALAdapter,
    BraketLocalHALAdapter,
    braket_circuit_to_workload,
)


def _bell_circuit() -> Circuit:
    return Circuit().h(0).cnot(0, 1)


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

        def run(self, circuit, shots: int):
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


def test_braket_aws_adapter_rejects_task_without_id() -> None:
    """AWS Braket adapter should fail closed when provider task id is missing."""

    class FakeTask:
        def state(self) -> str:
            return "COMPLETED"

        def result(self):
            return type("FakeTaskResult", (), {"measurement_counts": {"0": 1}})()

        def cancel(self) -> None:
            self.cancelled = True

    class FakeDevice:
        name = "fake-braket-device"

        def run(self, circuit, shots: int):
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

        def run(self, circuit, shots: int):
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
    profile = HardwareAbstractionLayer.with_builtin_profiles().profile("aws_braket_ionq")
    with pytest.raises(ValueError, match="Braket device ARN"):
        BraketAwsHALAdapter(
            profile,
            device_arn="arn:aws:braket:us-east-1::device/qpu/ionq/\naria-1",
            device_factory=lambda arn: arn,
        )


def test_braket_aws_device_arn_trims_padding() -> None:
    profile = HardwareAbstractionLayer.with_builtin_profiles().profile("aws_braket_ionq")
    adapter = BraketAwsHALAdapter(
        profile,
        device_arn="  arn:aws:braket:us-east-1::device/qpu/ionq/aria-1  ",
        device_factory=lambda arn: arn,
    )
    assert adapter._device_arn == "arn:aws:braket:us-east-1::device/qpu/ionq/aria-1"
