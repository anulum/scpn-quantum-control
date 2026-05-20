# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Braket HAL adapter tests
"""Tests for Braket adapters behind the provider-neutral HAL."""

from __future__ import annotations

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
    assert result.counts == {"0": 4, "1": 2}
    assert result.metadata["execution_mode"] == "braket_aws"
    assert result.metadata["approval_id"] == "approved-braket"
