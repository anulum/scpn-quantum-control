# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- PennyLane HAL adapter tests
"""Tests for PennyLane execution behind the provider-neutral HAL."""

from __future__ import annotations

import pytest

from scpn_quantum_control.hardware.hal import HardwareAbstractionLayer, QuantumWorkload
from scpn_quantum_control.hardware.hal_pennylane import (
    PennyLaneDeviceHALAdapter,
    pennylane_gate_workload,
)


def test_pennylane_adapter_executes_native_gate_workload() -> None:
    """A Bell-state workload should execute through a PennyLane device."""

    pytest.importorskip("pennylane")
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    hal.register_backend(PennyLaneDeviceHALAdapter(hal.profile("local_pennylane")))
    workload = pennylane_gate_workload(
        [
            {"gate": "h", "wires": [0]},
            {"gate": "cnot", "wires": [0, 1]},
        ],
        workload_id="pl_bell",
        n_qubits=2,
        shots=128,
        metadata={"seed": 7},
    )

    job = hal.submit("local_pennylane", workload)
    result = hal.result(job)

    assert job.status == "completed"
    assert job.job_id.startswith("local_pennylane:pl_bell:")
    assert str(job.metadata["provider_job_id"]).startswith("pennylane-local:pl_bell:")
    assert result.status == "completed"
    assert sum(result.counts.values()) == 128
    assert set(result.counts) <= {"00", "11"}
    assert result.metadata["execution_mode"] == "pennylane_device"
    assert result.metadata["device_name"] == "default.qubit"


def test_pennylane_adapter_fails_closed_on_unknown_gate() -> None:
    """Unsupported PennyLane gate payloads must fail before execution."""

    hal = HardwareAbstractionLayer.with_builtin_profiles()
    hal.register_backend(PennyLaneDeviceHALAdapter(hal.profile("local_pennylane")))
    workload = QuantumWorkload(
        workload_id="pl_bad_gate",
        ir_format="pennylane",
        program='{"schema":"scpn.pennylane.native_gates.v1","instructions":[{"gate":"not_a_gate","wires":[0],"params":[]}]}',
        n_qubits=1,
        shots=4,
    )

    with pytest.raises(ValueError, match="unsupported PennyLane gate"):
        hal.submit("local_pennylane", workload)


def test_pennylane_local_lineage_is_unique_per_submission() -> None:
    pytest.importorskip("pennylane")
    hal = HardwareAbstractionLayer.with_builtin_profiles()
    hal.register_backend(PennyLaneDeviceHALAdapter(hal.profile("local_pennylane")))
    workload = pennylane_gate_workload(
        [{"gate": "x", "wires": [0]}],
        workload_id="pl_unique",
        n_qubits=1,
        shots=4,
    )

    first = hal.submit("local_pennylane", workload)
    second = hal.submit("local_pennylane", workload)

    assert first.job_id != second.job_id
    assert first.metadata["provider_job_id"] != second.metadata["provider_job_id"]
