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

from scpn_quantum_control.hardware import hal_pennylane as pennylane_mod
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


def test_pennylane_adapter_rejects_shot_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """PennyLane adapter must fail closed when sampled counts diverge from expected shots."""

    class _FakeQml:
        @staticmethod
        def device(*args: object, **kwargs: object) -> object:
            del args, kwargs
            return object()

    monkeypatch.setattr(pennylane_mod, "_load_pennylane", lambda: _FakeQml())
    monkeypatch.setattr(
        pennylane_mod,
        "_execute_native_gates",
        lambda qml, device, instructions, n_qubits, shots: {"0" * n_qubits: max(shots - 1, 0)},
    )

    hal = HardwareAbstractionLayer.with_builtin_profiles()
    hal.register_backend(PennyLaneDeviceHALAdapter(hal.profile("local_pennylane")))
    workload = pennylane_gate_workload(
        [{"gate": "x", "wires": [0]}],
        workload_id="pl_shot_mismatch",
        n_qubits=1,
        shots=4,
    )

    with pytest.raises(ValueError, match="shot count mismatch"):
        hal.submit("local_pennylane", workload)


def test_pennylane_provider_job_id_uses_strict_validator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PennyLane lineage IDs must pass through strict provider-id validation."""

    seen: dict[str, str] = {}

    def _fake_strict_provider_job_id(value: object, *, field_name: str) -> str:
        seen["value"] = str(value)
        seen["field_name"] = field_name
        raise ValueError("provider job id sentinel")

    monkeypatch.setattr(pennylane_mod, "strict_provider_job_id", _fake_strict_provider_job_id)

    with pytest.raises(ValueError, match="provider job id sentinel"):
        pennylane_mod._provider_job_id(
            QuantumWorkload(
                workload_id="pl_bad_id",
                ir_format="pennylane",
                program='{"schema":"scpn.pennylane.native_gates.v1","instructions":[]}',
                n_qubits=1,
                shots=1,
            )
        )

    assert seen["field_name"] == "PennyLane provider job id"
    assert seen["value"].startswith("pennylane-local:pl_bad_id:")


def test_pennylane_provider_job_id_trims_padding() -> None:
    """PennyLane lineage IDs should remain canonical without surrounding whitespace."""

    provider_job_id = pennylane_mod._provider_job_id(
        QuantumWorkload(
            workload_id="pl_trim",
            ir_format="pennylane",
            program='{"schema":"scpn.pennylane.native_gates.v1","instructions":[]}',
            n_qubits=1,
            shots=1,
        )
    )
    assert provider_job_id == provider_job_id.strip()
