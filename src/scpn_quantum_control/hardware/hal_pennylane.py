# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- PennyLane adapter for the hardware HAL
"""PennyLane-backed adapter for :mod:`scpn_quantum_control.hardware.hal`."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from numbers import Real
from time import time_ns
from typing import Any, cast

import numpy as np

from ._count_integrity import strict_provider_job_id, strict_shot_conservation
from .hal import BackendProfile, QuantumJobRef, QuantumJobResult, QuantumWorkload

_SUPPORTED_GATES = frozenset(
    {
        "h",
        "x",
        "y",
        "z",
        "s",
        "t",
        "rx",
        "ry",
        "rz",
        "cnot",
        "cx",
        "cz",
        "swap",
    }
)
_PARAMETER_COUNTS: Mapping[str, int] = {
    "rx": 1,
    "ry": 1,
    "rz": 1,
}
_WIRE_COUNTS: Mapping[str, int] = {
    "h": 1,
    "x": 1,
    "y": 1,
    "z": 1,
    "s": 1,
    "t": 1,
    "rx": 1,
    "ry": 1,
    "rz": 1,
    "cnot": 2,
    "cx": 2,
    "cz": 2,
    "swap": 2,
}


def pennylane_gate_workload(
    instructions: Sequence[Mapping[str, object]],
    *,
    workload_id: str,
    n_qubits: int,
    shots: int,
    metadata: Mapping[str, object] | None = None,
) -> QuantumWorkload:
    """Encode a strict PennyLane native-gate instruction payload as HAL work."""

    payload = {
        "schema": "scpn.pennylane.native_gates.v1",
        "instructions": [
            _normalise_instruction(instruction, n_qubits) for instruction in instructions
        ],
    }
    return QuantumWorkload(
        workload_id=workload_id,
        ir_format="pennylane",
        program=json.dumps(payload, sort_keys=True, separators=(",", ":")),
        n_qubits=n_qubits,
        shots=shots,
        metadata=dict(metadata or {}),
    )


class PennyLaneDeviceHALAdapter:
    """Local PennyLane device adapter implementing the HAL protocol."""

    def __init__(
        self,
        profile: BackendProfile,
        *,
        device_name: str = "default.qubit",
        device: Any | None = None,
        device_kwargs: Mapping[str, object] | None = None,
    ) -> None:
        if profile.backend_id != "local_pennylane":
            raise ValueError("PennyLaneDeviceHALAdapter requires the local_pennylane profile")
        self.profile = profile
        self.backend_id = profile.backend_id
        self.device_name = _normalise_device_name(device_name)
        self._device = device
        self._device_kwargs = dict(device_kwargs or {})
        self._jobs: dict[str, QuantumJobRef] = {}
        self._results: dict[str, QuantumJobResult] = {}

    def submit(
        self, workload: QuantumWorkload, *, approval_id: str | None = None
    ) -> QuantumJobRef:
        """Submit a workload to the backend and return its job reference."""
        del approval_id
        instructions = _decode_native_gate_payload(workload)
        qml = _load_pennylane()
        device = self._device or qml.device(
            self.device_name,
            wires=workload.n_qubits,
            shots=workload.shots,
            **self._device_kwargs,
        )
        counts = _execute_native_gates(
            qml, device, instructions, workload.n_qubits, workload.shots
        )
        observed_shots = strict_shot_conservation(counts, expected_shots=workload.shots)
        provider_job_id = _provider_job_id(workload)
        job_id = _hal_job_id(self.backend_id, workload.workload_id, provider_job_id)
        job = QuantumJobRef(
            job_id=job_id,
            backend_id=self.backend_id,
            workload_id=workload.workload_id,
            status="completed",
            metadata={
                "provider_job_id": provider_job_id,
                "execution_mode": "pennylane_device",
                "ir_format": workload.ir_format,
                "device_name": self.device_name,
            },
        )
        result = QuantumJobResult(
            job=job,
            status="completed",
            counts=counts,
            shots=observed_shots,
            metadata={
                "execution_mode": "pennylane_device",
                "ir_format": workload.ir_format,
                "device_name": self.device_name,
                "timestamp": _utc_now(),
            },
        )
        self._jobs[job.job_id] = job
        self._results[job.job_id] = result
        return job

    def status(self, job: QuantumJobRef) -> str:
        """Return the current status for a submitted backend job."""
        stored = self._jobs.get(job.job_id)
        if stored is None:
            raise KeyError(f"unknown job_id: {job.job_id}")
        return stored.status

    def result(self, job: QuantumJobRef) -> QuantumJobResult:
        """Return the completed result for a submitted backend job."""
        result = self._results.get(job.job_id)
        if result is None:
            raise KeyError(f"unknown job_id: {job.job_id}")
        return result

    def cancel(self, job: QuantumJobRef) -> QuantumJobRef:
        """Request cancellation for a submitted backend job."""
        if job.job_id not in self._jobs:
            raise KeyError(f"unknown job_id: {job.job_id}")
        cancelled = QuantumJobRef(
            job_id=job.job_id,
            backend_id=job.backend_id,
            workload_id=job.workload_id,
            status="cancelled",
            metadata=job.metadata,
        )
        self._jobs[job.job_id] = cancelled
        return cancelled


def _normalise_instruction(instruction: Mapping[str, object], n_qubits: int) -> dict[str, object]:
    gate = str(instruction.get("gate", "")).lower()
    if gate not in _SUPPORTED_GATES:
        raise ValueError(f"unsupported PennyLane gate: {gate}")
    wires = instruction.get("wires")
    if not isinstance(wires, Sequence) or isinstance(wires, str):
        raise ValueError("PennyLane instruction wires must be a sequence of integers")
    wire_tuple = tuple(_normalise_wire(wire) for wire in wires)
    if not wire_tuple:
        raise ValueError("PennyLane instruction wires must not be empty")
    expected_wires = _WIRE_COUNTS[gate]
    if len(wire_tuple) != expected_wires:
        raise ValueError(f"PennyLane gate {gate} requires {expected_wires} wires")
    if len(set(wire_tuple)) != len(wire_tuple):
        raise ValueError("PennyLane instruction wires must be unique")
    if any(wire < 0 or wire >= n_qubits for wire in wire_tuple):
        raise ValueError("PennyLane instruction wire is outside the workload register")
    params = instruction.get("params", ())
    if not isinstance(params, Sequence) or isinstance(params, str):
        raise ValueError("PennyLane instruction params must be a sequence of numbers")
    param_tuple = tuple(_normalise_parameter(param) for param in params)
    expected_params = _PARAMETER_COUNTS.get(gate, 0)
    if len(param_tuple) != expected_params:
        raise ValueError(f"PennyLane gate {gate} requires {expected_params} parameters")
    return {
        "gate": gate,
        "wires": list(wire_tuple),
        "params": list(param_tuple),
    }


def _normalise_wire(wire: object) -> int:
    if isinstance(wire, bool) or not isinstance(wire, int):
        raise ValueError("PennyLane instruction wires must be integers")
    return wire


def _normalise_parameter(parameter: object) -> float:
    if isinstance(parameter, bool) or not isinstance(parameter, Real):
        raise ValueError("PennyLane instruction params must be real numbers")
    value = float(parameter)
    if not math.isfinite(value):
        raise ValueError("PennyLane instruction params must be finite")
    return value


def _normalise_device_name(device_name: str) -> str:
    normalised = str(device_name).strip()
    if not normalised:
        raise ValueError("PennyLane device name must not be empty")
    if any(ord(character) < 32 or ord(character) == 127 for character in normalised):
        raise ValueError("PennyLane device name must not contain control characters")
    return normalised


def _decode_native_gate_payload(workload: QuantumWorkload) -> tuple[dict[str, object], ...]:
    if workload.ir_format != "pennylane":
        raise ValueError("PennyLane adapter requires pennylane workloads")
    try:
        payload = json.loads(workload.program)
    except json.JSONDecodeError as exc:
        raise ValueError("PennyLane workload is not valid JSON") from exc
    if payload.get("schema") != "scpn.pennylane.native_gates.v1":
        raise ValueError("unsupported PennyLane workload schema")
    instructions = payload.get("instructions")
    if not isinstance(instructions, list):
        raise ValueError("PennyLane workload instructions must be a list")
    normalised: list[dict[str, object]] = []
    for item in instructions:
        if not isinstance(item, Mapping):
            raise ValueError("PennyLane workload instructions must be objects")
        normalised.append(_normalise_instruction(item, workload.n_qubits))
    return tuple(normalised)


def _execute_native_gates(
    qml: Any,
    device: Any,
    instructions: Sequence[Mapping[str, object]],
    n_qubits: int,
    shots: int,
) -> dict[str, int]:
    def _circuit() -> Any:
        for instruction in instructions:
            _apply_gate(qml, instruction)
        return qml.sample(wires=list(range(n_qubits)))

    circuit = qml.qnode(device)(_circuit)
    samples = np.asarray(circuit(), dtype=int)
    if samples.ndim == 1:
        samples = samples.reshape((shots, n_qubits))
    counts: dict[str, int] = {}
    for row in samples:
        bitstring = "".join(str(int(bit)) for bit in row)
        counts[bitstring] = counts.get(bitstring, 0) + 1
    return counts


def _apply_gate(qml: Any, instruction: Mapping[str, object]) -> None:
    gate = str(instruction["gate"])
    wires = tuple(cast(Sequence[int], instruction["wires"]))
    params = tuple(cast(Sequence[float], instruction["params"]))
    if gate == "h":
        _require_arity(gate, wires, 1)
        qml.Hadamard(wires=wires[0])
    elif gate == "x":
        _require_arity(gate, wires, 1)
        qml.PauliX(wires=wires[0])
    elif gate == "y":
        _require_arity(gate, wires, 1)
        qml.PauliY(wires=wires[0])
    elif gate == "z":
        _require_arity(gate, wires, 1)
        qml.PauliZ(wires=wires[0])
    elif gate == "s":
        _require_arity(gate, wires, 1)
        qml.S(wires=wires[0])
    elif gate == "t":
        _require_arity(gate, wires, 1)
        qml.T(wires=wires[0])
    elif gate == "rx":
        _require_arity(gate, wires, 1)
        _require_params(gate, params, 1)
        qml.RX(params[0], wires=wires[0])
    elif gate == "ry":
        _require_arity(gate, wires, 1)
        _require_params(gate, params, 1)
        qml.RY(params[0], wires=wires[0])
    elif gate == "rz":
        _require_arity(gate, wires, 1)
        _require_params(gate, params, 1)
        qml.RZ(params[0], wires=wires[0])
    elif gate in {"cnot", "cx"}:
        _require_arity(gate, wires, 2)
        qml.CNOT(wires=wires)
    elif gate == "cz":
        _require_arity(gate, wires, 2)
        qml.CZ(wires=wires)
    elif gate == "swap":
        _require_arity(gate, wires, 2)
        qml.SWAP(wires=wires)
    else:
        raise ValueError(f"unsupported PennyLane gate: {gate}")


def _require_arity(gate: str, wires: Sequence[int], expected: int) -> None:
    if len(wires) != expected:
        raise ValueError(f"PennyLane gate {gate} requires {expected} wires")


def _require_params(gate: str, params: Sequence[float], expected: int) -> None:
    if len(params) != expected:
        raise ValueError(f"PennyLane gate {gate} requires {expected} parameters")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _provider_job_id(workload: QuantumWorkload) -> str:
    program_digest = _program_digest(workload.program)
    submitted_ns = time_ns()
    raw_provider_job_id = f"pennylane-local:{workload.workload_id}:{program_digest}:{submitted_ns}"
    return strict_provider_job_id(raw_provider_job_id, field_name="PennyLane provider job id")


def _program_digest(program: str) -> str:
    return hashlib.sha256(program.encode("utf-8")).hexdigest()[:12]


def _hal_job_id(backend_id: str, workload_id: str, provider_job_id: str) -> str:
    digest = hashlib.sha256(f"{backend_id}|{provider_job_id}".encode()).hexdigest()[:12]
    return f"{backend_id}:{workload_id}:{digest}"


def _load_pennylane() -> Any:
    try:
        import pennylane as qml

        return qml
    except Exception as exc:
        raise RuntimeError("pennylane is required for PennyLaneDeviceHALAdapter") from exc


__all__ = [
    "PennyLaneDeviceHALAdapter",
    "pennylane_gate_workload",
]
