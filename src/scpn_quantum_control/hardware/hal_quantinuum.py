# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Quantinuum adapter for the hardware HAL
"""Quantinuum pytket adapter for :mod:`scpn_quantum_control.hardware.hal`."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from importlib import import_module
from typing import Any

from ._count_integrity import (
    strict_integer_value,
    strict_non_negative_count,
    strict_provider_job_id,
    strict_shot_conservation,
)
from .hal import BackendProfile, QuantumJobRef, QuantumJobResult, QuantumWorkload

QUANTINUUM_EXECUTION_MODE = "quantinuum_pytket"
QUANTINUUM_TKET_SCHEMA = "scpn.quantinuum.tket_circuit.v1"


def quantinuum_tket_workload(
    circuit: Mapping[str, object] | str | Any,
    *,
    workload_id: str,
    n_qubits: int,
    shots: int,
    metadata: Mapping[str, object] | None = None,
) -> QuantumWorkload:
    """Encode a pytket circuit dictionary as a HAL workload for Quantinuum."""

    if isinstance(circuit, str):
        circuit_payload = json.loads(circuit)
    elif hasattr(circuit, "to_dict"):
        circuit_payload = circuit.to_dict()
    else:
        circuit_payload = dict(circuit)
    payload = {
        "schema": QUANTINUUM_TKET_SCHEMA,
        "circuit": circuit_payload,
    }
    return QuantumWorkload(
        workload_id=workload_id,
        ir_format="tket",
        program=json.dumps(payload, sort_keys=True, separators=(",", ":")),
        n_qubits=n_qubits,
        shots=shots,
        metadata=dict(metadata or {}),
    )


class QuantinuumCloudHALAdapter:
    """pytket-quantinuum adapter implementing the provider-neutral HAL protocol."""

    def __init__(
        self,
        profile: BackendProfile,
        *,
        backend: Any | None = None,
        machine: str | None = None,
        backend_factory: Callable[[str], Any] | None = None,
        circuit_factory: Callable[[QuantumWorkload], Any] | None = None,
        compile_circuit: bool = True,
    ) -> None:
        if profile.backend_id != "quantinuum_cloud":
            raise ValueError("QuantinuumCloudHALAdapter requires the quantinuum_cloud profile")
        if backend is None and machine is None:
            raise ValueError("backend or machine is required")
        self.profile = profile
        self.backend_id = profile.backend_id
        self._backend = backend
        self._machine = strict_provider_job_id(
            machine or "injected", field_name="Quantinuum machine"
        )
        self._backend_factory = backend_factory
        self._circuit_factory = circuit_factory
        self._compile_circuit = compile_circuit
        self._jobs: dict[str, QuantumJobRef] = {}
        self._handles: dict[str, Any] = {}

    def submit(
        self, workload: QuantumWorkload, *, approval_id: str | None = None
    ) -> QuantumJobRef:
        """Submit a workload to the backend and return its job reference."""
        if not approval_id:
            raise PermissionError("approval_id is required for Quantinuum submission")
        if workload.ir_format != "tket":
            raise ValueError("Quantinuum direct adapter requires tket workloads")

        circuit = self._build_circuit(workload)
        executable = self._compile(circuit)
        handle = self._backend_client().process_circuit(executable, n_shots=workload.shots)
        provider_job_id = _provider_job_id(handle)
        job = QuantumJobRef(
            job_id=_job_id(self.backend_id, workload.workload_id, provider_job_id),
            backend_id=self.backend_id,
            workload_id=workload.workload_id,
            status="submitted",
            metadata={
                "approval_id": approval_id,
                "provider_job_id": provider_job_id,
                "execution_mode": QUANTINUUM_EXECUTION_MODE,
                "machine": self._machine,
                "ir_format": workload.ir_format,
                "n_qubits": workload.n_qubits,
                "shots": workload.shots,
            },
        )
        self._jobs[job.job_id] = job
        self._handles[job.job_id] = handle
        return job

    def status(self, job: QuantumJobRef) -> str:
        """Return the current status for a submitted backend job."""
        handle = self._handle(job)
        return _normalise_status(self._backend_client().circuit_status(handle))

    def result(self, job: QuantumJobRef) -> QuantumJobResult:
        """Return the completed result for a submitted backend job."""
        stored = self._job(job)
        handle = self._handle(job)
        result = self._backend_client().get_result(handle)
        counts = _normalise_counts(result.get_counts())
        expected_shots = _int_metadata(stored.metadata.get("shots"), fallback=0)
        observed_shots = strict_shot_conservation(counts, expected_shots=expected_shots)
        return QuantumJobResult(
            job=stored,
            status="completed",
            counts=counts,
            shots=observed_shots,
            metadata={
                "approval_id": stored.metadata.get("approval_id"),
                "execution_mode": QUANTINUUM_EXECUTION_MODE,
                "machine": self._machine,
                "timestamp": _utc_now(),
            },
        )

    def cancel(self, job: QuantumJobRef) -> QuantumJobRef:
        """Request cancellation for a submitted backend job."""
        stored = self._job(job)
        handle = self._handle(job)
        self._backend_client().cancel(handle)
        cancelled = QuantumJobRef(
            job_id=stored.job_id,
            backend_id=stored.backend_id,
            workload_id=stored.workload_id,
            status="cancelled",
            metadata=stored.metadata,
        )
        self._jobs[job.job_id] = cancelled
        return cancelled

    def _backend_client(self) -> Any:
        if self._backend is not None:
            return self._backend
        factory = self._backend_factory or _default_backend_factory
        self._backend = factory(self._machine)
        return self._backend

    def _build_circuit(self, workload: QuantumWorkload) -> Any:
        factory = self._circuit_factory or _default_circuit_factory
        return factory(workload)

    def _compile(self, circuit: Any) -> Any:
        if not self._compile_circuit:
            return circuit
        compile_method = getattr(self._backend_client(), "get_compiled_circuit", None)
        if compile_method is None:
            raise TypeError("Quantinuum backend object does not provide get_compiled_circuit()")
        return compile_method(circuit)

    def _job(self, job: QuantumJobRef) -> QuantumJobRef:
        stored = self._jobs.get(job.job_id)
        if stored is None:
            raise KeyError(f"unknown job_id: {job.job_id}")
        return stored

    def _handle(self, job: QuantumJobRef) -> Any:
        self._job(job)
        return self._handles[job.job_id]


def _default_backend_factory(machine: str) -> Any:
    try:
        quantinuum_module = import_module("pytket.extensions.quantinuum")
    except Exception as exc:
        raise RuntimeError("pytket-quantinuum is required for QuantinuumCloudHALAdapter") from exc
    return quantinuum_module.QuantinuumBackend(machine)


def _default_circuit_factory(workload: QuantumWorkload) -> Any:
    try:
        payload = json.loads(workload.program)
    except json.JSONDecodeError as exc:
        raise ValueError("Quantinuum workload is not valid JSON") from exc
    if payload.get("schema") != QUANTINUUM_TKET_SCHEMA:
        raise ValueError("unsupported Quantinuum workload schema")
    circuit_payload = payload.get("circuit")
    if not isinstance(circuit_payload, Mapping):
        raise ValueError("Quantinuum workload circuit must be a JSON object")
    try:
        circuit_cls = import_module("pytket.circuit").Circuit
    except Exception as exc:
        raise RuntimeError("pytket is required to construct Quantinuum circuits") from exc
    return circuit_cls.from_dict(dict(circuit_payload))


def _normalise_counts(raw_counts: Mapping[Any, int]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for raw_key, raw_count in raw_counts.items():
        count = strict_non_negative_count(raw_count)
        bitstring = _normalise_bit_key(raw_key)
        counts[bitstring] = counts.get(bitstring, 0) + count
    if not counts:
        raise ValueError("Quantinuum result did not contain counts")
    return counts


def _normalise_bit_key(raw_key: Any) -> str:
    if isinstance(raw_key, str):
        bits = [int(bit) for bit in raw_key]
    elif hasattr(raw_key, "to_list"):
        bits = [int(bit) for bit in raw_key.to_list()]
    elif isinstance(raw_key, Sequence):
        bits = [int(bit) for bit in raw_key]
    else:
        raise ValueError("Quantinuum count keys must be bitstrings or bit sequences")
    if not bits:
        raise ValueError("Quantinuum count keys must not be empty")
    if any(bit not in (0, 1) for bit in bits):
        raise ValueError("Quantinuum count keys must contain binary values")
    return "".join(str(bit) for bit in bits)


def _normalise_status(status: Any) -> str:
    raw_status = getattr(status, "status", status)
    if hasattr(raw_status, "name"):
        raw_status = raw_status.name
    text = str(raw_status).split(".")[-1].strip().lower().replace(" ", "_")
    return {
        "done": "completed",
        "complete": "completed",
        "completed": "completed",
        "finished": "completed",
        "success": "completed",
        "succeeded": "completed",
        "queued": "queued",
        "pending": "queued",
        "running": "running",
        "in_progress": "running",
        "in-progress": "running",
        "inprogress": "running",
        "initializing": "submitted",
        "initialising": "submitted",
        "starting": "submitted",
        "creating": "submitted",
        "created": "submitted",
        "cancelled": "cancelled",
        "canceled": "cancelled",
        "aborting": "cancelled",
        "cancelling": "cancelled",
        "canceling": "cancelled",
        "failed": "failed",
        "error": "failed",
    }.get(text, "unknown")


def _provider_job_id(handle: Any) -> str:
    for attr in ("job_id", "id", "handle", "task_id"):
        value = getattr(handle, attr, None)
        if callable(value):
            value = value()
        if value is not None and str(value).strip():
            return strict_provider_job_id(value, field_name="Quantinuum provider job id")
    if isinstance(handle, Mapping):
        for key in ("job_id", "id", "handle", "task_id"):
            value = handle.get(key)
            if value is not None and str(value).strip():
                return strict_provider_job_id(value, field_name="Quantinuum provider job id")

    provider_job_id = str(handle).strip()
    if (
        not provider_job_id
        or provider_job_id.lower() == "none"
        or (
            provider_job_id.startswith("<")
            and provider_job_id.endswith(">")
            and " object at 0x" in provider_job_id
        )
    ):
        raise ValueError("Quantinuum backend process_circuit returned an invalid provider handle")
    return strict_provider_job_id(provider_job_id, field_name="Quantinuum provider job id")


def _job_id(backend_id: str, workload_id: str, provider_job_id: str) -> str:
    digest = hashlib.sha256(provider_job_id.encode("utf-8")).hexdigest()[:12]
    return f"{backend_id}:{workload_id}:{digest}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _int_metadata(value: object, *, fallback: int) -> int:
    try:
        integer = strict_integer_value(value, field_name="Quantinuum metadata")
    except ValueError:
        return fallback
    return integer if integer > 0 else fallback


__all__ = [
    "QUANTINUUM_EXECUTION_MODE",
    "QUANTINUUM_TKET_SCHEMA",
    "QuantinuumCloudHALAdapter",
    "quantinuum_tket_workload",
]
