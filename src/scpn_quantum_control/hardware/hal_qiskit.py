# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Qiskit adapters for the hardware HAL
"""Qiskit-backed adapters for :mod:`scpn_quantum_control.hardware.hal`."""

from __future__ import annotations

import base64
import io
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any, cast

from qiskit import QuantumCircuit, qasm3, qpy, transpile
from qiskit.qpy import dump as qpy_dump

from ._count_integrity import (
    strict_binary_bitstring_key,
    strict_integer_value,
    strict_non_negative_count,
    strict_provider_job_id,
    strict_shot_conservation,
)
from .hal import BackendProfile, QuantumJobRef, QuantumJobResult, QuantumWorkload
from .runner import _extract_counts


def qiskit_circuit_to_workload(
    circuit: QuantumCircuit,
    *,
    workload_id: str,
    shots: int,
    metadata: dict[str, object] | None = None,
) -> QuantumWorkload:
    """Encode a Qiskit circuit as a QPY-backed HAL workload."""

    if not isinstance(circuit, QuantumCircuit):
        raise TypeError("circuit must be a qiskit.QuantumCircuit")
    return QuantumWorkload(
        workload_id=workload_id,
        ir_format="qiskit_qpy",
        program=_circuit_to_qpy_b64(circuit),
        n_qubits=circuit.num_qubits,
        shots=shots,
        metadata=metadata or {},
    )


def qiskit_circuit_to_qasm3_workload(
    circuit: QuantumCircuit,
    *,
    workload_id: str,
    shots: int,
    metadata: dict[str, object] | None = None,
) -> QuantumWorkload:
    """Encode a Qiskit circuit as an OpenQASM 3 HAL workload."""

    if not isinstance(circuit, QuantumCircuit):
        raise TypeError("circuit must be a qiskit.QuantumCircuit")
    return QuantumWorkload(
        workload_id=workload_id,
        ir_format="openqasm3",
        program=qasm3.dumps(circuit),
        n_qubits=circuit.num_qubits,
        shots=shots,
        metadata=metadata or {},
    )


class QiskitAerHALAdapter:
    """Local Aer adapter implementing the provider-neutral HAL protocol."""

    def __init__(self, profile: BackendProfile, *, backend: Any | None = None) -> None:
        if profile.backend_id != "local_qiskit_aer":
            raise ValueError("QiskitAerHALAdapter requires the local_qiskit_aer profile")
        self.profile = profile
        self.backend_id = profile.backend_id
        self._backend = backend
        self._jobs: dict[str, QuantumJobRef] = {}
        self._results: dict[str, QuantumJobResult] = {}

    def submit(
        self, workload: QuantumWorkload, *, approval_id: str | None = None
    ) -> QuantumJobRef:
        """Submit a workload to the backend and return its job reference."""
        del approval_id
        circuit = _workload_to_qiskit_circuit(workload)
        backend = self._backend or _default_aer_backend()
        compiled = transpile(circuit, backend)
        provider_job = backend.run(compiled, shots=workload.shots)
        counts = _normalise_counts(provider_job.result().get_counts())
        observed_shots = strict_shot_conservation(counts, expected_shots=workload.shots)
        job_id = _provider_job_id(provider_job, provider_name="qiskit_aer")
        job = QuantumJobRef(
            job_id=job_id,
            backend_id=self.backend_id,
            workload_id=workload.workload_id,
            status="completed",
            metadata={
                "provider_job_id": job_id,
                "execution_mode": "qiskit_aer",
                "ir_format": workload.ir_format,
            },
        )
        result = QuantumJobResult(
            job=job,
            status="completed",
            counts=counts,
            shots=observed_shots,
            metadata={
                "execution_mode": "qiskit_aer",
                "ir_format": workload.ir_format,
                "backend_name": _backend_name(backend),
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


class QiskitRuntimeHALAdapter:
    """IBM Runtime Sampler adapter implementing the HAL protocol."""

    def __init__(
        self,
        profile: BackendProfile,
        *,
        backend: Any,
        sampler_factory: Callable[..., Any] | None = None,
        timeout_s: float = 600.0,
    ) -> None:
        if profile.backend_id != "ibm_quantum":
            raise ValueError("QiskitRuntimeHALAdapter requires the ibm_quantum profile")
        self.profile = profile
        self.backend_id = profile.backend_id
        self._backend = backend
        self._sampler_factory = sampler_factory
        self.timeout_s = timeout_s
        self._provider_jobs: dict[str, Any] = {}
        self._jobs: dict[str, QuantumJobRef] = {}
        self._results: dict[str, QuantumJobResult] = {}

    def submit(
        self, workload: QuantumWorkload, *, approval_id: str | None = None
    ) -> QuantumJobRef:
        """Submit a workload to the backend and return its job reference."""
        if not approval_id:
            raise PermissionError("approval_id is required for IBM Runtime submission")
        circuit = _workload_to_qiskit_circuit(workload)
        sampler_factory = self._sampler_factory or _runtime_sampler_factory()
        sampler = sampler_factory(mode=self._backend)
        sampler.options.default_shots = workload.shots
        provider_job = sampler.run([circuit])
        job_id = _provider_job_id(provider_job, provider_name="qiskit_runtime")
        job = QuantumJobRef(
            job_id=job_id,
            backend_id=self.backend_id,
            workload_id=workload.workload_id,
            status="submitted",
            metadata={
                "approval_id": approval_id,
                "execution_mode": "qiskit_runtime_sampler",
                "backend_name": _backend_name(self._backend),
                "ir_format": workload.ir_format,
                "shots": workload.shots,
            },
        )
        self._provider_jobs[job.job_id] = provider_job
        self._jobs[job.job_id] = job
        return job

    def status(self, job: QuantumJobRef) -> str:
        """Return the current status for a submitted backend job."""
        provider_job = self._provider_job(job)
        status = provider_job.status()
        return _normalise_status(getattr(status, "name", status))

    def result(self, job: QuantumJobRef) -> QuantumJobResult:
        """Return the completed result for a submitted backend job."""
        cached = self._results.get(job.job_id)
        if cached is not None:
            return cached
        provider_job = self._provider_job(job)
        runtime_result = provider_job.result(timeout=self.timeout_s)
        counts: dict[str, int] = {}
        for pub_result in runtime_result:
            for bitstring, count in _normalise_counts(_extract_counts(pub_result)).items():
                counts[bitstring] = counts.get(bitstring, 0) + count
        expected_shots = strict_integer_value(job.metadata.get("shots", 0), field_name="shots")
        observed_shots = strict_shot_conservation(counts, expected_shots=expected_shots)
        result = QuantumJobResult(
            job=job,
            status="completed",
            counts=counts,
            shots=observed_shots,
            metadata={
                "approval_id": job.metadata.get("approval_id"),
                "execution_mode": "qiskit_runtime_sampler",
                "backend_name": job.metadata.get("backend_name"),
                "ir_format": job.metadata.get("ir_format"),
                "timestamp": _utc_now(),
            },
        )
        self._results[job.job_id] = result
        return result

    def cancel(self, job: QuantumJobRef) -> QuantumJobRef:
        """Request cancellation for a submitted backend job."""
        provider_job = self._provider_job(job)
        provider_job.cancel()
        cancelled = QuantumJobRef(
            job_id=job.job_id,
            backend_id=job.backend_id,
            workload_id=job.workload_id,
            status="cancelled",
            metadata=job.metadata,
        )
        self._jobs[job.job_id] = cancelled
        return cancelled

    def _provider_job(self, job: QuantumJobRef) -> Any:
        provider_job = self._provider_jobs.get(job.job_id)
        if provider_job is None:
            raise KeyError(f"unknown job_id: {job.job_id}")
        return provider_job


def _workload_to_qiskit_circuit(workload: QuantumWorkload) -> QuantumCircuit:
    if workload.ir_format == "qiskit_qpy":
        return _qpy_b64_to_circuit(workload.program)
    if workload.ir_format == "openqasm3":
        try:
            return qasm3.loads(workload.program)
        except Exception as exc:
            raise ValueError("OpenQASM 3 workload could not be decoded by Qiskit") from exc
    raise ValueError("Qiskit adapters require qiskit_qpy or OpenQASM 3 workloads")


def _circuit_to_qpy_b64(circuit: QuantumCircuit) -> str:
    buffer = io.BytesIO()
    qpy_dump(circuit, buffer)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _qpy_b64_to_circuit(payload: str) -> QuantumCircuit:
    try:
        data = base64.b64decode(payload.encode("ascii"), validate=True)
        circuits = _reviewed_qpy_load_circuits(data)
    except Exception as exc:
        raise ValueError("qiskit_qpy workload could not be decoded") from exc
    if len(circuits) != 1:
        raise ValueError("qiskit_qpy workload must contain exactly one circuit")
    circuit = circuits[0]
    if not isinstance(circuit, QuantumCircuit):
        raise TypeError("qiskit_qpy payload did not decode to a QuantumCircuit")
    return circuit


def _reviewed_qpy_load_circuits(data: bytes) -> list[QuantumCircuit]:
    """Decode trusted in-process QPY bytes behind the reviewed HAL wrapper."""

    return cast(list[QuantumCircuit], qpy.load(io.BytesIO(data)))


def _default_aer_backend() -> Any:
    try:
        from qiskit_aer import AerSimulator

        return AerSimulator()
    except Exception as exc:
        raise RuntimeError("qiskit-aer is required for QiskitAerHALAdapter") from exc


def _runtime_sampler_factory() -> Callable[..., Any]:
    try:
        from qiskit_ibm_runtime import SamplerV2

        return cast(Callable[..., Any], SamplerV2)
    except Exception as exc:
        raise RuntimeError("qiskit-ibm-runtime is required for QiskitRuntimeHALAdapter") from exc


def _normalise_counts(counts: dict[Any, Any]) -> dict[str, int]:
    normalised: dict[str, int] = {}
    for key, value in counts.items():
        bitstring = strict_binary_bitstring_key(key, field_name="Qiskit count key")
        count = strict_non_negative_count(value)
        normalised[bitstring] = normalised.get(bitstring, 0) + count
    if not normalised:
        raise ValueError("Qiskit result did not contain any counts")
    return normalised


def _backend_name(backend: Any) -> str:
    name = getattr(backend, "name", None)
    if callable(name):
        return strict_provider_job_id(name(), field_name="Qiskit backend name")
    if name is not None:
        return strict_provider_job_id(name, field_name="Qiskit backend name")
    return strict_provider_job_id(backend.__class__.__name__, field_name="Qiskit backend name")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _provider_job_id(provider_job: Any, *, provider_name: str) -> str:
    job_id_attr = getattr(provider_job, "job_id", None)
    raw_job_id = job_id_attr() if callable(job_id_attr) else job_id_attr
    job_id = str(raw_job_id).strip() if raw_job_id is not None else ""
    if not job_id:
        raise ValueError(f"{provider_name} job object does not expose a provider job id")
    return strict_provider_job_id(job_id, field_name=f"{provider_name} provider job id")


def _normalise_status(value: object, *, default: str = "unknown") -> str:
    text = str(value or default).split(".")[-1].strip().lower().replace(" ", "_")
    return {
        "complete": "completed",
        "completed": "completed",
        "success": "completed",
        "succeeded": "completed",
        "done": "completed",
        "finished": "completed",
        "running": "running",
        "in_progress": "running",
        "in-progress": "running",
        "inprogress": "running",
        "submitted": "submitted",
        "queued": "queued",
        "pending": "queued",
        "cancelled": "cancelled",
        "canceled": "cancelled",
        "failed": "failed",
        "error": "failed",
    }.get(text, default)


__all__ = [
    "QiskitAerHALAdapter",
    "QiskitRuntimeHALAdapter",
    "qiskit_circuit_to_qasm3_workload",
    "qiskit_circuit_to_workload",
]
