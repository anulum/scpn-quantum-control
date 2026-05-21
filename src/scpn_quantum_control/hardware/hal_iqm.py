# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- IQM adapter for the hardware HAL
"""IQM Qiskit adapter for :mod:`scpn_quantum_control.hardware.hal`."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from datetime import datetime, timezone
from importlib import import_module
from typing import Any

from qiskit import QuantumCircuit, transpile

from ._count_integrity import (
    strict_binary_bitstring_key,
    strict_integer_value,
    strict_non_negative_count,
    strict_provider_job_id,
    strict_shot_conservation,
)
from .hal import BackendProfile, QuantumJobRef, QuantumJobResult, QuantumWorkload
from .hal_qiskit import qiskit_circuit_to_workload

IQM_EXECUTION_MODE = "iqm_qiskit"


def iqm_qiskit_workload(
    circuit: QuantumCircuit,
    *,
    workload_id: str,
    shots: int,
    metadata: dict[str, object] | None = None,
) -> QuantumWorkload:
    """Encode a Qiskit circuit as a HAL workload for IQM execution."""

    return qiskit_circuit_to_workload(
        circuit,
        workload_id=workload_id,
        shots=shots,
        metadata=metadata,
    )


class IQMHALAdapter:
    """IQM Qiskit adapter implementing the provider-neutral HAL protocol."""

    def __init__(
        self,
        profile: BackendProfile,
        *,
        backend: Any | None = None,
        server_url: str | None = None,
        quantum_computer: str | None = None,
        import_module: Callable[[str], Any] = import_module,
        timeout_s: float = 600.0,
        optimisation_level: int = 1,
        compile_circuit: bool = True,
    ) -> None:
        if profile.backend_id != "iqm_cloud":
            raise ValueError("IQMHALAdapter requires the iqm_cloud profile")
        if timeout_s <= 0.0:
            raise ValueError("timeout_s must be positive")
        if optimisation_level not in {0, 1, 2, 3}:
            raise ValueError("optimisation_level must be 0, 1, 2, or 3")
        self.profile = profile
        self.backend_id = profile.backend_id
        self._backend = backend
        self._server_url = server_url
        self._quantum_computer = quantum_computer
        self._import_module = import_module
        self.timeout_s = timeout_s
        self.optimisation_level = optimisation_level
        self._compile_circuit = compile_circuit
        self._jobs: dict[str, QuantumJobRef] = {}
        self._provider_jobs: dict[str, Any] = {}
        self._results: dict[str, QuantumJobResult] = {}

    def submit(
        self, workload: QuantumWorkload, *, approval_id: str | None = None
    ) -> QuantumJobRef:
        """Submit a workload to the backend and return its job reference."""
        if not approval_id:
            raise PermissionError("approval_id is required for IQM submission")
        if workload.ir_format != "qiskit_qpy":
            raise ValueError("IQM direct adapter requires qiskit_qpy workloads")
        backend = self._backend_client()
        circuit = self._compile(_workload_to_qiskit_circuit(workload), backend)
        provider_job = backend.run([circuit], shots=workload.shots)
        provider_job_id = _job_id(provider_job)
        backend_name = _backend_name(backend)
        job = QuantumJobRef(
            job_id=_hal_job_id(self.backend_id, workload.workload_id, provider_job_id),
            backend_id=self.backend_id,
            workload_id=workload.workload_id,
            status="submitted",
            metadata={
                "approval_id": approval_id,
                "provider_job_id": provider_job_id,
                "execution_mode": IQM_EXECUTION_MODE,
                "backend_name": backend_name,
                "quantum_computer": self._quantum_computer,
                "ir_format": workload.ir_format,
                "n_qubits": workload.n_qubits,
                "shots": workload.shots,
                **dict(workload.metadata),
            },
        )
        self._jobs[job.job_id] = job
        self._provider_jobs[job.job_id] = provider_job
        return job

    def status(self, job: QuantumJobRef) -> str:
        """Return the current status for a submitted backend job."""
        provider_job = self._provider_job(job)
        status = getattr(provider_job, "status", None)
        if callable(status):
            return _normalise_status(status())
        return _normalise_status(getattr(provider_job, "_status", "unknown"))

    def result(self, job: QuantumJobRef) -> QuantumJobResult:
        """Return the completed result for a submitted backend job."""
        cached = self._results.get(job.job_id)
        if cached is not None:
            return cached
        stored = self._job(job)
        provider_job = self._provider_job(job)
        result_method = getattr(provider_job, "result", None)
        if not callable(result_method):
            raise TypeError("IQM provider job does not provide result()")
        provider_result = result_method(timeout=self.timeout_s)
        counts = _extract_counts(provider_result)
        expected_shots = strict_integer_value(stored.metadata.get("shots", 0), field_name="shots")
        observed_shots = strict_shot_conservation(counts, expected_shots=expected_shots)
        result = QuantumJobResult(
            job=stored,
            status="completed",
            counts=counts,
            shots=observed_shots,
            metadata={
                "approval_id": stored.metadata.get("approval_id"),
                "provider_job_id": stored.metadata.get("provider_job_id"),
                "execution_mode": IQM_EXECUTION_MODE,
                "backend_name": stored.metadata.get("backend_name"),
                "timestamp": _utc_now(),
            },
        )
        self._results[job.job_id] = result
        return result

    def cancel(self, job: QuantumJobRef) -> QuantumJobRef:
        """Request cancellation for a submitted backend job."""
        stored = self._job(job)
        provider_job = self._provider_job(job)
        cancel = getattr(provider_job, "cancel", None)
        if not callable(cancel):
            raise ValueError("IQM provider job does not support cancellation")
        cancel()
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
        if not self._server_url:
            raise RuntimeError("server_url is required when an IQM backend is not injected")
        try:
            provider_module = self._import_module("iqm.qiskit_iqm.iqm_provider")
        except ModuleNotFoundError as exc:
            raise ImportError(
                "iqm-client[qiskit] is required for IQM HAL execution; install it in "
                "an isolated runner environment such as `.venv-iqm` because current "
                "IQM client releases pin Qiskit below the repository's main Qiskit floor."
            ) from exc
        provider = provider_module.IQMProvider(
            self._server_url,
            quantum_computer=self._quantum_computer,
        )
        get_backend = getattr(provider, "get_backend", None)
        self._backend = get_backend() if callable(get_backend) else provider.backend()
        return self._backend

    def _compile(self, circuit: QuantumCircuit, backend: Any) -> QuantumCircuit:
        if not self._compile_circuit:
            return circuit
        try:
            return transpile(circuit, backend=backend, optimization_level=self.optimisation_level)
        except Exception:
            return transpile(circuit, optimization_level=self.optimisation_level)

    def _job(self, job: QuantumJobRef) -> QuantumJobRef:
        stored = self._jobs.get(job.job_id)
        if stored is None:
            raise KeyError(f"unknown job_id: {job.job_id}")
        return stored

    def _provider_job(self, job: QuantumJobRef) -> Any:
        self._job(job)
        return self._provider_jobs[job.job_id]


def _workload_to_qiskit_circuit(workload: QuantumWorkload) -> QuantumCircuit:
    from .hal_qiskit import _workload_to_qiskit_circuit as decode

    return decode(workload)


def _extract_counts(result: Any) -> dict[str, int]:
    get_counts = getattr(result, "get_counts", None)
    if callable(get_counts):
        try:
            raw = get_counts()
        except TypeError:
            raw = get_counts(0)
        if isinstance(raw, list):
            if len(raw) != 1:
                raise RuntimeError("IQM single-circuit execution returned multiple count maps")
            raw = raw[0]
        return _normalise_counts(raw)
    results = getattr(result, "results", None)
    if isinstance(results, list) and len(results) == 1:
        data = getattr(results[0], "data", None)
        counts = getattr(data, "counts", None)
        if isinstance(counts, dict):
            return _normalise_counts(counts)
    raise RuntimeError("Could not extract IQM counts from backend result")


def _normalise_counts(raw: Any) -> dict[str, int]:
    if not isinstance(raw, dict):
        raise TypeError("IQM counts must be a mapping")
    counts: dict[str, int] = {}
    for bitstring, count in raw.items():
        key = strict_binary_bitstring_key(bitstring, field_name="IQM count key")
        value = strict_non_negative_count(count)
        counts[key] = counts.get(key, 0) + value
    if not counts:
        raise ValueError("IQM result did not contain any counts")
    return counts


def _backend_name(backend: Any) -> str:
    name = getattr(backend, "name", None)
    if callable(name):
        return str(name())
    if name:
        return str(name)
    return type(backend).__name__


def _job_id(job: Any) -> str:
    job_id = getattr(job, "job_id", None)
    if callable(job_id):
        return strict_provider_job_id(job_id(), field_name="IQM provider job id")
    if job_id:
        return strict_provider_job_id(job_id, field_name="IQM provider job id")
    raise ValueError("IQM backend job does not expose a provider job id")


def _hal_job_id(backend_id: str, workload_id: str, provider_job_id: str) -> str:
    digest = hashlib.sha256(provider_job_id.encode("utf-8")).hexdigest()[:12]
    return f"{backend_id}:{workload_id}:{digest}"


def _normalise_status(value: object) -> str:
    text = str(value).split(".")[-1].strip().lower().replace(" ", "_")
    return {
        "done": "completed",
        "complete": "completed",
        "completed": "completed",
        "success": "completed",
        "succeeded": "completed",
        "finished": "completed",
        "queued": "queued",
        "running": "running",
        "cancelled": "cancelled",
        "canceled": "cancelled",
        "failed": "failed",
    }.get(text, "unknown")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "IQMHALAdapter",
    "IQM_EXECUTION_MODE",
    "iqm_qiskit_workload",
]
