# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Strangeworks adapter for the hardware HAL
"""Strangeworks Compute adapter for :mod:`scpn_quantum_control.hardware.hal`."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from importlib import import_module
from typing import Any

from ._count_integrity import (
    strict_fixed_width_bitstring_key,
    strict_integer_value,
    strict_non_negative_count,
    strict_provider_job_id,
    strict_shot_conservation,
)
from .hal import BackendProfile, QuantumJobRef, QuantumJobResult, QuantumWorkload


def strangeworks_program_to_workload(
    program: str,
    *,
    workload_id: str,
    ir_format: str,
    n_qubits: int,
    shots: int,
    metadata: Mapping[str, object] | None = None,
) -> QuantumWorkload:
    """Encode a Strangeworks-supported program string as a HAL workload."""

    return QuantumWorkload(
        workload_id=workload_id,
        ir_format=ir_format,
        program=program,
        n_qubits=n_qubits,
        shots=shots,
        metadata=dict(metadata or {}),
    )


class StrangeworksComputeHALAdapter:
    """Strangeworks Compute adapter implementing the provider-neutral HAL protocol."""

    def __init__(
        self,
        profile: BackendProfile,
        *,
        backend: Any | None = None,
        workspace: Any | None = None,
        backend_id: str | None = None,
        workspace_factory: Callable[[], Any] | None = None,
        program_factory: Callable[[QuantumWorkload], Any] | None = None,
        submit_kwargs: Mapping[str, object] | None = None,
    ) -> None:
        if profile.broker != "strangeworks":
            raise ValueError("StrangeworksComputeHALAdapter requires a Strangeworks profile")
        if backend is None and workspace is None and workspace_factory is None:
            raise ValueError(
                "backend, workspace, or workspace_factory is required for Strangeworks submission"
            )
        if backend is None and backend_id is None:
            raise ValueError("backend_id is required when no Strangeworks backend is injected")
        self.profile = profile
        self.backend_id = profile.backend_id
        self._backend = backend
        self._workspace = workspace
        self._backend_route_id = (
            strict_provider_job_id(backend_id, field_name="Strangeworks backend id")
            if backend_id is not None
            else None
        )
        self._workspace_factory = workspace_factory
        self._program_factory = program_factory
        self._submit_kwargs = dict(submit_kwargs or {})
        self._provider_jobs: dict[str, Any] = {}
        self._jobs: dict[str, QuantumJobRef] = {}
        self._results: dict[str, QuantumJobResult] = {}

    def submit(
        self, workload: QuantumWorkload, *, approval_id: str | None = None
    ) -> QuantumJobRef:
        """Submit a workload to the backend and return its job reference."""
        if not approval_id:
            raise PermissionError("approval_id is required for Strangeworks submission")
        backend = self._backend or self._load_backend()
        run_input = self._program_factory(workload) if self._program_factory else workload.program
        provider_job = backend.run(
            run_input,
            shots=workload.shots,
            name=workload.workload_id,
            metadata={
                "approval_id": approval_id,
                "broker": "strangeworks",
                "workload_id": workload.workload_id,
                "ir_format": workload.ir_format,
                **dict(workload.metadata),
            },
            **self._submit_kwargs,
        )
        provider_job_id = _job_id(provider_job)
        job = QuantumJobRef(
            job_id=provider_job_id,
            backend_id=self.backend_id,
            workload_id=workload.workload_id,
            status="submitted",
            metadata={
                "approval_id": approval_id,
                "provider_job_id": provider_job_id,
                "execution_mode": "strangeworks_compute",
                "ir_format": workload.ir_format,
                "backend_id": _backend_id(backend),
                "n_qubits": workload.n_qubits,
                "shots": workload.shots,
            },
        )
        self._provider_jobs[job.job_id] = provider_job
        self._jobs[job.job_id] = job
        return job

    def status(self, job: QuantumJobRef) -> str:
        """Return the current status for a submitted backend job."""
        provider_job = self._provider_job(job)
        status_method = getattr(provider_job, "status", None)
        status = (
            status_method()
            if callable(status_method)
            else getattr(provider_job, "status", "unknown")
        )
        return _normalise_status(getattr(status, "name", status))

    def result(self, job: QuantumJobRef) -> QuantumJobResult:
        """Return the completed result for a submitted backend job."""
        cached = self._results.get(job.job_id)
        if cached is not None:
            return cached
        provider_job = self._provider_job(job)
        result_method = getattr(provider_job, "result", None)
        if not callable(result_method):
            raise TypeError("Strangeworks provider job does not provide result()")
        provider_result = result_method()
        n_qubits = strict_integer_value(job.metadata.get("n_qubits", 0), field_name="n_qubits")
        counts = _extract_counts(provider_result, n_qubits=n_qubits)
        expected_shots = strict_integer_value(job.metadata.get("shots", 0), field_name="shots")
        observed_shots = strict_shot_conservation(counts, expected_shots=expected_shots)
        result = QuantumJobResult(
            job=job,
            status="completed",
            counts=counts,
            shots=observed_shots,
            metadata={
                "approval_id": job.metadata.get("approval_id"),
                "execution_mode": "strangeworks_compute",
                "ir_format": job.metadata.get("ir_format"),
                "backend_id": job.metadata.get("backend_id"),
                "timestamp": _utc_now(),
            },
        )
        self._results[job.job_id] = result
        return result

    def cancel(self, job: QuantumJobRef) -> QuantumJobRef:
        """Request cancellation for a submitted backend job."""
        provider_job = self._provider_job(job)
        cancel = getattr(provider_job, "cancel", None)
        if callable(cancel):
            cancel()
        cancelled = QuantumJobRef(
            job_id=job.job_id,
            backend_id=job.backend_id,
            workload_id=job.workload_id,
            status="cancelled",
            metadata=job.metadata,
        )
        self._jobs[job.job_id] = cancelled
        return cancelled

    def _load_backend(self) -> Any:
        if self._backend_route_id is None:
            raise ValueError("backend_id is required when no Strangeworks backend is injected")
        workspace = self._workspace
        if workspace is None:
            workspace = (
                self._workspace_factory()
                if self._workspace_factory
                else _default_strangeworks_workspace()
            )
            self._workspace = workspace
        for method_name in ("get_backend", "backend", "get_resource", "resource"):
            method = getattr(workspace, method_name, None)
            if callable(method):
                return method(self._backend_route_id)
        raise TypeError("Strangeworks workspace does not expose a backend/resource lookup method")

    def _provider_job(self, job: QuantumJobRef) -> Any:
        provider_job = self._provider_jobs.get(job.job_id)
        if provider_job is None:
            raise KeyError(f"unknown job_id: {job.job_id}")
        return provider_job


def _default_strangeworks_workspace() -> Any:
    try:
        strangeworks = import_module("strangeworks")

        workspace = getattr(strangeworks, "workspace", None)
        if callable(workspace):
            return workspace()
        default_workspace = getattr(strangeworks, "default_workspace", None)
        if callable(default_workspace):
            return default_workspace()
    except Exception as exc:
        raise RuntimeError("strangeworks is required for StrangeworksComputeHALAdapter") from exc
    raise RuntimeError(
        "automatic Strangeworks workspace construction requires an authenticated SDK; "
        "inject workspace or workspace_factory"
    )


def _extract_counts(provider_result: Any, *, n_qubits: int) -> dict[str, int]:
    if n_qubits <= 0:
        raise ValueError("Strangeworks result decoding requires a positive n_qubits")
    data = getattr(provider_result, "data", None)
    if data is not None:
        for attr in ("get_counts", "measurement_counts"):
            value = getattr(data, attr, None)
            if callable(value):
                return _normalise_counts(value(), n_qubits=n_qubits)
            if value is not None:
                return _normalise_counts(value, n_qubits=n_qubits)
    for attr in ("get_counts", "measurement_counts", "counts"):
        value = getattr(provider_result, attr, None)
        if callable(value):
            return _normalise_counts(value(), n_qubits=n_qubits)
        if value is not None:
            return _normalise_counts(value, n_qubits=n_qubits)
    if isinstance(provider_result, Mapping):
        for key in ("counts", "measurement_counts"):
            if key in provider_result:
                return _normalise_counts(provider_result[key], n_qubits=n_qubits)
    raise ValueError("Strangeworks result does not contain measurement counts")


def _normalise_counts(counts: Any, *, n_qubits: int) -> dict[str, int]:
    if isinstance(counts, list):
        if len(counts) != 1:
            raise ValueError(
                "Strangeworks batch results must be split before HAL count extraction"
            )
        counts = counts[0]
    if not isinstance(counts, Mapping):
        raise ValueError("Strangeworks measurement counts must be a mapping")
    normalised: dict[str, int] = {}
    for bitstring, count in counts.items():
        key = strict_fixed_width_bitstring_key(
            bitstring, width=n_qubits, field_name="Strangeworks count key"
        )
        value = strict_non_negative_count(count)
        normalised[key] = normalised.get(key, 0) + value
    if not normalised:
        raise ValueError("Strangeworks result contains an empty count map")
    return normalised


def _job_id(provider_job: Any) -> str:
    for attr in ("id", "job_id"):
        value = getattr(provider_job, attr, None)
        if callable(value):
            value = value()
        if value:
            return strict_provider_job_id(value, field_name="Strangeworks provider job id")
    raise ValueError("Strangeworks backend.run() returned a job object without an id")


def _backend_id(backend: Any) -> str:
    for attr in ("id", "backend_id", "name"):
        value = getattr(backend, attr, None)
        if callable(value):
            value = value()
        if value:
            return strict_provider_job_id(value, field_name="Strangeworks backend id")
    return strict_provider_job_id(backend.__class__.__name__, field_name="Strangeworks backend id")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalise_status(value: object, *, default: str = "unknown") -> str:
    text = str(value or default).split(".")[-1].strip().lower().replace(" ", "_")
    return {
        "complete": "completed",
        "completed": "completed",
        "success": "completed",
        "succeeded": "completed",
        "finished": "completed",
        "running": "running",
        "in_progress": "running",
        "submitted": "submitted",
        "queued": "queued",
        "pending": "queued",
        "cancelled": "cancelled",
        "canceled": "cancelled",
        "failed": "failed",
        "error": "failed",
    }.get(text, default)


__all__ = [
    "StrangeworksComputeHALAdapter",
    "strangeworks_program_to_workload",
]
