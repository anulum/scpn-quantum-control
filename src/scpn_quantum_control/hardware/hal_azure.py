# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Azure Quantum adapter for the hardware HAL
"""Azure Quantum adapter for :mod:`scpn_quantum_control.hardware.hal`."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from typing import Any

from ._count_integrity import (
    strict_fixed_width_bitstring_key,
    strict_integer_value,
    strict_non_negative_count,
    strict_provider_job_id,
    strict_shot_conservation,
)
from .hal import BackendProfile, QuantumJobRef, QuantumJobResult, QuantumWorkload

InputParamsFactory = Callable[[QuantumWorkload], Mapping[str, object] | None]
TargetFactory = Callable[[Any, str], Any]


def azure_openqasm3_to_workload(
    program: str,
    *,
    workload_id: str,
    n_qubits: int,
    shots: int,
    metadata: Mapping[str, object] | None = None,
) -> QuantumWorkload:
    """Build an Azure Quantum OpenQASM 3 HAL workload."""

    return QuantumWorkload(
        workload_id=workload_id,
        ir_format="openqasm3",
        program=program,
        n_qubits=n_qubits,
        shots=shots,
        metadata=dict(metadata or {}),
    )


class AzureQuantumHALAdapter:
    """Azure Quantum target adapter implementing the HAL protocol."""

    def __init__(
        self,
        profile: BackendProfile,
        *,
        target: Any | None = None,
        workspace: Any | None = None,
        target_name: str | None = None,
        target_factory: TargetFactory | None = None,
        input_params_factory: InputParamsFactory | None = None,
        submit_kwargs: Mapping[str, object] | None = None,
    ) -> None:
        if not profile.backend_id.startswith("azure_quantum_"):
            raise ValueError("AzureQuantumHALAdapter requires an azure_quantum profile")
        if target is None and (workspace is None or target_name is None or target_factory is None):
            raise ValueError("target or workspace+target_name+target_factory is required")
        self.profile = profile
        self.backend_id = profile.backend_id
        self._target = target
        self._workspace = workspace
        self._target_name = (
            strict_provider_job_id(target_name, field_name="Azure target name")
            if target_name is not None
            else None
        )
        self._target_factory = target_factory
        self._input_params_factory = input_params_factory
        self._submit_kwargs = dict(submit_kwargs or {})
        self._jobs: dict[str, QuantumJobRef] = {}
        self._provider_jobs: dict[str, Any] = {}
        self._results: dict[str, QuantumJobResult] = {}

    def submit(
        self, workload: QuantumWorkload, *, approval_id: str | None = None
    ) -> QuantumJobRef:
        """Submit a workload to the backend and return its job reference."""
        if not approval_id:
            raise PermissionError("approval_id is required for Azure Quantum submission")
        if workload.ir_format != "openqasm3":
            raise ValueError("Azure Quantum adapter requires OpenQASM 3 workloads")
        target = self._target or self._load_target()
        input_params = (
            dict(self._input_params_factory(workload) or {})
            if self._input_params_factory is not None
            else None
        )
        provider_job = target.submit(
            workload.program,
            name=workload.workload_id,
            shots=workload.shots,
            input_params=input_params,
            **self._submit_kwargs,
        )
        job_id = _job_id(provider_job)
        job = QuantumJobRef(
            job_id=job_id,
            backend_id=self.backend_id,
            workload_id=workload.workload_id,
            status="submitted",
            metadata={
                "approval_id": approval_id,
                "provider_job_id": job_id,
                "execution_mode": "azure_quantum",
                "ir_format": workload.ir_format,
                "target_name": _target_name(target),
                "n_qubits": workload.n_qubits,
                "shots": workload.shots,
            },
        )
        self._jobs[job.job_id] = job
        self._provider_jobs[job.job_id] = provider_job
        return job

    def status(self, job: QuantumJobRef) -> str:
        """Return the current status for a submitted backend job."""
        provider_job = self._provider_job(job)
        refresh = getattr(provider_job, "refresh", None)
        if callable(refresh):
            refresh()
        details = getattr(provider_job, "details", None)
        detail_status = getattr(details, "status", None)
        if detail_status is not None:
            return _normalise_status(detail_status)
        status = getattr(provider_job, "status", None)
        if callable(status):
            return _normalise_status(status())
        if status is not None:
            return _normalise_status(status)
        return self._jobs[job.job_id].status

    def result(self, job: QuantumJobRef) -> QuantumJobResult:
        """Return the completed result for a submitted backend job."""
        cached = self._results.get(job.job_id)
        if cached is not None:
            return cached
        provider_job = self._provider_job(job)
        payload = _job_results(provider_job)
        n_qubits = strict_integer_value(job.metadata.get("n_qubits", 0), field_name="n_qubits")
        counts = _extract_counts(payload, n_qubits=n_qubits)
        expected_shots = strict_integer_value(job.metadata.get("shots", 0), field_name="shots")
        observed_shots = strict_shot_conservation(counts, expected_shots=expected_shots)
        result = QuantumJobResult(
            job=job,
            status="completed",
            counts=counts,
            shots=observed_shots,
            metadata={
                "approval_id": job.metadata.get("approval_id"),
                "execution_mode": "azure_quantum",
                "ir_format": job.metadata.get("ir_format"),
                "target_name": job.metadata.get("target_name"),
                "timestamp": _utc_now(),
            },
        )
        self._results[job.job_id] = result
        return result

    def cancel(self, job: QuantumJobRef) -> QuantumJobRef:
        """Request cancellation for a submitted backend job."""
        provider_job = self._provider_job(job)
        cancel = getattr(provider_job, "cancel", None)
        if not callable(cancel):
            raise ValueError("Azure Quantum job object does not expose cancel()")
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

    def _load_target(self) -> Any:
        if self._workspace is None or self._target_name is None or self._target_factory is None:
            raise ValueError("workspace, target_name, and target_factory are required")
        return self._target_factory(self._workspace, self._target_name)

    def _provider_job(self, job: QuantumJobRef) -> Any:
        provider_job = self._provider_jobs.get(job.job_id)
        if provider_job is None:
            raise KeyError(f"unknown job_id: {job.job_id}")
        return provider_job


def _job_id(provider_job: Any) -> str:
    for attr in ("id", "job_id", "name"):
        value = getattr(provider_job, attr, None)
        if callable(value):
            value = value()
        if value:
            return strict_provider_job_id(value, field_name="Azure provider job id")
    raise ValueError("Azure Quantum submit() returned a job object without an id")


def _job_results(provider_job: Any) -> Any:
    for name in ("get_results", "result", "results"):
        method = getattr(provider_job, name, None)
        if callable(method):
            return method()
    raise ValueError("Azure Quantum job object does not expose results")


def _extract_counts(payload: Any, *, n_qubits: int) -> dict[str, int]:
    if n_qubits <= 0:
        raise ValueError("Azure Quantum result decoding requires a positive n_qubits")
    if isinstance(payload, Mapping):
        for key in ("counts", "histogram", "measurement_counts", "MeasurementCounts"):
            candidate = payload.get(key)
            if isinstance(candidate, Mapping):
                counts: dict[str, int] = {}
                for bitstring, count in candidate.items():
                    normalised_key = strict_fixed_width_bitstring_key(
                        bitstring, width=n_qubits, field_name="Azure count key"
                    )
                    value = strict_non_negative_count(count)
                    counts[normalised_key] = counts.get(normalised_key, 0) + value
                if not counts:
                    raise ValueError("Azure Quantum result payload contains an empty count map")
                return counts
        if all(isinstance(key, str) for key in payload):
            payload_counts: dict[str, int] = {}
            for bitstring, count in payload.items():
                normalised_key = strict_fixed_width_bitstring_key(
                    bitstring, width=n_qubits, field_name="Azure count key"
                )
                value = strict_non_negative_count(count)
                payload_counts[normalised_key] = payload_counts.get(normalised_key, 0) + value
            if not payload_counts:
                raise ValueError("Azure Quantum result payload contains an empty count map")
            return payload_counts
    raise ValueError("Azure Quantum result payload does not contain shot counts")


def _target_name(target: Any) -> str:
    name = getattr(target, "name", None)
    if callable(name):
        return strict_provider_job_id(name(), field_name="Azure target name")
    if name is not None:
        return strict_provider_job_id(name, field_name="Azure target name")
    return strict_provider_job_id(target.__class__.__name__, field_name="Azure target name")


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
        "in-progress": "running",
        "submitted": "submitted",
        "queued": "queued",
        "pending": "queued",
        "cancelled": "cancelled",
        "canceled": "cancelled",
        "failed": "failed",
        "error": "failed",
    }.get(text, default)


__all__ = [
    "AzureQuantumHALAdapter",
    "azure_openqasm3_to_workload",
]
