# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — OQC QCAAS adapter for the hardware HAL
"""Direct OQC QCAAS adapter for the provider-neutral HAL."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from importlib import import_module
from typing import Any

from ._count_integrity import (
    strict_binary_bitstring_key,
    strict_integer_value,
    strict_non_negative_count,
    strict_provider_job_id,
    strict_shot_conservation,
)
from .hal import BackendProfile, QuantumJobRef, QuantumJobResult, QuantumWorkload

OQC_EXECUTION_MODE = "oqc_qcaas_openqasm3"


def oqc_openqasm3_workload(
    program: str,
    *,
    workload_id: str,
    n_qubits: int,
    shots: int,
    metadata: Mapping[str, object] | None = None,
) -> QuantumWorkload:
    """Encode an OpenQASM 3 program as an OQC HAL workload."""

    _validate_openqasm3(program)
    return QuantumWorkload(
        workload_id=workload_id,
        ir_format="openqasm3",
        program=program,
        n_qubits=n_qubits,
        shots=shots,
        metadata=dict(metadata or {}),
    )


class OQCHALAdapter:
    """OQC QCAAS client adapter implementing the provider-neutral HAL protocol."""

    def __init__(
        self,
        profile: BackendProfile,
        *,
        client: Any | None = None,
        client_factory: Callable[[], Any] | None = None,
        target: str = "default",
    ) -> None:
        if profile.backend_id != "oqc_cloud":
            raise ValueError("OQCHALAdapter requires the oqc_cloud profile")
        self.profile = profile
        self.backend_id = profile.backend_id
        self._client = client
        self._client_factory = client_factory
        self._target = strict_provider_job_id(target, field_name="OQC target")
        self._jobs: dict[str, QuantumJobRef] = {}
        self._provider_jobs: dict[str, Any] = {}
        self._results: dict[str, QuantumJobResult] = {}

    def submit(
        self, workload: QuantumWorkload, *, approval_id: str | None = None
    ) -> QuantumJobRef:
        """Submit a workload to the backend and return its job reference."""
        if not approval_id:
            raise PermissionError("approval_id is required for OQC submission")
        if workload.ir_format != "openqasm3":
            raise ValueError("OQC direct adapter requires openqasm3 workloads")
        _validate_openqasm3(workload.program)
        submit = getattr(self._client_for(), "submit", None)
        if not callable(submit):
            raise TypeError("OQC client object does not provide submit()")
        provider_job = submit(
            program=workload.program,
            shots=workload.shots,
            target=self._target,
            name=workload.workload_id,
        )
        provider_job_id = _provider_job_id(provider_job)
        hal_job_id = _hal_job_id(self.backend_id, workload.workload_id, provider_job_id)
        job = QuantumJobRef(
            job_id=hal_job_id,
            backend_id=self.backend_id,
            workload_id=workload.workload_id,
            status="submitted",
            metadata={
                "approval_id": approval_id,
                "provider_job_id": provider_job_id,
                "execution_mode": OQC_EXECUTION_MODE,
                "target": self._target,
                "ir_format": workload.ir_format,
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
        status = getattr(provider_job, "status", None)
        if callable(status):
            status = status()
        return _normalise_status(status)

    def result(self, job: QuantumJobRef) -> QuantumJobResult:
        """Return the completed result for a submitted backend job."""
        cached = self._results.get(job.job_id)
        if cached is not None:
            return cached
        stored = self._job(job)
        provider_job = self._provider_job(job)
        result_method = getattr(provider_job, "result", None)
        if not callable(result_method):
            raise TypeError("OQC provider job does not provide result()")
        counts = _normalise_counts(_extract_counts(result_method()))
        expected_shots = strict_integer_value(stored.metadata.get("shots", 0), field_name="shots")
        observed_shots = strict_shot_conservation(counts, expected_shots=expected_shots)
        result = QuantumJobResult(
            job=stored,
            status="completed",
            counts=counts,
            shots=observed_shots,
            metadata={
                "approval_id": stored.metadata.get("approval_id"),
                "execution_mode": OQC_EXECUTION_MODE,
                "target": self._target,
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
        if callable(cancel):
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

    def _client_for(self) -> Any:
        if self._client is not None:
            return self._client
        if self._client_factory is not None:
            self._client = self._client_factory()
            return self._client
        self._client = _default_client_factory()
        return self._client

    def _job(self, job: QuantumJobRef) -> QuantumJobRef:
        stored = self._jobs.get(job.job_id)
        if stored is None:
            raise KeyError(f"unknown job_id: {job.job_id}")
        return stored

    def _provider_job(self, job: QuantumJobRef) -> Any:
        self._job(job)
        return self._provider_jobs[job.job_id]


def _default_client_factory() -> Any:
    try:
        import_module("qcaas_client")
    except Exception as exc:
        raise RuntimeError(
            "oqc-qcaas-client is required for OQCHALAdapter; inject client_factory"
        ) from exc
    raise RuntimeError(
        "automatic OQC client construction requires a calibrated OQC client; inject client_factory"
    )


def _validate_openqasm3(program: str) -> None:
    if not isinstance(program, str) or not program.strip():
        raise ValueError("OQC OpenQASM 3 program must be non-empty")
    first_line = program.lstrip().splitlines()[0].strip()
    if not first_line.startswith("OPENQASM 3.0"):
        raise ValueError("OQC direct adapter requires an OPENQASM 3.0 program")


def _extract_counts(result: object) -> object:
    if isinstance(result, Mapping):
        for key in ("counts", "results", "histogram"):
            value = result.get(key)
            if isinstance(value, Mapping):
                return value
    counts = getattr(result, "counts", None)
    if isinstance(counts, Mapping):
        return counts
    results = getattr(result, "results", None)
    if isinstance(results, Mapping):
        return results
    raise RuntimeError("Could not extract OQC counts from provider result")


def _normalise_counts(raw: object) -> dict[str, int]:
    if not isinstance(raw, Mapping):
        raise TypeError("OQC counts must be a mapping")
    counts: dict[str, int] = {}
    for bitstring, count in raw.items():
        key = strict_binary_bitstring_key(bitstring, field_name="OQC count key")
        value = strict_non_negative_count(count)
        counts[key] = counts.get(key, 0) + value
    if not counts:
        raise ValueError("OQC result did not contain any counts")
    return counts


def _normalise_status(value: object) -> str:
    text = str(value).split(".")[-1].strip().lower().replace(" ", "_")
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


def _coerce_int(value: object, *, field_name: str) -> int:
    return strict_integer_value(value, field_name=field_name)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _provider_job_id(provider_job: object) -> str:
    for attr in ("id", "job_id", "handle"):
        value = getattr(provider_job, attr, None)
        if callable(value):
            value = value()
        if value is not None and str(value).strip():
            return strict_provider_job_id(value, field_name="OQC provider job id")
    raise ValueError("OQC provider job does not expose a provider job id")


def _hal_job_id(backend_id: str, workload_id: str, provider_job_id: str) -> str:
    digest = hashlib.sha256(provider_job_id.encode("utf-8")).hexdigest()[:12]
    return f"{backend_id}:{workload_id}:{digest}"


__all__ = ["OQC_EXECUTION_MODE", "OQCHALAdapter", "oqc_openqasm3_workload"]
