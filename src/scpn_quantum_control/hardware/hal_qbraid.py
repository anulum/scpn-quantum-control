# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- qBraid adapter for the hardware HAL
"""qBraid runtime adapter for :mod:`scpn_quantum_control.hardware.hal`."""

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


def qbraid_program_to_workload(
    program: str,
    *,
    workload_id: str,
    ir_format: str,
    n_qubits: int,
    shots: int,
    metadata: Mapping[str, object] | None = None,
) -> QuantumWorkload:
    """Encode a qBraid-supported program string as a HAL workload."""

    return QuantumWorkload(
        workload_id=workload_id,
        ir_format=ir_format,
        program=program,
        n_qubits=n_qubits,
        shots=shots,
        metadata=dict(metadata or {}),
    )


class QbraidRuntimeHALAdapter:
    """qBraid cloud adapter implementing the provider-neutral HAL protocol."""

    def __init__(
        self,
        profile: BackendProfile,
        *,
        device: Any | None = None,
        provider: Any | None = None,
        device_id: str | None = None,
        provider_factory: Callable[[], Any] | None = None,
        program_factory: Callable[[QuantumWorkload], Any] | None = None,
        submit_kwargs: Mapping[str, object] | None = None,
    ) -> None:
        if profile.broker != "qbraid":
            raise ValueError("QbraidRuntimeHALAdapter requires a qBraid profile")
        if device is None and provider is None and provider_factory is None:
            raise ValueError(
                "device, provider, or provider_factory is required for qBraid submission"
            )
        if device is None and device_id is None:
            raise ValueError("device_id is required when no qBraid device is injected")
        self.profile = profile
        self.backend_id = profile.backend_id
        self._device = device
        self._provider = provider
        self._device_id = (
            strict_provider_job_id(device_id, field_name="qBraid device id")
            if device_id is not None
            else None
        )
        self._provider_factory = provider_factory
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
            raise PermissionError("approval_id is required for qBraid submission")
        device = self._device or self._load_device()
        run_input = self._program_factory(workload) if self._program_factory else workload.program
        provider_job = device.run(
            run_input,
            shots=workload.shots,
            name=workload.workload_id,
            metadata={
                "approval_id": approval_id,
                "workload_id": workload.workload_id,
                "ir_format": workload.ir_format,
                **dict(workload.metadata),
            },
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
                "execution_mode": "qbraid_runtime",
                "ir_format": workload.ir_format,
                "device_id": _device_id(device),
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
        provider_result = provider_job.result()
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
                "execution_mode": "qbraid_runtime",
                "ir_format": job.metadata.get("ir_format"),
                "device_id": job.metadata.get("device_id"),
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

    def _load_device(self) -> Any:
        if self._device_id is None:
            raise ValueError("device_id is required when no qBraid device is injected")
        provider = self._provider
        if provider is None:
            provider = (
                self._provider_factory() if self._provider_factory else _default_qbraid_provider()
            )
            self._provider = provider
        return provider.get_device(self._device_id)

    def _provider_job(self, job: QuantumJobRef) -> Any:
        provider_job = self._provider_jobs.get(job.job_id)
        if provider_job is None:
            raise KeyError(f"unknown job_id: {job.job_id}")
        return provider_job


def _default_qbraid_provider() -> Any:
    try:
        from qbraid.runtime import QbraidProvider

        return QbraidProvider()
    except Exception as exc:
        raise RuntimeError("qbraid is required for QbraidRuntimeHALAdapter") from exc


def _extract_counts(provider_result: Any, *, n_qubits: int) -> dict[str, int]:
    if n_qubits <= 0:
        raise ValueError("qBraid result decoding requires a positive n_qubits")
    data = getattr(provider_result, "data", None)
    if data is not None:
        get_counts = getattr(data, "get_counts", None)
        if callable(get_counts):
            return _normalise_counts(get_counts(), n_qubits=n_qubits)
        measurement_counts = getattr(data, "measurement_counts", None)
        if measurement_counts is not None:
            return _normalise_counts(measurement_counts, n_qubits=n_qubits)
    get_counts = getattr(provider_result, "get_counts", None)
    if callable(get_counts):
        return _normalise_counts(get_counts(), n_qubits=n_qubits)
    measurement_counts_method = getattr(provider_result, "measurement_counts", None)
    if callable(measurement_counts_method):
        return _normalise_counts(measurement_counts_method(), n_qubits=n_qubits)
    if isinstance(provider_result, Mapping):
        for key in ("counts", "measurement_counts"):
            if key in provider_result:
                return _normalise_counts(provider_result[key], n_qubits=n_qubits)
    raise ValueError("qBraid result does not contain measurement counts")


def _normalise_counts(counts: Any, *, n_qubits: int) -> dict[str, int]:
    if isinstance(counts, list):
        if len(counts) != 1:
            raise ValueError("qBraid batch results must be split before HAL count extraction")
        counts = counts[0]
    if not isinstance(counts, Mapping):
        raise ValueError("qBraid measurement counts must be a mapping")
    normalised: dict[str, int] = {}
    for bitstring, count in counts.items():
        key = strict_fixed_width_bitstring_key(
            bitstring, width=n_qubits, field_name="qBraid count key"
        )
        value = strict_non_negative_count(count)
        normalised[key] = normalised.get(key, 0) + value
    if not normalised:
        raise ValueError("qBraid result contains an empty count map")
    return normalised


def _job_id(provider_job: Any) -> str:
    for attr in ("id", "job_id"):
        value = getattr(provider_job, attr, None)
        if callable(value):
            value = value()
        if value:
            return strict_provider_job_id(value, field_name="qBraid provider job id")
    raise ValueError("qBraid device.run() returned a job object without an id")


def _device_id(device: Any) -> str:
    for attr in ("id", "device_id", "name"):
        value = getattr(device, attr, None)
        if callable(value):
            value = value()
        if value:
            return strict_provider_job_id(value, field_name="qBraid device id")
    return strict_provider_job_id(device.__class__.__name__, field_name="qBraid device id")


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
    "QbraidRuntimeHALAdapter",
    "qbraid_program_to_workload",
]
