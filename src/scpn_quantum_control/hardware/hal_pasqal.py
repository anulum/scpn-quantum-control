# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Pasqal Pulser adapter for the hardware HAL
"""Pasqal/Pulser adapter for :mod:`scpn_quantum_control.hardware.hal`."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
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

PASQAL_PULSER_SCHEMA = "pulser_sequence_plan_v1"
PASQAL_PULSER_EXECUTION_MODE = "pasqal_pulser"


def pulser_sequence_workload(
    payload: Mapping[str, object] | str,
    *,
    workload_id: str,
    n_qubits: int,
    shots: int,
    metadata: Mapping[str, object] | None = None,
) -> QuantumWorkload:
    """Encode a Pulser sequence plan as a HAL workload for Pasqal."""

    decoded = (
        _json_mapping(payload, field_name="Pulser payload")
        if isinstance(payload, str)
        else dict(payload)
    )
    _validate_pulser_payload(decoded, n_qubits)
    return QuantumWorkload(
        workload_id=workload_id,
        ir_format="pulser",
        program=json.dumps(decoded, sort_keys=True, separators=(",", ":")),
        n_qubits=n_qubits,
        shots=shots,
        metadata=dict(metadata or {}),
    )


class PasqalPulserHALAdapter:
    """Pasqal/Pulser client adapter implementing the provider-neutral HAL protocol."""

    def __init__(
        self,
        profile: BackendProfile,
        *,
        client: Any | None = None,
        client_factory: Callable[[dict[str, object]], Any] | None = None,
        target: str | None = None,
    ) -> None:
        if profile.backend_id != "pasqal_cloud":
            raise ValueError("PasqalPulserHALAdapter requires the pasqal_cloud profile")
        self.profile = profile
        self.backend_id = profile.backend_id
        self._client = client
        self._client_factory = client_factory
        self._target = strict_provider_job_id(target or "injected", field_name="Pasqal target")
        self._jobs: dict[str, QuantumJobRef] = {}
        self._provider_jobs: dict[str, Any] = {}
        self._results: dict[str, QuantumJobResult] = {}

    def submit(
        self, workload: QuantumWorkload, *, approval_id: str | None = None
    ) -> QuantumJobRef:
        """Submit a workload to the backend and return its job reference."""
        if not approval_id:
            raise PermissionError("approval_id is required for Pasqal submission")
        if workload.ir_format != "pulser":
            raise ValueError("Pasqal direct adapter requires pulser workloads")
        sequence = _decode_payload(workload)
        _validate_pulser_payload(sequence, workload.n_qubits)
        client = self._client_for(sequence)
        provider_job = client.submit(
            sequence=sequence,
            shots=workload.shots,
            job_name=workload.workload_id,
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
                "execution_mode": PASQAL_PULSER_EXECUTION_MODE,
                "target": self._target,
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
            raise TypeError("Pasqal provider job does not provide result()")
        counts = _normalise_counts(_extract_counts(result_method()))
        expected_shots = strict_integer_value(
            stored.metadata.get("shots"),
            field_name="Pasqal expected shots",
        )
        observed_shots = strict_shot_conservation(counts, expected_shots=expected_shots)
        result = QuantumJobResult(
            job=stored,
            status="completed",
            counts=counts,
            shots=observed_shots,
            metadata={
                "approval_id": stored.metadata.get("approval_id"),
                "provider_job_id": stored.metadata.get("provider_job_id"),
                "execution_mode": PASQAL_PULSER_EXECUTION_MODE,
                "target": stored.metadata.get("target"),
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
            raise ValueError("Pasqal provider job does not support cancellation")
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

    def _client_for(self, sequence: dict[str, object]) -> Any:
        if self._client is not None:
            return self._client
        if self._client_factory is not None:
            self._client = self._client_factory(sequence)
            return self._client
        self._client = _default_client_factory(sequence)
        return self._client

    def _job(self, job: QuantumJobRef) -> QuantumJobRef:
        stored = self._jobs.get(job.job_id)
        if stored is None:
            raise KeyError(f"unknown job_id: {job.job_id}")
        return stored

    def _provider_job(self, job: QuantumJobRef) -> Any:
        self._job(job)
        return self._provider_jobs[job.job_id]


def _default_client_factory(sequence: dict[str, object]) -> Any:
    del sequence
    try:
        import_module("pulser")
    except Exception as exc:
        raise RuntimeError(
            "pulser and a Pasqal cloud client are required for PasqalPulserHALAdapter; "
            "inject client_factory for this workload"
        ) from exc
    raise RuntimeError(
        "automatic Pasqal client construction requires a calibrated Pasqal client; "
        "inject client_factory for this workload"
    )


def _decode_payload(workload: QuantumWorkload) -> dict[str, object]:
    return _json_mapping(workload.program, field_name="Pulser workload")


def _json_mapping(source: Mapping[str, object] | str, *, field_name: str) -> dict[str, object]:
    if isinstance(source, Mapping):
        return dict(source)
    try:
        payload = json.loads(source)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field_name} is not valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"{field_name} must be a JSON object")
    return dict(payload)


def _validate_pulser_payload(payload: Mapping[str, object], n_qubits: int) -> None:
    if payload.get("schema") != PASQAL_PULSER_SCHEMA:
        raise ValueError(f"unsupported Pulser schema; expected {PASQAL_PULSER_SCHEMA}")
    duration = _coerce_float(payload.get("duration"), field_name="Pulser duration")
    if duration <= 0.0:
        raise ValueError("Pulser duration must be positive")
    register = payload.get("register")
    if not isinstance(register, Mapping) or len(register) != n_qubits:
        raise ValueError("Pulser register must map every qubit site to coordinates")
    for site, coordinates in register.items():
        _coerce_int(site, field_name="Pulser register site")
        if (
            not isinstance(coordinates, Sequence)
            or isinstance(coordinates, str)
            or len(coordinates) != 2
        ):
            raise ValueError("Pulser register coordinates must contain x and y")
        [_coerce_float(value, field_name="Pulser register coordinate") for value in coordinates]
    channel = payload.get("rydberg_channel")
    if not isinstance(channel, str) or not channel:
        raise ValueError("Pulser rydberg_channel must be non-empty")
    _validate_rabi_envelope(payload.get("rabi_envelope"))
    _validate_site_terms(payload.get("local_detunings"), field_name="local_detunings")
    _validate_edge_terms(payload.get("interaction_terms"), field_name="interaction_terms")
    _validate_edge_terms(payload.get("fim_feedback_terms", []), field_name="fim_feedback_terms")


def _validate_rabi_envelope(value: object) -> None:
    if not isinstance(value, Sequence) or isinstance(value, str) or not value:
        raise ValueError("rabi_envelope must be a non-empty sequence")
    last_time = -float("inf")
    for point in value:
        if not isinstance(point, Mapping):
            raise ValueError("rabi_envelope entries must be mappings")
        time = _coerce_float(point.get("time"), field_name="rabi_envelope time")
        _coerce_float(point.get("amplitude"), field_name="rabi_envelope amplitude")
        _coerce_float(point.get("phase"), field_name="rabi_envelope phase")
        if time < last_time:
            raise ValueError("rabi_envelope times must be monotonic")
        last_time = time


def _validate_site_terms(value: object, *, field_name: str) -> None:
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise ValueError(f"{field_name} must be a sequence")
    for term in value:
        if not isinstance(term, Mapping):
            raise ValueError(f"{field_name} entries must be mappings")
        _coerce_int(term.get("site"), field_name=f"{field_name} site")
        _coerce_float(term.get("detuning"), field_name=f"{field_name} detuning")


def _validate_edge_terms(value: object, *, field_name: str) -> None:
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise ValueError(f"{field_name} must be a sequence")
    for term in value:
        if not isinstance(term, Mapping):
            raise ValueError(f"{field_name} entries must be mappings")
        _coerce_int(term.get("source"), field_name=f"{field_name} source")
        _coerce_int(term.get("target"), field_name=f"{field_name} target")
        _coerce_float(term.get("coefficient"), field_name=f"{field_name} coefficient")


def _extract_counts(result: object) -> object:
    if isinstance(result, Mapping):
        for key in ("counter", "counts", "samples"):
            value = result.get(key)
            if isinstance(value, Mapping):
                return value
    counts = getattr(result, "counts", None)
    if isinstance(counts, Mapping):
        return counts
    counter = getattr(result, "counter", None)
    if isinstance(counter, Mapping):
        return counter
    raise RuntimeError("Could not extract Pasqal counts from provider result")


def _normalise_counts(raw: object) -> dict[str, int]:
    if not isinstance(raw, Mapping):
        raise TypeError("Pasqal counts must be a mapping")
    counts: dict[str, int] = {}
    for bitstring, count in raw.items():
        key = strict_binary_bitstring_key(bitstring, field_name="Pasqal count key")
        value = strict_non_negative_count(count)
        counts[key] = counts.get(key, 0) + value
    if not counts:
        raise ValueError("Pasqal result did not contain any counts")
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
        "cancelled": "cancelled",
        "canceled": "cancelled",
        "failed": "failed",
        "error": "failed",
    }.get(text, "unknown")


def _coerce_int(value: object, *, field_name: str) -> int:
    return strict_integer_value(value, field_name=field_name)


def _coerce_float(value: object, *, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be numeric")
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric") from exc


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _provider_job_id(provider_job: object) -> str:
    for attr in ("id", "job_id", "handle"):
        value = getattr(provider_job, attr, None)
        if callable(value):
            value = value()
        if value is not None and str(value).strip():
            return strict_provider_job_id(value, field_name="Pasqal provider job id")
    raise ValueError("Pasqal provider job does not expose a provider job id")


def _hal_job_id(backend_id: str, workload_id: str, provider_job_id: str) -> str:
    digest = hashlib.sha256(provider_job_id.encode("utf-8")).hexdigest()[:12]
    return f"{backend_id}:{workload_id}:{digest}"


__all__ = [
    "PASQAL_PULSER_EXECUTION_MODE",
    "PASQAL_PULSER_SCHEMA",
    "PasqalPulserHALAdapter",
    "pulser_sequence_workload",
]
