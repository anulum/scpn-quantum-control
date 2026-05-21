# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Quandela Perceval adapter for the hardware HAL
"""Direct Quandela/Perceval adapter for the provider-neutral HAL."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from importlib import import_module
from typing import Any

from ._count_integrity import (
    strict_integer_value,
    strict_provider_job_id,
    strict_shot_conservation,
)
from .hal import BackendProfile, QuantumJobRef, QuantumJobResult, QuantumWorkload

QUANDELA_PERCEVAL_SCHEMA = "scpn.quandela.perceval.v1"
QUANDELA_EXECUTION_MODE = "quandela_perceval"


def quandela_perceval_workload(
    payload: Mapping[str, object] | str,
    *,
    workload_id: str,
    n_modes: int,
    shots: int,
    metadata: Mapping[str, object] | None = None,
) -> QuantumWorkload:
    """Encode a Perceval photonic plan as a Quandela HAL workload."""

    decoded = (
        _json_mapping(payload, field_name="Quandela payload")
        if isinstance(payload, str)
        else dict(payload)
    )
    _validate_perceval_payload(decoded, n_modes)
    return QuantumWorkload(
        workload_id=workload_id,
        ir_format="perceval",
        program=json.dumps(decoded, sort_keys=True, separators=(",", ":")),
        n_qubits=n_modes,
        shots=shots,
        metadata=dict(metadata or {}),
    )


class QuandelaPercevalHALAdapter:
    """Quandela/Perceval adapter implementing the provider-neutral HAL protocol."""

    def __init__(
        self,
        profile: BackendProfile,
        *,
        processor: Any | None = None,
        processor_factory: Callable[[dict[str, object]], Any] | None = None,
        sampler_factory: Callable[[Any], Any] | None = None,
        target: str | None = None,
    ) -> None:
        if profile.backend_id != "quandela_cloud":
            raise ValueError("QuandelaPercevalHALAdapter requires the quandela_cloud profile")
        self.profile = profile
        self.backend_id = profile.backend_id
        self._processor = processor
        self._processor_factory = processor_factory
        self._sampler_factory = sampler_factory
        self._target = strict_provider_job_id(target or "injected", field_name="Quandela target")
        self._jobs: dict[str, QuantumJobRef] = {}
        self._results: dict[str, QuantumJobResult] = {}

    def submit(
        self, workload: QuantumWorkload, *, approval_id: str | None = None
    ) -> QuantumJobRef:
        """Submit a workload to the backend and return its job reference."""
        if not approval_id:
            raise PermissionError("approval_id is required for Quandela submission")
        if workload.ir_format != "perceval":
            raise ValueError("Quandela direct adapter requires perceval workloads")
        plan = _decode_payload(workload)
        _validate_perceval_payload(plan, workload.n_qubits)
        processor = self._processor_for(plan)
        raw_result = self._sample(processor, workload.shots)
        counts = _normalise_counts(_extract_counts(raw_result))
        observed_shots = strict_shot_conservation(counts, expected_shots=workload.shots)
        provider_job_id = _provider_job_id(raw_result)
        hal_job_id = _hal_job_id(self.backend_id, workload.workload_id, provider_job_id)
        job = QuantumJobRef(
            job_id=hal_job_id,
            backend_id=self.backend_id,
            workload_id=workload.workload_id,
            status="completed",
            metadata={
                "approval_id": approval_id,
                "provider_job_id": provider_job_id,
                "execution_mode": QUANDELA_EXECUTION_MODE,
                "target": self._target,
                "ir_format": workload.ir_format,
                "n_modes": workload.n_qubits,
                "shots": workload.shots,
            },
        )
        result = QuantumJobResult(
            job=job,
            status="completed",
            counts=counts,
            shots=observed_shots,
            metadata={
                "approval_id": approval_id,
                "execution_mode": QUANDELA_EXECUTION_MODE,
                "target": self._target,
                "timestamp": _utc_now(),
            },
        )
        self._jobs[job.job_id] = job
        self._results[job.job_id] = result
        return job

    def status(self, job: QuantumJobRef) -> str:
        """Return the current status for a submitted backend job."""
        return self._job(job).status

    def result(self, job: QuantumJobRef) -> QuantumJobResult:
        """Return the completed result for a submitted backend job."""
        result = self._results.get(job.job_id)
        if result is None:
            raise KeyError(f"unknown job_id: {job.job_id}")
        return result

    def cancel(self, job: QuantumJobRef) -> QuantumJobRef:
        """Request cancellation for a submitted backend job."""
        stored = self._job(job)
        cancelled = QuantumJobRef(
            job_id=stored.job_id,
            backend_id=stored.backend_id,
            workload_id=stored.workload_id,
            status="cancelled",
            metadata=stored.metadata,
        )
        self._jobs[job.job_id] = cancelled
        return cancelled

    def _processor_for(self, plan: dict[str, object]) -> Any:
        if self._processor is not None:
            return self._processor
        if self._processor_factory is not None:
            self._processor = self._processor_factory(plan)
            return self._processor
        self._processor = _default_processor_factory(plan)
        return self._processor

    def _sample(self, processor: Any, shots: int) -> object:
        if self._sampler_factory is not None:
            sampler = self._sampler_factory(processor)
            samples = getattr(sampler, "samples", None)
            if not callable(samples):
                raise TypeError("Quandela sampler object does not provide samples()")
            return samples(count=shots)
        samples = getattr(processor, "samples", None)
        if callable(samples):
            return samples(shots)
        sample_count = getattr(processor, "sample_count", None)
        if callable(sample_count):
            return sample_count(shots)
        raise TypeError("Quandela processor object does not provide samples()")

    def _job(self, job: QuantumJobRef) -> QuantumJobRef:
        stored = self._jobs.get(job.job_id)
        if stored is None:
            raise KeyError(f"unknown job_id: {job.job_id}")
        return stored


def _default_processor_factory(plan: dict[str, object]) -> Any:
    del plan
    try:
        import_module("perceval")
    except Exception as exc:
        raise RuntimeError(
            "perceval and a calibrated Quandela processor are required; "
            "inject processor_factory for this workload"
        ) from exc
    raise RuntimeError(
        "automatic Quandela processor construction requires a calibrated Quandela client; "
        "inject processor_factory for this workload"
    )


def _decode_payload(workload: QuantumWorkload) -> dict[str, object]:
    return _json_mapping(workload.program, field_name="Quandela workload")


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


def _validate_perceval_payload(payload: Mapping[str, object], n_modes: int) -> None:
    if payload.get("schema") != QUANDELA_PERCEVAL_SCHEMA:
        raise ValueError(f"unsupported Quandela schema; expected {QUANDELA_PERCEVAL_SCHEMA}")
    modes = _coerce_int(payload.get("modes"), field_name="modes")
    if modes != n_modes:
        raise ValueError("Quandela modes must match HAL workload mode count")
    input_state = payload.get("input_state")
    if not isinstance(input_state, Sequence) or isinstance(input_state, str):
        raise ValueError("Quandela input_state must be a sequence")
    if len(input_state) != n_modes:
        raise ValueError("Quandela input_state length must match modes")
    for occupation in input_state:
        value = _coerce_int(occupation, field_name="input_state occupation")
        if value < 0:
            raise ValueError("input_state occupations must be non-negative")
    components = payload.get("components")
    if not isinstance(components, Sequence) or isinstance(components, str):
        raise ValueError("Quandela components must be a sequence")
    for component in components:
        _validate_component(component, n_modes)
    postselection = payload.get("postselection", {})
    if not isinstance(postselection, Mapping):
        raise ValueError("Quandela postselection must be a mapping")
    if "min_detected_photons" in postselection:
        minimum = _coerce_int(
            postselection.get("min_detected_photons"),
            field_name="min_detected_photons",
        )
        if minimum < 0:
            raise ValueError("min_detected_photons must be non-negative")


def _validate_component(component: object, n_modes: int) -> None:
    if not isinstance(component, Mapping):
        raise ValueError("Quandela component entries must be mappings")
    component_type = component.get("type")
    if not isinstance(component_type, str) or not component_type:
        raise ValueError("Quandela component type must be non-empty")
    if component_type == "beam_splitter":
        modes = component.get("modes")
        if not isinstance(modes, Sequence) or isinstance(modes, str) or len(modes) != 2:
            raise ValueError("beam_splitter modes must contain two mode indices")
        first = _mode_index(modes[0], n_modes)
        second = _mode_index(modes[1], n_modes)
        if first == second:
            raise ValueError("beam_splitter modes must be distinct")
        _coerce_float(component.get("theta"), field_name="beam_splitter theta")
        return
    if component_type == "phase_shifter":
        _mode_index(component.get("mode"), n_modes)
        _coerce_float(component.get("phi"), field_name="phase_shifter phi")
        return
    raise ValueError(f"unsupported Quandela component type: {component_type}")


def _mode_index(value: object, n_modes: int) -> int:
    index = _coerce_int(value, field_name="mode")
    if index < 0 or index >= n_modes:
        raise ValueError("mode index out of range")
    return index


def _extract_counts(result: object) -> object:
    if isinstance(result, Mapping):
        for key in ("results", "counts", "samples"):
            value = result.get(key)
            if isinstance(value, Mapping):
                return value
    counts = getattr(result, "counts", None)
    if isinstance(counts, Mapping):
        return counts
    results = getattr(result, "results", None)
    if isinstance(results, Mapping):
        return results
    raise RuntimeError("Could not extract Quandela counts from provider result")


def _provider_job_id(result: object) -> str:
    for attr in ("job_id", "id", "task_id"):
        value = getattr(result, attr, None)
        if callable(value):
            value = value()
        if value is not None and str(value).strip():
            return strict_provider_job_id(value, field_name="Quandela provider job id")
    if isinstance(result, Mapping):
        for key in ("job_id", "id", "task_id"):
            value = result.get(key)
            if value is not None and str(value).strip():
                return strict_provider_job_id(value, field_name="Quandela provider job id")
    raise ValueError("Quandela result does not expose a provider job id")


def _hal_job_id(backend_id: str, workload_id: str, provider_job_id: str) -> str:
    digest = hashlib.sha256(f"{backend_id}|{provider_job_id}".encode()).hexdigest()[:12]
    return f"{backend_id}:{workload_id}:{digest}"


def _normalise_counts(raw: object) -> dict[str, int]:
    if not isinstance(raw, Mapping):
        raise TypeError("Quandela counts must be a mapping")
    counts: dict[str, int] = {}
    for state, count in raw.items():
        key = str(state).strip()
        value = _coerce_int(count, field_name="count")
        if not key:
            raise ValueError("counts keys must be non-empty photonic states")
        if value < 0:
            raise ValueError("counts values must be non-negative integers")
        counts[key] = counts.get(key, 0) + value
    if not counts:
        raise ValueError("Quandela result did not contain any counts")
    return counts


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


__all__ = [
    "QUANDELA_EXECUTION_MODE",
    "QUANDELA_PERCEVAL_SCHEMA",
    "QuandelaPercevalHALAdapter",
    "quandela_perceval_workload",
]
