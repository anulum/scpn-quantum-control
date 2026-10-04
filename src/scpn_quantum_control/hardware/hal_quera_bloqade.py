# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — HAL quera bloqade module
# scpn-quantum-control -- QuEra Bloqade adapter for the hardware HAL
"""QuEra Bloqade adapter for :mod:`scpn_quantum_control.hardware.hal`."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from importlib import import_module
from math import isfinite
from typing import Any, cast

from ._count_integrity import (
    strict_integer_value,
    strict_non_negative_count,
    strict_provider_job_id,
    strict_shot_conservation,
)
from .hal import (
    BackendProfile,
    QuantumJobRef,
    QuantumJobResult,
    QuantumWorkload,
    _resolve_stored_job,
    _validate_workload_for_profile,
)
from .provider_modalities import AnalogObservation, ModalitySemantics
from .provider_semantics import modality_submission_semantics

BLOQADE_AHS_SCHEMA = "bloqade_ahs_plan_v1"
QUERA_BLOQADE_EXECUTION_MODE = "quera_bloqade"


def bloqade_ahs_workload(
    payload: Mapping[str, object] | str,
    *,
    workload_id: str,
    n_qubits: int,
    shots: int,
    metadata: Mapping[str, object] | None = None,
    capture_semantics: bool = False,
    requested_target: str | None = None,
) -> QuantumWorkload:
    """Encode original ordered sites and an optional analog request companion.

    Parameters
    ----------
    payload
        AHS plan mapping or JSON object with ordered atom indices, positions,
        finite amplitude/phase schedules and positive duration. Captured JSON
        strings retain their exact original bytes; mappings retain site order.
    workload_id
        Stable caller identity for this original plan.
    n_qubits
        Legacy field containing the declared native site count. It does not
        infer gate-model qubits or readout polarity.
    shots
        Positive integral number of native samples.
    metadata
        Scalar caller annotations separate from execution settings.
    capture_semantics
        Retain an independent source digest and original native site order.
        Default false preserves the legacy workload codec.
    requested_target
        Optional exact declared routine selector, requiring native capture.

    Returns
    -------
    QuantumWorkload
        Original Bloqade plan and optional versioned analog companion.

    Raises
    ------
    ValueError
        If plan structure, native values, site count or target admission fails.

    """
    if isinstance(payload, str):
        decoded = _json_mapping(payload, field_name="Bloqade payload")
    else:
        decoded = dict(payload)
    _validate_bloqade_payload(decoded, n_qubits)
    if not capture_semantics and requested_target is not None:
        raise ValueError("target pin requires native semantics capture")
    program = (
        payload
        if isinstance(payload, str) and capture_semantics
        else json.dumps(decoded, sort_keys=not capture_semantics, separators=(",", ":"))
    )
    axes = _site_order(decoded)
    semantics = (
        ModalitySemantics(
            program_sha256=hashlib.sha256(program.encode()).hexdigest(),
            modality="analog",
            native_axes=axes,
            requested_target=requested_target,
        )
        if capture_semantics
        else None
    )
    return QuantumWorkload(
        workload_id=workload_id,
        ir_format="bloqade",
        program=program,
        n_qubits=n_qubits,
        shots=shots,
        metadata=dict(metadata or {}),
        semantics=semantics,
    )


class QuEraBloqadeHALAdapter:
    """Execute declared Bloqade plans through an explicit native-compatible routine.

    Parameters
    ----------
    profile
        QuEra route declaring supported IR and resource limits.
    routine
        Optional already prepared routine. Captured submission requires its
        exact original plan digest; opaque routines have caller provenance.
    routine_name
        Declared route selector, default injected only when absent. This name
        is an adapter selector and does not attest an SDK or physical device.
    routine_factory
        Optional builder called separately with each admitted original workload.
        Explicit callable objects are retained irrespective of truth value.
    prepared_program_sha256
        Exact source digest for a captured plan using an injected routine.

    Raises
    ------
    ValueError
        If profile, route configuration or explicit selector is invalid.

    """

    supports_provider_semantics = True

    def __init__(
        self,
        profile: BackendProfile,
        *,
        routine: Any | None = None,
        routine_name: str | None = None,
        routine_factory: Callable[[QuantumWorkload], Any] | None = None,
        prepared_program_sha256: str | None = None,
    ) -> None:
        if profile.backend_id != "quera_bloqade":
            raise ValueError("QuEraBloqadeHALAdapter requires the quera_bloqade profile")
        if routine is None and routine_factory is None:
            raise ValueError("routine or routine_factory is required")
        self.profile = profile
        self.backend_id = profile.backend_id
        self._routine = routine
        self._routine_name = strict_provider_job_id(
            routine_name if routine_name is not None else "injected",
            field_name="QuEra routine name",
        )
        self._routine_factory = routine_factory
        self._prepared_program_sha256 = prepared_program_sha256
        self._jobs: dict[str, QuantumJobRef] = {}
        self._batches: dict[str, Any] = {}
        self._results: dict[str, QuantumJobResult] = {}

    def submit(
        self, workload: QuantumWorkload, *, approval_id: str | None = None
    ) -> QuantumJobRef:
        """Admit original source, ordered sites and target before building a routine.

        Parameters
        ----------
        workload
            Original supported AHS plan with optional native analog companion.
        approval_id
            Required caller authorization for submission.

        Returns
        -------
        QuantumJobRef
            Stored job retaining source, exact sampling settings and selector
            provenance. No gate measurement map or physical calibration is inferred.

        Raises
        ------
        PermissionError
            If caller approval is absent.
        ValueError
            If profile, source, IR, native axes, target or prepared digest disagrees.
        RuntimeError
            If automatic construction lacks a calibrated provider builder.

        """
        if not approval_id:
            raise PermissionError("approval_id is required for QuEra Bloqade submission")
        if workload.ir_format != "bloqade":
            raise ValueError("QuEra Bloqade direct adapter requires bloqade workloads")
        _validate_workload_for_profile(self.profile, workload)
        plan = _decode_payload(workload)
        _validate_bloqade_payload(plan, workload.n_qubits)
        submission = modality_submission_semantics(
            workload.semantics,
            program=workload.program,
            ir_format=workload.ir_format,
            native_axes=_site_order(plan),
            modality="analog",
            target_name=self._routine_name,
            shots=workload.shots,
            caller_precompiled=self._routine is not None,
        )
        if (
            submission is not None
            and self._routine is not None
            and self._prepared_program_sha256 != submission.request.program_sha256
        ):
            raise ValueError("prepared Bloqade routine requires its exact original program digest")
        routine = self._routine_for(workload)
        batch = routine.run(shots=workload.shots, name=workload.workload_id)
        provider_job_id = _provider_job_id(batch)
        hal_job_id = _hal_job_id(self.backend_id, workload.workload_id, provider_job_id)
        job = QuantumJobRef(
            job_id=hal_job_id,
            backend_id=self.backend_id,
            workload_id=workload.workload_id,
            status="submitted",
            submission=submission,
            metadata={
                "approval_id": approval_id,
                "provider_job_id": provider_job_id,
                "execution_mode": QUERA_BLOQADE_EXECUTION_MODE,
                "routine_name": self._routine_name,
                "ir_format": workload.ir_format,
                "n_qubits": workload.n_qubits,
                "shots": workload.shots,
            },
        )
        self._jobs[job.job_id] = job
        self._batches[job.job_id] = batch
        return job

    def status(self, job: QuantumJobRef) -> str:
        """Retrieve provider lifecycle after checking the original stored identity.

        Parameters
        ----------
        job
            Original or recovered handle for the same backend and workload.

        Returns
        -------
        str
            Canonical native lifecycle, optionally obtained after provider fetch.

        Raises
        ------
        KeyError
            If the stored submission or batch is unavailable.
        ValueError
            If durable handle identity disagrees.

        """
        batch = self._batch(job)
        if callable(getattr(batch, "fetch", None)):
            batch = batch.fetch()
            self._batches[job.job_id] = batch
        return _normalise_status(getattr(batch, "status", "completed"))

    def result(self, job: QuantumJobRef) -> QuantumJobResult:
        """Retain unchanged native readouts and exact sampling conservation.

        Parameters
        ----------
        job
            Handle matching the original stored backend and workload.

        Returns
        -------
        QuantumJobResult
            Immutable compatibility histogram and optional analog observation
            retaining native site order, counts or ordered per-shot samples.
            Atom-loss interpretation and readout polarity remain unknown.
            Repeated successful retrieval returns the same stored result.

        Raises
        ------
        KeyError
            If original submission or retained batch is unavailable.
        ValueError
            If identity, channel, native values, sample width or shot total fails.

        """
        stored = self._job(job)
        cached = self._results.get(stored.job_id)
        if cached is not None:
            return cached
        batch = self._batch(job)
        if callable(getattr(batch, "fetch", None)):
            batch = batch.fetch()
            self._batches[job.job_id] = batch
        raw_readout = _extract_bitstrings(batch)
        counts = _normalise_counts(raw_readout)
        expected_shots = strict_integer_value(
            stored.metadata.get("shots"),
            field_name="Bloqade expected shots",
        )
        observed_shots = strict_shot_conservation(counts, expected_shots=expected_shots)
        observation = None
        if stored.submission is not None:
            request = stored.submission.request
            if not isinstance(request, ModalitySemantics):
                raise ValueError("native Bloqade result requires analog plan semantics")
            if isinstance(raw_readout, Mapping):
                if any(not isinstance(key, str) for key in raw_readout):
                    raise ValueError("native analog count channel requires unchanged string keys")
                observation = AnalogObservation(
                    request=request,
                    shots=observed_shots,
                    raw_counts=cast(Mapping[str, int], raw_readout),
                )
            else:
                observation = AnalogObservation(
                    request=request,
                    shots=observed_shots,
                    raw_samples=tuple(
                        value if isinstance(value, str) else tuple(value) for value in raw_readout
                    ),
                )
        result = QuantumJobResult(
            job=stored,
            status="completed",
            counts=counts,
            shots=observed_shots,
            provider_observation=observation,
            metadata={
                "approval_id": stored.metadata.get("approval_id"),
                "execution_mode": QUERA_BLOQADE_EXECUTION_MODE,
                "routine_name": self._routine_name,
                "timestamp": _utc_now(),
            },
        )
        self._results[job.job_id] = result
        return result

    def cancel(self, job: QuantumJobRef) -> QuantumJobRef:
        """Request cancellation while retaining source and completed raw evidence.

        Parameters
        ----------
        job
            Handle matching the original stored backend and workload.

        Returns
        -------
        QuantumJobRef
            Legacy cancellation annotation preserving the native submission.
            Missing provider cancel remains supported; this annotation does not
            attest physical provider cancellation.

        Raises
        ------
        KeyError
            If original submission or batch is unavailable.
        ValueError
            If durable handle identity differs.

        """
        stored = self._job(job)
        batch = self._batch(job)
        cancel = getattr(batch, "cancel", None)
        if callable(cancel):
            cancel()
        cancelled = QuantumJobRef(
            job_id=stored.job_id,
            backend_id=stored.backend_id,
            workload_id=stored.workload_id,
            status="cancelled",
            submission=stored.submission,
            metadata=stored.metadata,
        )
        self._jobs[job.job_id] = cancelled
        return cancelled

    def _routine_for(self, workload: QuantumWorkload) -> Any:
        if self._routine is not None:
            return self._routine
        factory = (
            self._routine_factory
            if self._routine_factory is not None
            else _default_routine_factory
        )
        return factory(workload)

    def _job(self, job: QuantumJobRef) -> QuantumJobRef:
        return _resolve_stored_job(job, self._jobs)

    def _batch(self, job: QuantumJobRef) -> Any:
        self._job(job)
        return self._batches[job.job_id]


def _default_routine_factory(workload: QuantumWorkload) -> Any:
    try:
        import_module("bloqade")
    except Exception as exc:
        raise RuntimeError("bloqade is required for QuEraBloqadeHALAdapter") from exc
    raise RuntimeError(
        "automatic Bloqade routine construction requires a calibrated provider builder; "
        "inject routine_factory for this workload"
    )


def _decode_payload(workload: QuantumWorkload) -> dict[str, object]:
    return _json_mapping(workload.program, field_name="Bloqade workload")


def _site_order(plan: Mapping[str, object]) -> tuple[str | int, ...]:
    """Read original native site indices in the validated plan's array order."""
    atoms = cast(Sequence[Mapping[str, object]], plan["atoms"])
    return tuple(
        strict_integer_value(atom["index"], field_name="Bloqade site index") for atom in atoms
    )


def _json_mapping(source: str, *, field_name: str) -> dict[str, object]:
    try:
        payload = json.loads(source)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field_name} is not valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"{field_name} must be a JSON object")
    return dict(payload)


def _validate_bloqade_payload(payload: Mapping[str, object], n_qubits: int) -> None:
    if payload.get("schema") != BLOQADE_AHS_SCHEMA:
        raise ValueError("unsupported Bloqade AHS schema")
    atoms = payload.get("atoms")
    if not isinstance(atoms, Sequence) or isinstance(atoms, str):
        raise ValueError("Bloqade AHS payload atoms must be a sequence")
    if len(atoms) != n_qubits:
        raise ValueError("Bloqade AHS atom count does not match workload qubit count")
    for atom in atoms:
        if not isinstance(atom, Mapping):
            raise ValueError("Bloqade AHS atom entries must be JSON objects")
        _coerce_int(atom.get("index"), field_name="Bloqade atom index")
        position = atom.get("position")
        if not isinstance(position, Sequence) or isinstance(position, str) or len(position) != 2:
            raise ValueError("Bloqade atom position must contain x and y coordinates")
        [_coerce_float(value, field_name="Bloqade atom coordinate") for value in position]
    for field in ("rabi_amplitude_piecewise_linear", "rabi_phase_piecewise_linear"):
        _validate_schedule(payload.get(field), field_name=field)
    duration = _coerce_float(payload.get("duration"), field_name="Bloqade duration")
    if duration <= 0.0:
        raise ValueError("Bloqade duration must be positive")


def _validate_schedule(value: object, *, field_name: str) -> None:
    if not isinstance(value, Sequence) or isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty sequence")
    for point in value:
        if not isinstance(point, Sequence) or isinstance(point, str) or len(point) != 2:
            raise ValueError(f"{field_name} entries must contain time and value")
        _coerce_float(point[0], field_name=f"{field_name} time")
        _coerce_float(point[1], field_name=f"{field_name} value")


def _extract_bitstrings(batch: Any) -> Sequence[Any] | Mapping[Any, int]:
    if isinstance(batch, Mapping):
        if "counts" in batch:
            return cast(Mapping[Any, int], batch["counts"])
        if "bitstrings" in batch:
            return cast(Sequence[Any], batch["bitstrings"])
    report = batch.report() if callable(getattr(batch, "report", None)) else batch
    if isinstance(report, Mapping):
        if "counts" in report:
            return cast(Mapping[Any, int], report["counts"])
        if "bitstrings" in report:
            return cast(Sequence[Any], report["bitstrings"])
    for attr in ("counts", "bitstrings", "raw_bitstrings"):
        value = getattr(report, attr, None)
        if value is not None:
            return cast(Sequence[Any] | Mapping[Any, int], value)
    raise ValueError("Bloqade batch report does not contain bitstrings or counts")


def _normalise_counts(source: Sequence[Any] | Mapping[Any, object]) -> dict[str, int]:
    if isinstance(source, Mapping):
        items = source.items()
    else:
        observed: dict[str, int] = {}
        for value in source:
            bitstring = _normalise_bitstring(value)
            observed[bitstring] = observed.get(bitstring, 0) + 1
        items = observed.items()
    counts: dict[str, int] = {}
    for raw_bitstring, raw_count in items:
        bitstring = _normalise_bitstring(raw_bitstring)
        count = strict_non_negative_count(raw_count)
        counts[bitstring] = counts.get(bitstring, 0) + count
    if not counts:
        raise ValueError("Bloqade result did not contain shots")
    return counts


def _normalise_bitstring(value: Any) -> str:
    if isinstance(value, (str, Sequence)):
        bits = [strict_integer_value(bit, field_name="Bloqade bit") for bit in value]
    else:
        raise ValueError("Bloqade bitstrings must be strings or bit sequences")
    if not bits:
        raise ValueError("Bloqade bitstrings must not be empty")
    if any(bit not in (0, 1) for bit in bits):
        raise ValueError("Bloqade bitstrings must contain binary values")
    return "".join(str(bit) for bit in bits)


def _normalise_status(status: object) -> str:
    raw_status = getattr(status, "name", status)
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
        "cancelled": "cancelled",
        "canceled": "cancelled",
        "failed": "failed",
        "error": "failed",
    }.get(text, "unknown")


def _provider_job_id(batch: object) -> str:
    for attr in ("id", "job_id", "handle", "task_id"):
        value = getattr(batch, attr, None)
        if callable(value):
            value = value()
        if value is not None and str(value).strip():
            return strict_provider_job_id(value, field_name="Bloqade provider job id")
    raise ValueError("Bloqade batch does not expose a provider job id")


def _hal_job_id(backend_id: str, workload_id: str, provider_job_id: str) -> str:
    digest = hashlib.sha256(provider_job_id.encode("utf-8")).hexdigest()[:12]
    return f"{backend_id}:{workload_id}:{digest}"


def _coerce_int(value: object, *, field_name: str) -> int:
    return strict_integer_value(value, field_name=field_name)


def _coerce_float(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise ValueError(f"{field_name} must be numeric")
    try:
        result = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{field_name} must be numeric") from exc
    if not isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


__all__ = [
    "BLOQADE_AHS_SCHEMA",
    "QUERA_BLOQADE_EXECUTION_MODE",
    "QuEraBloqadeHALAdapter",
    "bloqade_ahs_workload",
]
