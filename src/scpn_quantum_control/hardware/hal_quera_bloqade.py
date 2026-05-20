# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- QuEra Bloqade adapter for the hardware HAL
"""QuEra Bloqade adapter for :mod:`scpn_quantum_control.hardware.hal`."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from importlib import import_module
from typing import Any, cast

from .hal import BackendProfile, QuantumJobRef, QuantumJobResult, QuantumWorkload

BLOQADE_AHS_SCHEMA = "bloqade_ahs_plan_v1"
QUERA_BLOQADE_EXECUTION_MODE = "quera_bloqade"


def bloqade_ahs_workload(
    payload: Mapping[str, object] | str,
    *,
    workload_id: str,
    n_qubits: int,
    shots: int,
    metadata: Mapping[str, object] | None = None,
) -> QuantumWorkload:
    """Encode a Bloqade analogue Hamiltonian plan as a HAL workload."""

    if isinstance(payload, str):
        decoded = _json_mapping(payload, field_name="Bloqade payload")
    else:
        decoded = dict(payload)
    _validate_bloqade_payload(decoded, n_qubits)
    return QuantumWorkload(
        workload_id=workload_id,
        ir_format="bloqade",
        program=json.dumps(decoded, sort_keys=True, separators=(",", ":")),
        n_qubits=n_qubits,
        shots=shots,
        metadata=dict(metadata or {}),
    )


class QuEraBloqadeHALAdapter:
    """Bloqade routine adapter implementing the provider-neutral HAL protocol."""

    def __init__(
        self,
        profile: BackendProfile,
        *,
        routine: Any | None = None,
        routine_name: str | None = None,
        routine_factory: Callable[[QuantumWorkload], Any] | None = None,
    ) -> None:
        if profile.backend_id != "quera_bloqade":
            raise ValueError("QuEraBloqadeHALAdapter requires the quera_bloqade profile")
        if routine is None and routine_factory is None:
            raise ValueError("routine or routine_factory is required")
        self.profile = profile
        self.backend_id = profile.backend_id
        self._routine = routine
        self._routine_name = routine_name or "injected"
        self._routine_factory = routine_factory
        self._jobs: dict[str, QuantumJobRef] = {}
        self._batches: dict[str, Any] = {}
        self._results: dict[str, QuantumJobResult] = {}

    def submit(
        self, workload: QuantumWorkload, *, approval_id: str | None = None
    ) -> QuantumJobRef:
        """Submit a workload to the backend and return its job reference."""
        if not approval_id:
            raise PermissionError("approval_id is required for QuEra Bloqade submission")
        if workload.ir_format != "bloqade":
            raise ValueError("QuEra Bloqade direct adapter requires bloqade workloads")
        _validate_bloqade_payload(_decode_payload(workload), workload.n_qubits)
        routine = self._routine_for(workload)
        batch = routine.run(shots=workload.shots, name=workload.workload_id)
        provider_job_id = f"{self.backend_id}:{workload.workload_id}"
        job = QuantumJobRef(
            job_id=provider_job_id,
            backend_id=self.backend_id,
            workload_id=workload.workload_id,
            status="submitted",
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
        """Return the current status for a submitted backend job."""
        batch = self._batch(job)
        if callable(getattr(batch, "fetch", None)):
            batch = batch.fetch()
            self._batches[job.job_id] = batch
        return _normalise_status(getattr(batch, "status", "completed"))

    def result(self, job: QuantumJobRef) -> QuantumJobResult:
        """Return the completed result for a submitted backend job."""
        cached = self._results.get(job.job_id)
        if cached is not None:
            return cached
        stored = self._job(job)
        batch = self._batch(job)
        if callable(getattr(batch, "fetch", None)):
            batch = batch.fetch()
            self._batches[job.job_id] = batch
        counts = _normalise_counts(_extract_bitstrings(batch))
        result = QuantumJobResult(
            job=stored,
            status="completed",
            counts=counts,
            shots=sum(counts.values()),
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
        """Request cancellation for a submitted backend job."""
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
            metadata=stored.metadata,
        )
        self._jobs[job.job_id] = cancelled
        return cancelled

    def _routine_for(self, workload: QuantumWorkload) -> Any:
        if self._routine is not None:
            return self._routine
        factory = self._routine_factory or _default_routine_factory
        self._routine = factory(workload)
        return self._routine

    def _job(self, job: QuantumJobRef) -> QuantumJobRef:
        stored = self._jobs.get(job.job_id)
        if stored is None:
            raise KeyError(f"unknown job_id: {job.job_id}")
        return stored

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


def _normalise_counts(source: Sequence[Any] | Mapping[Any, int]) -> dict[str, int]:
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
        count = int(raw_count)
        if count < 0:
            raise ValueError("Bloqade counts must be non-negative")
        counts[bitstring] = counts.get(bitstring, 0) + count
    if not counts:
        raise ValueError("Bloqade result did not contain shots")
    return counts


def _normalise_bitstring(value: Any) -> str:
    if isinstance(value, (str, Sequence)):
        bits = [int(bit) for bit in value]
    else:
        raise ValueError("Bloqade bitstrings must be strings or bit sequences")
    if not bits:
        raise ValueError("Bloqade bitstrings must not be empty")
    if any(bit not in (0, 1) for bit in bits):
        raise ValueError("Bloqade bitstrings must contain binary values")
    return "".join(str(bit) for bit in bits)


def _normalise_status(status: object) -> str:
    raw_status = getattr(status, "name", status)
    return str(raw_status).split(".")[-1].lower().replace(" ", "_")


def _coerce_int(value: object, *, field_name: str) -> int:
    if isinstance(value, int):
        return value
    try:
        return int(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer") from exc


def _coerce_float(value: object, *, field_name: str) -> float:
    if isinstance(value, int | float):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric") from exc


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "BLOQADE_AHS_SCHEMA",
    "QUERA_BLOQADE_EXECUTION_MODE",
    "QuEraBloqadeHALAdapter",
    "bloqade_ahs_workload",
]
