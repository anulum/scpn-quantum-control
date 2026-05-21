# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- IonQ adapter for the hardware HAL
"""Direct IonQ Cloud adapter for :mod:`scpn_quantum_control.hardware.hal`."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from importlib import import_module
from typing import Any

from .hal import BackendProfile, QuantumJobRef, QuantumJobResult, QuantumWorkload

IONQ_API_V04_BASE_URL = "https://api.ionq.co/v0.4"
IONQ_CIRCUIT_SCHEMA = "scpn.ionq.qis_circuit.v1"


def ionq_qis_workload(
    circuit: Sequence[Mapping[str, object]],
    *,
    workload_id: str,
    n_qubits: int,
    shots: int,
    gateset: str = "qis",
    metadata: Mapping[str, object] | None = None,
) -> QuantumWorkload:
    """Encode IonQ's language-neutral QIS circuit JSON as a HAL workload."""

    payload = {
        "schema": IONQ_CIRCUIT_SCHEMA,
        "input": {
            "qubits": n_qubits,
            "gateset": gateset,
            "circuit": [_normalise_ionq_gate(gate, n_qubits) for gate in circuit],
        },
    }
    return QuantumWorkload(
        workload_id=workload_id,
        ir_format="ionq_json",
        program=json.dumps(payload, sort_keys=True, separators=(",", ":")),
        n_qubits=n_qubits,
        shots=shots,
        metadata=dict(metadata or {}),
    )


class IonQCloudHALAdapter:
    """IonQ v0.4 REST adapter implementing the provider-neutral HAL protocol."""

    def __init__(
        self,
        profile: BackendProfile,
        *,
        client: Any | None = None,
        api_key: str | None = None,
        api_key_env: str = "IONQ_API_KEY",
        backend: str = "simulator",
        base_url: str = IONQ_API_V04_BASE_URL,
        settings: Mapping[str, object] | None = None,
        timeout_s: float = 30.0,
    ) -> None:
        if profile.backend_id != "ionq_cloud":
            raise ValueError("IonQCloudHALAdapter requires the ionq_cloud profile")
        resolved_api_key = api_key or os.environ.get(api_key_env)
        if not resolved_api_key:
            raise ValueError("api_key or IONQ_API_KEY environment variable is required")
        self.profile = profile
        self.backend_id = profile.backend_id
        self._client = client
        self._api_key = resolved_api_key
        self._backend = backend
        self._base_url = base_url.rstrip("/")
        self._settings = dict(settings or {})
        self.timeout_s = timeout_s
        self._jobs: dict[str, QuantumJobRef] = {}
        self._results: dict[str, QuantumJobResult] = {}

    def submit(
        self, workload: QuantumWorkload, *, approval_id: str | None = None
    ) -> QuantumJobRef:
        """Submit a workload to the backend and return its job reference."""
        if not approval_id:
            raise PermissionError("approval_id is required for IonQ submission")
        payload = self._build_job_payload(workload, approval_id)
        response = self._post_json(f"{self._base_url}/jobs", payload)
        job_id = str(response.get("id") or "")
        if not job_id:
            raise ValueError("IonQ job creation response did not contain an id")
        status = _normalise_status(response.get("status"))
        job = QuantumJobRef(
            job_id=job_id,
            backend_id=self.backend_id,
            workload_id=workload.workload_id,
            status=status,
            metadata={
                "approval_id": approval_id,
                "provider_job_id": job_id,
                "execution_mode": "ionq_cloud_api_v0.4",
                "backend": self._backend,
                "ir_format": workload.ir_format,
                "n_qubits": workload.n_qubits,
                "shots": workload.shots,
            },
        )
        self._jobs[job.job_id] = job
        return job

    def status(self, job: QuantumJobRef) -> str:
        """Return the current status for a submitted backend job."""
        payload = self._get_json(f"{self._base_url}/jobs/{job.job_id}")
        return _normalise_status(payload.get("status"))

    def result(self, job: QuantumJobRef) -> QuantumJobResult:
        """Return the completed result for a submitted backend job."""
        cached = self._results.get(job.job_id)
        if cached is not None:
            return cached
        stored = self._job(job)
        probabilities = self._get_json(f"{self._base_url}/jobs/{job.job_id}/results/probabilities")
        shots = _int_metadata(stored.metadata.get("shots"), fallback=0) or _job_shots(stored)
        n_qubits = _job_n_qubits(stored)
        counts = _probabilities_to_counts(probabilities, shots, n_qubits)
        result = QuantumJobResult(
            job=stored,
            status="completed",
            counts=counts,
            shots=sum(counts.values()),
            metadata={
                "approval_id": stored.metadata.get("approval_id"),
                "execution_mode": "ionq_cloud_api_v0.4",
                "backend": stored.metadata.get("backend"),
                "ir_format": stored.metadata.get("ir_format"),
                "timestamp": _utc_now(),
            },
        )
        self._results[job.job_id] = result
        return result

    def cancel(self, job: QuantumJobRef) -> QuantumJobRef:
        """Request cancellation for a submitted backend job."""
        self._job(job)
        payload = self._put_json(f"{self._base_url}/jobs/{job.job_id}/status/cancel")
        cancelled = QuantumJobRef(
            job_id=job.job_id,
            backend_id=job.backend_id,
            workload_id=job.workload_id,
            status=_normalise_status(payload.get("status"), default="cancelled"),
            metadata=job.metadata,
        )
        self._jobs[job.job_id] = cancelled
        return cancelled

    def _build_job_payload(self, workload: QuantumWorkload, approval_id: str) -> dict[str, object]:
        if workload.ir_format != "ionq_json":
            raise ValueError("IonQ direct adapter requires ionq_json workloads")
        ionq_input = _decode_ionq_input(workload)
        payload: dict[str, object] = {
            "type": "ionq.circuit.v1",
            "name": workload.workload_id,
            "metadata": {
                "approval_id": approval_id,
                "workload_id": workload.workload_id,
                **dict(workload.metadata),
            },
            "shots": workload.shots,
            "backend": self._backend,
            "input": ionq_input,
        }
        if self._settings:
            payload["settings"] = self._settings
        return payload

    def _post_json(self, url: str, payload: Mapping[str, object]) -> dict[str, object]:
        response = self._http_client().post(
            url,
            headers=self._headers(),
            json=dict(payload),
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        return _json_mapping(response.json())

    def _get_json(self, url: str) -> dict[str, object]:
        response = self._http_client().get(url, headers=self._headers(), timeout=self.timeout_s)
        response.raise_for_status()
        return _json_mapping(response.json())

    def _put_json(self, url: str) -> dict[str, object]:
        response = self._http_client().put(url, headers=self._headers(), timeout=self.timeout_s)
        response.raise_for_status()
        return _json_mapping(response.json())

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"apiKey {self._api_key}",
            "Content-Type": "application/json",
        }

    def _http_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            requests = import_module("requests")
            self._client = requests.Session()
            return self._client
        except Exception as exc:
            raise RuntimeError("requests is required for IonQCloudHALAdapter") from exc

    def _job(self, job: QuantumJobRef) -> QuantumJobRef:
        stored = self._jobs.get(job.job_id)
        if stored is None:
            raise KeyError(f"unknown job_id: {job.job_id}")
        return stored


def _decode_ionq_input(workload: QuantumWorkload) -> dict[str, object]:
    try:
        payload = json.loads(workload.program)
    except json.JSONDecodeError as exc:
        raise ValueError("IonQ workload is not valid JSON") from exc
    if payload.get("schema") != IONQ_CIRCUIT_SCHEMA:
        raise ValueError("unsupported IonQ workload schema")
    ionq_input = payload.get("input")
    if not isinstance(ionq_input, Mapping):
        raise ValueError("IonQ workload input must be a JSON object")
    qubits = _coerce_int(ionq_input.get("qubits"), field_name="IonQ qubits")
    if qubits != workload.n_qubits:
        raise ValueError("IonQ workload qubit count does not match HAL workload")
    circuit = ionq_input.get("circuit")
    if not isinstance(circuit, list):
        raise ValueError("IonQ workload circuit must be a list")
    return {
        "qubits": workload.n_qubits,
        "gateset": str(ionq_input.get("gateset", "qis")),
        "circuit": [_normalise_ionq_gate(gate, workload.n_qubits) for gate in circuit],
    }


def _normalise_ionq_gate(gate: Mapping[str, object], n_qubits: int) -> dict[str, object]:
    name = str(gate.get("gate", "")).lower()
    if not name:
        raise ValueError("IonQ gate name must be non-empty")
    result: dict[str, object] = {"gate": name}
    for field in ("target", "control", "control1", "control2"):
        if field in gate:
            result[field] = _normalise_wire(gate[field], n_qubits, field)
    if "targets" in gate:
        targets = gate["targets"]
        if not isinstance(targets, Sequence) or isinstance(targets, str):
            raise ValueError("IonQ gate targets must be a sequence of integers")
        result["targets"] = [_normalise_wire(target, n_qubits, "targets") for target in targets]
    if not any(field in result for field in ("target", "targets")):
        raise ValueError("IonQ gate requires a target or targets field")
    for field in ("rotation", "phase", "angle"):
        if field in gate:
            result[field] = _coerce_float(gate[field], field_name=f"IonQ {field}")
    return result


def _normalise_wire(value: object, n_qubits: int, field_name: str) -> int:
    wire = _coerce_int(value, field_name=f"IonQ {field_name}")
    if wire < 0 or wire >= n_qubits:
        raise ValueError(f"IonQ {field_name} wire is outside the workload register")
    return wire


def _probabilities_to_counts(
    probabilities: Mapping[str, object], shots: int, n_qubits: int
) -> dict[str, int]:
    if shots <= 0:
        raise ValueError("IonQ result conversion requires a positive shot count")
    weighted: list[tuple[str, int, float]] = []
    assigned = 0
    for key, value in probabilities.items():
        probability = _coerce_float(value, field_name="IonQ probability")
        raw_count = probability * shots
        count = int(raw_count)
        assigned += count
        bitstring = _ionq_key_to_bitstring(key, n_qubits)
        weighted.append((bitstring, count, raw_count - count))
    remaining = shots - assigned
    weighted.sort(key=lambda item: item[2], reverse=True)
    counts: dict[str, int] = {}
    for index, (bitstring, count, _fraction) in enumerate(weighted):
        increment = 1 if index < remaining else 0
        total = count + increment
        if total:
            counts[bitstring] = counts.get(bitstring, 0) + total
    return counts


def _ionq_key_to_bitstring(key: str, n_qubits: int) -> str:
    if key.isdigit():
        return format(int(key), f"0{n_qubits}b")
    if set(key) <= {"0", "1"}:
        if len(key) > n_qubits:
            raise ValueError("IonQ probability key is wider than the workload register")
        return key.zfill(n_qubits)
    raise ValueError("IonQ probability keys must be decimal states or bitstrings")


def _json_mapping(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("IonQ API response must be a JSON object")
    return dict(value)


def _int_metadata(value: object, *, fallback: int) -> int:
    if value is None:
        return fallback
    return _coerce_int(value, field_name="IonQ metadata")


def _job_shots(job: QuantumJobRef) -> int:
    shots = job.metadata.get("shots")
    if shots is None:
        raise ValueError("IonQ job metadata does not contain shots")
    return _coerce_int(shots, field_name="IonQ shots")


def _job_n_qubits(job: QuantumJobRef) -> int:
    n_qubits = job.metadata.get("n_qubits")
    if n_qubits is None:
        raise ValueError("IonQ job metadata does not contain n_qubits")
    value = _coerce_int(n_qubits, field_name="IonQ n_qubits")
    if value <= 0:
        raise ValueError("IonQ n_qubits must be positive")
    return value


def _coerce_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip():
        return int(value)
    raise ValueError(f"{field_name} must be an integer")


def _coerce_float(value: object, *, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be numeric")
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str) and value.strip():
        return float(value)
    raise ValueError(f"{field_name} must be numeric")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalise_status(value: object, *, default: str = "unknown") -> str:
    text = str(value or default).strip().lower()
    return {
        "complete": "completed",
        "completed": "completed",
        "succeeded": "completed",
        "success": "completed",
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
    "IONQ_API_V04_BASE_URL",
    "IonQCloudHALAdapter",
    "ionq_qis_workload",
]
