# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Cirq adapter for the hardware HAL
"""Local Cirq simulator adapter for the provider-neutral HAL."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from importlib import import_module
from typing import Any

from ._count_integrity import (
    strict_integer_value,
    strict_provider_job_id,
    strict_shot_conservation,
)
from .hal import BackendProfile, QuantumJobRef, QuantumJobResult, QuantumWorkload

CIRQ_EXECUTION_MODE = "local_cirq_simulator"


def cirq_circuit_workload(
    circuit: object,
    *,
    workload_id: str,
    n_qubits: int,
    shots: int,
    metadata: Mapping[str, object] | None = None,
) -> QuantumWorkload:
    """Encode a Cirq circuit handle or serialised circuit as a HAL workload."""
    if isinstance(circuit, str):
        program = circuit
    else:
        program = repr(circuit)
    return QuantumWorkload(
        workload_id=workload_id,
        ir_format="cirq",
        program=program,
        n_qubits=n_qubits,
        shots=shots,
        metadata=dict(metadata or {}),
    )


class CirqLocalHALAdapter:
    """Local Cirq simulator adapter implementing the HAL protocol."""

    def __init__(
        self,
        profile: BackendProfile,
        *,
        simulator: Any | None = None,
        simulator_factory: Callable[[], Any] | None = None,
        circuit_factory: Callable[[str], Any] | None = None,
        measurement_key: str = "m",
    ) -> None:
        if profile.backend_id != "local_cirq":
            raise ValueError("CirqLocalHALAdapter requires the local_cirq profile")
        self.profile = profile
        self.backend_id = profile.backend_id
        self._simulator = simulator
        self._simulator_factory = simulator_factory
        self._circuit_factory = circuit_factory
        self._measurement_key = measurement_key
        self._jobs: dict[str, QuantumJobRef] = {}
        self._results: dict[str, QuantumJobResult] = {}

    def submit(
        self, workload: QuantumWorkload, *, approval_id: str | None = None
    ) -> QuantumJobRef:
        """Submit a workload to the backend and return its job reference."""
        del approval_id
        if workload.ir_format != "cirq":
            raise ValueError("Cirq local adapter requires cirq workloads")
        circuit = self._build_circuit(workload.program)
        run = getattr(self._simulator_for(), "run", None)
        if not callable(run):
            raise TypeError("Cirq simulator object does not provide run()")
        raw_result = run(circuit, repetitions=workload.shots)
        counts = _normalise_histogram_counts(raw_result, self._measurement_key, workload.n_qubits)
        observed_shots = strict_shot_conservation(counts, expected_shots=workload.shots)
        provider_job_id = _provider_job_id(raw_result)
        hal_job_id = _hal_job_id(self.backend_id, workload.workload_id, provider_job_id)
        job = QuantumJobRef(
            job_id=hal_job_id,
            backend_id=self.backend_id,
            workload_id=workload.workload_id,
            status="completed",
            metadata={
                "provider_job_id": provider_job_id,
                "execution_mode": CIRQ_EXECUTION_MODE,
                "measurement_key": self._measurement_key,
                "ir_format": workload.ir_format,
                "n_qubits": workload.n_qubits,
                "shots": workload.shots,
            },
        )
        result = QuantumJobResult(
            job=job,
            status="completed",
            counts=counts,
            shots=observed_shots,
            metadata={
                "execution_mode": CIRQ_EXECUTION_MODE,
                "measurement_key": self._measurement_key,
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

    def _build_circuit(self, source: str) -> Any:
        if self._circuit_factory is not None:
            return self._circuit_factory(source)
        return _default_circuit_factory(source)

    def _simulator_for(self) -> Any:
        if self._simulator is not None:
            return self._simulator
        factory = self._simulator_factory or _default_simulator_factory
        self._simulator = factory()
        return self._simulator

    def _job(self, job: QuantumJobRef) -> QuantumJobRef:
        stored = self._jobs.get(job.job_id)
        if stored is None:
            raise KeyError(f"unknown job_id: {job.job_id}")
        return stored


def _default_circuit_factory(source: str) -> Any:
    try:
        import_module("cirq")
    except Exception as exc:
        raise RuntimeError("cirq-core is required for CirqLocalHALAdapter") from exc
    raise RuntimeError("automatic Cirq circuit construction requires circuit_factory")


def _default_simulator_factory() -> Any:
    try:
        cirq = import_module("cirq")
    except Exception as exc:
        raise RuntimeError("cirq-core is required for CirqLocalHALAdapter") from exc
    return cirq.Simulator()


def _normalise_histogram_counts(result: object, key: str, n_qubits: int) -> dict[str, int]:
    histogram = getattr(result, "histogram", None)
    if callable(histogram):
        raw = histogram(key=key)
    elif isinstance(result, Mapping):
        raw = result.get("histogram", result.get("counts"))
    else:
        raw = None
    if not isinstance(raw, Mapping):
        raise ValueError("Cirq result does not expose histogram counts")
    counts: dict[str, int] = {}
    for state, count in raw.items():
        value = _coerce_int(count, field_name="count")
        if value < 0:
            raise ValueError("counts values must be non-negative integers")
        bitstring = _state_to_bitstring(state, n_qubits)
        counts[bitstring] = counts.get(bitstring, 0) + value
    if not counts:
        raise ValueError("Cirq result did not contain any counts")
    return counts


def _state_to_bitstring(state: object, n_qubits: int) -> str:
    if isinstance(state, str):
        text = state.strip()
        if not text:
            raise ValueError("Cirq histogram state keys must be non-empty")
        return text
    value = _coerce_int(state, field_name="histogram state")
    if value < 0:
        raise ValueError("histogram state must be non-negative")
    return format(value, f"0{n_qubits}b")


def _coerce_int(value: object, *, field_name: str) -> int:
    return strict_integer_value(value, field_name=field_name)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _provider_job_id(raw_result: object) -> str:
    for attr in ("job_id", "id", "task_id"):
        value = getattr(raw_result, attr, None)
        if callable(value):
            value = value()
        if value is not None and str(value).strip():
            return strict_provider_job_id(value, field_name="Cirq provider job id")
    metadata = getattr(raw_result, "metadata", None)
    if isinstance(metadata, Mapping):
        for key in ("job_id", "id", "task_id"):
            value = metadata.get(key)
            if value is not None and str(value).strip():
                return strict_provider_job_id(value, field_name="Cirq provider job id")
    raise ValueError("Cirq result does not expose a provider job id")


def _hal_job_id(backend_id: str, workload_id: str, provider_job_id: str) -> str:
    digest = hashlib.sha256(provider_job_id.encode("utf-8")).hexdigest()[:12]
    return f"{backend_id}:{workload_id}:{digest}"


__all__ = ["CIRQ_EXECUTION_MODE", "CirqLocalHALAdapter", "cirq_circuit_workload"]
