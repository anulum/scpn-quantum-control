# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- AWS Braket adapters for the hardware HAL
"""Amazon Braket adapters for :mod:`scpn_quantum_control.hardware.hal`."""

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


def braket_circuit_to_workload(
    circuit: Any,
    *,
    workload_id: str,
    shots: int,
    metadata: Mapping[str, object] | None = None,
) -> QuantumWorkload:
    """Encode a Braket circuit as an OpenQASM 3 HAL workload."""
    from braket.circuits import Circuit

    if not isinstance(circuit, Circuit):
        raise TypeError("circuit must be a braket.circuits.Circuit")
    program = circuit.to_ir(ir_type="OPENQASM").source
    return QuantumWorkload(
        workload_id=workload_id,
        ir_format="openqasm3",
        program=program,
        n_qubits=len(circuit.qubits),
        shots=shots,
        metadata=dict(metadata or {}),
    )


class BraketLocalHALAdapter:
    """Local Amazon Braket simulator adapter implementing the HAL protocol."""

    def __init__(self, profile: BackendProfile, *, device: Any | None = None) -> None:
        if profile.backend_id not in {"local_braket_sv", "local_braket_dm"}:
            raise ValueError("BraketLocalHALAdapter requires a local Braket gate-model profile")
        self.profile = profile
        self.backend_id = profile.backend_id
        self._device = device
        self._jobs: dict[str, QuantumJobRef] = {}
        self._results: dict[str, QuantumJobResult] = {}

    def submit(
        self, workload: QuantumWorkload, *, approval_id: str | None = None
    ) -> QuantumJobRef:
        """Submit a workload to the backend and return its job reference."""
        del approval_id
        circuit = _workload_to_braket_circuit(workload)
        device = self._device or _default_local_device(self.profile.backend_id)
        task = device.run(circuit, shots=workload.shots)
        task_result = task.result()
        counts = _extract_braket_counts(task_result, n_qubits=workload.n_qubits)
        task_id = _task_id(task)
        job = QuantumJobRef(
            job_id=task_id,
            backend_id=self.backend_id,
            workload_id=workload.workload_id,
            status="completed",
            metadata={
                "provider_task_id": task_id,
                "execution_mode": "braket_local",
                "ir_format": workload.ir_format,
                "n_qubits": workload.n_qubits,
                "shots": workload.shots,
            },
        )
        observed_shots = strict_shot_conservation(counts, expected_shots=workload.shots)
        result = QuantumJobResult(
            job=job,
            status="completed",
            counts=counts,
            shots=observed_shots,
            metadata={
                "execution_mode": "braket_local",
                "ir_format": workload.ir_format,
                "device_name": _device_name(device),
                "timestamp": _utc_now(),
            },
        )
        self._jobs[job.job_id] = job
        self._results[job.job_id] = result
        return job

    def status(self, job: QuantumJobRef) -> str:
        """Return the current status for a submitted backend job."""
        stored = self._jobs.get(job.job_id)
        if stored is None:
            raise KeyError(f"unknown job_id: {job.job_id}")
        return stored.status

    def result(self, job: QuantumJobRef) -> QuantumJobResult:
        """Return the completed result for a submitted backend job."""
        result = self._results.get(job.job_id)
        if result is None:
            raise KeyError(f"unknown job_id: {job.job_id}")
        return result

    def cancel(self, job: QuantumJobRef) -> QuantumJobRef:
        """Request cancellation for a submitted backend job."""
        if job.job_id not in self._jobs:
            raise KeyError(f"unknown job_id: {job.job_id}")
        cancelled = QuantumJobRef(
            job_id=job.job_id,
            backend_id=job.backend_id,
            workload_id=job.workload_id,
            status="cancelled",
            metadata=job.metadata,
        )
        self._jobs[job.job_id] = cancelled
        return cancelled


class BraketAwsHALAdapter:
    """AWS Braket cloud adapter implementing the HAL protocol."""

    def __init__(
        self,
        profile: BackendProfile,
        *,
        device: Any | None = None,
        device_arn: str | None = None,
        device_factory: Callable[[str], Any] | None = None,
    ) -> None:
        if not profile.backend_id.startswith("aws_braket_"):
            raise ValueError("BraketAwsHALAdapter requires an aws_braket profile")
        if device is None and not device_arn:
            raise ValueError("device or device_arn is required for AWS Braket submission")
        self.profile = profile
        self.backend_id = profile.backend_id
        self._device = device
        self._device_arn = (
            strict_provider_job_id(device_arn, field_name="Braket device ARN")
            if device_arn is not None
            else None
        )
        self._device_factory = device_factory
        self._tasks: dict[str, Any] = {}
        self._jobs: dict[str, QuantumJobRef] = {}
        self._results: dict[str, QuantumJobResult] = {}

    def submit(
        self, workload: QuantumWorkload, *, approval_id: str | None = None
    ) -> QuantumJobRef:
        """Submit a workload to the backend and return its job reference."""
        if not approval_id:
            raise PermissionError("approval_id is required for AWS Braket submission")
        circuit = _workload_to_braket_circuit(workload)
        device = self._device or self._load_device()
        task = device.run(circuit, shots=workload.shots)
        task_id = _task_id(task)
        job = QuantumJobRef(
            job_id=task_id,
            backend_id=self.backend_id,
            workload_id=workload.workload_id,
            status="submitted",
            metadata={
                "approval_id": approval_id,
                "provider_task_id": task_id,
                "execution_mode": "braket_aws",
                "ir_format": workload.ir_format,
                "device_name": _device_name(device),
                "n_qubits": workload.n_qubits,
                "shots": workload.shots,
            },
        )
        self._tasks[job.job_id] = task
        self._jobs[job.job_id] = job
        return job

    def status(self, job: QuantumJobRef) -> str:
        """Return the current status for a submitted backend job."""
        task = self._task(job)
        state = task.state()
        return _normalise_status(getattr(state, "name", state))

    def result(self, job: QuantumJobRef) -> QuantumJobResult:
        """Return the completed result for a submitted backend job."""
        cached = self._results.get(job.job_id)
        if cached is not None:
            return cached
        task = self._task(job)
        n_qubits = strict_integer_value(job.metadata.get("n_qubits", 0), field_name="n_qubits")
        counts = _extract_braket_counts(task.result(), n_qubits=n_qubits)
        expected_shots = strict_integer_value(job.metadata.get("shots", 0), field_name="shots")
        observed_shots = strict_shot_conservation(counts, expected_shots=expected_shots)
        result = QuantumJobResult(
            job=job,
            status="completed",
            counts=counts,
            shots=observed_shots,
            metadata={
                "approval_id": job.metadata.get("approval_id"),
                "execution_mode": "braket_aws",
                "ir_format": job.metadata.get("ir_format"),
                "device_name": job.metadata.get("device_name"),
                "timestamp": _utc_now(),
            },
        )
        self._results[job.job_id] = result
        return result

    def cancel(self, job: QuantumJobRef) -> QuantumJobRef:
        """Request cancellation for a submitted backend job."""
        task = self._task(job)
        task.cancel()
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
        if self._device_arn is None:
            raise ValueError("device_arn is required when no device is injected")
        if self._device_factory is not None:
            return self._device_factory(self._device_arn)
        from braket.aws import AwsDevice

        return AwsDevice(self._device_arn)

    def _task(self, job: QuantumJobRef) -> Any:
        task = self._tasks.get(job.job_id)
        if task is None:
            raise KeyError(f"unknown job_id: {job.job_id}")
        return task


def _workload_to_braket_circuit(workload: QuantumWorkload) -> Any:
    if workload.ir_format != "openqasm3":
        raise ValueError("Braket adapters require OpenQASM 3 workloads")
    from braket.circuits import Circuit

    try:
        return Circuit.from_ir(workload.program)
    except Exception as exc:
        raise ValueError("OpenQASM 3 workload could not be decoded by Braket") from exc


def _default_local_device(backend_id: str) -> Any:
    try:
        from braket.devices import LocalSimulator

        if backend_id == "local_braket_dm":
            return LocalSimulator("braket_dm")
        return LocalSimulator("braket_sv")
    except Exception as exc:
        raise RuntimeError("amazon-braket-sdk is required for BraketLocalHALAdapter") from exc


def _extract_braket_counts(task_result: Any, *, n_qubits: int) -> dict[str, int]:
    if n_qubits <= 0:
        raise ValueError("Braket result decoding requires a positive n_qubits")
    counts = getattr(task_result, "measurement_counts", None)
    if counts is None:
        raise ValueError("Braket task result does not contain measurement_counts")
    normalised: dict[str, int] = {}
    for bitstring, count in counts.items():
        key = strict_fixed_width_bitstring_key(
            bitstring, width=n_qubits, field_name="Braket count key"
        )
        value = strict_non_negative_count(count)
        normalised[key] = normalised.get(key, 0) + value
    counts = normalised
    if not counts:
        raise ValueError("Braket task result contains an empty count map")
    return counts


def _task_id(task: Any) -> str:
    task_id = getattr(task, "id", None)
    if task_id:
        return strict_provider_job_id(task_id, field_name="Braket provider task id")
    raise ValueError("Braket task object does not expose a provider task id")


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
        "inprogress": "running",
        "initializing": "submitted",
        "initialising": "submitted",
        "starting": "submitted",
        "creating": "submitted",
        "created": "submitted",
        "submitted": "submitted",
        "queued": "queued",
        "pending": "queued",
        "cancelled": "cancelled",
        "canceled": "cancelled",
        "aborting": "cancelled",
        "cancelling": "cancelled",
        "canceling": "cancelled",
        "failed": "failed",
        "error": "failed",
    }.get(text, default)


def _device_name(device: Any) -> str:
    name = getattr(device, "name", None)
    if callable(name):
        return strict_provider_job_id(name(), field_name="Braket device name")
    if name is not None:
        return strict_provider_job_id(name, field_name="Braket device name")
    return strict_provider_job_id(device.__class__.__name__, field_name="Braket device name")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "BraketAwsHALAdapter",
    "BraketLocalHALAdapter",
    "braket_circuit_to_workload",
]
