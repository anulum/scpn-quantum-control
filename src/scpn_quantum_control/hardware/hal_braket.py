# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — HAL braket module
# scpn-quantum-control -- AWS Braket adapters for the hardware HAL
"""Amazon Braket adapters for :mod:`scpn_quantum_control.hardware.hal`."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

from ._count_integrity import (
    strict_fixed_width_bitstring_key,
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
from .provider_capability_core import ProviderCapabilitySnapshot
from .provider_semantics import GateModelObservation, SubmissionSemantics, WorkloadSemantics
from .provider_submission_gate import require_submit_time_capability


def braket_circuit_to_workload(
    circuit: Any,
    *,
    workload_id: str,
    shots: int,
    metadata: Mapping[str, object] | None = None,
    capture_semantics: bool = False,
    requested_target: str | None = None,
    parameter_bindings: Mapping[str, float] | None = None,
) -> QuantumWorkload:
    """Encode the original native circuit and optional static sampling contract.

    Parameters
    ----------
    circuit
        Native Braket ``Circuit``. Shared free symbols retain their original
        OpenQASM names; this function does not bind or execute the circuit.
    workload_id
        Stable caller identity for the unchanged source payload.
    shots
        Positive integral number of samples, conserved by result decoding.
    metadata
        Caller annotations separate from source and execution settings.
    capture_semantics
        Capture static final measurement order, shared symbol uses and the
        SHA-256 of the original OpenQASM. Default false preserves legacy output.
    requested_target
        Optional exact native device name. Requires capture and must equal
        the selected device's name before its ``run`` method is called.
    parameter_bindings
        Finite real values keyed by original native free-symbol names.
        Requires capture; submission requires every original free symbol.

    Returns
    -------
    QuantumWorkload
        Original OpenQASM with a separate versioned companion when requested.
        Native measured-qubit order uses the leftmost count bit first; no
        logical qubit permutation or count reversal is performed here.

    Raises
    ------
    TypeError
        If the source is not a native Braket circuit.
    ValueError
        If settings require capture, bindings are invalid, or a gate follows
        final measurement in the admitted static sampling subset.

    """
    from braket.circuits import Circuit

    if not isinstance(circuit, Circuit):
        raise TypeError("circuit must be a braket.circuits.Circuit")
    if not capture_semantics and (requested_target is not None or parameter_bindings is not None):
        raise ValueError("target or parameter bindings require native semantics capture")
    program = circuit.to_ir(ir_type="OPENQASM").source
    semantics = (
        _braket_workload_semantics(
            circuit,
            program,
            requested_target=requested_target,
            parameter_bindings=parameter_bindings,
        )
        if capture_semantics
        else None
    )
    return QuantumWorkload(
        workload_id=workload_id,
        ir_format="openqasm3",
        program=program,
        n_qubits=semantics.n_qubits if semantics is not None else len(circuit.qubits),
        shots=shots,
        metadata=dict(metadata or {}),
        semantics=semantics,
    )


class BraketLocalHALAdapter:
    """Execute native Braket workloads on the selected local simulator.

    Parameters
    ----------
    profile
        Built-in statevector or density-matrix Braket profile.
    device
        Optional native-compatible transport. An injected object is retained
        regardless of its truth value; absence selects the profile's simulator.

    """

    supports_provider_semantics = True

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
        """Validate and execute one native workload without implicit target changes.

        Parameters
        ----------
        workload
            Original OpenQASM and optional captured static measurement contract.
        approval_id
            Unused for local simulator execution.

        Returns
        -------
        QuantumJobRef
            Completed handle with stored submission settings and raw count
            observation when capture was requested.

        Raises
        ------
        ValueError
            If source, profile, target, bindings or provider counts are invalid.

        """
        del approval_id
        _validate_workload_for_profile(self.profile, workload)
        circuit = _workload_to_braket_circuit(workload)
        device = (
            self._device
            if self._device is not None
            else _default_local_device(self.profile.backend_id)
        )
        circuit, submission = _prepare_braket_submission(workload, circuit, _device_name(device))
        task = device.run(circuit, shots=workload.shots)
        task_result = task.result()
        observation = _braket_observation(task_result, submission, workload.shots)
        counts = (
            dict(observation.counts)
            if observation is not None
            else _extract_braket_counts(task_result, n_qubits=workload.n_qubits)
        )
        task_id = _task_id(task)
        job = QuantumJobRef(
            job_id=task_id,
            backend_id=self.backend_id,
            workload_id=workload.workload_id,
            status="completed",
            submission=submission,
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
            provider_observation=observation,
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
        """Return the terminal state of a retained local submission.

        Parameters
        ----------
        job
            Handle matching the stored job, backend and workload identity.

        Returns
        -------
        str
            Completed lifecycle state.

        Raises
        ------
        KeyError
            If the job is unknown.
        ValueError
            If backend or workload identity differs.

        """
        return _resolve_stored_job(job, self._jobs).status

    def result(self, job: QuantumJobRef) -> QuantumJobResult:
        """Return immutable captured output or the legacy count result.

        Parameters
        ----------
        job
            Handle matching the original stored submission identity.

        Returns
        -------
        QuantumJobResult
            Previously validated result; repeated calls retain the same object.

        Raises
        ------
        KeyError
            If the job is unknown.
        ValueError
            If backend or workload identity differs.

        """
        stored = _resolve_stored_job(job, self._jobs)
        return self._results[stored.job_id]

    def cancel(self, job: QuantumJobRef) -> QuantumJobRef:
        """Preserve terminal local evidence when cancellation arrives late.

        Parameters
        ----------
        job
            Handle matching the original stored submission identity.

        Returns
        -------
        QuantumJobRef
            Stored completed handle without changing source or raw evidence.

        Raises
        ------
        KeyError
            If the job is unknown.
        ValueError
            If backend or workload identity differs.

        """
        return _resolve_stored_job(job, self._jobs)


class BraketAwsHALAdapter:
    """Submit native Braket circuits through an explicitly configured AWS device.

    Parameters
    ----------
    profile
        AWS Braket gate-model route whose resource and IR limits are enforced.
    device
        Optional injected native-compatible device, retained even when falsey.
    device_arn
        Explicit ARN required when a device is not injected.
    device_factory
        Optional constructor receiving the exact configured ARN.
    capability_probe
        Optional no-submit metadata probe paired with a calibration age limit.
    max_calibration_age_seconds
        Finite nonnegative age limit in seconds, paired with the probe.

    """

    supports_provider_semantics = True

    def __init__(
        self,
        profile: BackendProfile,
        *,
        device: Any | None = None,
        device_arn: str | None = None,
        device_factory: Callable[[str], Any] | None = None,
        capability_probe: Callable[[], ProviderCapabilitySnapshot] | None = None,
        max_calibration_age_seconds: float | None = None,
    ) -> None:
        if not profile.backend_id.startswith("aws_braket_"):
            raise ValueError("BraketAwsHALAdapter requires an aws_braket profile")
        if device is None and not device_arn:
            raise ValueError("device or device_arn is required for AWS Braket submission")
        if (capability_probe is None) != (max_calibration_age_seconds is None):
            raise ValueError("capability_probe and max_calibration_age_seconds must be paired")
        self.profile = profile
        self.backend_id = profile.backend_id
        self._device = device
        self._device_arn = (
            strict_provider_job_id(device_arn, field_name="Braket device ARN")
            if device_arn is not None
            else None
        )
        self._device_factory = device_factory
        self._capability_probe = capability_probe
        self._max_calibration_age_seconds = max_calibration_age_seconds
        self._tasks: dict[str, Any] = {}
        self._jobs: dict[str, QuantumJobRef] = {}
        self._results: dict[str, QuantumJobResult] = {}

    def submit(
        self, workload: QuantumWorkload, *, approval_id: str | None = None
    ) -> QuantumJobRef:
        """Submit once after approval and native source/target admission.

        Parameters
        ----------
        workload
            Original OpenQASM and optional static gate-model companion.
        approval_id
            Required caller authorization recorded with the submitted job.

        Returns
        -------
        QuantumJobRef
            Submitted handle retaining the original payload, requested and
            effective shots, native target and compiled payload digest.

        Raises
        ------
        PermissionError
            If approval is absent.
        ValueError
            If source, target, bindings, resource limits or metadata disagree.

        """
        if not approval_id:
            raise PermissionError("approval_id is required for AWS Braket submission")
        _validate_workload_for_profile(self.profile, workload)
        circuit = _workload_to_braket_circuit(workload)
        device = self._device if self._device is not None else self._load_device()
        circuit, submission = _prepare_braket_submission(workload, circuit, _device_name(device))
        calibration_metadata: dict[str, object] = {}
        if self._capability_probe is not None:
            assert self._max_calibration_age_seconds is not None
            decision = require_submit_time_capability(
                self._capability_probe(),
                profile=self.profile,
                target_name=_device_name(device),
                workload=workload,
                max_calibration_age_seconds=self._max_calibration_age_seconds,
                checked_at=datetime.now(UTC),
            )
            calibration_metadata = {
                "calibration_timestamp": decision.snapshot.calibration_timestamp,
                "calibration_checked_at": decision.freshness_as_of,
                "calibration_max_age_seconds": decision.max_calibration_age_seconds,
            }
        task = device.run(circuit, shots=workload.shots)
        task_id = _task_id(task)
        job = QuantumJobRef(
            job_id=task_id,
            backend_id=self.backend_id,
            workload_id=workload.workload_id,
            status="submitted",
            submission=submission,
            metadata={
                "approval_id": approval_id,
                "provider_task_id": task_id,
                "execution_mode": "braket_aws",
                "ir_format": workload.ir_format,
                "device_name": _device_name(device),
                "n_qubits": workload.n_qubits,
                "shots": workload.shots,
                **calibration_metadata,
            },
        )
        self._tasks[job.job_id] = task
        self._jobs[job.job_id] = job
        return job

    def status(self, job: QuantumJobRef) -> str:
        """Read provider lifecycle after checking stored submission identity.

        Parameters
        ----------
        job
            Original or recovered handle with unchanged durable identity.

        Returns
        -------
        str
            Canonical provider state, or completed when a result is retained.

        Raises
        ------
        KeyError
            If the job is unknown.
        ValueError
            If backend or workload identity differs.

        """
        job = _resolve_stored_job(job, self._jobs)
        if job.job_id in self._results:
            return "completed"
        task = self._task(job)
        state = task.state()
        return _normalise_status(getattr(state, "name", state))

    def result(self, job: QuantumJobRef) -> QuantumJobResult:
        """Return evidence decoded against the stored submission settings.

        Parameters
        ----------
        job
            Original or recovered handle. Its backend and workload identity
            must match a submission retained by this adapter.

        Returns
        -------
        QuantumJobResult
            Result using stored shot and provenance metadata, including when
            a recovered handle has different lifecycle annotations.

        Raises
        ------
        KeyError
            If this adapter has no retained submission for the job.
        ValueError
            If identity differs or provider counts violate stored settings.

        """
        job = _resolve_stored_job(job, self._jobs)
        cached = self._results.get(job.job_id)
        if cached is not None:
            return cached
        task = self._task(job)
        n_qubits = strict_integer_value(job.metadata.get("n_qubits", 0), field_name="n_qubits")
        expected_shots = strict_integer_value(job.metadata.get("shots", 0), field_name="shots")
        native_result = task.result()
        observation = _braket_observation(native_result, job.submission, expected_shots)
        counts = (
            dict(observation.counts)
            if observation is not None
            else _extract_braket_counts(native_result, n_qubits=n_qubits)
        )
        observed_shots = strict_shot_conservation(counts, expected_shots=expected_shots)
        result = QuantumJobResult(
            job=job,
            status="completed",
            counts=counts,
            shots=observed_shots,
            provider_observation=observation,
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
        """Request cancellation without replacing stored submission metadata.

        Parameters
        ----------
        job
            Handle with the same durable identity as the stored submission.

        Returns
        -------
        QuantumJobRef
            Updated lifecycle handle retaining original submission metadata.

        Raises
        ------
        KeyError
            If no submission is retained by this adapter.
        ValueError
            If the supplied identity differs from the stored identity.

        """
        job = _resolve_stored_job(job, self._jobs)
        if job.job_id in self._results:
            observed = "completed"
        else:
            task = self._task(job)
            observed = _normalise_status(task.state())
            if observed not in {"completed", "cancelled", "failed"}:
                task.cancel()
                observed = _normalise_status(task.state())
            if job.job_id in self._results:
                observed = "completed"
        updated = replace(job, status=observed)
        self._jobs[job.job_id] = updated
        return updated

    def _load_device(self) -> Any:
        if self._device_arn is None:
            raise ValueError("device_arn is required when no device is injected")
        if self._device_factory is not None:
            return self._device_factory(self._device_arn)
        from braket.aws import AwsDevice

        return AwsDevice(self._device_arn)

    def _task(self, job: QuantumJobRef) -> Any:
        job = _resolve_stored_job(job, self._jobs)
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


def _braket_workload_semantics(
    circuit: Any,
    program: str,
    *,
    requested_target: str | None = None,
    parameter_bindings: Mapping[str, float] | None = None,
) -> WorkloadSemantics:
    """Read native Braket measurement order and shared OpenQASM symbol uses."""
    from braket.circuits import FreeParameterExpression

    measurement: list[tuple[int, int]] = []
    uses: dict[str, list[tuple[int, int]]] = {}
    measured = False
    for index, instruction in enumerate(circuit.instructions):
        if instruction.operator.name == "Measure":
            measured = True
            for qubit in instruction.target:
                measurement.append((int(qubit), len(measurement)))
        elif measured:
            raise ValueError("native Braket gate follows final measurement")
        for argument, expression in enumerate(getattr(instruction.operator, "parameters", ())):
            if isinstance(expression, FreeParameterExpression):
                for symbol in expression.expression.free_symbols:
                    uses.setdefault(str(symbol), []).append((index, argument))
    if not measurement:
        measurement = [(int(qubit), index) for index, qubit in enumerate(sorted(circuit.qubits))]
    parameters = tuple((name, "braket:" + name, tuple(uses[name])) for name in sorted(uses))
    values: list[tuple[str, float]] = []
    for name, value in sorted((parameter_bindings or {}).items()):
        if name not in uses:
            raise ValueError("Braket bindings must name original native parameters")
        values.append(("braket:" + name, value))
    width = max(int(qubit) for qubit in circuit.qubits) + 1
    return WorkloadSemantics(
        program_sha256=hashlib.sha256(program.encode()).hexdigest(),
        n_qubits=width,
        n_clbits=len(measurement),
        measurement_map=tuple(measurement),
        classical_registers=(("b", tuple(range(len(measurement)))),),
        parameters=parameters,
        parameter_values=tuple(values),
        requested_target=requested_target,
        count_bit_order="classical_lsb_left",
    )


def _prepare_braket_submission(
    workload: QuantumWorkload,
    circuit: Any,
    target_name: str,
) -> tuple[Any, SubmissionSemantics | None]:
    """Bind the unchanged native source and explicit symbols before device run."""
    request = workload.semantics
    if request is None:
        if circuit.parameters:
            raise ValueError(
                "Braket sampled execution requires explicit native parameter bindings"
            )
        return circuit, None
    if not isinstance(request, WorkloadSemantics):
        raise ValueError("Braket circuit submission requires native gate-model semantics")
    values = {
        name: dict(request.parameter_values)[identity]
        for name, identity, _ in request.parameters
        if identity in dict(request.parameter_values)
    }
    observed = _braket_workload_semantics(
        circuit,
        workload.program,
        requested_target=request.requested_target,
        parameter_bindings=values,
    )
    if observed != request or len(values) != len(request.parameters):
        raise ValueError("native Braket request wiring or parameter bindings differ")
    bound = circuit.make_bound_circuit(values, strict=True) if values else circuit
    submission = SubmissionSemantics(
        request=request,
        original_program=workload.program,
        ir_format=workload.ir_format,
        requested_shots=workload.shots,
        effective_shots=workload.shots,
        target_name=target_name,
        compilation="native_provider",
        compiled_program_sha256=hashlib.sha256(
            bound.to_ir(ir_type="OPENQASM").source.encode()
        ).hexdigest(),
    )
    return bound, submission


def _braket_observation(
    result: Any,
    submission: SubmissionSemantics | None,
    shots: int,
) -> GateModelObservation | None:
    """Retain native count keys and verify actual measured-qubit output order."""
    if submission is None:
        return None
    request = submission.require_gate_request()
    measured = getattr(result, "measured_qubits", None)
    if measured != [qubit for qubit, _ in request.measurement_map]:
        raise ValueError("native Braket measured-qubit order differs from stored request")
    counts = getattr(result, "measurement_counts", None)
    if not isinstance(counts, Mapping):
        raise ValueError("native Braket observation requires measurement_counts")
    return GateModelObservation(request=request, raw_counts=counts, shots=shots)


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
    return datetime.now(UTC).isoformat()


__all__ = [
    "BraketAwsHALAdapter",
    "BraketLocalHALAdapter",
    "braket_circuit_to_workload",
]
