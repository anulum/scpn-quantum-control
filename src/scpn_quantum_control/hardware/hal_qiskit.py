# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — HAL Qiskit module
# scpn-quantum-control -- Qiskit adapters for the hardware HAL
"""Qiskit-backed adapters for :mod:`scpn_quantum_control.hardware.hal`."""

from __future__ import annotations

import base64
import hashlib
import io
from collections.abc import Callable, Mapping
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any, cast

from qiskit import QuantumCircuit, qasm3, qpy, transpile
from qiskit.circuit import Parameter
from qiskit.primitives import PrimitiveJob
from qiskit.primitives.containers import SamplerPubResult
from qiskit.qpy import dump as qpy_dump

from ._count_integrity import (
    strict_binary_bitstring_key,
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
from .provider_measurement import (
    bind_qiskit_workload,
    native_runtime_gate_observation,
    qiskit_submission_semantics,
    qiskit_workload_semantics,
    require_runtime_sample_buffers,
)
from .provider_semantics import GateModelObservation
from .provider_submission_gate import require_submit_time_capability
from .runner import _extract_counts


def qiskit_circuit_to_workload(
    circuit: QuantumCircuit,
    *,
    workload_id: str,
    shots: int,
    metadata: dict[str, object] | None = None,
    capture_semantics: bool = False,
    requested_target: str | None = None,
    parameter_bindings: Mapping[Parameter, float] | None = None,
) -> QuantumWorkload:
    """Encode the unchanged native source with an optional QPY sampling contract.

    Parameters
    ----------
    circuit
        Original Qiskit circuit. QPY retains actual shared ``Parameter`` UUIDs.
    workload_id
        Stable caller identity for this encoded source.
    shots
        Positive integral number of samples, conserved by result decoding.
    metadata
        Caller scalar annotations separate from execution settings.
    capture_semantics
        Capture source digest, final measurement map, register order and native
        parameter uses. Default false preserves the legacy workload contract.
    requested_target
        Optional exact native backend name, admitted before execution. Requires
        capture; no implicit replacement device is allowed.
    parameter_bindings
        Finite real values keyed by actual original native ``Parameter`` objects.
        Requires capture. Values are applied to a copy at submission; the original
        shared parameter identities and source QPY are retained unchanged.

    Returns
    -------
    QuantumWorkload
        Original base64 QPY with an independent versioned sampling companion.

    Raises
    ------
    TypeError
        If the input is not a native Qiskit circuit.
    ValueError
        If settings require capture, bindings are invalid, or the circuit is
        outside the supported static final-measurement subset.

    """
    if not isinstance(circuit, QuantumCircuit):
        raise TypeError("circuit must be a qiskit.QuantumCircuit")
    if requested_target is not None and not capture_semantics:
        raise ValueError("requested_target requires native semantics capture")
    if parameter_bindings is not None and not capture_semantics:
        raise ValueError("parameter_bindings requires native semantics capture")
    program = _circuit_to_qpy_b64(circuit)
    return QuantumWorkload(
        workload_id=workload_id,
        ir_format="qiskit_qpy",
        program=program,
        n_qubits=circuit.num_qubits,
        shots=shots,
        metadata=metadata or {},
        semantics=(
            qiskit_workload_semantics(
                circuit,
                program,
                requested_target=requested_target,
                parameter_bindings=parameter_bindings,
            )
            if capture_semantics
            else None
        ),
    )


def qiskit_circuit_to_qasm3_workload(
    circuit: QuantumCircuit,
    *,
    workload_id: str,
    shots: int,
    metadata: dict[str, object] | None = None,
    capture_semantics: bool = False,
    requested_target: str | None = None,
) -> QuantumWorkload:
    """Encode OpenQASM 3 and optionally capture fully bound static sampling.

    Parameters
    ----------
    circuit
        Original native circuit. Captured sampling requires no free parameters,
        because OpenQASM does not preserve original native ``Parameter`` UUIDs.
    workload_id
        Stable caller identity for the original exported source.
    shots
        Positive integral number of samples, conserved by result decoding.
    metadata
        Caller scalar annotations separate from execution settings.
    capture_semantics
        Capture source digest, static measurement map and register order.
        Default false preserves existing OpenQASM payload construction.
    requested_target
        Optional exact native backend name. Requires capture and is checked
        against the selected backend before native transport execution.

    Returns
    -------
    QuantumWorkload
        Original OpenQASM text and optional independent sampling companion.

    Raises
    ------
    TypeError
        If the input is not a native Qiskit circuit.
    ValueError
        If a target requires capture, the circuit is outside the static subset,
        or capture would lose original free-parameter identity. Use QPY and
        its original native parameter bindings for the latter case.

    """
    if not isinstance(circuit, QuantumCircuit):
        raise TypeError("circuit must be a qiskit.QuantumCircuit")
    if requested_target is not None and not capture_semantics:
        raise ValueError("requested_target requires native semantics capture")
    if capture_semantics and circuit.parameters:
        raise ValueError("OpenQASM 3 cannot preserve original parameter UUIDs; use QPY")
    program = qasm3.dumps(circuit)
    return QuantumWorkload(
        workload_id=workload_id,
        ir_format="openqasm3",
        program=program,
        n_qubits=circuit.num_qubits,
        shots=shots,
        metadata=metadata or {},
        semantics=(
            qiskit_workload_semantics(circuit, program, requested_target=requested_target)
            if capture_semantics
            else None
        ),
    )


class QiskitAerHALAdapter:
    """Execute native Qiskit sampling on the selected Aer-compatible backend.

    Parameters
    ----------
    profile
        Built-in local Qiskit Aer route.
    backend
        Optional native-compatible backend retained independently of truthiness.
        Absence selects the default native Aer simulator.

    """

    supports_provider_semantics = True

    def __init__(self, profile: BackendProfile, *, backend: Any | None = None) -> None:
        if profile.backend_id != "local_qiskit_aer":
            raise ValueError("QiskitAerHALAdapter requires the local_qiskit_aer profile")
        self.profile = profile
        self.backend_id = profile.backend_id
        self._backend = backend
        self._jobs: dict[str, QuantumJobRef] = {}
        self._results: dict[str, QuantumJobResult] = {}

    def submit(
        self, workload: QuantumWorkload, *, approval_id: str | None = None
    ) -> QuantumJobRef:
        """Bind and compile a copy for the exact native target before execution.

        Parameters
        ----------
        workload
            Original QPY or OpenQASM with an optional native sampling companion.
        approval_id
            Unused for local simulator execution.

        Returns
        -------
        QuantumJobRef
            Completed stored handle preserving the original payload and captured
            requested/effective settings, including targeted compilation digest.

        Raises
        ------
        ValueError
            If source, target, static subset, bindings or provider counts disagree.

        """
        del approval_id
        _validate_workload_for_profile(self.profile, workload)
        circuit = _workload_to_qiskit_circuit(workload)
        backend = self._backend if self._backend is not None else _default_aer_backend()
        submission = qiskit_submission_semantics(
            workload, circuit, target_name=_backend_name(backend)
        )
        compiled = transpile(bind_qiskit_workload(workload, circuit), backend)
        if submission is not None:
            submission = replace(
                submission,
                compilation="targeted",
                compiled_program_sha256=hashlib.sha256(
                    _circuit_to_qpy_b64(compiled).encode()
                ).hexdigest(),
            )
        provider_job = backend.run(compiled, shots=workload.shots)
        raw_counts = provider_job.result().get_counts()
        observation = (
            GateModelObservation(
                request=submission.require_gate_request(),
                raw_counts=raw_counts,
                shots=workload.shots,
            )
            if submission is not None
            else None
        )
        counts = (
            dict(observation.counts) if observation is not None else _normalise_counts(raw_counts)
        )
        observed_shots = strict_shot_conservation(counts, expected_shots=workload.shots)
        job_id = _provider_job_id(provider_job, provider_name="qiskit_aer")
        job = QuantumJobRef(
            job_id=job_id,
            backend_id=self.backend_id,
            workload_id=workload.workload_id,
            status="completed",
            submission=submission,
            metadata={
                "provider_job_id": job_id,
                "execution_mode": "qiskit_aer",
                "ir_format": workload.ir_format,
            },
        )
        result = QuantumJobResult(
            job=job,
            status="completed",
            counts=counts,
            shots=observed_shots,
            provider_observation=observation,
            metadata={
                "execution_mode": "qiskit_aer",
                "ir_format": workload.ir_format,
                "backend_name": _backend_name(backend),
                "timestamp": _utc_now(),
            },
        )
        self._jobs[job.job_id] = job
        self._results[job.job_id] = result
        return job

    def status(self, job: QuantumJobRef) -> str:
        """Return the retained terminal state after checking durable identity.

        Parameters
        ----------
        job
            Original or recovered handle for the stored backend and workload.

        Returns
        -------
        str
            Completed state of the retained local execution.

        Raises
        ------
        KeyError
            If the job is unknown.
        ValueError
            If backend or workload identity differs.

        """
        return _resolve_stored_job(job, self._jobs).status

    def result(self, job: QuantumJobRef) -> QuantumJobResult:
        """Return the same validated result without rewriting prior evidence.

        Parameters
        ----------
        job
            Handle matching the exact stored submission identity.

        Returns
        -------
        QuantumJobResult
            Retained immutable counts and optional native measurement observation.

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
            Retained completed handle without changing source or raw evidence.

        Raises
        ------
        KeyError
            If the job is unknown.
        ValueError
            If backend or workload identity differs.

        """
        return _resolve_stored_job(job, self._jobs)


class QiskitRuntimeHALAdapter:
    """Submit one native Qiskit circuit through the configured Runtime sampler.

    Parameters
    ----------
    profile
        IBM Quantum route whose resource and IR admission is enforced.
    backend
        Exact native backend used for target admission and captured compilation.
    sampler_factory
        Optional native sampler constructor receiving this backend as ``mode``.
        An injected callable is retained independently of truthiness.
    timeout_s
        Provider result-retrieval timeout in seconds, default 600. Native local
        ``PrimitiveJob`` retrieval uses its supported no-timeout signature.
    capability_probe
        Optional no-submit probe paired with a calibration age limit.
    max_calibration_age_seconds
        Finite nonnegative calibration age limit in seconds, paired with the probe.

    """

    supports_provider_semantics = True

    def __init__(
        self,
        profile: BackendProfile,
        *,
        backend: Any,
        sampler_factory: Callable[..., Any] | None = None,
        timeout_s: float = 600.0,
        capability_probe: Callable[[], ProviderCapabilitySnapshot] | None = None,
        max_calibration_age_seconds: float | None = None,
    ) -> None:
        if profile.backend_id != "ibm_quantum":
            raise ValueError("QiskitRuntimeHALAdapter requires the ibm_quantum profile")
        if (capability_probe is None) != (max_calibration_age_seconds is None):
            raise ValueError("capability_probe and max_calibration_age_seconds must be paired")
        self.profile = profile
        self.backend_id = profile.backend_id
        self._backend = backend
        self._sampler_factory = sampler_factory
        self.timeout_s = timeout_s
        self._capability_probe = capability_probe
        self._max_calibration_age_seconds = max_calibration_age_seconds
        self._provider_jobs: dict[str, Any] = {}
        self._jobs: dict[str, QuantumJobRef] = {}
        self._results: dict[str, QuantumJobResult] = {}

    def submit(
        self, workload: QuantumWorkload, *, approval_id: str | None = None
    ) -> QuantumJobRef:
        """Admit source, target and native sample buffers before constructing a sampler.

        Parameters
        ----------
        workload
            Original QPY or bound OpenQASM source and optional static sampling
            contract. Shared free parameters require QPY and original bindings.
        approval_id
            Required caller authorization stored with the submitted job.

        Returns
        -------
        QuantumJobRef
            Submitted handle retaining source, target, exact shot settings and
            targeted compilation digest when capture was requested.

        Raises
        ------
        PermissionError
            If approval is absent.
        ValueError
            If source, target, bindings, profile or live metadata disagree.
        MemoryError
            If declared native register buffers exceed the active byte budget.
            This admission is a snapshot, not a memory reservation or whole-process
            allocation guarantee.

        """
        if not approval_id:
            raise PermissionError("approval_id is required for IBM Runtime submission")
        _validate_workload_for_profile(self.profile, workload)
        circuit = _workload_to_qiskit_circuit(workload)
        submission = qiskit_submission_semantics(
            workload, circuit, target_name=_backend_name(self._backend)
        )
        if submission is not None:
            require_runtime_sample_buffers(submission.require_gate_request(), workload.shots)
        circuit = bind_qiskit_workload(workload, circuit)
        if submission is not None:
            circuit = transpile(circuit, backend=self._backend)
            submission = replace(
                submission,
                compilation="targeted",
                compiled_program_sha256=hashlib.sha256(
                    _circuit_to_qpy_b64(circuit).encode()
                ).hexdigest(),
            )
        calibration_metadata: dict[str, object] = {}
        if self._capability_probe is not None:
            assert self._max_calibration_age_seconds is not None
            decision = require_submit_time_capability(
                self._capability_probe(),
                profile=self.profile,
                target_name=_backend_name(self._backend),
                workload=workload,
                max_calibration_age_seconds=self._max_calibration_age_seconds,
                checked_at=datetime.now(UTC),
            )
            calibration_metadata = {
                "calibration_timestamp": decision.snapshot.calibration_timestamp,
                "calibration_checked_at": decision.freshness_as_of,
                "calibration_max_age_seconds": decision.max_calibration_age_seconds,
            }
        sampler_factory = (
            self._sampler_factory
            if self._sampler_factory is not None
            else _runtime_sampler_factory()
        )
        sampler = sampler_factory(mode=self._backend)
        sampler.options.default_shots = workload.shots
        provider_job = sampler.run([circuit])
        job_id = _provider_job_id(provider_job, provider_name="qiskit_runtime")
        job = QuantumJobRef(
            job_id=job_id,
            backend_id=self.backend_id,
            workload_id=workload.workload_id,
            status="submitted",
            submission=submission,
            metadata={
                "approval_id": approval_id,
                "execution_mode": "qiskit_runtime_sampler",
                "backend_name": _backend_name(self._backend),
                "ir_format": workload.ir_format,
                "shots": workload.shots,
                **calibration_metadata,
            },
        )
        self._provider_jobs[job.job_id] = provider_job
        self._jobs[job.job_id] = job
        return job

    def status(self, job: QuantumJobRef) -> str:
        """Read provider lifecycle after checking the exact retained job identity.

        Parameters
        ----------
        job
            Original or recovered handle with unchanged durable identity.

        Returns
        -------
        str
            Canonical provider lifecycle, or completed for a retained result.

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
        provider_job = self._provider_job(job)
        status = provider_job.status()
        return _normalise_status(getattr(status, "name", status))

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
            a recovered handle has different lifecycle annotations. Captured
            sampling requires exactly one actual native ``SamplerPubResult``;
            its ordered uint8 packed register arrays and native joint counts
            remain separate from the legacy summary. Packed arrays have shape
            ``(shots, ceil(register_bits / 8))`` without parameter broadcasting.

        Raises
        ------
        KeyError
            If this adapter has no retained submission for the job.
        ValueError
            If identity, PUB cardinality or counts violate stored settings.
        TypeError
            If captured register channels are not actual native BitArrays.
        MemoryError
            If native packed buffers exceed the active byte budget before copying.

        """
        job = _resolve_stored_job(job, self._jobs)
        cached = self._results.get(job.job_id)
        if cached is not None:
            return cached
        provider_job = self._provider_job(job)
        runtime_result = (
            provider_job.result()
            if isinstance(provider_job, PrimitiveJob)
            else provider_job.result(timeout=self.timeout_s)
        )
        counts: dict[str, int] = {}
        observation: GateModelObservation | None = None
        expected_shots = strict_integer_value(job.metadata.get("shots", 0), field_name="shots")
        if job.submission is not None:
            iterator = iter(runtime_result)
            absent = object()
            pub_result = next(iterator, absent)
            if pub_result is absent or next(iterator, absent) is not absent:
                raise ValueError("native single-circuit semantics requires exactly one PUB result")
            if not isinstance(pub_result, SamplerPubResult):
                raise ValueError("native semantics requires an actual SamplerPubResult")
            observation = native_runtime_gate_observation(
                pub_result,
                job.submission.require_gate_request(),
                shots=expected_shots,
            )
            counts = dict(observation.counts)
        else:
            for pub_result in runtime_result:
                if isinstance(pub_result, SamplerPubResult):
                    raw = pub_result.join_data(list(pub_result.data)).get_counts()
                else:
                    raw = _extract_counts(pub_result)
                for bitstring, count in _normalise_counts(raw).items():
                    counts[bitstring] = counts.get(bitstring, 0) + count
        observed_shots = strict_shot_conservation(counts, expected_shots=expected_shots)
        result = QuantumJobResult(
            job=job,
            status="completed",
            counts=counts,
            shots=observed_shots,
            provider_observation=observation,
            metadata={
                "approval_id": job.metadata.get("approval_id"),
                "execution_mode": "qiskit_runtime_sampler",
                "backend_name": job.metadata.get("backend_name"),
                "ir_format": job.metadata.get("ir_format"),
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
            provider_job = self._provider_job(job)
            observed = _normalise_status(provider_job.status())
            if observed not in {"completed", "cancelled", "failed"}:
                provider_job.cancel()
                observed = _normalise_status(provider_job.status())
            if job.job_id in self._results:
                observed = "completed"
        updated = replace(job, status=observed)
        self._jobs[job.job_id] = updated
        return updated

    def _provider_job(self, job: QuantumJobRef) -> Any:
        job = _resolve_stored_job(job, self._jobs)
        provider_job = self._provider_jobs.get(job.job_id)
        if provider_job is None:
            raise KeyError(f"unknown job_id: {job.job_id}")
        return provider_job


def _workload_to_qiskit_circuit(workload: QuantumWorkload) -> QuantumCircuit:
    if workload.ir_format == "qiskit_qpy":
        return _qpy_b64_to_circuit(workload.program)
    if workload.ir_format == "openqasm3":
        try:
            return qasm3.loads(workload.program)
        except Exception as exc:
            raise ValueError("OpenQASM 3 workload could not be decoded by Qiskit") from exc
    raise ValueError("Qiskit adapters require qiskit_qpy or OpenQASM 3 workloads")


def _circuit_to_qpy_b64(circuit: QuantumCircuit) -> str:
    buffer = io.BytesIO()
    qpy_dump(circuit, buffer)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _qpy_b64_to_circuit(payload: str) -> QuantumCircuit:
    try:
        data = base64.b64decode(payload.encode("ascii"), validate=True)
        circuits = _reviewed_qpy_load_circuits(data)
    except Exception as exc:
        raise ValueError("qiskit_qpy workload could not be decoded") from exc
    if len(circuits) != 1:
        raise ValueError("qiskit_qpy workload must contain exactly one circuit")
    circuit = circuits[0]
    if not isinstance(circuit, QuantumCircuit):
        raise TypeError("qiskit_qpy payload did not decode to a QuantumCircuit")
    return circuit


def _reviewed_qpy_load_circuits(data: bytes) -> list[QuantumCircuit]:
    """Decode trusted in-process QPY bytes behind the reviewed HAL wrapper."""
    return cast(list[QuantumCircuit], qpy.load(io.BytesIO(data)))


def _default_aer_backend() -> Any:
    try:
        from qiskit_aer import AerSimulator

        return AerSimulator()
    except Exception as exc:
        raise RuntimeError("qiskit-aer is required for QiskitAerHALAdapter") from exc


def _runtime_sampler_factory() -> Callable[..., Any]:
    try:
        from qiskit_ibm_runtime import SamplerV2

        return cast(Callable[..., Any], SamplerV2)
    except Exception as exc:
        raise RuntimeError("qiskit-ibm-runtime is required for QiskitRuntimeHALAdapter") from exc


def _normalise_counts(counts: dict[Any, Any]) -> dict[str, int]:
    normalised: dict[str, int] = {}
    for key, value in counts.items():
        bitstring = strict_binary_bitstring_key(key, field_name="Qiskit count key")
        count = strict_non_negative_count(value)
        normalised[bitstring] = normalised.get(bitstring, 0) + count
    if not normalised:
        raise ValueError("Qiskit result did not contain any counts")
    return normalised


def _backend_name(backend: Any) -> str:
    name = getattr(backend, "name", None)
    if callable(name):
        return strict_provider_job_id(name(), field_name="Qiskit backend name")
    if name is not None:
        return strict_provider_job_id(name, field_name="Qiskit backend name")
    return strict_provider_job_id(backend.__class__.__name__, field_name="Qiskit backend name")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _provider_job_id(provider_job: Any, *, provider_name: str) -> str:
    job_id_attr = getattr(provider_job, "job_id", None)
    raw_job_id = job_id_attr() if callable(job_id_attr) else job_id_attr
    job_id = str(raw_job_id).strip() if raw_job_id is not None else ""
    if not job_id:
        raise ValueError(f"{provider_name} job object does not expose a provider job id")
    return strict_provider_job_id(job_id, field_name=f"{provider_name} provider job id")


def _normalise_status(value: object, *, default: str = "unknown") -> str:
    text = str(value or default).split(".")[-1].strip().lower().replace(" ", "_")
    return {
        "complete": "completed",
        "completed": "completed",
        "success": "completed",
        "succeeded": "completed",
        "done": "completed",
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


__all__ = [
    "QiskitAerHALAdapter",
    "QiskitRuntimeHALAdapter",
    "qiskit_circuit_to_qasm3_workload",
    "qiskit_circuit_to_workload",
]
