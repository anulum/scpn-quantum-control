# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — HAL iqm module
# scpn-quantum-control -- IQM adapter for the hardware HAL
"""IQM Qiskit adapter for :mod:`scpn_quantum_control.hardware.hal`."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from importlib import import_module
from typing import Any

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter

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
from .hal_qiskit import _circuit_to_qpy_b64, qiskit_circuit_to_workload
from .iqm_backend import IQMTargetCompilationError
from .provider_measurement import bind_qiskit_workload, qiskit_submission_semantics
from .provider_semantics import GateModelObservation

IQM_EXECUTION_MODE = "iqm_qiskit"


def iqm_qiskit_workload(
    circuit: QuantumCircuit,
    *,
    workload_id: str,
    shots: int,
    metadata: dict[str, object] | None = None,
    capture_semantics: bool = False,
    requested_target: str | None = None,
    parameter_bindings: Mapping[Parameter, float] | None = None,
) -> QuantumWorkload:
    """Encode unchanged native QPY and an optional IQM sampling contract.

    Parameters
    ----------
    circuit
        Original Qiskit circuit, retaining native shared parameter UUIDs in QPY.
    workload_id
        Stable caller identity for the original encoded source.
    shots
        Positive integral requested sample total.
    metadata
        Scalar annotations separate from source and execution settings.
    capture_semantics
        Capture static measurement map, register order, original parameter
        uses and source digest. Default false preserves legacy construction.
    requested_target
        Optional exact native backend name, requiring capture.
    parameter_bindings
        Finite real values keyed by actual original native Parameters, requiring
        capture. Submission requires every free parameter and binds a copy.

    Returns
    -------
    QuantumWorkload
        Original QPY and optional separate versioned sampling companion.

    Raises
    ------
    TypeError
        If the source is not a native Qiskit circuit.
    ValueError
        If capture, static-subset, source or binding admission fails.

    """
    return qiskit_circuit_to_workload(
        circuit,
        workload_id=workload_id,
        shots=shots,
        metadata=metadata,
        capture_semantics=capture_semantics,
        requested_target=requested_target,
        parameter_bindings=parameter_bindings,
    )


class IQMHALAdapter:
    """Execute original QPY through an explicitly configured IQM-compatible client.

    Parameters
    ----------
    profile
        IQM Cloud route with declared resource and IR admission.
    backend
        Optional injected native-compatible backend. Absence uses the explicit
        server URL; decoded source and bindings qualify before loading a client.
    server_url
        Required configured IQM endpoint when no backend is injected.
    quantum_computer
        Optional configured native computer selector, recorded separately from
        the actual backend name used for target admission.
    import_module
        Lazy SDK importer; this adapter never installs an optional dependency.
    timeout_s
        Positive provider result-retrieval timeout in seconds, default 600.
    optimisation_level
        Qiskit target-compilation level, one of 0, 1, 2 or 3.
    compile_circuit
        Compile for the selected native target when true. False retains an
        explicit caller-precompiled boundary without claiming target compilation.

    """

    supports_provider_semantics = True

    def __init__(
        self,
        profile: BackendProfile,
        *,
        backend: Any | None = None,
        server_url: str | None = None,
        quantum_computer: str | None = None,
        import_module: Callable[[str], Any] = import_module,
        timeout_s: float = 600.0,
        optimisation_level: int = 1,
        compile_circuit: bool = True,
    ) -> None:
        if profile.backend_id != "iqm_cloud":
            raise ValueError("IQMHALAdapter requires the iqm_cloud profile")
        if timeout_s <= 0.0:
            raise ValueError("timeout_s must be positive")
        if optimisation_level not in {0, 1, 2, 3}:
            raise ValueError("optimisation_level must be 0, 1, 2, or 3")
        self.profile = profile
        self.backend_id = profile.backend_id
        self._backend = backend
        self._server_url = server_url
        self._quantum_computer = (
            strict_provider_job_id(quantum_computer, field_name="IQM quantum computer")
            if quantum_computer is not None
            else None
        )
        self._import_module = import_module
        self.timeout_s = timeout_s
        self.optimisation_level = optimisation_level
        self._compile_circuit = compile_circuit
        self._jobs: dict[str, QuantumJobRef] = {}
        self._provider_jobs: dict[str, Any] = {}
        self._results: dict[str, QuantumJobResult] = {}

    def submit(
        self, workload: QuantumWorkload, *, approval_id: str | None = None
    ) -> QuantumJobRef:
        """Admit original source and bindings before client construction or execution.

        Parameters
        ----------
        workload
            Original QPY and optional static native request. Metadata cannot
            replace source, target, shot settings or adapter-owned fields.
        approval_id
            Required caller authorization stored with the submitted job.

        Returns
        -------
        QuantumJobRef
            Submitted handle preserving original payload, shared identities,
            requested/effective shots and actual target. Captured compilation
            has a digest; caller-precompiled mode is labelled separately.

        Raises
        ------
        PermissionError
            If approval is absent.
        ValueError
            If profile, source, IR, bindings or selected target disagree.
        IQMTargetCompilationError
            If target compilation fails; the original cause remains available
            and no target-free retry or provider run follows.
        ImportError
            If lazy optional IQM SDK loading fails.

        """
        if not approval_id:
            raise PermissionError("approval_id is required for IQM submission")
        _validate_workload_for_profile(self.profile, workload)
        if workload.ir_format != "qiskit_qpy":
            raise ValueError("IQM direct adapter requires qiskit_qpy workloads")
        original = _workload_to_qiskit_circuit(workload)
        bound = bind_qiskit_workload(workload, original)
        backend = self._backend_client()
        backend_name = _backend_name(backend)
        qiskit_submission_semantics(workload, original, target_name=backend_name)
        circuit = self._compile(bound, backend)
        submission = qiskit_submission_semantics(
            workload,
            original,
            target_name=backend_name,
            compiled_program=(
                _circuit_to_qpy_b64(circuit)
                if self._compile_circuit and workload.semantics is not None
                else None
            ),
            compilation="targeted" if self._compile_circuit else "caller_precompiled",
        )
        provider_job = backend.run([circuit], shots=workload.shots)
        provider_job_id = _job_id(provider_job)
        job = QuantumJobRef(
            job_id=_hal_job_id(self.backend_id, workload.workload_id, provider_job_id),
            backend_id=self.backend_id,
            workload_id=workload.workload_id,
            status="submitted",
            submission=submission,
            metadata={
                "approval_id": approval_id,
                "provider_job_id": provider_job_id,
                "execution_mode": IQM_EXECUTION_MODE,
                "backend_name": backend_name,
                "quantum_computer": self._quantum_computer,
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
        """Read native provider lifecycle after checking stored submission identity.

        Parameters
        ----------
        job
            Original or recovered handle for the same backend and workload.

        Returns
        -------
        str
            Canonical provider lifecycle or unknown when no state is exposed.

        Raises
        ------
        KeyError
            If the submission or retained provider job is unavailable.
        ValueError
            If durable identity differs from the stored submission.

        """
        provider_job = self._provider_job(job)
        status = getattr(provider_job, "status", None)
        if callable(status):
            return _normalise_status(status())
        return _normalise_status(getattr(provider_job, "_status", "unknown"))

    def result(self, job: QuantumJobRef) -> QuantumJobResult:
        """Retain native count labels against the original stored sampling request.

        Parameters
        ----------
        job
            Original or recovered handle with unchanged durable identity.

        Returns
        -------
        QuantumJobResult
            Immutable counts and optional native gate-model observation using
            stored source, measurement order and exact shot settings. Repeated
            successful retrieval returns the same retained result.

        Raises
        ------
        KeyError
            If submission or retained transport is unavailable.
        ValueError
            If identity, count labels or native shot conservation disagree.
        TypeError
            If the provider lacks result retrieval or returns an invalid channel.
        RuntimeError
            If a single-circuit result has no usable count map or several maps.

        """
        stored = self._job(job)
        cached = self._results.get(job.job_id)
        if cached is not None:
            return cached
        provider_job = self._provider_job(job)
        result_method = getattr(provider_job, "result", None)
        if not callable(result_method):
            raise TypeError("IQM provider job does not provide result()")
        provider_result = result_method(timeout=self.timeout_s)
        expected_shots = strict_integer_value(stored.metadata.get("shots", 0), field_name="shots")
        observation: GateModelObservation | None = None
        if stored.submission is not None:
            observation = GateModelObservation(
                request=stored.submission.require_gate_request(),
                raw_counts=_raw_counts(provider_result),
                shots=expected_shots,
            )
            counts = dict(observation.counts)
        else:
            counts = _extract_counts(provider_result)
        observed_shots = strict_shot_conservation(counts, expected_shots=expected_shots)
        result = QuantumJobResult(
            job=stored,
            status="completed",
            counts=counts,
            shots=observed_shots,
            provider_observation=observation,
            metadata={
                "approval_id": stored.metadata.get("approval_id"),
                "provider_job_id": stored.metadata.get("provider_job_id"),
                "execution_mode": IQM_EXECUTION_MODE,
                "backend_name": stored.metadata.get("backend_name"),
                "timestamp": _utc_now(),
            },
        )
        self._results[job.job_id] = result
        return result

    def cancel(self, job: QuantumJobRef) -> QuantumJobRef:
        """Request provider cancellation while retaining the original native contract.

        Parameters
        ----------
        job
            Handle with the exact backend and workload identity of the stored job.

        Returns
        -------
        QuantumJobRef
            Cancellation-request handle retaining original metadata and optional
            submission companion. This legacy lifecycle annotation does not
            attest that the physical provider cancelled the work.

        Raises
        ------
        KeyError
            If the original submission or retained provider job is unavailable.
        ValueError
            If identity differs or the provider has no cancellation operation.

        """
        stored = self._job(job)
        provider_job = self._provider_job(job)
        cancel = getattr(provider_job, "cancel", None)
        if not callable(cancel):
            raise ValueError("IQM provider job does not support cancellation")
        cancel()
        cancelled = QuantumJobRef(
            job_id=stored.job_id,
            backend_id=stored.backend_id,
            workload_id=stored.workload_id,
            status="cancelled",
            metadata=stored.metadata,
            submission=stored.submission,
        )
        self._jobs[job.job_id] = cancelled
        return cancelled

    def _backend_client(self) -> Any:
        if self._backend is not None:
            return self._backend
        if not self._server_url:
            raise RuntimeError("server_url is required when an IQM backend is not injected")
        try:
            provider_module = self._import_module("iqm.qiskit_iqm.iqm_provider")
        except ModuleNotFoundError as exc:
            raise ImportError(
                "iqm-client[qiskit] is required for IQM HAL execution; install it in "
                "an isolated runner environment such as `.venv-iqm` because current "
                "IQM client releases pin Qiskit below the repository's main Qiskit floor."
            ) from exc
        provider = provider_module.IQMProvider(
            self._server_url,
            quantum_computer=self._quantum_computer,
        )
        get_backend = getattr(provider, "get_backend", None)
        self._backend = get_backend() if callable(get_backend) else provider.backend()
        return self._backend

    def _compile(self, circuit: QuantumCircuit, backend: Any) -> QuantumCircuit:
        if not self._compile_circuit:
            return circuit
        try:
            return transpile(circuit, backend=backend, optimization_level=self.optimisation_level)
        except Exception as exc:
            raise IQMTargetCompilationError(
                "circuit cannot be compiled for the selected IQM target"
            ) from exc

    def _job(self, job: QuantumJobRef) -> QuantumJobRef:
        return _resolve_stored_job(job, self._jobs)

    def _provider_job(self, job: QuantumJobRef) -> Any:
        self._job(job)
        return self._provider_jobs[job.job_id]


def _workload_to_qiskit_circuit(workload: QuantumWorkload) -> QuantumCircuit:
    from .hal_qiskit import _workload_to_qiskit_circuit as decode

    return decode(workload)


def _extract_counts(result: Any) -> dict[str, int]:
    return _normalise_counts(_raw_counts(result))


def _raw_counts(result: Any) -> dict[Any, Any]:
    get_counts = getattr(result, "get_counts", None)
    if callable(get_counts):
        try:
            raw = get_counts()
        except TypeError:
            raw = get_counts(0)
        if isinstance(raw, list):
            if len(raw) != 1:
                raise RuntimeError("IQM single-circuit execution returned multiple count maps")
            raw = raw[0]
        if not isinstance(raw, dict):
            raise TypeError("IQM counts must be a mapping")
        return raw
    results = getattr(result, "results", None)
    if isinstance(results, list) and len(results) == 1:
        data = getattr(results[0], "data", None)
        counts = getattr(data, "counts", None)
        if isinstance(counts, dict):
            return counts
    raise RuntimeError("Could not extract IQM counts from backend result")


def _normalise_counts(raw: Any) -> dict[str, int]:
    if not isinstance(raw, dict):
        raise TypeError("IQM counts must be a mapping")
    counts: dict[str, int] = {}
    for bitstring, count in raw.items():
        key = strict_binary_bitstring_key(bitstring, field_name="IQM count key")
        value = strict_non_negative_count(count)
        counts[key] = counts.get(key, 0) + value
    if not counts:
        raise ValueError("IQM result did not contain any counts")
    return counts


def _backend_name(backend: Any) -> str:
    name = getattr(backend, "name", None)
    if callable(name):
        return strict_provider_job_id(name(), field_name="IQM backend name")
    if name is not None:
        return strict_provider_job_id(name, field_name="IQM backend name")
    return strict_provider_job_id(type(backend).__name__, field_name="IQM backend name")


def _job_id(job: Any) -> str:
    job_id = getattr(job, "job_id", None)
    if callable(job_id):
        return strict_provider_job_id(job_id(), field_name="IQM provider job id")
    if job_id:
        return strict_provider_job_id(job_id, field_name="IQM provider job id")
    raise ValueError("IQM backend job does not expose a provider job id")


def _hal_job_id(backend_id: str, workload_id: str, provider_job_id: str) -> str:
    digest = hashlib.sha256(provider_job_id.encode("utf-8")).hexdigest()[:12]
    return f"{backend_id}:{workload_id}:{digest}"


def _normalise_status(value: object) -> str:
    text = str(value).split(".")[-1].strip().lower().replace(" ", "_")
    return {
        "done": "completed",
        "complete": "completed",
        "completed": "completed",
        "success": "completed",
        "succeeded": "completed",
        "finished": "completed",
        "queued": "queued",
        "pending": "queued",
        "running": "running",
        "in_progress": "running",
        "in-progress": "running",
        "inprogress": "running",
        "initializing": "submitted",
        "initialising": "submitted",
        "starting": "submitted",
        "creating": "submitted",
        "created": "submitted",
        "cancelled": "cancelled",
        "canceled": "cancelled",
        "aborting": "cancelled",
        "cancelling": "cancelled",
        "canceling": "cancelled",
        "failed": "failed",
        "error": "failed",
    }.get(text, "unknown")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


__all__ = [
    "IQMHALAdapter",
    "IQM_EXECUTION_MODE",
    "iqm_qiskit_workload",
]
