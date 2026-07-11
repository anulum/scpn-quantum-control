# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Phase Qiskit Bridge
"""Compatibility facade for Qiskit gradient and Runtime evidence routes.

Immutable result/evidence records, provider-method registries, validation, and
serialization live in :mod:`.qiskit_bridge_contracts`; shifted-circuit generation
and local deterministic/finite-shot gradients live in :mod:`.qiskit_gradients`.
This module retains Runtime capture builders and maturity orchestration while those
responsibilities undergo bounded decomposition.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

from ..differentiable import ParameterShiftRule
from .provider_hardware_gradient_audit import (
    ProviderHardwareGradientPreparationAuditResult,
    run_provider_hardware_gradient_preparation_audit,
)
from .qiskit_bridge_contracts import (
    _QISKIT_PROVIDER_GRADIENT_METHOD_COMMON_METADATA_KEYS as _QISKIT_PROVIDER_GRADIENT_METHOD_COMMON_METADATA_KEYS,
)
from .qiskit_bridge_contracts import (
    _QISKIT_PROVIDER_GRADIENT_METHOD_SCHEMAS as _QISKIT_PROVIDER_GRADIENT_METHOD_SCHEMAS,
)
from .qiskit_bridge_contracts import (
    _QISKIT_PROVIDER_GRADIENT_METHOD_SPECIFIC_METADATA_KEYS as _QISKIT_PROVIDER_GRADIENT_METHOD_SPECIFIC_METADATA_KEYS,
)
from .qiskit_bridge_contracts import (
    QISKIT_PROVIDER_EVIDENCE_REVIEW_AS_OF_UTC as QISKIT_PROVIDER_EVIDENCE_REVIEW_AS_OF_UTC,
)
from .qiskit_bridge_contracts import (
    QISKIT_PROVIDER_GRADIENT_METHODS as QISKIT_PROVIDER_GRADIENT_METHODS,
)
from .qiskit_bridge_contracts import (
    FloatArray as FloatArray,
)
from .qiskit_bridge_contracts import (
    QiskitCalibrationStatevectorComparisonArtifact as QiskitCalibrationStatevectorComparisonArtifact,
)
from .qiskit_bridge_contracts import (
    QiskitMaturityAuditResult as QiskitMaturityAuditResult,
)
from .qiskit_bridge_contracts import (
    QiskitParameterShiftGradientResult as QiskitParameterShiftGradientResult,
)
from .qiskit_bridge_contracts import (
    QiskitParameterShiftRecord as QiskitParameterShiftRecord,
)
from .qiskit_bridge_contracts import (
    QiskitProviderGradientWorkflowArtifact as QiskitProviderGradientWorkflowArtifact,
)
from .qiskit_bridge_contracts import (
    QiskitRawCountReplayArtifact as QiskitRawCountReplayArtifact,
)
from .qiskit_bridge_contracts import (
    QiskitRuntimePrimitiveExecutionArtifact as QiskitRuntimePrimitiveExecutionArtifact,
)
from .qiskit_bridge_contracts import (
    QiskitRuntimeQPUExecutionArtifact as QiskitRuntimeQPUExecutionArtifact,
)
from .qiskit_bridge_contracts import (
    QiskitRuntimeQPUProviderEvidenceBundle as QiskitRuntimeQPUProviderEvidenceBundle,
)
from .qiskit_bridge_contracts import (
    _as_finite_scalar as _as_finite_scalar,
)
from .qiskit_bridge_contracts import (
    _as_finite_vector as _as_finite_vector,
)
from .qiskit_bridge_contracts import (
    _as_positive_scalar as _as_positive_scalar,
)
from .qiskit_bridge_contracts import (
    _normalise_metadata_text as _normalise_metadata_text,
)
from .qiskit_bridge_contracts import (
    _normalise_non_negative_int as _normalise_non_negative_int,
)
from .qiskit_bridge_contracts import (
    _normalise_positive_int as _normalise_positive_int,
)
from .qiskit_bridge_contracts import (
    _normalise_provider_gradient_method_metadata as _normalise_provider_gradient_method_metadata,
)
from .qiskit_bridge_contracts import (
    _normalise_qiskit_provider_gradient_method as _normalise_qiskit_provider_gradient_method,
)
from .qiskit_bridge_contracts import (
    _normalise_qiskit_runtime_primitive as _normalise_qiskit_runtime_primitive,
)
from .qiskit_bridge_contracts import (
    _normalise_sha256_metadata_digest as _normalise_sha256_metadata_digest,
)
from .qiskit_bridge_contracts import (
    _normalise_shots as _normalise_shots,
)
from .qiskit_bridge_contracts import (
    _normalise_utc_timestamp as _normalise_utc_timestamp,
)
from .qiskit_bridge_contracts import (
    _require_matching_evidence_field as _require_matching_evidence_field,
)
from .qiskit_bridge_contracts import (
    _require_provider_gradient_method_metadata_value as _require_provider_gradient_method_metadata_value,
)
from .qiskit_bridge_contracts import (
    _result_to_dict as _result_to_dict,
)
from .qiskit_bridge_contracts import (
    _utc_timestamp as _utc_timestamp,
)
from .qiskit_bridge_contracts import (
    _validate_provider_gradient_method_metadata_consistency as _validate_provider_gradient_method_metadata_consistency,
)
from .qiskit_bridge_contracts import (
    _validate_runtime_qpu_evidence_chain as _validate_runtime_qpu_evidence_chain,
)
from .qiskit_bridge_contracts import (
    _validate_runtime_qpu_mode as _validate_runtime_qpu_mode,
)
from .qiskit_bridge_contracts import (
    _validate_sha256_digest as _validate_sha256_digest,
)
from .qiskit_gradients import (
    _bind_circuit as _bind_circuit,
)
from .qiskit_gradients import (
    _expectation as _expectation,
)
from .qiskit_gradients import (
    _expectation_and_variance as _expectation_and_variance,
)
from .qiskit_gradients import (
    _normalise_parameters as _normalise_parameters,
)
from .qiskit_gradients import (
    _parameter_shift_terms as _parameter_shift_terms,
)
from .qiskit_gradients import (
    _validate_circuit_parameters as _validate_circuit_parameters,
)
from .qiskit_gradients import (
    execute_qiskit_finite_shot_parameter_shift as execute_qiskit_finite_shot_parameter_shift,
)
from .qiskit_gradients import (
    execute_qiskit_statevector_parameter_shift as execute_qiskit_statevector_parameter_shift,
)
from .qiskit_gradients import (
    generate_qiskit_parameter_shift_circuits as generate_qiskit_parameter_shift_circuits,
)


def build_qiskit_runtime_qpu_execution_artifact(
    *,
    artifact_id: str,
    provider_name: str,
    primitive_name: str,
    backend_name: str,
    job_id: str,
    session_id: str | None,
    circuit_fingerprint: str,
    observable_fingerprint: str | None,
    parameter_digest: str,
    result_digest: str,
    metadata_digest: str,
    transpiled_circuit_digest: str,
    live_execution_ticket: str,
    backend_allowlist_id: str,
    shot_budget_id: str,
    runtime_session_mode: str,
    shots: int,
) -> QiskitRuntimeQPUExecutionArtifact:
    """Build Runtime QPU evidence from captured EstimatorV2/SamplerV2 metadata.

    The helper is intentionally no-submit: callers must provide digests and
    identifiers captured from an approved Qiskit Runtime execution. The returned
    artefact validates the Runtime primitive family, backend/session mode,
    positive shot count, ticket/allowlist/budget IDs, and SHA-256 digests before
    it can be attached to the Qiskit maturity audit.
    """
    return QiskitRuntimeQPUExecutionArtifact(
        artifact_id=artifact_id,
        provider_name=provider_name,
        primitive_name=primitive_name,
        backend_name=backend_name,
        job_id=job_id,
        session_id=session_id,
        circuit_fingerprint=circuit_fingerprint,
        observable_fingerprint=observable_fingerprint,
        parameter_digest=parameter_digest,
        result_digest=result_digest,
        metadata_digest=metadata_digest,
        transpiled_circuit_digest=transpiled_circuit_digest,
        live_execution_ticket=live_execution_ticket,
        backend_allowlist_id=backend_allowlist_id,
        shot_budget_id=shot_budget_id,
        runtime_session_mode=runtime_session_mode,
        shots=shots,
        hardware_execution=True,
    )


def build_qiskit_provider_gradient_workflow_artifact(
    *,
    artifact_id: str,
    provider_name: str,
    backend_name: str,
    job_id: str,
    primitive_name: str,
    gradient_method: str,
    circuit_fingerprint: str,
    observable_fingerprint: str | None,
    parameter_digest: str,
    gradient_digest: str,
    metadata_digest: str,
    shots: int,
    parameter_count: int,
    gradient_dimension: int,
    hardware_execution: bool,
    live_ticket_id: str,
    method_metadata: Mapping[str, object] | None = None,
    claim_boundary: str = "qiskit_provider_gradient_workflow_capture",
) -> QiskitProviderGradientWorkflowArtifact:
    """Build no-submit evidence for a captured Qiskit provider-gradient workflow.

    Parameters
    ----------
    artifact_id:
        Stable evidence identifier for the captured workflow.
    provider_name, backend_name, job_id:
        Runtime provider, backend, and job identifiers from the captured run.
    primitive_name:
        Runtime primitive family, normalised to ``EstimatorV2`` or
        ``SamplerV2``.
    gradient_method:
        Captured gradient method. Accepted values are ``parameter_shift``,
        ``finite_difference``, ``lcu``, ``spsa``, ``qgt``, and ``qfi``.
    circuit_fingerprint, observable_fingerprint:
        Circuit identity and optional observable identity. Estimator workflows
        require an observable fingerprint; Sampler workflows reject one.
    parameter_digest, gradient_digest, metadata_digest:
        ``sha256:<64-hex>`` digests for the parameter payload, gradient payload,
        and workflow metadata payload.
    shots:
        Positive Runtime shot count for the captured workflow.
    parameter_count, gradient_dimension:
        Positive parameter width and gradient width; they must match.
    hardware_execution:
        Must be true because this artefact represents captured live Runtime
        provider-gradient evidence.
    live_ticket_id:
        Approval ticket for the captured live workflow.
    method_metadata:
        Method-specific provenance for the captured gradient workflow. Each
        method requires a matching schema name, artifact ID, workflow version,
        and method-specific fields: shift-rule metadata for parameter-shift,
        stencil metadata for finite-difference, generator metadata for LCU,
        perturbation metadata for SPSA, and matrix metadata for QGT/QFI.
    claim_boundary:
        Explicit boundary label retained in serialized evidence.

    Returns
    -------
    QiskitProviderGradientWorkflowArtifact
        Validated workflow metadata ready for maturity-audit attachment.

    Raises
    ------
    ValueError
        If primitive, method, digest, shot, dimension, observable, live-ticket,
        or hardware-execution evidence is malformed.
    """
    return QiskitProviderGradientWorkflowArtifact(
        artifact_id=artifact_id,
        provider_name=provider_name,
        backend_name=backend_name,
        job_id=job_id,
        primitive_name=primitive_name,
        gradient_method=gradient_method,
        circuit_fingerprint=circuit_fingerprint,
        observable_fingerprint=observable_fingerprint,
        parameter_digest=parameter_digest,
        gradient_digest=gradient_digest,
        metadata_digest=metadata_digest,
        shots=shots,
        parameter_count=parameter_count,
        gradient_dimension=gradient_dimension,
        hardware_execution=hardware_execution,
        live_ticket_id=live_ticket_id,
        method_metadata={} if method_metadata is None else dict(method_metadata),
        claim_boundary=claim_boundary,
    )


def build_qiskit_runtime_qpu_provider_evidence_bundle(
    *,
    artifact_id: str,
    runtime_qpu_execution_artifact: QiskitRuntimeQPUExecutionArtifact,
    raw_count_replay_artifact: QiskitRawCountReplayArtifact,
    calibration_comparison_artifact: QiskitCalibrationStatevectorComparisonArtifact,
    captured_at_utc: str,
    valid_until_utc: str,
    isolated_benchmark_artifact_id: str | None = None,
) -> QiskitRuntimeQPUProviderEvidenceBundle:
    """Build a no-submit Qiskit Runtime QPU provider evidence bundle.

    The bundle ties one Runtime QPU execution artefact to its matching
    raw-count replay and calibration/statevector comparison artefacts. An
    isolated benchmark artefact ID may be attached when benchmark evidence has
    been produced under the repository benchmark-isolation policy. Capture and
    expiry timestamps bound the review window used by provider-exceedance
    audits.
    """
    return QiskitRuntimeQPUProviderEvidenceBundle(
        artifact_id=artifact_id,
        runtime_qpu_execution_artifact=runtime_qpu_execution_artifact,
        raw_count_replay_artifact=raw_count_replay_artifact,
        calibration_comparison_artifact=calibration_comparison_artifact,
        captured_at_utc=captured_at_utc,
        valid_until_utc=valid_until_utc,
        isolated_benchmark_artifact_id=isolated_benchmark_artifact_id,
    )


def run_qiskit_maturity_audit(
    circuit: QuantumCircuit,
    observable: object,
    parameters: Sequence[Parameter],
    values: ArrayLike,
    *,
    shots: int,
    rule: ParameterShiftRule | None = None,
    shift: float = float(np.pi / 2.0),
    confidence_level: float = 0.95,
    confidence_z: float = 1.959963984540054,
    provider_preparation_audit: ProviderHardwareGradientPreparationAuditResult | None = None,
    runtime_primitive_artifact: QiskitRuntimePrimitiveExecutionArtifact | None = None,
    runtime_qpu_execution_artifact: QiskitRuntimeQPUExecutionArtifact | None = None,
    raw_count_replay_artifact: QiskitRawCountReplayArtifact | None = None,
    calibration_comparison_artifact: QiskitCalibrationStatevectorComparisonArtifact | None = None,
    qpu_provider_evidence_bundle: QiskitRuntimeQPUProviderEvidenceBundle | None = None,
    provider_gradient_workflow_artifacts: Sequence[QiskitProviderGradientWorkflowArtifact]
    | None = None,
    evidence_freshness_as_of_utc: str = QISKIT_PROVIDER_EVIDENCE_REVIEW_AS_OF_UTC,
) -> QiskitMaturityAuditResult:
    """Aggregate Qiskit local-gradient evidence and provider-level blockers.

    The audit records fully bound shifted-circuit generation, deterministic
    local statevector reference gradients, finite-shot surrogate uncertainty,
    and the no-submit provider hardware-gradient preparation audit. Runtime QPU
    evidence, raw-count replay, and calibration comparison artefacts may be
    attached only as a matching live execution chain. The audit does not create
    promotion-grade isolated benchmark evidence; it only records supplied
    artifact IDs and keeps the corresponding gate blocked when evidence is
    absent.
    """
    parameter_tuple = _normalise_parameters(parameters)
    values_vector = _as_finite_vector("values", values, width=len(parameter_tuple))
    shot_count = _normalise_shots(shots)
    isolated_benchmark_artifact_id: str | None = None
    if qpu_provider_evidence_bundle is not None:
        if (
            runtime_qpu_execution_artifact is not None
            or raw_count_replay_artifact is not None
            or calibration_comparison_artifact is not None
        ):
            raise ValueError(
                "qpu_provider_evidence_bundle cannot be combined with individual QPU artefacts"
            )
        _validate_qiskit_provider_evidence_bundle_freshness(
            qpu_provider_evidence_bundle,
            as_of_utc=evidence_freshness_as_of_utc,
        )
        runtime_qpu_execution_artifact = (
            qpu_provider_evidence_bundle.runtime_qpu_execution_artifact
        )
        raw_count_replay_artifact = qpu_provider_evidence_bundle.raw_count_replay_artifact
        calibration_comparison_artifact = (
            qpu_provider_evidence_bundle.calibration_comparison_artifact
        )
        isolated_benchmark_artifact_id = (
            qpu_provider_evidence_bundle.isolated_benchmark_artifact_id
        )
    _validate_runtime_qpu_evidence_chain(
        runtime_qpu_execution_artifact=runtime_qpu_execution_artifact,
        raw_count_replay_artifact=raw_count_replay_artifact,
        calibration_comparison_artifact=calibration_comparison_artifact,
    )
    provider_gradient_workflow_tuple = _normalise_provider_gradient_workflow_artifacts(
        provider_gradient_workflow_artifacts,
        runtime_qpu_execution_artifact=runtime_qpu_execution_artifact,
    )
    shifted_records = generate_qiskit_parameter_shift_circuits(
        circuit,
        parameter_tuple,
        values_vector,
        rule=rule,
        shift=shift,
    )
    statevector_reference = execute_qiskit_statevector_parameter_shift(
        circuit,
        observable,
        parameter_tuple,
        values_vector,
        rule=rule,
        shift=shift,
    )
    finite_shot_surrogate = execute_qiskit_finite_shot_parameter_shift(
        circuit,
        observable,
        parameter_tuple,
        values_vector,
        shots=shot_count,
        rule=rule,
        shift=shift,
        confidence_level=confidence_level,
        confidence_z=confidence_z,
    )
    preparation_audit = (
        provider_preparation_audit
        if provider_preparation_audit is not None
        else run_provider_hardware_gradient_preparation_audit()
    )
    delta = statevector_reference.gradient - finite_shot_surrogate.gradient
    max_abs_error = float(np.max(np.abs(delta))) if delta.size else 0.0
    local_reference_metadata: dict[str, object] = {
        "parameter_count": len(parameter_tuple),
        "shifted_record_count": len(shifted_records),
        "statevector_evaluations": statevector_reference.evaluations,
        "finite_shot_evaluations": finite_shot_surrogate.total_evaluations,
        "shots": shot_count,
        "finite_shot_total_shots": finite_shot_surrogate.total_shots,
        "statevector_finite_shot_max_abs_error": max_abs_error,
        "provider_preparation_record_count": preparation_audit.record_count,
        "provider_preparation_approved_count": preparation_audit.approved_count,
        "provider_preparation_blocked_count": preparation_audit.blocked_count,
        "provider_preparation_hardware_execution_count": (
            preparation_audit.hardware_execution_count
        ),
        "provider_preparation_gradient_available_count": (
            preparation_audit.gradient_available_count
        ),
    }
    if runtime_primitive_artifact is not None:
        local_reference_metadata["runtime_primitive_artifact_id"] = (
            runtime_primitive_artifact.artifact_id
        )
        local_reference_metadata["runtime_primitive_name"] = (
            runtime_primitive_artifact.primitive_name
        )
        local_reference_metadata["runtime_primitive_backend_name"] = (
            runtime_primitive_artifact.backend_name
        )
    if runtime_qpu_execution_artifact is not None:
        local_reference_metadata["runtime_qpu_execution_artifact_id"] = (
            runtime_qpu_execution_artifact.artifact_id
        )
        local_reference_metadata["runtime_qpu_primitive_name"] = (
            runtime_qpu_execution_artifact.primitive_name
        )
        local_reference_metadata["runtime_qpu_backend_name"] = (
            runtime_qpu_execution_artifact.backend_name
        )
        local_reference_metadata["runtime_qpu_shots"] = runtime_qpu_execution_artifact.shots
    if raw_count_replay_artifact is not None:
        local_reference_metadata["raw_count_replay_artifact_id"] = (
            raw_count_replay_artifact.artifact_id
        )
        local_reference_metadata["raw_count_replay_shots"] = raw_count_replay_artifact.shots
        local_reference_metadata["raw_count_replay_standard_error"] = (
            raw_count_replay_artifact.standard_error
        )
    if calibration_comparison_artifact is not None:
        local_reference_metadata["calibration_comparison_artifact_id"] = (
            calibration_comparison_artifact.artifact_id
        )
        local_reference_metadata["calibration_comparison_max_abs_error"] = (
            calibration_comparison_artifact.max_abs_error
        )
        local_reference_metadata["calibration_comparison_tolerance"] = (
            calibration_comparison_artifact.tolerance
        )
    if qpu_provider_evidence_bundle is not None:
        local_reference_metadata["qpu_provider_evidence_bundle_id"] = (
            qpu_provider_evidence_bundle.artifact_id
        )
        local_reference_metadata["qpu_provider_evidence_captured_at_utc"] = (
            qpu_provider_evidence_bundle.captured_at_utc
        )
        local_reference_metadata["qpu_provider_evidence_valid_until_utc"] = (
            qpu_provider_evidence_bundle.valid_until_utc
        )
    if isolated_benchmark_artifact_id is not None:
        local_reference_metadata["isolated_benchmark_artifact_id"] = isolated_benchmark_artifact_id
    if provider_gradient_workflow_tuple:
        local_reference_metadata["provider_gradient_workflow_artifact_count"] = len(
            provider_gradient_workflow_tuple
        )
        local_reference_metadata["provider_gradient_workflow_methods"] = tuple(
            sorted(artifact.gradient_method for artifact in provider_gradient_workflow_tuple)
        )
        local_reference_metadata["provider_gradient_workflow_method_schemas"] = tuple(
            sorted(
                f"{artifact.gradient_method}:{artifact.method_metadata['method_schema']}"
                for artifact in provider_gradient_workflow_tuple
            )
        )
    evidence: dict[str, object] = {
        "shifted_circuit_records": shifted_records,
        "statevector_reference": statevector_reference,
        "finite_shot_surrogate": finite_shot_surrogate,
        "provider_preparation_audit": preparation_audit,
        "runtime_primitive_artifact": runtime_primitive_artifact,
        "runtime_qpu_execution_artifact": runtime_qpu_execution_artifact,
        "raw_count_replay_artifact": raw_count_replay_artifact,
        "calibration_comparison_artifact": calibration_comparison_artifact,
        "qpu_provider_evidence_bundle": qpu_provider_evidence_bundle,
        "provider_gradient_workflow_artifacts": provider_gradient_workflow_tuple,
    }
    statevector_comparison_passed = max_abs_error <= 1e-10
    provider_policy_passed = (
        preparation_audit.passed
        and preparation_audit.hardware_execution_count == 0
        and preparation_audit.gradient_available_count == 0
    )
    required_capabilities = {
        "shifted_circuit_generation": "passed" if shifted_records else "failed",
        "statevector_reference": "passed"
        if statevector_reference.gradient.shape == values_vector.shape
        else "failed",
        "finite_shot_surrogate_uncertainty": "passed"
        if finite_shot_surrogate.gradient.shape == values_vector.shape
        else "failed",
        "statevector_reference_comparison": (
            "passed" if statevector_comparison_passed else "failed"
        ),
        "provider_hardware_preparation_policy": ("passed" if provider_policy_passed else "failed"),
        "backend_allowlist_policy": "passed" if provider_policy_passed else "failed",
        "calibration_snapshot_policy": "passed" if provider_policy_passed else "failed",
        "runtime_primitive_execution_evidence": (
            "passed"
            if runtime_primitive_artifact is not None or runtime_qpu_execution_artifact is not None
            else "blocked"
        ),
        "live_qpu_execution_ticket": (
            "passed" if runtime_qpu_execution_artifact is not None else "blocked"
        ),
        "raw_count_capture_replay_harness": (
            "passed" if raw_count_replay_artifact is not None else "blocked"
        ),
        "live_backend_statevector_reference_comparison": (
            "passed" if calibration_comparison_artifact is not None else "blocked"
        ),
        "promotion_grade_isolated_benchmarks": (
            "passed" if isolated_benchmark_artifact_id is not None else "blocked"
        ),
        "provider_gradient_workflow_evidence": (
            "passed" if provider_gradient_workflow_tuple else "blocked"
        ),
    }
    local_gradient_ready = all(
        required_capabilities[name] == "passed"
        for name in (
            "shifted_circuit_generation",
            "statevector_reference",
            "finite_shot_surrogate_uncertainty",
            "statevector_reference_comparison",
            "provider_hardware_preparation_policy",
            "backend_allowlist_policy",
            "calibration_snapshot_policy",
        )
    )
    open_gaps = tuple(name for name, status in required_capabilities.items() if status != "passed")
    return QiskitMaturityAuditResult(
        local_gradient_ready=local_gradient_ready,
        ready_for_provider_exceedance=local_gradient_ready and not open_gaps,
        evidence=evidence,
        required_capabilities=required_capabilities,
        local_reference_metadata=local_reference_metadata,
        open_gaps=open_gaps,
    )


def _validate_qiskit_provider_evidence_bundle_freshness(
    qpu_provider_evidence_bundle: QiskitRuntimeQPUProviderEvidenceBundle,
    *,
    as_of_utc: str,
) -> None:
    valid_until = _utc_timestamp(
        "qpu_provider_evidence_bundle.valid_until_utc",
        qpu_provider_evidence_bundle.valid_until_utc,
    )
    if valid_until <= _utc_timestamp("evidence_freshness_as_of_utc", as_of_utc):
        raise ValueError(
            "qpu_provider_evidence_bundle.valid_until_utc is stale for the review cutoff"
        )


def _normalise_provider_gradient_workflow_artifacts(
    artifacts: Sequence[QiskitProviderGradientWorkflowArtifact] | None,
    *,
    runtime_qpu_execution_artifact: QiskitRuntimeQPUExecutionArtifact | None,
) -> tuple[QiskitProviderGradientWorkflowArtifact, ...]:
    if artifacts is None:
        return ()
    artifact_tuple = tuple(artifacts)
    if not artifact_tuple:
        return ()
    methods = frozenset(artifact.gradient_method for artifact in artifact_tuple)
    if methods != QISKIT_PROVIDER_GRADIENT_METHODS:
        missing = sorted(QISKIT_PROVIDER_GRADIENT_METHODS - methods)
        extra = sorted(methods - QISKIT_PROVIDER_GRADIENT_METHODS)
        raise ValueError(
            "provider gradient workflow methods must cover "
            f"{sorted(QISKIT_PROVIDER_GRADIENT_METHODS)}; missing={missing}; extra={extra}"
        )
    if len(methods) != len(artifact_tuple):
        raise ValueError("provider gradient workflow methods must not contain duplicates")
    artifact_ids = frozenset(artifact.artifact_id for artifact in artifact_tuple)
    if len(artifact_ids) != len(artifact_tuple):
        raise ValueError("provider gradient workflow artifact_id values must be unique")
    if runtime_qpu_execution_artifact is not None:
        for artifact in artifact_tuple:
            _validate_provider_gradient_workflow_chain(
                artifact,
                runtime_qpu_execution_artifact=runtime_qpu_execution_artifact,
            )
    return artifact_tuple


def _validate_provider_gradient_workflow_chain(
    artifact: QiskitProviderGradientWorkflowArtifact,
    *,
    runtime_qpu_execution_artifact: QiskitRuntimeQPUExecutionArtifact,
) -> None:
    _require_matching_evidence_field(
        "provider_gradient_workflow_artifact.provider_name",
        artifact.provider_name,
        runtime_qpu_execution_artifact.provider_name,
    )
    _require_matching_evidence_field(
        "provider_gradient_workflow_artifact.backend_name",
        artifact.backend_name,
        runtime_qpu_execution_artifact.backend_name,
    )
    _require_matching_evidence_field(
        "provider_gradient_workflow_artifact.job_id",
        artifact.job_id,
        runtime_qpu_execution_artifact.job_id,
    )
    _require_matching_evidence_field(
        "provider_gradient_workflow_artifact.primitive_name",
        artifact.primitive_name,
        runtime_qpu_execution_artifact.primitive_name,
    )
    _require_matching_evidence_field(
        "provider_gradient_workflow_artifact.circuit_fingerprint",
        artifact.circuit_fingerprint,
        runtime_qpu_execution_artifact.circuit_fingerprint,
    )
    _require_matching_optional_evidence_field(
        "provider_gradient_workflow_artifact.observable_fingerprint",
        artifact.observable_fingerprint,
        runtime_qpu_execution_artifact.observable_fingerprint,
    )
    _require_matching_evidence_field(
        "provider_gradient_workflow_artifact.parameter_digest",
        artifact.parameter_digest,
        runtime_qpu_execution_artifact.parameter_digest,
    )
    _require_matching_evidence_field(
        "provider_gradient_workflow_artifact.live_ticket_id",
        artifact.live_ticket_id,
        runtime_qpu_execution_artifact.live_execution_ticket,
    )
    if artifact.shots != runtime_qpu_execution_artifact.shots:
        raise ValueError("provider_gradient_workflow_artifact.shots must match Runtime QPU shots")


def _require_matching_optional_evidence_field(
    field_name: str,
    observed: str | None,
    expected: str | None,
) -> None:
    if observed != expected:
        raise ValueError(f"{field_name} must match Runtime QPU evidence")


__all__ = [
    "QiskitCalibrationStatevectorComparisonArtifact",
    "QiskitMaturityAuditResult",
    "QiskitParameterShiftGradientResult",
    "QiskitParameterShiftRecord",
    "QiskitProviderGradientWorkflowArtifact",
    "QiskitRawCountReplayArtifact",
    "QiskitRuntimePrimitiveExecutionArtifact",
    "QiskitRuntimeQPUProviderEvidenceBundle",
    "QiskitRuntimeQPUExecutionArtifact",
    "build_qiskit_provider_gradient_workflow_artifact",
    "build_qiskit_runtime_qpu_execution_artifact",
    "build_qiskit_runtime_qpu_provider_evidence_bundle",
    "execute_qiskit_finite_shot_parameter_shift",
    "execute_qiskit_statevector_parameter_shift",
    "generate_qiskit_parameter_shift_circuits",
    "run_qiskit_maturity_audit",
]
