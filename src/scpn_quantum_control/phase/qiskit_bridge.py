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
Runtime capture builders, provider evidence assembly, and maturity orchestration live
in :mod:`.qiskit_runtime`. This module is the shallow compatibility facade that
re-exports those exact public and private objects.
"""

from __future__ import annotations

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
from .qiskit_runtime import (
    _normalise_provider_gradient_workflow_artifacts as _normalise_provider_gradient_workflow_artifacts,
)
from .qiskit_runtime import (
    _require_matching_optional_evidence_field as _require_matching_optional_evidence_field,
)
from .qiskit_runtime import (
    _validate_provider_gradient_workflow_chain as _validate_provider_gradient_workflow_chain,
)
from .qiskit_runtime import (
    _validate_qiskit_provider_evidence_bundle_freshness as _validate_qiskit_provider_evidence_bundle_freshness,
)
from .qiskit_runtime import (
    build_qiskit_provider_gradient_workflow_artifact as build_qiskit_provider_gradient_workflow_artifact,
)
from .qiskit_runtime import (
    build_qiskit_runtime_qpu_execution_artifact as build_qiskit_runtime_qpu_execution_artifact,
)
from .qiskit_runtime import (
    build_qiskit_runtime_qpu_provider_evidence_bundle as build_qiskit_runtime_qpu_provider_evidence_bundle,
)
from .qiskit_runtime import (
    run_qiskit_maturity_audit as run_qiskit_maturity_audit,
)

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
