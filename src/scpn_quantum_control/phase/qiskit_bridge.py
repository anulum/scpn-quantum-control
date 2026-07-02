# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Phase Qiskit Bridge
"""Qiskit parameter-shift circuit generation and local gradient execution."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Operator, Statevector

from ..differentiable import ParameterShiftRule
from .provider_gradient import (
    ProviderExpectationSample,
    ProviderGradientExecutionResult,
    execute_provider_parameter_shift_gradient,
)
from .provider_hardware_gradient_audit import (
    ProviderHardwareGradientPreparationAuditResult,
    run_provider_hardware_gradient_preparation_audit,
)

FloatArray: TypeAlias = NDArray[np.float64]
QISKIT_PROVIDER_GRADIENT_METHODS = frozenset(
    {"parameter_shift", "finite_difference", "lcu", "spsa", "qgt", "qfi"}
)
QISKIT_PROVIDER_EVIDENCE_REVIEW_AS_OF_UTC = "2026-06-27T00:00:00Z"
_QISKIT_PROVIDER_GRADIENT_METHOD_COMMON_METADATA_KEYS = frozenset(
    {"method_schema", "method_artifact_id", "workflow_version"}
)
_QISKIT_PROVIDER_GRADIENT_METHOD_SCHEMAS: dict[str, str] = {
    "parameter_shift": "parameter_shift_shift_rule",
    "finite_difference": "finite_difference_stencil",
    "lcu": "linear_combination_unitary",
    "spsa": "spsa_perturbation",
    "qgt": "quantum_geometric_tensor",
    "qfi": "quantum_fisher_information",
}
_QISKIT_PROVIDER_GRADIENT_METHOD_SPECIFIC_METADATA_KEYS: dict[str, frozenset[str]] = {
    "parameter_shift": frozenset({"shift_rule_id", "shift_count"}),
    "finite_difference": frozenset({"stencil", "step_size"}),
    "lcu": frozenset({"generator_digest", "term_count"}),
    "spsa": frozenset({"perturbation_seed", "perturbation_count"}),
    "qgt": frozenset({"qgt_digest", "matrix_dimension"}),
    "qfi": frozenset({"qfi_digest", "matrix_dimension"}),
}


@dataclass(frozen=True)
class QiskitParameterShiftRecord:
    """One Qiskit plus/minus shifted-circuit pair."""

    parameter_index: int
    shift_index: int
    shift: float
    coefficient: float
    parameter_name: str
    plus_values: FloatArray
    minus_values: FloatArray
    plus_circuit: QuantumCircuit
    minus_circuit: QuantumCircuit

    def __post_init__(self) -> None:
        """Validate shifted-circuit metadata and bound circuit values."""
        if isinstance(self.parameter_index, bool) or self.parameter_index < 0:
            raise ValueError("parameter_index must be a non-negative integer")
        if isinstance(self.shift_index, bool) or self.shift_index < 0:
            raise ValueError("shift_index must be a non-negative integer")
        shift = _as_positive_scalar("shift", self.shift)
        coefficient = _as_finite_scalar("coefficient", self.coefficient)
        if not self.parameter_name:
            raise ValueError("parameter_name must be non-empty")
        plus_values = _as_finite_vector("plus_values", self.plus_values)
        minus_values = _as_finite_vector("minus_values", self.minus_values)
        if plus_values.shape != minus_values.shape:
            raise ValueError("plus_values and minus_values must have matching shapes")
        if self.plus_circuit.num_parameters != 0 or self.minus_circuit.num_parameters != 0:
            raise ValueError("shifted Qiskit circuits must be fully bound")
        object.__setattr__(self, "shift", shift)
        object.__setattr__(self, "coefficient", coefficient)
        object.__setattr__(self, "plus_values", plus_values)
        object.__setattr__(self, "minus_values", minus_values)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible shifted-circuit metadata."""
        return {
            "parameter_index": self.parameter_index,
            "shift_index": self.shift_index,
            "shift": self.shift,
            "coefficient": self.coefficient,
            "parameter_name": self.parameter_name,
            "plus_values": self.plus_values.tolist(),
            "minus_values": self.minus_values.tolist(),
            "plus_depth": self.plus_circuit.depth(),
            "minus_depth": self.minus_circuit.depth(),
            "plus_size": self.plus_circuit.size(),
            "minus_size": self.minus_circuit.size(),
        }


@dataclass(frozen=True)
class QiskitParameterShiftGradientResult:
    """Local Qiskit Statevector parameter-shift gradient result."""

    value: float
    gradient: FloatArray
    records: tuple[QiskitParameterShiftRecord, ...]
    method: str
    evaluations: int
    claim_boundary: str

    def __post_init__(self) -> None:
        """Validate local Qiskit gradient result metadata."""
        value = _as_finite_scalar("value", self.value)
        gradient = _as_finite_vector("gradient", self.gradient)
        if self.evaluations <= 0:
            raise ValueError("evaluations must be positive")
        if not self.claim_boundary:
            raise ValueError("claim_boundary must be non-empty")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "gradient", gradient)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible Qiskit gradient metadata."""
        return {
            "value": self.value,
            "gradient": self.gradient.tolist(),
            "records": [record.to_dict() for record in self.records],
            "method": self.method,
            "evaluations": self.evaluations,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class QiskitRuntimePrimitiveExecutionArtifact:
    """Validated Qiskit Runtime primitive execution evidence."""

    artifact_id: str
    provider_name: str
    primitive_name: str
    backend_name: str
    job_id: str
    session_id: str | None
    circuit_fingerprint: str
    observable_fingerprint: str
    parameter_digest: str
    result_digest: str
    metadata_digest: str
    shots: int | None
    hardware_execution: bool = False
    raw_result_replay_artifact_id: str | None = None

    def __post_init__(self) -> None:
        """Validate no-submit Runtime primitive execution metadata."""
        for field_name in (
            "artifact_id",
            "provider_name",
            "primitive_name",
            "backend_name",
            "job_id",
            "circuit_fingerprint",
            "observable_fingerprint",
        ):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"{field_name} must be non-empty")
        if self.session_id is not None and not self.session_id.strip():
            raise ValueError("session_id must be non-empty when provided")
        if self.raw_result_replay_artifact_id is not None and not (
            self.raw_result_replay_artifact_id.strip()
        ):
            raise ValueError("raw_result_replay_artifact_id must be non-empty when provided")
        if self.shots is not None:
            _normalise_shots(self.shots)
        if self.hardware_execution:
            raise ValueError(
                "runtime primitive execution artefacts must not claim hardware execution"
            )
        for field_name in ("parameter_digest", "result_digest", "metadata_digest"):
            _validate_sha256_digest(field_name, str(getattr(self, field_name)))

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible Qiskit Runtime primitive metadata."""
        return {
            "artifact_id": self.artifact_id,
            "provider_name": self.provider_name,
            "primitive_name": self.primitive_name,
            "backend_name": self.backend_name,
            "job_id": self.job_id,
            "session_id": self.session_id,
            "circuit_fingerprint": self.circuit_fingerprint,
            "observable_fingerprint": self.observable_fingerprint,
            "parameter_digest": self.parameter_digest,
            "result_digest": self.result_digest,
            "metadata_digest": self.metadata_digest,
            "shots": self.shots,
            "hardware_execution": self.hardware_execution,
            "raw_result_replay_artifact_id": self.raw_result_replay_artifact_id,
        }


@dataclass(frozen=True)
class QiskitRuntimeQPUExecutionArtifact:
    """Validated Qiskit Runtime EstimatorV2/SamplerV2 QPU execution evidence."""

    artifact_id: str
    provider_name: str
    primitive_name: str
    backend_name: str
    job_id: str
    session_id: str | None
    circuit_fingerprint: str
    observable_fingerprint: str | None
    parameter_digest: str
    result_digest: str
    metadata_digest: str
    transpiled_circuit_digest: str
    live_execution_ticket: str
    backend_allowlist_id: str
    shot_budget_id: str
    runtime_session_mode: str
    shots: int
    hardware_execution: bool = True

    def __post_init__(self) -> None:
        """Validate ticketed Runtime QPU execution evidence metadata."""
        for field_name in (
            "artifact_id",
            "provider_name",
            "primitive_name",
            "backend_name",
            "job_id",
            "circuit_fingerprint",
            "live_execution_ticket",
            "backend_allowlist_id",
            "shot_budget_id",
            "runtime_session_mode",
        ):
            object.__setattr__(
                self,
                field_name,
                _normalise_metadata_text(field_name, getattr(self, field_name)),
            )
        primitive_name = _normalise_qiskit_runtime_primitive(self.primitive_name)
        object.__setattr__(self, "primitive_name", primitive_name)
        if self.session_id is not None:
            object.__setattr__(
                self,
                "session_id",
                _normalise_metadata_text("session_id", self.session_id),
            )
        if self.observable_fingerprint is not None:
            object.__setattr__(
                self,
                "observable_fingerprint",
                _normalise_metadata_text(
                    "observable_fingerprint",
                    self.observable_fingerprint,
                ),
            )
        if primitive_name == "EstimatorV2" and self.observable_fingerprint is None:
            raise ValueError("observable_fingerprint is required for EstimatorV2 evidence")
        if primitive_name == "SamplerV2" and self.observable_fingerprint is not None:
            raise ValueError("observable_fingerprint must be absent for SamplerV2 evidence")
        object.__setattr__(self, "shots", _normalise_shots(self.shots))
        if not self.hardware_execution:
            raise ValueError("runtime QPU execution artefacts must cite hardware execution")
        _validate_runtime_qpu_mode(self.runtime_session_mode)
        for field_name in (
            "parameter_digest",
            "result_digest",
            "metadata_digest",
            "transpiled_circuit_digest",
        ):
            _validate_sha256_digest(field_name, str(getattr(self, field_name)))

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible Qiskit Runtime QPU execution metadata."""
        return {
            "artifact_id": self.artifact_id,
            "provider_name": self.provider_name,
            "primitive_name": self.primitive_name,
            "backend_name": self.backend_name,
            "job_id": self.job_id,
            "session_id": self.session_id,
            "circuit_fingerprint": self.circuit_fingerprint,
            "observable_fingerprint": self.observable_fingerprint,
            "parameter_digest": self.parameter_digest,
            "result_digest": self.result_digest,
            "metadata_digest": self.metadata_digest,
            "transpiled_circuit_digest": self.transpiled_circuit_digest,
            "live_execution_ticket": self.live_execution_ticket,
            "backend_allowlist_id": self.backend_allowlist_id,
            "shot_budget_id": self.shot_budget_id,
            "runtime_session_mode": self.runtime_session_mode,
            "shots": self.shots,
            "hardware_execution": self.hardware_execution,
        }


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


@dataclass(frozen=True)
class QiskitRawCountReplayArtifact:
    """Validated Qiskit raw-count capture and replay evidence."""

    artifact_id: str
    provider_name: str
    backend_name: str
    job_id: str
    circuit_fingerprint: str
    counts_digest: str
    replay_digest: str
    shots: int
    measured_qubits: int
    expectation_value: float
    standard_error: float
    hardware_execution: bool
    live_ticket_id: str

    def __post_init__(self) -> None:
        """Validate raw-count replay evidence against live QPU metadata."""
        for field_name in (
            "artifact_id",
            "provider_name",
            "backend_name",
            "job_id",
            "circuit_fingerprint",
            "live_ticket_id",
        ):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"{field_name} must be non-empty")
        _normalise_shots(self.shots)
        if isinstance(self.measured_qubits, bool) or self.measured_qubits <= 0:
            raise ValueError("measured_qubits must be a positive integer")
        expectation_value = _as_finite_scalar("expectation_value", self.expectation_value)
        if not -1.0 <= expectation_value <= 1.0:
            raise ValueError("expectation_value must be in [-1, 1]")
        standard_error = _as_finite_scalar("standard_error", self.standard_error)
        if standard_error < 0.0:
            raise ValueError("standard_error must be non-negative")
        if not self.hardware_execution:
            raise ValueError("raw-count replay artefacts must cite hardware execution")
        _validate_sha256_digest("counts_digest", self.counts_digest)
        _validate_sha256_digest("replay_digest", self.replay_digest)
        object.__setattr__(self, "expectation_value", expectation_value)
        object.__setattr__(self, "standard_error", standard_error)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible raw-count replay metadata."""
        return {
            "artifact_id": self.artifact_id,
            "provider_name": self.provider_name,
            "backend_name": self.backend_name,
            "job_id": self.job_id,
            "circuit_fingerprint": self.circuit_fingerprint,
            "counts_digest": self.counts_digest,
            "replay_digest": self.replay_digest,
            "shots": self.shots,
            "measured_qubits": self.measured_qubits,
            "expectation_value": self.expectation_value,
            "standard_error": self.standard_error,
            "hardware_execution": self.hardware_execution,
            "live_ticket_id": self.live_ticket_id,
        }


@dataclass(frozen=True)
class QiskitCalibrationStatevectorComparisonArtifact:
    """Validated live-backend calibration and statevector comparison evidence."""

    artifact_id: str
    provider_name: str
    backend_name: str
    calibration_snapshot_id: str
    statevector_reference_artifact_id: str
    circuit_fingerprint: str
    calibration_digest: str
    comparison_digest: str
    max_abs_error: float
    tolerance: float
    hardware_execution: bool
    live_ticket_id: str

    def __post_init__(self) -> None:
        """Validate calibration/statevector comparison evidence metadata."""
        for field_name in (
            "artifact_id",
            "provider_name",
            "backend_name",
            "calibration_snapshot_id",
            "statevector_reference_artifact_id",
            "circuit_fingerprint",
            "live_ticket_id",
        ):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"{field_name} must be non-empty")
        max_abs_error = _as_finite_scalar("max_abs_error", self.max_abs_error)
        tolerance = _as_positive_scalar("tolerance", self.tolerance)
        if max_abs_error < 0.0:
            raise ValueError("max_abs_error must be non-negative")
        if max_abs_error > tolerance:
            raise ValueError("max_abs_error must be within tolerance")
        if not self.hardware_execution:
            raise ValueError("calibration comparison artefacts must cite hardware execution")
        _validate_sha256_digest("calibration_digest", self.calibration_digest)
        _validate_sha256_digest("comparison_digest", self.comparison_digest)
        object.__setattr__(self, "max_abs_error", max_abs_error)
        object.__setattr__(self, "tolerance", tolerance)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible calibration comparison metadata."""
        return {
            "artifact_id": self.artifact_id,
            "provider_name": self.provider_name,
            "backend_name": self.backend_name,
            "calibration_snapshot_id": self.calibration_snapshot_id,
            "statevector_reference_artifact_id": self.statevector_reference_artifact_id,
            "circuit_fingerprint": self.circuit_fingerprint,
            "calibration_digest": self.calibration_digest,
            "comparison_digest": self.comparison_digest,
            "max_abs_error": self.max_abs_error,
            "tolerance": self.tolerance,
            "hardware_execution": self.hardware_execution,
            "live_ticket_id": self.live_ticket_id,
        }


@dataclass(frozen=True)
class QiskitProviderGradientWorkflowArtifact:
    """Validated Qiskit Runtime provider-gradient workflow evidence.

    The artefact records captured Runtime metadata for one provider-gradient
    method. It does not submit a provider job; callers must supply job IDs,
    SHA-256 digests, shot counts, parameter dimensions, and the live-ticket
    citation captured from an approved Runtime workflow.
    """

    artifact_id: str
    provider_name: str
    backend_name: str
    job_id: str
    primitive_name: str
    gradient_method: str
    circuit_fingerprint: str
    observable_fingerprint: str | None
    parameter_digest: str
    gradient_digest: str
    metadata_digest: str
    shots: int
    parameter_count: int
    gradient_dimension: int
    hardware_execution: bool
    live_ticket_id: str
    method_metadata: Mapping[str, object] = field(default_factory=dict)
    claim_boundary: str = "qiskit_provider_gradient_workflow_capture"

    def __post_init__(self) -> None:
        """Validate captured Runtime provider-gradient workflow metadata."""
        for field_name in (
            "artifact_id",
            "provider_name",
            "backend_name",
            "job_id",
            "circuit_fingerprint",
            "live_ticket_id",
            "claim_boundary",
        ):
            object.__setattr__(
                self,
                field_name,
                _normalise_metadata_text(field_name, getattr(self, field_name)),
            )
        primitive_name = _normalise_qiskit_runtime_primitive(self.primitive_name)
        object.__setattr__(self, "primitive_name", primitive_name)
        gradient_method = _normalise_qiskit_provider_gradient_method(self.gradient_method)
        object.__setattr__(self, "gradient_method", gradient_method)
        if self.observable_fingerprint is not None:
            object.__setattr__(
                self,
                "observable_fingerprint",
                _normalise_metadata_text(
                    "observable_fingerprint",
                    self.observable_fingerprint,
                ),
            )
        if primitive_name == "EstimatorV2" and self.observable_fingerprint is None:
            raise ValueError("observable_fingerprint is required for EstimatorV2 workflows")
        if primitive_name == "SamplerV2" and self.observable_fingerprint is not None:
            raise ValueError("observable_fingerprint must be absent for SamplerV2 workflows")
        object.__setattr__(self, "shots", _normalise_shots(self.shots))
        object.__setattr__(
            self,
            "parameter_count",
            _normalise_positive_int("parameter_count", self.parameter_count),
        )
        object.__setattr__(
            self,
            "gradient_dimension",
            _normalise_positive_int("gradient_dimension", self.gradient_dimension),
        )
        if self.gradient_dimension != self.parameter_count:
            raise ValueError("gradient_dimension must match parameter_count")
        if not self.hardware_execution:
            raise ValueError("provider-gradient workflow artefacts must cite hardware execution")
        for field_name in ("parameter_digest", "gradient_digest", "metadata_digest"):
            _validate_sha256_digest(field_name, str(getattr(self, field_name)))
        method_metadata = _normalise_provider_gradient_method_metadata(
            gradient_method,
            self.method_metadata,
        )
        _validate_provider_gradient_method_metadata_consistency(
            gradient_method,
            method_metadata,
            parameter_count=self.parameter_count,
        )
        object.__setattr__(self, "method_metadata", method_metadata)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible Qiskit provider-gradient workflow metadata."""
        return {
            "artifact_id": self.artifact_id,
            "provider_name": self.provider_name,
            "backend_name": self.backend_name,
            "job_id": self.job_id,
            "primitive_name": self.primitive_name,
            "gradient_method": self.gradient_method,
            "circuit_fingerprint": self.circuit_fingerprint,
            "observable_fingerprint": self.observable_fingerprint,
            "parameter_digest": self.parameter_digest,
            "gradient_digest": self.gradient_digest,
            "metadata_digest": self.metadata_digest,
            "shots": self.shots,
            "parameter_count": self.parameter_count,
            "gradient_dimension": self.gradient_dimension,
            "hardware_execution": self.hardware_execution,
            "live_ticket_id": self.live_ticket_id,
            "method_metadata": dict(self.method_metadata),
            "claim_boundary": self.claim_boundary,
        }


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


@dataclass(frozen=True)
class QiskitRuntimeQPUProviderEvidenceBundle:
    """Validated Runtime QPU, replay, calibration, and benchmark evidence chain."""

    artifact_id: str
    runtime_qpu_execution_artifact: QiskitRuntimeQPUExecutionArtifact
    raw_count_replay_artifact: QiskitRawCountReplayArtifact
    calibration_comparison_artifact: QiskitCalibrationStatevectorComparisonArtifact
    captured_at_utc: str
    valid_until_utc: str
    isolated_benchmark_artifact_id: str | None = None
    claim_boundary: str = "qiskit_runtime_qpu_provider_evidence_bundle"

    def __post_init__(self) -> None:
        """Validate bundled Runtime QPU provider evidence metadata."""
        object.__setattr__(
            self,
            "artifact_id",
            _normalise_metadata_text("artifact_id", self.artifact_id),
        )
        object.__setattr__(
            self,
            "claim_boundary",
            _normalise_metadata_text("claim_boundary", self.claim_boundary),
        )
        object.__setattr__(
            self,
            "captured_at_utc",
            _normalise_utc_timestamp("captured_at_utc", self.captured_at_utc),
        )
        object.__setattr__(
            self,
            "valid_until_utc",
            _normalise_utc_timestamp("valid_until_utc", self.valid_until_utc),
        )
        if _utc_timestamp("valid_until_utc", self.valid_until_utc) <= _utc_timestamp(
            "captured_at_utc",
            self.captured_at_utc,
        ):
            raise ValueError("valid_until_utc must be after captured_at_utc")
        if self.isolated_benchmark_artifact_id is not None:
            object.__setattr__(
                self,
                "isolated_benchmark_artifact_id",
                _normalise_metadata_text(
                    "isolated_benchmark_artifact_id",
                    self.isolated_benchmark_artifact_id,
                ),
            )
        _validate_runtime_qpu_evidence_chain(
            runtime_qpu_execution_artifact=self.runtime_qpu_execution_artifact,
            raw_count_replay_artifact=self.raw_count_replay_artifact,
            calibration_comparison_artifact=self.calibration_comparison_artifact,
        )

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible Runtime QPU provider evidence metadata."""
        return {
            "artifact_id": self.artifact_id,
            "runtime_qpu_execution_artifact": (self.runtime_qpu_execution_artifact.to_dict()),
            "raw_count_replay_artifact": self.raw_count_replay_artifact.to_dict(),
            "calibration_comparison_artifact": (self.calibration_comparison_artifact.to_dict()),
            "captured_at_utc": self.captured_at_utc,
            "valid_until_utc": self.valid_until_utc,
            "isolated_benchmark_artifact_id": self.isolated_benchmark_artifact_id,
            "claim_boundary": self.claim_boundary,
        }


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


@dataclass(frozen=True)
class QiskitMaturityAuditResult:
    """Aggregate Qiskit local-gradient evidence and provider-execution blockers."""

    local_gradient_ready: bool
    ready_for_provider_exceedance: bool
    evidence: dict[str, object]
    required_capabilities: dict[str, str]
    local_reference_metadata: dict[str, object]
    open_gaps: tuple[str, ...]
    claim_boundary: str = "bounded_qiskit_provider_maturity_audit"

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready Qiskit maturity evidence."""
        return {
            "local_gradient_ready": self.local_gradient_ready,
            "ready_for_provider_exceedance": self.ready_for_provider_exceedance,
            "evidence": {name: _result_to_dict(result) for name, result in self.evidence.items()},
            "required_capabilities": dict(self.required_capabilities),
            "local_reference_metadata": dict(self.local_reference_metadata),
            "open_gaps": list(self.open_gaps),
            "claim_boundary": self.claim_boundary,
        }


def generate_qiskit_parameter_shift_circuits(
    circuit: QuantumCircuit,
    parameters: Sequence[Parameter],
    values: ArrayLike,
    *,
    rule: ParameterShiftRule | None = None,
    shift: float = float(np.pi / 2.0),
) -> tuple[QiskitParameterShiftRecord, ...]:
    """Generate fully bound Qiskit plus/minus circuits for parameter shift."""
    parameter_tuple = _normalise_parameters(parameters)
    values_vector = _as_finite_vector("values", values, width=len(parameter_tuple))
    terms = _parameter_shift_terms(rule, shift)
    _validate_circuit_parameters(circuit, parameter_tuple)

    records: list[QiskitParameterShiftRecord] = []
    for index, parameter in enumerate(parameter_tuple):
        for shift_index, (shift_value, coefficient) in enumerate(terms):
            plus_values = values_vector.copy()
            minus_values = values_vector.copy()
            plus_values[index] += shift_value
            minus_values[index] -= shift_value
            records.append(
                QiskitParameterShiftRecord(
                    parameter_index=index,
                    shift_index=shift_index,
                    shift=shift_value,
                    coefficient=coefficient,
                    parameter_name=parameter.name,
                    plus_values=plus_values,
                    minus_values=minus_values,
                    plus_circuit=_bind_circuit(circuit, parameter_tuple, plus_values),
                    minus_circuit=_bind_circuit(circuit, parameter_tuple, minus_values),
                )
            )
    return tuple(records)


def execute_qiskit_statevector_parameter_shift(
    circuit: QuantumCircuit,
    observable: object,
    parameters: Sequence[Parameter],
    values: ArrayLike,
    *,
    rule: ParameterShiftRule | None = None,
    shift: float = float(np.pi / 2.0),
) -> QiskitParameterShiftGradientResult:
    """Evaluate a local Qiskit Statevector value and parameter-shift gradient."""
    parameter_tuple = _normalise_parameters(parameters)
    values_vector = _as_finite_vector("values", values, width=len(parameter_tuple))
    terms = _parameter_shift_terms(rule, shift)
    _validate_circuit_parameters(circuit, parameter_tuple)
    base_circuit = _bind_circuit(circuit, parameter_tuple, values_vector)
    value = _expectation(base_circuit, observable)
    records = generate_qiskit_parameter_shift_circuits(
        circuit,
        parameter_tuple,
        values_vector,
        rule=rule,
        shift=shift,
    )
    gradient = np.zeros(values_vector.size, dtype=np.float64)
    for record in records:
        gradient[record.parameter_index] += record.coefficient * (
            _expectation(record.plus_circuit, observable)
            - _expectation(record.minus_circuit, observable)
        )
    return QiskitParameterShiftGradientResult(
        value=value,
        gradient=gradient,
        records=records,
        method="qiskit_statevector_parameter_shift"
        if len(terms) == 1
        else "qiskit_statevector_multi_frequency_parameter_shift",
        evaluations=1 + 2 * len(records),
        claim_boundary=(
            "local Qiskit Statevector parameter-shift execution for fully bound circuits; "
            "not hardware execution, provider submission, or finite-shot evidence"
        ),
    )


def execute_qiskit_finite_shot_parameter_shift(
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
) -> ProviderGradientExecutionResult:
    """Evaluate Qiskit shifted circuits through the provider-safe finite-shot contract."""
    shot_count = _normalise_shots(shots)
    parameter_tuple = _normalise_parameters(parameters)
    values_vector = _as_finite_vector("values", values, width=len(parameter_tuple))
    _validate_circuit_parameters(circuit, parameter_tuple)

    def sampler(shifted_values: FloatArray, sample_shots: int | None) -> ProviderExpectationSample:
        bound = _bind_circuit(circuit, parameter_tuple, shifted_values)
        value, variance = _expectation_and_variance(bound, observable)
        return ProviderExpectationSample(
            value=value,
            variance=variance,
            shots=sample_shots,
            metadata={
                "engine": "qiskit_statevector_finite_shot_surrogate",
                "observable_type": type(observable).__name__,
            },
        )

    return execute_provider_parameter_shift_gradient(
        sampler,
        values_vector,
        backend="finite_shot_simulator",
        shots=shot_count,
        rule=rule,
        shift=shift,
        confidence_level=confidence_level,
        confidence_z=confidence_z,
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


def _result_to_dict(result: object) -> object:
    to_dict = getattr(result, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    if isinstance(result, tuple):
        return [_result_to_dict(item) for item in result]
    return result


def _expectation(circuit: QuantumCircuit, observable: object) -> float:
    state = Statevector.from_instruction(circuit)
    value = state.expectation_value(observable)
    return _as_finite_scalar("Qiskit expectation value", np.real(value))


def _expectation_and_variance(circuit: QuantumCircuit, observable: object) -> tuple[float, float]:
    state = Statevector.from_instruction(circuit)
    value = _as_finite_scalar(
        "Qiskit expectation value",
        np.real(state.expectation_value(observable)),
    )
    matrix = Operator(observable).data
    vector = np.asarray(state.data, dtype=np.complex128)
    second_moment = _as_finite_scalar(
        "Qiskit expectation second moment",
        np.real(np.vdot(matrix @ vector, matrix @ vector)),
    )
    variance = max(0.0, second_moment - value * value)
    return value, variance


def _bind_circuit(
    circuit: QuantumCircuit,
    parameters: tuple[Parameter, ...],
    values: FloatArray,
) -> QuantumCircuit:
    mapping = {parameter: float(values[index]) for index, parameter in enumerate(parameters)}
    bound = circuit.assign_parameters(mapping, inplace=False)
    if bound.num_parameters != 0:
        raise ValueError("all circuit parameters must be supplied before Qiskit execution")
    return bound


def _validate_circuit_parameters(
    circuit: QuantumCircuit,
    parameters: tuple[Parameter, ...],
) -> None:
    supplied = set(parameters)
    circuit_parameters = set(circuit.parameters)
    if not circuit_parameters:
        raise ValueError("circuit must contain at least one trainable parameter")
    if not circuit_parameters.issubset(supplied):
        raise ValueError("all circuit parameters must be listed in parameters")


def _normalise_parameters(parameters: Sequence[Parameter]) -> tuple[Parameter, ...]:
    parameter_tuple = tuple(parameters)
    if not parameter_tuple:
        raise ValueError("parameters must contain at least one Qiskit Parameter")
    if len(set(parameter_tuple)) != len(parameter_tuple):
        raise ValueError("parameters must not contain duplicates")
    for parameter in parameter_tuple:
        if not isinstance(parameter, Parameter):
            raise ValueError("parameters must contain Qiskit Parameter objects")
    return parameter_tuple


def _parameter_shift_terms(
    rule: ParameterShiftRule | None,
    shift: float,
) -> tuple[tuple[float, float], ...]:
    if rule is not None:
        return tuple(
            (
                _as_positive_scalar("rule.shift", shift_value),
                _as_finite_scalar("rule.coefficient", coefficient),
            )
            for shift_value, coefficient in rule.terms
        )
    shift_value = _as_positive_scalar("shift", shift)
    denominator = 2.0 * np.sin(shift_value)
    if abs(denominator) <= 1.0e-15:
        raise ValueError("shift must not make the parameter-shift denominator singular")
    return ((shift_value, float(1.0 / denominator)),)


def _as_finite_vector(
    name: str,
    value: ArrayLike,
    *,
    width: int | None = None,
) -> FloatArray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if width is not None and vector.shape != (width,):
        raise ValueError(f"{name} must have shape ({width},), got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector.astype(np.float64, copy=True)


def _as_finite_scalar(name: str, value: object) -> float:
    raw = np.asarray(value)
    if raw.shape != () or raw.dtype.kind in {"b", "O", "S", "U", "c"}:
        raise ValueError(f"{name} must be a real numeric scalar")
    scalar = float(raw)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def _as_positive_scalar(name: str, value: object) -> float:
    scalar = _as_finite_scalar(name, value)
    if scalar <= 0.0:
        raise ValueError(f"{name} must be positive")
    return scalar


def _validate_sha256_digest(name: str, value: str) -> None:
    hex_digest = value.removeprefix("sha256:")
    if not (
        value.startswith("sha256:")
        and len(hex_digest) == 64
        and all(char in "0123456789abcdefABCDEF" for char in hex_digest)
    ):
        raise ValueError(f"{name} must be a sha256:<64-hex> digest")


def _normalise_metadata_text(field_name: str, value: object) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} must be non-empty")
    if any(ord(character) < 32 or ord(character) == 127 for character in text):
        raise ValueError(f"{field_name} must not contain control characters")
    return text


def _normalise_utc_timestamp(field_name: str, value: object) -> str:
    timestamp = _utc_timestamp(field_name, value)
    return timestamp.isoformat().replace("+00:00", "Z")


def _utc_timestamp(field_name: str, value: object) -> datetime:
    text = _normalise_metadata_text(field_name, value)
    try:
        timestamp = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO-8601 UTC timestamp") from exc
    if timestamp.tzinfo is None:
        raise ValueError(f"{field_name} must include a UTC offset")
    return timestamp.astimezone(timezone.utc).replace(microsecond=0)


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


def _normalise_qiskit_runtime_primitive(primitive_name: str) -> str:
    primitive = primitive_name.strip()
    aliases = {
        "estimator": "EstimatorV2",
        "estimatorv2": "EstimatorV2",
        "sampler": "SamplerV2",
        "samplerv2": "SamplerV2",
    }
    try:
        return aliases[primitive.lower()]
    except KeyError as exc:
        raise ValueError("primitive_name must be EstimatorV2 or SamplerV2") from exc


def _normalise_qiskit_provider_gradient_method(gradient_method: str) -> str:
    method = gradient_method.strip().lower().replace("-", "_")
    aliases = {
        "parameter_shift": "parameter_shift",
        "finite_difference": "finite_difference",
        "lcu": "lcu",
        "spsa": "spsa",
        "qgt": "qgt",
        "qfi": "qfi",
    }
    try:
        return aliases[method]
    except KeyError as exc:
        raise ValueError(
            "gradient_method must be one of finite_difference, lcu, "
            "parameter_shift, qfi, qgt, or spsa"
        ) from exc


def _normalise_provider_gradient_method_metadata(
    gradient_method: str,
    method_metadata: Mapping[str, object],
) -> dict[str, object]:
    """Validate and normalise method-specific provider-gradient provenance."""
    if not method_metadata:
        raise ValueError("method_metadata is required for provider-gradient workflow artefacts")
    required_keys = (
        _QISKIT_PROVIDER_GRADIENT_METHOD_COMMON_METADATA_KEYS
        | _QISKIT_PROVIDER_GRADIENT_METHOD_SPECIFIC_METADATA_KEYS[gradient_method]
    )
    metadata_keys = frozenset(method_metadata)
    unsupported_keys = sorted(metadata_keys - required_keys)
    if unsupported_keys:
        joined_keys = ", ".join(unsupported_keys)
        raise ValueError(f"method_metadata contains unsupported keys: {joined_keys}")
    missing_keys = sorted(required_keys - metadata_keys)
    if missing_keys:
        joined_keys = ", ".join(missing_keys)
        raise ValueError(f"method_metadata is missing required keys: {joined_keys}")

    normalised: dict[str, object] = {
        "method_schema": _normalise_metadata_text(
            "method_metadata.method_schema",
            _require_provider_gradient_method_metadata_value(
                method_metadata,
                "method_schema",
            ),
        ),
        "method_artifact_id": _normalise_metadata_text(
            "method_metadata.method_artifact_id",
            _require_provider_gradient_method_metadata_value(
                method_metadata,
                "method_artifact_id",
            ),
        ),
        "workflow_version": _normalise_metadata_text(
            "method_metadata.workflow_version",
            _require_provider_gradient_method_metadata_value(
                method_metadata,
                "workflow_version",
            ),
        ),
    }
    expected_schema = _QISKIT_PROVIDER_GRADIENT_METHOD_SCHEMAS[gradient_method]
    if normalised["method_schema"] != expected_schema:
        raise ValueError(
            f"method_metadata.method_schema must be {expected_schema!r} for {gradient_method}"
        )

    if gradient_method == "parameter_shift":
        normalised["shift_rule_id"] = _normalise_metadata_text(
            "method_metadata.shift_rule_id",
            _require_provider_gradient_method_metadata_value(
                method_metadata,
                "shift_rule_id",
            ),
        )
        normalised["shift_count"] = _normalise_positive_int(
            "method_metadata.shift_count",
            _require_provider_gradient_method_metadata_value(method_metadata, "shift_count"),
        )
    elif gradient_method == "finite_difference":
        normalised["stencil"] = _normalise_metadata_text(
            "method_metadata.stencil",
            _require_provider_gradient_method_metadata_value(method_metadata, "stencil"),
        )
        normalised["step_size"] = _as_positive_scalar(
            "method_metadata.step_size",
            _require_provider_gradient_method_metadata_value(method_metadata, "step_size"),
        )
    elif gradient_method == "lcu":
        normalised["generator_digest"] = _normalise_sha256_metadata_digest(
            "method_metadata.generator_digest",
            _require_provider_gradient_method_metadata_value(
                method_metadata,
                "generator_digest",
            ),
        )
        normalised["term_count"] = _normalise_positive_int(
            "method_metadata.term_count",
            _require_provider_gradient_method_metadata_value(method_metadata, "term_count"),
        )
    elif gradient_method == "spsa":
        normalised["perturbation_seed"] = _normalise_non_negative_int(
            "method_metadata.perturbation_seed",
            _require_provider_gradient_method_metadata_value(
                method_metadata,
                "perturbation_seed",
            ),
        )
        normalised["perturbation_count"] = _normalise_positive_int(
            "method_metadata.perturbation_count",
            _require_provider_gradient_method_metadata_value(
                method_metadata,
                "perturbation_count",
            ),
        )
    elif gradient_method == "qgt":
        normalised["qgt_digest"] = _normalise_sha256_metadata_digest(
            "method_metadata.qgt_digest",
            _require_provider_gradient_method_metadata_value(method_metadata, "qgt_digest"),
        )
        normalised["matrix_dimension"] = _normalise_positive_int(
            "method_metadata.matrix_dimension",
            _require_provider_gradient_method_metadata_value(
                method_metadata,
                "matrix_dimension",
            ),
        )
    elif gradient_method == "qfi":
        normalised["qfi_digest"] = _normalise_sha256_metadata_digest(
            "method_metadata.qfi_digest",
            _require_provider_gradient_method_metadata_value(method_metadata, "qfi_digest"),
        )
        normalised["matrix_dimension"] = _normalise_positive_int(
            "method_metadata.matrix_dimension",
            _require_provider_gradient_method_metadata_value(
                method_metadata,
                "matrix_dimension",
            ),
        )
    else:  # pragma: no cover - guarded by _normalise_qiskit_provider_gradient_method.
        raise ValueError("gradient_method must be a supported provider-gradient method")
    return normalised


def _require_provider_gradient_method_metadata_value(
    method_metadata: Mapping[str, object],
    key: str,
) -> object:
    """Return a required method-metadata value with a stable error message."""
    if key not in method_metadata:
        raise ValueError(f"method_metadata.{key} is required")
    return method_metadata[key]


def _normalise_sha256_metadata_digest(field_name: str, value: object) -> str:
    """Return a normalised sha256 digest captured inside method metadata."""
    digest = _normalise_metadata_text(field_name, value)
    _validate_sha256_digest(field_name, digest)
    return digest


def _validate_provider_gradient_method_metadata_consistency(
    gradient_method: str,
    method_metadata: Mapping[str, object],
    *,
    parameter_count: int,
) -> None:
    """Validate metadata fields that depend on the workflow parameter width."""
    if gradient_method not in {"qgt", "qfi"}:
        return
    matrix_dimension = method_metadata["matrix_dimension"]
    if matrix_dimension != parameter_count:
        raise ValueError(
            f"method_metadata.matrix_dimension must match parameter_count for {gradient_method}"
        )


def _validate_runtime_qpu_mode(runtime_session_mode: str) -> None:
    mode = runtime_session_mode.lower()
    if not any(token in mode for token in ("qpu", "hardware", "live")):
        raise ValueError("runtime_session_mode must identify live QPU execution")
    if "simulator" in mode or "offline" in mode or "replay" in mode:
        raise ValueError("runtime_session_mode must not identify simulator or replay execution")


def _validate_runtime_qpu_evidence_chain(
    *,
    runtime_qpu_execution_artifact: QiskitRuntimeQPUExecutionArtifact | None,
    raw_count_replay_artifact: QiskitRawCountReplayArtifact | None,
    calibration_comparison_artifact: QiskitCalibrationStatevectorComparisonArtifact | None,
) -> None:
    if raw_count_replay_artifact is not None and runtime_qpu_execution_artifact is None:
        raise ValueError("raw-count replay artefact requires matching Runtime QPU evidence")
    if calibration_comparison_artifact is not None and runtime_qpu_execution_artifact is None:
        raise ValueError("calibration comparison artefact requires matching Runtime QPU evidence")
    if runtime_qpu_execution_artifact is None:
        return
    if raw_count_replay_artifact is not None:
        _require_matching_evidence_field(
            "raw_count_replay_artifact.provider_name",
            raw_count_replay_artifact.provider_name,
            runtime_qpu_execution_artifact.provider_name,
        )
        _require_matching_evidence_field(
            "raw_count_replay_artifact.backend_name",
            raw_count_replay_artifact.backend_name,
            runtime_qpu_execution_artifact.backend_name,
        )
        _require_matching_evidence_field(
            "raw_count_replay_artifact.job_id",
            raw_count_replay_artifact.job_id,
            runtime_qpu_execution_artifact.job_id,
        )
        _require_matching_evidence_field(
            "raw_count_replay_artifact.circuit_fingerprint",
            raw_count_replay_artifact.circuit_fingerprint,
            runtime_qpu_execution_artifact.circuit_fingerprint,
        )
        _require_matching_evidence_field(
            "raw_count_replay_artifact.live_ticket_id",
            raw_count_replay_artifact.live_ticket_id,
            runtime_qpu_execution_artifact.live_execution_ticket,
        )
        if raw_count_replay_artifact.shots != runtime_qpu_execution_artifact.shots:
            raise ValueError("raw_count_replay_artifact.shots must match Runtime QPU shots")
    if calibration_comparison_artifact is not None:
        _require_matching_evidence_field(
            "calibration_comparison_artifact.provider_name",
            calibration_comparison_artifact.provider_name,
            runtime_qpu_execution_artifact.provider_name,
        )
        _require_matching_evidence_field(
            "calibration_comparison_artifact.backend_name",
            calibration_comparison_artifact.backend_name,
            runtime_qpu_execution_artifact.backend_name,
        )
        _require_matching_evidence_field(
            "calibration_comparison_artifact.circuit_fingerprint",
            calibration_comparison_artifact.circuit_fingerprint,
            runtime_qpu_execution_artifact.circuit_fingerprint,
        )
        _require_matching_evidence_field(
            "calibration_comparison_artifact.live_ticket_id",
            calibration_comparison_artifact.live_ticket_id,
            runtime_qpu_execution_artifact.live_execution_ticket,
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


def _require_matching_evidence_field(field_name: str, observed: str, expected: str) -> None:
    if observed != expected:
        raise ValueError(f"{field_name} must match Runtime QPU evidence")


def _require_matching_optional_evidence_field(
    field_name: str,
    observed: str | None,
    expected: str | None,
) -> None:
    if observed != expected:
        raise ValueError(f"{field_name} must match Runtime QPU evidence")


def _normalise_positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _normalise_non_negative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _normalise_shots(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("shots must be a positive integer")
    return value


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
