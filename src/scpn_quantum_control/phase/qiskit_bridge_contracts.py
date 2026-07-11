# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Qiskit Bridge Contracts
"""Qiskit gradient, Runtime, and provider-evidence record contracts.

This one-way leaf owns immutable records, provider-method registries,
constructor validation, and JSON-ready serialization. It contains no shifted
circuit execution, provider callback, hardware preparation, maturity audit,
benchmark, or publication orchestration.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from qiskit import QuantumCircuit

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


def _result_to_dict(result: object) -> object:
    to_dict = getattr(result, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    if isinstance(result, tuple):
        return [_result_to_dict(item) for item in result]
    return result


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


def _require_matching_evidence_field(field_name: str, observed: str, expected: str) -> None:
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
    "QiskitParameterShiftRecord",
    "QiskitParameterShiftGradientResult",
    "QiskitRuntimePrimitiveExecutionArtifact",
    "QiskitRuntimeQPUExecutionArtifact",
    "QiskitRawCountReplayArtifact",
    "QiskitCalibrationStatevectorComparisonArtifact",
    "QiskitProviderGradientWorkflowArtifact",
    "QiskitRuntimeQPUProviderEvidenceBundle",
    "QiskitMaturityAuditResult",
]
