# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — external differentiable comparison contracts
"""Immutable records and field vocabulary for external differentiable comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .differentiable_catalyst_comparison import CatalystCompilerWorkflowComparison

ComparisonStatus = Literal["success", "hard_gap"]
ComparisonClosureStatus = Literal["implemented", "implementation_path", "permanent_boundary"]

PERMANENT_EXTERNAL_COMPARISON_BOUNDARIES = frozenset(
    {
        "unsupported_batching",
        "unsupported_transform",
        "unsupported_dtype",
        "unsupported_device",
    }
)

REQUIRED_EXTERNAL_COMPARISON_ROW_FIELDS = frozenset(
    {
        "case_id",
        "backend",
        "status",
        "failure_class",
        "value_error",
        "gradient_error",
        "runtime_seconds",
        "memory_peak_bytes",
        "batching_support",
        "transform_support",
        "dtype",
        "device",
        "source_of_truth",
        "setup_instructions",
        "claim_boundary",
        "dependency_versions",
        "toolchain",
        "catalyst_comparison",
        "closure_status",
        "closure_reason",
    }
)


@dataclass(frozen=True)
class ExternalComparisonRow:
    """One external framework/compiler comparison row."""

    case_id: str
    backend: str
    status: ComparisonStatus
    failure_class: str | None
    value_error: float | None
    gradient_error: float | None
    runtime_seconds: float | None
    memory_peak_bytes: int | None
    batching_support: str
    transform_support: str
    dtype: str
    device: str
    source_of_truth: str
    setup_instructions: str | None
    claim_boundary: str
    dependency_versions: dict[str, str] | None = None
    toolchain: dict[str, str] | None = None
    catalyst_comparison: CatalystCompilerWorkflowComparison | None = None

    def __post_init__(self) -> None:
        """Validate external comparison row evidence invariants."""
        if not self.case_id:
            raise ValueError("case_id must be non-empty")
        if not self.backend:
            raise ValueError("backend must be non-empty")
        if self.status not in {"success", "hard_gap"}:
            raise ValueError("status must be success or hard_gap")
        if self.source_of_truth != "scpn_reference":
            raise ValueError("source_of_truth must be scpn_reference")
        if not self.claim_boundary:
            raise ValueError("claim_boundary must be non-empty")
        if self.backend == "catalyst" and self.catalyst_comparison is None:
            raise ValueError("catalyst_comparison is required for Catalyst rows")
        if self.backend != "catalyst" and self.catalyst_comparison is not None:
            raise ValueError("catalyst_comparison is only valid for Catalyst rows")
        if self.status == "success":
            if (
                self.failure_class is not None
                or self.value_error is None
                or self.gradient_error is None
                or self.runtime_seconds is None
                or self.memory_peak_bytes is None
            ):
                raise ValueError("success rows require numeric evidence and no failure class")
            if self.value_error < 0 or self.gradient_error < 0:
                raise ValueError("success row errors must be non-negative")
        if self.status == "hard_gap" and (not self.failure_class or not self.setup_instructions):
            raise ValueError("hard_gap rows require failure_class and setup_instructions")
        if self.toolchain is not None and (
            any(not isinstance(key, str) or not key for key in self.toolchain)
            or any(not isinstance(value, str) or not value for value in self.toolchain.values())
        ):
            raise ValueError("toolchain metadata must map non-empty strings to non-empty strings")
        if self.dependency_versions is not None and (
            any(not isinstance(key, str) or not key for key in self.dependency_versions)
            or any(
                not isinstance(value, str) or not value
                for value in self.dependency_versions.values()
            )
        ):
            raise ValueError(
                "dependency version metadata must map non-empty strings to non-empty strings"
            )
        if not self.closure_reason.strip():
            raise ValueError("closure_reason must be non-empty")
        if self.status == "success" and self.closure_status != "implemented":
            raise ValueError("success rows must use implemented closure_status")
        if self.status == "hard_gap" and self.closure_status == "implemented":
            raise ValueError("hard_gap rows cannot use implemented closure_status")
        if (
            self.failure_class in PERMANENT_EXTERNAL_COMPARISON_BOUNDARIES
            and self.closure_status != "permanent_boundary"
        ):
            raise ValueError("unsupported route rows must use permanent_boundary closure_status")
        if (
            self.failure_class not in PERMANENT_EXTERNAL_COMPARISON_BOUNDARIES
            and self.status == "hard_gap"
            and self.closure_status != "implementation_path"
        ):
            raise ValueError("implementable hard_gap rows must use implementation_path")

    @property
    def closure_status(self) -> ComparisonClosureStatus:
        """Return how the row is closed for BL-12 audit purposes."""
        if self.status == "success":
            return "implemented"
        if self.failure_class in PERMANENT_EXTERNAL_COMPARISON_BOUNDARIES:
            return "permanent_boundary"
        return "implementation_path"

    @property
    def closure_reason(self) -> str:
        """Return the non-empty implementation or boundary reason for the row."""
        if self.status == "success":
            return "SCPN reference comparison passed with value and gradient evidence."
        if self.closure_status == "permanent_boundary":
            return str(self.setup_instructions)
        return str(self.setup_instructions)

    @property
    def artifact_fields_ready(self) -> bool:
        """Return whether this row is serializable as an evidence artefact."""
        payload = self.to_dict()
        return bool(
            self.case_id
            and self.backend
            and self.status
            and self.claim_boundary
            and REQUIRED_EXTERNAL_COMPARISON_ROW_FIELDS.issubset(payload)
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready row."""
        return {
            "case_id": self.case_id,
            "backend": self.backend,
            "status": self.status,
            "failure_class": self.failure_class,
            "value_error": self.value_error,
            "gradient_error": self.gradient_error,
            "runtime_seconds": self.runtime_seconds,
            "memory_peak_bytes": self.memory_peak_bytes,
            "batching_support": self.batching_support,
            "transform_support": self.transform_support,
            "dtype": self.dtype,
            "device": self.device,
            "source_of_truth": self.source_of_truth,
            "setup_instructions": self.setup_instructions,
            "claim_boundary": self.claim_boundary,
            "dependency_versions": (
                dict(self.dependency_versions) if self.dependency_versions is not None else None
            ),
            "toolchain": dict(self.toolchain) if self.toolchain is not None else None,
            "catalyst_comparison": (
                self.catalyst_comparison.to_dict()
                if self.catalyst_comparison is not None
                else None
            ),
            "closure_status": self.closure_status,
            "closure_reason": self.closure_reason,
        }


@dataclass(frozen=True)
class ExternalComparisonArtifact:
    """Written external comparison artefact paths and summary metadata."""

    artifact_id: str
    path: Path
    row_count: int
    success_count: int
    hard_gap_count: int
    hard_gap_closure_counts: dict[str, int]
    classification: str
    claim_boundary: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready artefact summary."""
        return {
            "artifact_id": self.artifact_id,
            "path": str(self.path),
            "row_count": self.row_count,
            "success_count": self.success_count,
            "hard_gap_count": self.hard_gap_count,
            "hard_gap_closure_counts": dict(self.hard_gap_closure_counts),
            "classification": self.classification,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class IdenticalCircuitGradientComparisonRow:
    """One same-circuit gradient comparison row against an external framework."""

    case_id: str
    backend: str
    status: ComparisonStatus
    failure_class: str | None
    circuit_fingerprint: str
    operations: tuple[tuple[object, ...], ...]
    observable: str
    parameter_values: tuple[float, ...]
    execution_mode: str
    shots: int | None
    scpn_value: float | None
    backend_value: float | None
    value_error: float | None
    scpn_gradient: tuple[float, ...] | None
    backend_gradient: tuple[float, ...] | None
    gradient_error: float | None
    evaluations: int | None
    dependency_versions: dict[str, str] | None
    claim_boundary: str
    performance_claim_eligible: bool = False

    def __post_init__(self) -> None:
        """Validate same-circuit comparison row evidence invariants."""
        if not self.case_id:
            raise ValueError("case_id must be non-empty")
        if self.backend not in {"qiskit", "pennylane"}:
            raise ValueError("backend must be qiskit or pennylane")
        if self.status not in {"success", "hard_gap"}:
            raise ValueError("status must be success or hard_gap")
        if not self.circuit_fingerprint:
            raise ValueError("circuit_fingerprint must be non-empty")
        if not self.operations:
            raise ValueError("operations must be non-empty")
        if not self.observable:
            raise ValueError("observable must be non-empty")
        if self.execution_mode != "exact_state":
            raise ValueError("execution_mode must be exact_state")
        if self.shots is not None:
            raise ValueError(
                "identical-circuit comparison uses exact-state mode; shots must be None"
            )
        if not self.claim_boundary:
            raise ValueError("claim_boundary must be non-empty")
        if self.status == "success":
            if (
                self.failure_class is not None
                or self.scpn_value is None
                or self.backend_value is None
                or self.value_error is None
                or self.scpn_gradient is None
                or self.backend_gradient is None
                or self.gradient_error is None
                or self.evaluations is None
            ):
                raise ValueError("success rows require numeric value and gradient evidence")
            if self.value_error < 0.0 or self.gradient_error < 0.0:
                raise ValueError("success row errors must be non-negative")
        if self.status == "hard_gap" and self.failure_class is None:
            raise ValueError("hard_gap rows require a failure_class")

    @property
    def artifact_fields_ready(self) -> bool:
        """Return whether the row carries the required same-circuit fields."""
        return bool(
            self.case_id
            and self.backend
            and self.circuit_fingerprint
            and self.operations
            and self.observable
            and self.execution_mode
            and self.claim_boundary
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready row."""
        return {
            "case_id": self.case_id,
            "backend": self.backend,
            "status": self.status,
            "failure_class": self.failure_class,
            "circuit_fingerprint": self.circuit_fingerprint,
            "operations": _jsonify_operations(self.operations),
            "observable": self.observable,
            "parameter_values": list(self.parameter_values),
            "execution_mode": self.execution_mode,
            "shots": self.shots,
            "scpn_value": self.scpn_value,
            "backend_value": self.backend_value,
            "value_error": self.value_error,
            "scpn_gradient": list(self.scpn_gradient) if self.scpn_gradient is not None else None,
            "backend_gradient": (
                list(self.backend_gradient) if self.backend_gradient is not None else None
            ),
            "gradient_error": self.gradient_error,
            "evaluations": self.evaluations,
            "dependency_versions": (
                dict(self.dependency_versions) if self.dependency_versions is not None else None
            ),
            "claim_boundary": self.claim_boundary,
            "performance_claim_eligible": self.performance_claim_eligible,
        }


@dataclass(frozen=True)
class IdenticalCircuitGradientComparisonArtifact:
    """Written same-circuit comparison artefact summary."""

    artifact_id: str
    path: Path
    row_count: int
    success_count: int
    hard_gap_count: int
    identical_circuit_ready: bool
    promotion_ready: bool
    classification: str
    claim_boundary: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready artefact summary."""
        return {
            "artifact_id": self.artifact_id,
            "path": str(self.path),
            "row_count": self.row_count,
            "success_count": self.success_count,
            "hard_gap_count": self.hard_gap_count,
            "identical_circuit_ready": self.identical_circuit_ready,
            "promotion_ready": self.promotion_ready,
            "classification": self.classification,
            "claim_boundary": self.claim_boundary,
        }


def _jsonify_operations(operations: tuple[tuple[object, ...], ...]) -> list[list[object]]:
    return [[_jsonify_operation_part(part) for part in operation] for operation in operations]


def _jsonify_operation_part(part: object) -> object:
    if isinstance(part, tuple):
        return [_jsonify_operation_part(item) for item in part]
    if isinstance(part, np.integer):
        return int(part)
    if isinstance(part, np.floating):
        return float(part)
    return part


__all__ = [
    "ComparisonClosureStatus",
    "ComparisonStatus",
    "ExternalComparisonArtifact",
    "ExternalComparisonRow",
    "IdenticalCircuitGradientComparisonArtifact",
    "IdenticalCircuitGradientComparisonRow",
    "PERMANENT_EXTERNAL_COMPARISON_BOUNDARIES",
    "REQUIRED_EXTERNAL_COMPARISON_ROW_FIELDS",
]
