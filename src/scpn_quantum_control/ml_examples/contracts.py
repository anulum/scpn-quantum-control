# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — convergence-example ML convergence evidence contracts
"""Immutable contracts for bounded QNN/QGNN/QSNN convergence examples."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Final

ML_CONVERGENCE_SCHEMA: Final[str] = "ml_convergence_examples.v1"
ML_CONVERGENCE_CLAIM_BOUNDARY: Final[str] = (
    "deterministic synthetic local QNN/QGNN/QSNN training evidence on frozen "
    "small tasks; no arbitrary-architecture, generalisation, SOTA, provider, "
    "QPU, neuromorphic-hardware, or production convergence claim"
)


class ModelFamily(str, Enum):
    """Model families covered by the ML convergence evidence suite."""

    QNN = "qnn"
    QGNN = "qgnn"
    QSNN = "qsnn"


class FrameworkStatus(str, Enum):
    """Complete framework-row outcomes without blank or inferred support."""

    RAN = "ran"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"
    NOT_APPLICABLE = "not_applicable"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True, slots=True)
class ConvergenceExampleSpec:
    """Frozen synthetic task and acceptance thresholds for one model family."""

    example_id: str
    family: ModelFamily
    seed: int
    task: str
    max_steps: int
    target_loss: float
    min_loss_drop: float
    backend: str = "statevector_simulator"
    hardware: bool = False

    def __post_init__(self) -> None:
        """Validate the bounded local task specification."""
        if not self.example_id.strip() or not self.task.strip() or not self.backend.strip():
            raise ValueError("example_id, task, and backend must be non-empty")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if isinstance(self.max_steps, bool) or not isinstance(self.max_steps, int):
            raise ValueError("max_steps must be an integer")
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        _non_negative_finite("target_loss", self.target_loss)
        _non_negative_finite("min_loss_drop", self.min_loss_drop)
        if self.hardware:
            raise ValueError("ML convergence examples must remain simulator-only")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready task specification."""
        return asdict(self) | {"family": self.family.value}


@dataclass(frozen=True, slots=True)
class ConvergenceCertificate:
    """Machine-checkable convergence threshold evidence for one frozen task."""

    spec: ConvergenceExampleSpec
    loss_history: tuple[float, ...]
    initial_loss: float
    final_loss: float
    best_loss: float
    loss_drop: float
    target_reached: bool
    loss_drop_reached: bool
    deterministic_replay: bool
    stop_reason: str
    metric_name: str | None = None
    metric_value: float | None = None
    metric_threshold: float | None = None
    details: tuple[tuple[str, object], ...] = ()
    claim_boundary: str = ML_CONVERGENCE_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate exact curve arithmetic and frozen acceptance booleans."""
        if not self.loss_history or not self.stop_reason.strip():
            raise ValueError("loss_history and stop_reason must be non-empty")
        if not all(math.isfinite(value) and value >= 0.0 for value in self.loss_history):
            raise ValueError("loss_history must contain finite non-negative values")
        if not math.isclose(self.initial_loss, self.loss_history[0], rel_tol=0.0, abs_tol=0.0):
            raise ValueError("initial_loss must equal the first loss-history value")
        if not math.isclose(self.final_loss, self.loss_history[-1], rel_tol=0.0, abs_tol=0.0):
            raise ValueError("final_loss must equal the last loss-history value")
        if not math.isclose(self.best_loss, min(self.loss_history), rel_tol=0.0, abs_tol=0.0):
            raise ValueError("best_loss must equal the minimum loss-history value")
        expected_drop = self.initial_loss - self.best_loss
        if not math.isclose(self.loss_drop, expected_drop, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError("loss_drop must equal initial_loss minus best_loss")
        if self.target_reached != (self.best_loss <= self.spec.target_loss):
            raise ValueError("target_reached disagrees with the frozen target loss")
        if self.loss_drop_reached != (self.loss_drop >= self.spec.min_loss_drop):
            raise ValueError("loss_drop_reached disagrees with the frozen minimum drop")
        metric_fields = (self.metric_name, self.metric_value, self.metric_threshold)
        if any(value is not None for value in metric_fields) and not all(
            value is not None for value in metric_fields
        ):
            raise ValueError("metric name, value, and threshold must be provided together")
        if self.metric_name is not None and not self.metric_name.strip():
            raise ValueError("metric_name must be non-empty when provided")
        if self.metric_value is not None:
            _finite("metric_value", self.metric_value)
            _finite("metric_threshold", self.metric_threshold)
        if len({key for key, _value in self.details}) != len(self.details):
            raise ValueError("certificate detail keys must be unique")
        if any(not key.strip() for key, _value in self.details):
            raise ValueError("certificate detail keys must be non-empty")

    @property
    def metric_reached(self) -> bool:
        """Return whether the optional metric threshold is met."""
        if self.metric_value is None or self.metric_threshold is None:
            return True
        return self.metric_value >= self.metric_threshold

    @property
    def passed(self) -> bool:
        """Return whether all preregistered convergence gates passed."""
        return bool(
            self.target_reached
            and self.loss_drop_reached
            and self.deterministic_replay
            and self.metric_reached
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready certificate."""
        return {
            "spec": self.spec.to_dict(),
            "loss_history": list(self.loss_history),
            "initial_loss": self.initial_loss,
            "final_loss": self.final_loss,
            "best_loss": self.best_loss,
            "loss_drop": self.loss_drop,
            "target_reached": self.target_reached,
            "loss_drop_reached": self.loss_drop_reached,
            "deterministic_replay": self.deterministic_replay,
            "stop_reason": self.stop_reason,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "metric_threshold": self.metric_threshold,
            "metric_reached": self.metric_reached,
            "details": {key: value for key, value in self.details},
            "passed": self.passed,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True, slots=True)
class FrameworkEvidenceRow:
    """One executed, unavailable, unsupported, or inapplicable framework row."""

    family: ModelFamily
    framework: str
    status: FrameworkStatus
    required: bool
    executed: bool
    passed: bool | None
    reason: str
    max_abs_error: float | None = None

    def __post_init__(self) -> None:
        """Validate row completeness and status semantics."""
        if not self.framework.strip() or not self.reason.strip():
            raise ValueError("framework and reason must be non-empty")
        if self.status in {FrameworkStatus.RAN, FrameworkStatus.FAILED}:
            if not self.executed or self.passed is None:
                raise ValueError("executed framework rows require an explicit pass result")
        elif self.executed or self.passed is not None:
            raise ValueError("unexecuted framework rows cannot carry a pass result")
        if self.status is FrameworkStatus.RAN and not self.passed:
            raise ValueError("a ran framework row must pass")
        if self.status is FrameworkStatus.FAILED and self.passed:
            raise ValueError("a failed framework row cannot pass")
        if self.max_abs_error is not None:
            _non_negative_finite("max_abs_error", self.max_abs_error)

    @property
    def gate_passed(self) -> bool:
        """Return whether this row satisfies its required/optional gate."""
        if not self.required:
            return self.status is not FrameworkStatus.FAILED
        return self.status is FrameworkStatus.RAN and self.passed is True

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready framework row."""
        return {
            "family": self.family.value,
            "framework": self.framework,
            "status": self.status.value,
            "required": self.required,
            "executed": self.executed,
            "passed": self.passed,
            "gate_passed": self.gate_passed,
            "reason": self.reason,
            "max_abs_error": self.max_abs_error,
        }


@dataclass(frozen=True, slots=True)
class ConvergenceSuiteEvidence:
    """Unified ML convergence certificates, framework rows, and notebook pointers."""

    certificates: tuple[ConvergenceCertificate, ...]
    framework_rows: tuple[FrameworkEvidenceRow, ...]
    notebook_pointers: tuple[tuple[ModelFamily, str], ...]
    schema: str = ML_CONVERGENCE_SCHEMA
    claim_boundary: str = ML_CONVERGENCE_CLAIM_BOUNDARY
    provider_execution: bool = False
    hardware_execution: bool = False

    def __post_init__(self) -> None:
        """Require complete, unique QNN/QGNN/QSNN evidence without blank cells."""
        families = {certificate.spec.family for certificate in self.certificates}
        if len(self.certificates) != len(families):
            raise ValueError("suite certificates must not repeat a model family")
        if families != set(ModelFamily):
            raise ValueError("suite must contain exactly one certificate per model family")
        row_families = {row.family for row in self.framework_rows}
        if row_families != set(ModelFamily):
            raise ValueError("framework matrix must cover every model family")
        notebook_families = {family for family, _path in self.notebook_pointers}
        if notebook_families != set(ModelFamily):
            raise ValueError("notebook pointers must cover every model family")
        if any(not path.strip() for _family, path in self.notebook_pointers):
            raise ValueError("notebook pointer values must be non-empty")
        if (
            self.schema != ML_CONVERGENCE_SCHEMA
            or self.claim_boundary != ML_CONVERGENCE_CLAIM_BOUNDARY
        ):
            raise ValueError("suite schema and claim boundary are fixed")
        if self.provider_execution or self.hardware_execution:
            raise ValueError(
                "ML convergence evidence must not record provider or hardware execution"
            )

    @property
    def passed(self) -> bool:
        """Return whether every convergence and required-framework gate passed."""
        return all(certificate.passed for certificate in self.certificates) and all(
            row.gate_passed for row in self.framework_rows
        )

    def to_payload(self) -> dict[str, object]:
        """Return the digestable JSON payload without an integrity digest."""
        return {
            "schema": self.schema,
            "claim_boundary": self.claim_boundary,
            "passed": self.passed,
            "provider_execution": self.provider_execution,
            "hardware_execution": self.hardware_execution,
            "certificates": [certificate.to_dict() for certificate in self.certificates],
            "framework_rows": [row.to_dict() for row in self.framework_rows],
            "notebook_pointers": [
                {"family": family.value, "path": path} for family, path in self.notebook_pointers
            ],
        }


def _finite(name: str, value: float | None) -> float:
    if value is None or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _non_negative_finite(name: str, value: float) -> float:
    scalar = _finite(name, value)
    if scalar < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return scalar


__all__ = [
    "ML_CONVERGENCE_CLAIM_BOUNDARY",
    "ML_CONVERGENCE_SCHEMA",
    "ConvergenceCertificate",
    "ConvergenceExampleSpec",
    "ConvergenceSuiteEvidence",
    "FrameworkEvidenceRow",
    "FrameworkStatus",
    "ModelFamily",
]
