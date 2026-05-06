# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Feedback capability probes
"""No-submit capability probes for S1 feedback target selection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from .feedback_submission import FeedbackSubmissionPackage

CapabilityStatus = Literal["ready", "blocked", "unknown"]


@dataclass(frozen=True)
class BackendCapabilitySnapshot:
    """Provider metadata snapshot needed for S1 without submitting jobs."""

    provider: str
    backend_name: str
    n_qubits: int
    basis_gates: tuple[str, ...] = field(default_factory=tuple)
    supported_features: tuple[str, ...] = field(default_factory=tuple)
    max_shots: int | None = None
    max_circuits: int | None = None
    simulator: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.provider:
            raise ValueError("provider must be non-empty")
        if not self.backend_name:
            raise ValueError("backend_name must be non-empty")
        if self.n_qubits < 1:
            raise ValueError("n_qubits must be positive")
        if self.max_shots is not None and self.max_shots < 1:
            raise ValueError("max_shots must be positive when provided")
        if self.max_circuits is not None and self.max_circuits < 1:
            raise ValueError("max_circuits must be positive when provided")


@dataclass(frozen=True)
class FeedbackCapabilityDecision:
    """Readiness decision produced from backend metadata."""

    backend: BackendCapabilitySnapshot
    status: CapabilityStatus
    reasons: tuple[str, ...]
    required_features: tuple[str, ...]
    missing_features: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Serialise the capability decision."""
        return {
            "provider": self.backend.provider,
            "backend_name": self.backend.backend_name,
            "status": self.status,
            "reasons": list(self.reasons),
            "required_features": list(self.required_features),
            "missing_features": list(self.missing_features),
            "n_qubits": self.backend.n_qubits,
            "max_shots": self.backend.max_shots,
            "max_circuits": self.backend.max_circuits,
            "simulator": self.backend.simulator,
        }


def required_s1_dynamic_features(package: FeedbackSubmissionPackage) -> tuple[str, ...]:
    """Return feature flags required by the S1 dynamic-circuit package."""
    features = ["cross_shot_batches"]
    if package.circuit.has_mid_circuit_measurement:
        features.append("mid_circuit_measurement")
    if package.circuit.has_conditional_control:
        features.append("conditional_control")
    if package.circuit.has_conditional_reset:
        features.append("conditional_reset")
    return tuple(features)


def assess_feedback_backend_capability(
    snapshot: BackendCapabilitySnapshot,
    package: FeedbackSubmissionPackage,
) -> FeedbackCapabilityDecision:
    """Assess one backend snapshot without submitting a job."""
    required = required_s1_dynamic_features(package)
    supported = set(snapshot.supported_features)
    missing = tuple(feature for feature in required if feature not in supported)
    reasons: list[str] = []
    if snapshot.n_qubits < package.circuit.n_qubits:
        reasons.append(
            f"backend has {snapshot.n_qubits} qubits but package requires {package.circuit.n_qubits}"
        )
    if snapshot.max_shots is not None and snapshot.max_shots < package.budget.shots_per_circuit:
        reasons.append(
            f"backend max_shots={snapshot.max_shots} below required {package.budget.shots_per_circuit}"
        )
    if snapshot.max_circuits is not None and snapshot.max_circuits < package.budget.circuits:
        reasons.append(
            f"backend max_circuits={snapshot.max_circuits} below required {package.budget.circuits}"
        )
    for feature in missing:
        reasons.append(f"missing required feature: {feature}")
    if not snapshot.supported_features:
        status: CapabilityStatus = "unknown"
        reasons.append("backend did not declare supported_features")
    elif reasons:
        status = "blocked"
    else:
        status = "ready"
        reasons.append("backend metadata satisfies S1 dynamic-circuit requirements")
    return FeedbackCapabilityDecision(
        backend=snapshot,
        status=status,
        reasons=tuple(reasons),
        required_features=required,
        missing_features=missing,
    )


def assess_feedback_backend_fleet(
    snapshots: Sequence[BackendCapabilitySnapshot],
    package: FeedbackSubmissionPackage,
) -> tuple[FeedbackCapabilityDecision, ...]:
    """Assess a fleet of backend metadata snapshots."""
    return tuple(assess_feedback_backend_capability(snapshot, package) for snapshot in snapshots)
