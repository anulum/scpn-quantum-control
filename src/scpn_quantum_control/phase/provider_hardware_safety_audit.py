# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Provider Hardware Safety Audit
"""Aggregate safety gate for differentiable provider and hardware-gradient paths."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .hardware_gradient_campaign import run_hardware_gradient_campaign_readiness_suite
from .provider_gradient_audit import run_provider_gradient_readiness_audit
from .provider_hardware_gradient_audit import (
    run_provider_hardware_gradient_preparation_audit,
)
from .qnode_provider_transforms import run_provider_qnode_transform_readiness_suite
from .qnode_tape import run_phase_qnode_tape_readiness_suite


@dataclass(frozen=True)
class DifferentiableProviderHardwareSafetySurface:
    """One audited differentiable provider/hardware safety surface."""

    name: str
    passed: bool
    record_count: int
    supported_count: int
    blocked_count: int
    hardware_execution_count: int
    gradient_available_count: int
    claim_boundary: str
    payload: Mapping[str, object]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("surface name must be non-empty")
        if self.record_count < 0:
            raise ValueError("record_count must be non-negative")
        if self.supported_count < 0 or self.blocked_count < 0:
            raise ValueError("supported_count and blocked_count must be non-negative")
        if self.hardware_execution_count < 0 or self.gradient_available_count < 0:
            raise ValueError(
                "hardware_execution_count and gradient_available_count must be non-negative"
            )
        if not self.claim_boundary.strip():
            raise ValueError("claim_boundary must be non-empty")
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "claim_boundary", self.claim_boundary.strip())

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready surface metadata."""

        return {
            "name": self.name,
            "passed": self.passed,
            "record_count": self.record_count,
            "supported_count": self.supported_count,
            "blocked_count": self.blocked_count,
            "hardware_execution_count": self.hardware_execution_count,
            "gradient_available_count": self.gradient_available_count,
            "claim_boundary": self.claim_boundary,
            "payload": dict(self.payload),
        }


@dataclass(frozen=True)
class DifferentiableProviderHardwareSafetyAuditResult:
    """Cross-surface gate for differentiable provider and hardware-gradient safety."""

    surfaces: tuple[DifferentiableProviderHardwareSafetySurface, ...]
    live_execution_ticket: str | None
    raw_count_replay_artifact_id: str | None
    calibration_snapshot_artifact_id: str | None
    statevector_comparison_artifact_id: str | None
    isolated_benchmark_artifact_id: str | None
    claim_boundary: str = "differentiable_provider_hardware_safety_audit"

    def __post_init__(self) -> None:
        if not self.surfaces:
            raise ValueError("at least one safety surface is required")
        if any(
            not isinstance(surface, DifferentiableProviderHardwareSafetySurface)
            for surface in self.surfaces
        ):
            raise ValueError("surfaces must contain DifferentiableProviderHardwareSafetySurface")
        for field_name in (
            "live_execution_ticket",
            "raw_count_replay_artifact_id",
            "calibration_snapshot_artifact_id",
            "statevector_comparison_artifact_id",
            "isolated_benchmark_artifact_id",
        ):
            value = getattr(self, field_name)
            if value is not None and not value.strip():
                raise ValueError(f"{field_name} must be non-empty when provided")
        if not self.claim_boundary.strip():
            raise ValueError("claim_boundary must be non-empty")
        object.__setattr__(self, "claim_boundary", self.claim_boundary.strip())

    @property
    def surface_count(self) -> int:
        """Return audited surface count."""

        return len(self.surfaces)

    @property
    def passed(self) -> bool:
        """Return whether every audited safety surface preserves its boundary."""

        return (
            all(surface.passed for surface in self.surfaces)
            and self.hardware_execution_count == 0
            and self.gradient_available_count == 0
        )

    @property
    def hardware_execution_count(self) -> int:
        """Return total hardware executions observed by safety surfaces."""

        return sum(surface.hardware_execution_count for surface in self.surfaces)

    @property
    def gradient_available_count(self) -> int:
        """Return hardware-gradient results produced by safety surfaces."""

        return sum(surface.gradient_available_count for surface in self.surfaces)

    @property
    def requires_live_ticket(self) -> bool:
        """Return whether promotion still needs live-ticket governance."""

        return self.live_execution_ticket is None

    @property
    def promotion_blockers(self) -> tuple[str, ...]:
        """Return artefacts required before hardware-gradient promotion."""

        blockers: list[str] = []
        if self.live_execution_ticket is None:
            blockers.append("live execution ticket missing")
        if self.raw_count_replay_artifact_id is None:
            blockers.append("raw-count replay artefact missing")
        if self.calibration_snapshot_artifact_id is None:
            blockers.append("calibration snapshot artefact missing")
        if self.statevector_comparison_artifact_id is None:
            blockers.append("statevector comparison artefact missing")
        if self.isolated_benchmark_artifact_id is None:
            blockers.append("isolated benchmark artefact missing")
        return tuple(blockers)

    @property
    def ready_for_hardware_gradient_promotion(self) -> bool:
        """Return whether live hardware-gradient promotion evidence is complete."""

        return self.passed and not self.promotion_blockers

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready aggregate safety evidence."""

        return {
            "passed": self.passed,
            "surface_count": self.surface_count,
            "hardware_execution_count": self.hardware_execution_count,
            "gradient_available_count": self.gradient_available_count,
            "requires_live_ticket": self.requires_live_ticket,
            "ready_for_hardware_gradient_promotion": self.ready_for_hardware_gradient_promotion,
            "promotion_blockers": list(self.promotion_blockers),
            "live_execution_ticket": self.live_execution_ticket,
            "raw_count_replay_artifact_id": self.raw_count_replay_artifact_id,
            "calibration_snapshot_artifact_id": self.calibration_snapshot_artifact_id,
            "statevector_comparison_artifact_id": self.statevector_comparison_artifact_id,
            "isolated_benchmark_artifact_id": self.isolated_benchmark_artifact_id,
            "claim_boundary": self.claim_boundary,
            "surfaces": [surface.to_dict() for surface in self.surfaces],
        }


def run_differentiable_provider_hardware_safety_audit(
    *,
    live_execution_ticket: str | None = None,
    raw_count_replay_artifact_id: str | None = None,
    calibration_snapshot_artifact_id: str | None = None,
    statevector_comparison_artifact_id: str | None = None,
    isolated_benchmark_artifact_id: str | None = None,
) -> DifferentiableProviderHardwareSafetyAuditResult:
    """Run the aggregate dry-run and policy-gated provider/hardware safety audit."""

    provider_gradient = run_provider_gradient_readiness_audit()
    provider_preparation = run_provider_hardware_gradient_preparation_audit()
    qnode_transforms = run_provider_qnode_transform_readiness_suite()
    qnode_tape = run_phase_qnode_tape_readiness_suite()
    campaign = run_hardware_gradient_campaign_readiness_suite()
    return DifferentiableProviderHardwareSafetyAuditResult(
        surfaces=(
            _surface_from_provider_gradient(provider_gradient),
            _surface_from_provider_preparation(provider_preparation),
            _surface_from_qnode_transforms(qnode_transforms),
            _surface_from_qnode_tape(qnode_tape),
            _surface_from_campaign(campaign),
        ),
        live_execution_ticket=live_execution_ticket,
        raw_count_replay_artifact_id=raw_count_replay_artifact_id,
        calibration_snapshot_artifact_id=calibration_snapshot_artifact_id,
        statevector_comparison_artifact_id=statevector_comparison_artifact_id,
        isolated_benchmark_artifact_id=isolated_benchmark_artifact_id,
    )


def _surface_from_provider_gradient(audit: Any) -> DifferentiableProviderHardwareSafetySurface:
    return DifferentiableProviderHardwareSafetySurface(
        name="provider_gradient_readiness",
        passed=bool(audit.passed),
        record_count=len(audit.records),
        supported_count=len(audit.supported_records),
        blocked_count=len(audit.blocked_records),
        hardware_execution_count=0,
        gradient_available_count=0,
        claim_boundary=str(audit.claim_boundary),
        payload=audit.to_dict(),
    )


def _surface_from_provider_preparation(audit: Any) -> DifferentiableProviderHardwareSafetySurface:
    return DifferentiableProviderHardwareSafetySurface(
        name="provider_hardware_gradient_preparation",
        passed=bool(audit.passed),
        record_count=audit.record_count,
        supported_count=audit.approved_count,
        blocked_count=audit.blocked_count,
        hardware_execution_count=audit.hardware_execution_count,
        gradient_available_count=audit.gradient_available_count,
        claim_boundary=str(audit.claim_boundary),
        payload=audit.to_dict(),
    )


def _surface_from_qnode_transforms(audit: Any) -> DifferentiableProviderHardwareSafetySurface:
    return DifferentiableProviderHardwareSafetySurface(
        name="provider_qnode_transform_readiness",
        passed=bool(audit.passed),
        record_count=audit.record_count,
        supported_count=audit.supported_count,
        blocked_count=audit.fail_closed_count,
        hardware_execution_count=int(audit.hardware_execution),
        gradient_available_count=0,
        claim_boundary=str(audit.claim_boundary),
        payload=audit.to_dict(),
    )


def _surface_from_qnode_tape(audit: Any) -> DifferentiableProviderHardwareSafetySurface:
    return DifferentiableProviderHardwareSafetySurface(
        name="phase_qnode_tape_readiness",
        passed=bool(audit.passed),
        record_count=audit.record_count,
        supported_count=audit.supported_count,
        blocked_count=audit.fail_closed_count,
        hardware_execution_count=int(audit.hardware_execution),
        gradient_available_count=0,
        claim_boundary=str(audit.claim_boundary),
        payload=audit.to_dict(),
    )


def _surface_from_campaign(audit: Any) -> DifferentiableProviderHardwareSafetySurface:
    return DifferentiableProviderHardwareSafetySurface(
        name="hardware_gradient_campaign_readiness",
        passed=bool(audit.passed),
        record_count=audit.plan_count,
        supported_count=audit.approved_count,
        blocked_count=audit.blocked_count,
        hardware_execution_count=audit.hardware_execution_count,
        gradient_available_count=audit.gradient_available_count,
        claim_boundary=str(audit.claim_boundary),
        payload=audit.to_dict(),
    )


__all__ = [
    "DifferentiableProviderHardwareSafetyAuditResult",
    "DifferentiableProviderHardwareSafetySurface",
    "run_differentiable_provider_hardware_safety_audit",
]
