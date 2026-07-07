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
from datetime import datetime, timezone
from typing import Any, TypeAlias

from .hardware_gradient_campaign import run_hardware_gradient_campaign_readiness_suite
from .provider_gradient_audit import run_provider_gradient_readiness_audit
from .provider_hardware_gradient_audit import (
    run_provider_hardware_gradient_preparation_audit,
)
from .qnode_provider_transforms import run_provider_qnode_transform_readiness_suite
from .qnode_tape import run_phase_qnode_tape_readiness_suite

DIFFERENTIABLE_PROVIDER_HARDWARE_SAFETY_REVIEW_AS_OF_UTC = "2026-06-27T00:00:00Z"


@dataclass(frozen=True)
class _DifferentiableProviderHardwareEvidenceChain:
    """Validated provider/hardware evidence chain for promotion review."""

    live_execution_ticket: str
    provider_name: str
    backend_id: str
    job_id: str
    circuit_fingerprint: str
    provider_allowlist_id: str
    shot_budget_id: str
    raw_count_replay_artifact_id: str
    raw_count_replay_digest: str
    raw_count_shots: int
    calibration_snapshot_artifact_id: str
    calibration_snapshot_digest: str
    statevector_comparison_artifact_id: str
    statevector_comparison_digest: str
    isolated_benchmark_artifact_id: str
    captured_at_utc: str
    valid_until_utc: str
    claim_boundary: str = "differentiable_provider_hardware_evidence_chain"

    def __post_init__(self) -> None:
        """Validate artifact identity, raw-count, digest, and freshness metadata."""
        for field_name in (
            "live_execution_ticket",
            "provider_name",
            "backend_id",
            "job_id",
            "circuit_fingerprint",
            "provider_allowlist_id",
            "shot_budget_id",
            "raw_count_replay_artifact_id",
            "calibration_snapshot_artifact_id",
            "statevector_comparison_artifact_id",
            "isolated_benchmark_artifact_id",
            "claim_boundary",
        ):
            object.__setattr__(
                self,
                field_name,
                _normalise_metadata_text(field_name, getattr(self, field_name)),
            )
        if (
            isinstance(self.raw_count_shots, bool)
            or not isinstance(self.raw_count_shots, int)
            or self.raw_count_shots <= 0
        ):
            raise ValueError("raw_count_shots must be a positive integer")
        for field_name in (
            "raw_count_replay_digest",
            "calibration_snapshot_digest",
            "statevector_comparison_digest",
        ):
            object.__setattr__(
                self,
                field_name,
                _normalise_sha256_digest(field_name, getattr(self, field_name)),
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

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready provider/hardware evidence-chain metadata."""
        return {
            "live_execution_ticket": self.live_execution_ticket,
            "provider_name": self.provider_name,
            "backend_id": self.backend_id,
            "job_id": self.job_id,
            "circuit_fingerprint": self.circuit_fingerprint,
            "provider_allowlist_id": self.provider_allowlist_id,
            "shot_budget_id": self.shot_budget_id,
            "raw_count_replay_artifact_id": self.raw_count_replay_artifact_id,
            "raw_count_replay_digest": self.raw_count_replay_digest,
            "raw_count_shots": self.raw_count_shots,
            "calibration_snapshot_artifact_id": self.calibration_snapshot_artifact_id,
            "calibration_snapshot_digest": self.calibration_snapshot_digest,
            "statevector_comparison_artifact_id": self.statevector_comparison_artifact_id,
            "statevector_comparison_digest": self.statevector_comparison_digest,
            "isolated_benchmark_artifact_id": self.isolated_benchmark_artifact_id,
            "captured_at_utc": self.captured_at_utc,
            "valid_until_utc": self.valid_until_utc,
            "claim_boundary": self.claim_boundary,
        }


DifferentiableProviderHardwareEvidenceChain: TypeAlias = (
    _DifferentiableProviderHardwareEvidenceChain
)


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
    evidence_chain: DifferentiableProviderHardwareEvidenceChain | None = None
    evidence_review_as_of_utc: str = DIFFERENTIABLE_PROVIDER_HARDWARE_SAFETY_REVIEW_AS_OF_UTC
    claim_boundary: str = "differentiable_provider_hardware_safety_audit"

    def __post_init__(self) -> None:
        if not self.surfaces:
            raise ValueError("at least one safety surface is required")
        if any(
            not isinstance(surface, DifferentiableProviderHardwareSafetySurface)
            for surface in self.surfaces
        ):
            raise ValueError("surfaces must contain DifferentiableProviderHardwareSafetySurface")
        object.__setattr__(
            self,
            "evidence_review_as_of_utc",
            _normalise_utc_timestamp("evidence_review_as_of_utc", self.evidence_review_as_of_utc),
        )
        if self.evidence_chain is not None:
            if not isinstance(self.evidence_chain, DifferentiableProviderHardwareEvidenceChain):
                raise ValueError(
                    "evidence_chain must be a DifferentiableProviderHardwareEvidenceChain"
                )
            _validate_evidence_chain_freshness(
                self.evidence_chain,
                as_of_utc=self.evidence_review_as_of_utc,
            )
            _mirror_evidence_chain_fields(self)
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
        if self.evidence_chain is None:
            blockers.append("validated provider hardware evidence chain missing")
        return tuple(blockers)

    @property
    def ready_for_hardware_gradient_promotion(self) -> bool:
        """Return whether live hardware-gradient promotion evidence is complete."""
        return self.passed and not self.promotion_blockers

    @property
    def evidence_chain_ready(self) -> bool:
        """Return whether a validated promotion evidence chain is attached."""
        return self.evidence_chain is not None

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready aggregate safety evidence."""
        return {
            "passed": self.passed,
            "surface_count": self.surface_count,
            "hardware_execution_count": self.hardware_execution_count,
            "gradient_available_count": self.gradient_available_count,
            "requires_live_ticket": self.requires_live_ticket,
            "ready_for_hardware_gradient_promotion": self.ready_for_hardware_gradient_promotion,
            "evidence_chain_ready": self.evidence_chain_ready,
            "promotion_blockers": list(self.promotion_blockers),
            "live_execution_ticket": self.live_execution_ticket,
            "raw_count_replay_artifact_id": self.raw_count_replay_artifact_id,
            "calibration_snapshot_artifact_id": self.calibration_snapshot_artifact_id,
            "statevector_comparison_artifact_id": self.statevector_comparison_artifact_id,
            "isolated_benchmark_artifact_id": self.isolated_benchmark_artifact_id,
            "evidence_chain": None
            if self.evidence_chain is None
            else self.evidence_chain.to_dict(),
            "evidence_review_as_of_utc": self.evidence_review_as_of_utc,
            "claim_boundary": self.claim_boundary,
            "surfaces": [surface.to_dict() for surface in self.surfaces],
        }


def run_differentiable_provider_hardware_safety_audit(
    *,
    evidence_chain: DifferentiableProviderHardwareEvidenceChain | None = None,
    evidence_review_as_of_utc: str = DIFFERENTIABLE_PROVIDER_HARDWARE_SAFETY_REVIEW_AS_OF_UTC,
    live_execution_ticket: str | None = None,
    raw_count_replay_artifact_id: str | None = None,
    calibration_snapshot_artifact_id: str | None = None,
    statevector_comparison_artifact_id: str | None = None,
    isolated_benchmark_artifact_id: str | None = None,
) -> DifferentiableProviderHardwareSafetyAuditResult:
    """Run the aggregate dry-run and policy-gated provider/hardware safety audit."""
    if evidence_chain is not None:
        if any(
            value is not None
            for value in (
                live_execution_ticket,
                raw_count_replay_artifact_id,
                calibration_snapshot_artifact_id,
                statevector_comparison_artifact_id,
                isolated_benchmark_artifact_id,
            )
        ):
            raise ValueError(
                "evidence_chain cannot be combined with legacy promotion artifact IDs"
            )
        _validate_evidence_chain_freshness(
            evidence_chain,
            as_of_utc=evidence_review_as_of_utc,
        )
        live_execution_ticket = evidence_chain.live_execution_ticket
        raw_count_replay_artifact_id = evidence_chain.raw_count_replay_artifact_id
        calibration_snapshot_artifact_id = evidence_chain.calibration_snapshot_artifact_id
        statevector_comparison_artifact_id = evidence_chain.statevector_comparison_artifact_id
        isolated_benchmark_artifact_id = evidence_chain.isolated_benchmark_artifact_id
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
        evidence_chain=evidence_chain,
        evidence_review_as_of_utc=evidence_review_as_of_utc,
    )


def _mirror_evidence_chain_fields(
    audit: DifferentiableProviderHardwareSafetyAuditResult,
) -> None:
    evidence_chain = audit.evidence_chain
    if evidence_chain is None:
        return
    field_map = {
        "live_execution_ticket": evidence_chain.live_execution_ticket,
        "raw_count_replay_artifact_id": evidence_chain.raw_count_replay_artifact_id,
        "calibration_snapshot_artifact_id": evidence_chain.calibration_snapshot_artifact_id,
        "statevector_comparison_artifact_id": evidence_chain.statevector_comparison_artifact_id,
        "isolated_benchmark_artifact_id": evidence_chain.isolated_benchmark_artifact_id,
    }
    for field_name, chain_value in field_map.items():
        current_value = getattr(audit, field_name)
        if current_value is None:
            object.__setattr__(audit, field_name, chain_value)
        elif current_value != chain_value:
            raise ValueError(f"{field_name} must match evidence_chain")


def _normalise_metadata_text(field_name: str, value: object) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} must be non-empty")
    if any(ord(character) < 32 or ord(character) == 127 for character in text):
        raise ValueError(f"{field_name} must not contain control characters")
    return text


def _normalise_sha256_digest(field_name: str, value: object) -> str:
    digest = str(value)
    hex_digest = digest.removeprefix("sha256:")
    if not (
        digest.startswith("sha256:")
        and len(hex_digest) == 64
        and all(char in "0123456789abcdefABCDEF" for char in hex_digest)
    ):
        raise ValueError(f"{field_name} must be a sha256:<64-hex> digest")
    return f"sha256:{hex_digest.lower()}"


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


def _validate_evidence_chain_freshness(
    evidence_chain: DifferentiableProviderHardwareEvidenceChain,
    *,
    as_of_utc: str,
) -> None:
    valid_until = _utc_timestamp(
        "evidence_chain.valid_until_utc",
        evidence_chain.valid_until_utc,
    )
    if valid_until <= _utc_timestamp("evidence_review_as_of_utc", as_of_utc):
        raise ValueError("evidence_chain.valid_until_utc is stale for the review cutoff")


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
    "DifferentiableProviderHardwareEvidenceChain",
    "DifferentiableProviderHardwareSafetyAuditResult",
    "DifferentiableProviderHardwareSafetySurface",
    "run_differentiable_provider_hardware_safety_audit",
]
