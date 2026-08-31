# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for provider hardware safety audit
"""Tests for the aggregate differentiable provider/hardware safety gate."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from typing import cast

import pytest

import scpn_quantum_control.phase as phase
from scpn_quantum_control.phase import (
    DifferentiableProviderHardwareEvidenceChain,
    DifferentiableProviderHardwareSafetyAuditResult,
    DifferentiableProviderHardwareSafetySurface,
    run_differentiable_provider_hardware_safety_audit,
)


def _hardware_evidence_chain(
    *,
    valid_until_utc: str = "2026-07-20T00:00:00Z",
) -> DifferentiableProviderHardwareEvidenceChain:
    return DifferentiableProviderHardwareEvidenceChain(
        live_execution_ticket="LIVE-2026-06-16-001",
        provider_name="ibm_quantum",
        backend_id="ibm_kingston",
        job_id="job-20260616-001",
        circuit_fingerprint="phase-qnode:ry-rx-pauli-z:v1",
        provider_allowlist_id="allowlist-heron-r2-20260616",
        shot_budget_id="shot-budget-4096-20260616",
        raw_count_replay_artifact_id="raw-counts-001",
        raw_count_replay_digest="sha256:" + "a" * 64,
        raw_count_shots=4096,
        calibration_snapshot_artifact_id="calibration-001",
        calibration_snapshot_digest="sha256:" + "b" * 64,
        statevector_comparison_artifact_id="statevector-001",
        statevector_comparison_digest="sha256:" + "c" * 64,
        isolated_benchmark_artifact_id="isolated-001",
        captured_at_utc="2026-06-16T00:00:00Z",
        valid_until_utc=valid_until_utc,
    )


def _safety_surface() -> DifferentiableProviderHardwareSafetySurface:
    return DifferentiableProviderHardwareSafetySurface(
        name="provider_gradient_readiness",
        passed=True,
        record_count=1,
        supported_count=1,
        blocked_count=0,
        hardware_execution_count=0,
        gradient_available_count=0,
        claim_boundary="provider_gradient_readiness",
        payload={"passed": True},
    )


def test_provider_hardware_safety_audit_aggregates_all_differentiable_surfaces() -> None:
    """Aggregate every offline surface without claiming hardware execution."""
    audit = run_differentiable_provider_hardware_safety_audit()
    payload = audit.to_dict()
    surface_names = {surface.name for surface in audit.surfaces}

    assert isinstance(audit, DifferentiableProviderHardwareSafetyAuditResult)
    assert audit.passed
    assert audit.surface_count == 5
    assert audit.hardware_execution_count == 0
    assert audit.gradient_available_count == 0
    assert audit.requires_live_ticket
    assert audit.ready_for_hardware_gradient_promotion is False
    assert surface_names == {
        "provider_gradient_readiness",
        "provider_hardware_gradient_preparation",
        "provider_qnode_transform_readiness",
        "phase_qnode_tape_readiness",
        "hardware_gradient_campaign_readiness",
    }
    assert payload["claim_boundary"] == "differentiable_provider_hardware_safety_audit"
    assert payload["promotion_blockers"] == [
        "live execution ticket missing",
        "raw-count replay artefact missing",
        "calibration snapshot artefact missing",
        "statevector comparison artefact missing",
        "isolated benchmark artefact missing",
        "validated provider hardware evidence chain missing",
    ]


def test_provider_hardware_safety_audit_payload_is_json_ready_and_exported() -> None:
    """Serialize the aggregate and expose its public phase-package symbols."""
    audit = run_differentiable_provider_hardware_safety_audit()
    payload = audit.to_dict()
    round_trip = json.loads(json.dumps(payload))
    surfaces = cast(list[dict[str, object]], round_trip["surfaces"])

    assert round_trip["passed"] is True
    assert surfaces[0]["passed"] is True
    assert surfaces[1]["hardware_execution_count"] == 0
    assert phase.run_differentiable_provider_hardware_safety_audit is (
        run_differentiable_provider_hardware_safety_audit
    )
    assert phase.DifferentiableProviderHardwareSafetyAuditResult is (
        DifferentiableProviderHardwareSafetyAuditResult
    )
    assert phase.DifferentiableProviderHardwareEvidenceChain is (
        DifferentiableProviderHardwareEvidenceChain
    )


def test_provider_hardware_safety_audit_requires_artifacts_for_promotion() -> None:
    """Keep legacy artifact identifiers insufficient for promotion."""
    audit = run_differentiable_provider_hardware_safety_audit(
        live_execution_ticket="LIVE-2026-06-16-001",
        raw_count_replay_artifact_id="raw-counts-001",
        calibration_snapshot_artifact_id="calibration-001",
        statevector_comparison_artifact_id="statevector-001",
        isolated_benchmark_artifact_id="isolated-001",
    )
    payload = audit.to_dict()

    assert audit.passed
    assert not audit.ready_for_hardware_gradient_promotion
    assert audit.promotion_blockers == ("validated provider hardware evidence chain missing",)
    assert payload["live_execution_ticket"] == "LIVE-2026-06-16-001"
    assert payload["evidence_chain_ready"] is False


def test_provider_hardware_safety_audit_accepts_validated_evidence_chain() -> None:
    """Accept a fresh, internally consistent evidence chain."""
    chain = _hardware_evidence_chain()

    audit = run_differentiable_provider_hardware_safety_audit(evidence_chain=chain)
    payload = audit.to_dict()
    chain_payload = cast(dict[str, object], payload["evidence_chain"])

    assert audit.passed
    assert audit.evidence_chain is chain
    assert audit.ready_for_hardware_gradient_promotion
    assert audit.promotion_blockers == ()
    assert audit.live_execution_ticket == "LIVE-2026-06-16-001"
    assert audit.raw_count_replay_artifact_id == "raw-counts-001"
    assert payload["evidence_chain_ready"] is True
    assert payload["evidence_review_as_of_utc"] == "2026-06-27T00:00:00Z"
    assert chain_payload["backend_id"] == "ibm_kingston"
    assert chain_payload["raw_count_shots"] == 4096


def test_provider_hardware_safety_audit_rejects_stale_evidence_chain() -> None:
    """Reject evidence that expired before the review cutoff."""
    chain = _hardware_evidence_chain(valid_until_utc="2026-06-21T00:00:00Z")

    with pytest.raises(ValueError, match="evidence_chain.valid_until_utc"):
        run_differentiable_provider_hardware_safety_audit(evidence_chain=chain)


def test_provider_hardware_safety_audit_rejects_mixed_legacy_and_chain_inputs() -> None:
    """Reject ambiguous mixtures of legacy identifiers and a sealed chain."""
    with pytest.raises(ValueError, match="evidence_chain"):
        run_differentiable_provider_hardware_safety_audit(
            evidence_chain=_hardware_evidence_chain(),
            live_execution_ticket="LIVE-2026-06-16-001",
        )


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (
            lambda: _hardware_evidence_chain(valid_until_utc="2026-06-15T00:00:00Z"),
            "valid_until_utc",
        ),
        (
            lambda: DifferentiableProviderHardwareEvidenceChain(
                live_execution_ticket="LIVE-2026-06-16-001",
                provider_name="ibm_quantum",
                backend_id="ibm_kingston",
                job_id="job-20260616-001",
                circuit_fingerprint="phase-qnode:ry-rx-pauli-z:v1",
                provider_allowlist_id="allowlist-heron-r2-20260616",
                shot_budget_id="shot-budget-4096-20260616",
                raw_count_replay_artifact_id="raw-counts-001",
                raw_count_replay_digest="not-a-digest",
                raw_count_shots=4096,
                calibration_snapshot_artifact_id="calibration-001",
                calibration_snapshot_digest="sha256:" + "b" * 64,
                statevector_comparison_artifact_id="statevector-001",
                statevector_comparison_digest="sha256:" + "c" * 64,
                isolated_benchmark_artifact_id="isolated-001",
                captured_at_utc="2026-06-16T00:00:00Z",
                valid_until_utc="2026-07-20T00:00:00Z",
            ),
            "raw_count_replay_digest",
        ),
        (
            lambda: DifferentiableProviderHardwareEvidenceChain(
                live_execution_ticket="LIVE-2026-06-16-001",
                provider_name="ibm_quantum",
                backend_id="ibm_kingston",
                job_id="job-20260616-001",
                circuit_fingerprint="phase-qnode:ry-rx-pauli-z:v1",
                provider_allowlist_id="allowlist-heron-r2-20260616",
                shot_budget_id="shot-budget-4096-20260616",
                raw_count_replay_artifact_id="raw-counts-001",
                raw_count_replay_digest="sha256:" + "a" * 64,
                raw_count_shots=True,
                calibration_snapshot_artifact_id="calibration-001",
                calibration_snapshot_digest="sha256:" + "b" * 64,
                statevector_comparison_artifact_id="statevector-001",
                statevector_comparison_digest="sha256:" + "c" * 64,
                isolated_benchmark_artifact_id="isolated-001",
                captured_at_utc="2026-06-16T00:00:00Z",
                valid_until_utc="2026-07-20T00:00:00Z",
            ),
            "raw_count_shots",
        ),
    ],
)
def test_provider_hardware_evidence_chain_rejects_malformed_fields(
    factory: Callable[[], object],
    match: str,
) -> None:
    """Reject malformed temporal, digest, and shot-count fields."""
    with pytest.raises(ValueError, match=match):
        factory()


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (lambda: replace(_safety_surface(), name=" "), "surface name"),
        (lambda: replace(_safety_surface(), record_count=-1), "record_count"),
        (lambda: replace(_safety_surface(), supported_count=-1), "supported_count"),
        (lambda: replace(_safety_surface(), blocked_count=-1), "supported_count"),
        (
            lambda: replace(_safety_surface(), hardware_execution_count=-1),
            "hardware_execution_count",
        ),
        (
            lambda: replace(_safety_surface(), gradient_available_count=-1),
            "hardware_execution_count",
        ),
        (lambda: replace(_safety_surface(), claim_boundary=" "), "claim_boundary"),
    ],
)
def test_provider_hardware_safety_surface_rejects_invalid_contracts(
    factory: Callable[[], object],
    match: str,
) -> None:
    """Reject blank identities and negative safety-surface counts."""
    with pytest.raises(ValueError, match=match):
        factory()


def test_provider_hardware_safety_surface_normalizes_public_metadata() -> None:
    """Normalize surface identity fields and preserve JSON-ready payloads."""
    surface = replace(
        _safety_surface(),
        name="  provider_gradient_readiness  ",
        claim_boundary="  provider_gradient_readiness  ",
    )

    assert surface.name == "provider_gradient_readiness"
    assert surface.claim_boundary == "provider_gradient_readiness"
    assert surface.to_dict()["payload"] == {"passed": True}


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (
            lambda: DifferentiableProviderHardwareSafetyAuditResult(
                surfaces=(),
                live_execution_ticket=None,
                raw_count_replay_artifact_id=None,
                calibration_snapshot_artifact_id=None,
                statevector_comparison_artifact_id=None,
                isolated_benchmark_artifact_id=None,
            ),
            "at least one safety surface",
        ),
        (
            lambda: DifferentiableProviderHardwareSafetyAuditResult(
                surfaces=(cast(DifferentiableProviderHardwareSafetySurface, object()),),
                live_execution_ticket=None,
                raw_count_replay_artifact_id=None,
                calibration_snapshot_artifact_id=None,
                statevector_comparison_artifact_id=None,
                isolated_benchmark_artifact_id=None,
            ),
            "surfaces must contain",
        ),
        (
            lambda: DifferentiableProviderHardwareSafetyAuditResult(
                surfaces=(_safety_surface(),),
                live_execution_ticket=None,
                raw_count_replay_artifact_id=None,
                calibration_snapshot_artifact_id=None,
                statevector_comparison_artifact_id=None,
                isolated_benchmark_artifact_id=None,
                evidence_chain=cast(DifferentiableProviderHardwareEvidenceChain, object()),
            ),
            "evidence_chain must be",
        ),
        (
            lambda: DifferentiableProviderHardwareSafetyAuditResult(
                surfaces=(_safety_surface(),),
                live_execution_ticket=" ",
                raw_count_replay_artifact_id=None,
                calibration_snapshot_artifact_id=None,
                statevector_comparison_artifact_id=None,
                isolated_benchmark_artifact_id=None,
            ),
            "live_execution_ticket",
        ),
        (
            lambda: DifferentiableProviderHardwareSafetyAuditResult(
                surfaces=(_safety_surface(),),
                live_execution_ticket=None,
                raw_count_replay_artifact_id=None,
                calibration_snapshot_artifact_id=None,
                statevector_comparison_artifact_id=None,
                isolated_benchmark_artifact_id=None,
                claim_boundary=" ",
            ),
            "claim_boundary",
        ),
        (
            lambda: DifferentiableProviderHardwareSafetyAuditResult(
                surfaces=(_safety_surface(),),
                live_execution_ticket="different-ticket",
                raw_count_replay_artifact_id=None,
                calibration_snapshot_artifact_id=None,
                statevector_comparison_artifact_id=None,
                isolated_benchmark_artifact_id=None,
                evidence_chain=_hardware_evidence_chain(),
            ),
            "live_execution_ticket must match",
        ),
    ],
)
def test_provider_hardware_safety_result_rejects_invalid_aggregate_contracts(
    factory: Callable[[], object],
    match: str,
) -> None:
    """Reject malformed surface collections, evidence, and aggregate metadata."""
    with pytest.raises(ValueError, match=match):
        factory()


def test_provider_hardware_safety_result_requires_passing_offline_surfaces() -> None:
    """Refuse promotion when a surface fails or reports hardware-side output."""
    baseline = run_differentiable_provider_hardware_safety_audit()
    failing = replace(baseline.surfaces[0], passed=False)
    hardware = replace(baseline.surfaces[0], hardware_execution_count=1)
    gradient = replace(baseline.surfaces[0], gradient_available_count=1)

    for surface in (failing, hardware, gradient):
        audit = replace(baseline, surfaces=(surface, *baseline.surfaces[1:]))
        assert audit.passed is False
        assert audit.ready_for_hardware_gradient_promotion is False


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (lambda: replace(_hardware_evidence_chain(), provider_name=" "), "provider_name"),
        (
            lambda: replace(_hardware_evidence_chain(), provider_name="ibm\nquantum"),
            "control characters",
        ),
        (
            lambda: replace(
                _hardware_evidence_chain(),
                raw_count_replay_digest="sha256:" + "a" * 63,
            ),
            "raw_count_replay_digest",
        ),
        (
            lambda: replace(
                _hardware_evidence_chain(),
                raw_count_replay_digest="sha256:" + "g" * 64,
            ),
            "raw_count_replay_digest",
        ),
        (
            lambda: replace(
                _hardware_evidence_chain(),
                raw_count_shots=cast(int, "4096"),
            ),
            "raw_count_shots",
        ),
        (
            lambda: replace(_hardware_evidence_chain(), raw_count_shots=-1),
            "raw_count_shots",
        ),
        (
            lambda: replace(_hardware_evidence_chain(), valid_until_utc="not-a-time"),
            "ISO-8601 UTC timestamp",
        ),
        (
            lambda: replace(
                _hardware_evidence_chain(),
                valid_until_utc="2026-07-20T00:00:00",
            ),
            "UTC offset",
        ),
    ],
)
def test_provider_hardware_evidence_chain_rejects_remaining_invalid_fields(
    factory: Callable[[], object],
    match: str,
) -> None:
    """Reject blank, controlled, malformed, untyped, and offset-free metadata."""
    with pytest.raises(ValueError, match=match):
        factory()


def test_provider_hardware_evidence_chain_normalizes_digest_and_timestamp() -> None:
    """Canonicalize uppercase digests and non-UTC offsets."""
    chain = replace(
        _hardware_evidence_chain(),
        raw_count_replay_digest="sha256:" + "A" * 64,
        captured_at_utc="2026-06-16T02:00:00+02:00",
    )

    assert chain.raw_count_replay_digest == "sha256:" + "a" * 64
    assert chain.captured_at_utc == "2026-06-16T00:00:00Z"


def test_provider_hardware_safety_result_mirrors_public_chain_fields() -> None:
    """Populate legacy read fields from a directly attached public chain."""
    chain = _hardware_evidence_chain()
    audit = DifferentiableProviderHardwareSafetyAuditResult(
        surfaces=(_safety_surface(),),
        live_execution_ticket=None,
        raw_count_replay_artifact_id=None,
        calibration_snapshot_artifact_id=None,
        statevector_comparison_artifact_id=None,
        isolated_benchmark_artifact_id=None,
        evidence_chain=chain,
    )

    assert audit.live_execution_ticket == chain.live_execution_ticket
    assert audit.raw_count_replay_artifact_id == chain.raw_count_replay_artifact_id
    assert audit.calibration_snapshot_artifact_id == chain.calibration_snapshot_artifact_id
    assert audit.statevector_comparison_artifact_id == chain.statevector_comparison_artifact_id
    assert audit.isolated_benchmark_artifact_id == chain.isolated_benchmark_artifact_id
