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
from typing import cast

import pytest

import scpn_quantum_control.phase as phase
from scpn_quantum_control.phase import (
    DifferentiableProviderHardwareEvidenceChain,
    DifferentiableProviderHardwareSafetyAuditResult,
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


def test_provider_hardware_safety_audit_aggregates_all_differentiable_surfaces() -> None:
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
    chain = _hardware_evidence_chain(valid_until_utc="2026-06-21T00:00:00Z")

    with pytest.raises(ValueError, match="evidence_chain.valid_until_utc"):
        run_differentiable_provider_hardware_safety_audit(evidence_chain=chain)


def test_provider_hardware_safety_audit_rejects_mixed_legacy_and_chain_inputs() -> None:
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
    with pytest.raises(ValueError, match=match):
        factory()
