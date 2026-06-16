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
from typing import cast

import scpn_quantum_control.phase as phase
from scpn_quantum_control.phase import (
    DifferentiableProviderHardwareSafetyAuditResult,
    run_differentiable_provider_hardware_safety_audit,
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
    assert audit.ready_for_hardware_gradient_promotion
    assert audit.promotion_blockers == ()
    assert payload["live_execution_ticket"] == "LIVE-2026-06-16-001"
