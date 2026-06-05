# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Provider Hardware Gradient Audit
"""Tests for provider hardware-gradient preparation readiness audits."""

from __future__ import annotations

import json

from scpn_quantum_control.phase import (
    ProviderHardwareGradientPreparationAuditResult,
    default_provider_hardware_gradient_preparation_scenarios,
    run_provider_hardware_gradient_preparation_audit,
)


def test_provider_hardware_gradient_preparation_audit_records_boundaries() -> None:
    audit = run_provider_hardware_gradient_preparation_audit()
    payload = audit.to_dict()

    assert isinstance(audit, ProviderHardwareGradientPreparationAuditResult)
    assert audit.passed
    assert audit.record_count == 6
    assert audit.approved_count == 2
    assert audit.blocked_count == 4
    assert audit.hardware_execution_count == 0
    assert audit.gradient_available_count == 0
    assert {record.scenario.name for record in audit.records} == {
        "bounded_dry_run_preparation",
        "ticketed_live_preparation",
        "missing_evidence_preparation",
        "shot_budget_exceeded_preparation",
        "unknown_provider_backend_preparation",
        "live_without_ticket_preparation",
    }
    assert json.loads(json.dumps(payload))["passed"] is True


def test_provider_hardware_gradient_preparation_audit_keeps_live_claim_closed() -> None:
    audit = run_provider_hardware_gradient_preparation_audit()
    ticketed = next(
        record for record in audit.records if record.scenario.name == "ticketed_live_preparation"
    )

    assert ticketed.passed
    assert ticketed.result.approved
    assert ticketed.result.mode == "live_ticketed"
    assert ticketed.result.hardware_execution is False
    assert ticketed.result.gradient_available is False
    assert "execution remains outside this policy record" in ticketed.result.claim_boundary


def test_provider_hardware_gradient_preparation_audit_exposes_blocked_reasons() -> None:
    audit = run_provider_hardware_gradient_preparation_audit()
    blocked = {record.scenario.name: record for record in audit.blocked_records}

    assert (
        "missing required evidence IDs" in blocked["missing_evidence_preparation"].failure_reason
    )
    assert "estimated total shots" in blocked["shot_budget_exceeded_preparation"].failure_reason
    assert "not allowlisted" in blocked["unknown_provider_backend_preparation"].failure_reason
    assert "live_execution_ticket" in blocked["live_without_ticket_preparation"].failure_reason


def test_default_provider_hardware_gradient_preparation_scenarios_are_json_ready() -> None:
    scenarios = default_provider_hardware_gradient_preparation_scenarios()
    payload = [scenario.to_dict() for scenario in scenarios]

    assert len(scenarios) == 6
    assert payload[0]["expected_approved"] is True
    assert payload[1]["dry_run_only"] is False
    assert payload[1]["has_live_execution_ticket"] is True
