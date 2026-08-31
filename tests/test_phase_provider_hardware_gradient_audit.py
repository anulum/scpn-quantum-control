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
import math
from collections.abc import Callable
from dataclasses import replace
from typing import cast

import numpy as np
import pytest

from scpn_quantum_control.phase import (
    ProviderHardwareGradientPreparationAuditResult,
    ProviderHardwareGradientPreparationScenario,
    default_provider_hardware_gradient_preparation_scenarios,
    run_provider_hardware_gradient_preparation_audit,
)


def test_provider_hardware_gradient_preparation_audit_records_boundaries() -> None:
    """Execute all built-in approval and fail-closed preparation routes."""
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
    """Keep ticketed preparation separate from execution and gradients."""
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
    """Expose each policy refusal through the public audit result."""
    audit = run_provider_hardware_gradient_preparation_audit()
    blocked = {record.scenario.name: record for record in audit.blocked_records}

    assert (
        "missing required evidence IDs" in blocked["missing_evidence_preparation"].failure_reason
    )
    assert "estimated total shots" in blocked["shot_budget_exceeded_preparation"].failure_reason
    assert "not allowlisted" in blocked["unknown_provider_backend_preparation"].failure_reason
    assert "live_execution_ticket" in blocked["live_without_ticket_preparation"].failure_reason


def test_default_provider_hardware_gradient_preparation_scenarios_are_json_ready() -> None:
    """Serialize built-in scenarios without provider-side effects."""
    scenarios = default_provider_hardware_gradient_preparation_scenarios()
    payload = [scenario.to_dict() for scenario in scenarios]

    assert len(scenarios) == 6
    assert payload[0]["expected_approved"] is True
    assert payload[1]["dry_run_only"] is False
    assert payload[1]["has_live_execution_ticket"] is True


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda base: replace(base, name=" "), "scenario name"),
        (lambda base: replace(base, provider=" "), "scenario provider"),
        (lambda base: replace(base, backend=" "), "scenario backend"),
        (
            lambda base: replace(base, values=np.array([], dtype=np.float64)),
            "scenario values",
        ),
        (
            lambda base: replace(base, values=np.array([[0.2]], dtype=np.float64)),
            "scenario values",
        ),
        (
            lambda base: replace(base, values=np.array([math.nan], dtype=np.float64)),
            "scenario values",
        ),
        (lambda base: replace(base, shots=cast(int, True)), "scenario shots"),
        (lambda base: replace(base, shots=0), "scenario shots"),
    ],
)
def test_provider_hardware_gradient_scenario_validation_is_strict(
    mutate: Callable[
        [ProviderHardwareGradientPreparationScenario],
        ProviderHardwareGradientPreparationScenario,
    ],
    message: str,
) -> None:
    """Reject malformed identity, value-vector, and shot inputs."""
    base = default_provider_hardware_gradient_preparation_scenarios()[0]

    with pytest.raises(ValueError, match=message):
        mutate(base)


def test_provider_hardware_gradient_scenario_normalises_metadata() -> None:
    """Trim labels and retain only populated evidence identifiers."""
    base = default_provider_hardware_gradient_preparation_scenarios()[0]

    scenario = replace(
        base,
        name="  dry-run  ",
        provider=" ibm_quantum ",
        backend=" ibm_quantum ",
        evidence_ids={"calibration": " ready ", "": "drop", "empty": " "},
        description="  bounded preparation  ",
    )

    assert scenario.name == "dry-run"
    assert scenario.provider == "ibm_quantum"
    assert scenario.backend == "ibm_quantum"
    assert scenario.evidence_ids == {"calibration": "ready"}
    assert scenario.description == "bounded preparation"
    assert scenario.to_dict()["policy"] is None


def test_provider_hardware_gradient_scenario_accepts_absent_evidence_map() -> None:
    """Normalize an absent evidence mapping to an empty immutable record field."""
    base = default_provider_hardware_gradient_preparation_scenarios()[0]

    scenario = replace(base, evidence_ids=None)

    assert scenario.evidence_ids == {}
    assert scenario.to_dict()["evidence_ids"] == {}


def test_provider_hardware_gradient_audit_reports_expectation_mismatch() -> None:
    """Expose an approved preparation whose declared expectation is wrong."""
    base = default_provider_hardware_gradient_preparation_scenarios()[0]
    scenario = replace(base, name="unexpected_approval", expected_approved=False)

    audit = run_provider_hardware_gradient_preparation_audit((scenario,))
    record = audit.records[0]

    assert not audit.passed
    assert audit.record_count == 1
    assert audit.approved_count == 1
    assert audit.blocked_count == 0
    assert audit.failing_records == (record,)
    assert record.approved
    assert not record.blocked
    assert not record.passed
    assert record.to_dict()["failure_reason"] == ""


def test_provider_hardware_gradient_audit_empty_selection_uses_defaults() -> None:
    """Treat an empty optional selection as a request for built-in scenarios."""
    audit = run_provider_hardware_gradient_preparation_audit(())

    assert audit.passed
    assert audit.record_count == 6
