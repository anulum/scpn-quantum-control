# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Provider Gradient Readiness Audit
"""Tests for phase/provider_gradient_audit.py readiness evidence."""

from __future__ import annotations

from typing import cast

import pytest

from scpn_quantum_control.phase import (
    ProviderGradientReadinessAuditResult,
    ProviderGradientReadinessScenario,
    default_provider_gradient_readiness_scenarios,
    run_provider_gradient_readiness_audit,
)


def test_provider_gradient_readiness_audit_covers_supported_and_blocked_routes() -> None:
    audit = run_provider_gradient_readiness_audit()
    names = {record.scenario.name for record in audit.records}

    assert isinstance(audit, ProviderGradientReadinessAuditResult)
    assert audit.passed
    assert len(audit.records) == 6
    assert len(audit.supported_records) == 3
    assert len(audit.blocked_records) == 3
    assert audit.failing_records == ()
    assert names == {
        "statevector_parameter_shift",
        "finite_shot_parameter_shift",
        "multi_frequency_finite_shot",
        "hardware_without_policy",
        "unknown_backend",
        "finite_shot_missing_variance",
    }


def test_provider_gradient_readiness_audit_verifies_gradient_references() -> None:
    audit = run_provider_gradient_readiness_audit()

    for record in audit.supported_records:
        assert record.result is not None
        assert record.max_abs_error is not None
        assert record.max_abs_error <= 1e-10
        assert record.result.total_evaluations == record.plan.evaluations
        assert record.result.claim_boundary.startswith("provider callback")

    finite_shot = next(
        record
        for record in audit.supported_records
        if record.scenario.name == "finite_shot_parameter_shift"
    )
    assert finite_shot.result is not None
    assert finite_shot.result.total_shots == 1600
    assert finite_shot.result.standard_error[0] > 0.0
    finite_shot_metadata = finite_shot.result.records[0].plus.metadata
    assert finite_shot_metadata is not None
    assert finite_shot_metadata["source_class"] == "synthetic_fixture"
    assert finite_shot_metadata["shift_direction"] == "plus"

    multi_frequency = next(
        record
        for record in audit.supported_records
        if record.scenario.name == "multi_frequency_finite_shot"
    )
    assert multi_frequency.result is not None
    assert multi_frequency.plan.shift_terms == 2
    assert multi_frequency.result.total_evaluations == 4
    assert multi_frequency.result.total_shots == 1200
    multi_frequency_metadata = multi_frequency.result.records[1].minus.metadata
    assert multi_frequency_metadata is not None
    assert multi_frequency_metadata["shift_index"] == 1


def test_provider_gradient_readiness_audit_records_fail_closed_reasons() -> None:
    audit = run_provider_gradient_readiness_audit()
    blocked = {record.scenario.name: record for record in audit.blocked_records}

    hardware = blocked["hardware_without_policy"]
    unknown = blocked["unknown_backend"]
    malformed = blocked["finite_shot_missing_variance"]

    assert hardware.plan.fail_closed
    assert hardware.plan.requires_hardware_approval
    assert hardware.failure_reason is not None
    assert "hardware gradient execution requires" in hardware.failure_reason

    assert unknown.plan.fail_closed
    assert "unknown backend has no registered gradient capability" in unknown.plan.reasons
    assert "statevector_simulator" in unknown.plan.alternatives

    assert not malformed.plan.fail_closed
    assert malformed.failure_reason is not None
    assert "variance" in malformed.failure_reason


def test_provider_gradient_readiness_audit_payload_is_json_ready() -> None:
    audit = run_provider_gradient_readiness_audit()
    payload = audit.to_dict()
    claim_boundary = cast(str, payload["claim_boundary"])
    records = cast(list[dict[str, object]], payload["records"])
    first_scenario = cast(dict[str, object], records[0]["scenario"])
    first_result = cast(dict[str, object], records[0]["result"])

    assert payload["passed"] is True
    assert "provider-gradient readiness audit only" in claim_boundary
    assert isinstance(records, list)
    assert first_scenario["name"] == "statevector_parameter_shift"
    assert first_result["gradient"]
    assert records[3]["result"] is None


def test_provider_gradient_readiness_scenario_validation_is_strict() -> None:
    base = default_provider_gradient_readiness_scenarios()[0]

    with pytest.raises(ValueError, match="scenario name"):
        ProviderGradientReadinessScenario(
            name=" ",
            backend=base.backend,
            values=base.values,
            shots=base.shots,
            rule=base.rule,
            expected_gradient=base.expected_gradient,
            expected_outcome=base.expected_outcome,
            description=base.description,
        )
    with pytest.raises(ValueError, match="shots"):
        ProviderGradientReadinessScenario(
            name=base.name,
            backend=base.backend,
            values=base.values,
            shots=0,
            rule=base.rule,
            expected_gradient=base.expected_gradient,
            expected_outcome=base.expected_outcome,
            description=base.description,
        )
