# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Differentiable Readiness Audit
"""Tests for the unified differentiable-programming readiness ledger."""

from __future__ import annotations

import json

from scpn_quantum_control.phase import (
    DifferentiableReadinessAuditResult,
    default_differentiable_readiness_surfaces,
    run_differentiable_readiness_audit,
)


def test_differentiable_readiness_audit_aggregates_supported_and_blocked_surfaces() -> None:
    """The readiness audit should aggregate all focused differentiable surfaces."""
    audit = run_differentiable_readiness_audit()
    payload = audit.to_dict()

    assert isinstance(audit, DifferentiableReadinessAuditResult)
    assert audit.passed
    assert audit.record_count == 9
    assert audit.passed_count == 9
    assert audit.failed_count == 0
    assert audit.supported_count > 0
    assert audit.blocked_count > 0
    assert audit.hardware_execution_count == 0
    assert audit.hardware_gradient_available_count == 0
    assert audit.blocked_boundaries
    assert json.loads(json.dumps(payload))["passed"] is True


def test_differentiable_readiness_audit_lists_expected_surfaces() -> None:
    """The readiness ledger should list every expected focused audit surface."""
    audit = run_differentiable_readiness_audit()

    assert {record.surface for record in audit.records} == {
        "gradient_support_matrix",
        "transform_nesting",
        "phase_qnode_tape",
        "phase_qnode_transforms",
        "phase_qnode_vector_transforms",
        "provider_gradient_readiness",
        "provider_qnode_transforms",
        "hardware_gradient_policy",
        "provider_hardware_gradient_preparation",
    }
    assert all(record.claim_boundary for record in audit.records)


def test_differentiable_readiness_audit_preserves_hardware_claim_boundary() -> None:
    """Hardware readiness rows should stay blocked without live gradient promotion."""
    audit = run_differentiable_readiness_audit()
    provider_hardware = next(
        record
        for record in audit.records
        if record.surface == "provider_hardware_gradient_preparation"
    )
    hardware_policy = next(
        record for record in audit.records if record.surface == "hardware_gradient_policy"
    )

    assert provider_hardware.passed
    assert provider_hardware.hardware_execution_count == 0
    assert provider_hardware.hardware_gradient_available_count == 0
    assert provider_hardware.blocked_count == 4
    assert hardware_policy.blocked_count == 5


def test_default_differentiable_readiness_surfaces_are_named_callables() -> None:
    """Default readiness surfaces should be stable named callable records."""
    surfaces = default_differentiable_readiness_surfaces()

    assert len(surfaces) == 9
    assert surfaces[0].surface == "gradient_support_matrix"
    assert all(callable(surface.runner) for surface in surfaces)
