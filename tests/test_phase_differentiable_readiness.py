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

import pytest

from scpn_quantum_control.phase import (
    DifferentiableReadinessAuditResult,
    DifferentiableReadinessSurface,
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


class _RecordsResult:
    """Readiness result that reports only per-record flags, as the provider ledgers do."""

    def __init__(self, records: list[object]) -> None:
        """Hold the records the payload exposes."""
        self._records = records

    def to_dict(self) -> dict[str, object]:
        """Return a payload with records and no aggregate counts."""
        return {"passed": True, "records": self._records}


class _AttributeResult:
    """Readiness result whose supported and blocked records are attributes, not counts."""

    supported_records = ("a", "b")
    blocked_records = ("c",)
    passed = False
    claim_boundary = "  attribute-backed readiness  "

    def to_dict(self) -> dict[str, object]:
        """Return a payload that carries the gradient count and an unserialisable value."""
        return {"gradient_available_count": 4, "marker": object, "nested": [{1: (2, 3)}]}


class _ListPayloadResult:
    """Readiness result whose to_dict is not a mapping."""

    def to_dict(self) -> list[str]:
        """Return a list, which the ledger records by its repr."""
        return ["listed"]

    def __repr__(self) -> str:
        """Name the result in the recorded payload."""
        return "listed readiness"


def test_readiness_surfaces_refuse_blank_names_and_report_their_runner() -> None:
    """A surface needs a name and a description and names its runner in its metadata."""

    def custom_runner() -> object:
        """Return a value without a payload."""
        return 7

    with pytest.raises(ValueError, match="^surface must be non-empty$"):
        DifferentiableReadinessSurface(" ", custom_runner, "described")
    with pytest.raises(ValueError, match="^description must be non-empty$"):
        DifferentiableReadinessSurface("custom", custom_runner, " ")
    surface = DifferentiableReadinessSurface(" custom ", custom_runner, " described ")

    assert surface.to_dict() == {
        "surface": "custom",
        "runner": "custom_runner",
        "description": "described",
    }


def test_readiness_audit_counts_records_by_their_flags() -> None:
    """Only boolean per-record flags count, nested results included, and give blocked reasons."""
    records: list[object] = [
        {"supported": True, "result": {"hardware_execution": True}},
        {"approved": True, "gradient_available": True},
        {"blocked": True, "failure_reason": "needs a ticket"},
        {"fail_closed": True, "result": {"claim_boundary": "local only"}},
        {"supported": False},
        {"supported": "yes", "approved": 1},
        {"supported": True, "hardware_execution": "yes", "result": {"hardware_execution": 1}},
        {"gradient_available": {"nested": True}},
        "not a record",
    ]
    surface = DifferentiableReadinessSurface("records", lambda: _RecordsResult(records), "flags")

    record = run_differentiable_readiness_audit([surface]).records[0]

    assert record.passed is True
    assert record.supported_count == 3
    assert record.blocked_count == 2
    assert record.hardware_execution_count == 1
    assert record.hardware_gradient_available_count == 1
    assert record.blocked_boundaries == ("needs a ticket", "local only")
    assert record.claim_boundary == "focused differentiable readiness surface"


def test_readiness_audit_reads_attribute_records_and_serialises_any_payload() -> None:
    """Attribute record lists are counted, and unserialisable payload values become text."""
    surfaces = [
        DifferentiableReadinessSurface("attributes", _AttributeResult, "attributes"),
        DifferentiableReadinessSurface("bare", lambda: 7, "no payload"),
        DifferentiableReadinessSurface("listed", lambda: _ListPayloadResult(), "list payload"),
        DifferentiableReadinessSurface("direct", lambda: _RecordsResult([]), "direct counts"),
    ]

    audit = run_differentiable_readiness_audit(surfaces)

    attributes, bare, listed, direct = audit.records
    assert (attributes.supported_count, attributes.blocked_count) == (2, 1)
    assert attributes.hardware_gradient_available_count == 4
    assert attributes.hardware_execution_count == 0
    assert attributes.claim_boundary == "attribute-backed readiness"
    payload = attributes.to_dict()["payload"]
    assert payload["marker"] == repr(object)
    assert payload["nested"] == [{"1": [2, 3]}]
    assert bare.to_dict()["payload"] == {"repr": "7"}
    assert (bare.supported_count, bare.blocked_count, bare.passed) == (0, 0, False)
    assert listed.to_dict()["payload"] == {"repr": "listed readiness"}
    assert (direct.hardware_execution_count, direct.hardware_gradient_available_count) == (0, 0)
    json.dumps(audit.to_dict())


def test_readiness_audit_prefers_direct_hardware_counts() -> None:
    """Direct hardware execution and gradient counts win over per-record flags."""

    class _DirectCounts(_RecordsResult):
        """Records plus direct hardware counts."""

        def to_dict(self) -> dict[str, object]:
            """Return direct counts beside a record that would count otherwise."""
            return {
                **super().to_dict(),
                "hardware_execution_count": 5,
                "hardware_gradient_available_count": 2,
            }

    records: list[object] = [{"hardware_execution": True, "gradient_available": True}]
    surface = DifferentiableReadinessSurface("direct", lambda: _DirectCounts(records), "direct")

    record = run_differentiable_readiness_audit([surface]).records[0]

    assert (record.hardware_execution_count, record.hardware_gradient_available_count) == (5, 2)
