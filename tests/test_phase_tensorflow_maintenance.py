# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — TensorFlow maintenance tests.
"""Tests for TensorFlow framework-parity maintenance decisions."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

import scpn_quantum_control.phase as phase
from scpn_quantum_control.phase.tensorflow_maintenance import (
    TENSORFLOW_MAINTENANCE_CLAIM_BOUNDARY,
    PhaseTensorFlowMaintenanceReport,
    PhaseTensorFlowMaintenanceRoute,
    run_tensorflow_maintenance_decision,
)


def test_tensorflow_maintenance_decision_rescopes_to_compatibility_only() -> None:
    """The decision keeps bounded routes maintained without broad Graph/XLA claims."""
    report = run_tensorflow_maintenance_decision()

    assert isinstance(report, PhaseTensorFlowMaintenanceReport)
    assert report.strategy == "compatibility_only"
    assert report.compatibility_only
    assert not report.graph_xla_parity_promoted
    assert not report.ready_for_provider_exceedance
    assert report.route("bounded_gradient_tape").decision == "compatibility_only"
    assert report.route("bounded_tf_function").decision == "compatibility_only"
    assert report.route("bounded_xla_request").decision == "compatibility_only"
    assert report.route("arbitrary_phase_qnode_tensorflow_lowering").decision == "blocked"
    assert report.route("full_graph_autodiff_through_simulator").decision == "blocked"
    assert "bounded_xla_request" in report.maintained_compatibility_routes
    assert "hardware_gradient_route" in report.blocked_routes
    assert "broad_graph_xla_parity" in report.stale_claim_blockers
    assert "broad Graph/XLA parity" in report.claim_boundary


def test_tensorflow_maintenance_payload_is_json_ready_and_exported() -> None:
    """The phase package exports the same TensorFlow maintenance decision surface."""
    report = phase.run_tensorflow_maintenance_decision()
    payload = report.to_dict()
    routes = payload["routes"]

    assert phase.PhaseTensorFlowMaintenanceReport is PhaseTensorFlowMaintenanceReport
    assert phase.PhaseTensorFlowMaintenanceRoute is PhaseTensorFlowMaintenanceRoute
    assert phase.TENSORFLOW_MAINTENANCE_CLAIM_BOUNDARY == TENSORFLOW_MAINTENANCE_CLAIM_BOUNDARY
    assert payload["strategy"] == "compatibility_only"
    assert payload["compatibility_only"] is True
    assert payload["graph_xla_parity_promoted"] is False
    assert isinstance(routes, dict)
    assert routes["bounded_qnn_tensor_gradient"]["fail_closed"] is True
    assert routes["bounded_qnn_tensor_gradient"]["decision"] == "compatibility_only"
    assert routes["hardware_gradient_route"]["decision"] == "blocked"
    assert routes["hardware_gradient_route"]["required_before_promotion"] == [
        "live_ticket",
        "raw_count_replay",
        "calibration_snapshot",
    ]


def test_tensorflow_maintenance_report_validates_duplicate_routes() -> None:
    """The report rejects duplicate route names."""
    route = PhaseTensorFlowMaintenanceRoute(
        name="duplicate",
        decision="blocked",
        evidence=("unit",),
        blocked_reasons=("blocked",),
        required_before_promotion=("evidence",),
    )

    with pytest.raises(ValueError, match="unique"):
        PhaseTensorFlowMaintenanceReport(
            strategy="compatibility_only",
            routes=(route, route),
        )


def test_tensorflow_maintenance_route_fails_closed_on_unknown_route() -> None:
    """Unknown TensorFlow decision routes fail closed."""
    report = run_tensorflow_maintenance_decision()

    with pytest.raises(KeyError, match="unknown TensorFlow maintenance route"):
        report.route("missing")


def _route(decision: str) -> PhaseTensorFlowMaintenanceRoute:
    """Return a shipped route with the given decision."""
    report = run_tensorflow_maintenance_decision()
    return next(route for route in report.routes if route.decision == decision)


@pytest.mark.parametrize(
    ("decision", "changes", "message"),
    [
        ("blocked", {"name": ""}, "route name must be non-empty"),
        ("blocked", {"decision": "promoted"}, "unknown TensorFlow maintenance decision"),
        ("blocked", {"evidence": ()}, "route evidence must be non-empty"),
        ("blocked", {"evidence": ("",)}, "route evidence must be non-empty"),
        ("blocked", {"blocked_reasons": ("",)}, "route blockers must be non-empty"),
        ("blocked", {"required_before_promotion": ("",)}, "promotion requirements must be"),
        ("blocked", {"blocked_reasons": ()}, "blocked TensorFlow maintenance routes require"),
        ("blocked", {"decision": "maintained"}, "maintained TensorFlow maintenance routes cannot"),
        ("blocked", {"claim_boundary": ""}, "route claim boundary must be non-empty"),
    ],
)
def test_tensorflow_maintenance_route_refuses_incomplete_or_contradictory_records(
    decision: str, changes: dict[str, Any], message: str
) -> None:
    """A route needs a name, a known decision, evidence and blockers that fit its decision."""
    with pytest.raises(ValueError, match=message):
        replace(_route(decision), **changes)


def test_tensorflow_maintenance_route_accepts_a_maintained_route_without_blockers() -> None:
    """A maintained route without blockers is valid and not fail-closed."""
    route = replace(_route("blocked"), decision="maintained", blocked_reasons=())

    assert route.fail_closed is False
    assert route.to_dict()["decision"] == "maintained"


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"strategy": "parity"}, "strategy must be compatibility_only"),
        ({"routes": ()}, "report requires routes"),
        ({"claim_boundary": ""}, "report claim boundary must be non-empty"),
    ],
)
def test_tensorflow_maintenance_report_refuses_another_strategy_no_routes_or_no_boundary(
    changes: dict[str, Any], message: str
) -> None:
    """The report stays compatibility-only, lists routes and states its claim boundary."""
    with pytest.raises(ValueError, match=message):
        replace(run_tensorflow_maintenance_decision(), **changes)
