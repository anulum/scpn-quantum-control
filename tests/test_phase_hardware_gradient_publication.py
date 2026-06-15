# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Hardware Gradient Publication Package
"""Tests for the hardware-gradient publication package scaffold."""

from __future__ import annotations

import json

import pytest

from scpn_quantum_control import phase
from scpn_quantum_control.phase.hardware_gradient_campaign import (
    HardwareGradientCampaignPlan,
    default_hardware_gradient_campaign_specs,
    plan_hardware_gradient_campaign,
)
from scpn_quantum_control.phase.hardware_gradient_publication import (
    HARDWARE_GRADIENT_PUBLICATION_TITLE,
    HardwareGradientPublicationPackage,
    build_hardware_gradient_publication_package,
)


def test_hardware_gradient_publication_package_is_json_ready_and_no_submit() -> None:
    package = build_hardware_gradient_publication_package()
    payload = package.to_dict()

    assert isinstance(package, HardwareGradientPublicationPackage)
    assert payload["title"] == HARDWARE_GRADIENT_PUBLICATION_TITLE
    assert payload["hardware_execution_count"] == 0
    assert payload["gradient_available_count"] == 0
    assert payload["claim_status"] == "pre_registered_no_submit_scaffold"
    assert payload["submission_ready"] is False
    assert "no-submit" in payload["claim_boundary"]
    assert json.loads(json.dumps(payload))["schema_version"].endswith(".v1")


def test_hardware_gradient_publication_package_covers_campaign_methods() -> None:
    package = build_hardware_gradient_publication_package()

    methods = {section.method for section in package.method_sections}

    assert methods == {"parameter_shift_vqe", "spsa"}
    assert {row.method for row in package.claim_ledger_rows} == methods
    assert {entry.method for entry in package.artifact_map} == methods
    assert all(section.statevector_reference_required for section in package.method_sections)
    assert all(section.raw_counts_required for section in package.method_sections)
    assert all(section.calibration_snapshot_required for section in package.method_sections)


def test_hardware_gradient_publication_artifact_map_requires_raw_replay_fields() -> None:
    package = build_hardware_gradient_publication_package()

    for entry in package.artifact_map:
        payload = entry.to_dict()
        assert payload["raw_counts_status"] == "required_not_captured"
        assert payload["statevector_reference_status"] == "required_not_captured"
        assert payload["backend_calibration_status"] == "required_not_captured"
        assert "evaluation_records" in payload["required_replay_fields"]
        assert "statevector_reference" in payload["required_replay_fields"]
        assert "hardware_execution" in payload["required_replay_fields"]


def test_hardware_gradient_publication_claim_rows_are_not_promoted() -> None:
    package = build_hardware_gradient_publication_package()

    for row in package.claim_ledger_rows:
        payload = row.to_dict()
        assert payload["promoted"] is False
        assert payload["claim_boundary"] == "planned_publication_row_no_hardware_evidence"
        assert payload["required_before_promotion"] == [
            "approved live execution ticket",
            "backend calibration snapshot",
            "raw hardware count artefact",
            "statevector reference gradient",
            "same-circuit competitor comparison",
            "claim-ledger artefact ID",
            "benchmark evidence ID",
        ]


def test_hardware_gradient_publication_benchmark_placeholders_are_explicit() -> None:
    package = build_hardware_gradient_publication_package()

    routes = {placeholder.route for placeholder in package.benchmark_placeholders}

    assert routes == {
        "scpn_statevector_reference",
        "pennylane_same_circuit",
        "qiskit_same_circuit",
    }
    for placeholder in package.benchmark_placeholders:
        payload = placeholder.to_dict()
        assert payload["status"] == "placeholder_not_executed"
        assert payload["same_circuit_required"] is True
        assert payload["same_parameters_required"] is True
        assert payload["same_observable_required"] is True
        assert payload["artifact_id"] is None


def test_hardware_gradient_publication_package_rejects_live_result_claims() -> None:
    plan = plan_hardware_gradient_campaign(default_hardware_gradient_campaign_specs()[0])
    invalid_plan = HardwareGradientCampaignPlan(
        spec=plan.spec,
        policy_decision=plan.policy_decision,
        hardware_execution=True,
        gradient_available=True,
        claim_boundary="invalid live result injected by test",
    )

    with pytest.raises(ValueError, match="publication scaffold cannot contain"):
        build_hardware_gradient_publication_package(plans=(invalid_plan,))


def test_hardware_gradient_publication_exports_from_phase_namespace() -> None:
    assert (
        phase.build_hardware_gradient_publication_package
        is build_hardware_gradient_publication_package
    )
    assert phase.HardwareGradientPublicationPackage is HardwareGradientPublicationPackage
