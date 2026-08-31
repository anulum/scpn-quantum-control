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
from dataclasses import replace
from typing import cast

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
    """Keep the default package serialisable and explicitly no-submit."""
    package = build_hardware_gradient_publication_package()
    payload = package.to_dict()

    assert isinstance(package, HardwareGradientPublicationPackage)
    assert payload["title"] == HARDWARE_GRADIENT_PUBLICATION_TITLE
    assert payload["hardware_execution_count"] == 0
    assert payload["gradient_available_count"] == 0
    assert payload["claim_status"] == "pre_registered_no_submit_scaffold"
    assert payload["submission_ready"] is False
    assert "no-submit" in cast(str, payload["claim_boundary"])
    assert json.loads(json.dumps(payload))["schema_version"].endswith(".v1")


def test_hardware_gradient_publication_package_covers_campaign_methods() -> None:
    """Cover every registered campaign method in all publication sections."""
    package = build_hardware_gradient_publication_package()

    methods = {section.method for section in package.method_sections}

    assert methods == {"parameter_shift_vqe", "spsa"}
    assert {row.method for row in package.claim_ledger_rows} == methods
    assert {entry.method for entry in package.artifact_map} == methods
    assert all(section.statevector_reference_required for section in package.method_sections)
    assert all(section.raw_counts_required for section in package.method_sections)
    assert all(section.calibration_snapshot_required for section in package.method_sections)


def test_hardware_gradient_publication_artifact_map_requires_raw_replay_fields() -> None:
    """Require raw replay and calibration fields before evidence promotion."""
    package = build_hardware_gradient_publication_package()

    for entry in package.artifact_map:
        payload = entry.to_dict()
        assert payload["raw_counts_status"] == "required_not_captured"
        assert payload["statevector_reference_status"] == "required_not_captured"
        assert payload["backend_calibration_status"] == "required_not_captured"
        required_replay_fields = cast(list[str], payload["required_replay_fields"])
        assert "evaluation_records" in required_replay_fields
        assert "statevector_reference" in required_replay_fields
        assert "hardware_execution" in required_replay_fields


def test_hardware_gradient_publication_claim_rows_are_not_promoted() -> None:
    """Keep claim-ledger rows unpromoted until every evidence item exists."""
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
    """Keep same-circuit benchmark placeholders explicit and unexecuted."""
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
    """Reject injected live-result claims from the no-submit scaffold."""
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
    """Export the publication builder and package through the phase namespace."""
    assert (
        phase.build_hardware_gradient_publication_package
        is build_hardware_gradient_publication_package
    )
    assert phase.HardwareGradientPublicationPackage is HardwareGradientPublicationPackage


def test_hardware_gradient_publication_rejects_empty_campaign_collection() -> None:
    """Reject a publication package with no campaign plans."""
    with pytest.raises(ValueError, match="at least one hardware-gradient campaign"):
        build_hardware_gradient_publication_package(plans=())


def test_hardware_gradient_publication_validates_preregistration_contract() -> None:
    """Reject blank and structurally empty preregistration fields."""
    preregistration = build_hardware_gradient_publication_package().preregistration

    with pytest.raises(ValueError, match="title must be non-empty"):
        replace(preregistration, title=" ")
    with pytest.raises(ValueError, match="secondary_endpoints"):
        replace(preregistration, secondary_endpoints=())
    with pytest.raises(ValueError, match="exclusion_rules"):
        replace(preregistration, exclusion_rules=())


def test_hardware_gradient_publication_renders_reviewer_markdown() -> None:
    """Render methods, artefacts, benchmarks, and the claim boundary."""
    markdown = build_hardware_gradient_publication_package().to_markdown()

    assert "# Hardware-Validated Quantum Gradients" in markdown
    assert "## Methods" in markdown
    assert "## Artefact Map" in markdown
    assert "## Benchmark Placeholders" in markdown
    assert "xy_parameter_shift_vqe_heron_r2_dry_run" in markdown
    assert "scpn_statevector_reference" in markdown
    assert "Claim boundary: no-submit publication scaffold" in markdown


def test_hardware_gradient_publication_requires_every_promotion_evidence_layer() -> None:
    """Require hardware output, promoted rows, and every artifact identifier."""
    package = build_hardware_gradient_publication_package()
    promoted_rows = tuple(
        replace(row, promoted=True, artifact_id="artifact", benchmark_id="benchmark")
        for row in package.claim_ledger_rows
    )
    mapped_artifacts = tuple(
        replace(entry, artifact_id="artifact") for entry in package.artifact_map
    )
    mapped_benchmarks = tuple(
        replace(entry, artifact_id="benchmark") for entry in package.benchmark_placeholders
    )

    assert replace(package, hardware_execution_count=1).submission_ready is False
    assert (
        replace(package, hardware_execution_count=1, gradient_available_count=1).submission_ready
        is False
    )
    assert (
        replace(
            package,
            hardware_execution_count=1,
            gradient_available_count=1,
            claim_ledger_rows=promoted_rows,
        ).submission_ready
        is False
    )
    assert (
        replace(
            package,
            hardware_execution_count=1,
            gradient_available_count=1,
            claim_ledger_rows=promoted_rows,
            artifact_map=mapped_artifacts,
        ).submission_ready
        is False
    )
    assert replace(
        package,
        hardware_execution_count=1,
        gradient_available_count=1,
        claim_ledger_rows=promoted_rows,
        artifact_map=mapped_artifacts,
        benchmark_placeholders=mapped_benchmarks,
    ).submission_ready


def test_hardware_gradient_publication_rejects_gradient_without_execution() -> None:
    """Reject injected gradient availability even without execution metadata."""
    plan = plan_hardware_gradient_campaign(default_hardware_gradient_campaign_specs()[0])
    invalid_plan = replace(plan, gradient_available=True)

    with pytest.raises(ValueError, match="publication scaffold cannot contain"):
        build_hardware_gradient_publication_package(plans=(invalid_plan,))
