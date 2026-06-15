# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Hardware Gradient Campaign Specs
"""Tests for no-submit hardware-gradient campaign specifications."""

from __future__ import annotations

import json

import pytest

from scpn_quantum_control import phase
from scpn_quantum_control.phase.hardware_gradient_campaign import (
    DEFAULT_CAMPAIGN_EVIDENCE_IDS,
    DEFAULT_HERON_R2_BACKENDS,
    HardwareGradientCampaignPlan,
    HardwareGradientCampaignSpec,
    HardwareGradientCampaignSuite,
    default_hardware_gradient_campaign_specs,
    plan_hardware_gradient_campaign,
    run_hardware_gradient_campaign_readiness_suite,
)


def test_default_hardware_gradient_campaign_specs_are_no_submit_and_json_ready() -> None:
    specs = default_hardware_gradient_campaign_specs()

    assert len(specs) == 2
    assert {spec.method for spec in specs} == {"parameter_shift_vqe", "spsa"}
    assert all(spec.dry_run_only for spec in specs)
    assert all(spec.backend in DEFAULT_HERON_R2_BACKENDS for spec in specs)
    assert all(spec.calibration_snapshot_required for spec in specs)
    assert all(spec.statevector_reference_required for spec in specs)
    assert all(spec.raw_counts_required for spec in specs)
    payload = [spec.to_dict() for spec in specs]
    assert json.loads(json.dumps(payload))[0]["schema_version"].endswith(".v1")


def test_parameter_shift_vqe_campaign_policy_counts_shifted_evaluations() -> None:
    spec = default_hardware_gradient_campaign_specs()[0]

    plan = plan_hardware_gradient_campaign(spec)

    assert isinstance(plan, HardwareGradientCampaignPlan)
    assert plan.approved_for_preparation
    assert not plan.hardware_execution
    assert not plan.gradient_available
    assert spec.evaluations == 12
    assert spec.estimated_total_shots == 6_144
    assert plan.policy_decision.evaluations == spec.evaluations
    assert plan.policy_decision.estimated_total_shots == spec.estimated_total_shots
    assert "no-submit" in plan.claim_boundary
    assert "parameter_shift_records" in spec.replay_schema().required_fields


def test_spsa_campaign_policy_counts_seeded_repetitions() -> None:
    spec = default_hardware_gradient_campaign_specs()[1]

    plan = plan_hardware_gradient_campaign(spec)
    payload = plan.to_dict()

    assert spec.method == "spsa"
    assert spec.seed == 17
    assert spec.perturbation_radius == pytest.approx(0.08)
    assert spec.evaluations == 8
    assert spec.estimated_total_shots == 4_096
    assert plan.approved_for_preparation
    assert plan.policy_decision.evaluations == spec.evaluations
    assert "perturbation_records" in spec.replay_schema().required_fields
    assert payload["hardware_execution"] is False
    assert payload["gradient_available"] is False


def test_hardware_gradient_campaign_suite_preserves_no_submit_boundary() -> None:
    suite = run_hardware_gradient_campaign_readiness_suite()

    assert isinstance(suite, HardwareGradientCampaignSuite)
    assert suite.passed
    assert suite.plan_count == 2
    assert suite.approved_count == 2
    assert suite.blocked_count == 0
    assert suite.hardware_execution_count == 0
    assert suite.gradient_available_count == 0
    assert "no QPU submission" in suite.claim_boundary


def test_hardware_gradient_campaign_rejects_missing_evidence_and_bad_allowlist() -> None:
    evidence = dict(DEFAULT_CAMPAIGN_EVIDENCE_IDS)
    evidence.pop("cost_budget_id")

    with pytest.raises(ValueError, match="missing campaign evidence IDs"):
        HardwareGradientCampaignSpec(
            name="missing_budget",
            method="parameter_shift_vqe",
            provider="ibm_quantum",
            backend="ibm_fez",
            n_params=2,
            shots_per_evaluation=256,
            shift_terms=1,
            spsa_repetitions=1,
            perturbation_radius=None,
            seed=None,
            evidence_ids=evidence,
            backend_allowlist=DEFAULT_HERON_R2_BACKENDS,
        )

    with pytest.raises(ValueError, match="not in the campaign allowlist"):
        HardwareGradientCampaignSpec(
            name="bad_backend",
            method="spsa",
            provider="ibm_quantum",
            backend="mystery_backend",
            n_params=2,
            shots_per_evaluation=256,
            shift_terms=1,
            spsa_repetitions=2,
            perturbation_radius=0.1,
            seed=1,
            evidence_ids=DEFAULT_CAMPAIGN_EVIDENCE_IDS,
            backend_allowlist=DEFAULT_HERON_R2_BACKENDS,
        )


def test_hardware_gradient_campaign_policy_blocks_budget_overrun() -> None:
    spec = HardwareGradientCampaignSpec(
        name="budget_overrun",
        method="parameter_shift_vqe",
        provider="ibm_quantum",
        backend="ibm_fez",
        n_params=20,
        shots_per_evaluation=2_048,
        shift_terms=1,
        spsa_repetitions=1,
        perturbation_radius=None,
        seed=None,
        evidence_ids=DEFAULT_CAMPAIGN_EVIDENCE_IDS,
        backend_allowlist=DEFAULT_HERON_R2_BACKENDS,
    )

    plan = plan_hardware_gradient_campaign(spec)

    assert plan.fail_closed
    assert not plan.approved_for_preparation
    assert "estimated total shots" in plan.policy_decision.failure_reason
    assert not plan.hardware_execution


def test_hardware_gradient_campaign_exports_from_phase_namespace() -> None:
    assert phase.HardwareGradientCampaignSpec is HardwareGradientCampaignSpec
    assert phase.HardwareGradientCampaignPlan is HardwareGradientCampaignPlan
    assert (
        phase.default_hardware_gradient_campaign_specs is default_hardware_gradient_campaign_specs
    )
    assert (
        phase.run_hardware_gradient_campaign_readiness_suite
        is run_hardware_gradient_campaign_readiness_suite
    )
