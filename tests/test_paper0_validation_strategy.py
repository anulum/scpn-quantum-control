# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 validation strategy tests
"""Executable fixture tests for Paper 0 Applied SCPN and Validation records."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.validation_strategy import (
    ValidationStrategyConfig,
    stage_order_index,
    validate_stage_order,
    validate_validation_strategy_fixture,
    validation_domain_coverage,
)


def test_validation_domain_coverage_classifies_all_source_domains() -> None:
    config = ValidationStrategyConfig()

    coverage = validation_domain_coverage(config)

    assert coverage["pathology"].target_type == "systems_state"
    assert coverage["societal_phase_transitions_l11"].target_type == "spin_glass_dynamics"
    assert coverage["ethical_governance_l15"].target_type == "ethical_lagrangian_cef"
    assert coverage["alignment_objective"].target_type == "ethical_functional_embedding"
    assert all("not empirical evidence" in target.claim_boundary for target in coverage.values())


def test_validation_stage_order_is_explicit_and_complete() -> None:
    config = ValidationStrategyConfig()

    assert (
        stage_order_index("Stage I")
        < stage_order_index("Stage II")
        < stage_order_index("Stage III")
    )
    assert validate_stage_order(config.stages) is True

    with pytest.raises(ValueError, match="unknown validation stage"):
        stage_order_index("Stage IV")


def test_validation_strategy_fixture_checks_prioritised_stage_contract() -> None:
    result = validate_validation_strategy_fixture()

    assert result.hardware_status == "simulator_only_no_provider_submission"
    assert result.stage_count == 3
    assert result.domain_count == 11
    assert result.stage_order_valid is True
    assert result.null_controls["duplicate_domain_rejection_label"] == pytest.approx(1.0)
    assert result.null_controls["unknown_stage_rejection_label"] == pytest.approx(1.0)
    assert "not empirical evidence" in result.claim_boundary


def test_validation_strategy_rejects_incomplete_or_duplicate_inputs() -> None:
    with pytest.raises(ValueError, match="at least one validation stage"):
        ValidationStrategyConfig(stages=())

    with pytest.raises(ValueError, match="duplicate domain"):
        ValidationStrategyConfig(
            domains=("pathology", "pathology"),
        )

    with pytest.raises(ValueError, match="required source domain"):
        ValidationStrategyConfig(domains=("pathology",))
