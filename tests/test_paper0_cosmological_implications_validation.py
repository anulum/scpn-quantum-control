# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 cosmological implications fixture tests
"""Tests for Paper 0 cosmological implications simulator fixtures."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.cosmological_implications_validation import (
    CosmologicalImplicationsConfig,
    ethical_renormalisation_delta,
    lambda_balance_score,
    theory_positioning_score,
    validate_cosmological_implications_fixture,
    validate_ethical_renormalisation_fixture,
)


def test_theory_positioning_requires_all_source_comparison_channels() -> None:
    config = CosmologicalImplicationsConfig()

    complete = theory_positioning_score(
        iit=True,
        orch_or=True,
        gnw=True,
        fep_predictive_coding=True,
        upde=True,
        l15=True,
        config=config,
    )
    partial = theory_positioning_score(
        iit=True,
        orch_or=False,
        gnw=True,
        fep_predictive_coding=False,
        upde=True,
        l15=False,
        config=config,
    )

    assert complete == pytest.approx(1.0)
    assert partial < config.positioning_threshold


def test_lambda_balance_and_ethical_renormalisation_are_bounded() -> None:
    config = CosmologicalImplicationsConfig()

    balanced = lambda_balance_score(
        expansion_balance=0.72,
        rg_flow_window=0.78,
        cosmic_attractor_access=0.74,
        config=config,
    )
    unbalanced = lambda_balance_score(
        expansion_balance=0.19,
        rg_flow_window=0.22,
        cosmic_attractor_access=0.31,
        config=config,
    )
    delta = ethical_renormalisation_delta(
        previous_cycle_stagnation=0.72,
        unsustainable_complexity=0.66,
        coupling_adjustment=0.81,
        l16_meta_optimisation=0.88,
        config=config,
    )

    assert balanced > config.lambda_balance_threshold
    assert unbalanced < config.lambda_balance_threshold
    assert delta > config.renormalisation_threshold


def test_ethical_renormalisation_fixture_has_rejection_controls() -> None:
    result = validate_ethical_renormalisation_fixture()

    assert result.renormalisation_delta > result.problem_metadata["renormalisation_threshold"]
    assert result.null_controls["missing_l16_meta_optimisation_rejection_label"] == 1.0
    assert result.null_controls["negative_coupling_adjustment_rejection_label"] == 1.0
    assert result.null_controls["unsupported_empirical_cosmology_rejection_label"] == 1.0


def test_invalid_cosmological_implications_config_rejects_bad_parameters() -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        CosmologicalImplicationsConfig(positioning_threshold=0.0)
    with pytest.raises(ValueError, match="finite and non-negative"):
        CosmologicalImplicationsConfig(iit_weight=-0.1)
    with pytest.raises(ValueError, match="in \\[0, 1\\]"):
        lambda_balance_score(
            expansion_balance=1.2,
            rg_flow_window=0.5,
            cosmic_attractor_access=0.5,
            config=CosmologicalImplicationsConfig(),
        )


def test_cosmological_implications_fixture_preserves_boundaries() -> None:
    result = validate_cosmological_implications_fixture()

    assert result.spec_keys == (
        "cosmological_implications.comparative_positioning_mapping",
        "cosmological_implications.ethical_selection_claim_boundary",
        "cosmological_implications.lambda_optimisation_context",
        "cosmological_implications.ethical_renormalisation_mechanism",
        "cosmological_implications.mmc_ccc_formalisation_boundary",
    )
    assert result.hardware_status == "simulator_only_no_provider_submission"
    assert result.positioning_score > result.config_thresholds["positioning_threshold"]
    assert result.lambda_balanced_score > result.lambda_unbalanced_score
    assert result.renormalisation.renormalisation_delta > 0.0
    assert result.source_ledger_span == ("P0R06290", "P0R06310")
    assert "not empirical evidence" in result.claim_boundary
