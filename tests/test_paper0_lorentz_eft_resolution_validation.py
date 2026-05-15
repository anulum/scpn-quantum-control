# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Lorentz EFT resolution validation tests
"""Tests for Paper 0 Lorentz-covariance/EFT-resolution validation."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.lorentz_eft_resolution_validation import (
    LorentzEFTResolutionConfig,
    classify_lorentz_eft_resolution_component,
    lorentz_eft_resolution_labels,
    validate_lorentz_eft_resolution_fixture,
)


def test_lorentz_eft_resolution_config_rejects_boundary_and_count_drift() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 25"):
        LorentzEFTResolutionConfig(expected_source_record_count=24)
    with pytest.raises(ValueError, match="expected_blank_record_count must equal 1"):
        LorentzEFTResolutionConfig(expected_blank_record_count=0)
    with pytest.raises(ValueError, match="expected_ghost_action_record_count must equal 5"):
        LorentzEFTResolutionConfig(expected_ghost_action_record_count=4)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R01103"):
        LorentzEFTResolutionConfig(next_source_boundary="P0R01102")


def test_lorentz_eft_resolution_classifiers_are_source_bounded() -> None:
    assert (
        classify_lorentz_eft_resolution_component("boundary_and_tension")
        == "lorentz_covariance_tension_and_eft_boundary"
    )
    assert (
        classify_lorentz_eft_resolution_component("fundamental_lorentz_invariant_action")
        == "lorentz_scalar_infoton_action_with_higher_dimension_fim_operator"
    )
    assert (
        classify_lorentz_eft_resolution_component("biological_medium_effective_metric")
        == "spontaneous_medium_effective_metric_and_infoton_kinetic_term"
    )
    assert (
        classify_lorentz_eft_resolution_component("consistency_implications")
        == "localized_lorentz_breaking_gauge_invariance_and_refractive_index_claims"
    )
    assert (
        classify_lorentz_eft_resolution_component("ghost_action_boundary")
        == "fim_background_gauge_fixing_ghost_action_and_separator_boundary"
    )
    with pytest.raises(ValueError, match="unknown Lorentz-EFT-resolution component"):
        classify_lorentz_eft_resolution_component("non_abelian_qualia")


def test_lorentz_eft_resolution_fixture_preserves_claim_boundary_and_null_controls() -> None:
    result = validate_lorentz_eft_resolution_fixture()

    assert result.source_ledger_span == ("P0R01078", "P0R01102")
    assert result.hardware_status == "source_methodology_no_experiment"
    assert result.source_record_count == 25
    assert result.blank_record_count == 1
    assert result.lorentz_tension_record_count == 4
    assert result.fundamental_action_record_count == 5
    assert result.biological_medium_record_count == 7
    assert result.consistency_record_count == 4
    assert result.ghost_action_record_count == 5
    assert result.next_source_boundary == "P0R01103"
    assert result.null_controls == {
        "lorentz_eft_resolution_is_source_claim_not_empirical_evidence": 1.0,
        "naive_fim_metric_replacement_is_marked_as_lorentz_violation": 1.0,
        "blank_record_p0r01079_and_separator_p0r01102_are_preserved": 1.0,
    }
    assert result.problem_metadata["protocol_state"] == (
        "source_lorentz_eft_resolution_only_no_experiment"
    )
    assert result.problem_metadata["source_ledger_ids"][0] == "P0R01078"
    assert result.problem_metadata["source_ledger_ids"][-1] == "P0R01102"


def test_lorentz_eft_resolution_labels_name_non_abelian_boundary() -> None:
    labels = lorentz_eft_resolution_labels()

    assert labels["section"] == "Formal Resolution of Lorentz Covariance"
    assert (
        labels["fundamental_action"] == "Lorentz scalar infoton action with Lambda_I suppression"
    )
    assert labels["effective_metric"] == "g_eff = eta - c/(2 Lambda_I^2) gF"
    assert labels["ghost_action"] == "Faddeev-Popov ghost action in FIM background"
    assert labels["next_boundary"] == "Beyond U(1): Non-Abelian Qualia Field"
