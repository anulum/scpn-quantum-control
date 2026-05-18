# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 I. The Unifying Computational Principle: Hierarchical Predictive Coding (HPC) validation tests
"""Tests for Paper 0 I. The Unifying Computational Principle: Hierarchical Predictive Coding (HPC) source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.i_the_unifying_computational_principle_hierarchical_predictive_coding_hp_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    ITheUnifyingComputationalPrincipleHierarchicalPredictiveCodingHpConfig,
    classify_i_the_unifying_computational_principle_hierarchical_predictive_coding_hp_component,
    i_the_unifying_computational_principle_hierarchical_predictive_coding_hp_labels,
    validate_i_the_unifying_computational_principle_hierarchical_predictive_coding_hp_fixture,
)


def test_i_the_unifying_computational_principle_hierarchical_predictive_coding_hp_fixture_preserves_source_boundary() -> (
    None
):
    result = (
        validate_i_the_unifying_computational_principle_hierarchical_predictive_coding_hp_fixture()
    )
    assert result.source_ledger_span == ("P0R03197", "P0R03207")
    assert result.source_record_count == 11
    assert result.component_count == 4
    assert result.next_source_boundary == "P0R03208"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_i_the_unifying_computational_principle_hierarchical_predictive_coding_hp_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R03197"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R03207"


def test_i_the_unifying_computational_principle_hierarchical_predictive_coding_hp_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "i_the_unifying_computational_principle_hierarchical_predictive_coding_hp",
        "1_the_generative_model_downward_projection",
        "2_inference_and_error_upward_filtering",
        "3_optimisation_free_energy_minimisation",
    ):
        assert (
            classify_i_the_unifying_computational_principle_hierarchical_predictive_coding_hp_component(
                component
            )
            == f"{component}_source_boundary"
        )
    labels = i_the_unifying_computational_principle_hierarchical_predictive_coding_hp_labels()
    assert (
        labels["section"]
        == "I. The Unifying Computational Principle: Hierarchical Predictive Coding (HPC)"
    )
    assert labels["next_boundary"] == "P0R03208"


def test_i_the_unifying_computational_principle_hierarchical_predictive_coding_hp_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 11"):
        ITheUnifyingComputationalPrincipleHierarchicalPredictiveCodingHpConfig(
            expected_source_record_count=10
        )
    with pytest.raises(ValueError, match="expected_component_count must equal 4"):
        ITheUnifyingComputationalPrincipleHierarchicalPredictiveCodingHpConfig(
            expected_component_count=5
        )
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R03208"):
        ITheUnifyingComputationalPrincipleHierarchicalPredictiveCodingHpConfig(
            next_source_boundary="P0R03207"
        )
    with pytest.raises(
        ValueError,
        match="unknown i_the_unifying_computational_principle_hierarchical_predictive_coding_hp component",
    ):
        classify_i_the_unifying_computational_principle_hierarchical_predictive_coding_hp_component(
            "empirical_validation_claim"
        )
