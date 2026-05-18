# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 II. Hierarchical Predictive Coding: The SCPN's Computational Algorithm validation tests
"""Tests for Paper 0 II. Hierarchical Predictive Coding: The SCPN's Computational Algorithm source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.ii_hierarchical_predictive_coding_the_scpn_s_computational_algorithm_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    IiHierarchicalPredictiveCodingTheScpnSComputationalAlgorithmConfig,
    classify_ii_hierarchical_predictive_coding_the_scpn_s_computational_algorithm_component,
    ii_hierarchical_predictive_coding_the_scpn_s_computational_algorithm_labels,
    validate_ii_hierarchical_predictive_coding_the_scpn_s_computational_algorithm_fixture,
)


def test_ii_hierarchical_predictive_coding_the_scpn_s_computational_algorithm_fixture_preserves_source_boundary() -> (
    None
):
    result = (
        validate_ii_hierarchical_predictive_coding_the_scpn_s_computational_algorithm_fixture()
    )
    assert result.source_ledger_span == ("P0R06156", "P0R06163")
    assert result.source_record_count == 8
    assert result.component_count == 2
    assert result.next_source_boundary == "P0R06164"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_ii_hierarchical_predictive_coding_the_scpn_s_computational_algorithm_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R06156"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R06163"


def test_ii_hierarchical_predictive_coding_the_scpn_s_computational_algorithm_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "ii_hierarchical_predictive_coding_the_scpn_s_computational_algorithm",
        "iii_the_upde_as_the_physical_implementation_of_free_energy_minimisation",
    ):
        assert (
            classify_ii_hierarchical_predictive_coding_the_scpn_s_computational_algorithm_component(
                component
            )
            == f"{component}_source_boundary"
        )
    labels = ii_hierarchical_predictive_coding_the_scpn_s_computational_algorithm_labels()
    assert (
        labels["section"]
        == "II. Hierarchical Predictive Coding: The SCPN's Computational Algorithm"
    )
    assert labels["next_boundary"] == "P0R06164"


def test_ii_hierarchical_predictive_coding_the_scpn_s_computational_algorithm_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 8"):
        IiHierarchicalPredictiveCodingTheScpnSComputationalAlgorithmConfig(
            expected_source_record_count=7
        )
    with pytest.raises(ValueError, match="expected_component_count must equal 2"):
        IiHierarchicalPredictiveCodingTheScpnSComputationalAlgorithmConfig(
            expected_component_count=3
        )
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R06164"):
        IiHierarchicalPredictiveCodingTheScpnSComputationalAlgorithmConfig(
            next_source_boundary="P0R06163"
        )
    with pytest.raises(
        ValueError,
        match="unknown ii_hierarchical_predictive_coding_the_scpn_s_computational_algorithm component",
    ):
        classify_ii_hierarchical_predictive_coding_the_scpn_s_computational_algorithm_component(
            "empirical_validation_claim"
        )
