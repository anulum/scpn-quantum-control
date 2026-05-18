# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 The Unifying Computational Principle: Hierarchical Predictive Coding (HPC) validation tests
"""Tests for Paper 0 The Unifying Computational Principle: Hierarchical Predictive Coding (HPC) source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.the_unifying_computational_principle_hierarchical_predictive_coding_hpc_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    TheUnifyingComputationalPrincipleHierarchicalPredictiveCodingHpcConfig,
    classify_the_unifying_computational_principle_hierarchical_predictive_coding_hpc_component,
    the_unifying_computational_principle_hierarchical_predictive_coding_hpc_labels,
    validate_the_unifying_computational_principle_hierarchical_predictive_coding_hpc_fixture,
)


def test_the_unifying_computational_principle_hierarchical_predictive_coding_hpc_fixture_preserves_source_boundary() -> (
    None
):
    result = (
        validate_the_unifying_computational_principle_hierarchical_predictive_coding_hpc_fixture()
    )
    assert result.source_ledger_span == ("P0R06147", "P0R06155")
    assert result.source_record_count == 9
    assert result.component_count == 2
    assert result.next_source_boundary == "P0R06156"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_the_unifying_computational_principle_hierarchical_predictive_coding_hpc_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R06147"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R06155"


def test_the_unifying_computational_principle_hierarchical_predictive_coding_hpc_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "the_unifying_computational_principle_hierarchical_predictive_coding_hpc",
        "i_the_free_energy_principle_the_imperative_to_minimise_surprise",
    ):
        assert (
            classify_the_unifying_computational_principle_hierarchical_predictive_coding_hpc_component(
                component
            )
            == f"{component}_source_boundary"
        )
    labels = the_unifying_computational_principle_hierarchical_predictive_coding_hpc_labels()
    assert (
        labels["section"]
        == "The Unifying Computational Principle: Hierarchical Predictive Coding (HPC)"
    )
    assert labels["next_boundary"] == "P0R06156"


def test_the_unifying_computational_principle_hierarchical_predictive_coding_hpc_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 9"):
        TheUnifyingComputationalPrincipleHierarchicalPredictiveCodingHpcConfig(
            expected_source_record_count=8
        )
    with pytest.raises(ValueError, match="expected_component_count must equal 2"):
        TheUnifyingComputationalPrincipleHierarchicalPredictiveCodingHpcConfig(
            expected_component_count=3
        )
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R06156"):
        TheUnifyingComputationalPrincipleHierarchicalPredictiveCodingHpcConfig(
            next_source_boundary="P0R06155"
        )
    with pytest.raises(
        ValueError,
        match="unknown the_unifying_computational_principle_hierarchical_predictive_coding_hpc component",
    ):
        classify_the_unifying_computational_principle_hierarchical_predictive_coding_hpc_component(
            "empirical_validation_claim"
        )
