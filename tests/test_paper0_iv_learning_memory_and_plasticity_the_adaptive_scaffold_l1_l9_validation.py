# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 IV. Learning, Memory, and Plasticity: The Adaptive Scaffold (L1-L9) validation tests
"""Tests for Paper 0 IV. Learning, Memory, and Plasticity: The Adaptive Scaffold (L1-L9) source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.iv_learning_memory_and_plasticity_the_adaptive_scaffold_l1_l9_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    IvLearningMemoryAndPlasticityTheAdaptiveScaffoldL1L9Config,
    classify_iv_learning_memory_and_plasticity_the_adaptive_scaffold_l1_l9_component,
    iv_learning_memory_and_plasticity_the_adaptive_scaffold_l1_l9_labels,
    validate_iv_learning_memory_and_plasticity_the_adaptive_scaffold_l1_l9_fixture,
)


def test_iv_learning_memory_and_plasticity_the_adaptive_scaffold_l1_l9_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_iv_learning_memory_and_plasticity_the_adaptive_scaffold_l1_l9_fixture()
    assert result.source_ledger_span == ("P0R05001", "P0R05008")
    assert result.source_record_count == 8
    assert result.component_count == 3
    assert result.next_source_boundary == "P0R05009"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_iv_learning_memory_and_plasticity_the_adaptive_scaffold_l1_l9_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R05001"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R05008"


def test_iv_learning_memory_and_plasticity_the_adaptive_scaffold_l1_l9_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "iv_learning_memory_and_plasticity_the_adaptive_scaffold_l1_l9",
        "1_the_multi_scale_memory_trace_the_engram",
        "2_the_mechanism_of_learning_hpc_optimisation_and_psi_guidance",
    ):
        assert (
            classify_iv_learning_memory_and_plasticity_the_adaptive_scaffold_l1_l9_component(
                component
            )
            == f"{component}_source_boundary"
        )
    labels = iv_learning_memory_and_plasticity_the_adaptive_scaffold_l1_l9_labels()
    assert (
        labels["section"] == "IV. Learning, Memory, and Plasticity: The Adaptive Scaffold (L1-L9)"
    )
    assert labels["next_boundary"] == "P0R05009"


def test_iv_learning_memory_and_plasticity_the_adaptive_scaffold_l1_l9_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 8"):
        IvLearningMemoryAndPlasticityTheAdaptiveScaffoldL1L9Config(expected_source_record_count=7)
    with pytest.raises(ValueError, match="expected_component_count must equal 3"):
        IvLearningMemoryAndPlasticityTheAdaptiveScaffoldL1L9Config(expected_component_count=4)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R05009"):
        IvLearningMemoryAndPlasticityTheAdaptiveScaffoldL1L9Config(next_source_boundary="P0R05008")
    with pytest.raises(
        ValueError,
        match="unknown iv_learning_memory_and_plasticity_the_adaptive_scaffold_l1_l9 component",
    ):
        classify_iv_learning_memory_and_plasticity_the_adaptive_scaffold_l1_l9_component(
            "empirical_validation_claim"
        )
