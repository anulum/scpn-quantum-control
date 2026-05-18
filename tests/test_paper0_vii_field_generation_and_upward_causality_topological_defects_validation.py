# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 VII. Field Generation and Upward Causality (Topological Defects) validation tests
"""Tests for Paper 0 VII. Field Generation and Upward Causality (Topological Defects) source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.vii_field_generation_and_upward_causality_topological_defects_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    ViiFieldGenerationAndUpwardCausalityTopologicalDefectsConfig,
    classify_vii_field_generation_and_upward_causality_topological_defects_component,
    validate_vii_field_generation_and_upward_causality_topological_defects_fixture,
    vii_field_generation_and_upward_causality_topological_defects_labels,
)


def test_vii_field_generation_and_upward_causality_topological_defects_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_vii_field_generation_and_upward_causality_topological_defects_fixture()
    assert result.source_ledger_span == ("P0R03250", "P0R03259")
    assert result.source_record_count == 10
    assert result.component_count == 4
    assert result.next_source_boundary == "P0R03260"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_vii_field_generation_and_upward_causality_topological_defects_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R03250"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R03259"


def test_vii_field_generation_and_upward_causality_topological_defects_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "vii_field_generation_and_upward_causality_topological_defects",
        "viii_the_combination_problem_and_panpsychist_fusion_quantum_mereology",
        "integrative_mechanisms_in_short",
        "i_the_unifying_computational_principle_hpc",
    ):
        assert (
            classify_vii_field_generation_and_upward_causality_topological_defects_component(
                component
            )
            == f"{component}_source_boundary"
        )
    labels = vii_field_generation_and_upward_causality_topological_defects_labels()
    assert labels["section"] == "VII. Field Generation and Upward Causality (Topological Defects)"
    assert labels["next_boundary"] == "P0R03260"


def test_vii_field_generation_and_upward_causality_topological_defects_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 10"):
        ViiFieldGenerationAndUpwardCausalityTopologicalDefectsConfig(
            expected_source_record_count=9
        )
    with pytest.raises(ValueError, match="expected_component_count must equal 4"):
        ViiFieldGenerationAndUpwardCausalityTopologicalDefectsConfig(expected_component_count=5)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R03260"):
        ViiFieldGenerationAndUpwardCausalityTopologicalDefectsConfig(
            next_source_boundary="P0R03259"
        )
    with pytest.raises(
        ValueError,
        match="unknown vii_field_generation_and_upward_causality_topological_defects component",
    ):
        classify_vii_field_generation_and_upward_causality_topological_defects_component(
            "empirical_validation_claim"
        )
