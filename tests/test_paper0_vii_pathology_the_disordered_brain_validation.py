# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 VII. Pathology: The Disordered Brain validation tests
"""Tests for Paper 0 VII. Pathology: The Disordered Brain source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.vii_pathology_the_disordered_brain_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    ViiPathologyTheDisorderedBrainConfig,
    classify_vii_pathology_the_disordered_brain_component,
    validate_vii_pathology_the_disordered_brain_fixture,
    vii_pathology_the_disordered_brain_labels,
)


def test_vii_pathology_the_disordered_brain_fixture_preserves_source_boundary() -> None:
    result = validate_vii_pathology_the_disordered_brain_fixture()
    assert result.source_ledger_span == ("P0R04534", "P0R04543")
    assert result.source_record_count == 10
    assert result.component_count == 3
    assert result.next_source_boundary == "P0R04544"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_vii_pathology_the_disordered_brain_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R04534"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R04543"


def test_vii_pathology_the_disordered_brain_classification_and_labels_are_explicit() -> None:
    for component in (
        "vii_pathology_the_disordered_brain",
        "the_embodied_engine_a_deeper_neurobiological_grounding_for_the_scpn",
        "introduction_to_the_deep_architecture_of_the_quantum_biological_interfac",
    ):
        assert (
            classify_vii_pathology_the_disordered_brain_component(component)
            == f"{component}_source_boundary"
        )
    labels = vii_pathology_the_disordered_brain_labels()
    assert labels["section"] == "VII. Pathology: The Disordered Brain"
    assert labels["next_boundary"] == "P0R04544"


def test_vii_pathology_the_disordered_brain_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 10"):
        ViiPathologyTheDisorderedBrainConfig(expected_source_record_count=9)
    with pytest.raises(ValueError, match="expected_component_count must equal 3"):
        ViiPathologyTheDisorderedBrainConfig(expected_component_count=4)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R04544"):
        ViiPathologyTheDisorderedBrainConfig(next_source_boundary="P0R04543")
    with pytest.raises(ValueError, match="unknown vii_pathology_the_disordered_brain component"):
        classify_vii_pathology_the_disordered_brain_component("empirical_validation_claim")
