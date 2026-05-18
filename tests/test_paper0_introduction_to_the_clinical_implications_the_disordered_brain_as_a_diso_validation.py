# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Introduction to The Clinical Implications: The Disordered Brain as a Disordered Architecture validation tests
"""Tests for Paper 0 Introduction to The Clinical Implications: The Disordered Brain as a Disordered Architecture source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.introduction_to_the_clinical_implications_the_disordered_brain_as_a_diso_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    IntroductionToTheClinicalImplicationsTheDisorderedBrainAsADisoConfig,
    classify_introduction_to_the_clinical_implications_the_disordered_brain_as_a_diso_component,
    introduction_to_the_clinical_implications_the_disordered_brain_as_a_diso_labels,
    validate_introduction_to_the_clinical_implications_the_disordered_brain_as_a_diso_fixture,
)


def test_introduction_to_the_clinical_implications_the_disordered_brain_as_a_diso_fixture_preserves_source_boundary() -> (
    None
):
    result = (
        validate_introduction_to_the_clinical_implications_the_disordered_brain_as_a_diso_fixture()
    )
    assert result.source_ledger_span == ("P0R04622", "P0R04629")
    assert result.source_record_count == 8
    assert result.component_count == 2
    assert result.next_source_boundary == "P0R04630"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_introduction_to_the_clinical_implications_the_disordered_brain_as_a_diso_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R04622"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R04629"


def test_introduction_to_the_clinical_implications_the_disordered_brain_as_a_diso_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "introduction_to_the_clinical_implications_the_disordered_brain_as_a_diso",
        "vi_clinical_implications_the_disordered_brain_as_a_disordered_architectu",
    ):
        assert (
            classify_introduction_to_the_clinical_implications_the_disordered_brain_as_a_diso_component(
                component
            )
            == f"{component}_source_boundary"
        )
    labels = introduction_to_the_clinical_implications_the_disordered_brain_as_a_diso_labels()
    assert (
        labels["section"]
        == "Introduction to The Clinical Implications: The Disordered Brain as a Disordered Architecture"
    )
    assert labels["next_boundary"] == "P0R04630"


def test_introduction_to_the_clinical_implications_the_disordered_brain_as_a_diso_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 8"):
        IntroductionToTheClinicalImplicationsTheDisorderedBrainAsADisoConfig(
            expected_source_record_count=7
        )
    with pytest.raises(ValueError, match="expected_component_count must equal 2"):
        IntroductionToTheClinicalImplicationsTheDisorderedBrainAsADisoConfig(
            expected_component_count=3
        )
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R04630"):
        IntroductionToTheClinicalImplicationsTheDisorderedBrainAsADisoConfig(
            next_source_boundary="P0R04629"
        )
    with pytest.raises(
        ValueError,
        match="unknown introduction_to_the_clinical_implications_the_disordered_brain_as_a_diso component",
    ):
        classify_introduction_to_the_clinical_implications_the_disordered_brain_as_a_diso_component(
            "empirical_validation_claim"
        )
