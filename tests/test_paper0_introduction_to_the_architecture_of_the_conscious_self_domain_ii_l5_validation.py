# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Introduction to The Architecture of the Conscious Self (Domain II: L5) validation tests
"""Tests for Paper 0 Introduction to The Architecture of the Conscious Self (Domain II: L5) source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.introduction_to_the_architecture_of_the_conscious_self_domain_ii_l5_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    IntroductionToTheArchitectureOfTheConsciousSelfDomainIiL5Config,
    classify_introduction_to_the_architecture_of_the_conscious_self_domain_ii_l5_component,
    introduction_to_the_architecture_of_the_conscious_self_domain_ii_l5_labels,
    validate_introduction_to_the_architecture_of_the_conscious_self_domain_ii_l5_fixture,
)


def test_introduction_to_the_architecture_of_the_conscious_self_domain_ii_l5_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_introduction_to_the_architecture_of_the_conscious_self_domain_ii_l5_fixture()
    assert result.source_ledger_span == ("P0R04589", "P0R04597")
    assert result.source_record_count == 9
    assert result.component_count == 3
    assert result.next_source_boundary == "P0R04598"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_introduction_to_the_architecture_of_the_conscious_self_domain_ii_l5_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R04589"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R04597"


def test_introduction_to_the_architecture_of_the_conscious_self_domain_ii_l5_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "introduction_to_the_architecture_of_the_conscious_self_domain_ii_l5",
        "iv_examination_of_the_architecture_of_the_conscious_self_domain_ii_l5",
        "hpc_and_the_canonical_microcircuit_the_engine_of_inference",
    ):
        assert (
            classify_introduction_to_the_architecture_of_the_conscious_self_domain_ii_l5_component(
                component
            )
            == f"{component}_source_boundary"
        )
    labels = introduction_to_the_architecture_of_the_conscious_self_domain_ii_l5_labels()
    assert (
        labels["section"]
        == "Introduction to The Architecture of the Conscious Self (Domain II: L5)"
    )
    assert labels["next_boundary"] == "P0R04598"


def test_introduction_to_the_architecture_of_the_conscious_self_domain_ii_l5_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 9"):
        IntroductionToTheArchitectureOfTheConsciousSelfDomainIiL5Config(
            expected_source_record_count=8
        )
    with pytest.raises(ValueError, match="expected_component_count must equal 3"):
        IntroductionToTheArchitectureOfTheConsciousSelfDomainIiL5Config(expected_component_count=4)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R04598"):
        IntroductionToTheArchitectureOfTheConsciousSelfDomainIiL5Config(
            next_source_boundary="P0R04597"
        )
    with pytest.raises(
        ValueError,
        match="unknown introduction_to_the_architecture_of_the_conscious_self_domain_ii_l5 component",
    ):
        classify_introduction_to_the_architecture_of_the_conscious_self_domain_ii_l5_component(
            "empirical_validation_claim"
        )
