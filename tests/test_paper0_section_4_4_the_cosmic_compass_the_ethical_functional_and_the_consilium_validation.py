# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 4.4 The Cosmic Compass: The Ethical Functional and the Consilium validation tests
"""Tests for Paper 0 4.4 The Cosmic Compass: The Ethical Functional and the Consilium source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.section_4_4_the_cosmic_compass_the_ethical_functional_and_the_consilium_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    Section44TheCosmicCompassTheEthicalFunctionalAndTheConsiliumConfig,
    classify_section_4_4_the_cosmic_compass_the_ethical_functional_and_the_consilium_component,
    section_4_4_the_cosmic_compass_the_ethical_functional_and_the_consilium_labels,
    validate_section_4_4_the_cosmic_compass_the_ethical_functional_and_the_consilium_fixture,
)


def test_section_4_4_the_cosmic_compass_the_ethical_functional_and_the_consilium_fixture_preserves_source_boundary() -> (
    None
):
    result = (
        validate_section_4_4_the_cosmic_compass_the_ethical_functional_and_the_consilium_fixture()
    )
    assert result.source_ledger_span == ("P0R03932", "P0R03944")
    assert result.source_record_count == 13
    assert result.component_count == 2
    assert result.next_source_boundary == "P0R03945"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_section_4_4_the_cosmic_compass_the_ethical_functional_and_the_consilium_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R03932"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R03944"


def test_section_4_4_the_cosmic_compass_the_ethical_functional_and_the_consilium_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "4_4_the_cosmic_compass_the_ethical_functional_and_the_consilium",
        "the_ethical_functional",
    ):
        assert (
            classify_section_4_4_the_cosmic_compass_the_ethical_functional_and_the_consilium_component(
                component
            )
            == f"{component}_source_boundary"
        )
    labels = section_4_4_the_cosmic_compass_the_ethical_functional_and_the_consilium_labels()
    assert labels["section"] == "4.4 The Cosmic Compass: The Ethical Functional and the Consilium"
    assert labels["next_boundary"] == "P0R03945"


def test_section_4_4_the_cosmic_compass_the_ethical_functional_and_the_consilium_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 13"):
        Section44TheCosmicCompassTheEthicalFunctionalAndTheConsiliumConfig(
            expected_source_record_count=12
        )
    with pytest.raises(ValueError, match="expected_component_count must equal 2"):
        Section44TheCosmicCompassTheEthicalFunctionalAndTheConsiliumConfig(
            expected_component_count=3
        )
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R03945"):
        Section44TheCosmicCompassTheEthicalFunctionalAndTheConsiliumConfig(
            next_source_boundary="P0R03944"
        )
    with pytest.raises(
        ValueError,
        match="unknown section_4_4_the_cosmic_compass_the_ethical_functional_and_the_consilium component",
    ):
        classify_section_4_4_the_cosmic_compass_the_ethical_functional_and_the_consilium_component(
            "empirical_validation_claim"
        )
