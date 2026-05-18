# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 The Scaling Law of Consciousness (SLC) validation tests
"""Tests for Paper 0 The Scaling Law of Consciousness (SLC) source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.the_scaling_law_of_consciousness_slc_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    TheScalingLawOfConsciousnessSlcConfig,
    classify_the_scaling_law_of_consciousness_slc_component,
    the_scaling_law_of_consciousness_slc_labels,
    validate_the_scaling_law_of_consciousness_slc_fixture,
)


def test_the_scaling_law_of_consciousness_slc_fixture_preserves_source_boundary() -> None:
    result = validate_the_scaling_law_of_consciousness_slc_fixture()
    assert result.source_ledger_span == ("P0R03478", "P0R03491")
    assert result.source_record_count == 14
    assert result.component_count == 1
    assert result.next_source_boundary == "P0R03492"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_the_scaling_law_of_consciousness_slc_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R03478"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R03491"


def test_the_scaling_law_of_consciousness_slc_classification_and_labels_are_explicit() -> None:
    for component in ("the_scaling_law_of_consciousness_slc",):
        assert (
            classify_the_scaling_law_of_consciousness_slc_component(component)
            == f"{component}_source_boundary"
        )
    labels = the_scaling_law_of_consciousness_slc_labels()
    assert labels["section"] == "The Scaling Law of Consciousness (SLC)"
    assert labels["next_boundary"] == "P0R03492"


def test_the_scaling_law_of_consciousness_slc_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 14"):
        TheScalingLawOfConsciousnessSlcConfig(expected_source_record_count=13)
    with pytest.raises(ValueError, match="expected_component_count must equal 1"):
        TheScalingLawOfConsciousnessSlcConfig(expected_component_count=2)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R03492"):
        TheScalingLawOfConsciousnessSlcConfig(next_source_boundary="P0R03491")
    with pytest.raises(ValueError, match="unknown the_scaling_law_of_consciousness_slc component"):
        classify_the_scaling_law_of_consciousness_slc_component("empirical_validation_claim")
