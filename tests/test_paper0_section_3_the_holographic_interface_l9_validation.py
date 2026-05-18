# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 3. The Holographic Interface (L9): validation tests
"""Tests for Paper 0 3. The Holographic Interface (L9): source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.section_3_the_holographic_interface_l9_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    Section3TheHolographicInterfaceL9Config,
    classify_section_3_the_holographic_interface_l9_component,
    section_3_the_holographic_interface_l9_labels,
    validate_section_3_the_holographic_interface_l9_fixture,
)


def test_section_3_the_holographic_interface_l9_fixture_preserves_source_boundary() -> None:
    result = validate_section_3_the_holographic_interface_l9_fixture()
    assert result.source_ledger_span == ("P0R05009", "P0R05016")
    assert result.source_record_count == 8
    assert result.component_count == 4
    assert result.next_source_boundary == "P0R05017"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_section_3_the_holographic_interface_l9_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R05009"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R05016"


def test_section_3_the_holographic_interface_l9_classification_and_labels_are_explicit() -> None:
    for component in (
        "3_the_holographic_interface_l9",
        "v_dynamics_across_the_lifespan_development_ageing_and_sleep",
        "1_development_the_ascent_to_criticality",
        "2_ageing_the_descent_from_criticality_and_decoherence",
    ):
        assert (
            classify_section_3_the_holographic_interface_l9_component(component)
            == f"{component}_source_boundary"
        )
    labels = section_3_the_holographic_interface_l9_labels()
    assert labels["section"] == "3. The Holographic Interface (L9):"
    assert labels["next_boundary"] == "P0R05017"


def test_section_3_the_holographic_interface_l9_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 8"):
        Section3TheHolographicInterfaceL9Config(expected_source_record_count=7)
    with pytest.raises(ValueError, match="expected_component_count must equal 4"):
        Section3TheHolographicInterfaceL9Config(expected_component_count=5)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R05017"):
        Section3TheHolographicInterfaceL9Config(next_source_boundary="P0R05016")
    with pytest.raises(
        ValueError, match="unknown section_3_the_holographic_interface_l9 component"
    ):
        classify_section_3_the_holographic_interface_l9_component("empirical_validation_claim")
