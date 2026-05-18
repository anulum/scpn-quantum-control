# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 The Mechanism of Influence: validation tests
"""Tests for Paper 0 The Mechanism of Influence: source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.the_mechanism_of_influence_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    TheMechanismOfInfluenceConfig,
    classify_the_mechanism_of_influence_component,
    the_mechanism_of_influence_labels,
    validate_the_mechanism_of_influence_fixture,
)


def test_the_mechanism_of_influence_fixture_preserves_source_boundary() -> None:
    result = validate_the_mechanism_of_influence_fixture()
    assert result.source_ledger_span == ("P0R02616", "P0R02623")
    assert result.source_record_count == 8
    assert result.component_count == 4
    assert result.next_source_boundary == "P0R02624"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_the_mechanism_of_influence_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R02616"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R02623"


def test_the_mechanism_of_influence_classification_and_labels_are_explicit() -> None:
    for component in (
        "the_mechanism_of_influence",
        "the_unified_phase_dynamics_equation_upde_the_spine_of_the_scpn",
        "the_upde_formalism",
        "components_of_the_upde",
    ):
        assert (
            classify_the_mechanism_of_influence_component(component)
            == f"{component}_source_boundary"
        )
    labels = the_mechanism_of_influence_labels()
    assert labels["section"] == "The Mechanism of Influence:"
    assert labels["next_boundary"] == "P0R02624"


def test_the_mechanism_of_influence_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 8"):
        TheMechanismOfInfluenceConfig(expected_source_record_count=7)
    with pytest.raises(ValueError, match="expected_component_count must equal 4"):
        TheMechanismOfInfluenceConfig(expected_component_count=5)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R02624"):
        TheMechanismOfInfluenceConfig(next_source_boundary="P0R02623")
    with pytest.raises(ValueError, match="unknown the_mechanism_of_influence component"):
        classify_the_mechanism_of_influence_component("empirical_validation_claim")
