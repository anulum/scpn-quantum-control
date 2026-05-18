# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 III. The Developmental and Plasticity Landscape (L3) validation tests
"""Tests for Paper 0 III. The Developmental and Plasticity Landscape (L3) source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.iii_the_developmental_and_plasticity_landscape_l3_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    IiiTheDevelopmentalAndPlasticityLandscapeL3Config,
    classify_iii_the_developmental_and_plasticity_landscape_l3_component,
    iii_the_developmental_and_plasticity_landscape_l3_labels,
    validate_iii_the_developmental_and_plasticity_landscape_l3_fixture,
)


def test_iii_the_developmental_and_plasticity_landscape_l3_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_iii_the_developmental_and_plasticity_landscape_l3_fixture()
    assert result.source_ledger_span == ("P0R04478", "P0R04487")
    assert result.source_record_count == 10
    assert result.component_count == 3
    assert result.next_source_boundary == "P0R04488"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_iii_the_developmental_and_plasticity_landscape_l3_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R04478"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R04487"


def test_iii_the_developmental_and_plasticity_landscape_l3_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "iii_the_developmental_and_plasticity_landscape_l3",
        "iv_the_dynamic_core_synchronisation_criticality_and_the_connectome_l4",
        "1_the_upde_in_the_brain_the_neural_symphony",
    ):
        assert (
            classify_iii_the_developmental_and_plasticity_landscape_l3_component(component)
            == f"{component}_source_boundary"
        )
    labels = iii_the_developmental_and_plasticity_landscape_l3_labels()
    assert labels["section"] == "III. The Developmental and Plasticity Landscape (L3)"
    assert labels["next_boundary"] == "P0R04488"


def test_iii_the_developmental_and_plasticity_landscape_l3_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 10"):
        IiiTheDevelopmentalAndPlasticityLandscapeL3Config(expected_source_record_count=9)
    with pytest.raises(ValueError, match="expected_component_count must equal 3"):
        IiiTheDevelopmentalAndPlasticityLandscapeL3Config(expected_component_count=4)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R04488"):
        IiiTheDevelopmentalAndPlasticityLandscapeL3Config(next_source_boundary="P0R04487")
    with pytest.raises(
        ValueError, match="unknown iii_the_developmental_and_plasticity_landscape_l3 component"
    ):
        classify_iii_the_developmental_and_plasticity_landscape_l3_component(
            "empirical_validation_claim"
        )
