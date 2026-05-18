# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 The Microbiome as a Foundational Control Layer validation tests
"""Tests for Paper 0 The Microbiome as a Foundational Control Layer source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.the_microbiome_as_a_foundational_control_layer_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    TheMicrobiomeAsAFoundationalControlLayerConfig,
    classify_the_microbiome_as_a_foundational_control_layer_component,
    the_microbiome_as_a_foundational_control_layer_labels,
    validate_the_microbiome_as_a_foundational_control_layer_fixture,
)


def test_the_microbiome_as_a_foundational_control_layer_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_the_microbiome_as_a_foundational_control_layer_fixture()
    assert result.source_ledger_span == ("P0R05479", "P0R05492")
    assert result.source_record_count == 14
    assert result.component_count == 2
    assert result.next_source_boundary == "P0R05493"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_the_microbiome_as_a_foundational_control_layer_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R05479"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R05492"


def test_the_microbiome_as_a_foundational_control_layer_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "the_microbiome_as_a_foundational_control_layer",
        "a_two_timescale_model_of_glial_neuronal_coupling",
    ):
        assert (
            classify_the_microbiome_as_a_foundational_control_layer_component(component)
            == f"{component}_source_boundary"
        )
    labels = the_microbiome_as_a_foundational_control_layer_labels()
    assert labels["section"] == "The Microbiome as a Foundational Control Layer"
    assert labels["next_boundary"] == "P0R05493"


def test_the_microbiome_as_a_foundational_control_layer_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 14"):
        TheMicrobiomeAsAFoundationalControlLayerConfig(expected_source_record_count=13)
    with pytest.raises(ValueError, match="expected_component_count must equal 2"):
        TheMicrobiomeAsAFoundationalControlLayerConfig(expected_component_count=3)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R05493"):
        TheMicrobiomeAsAFoundationalControlLayerConfig(next_source_boundary="P0R05492")
    with pytest.raises(
        ValueError, match="unknown the_microbiome_as_a_foundational_control_layer component"
    ):
        classify_the_microbiome_as_a_foundational_control_layer_component(
            "empirical_validation_claim"
        )
