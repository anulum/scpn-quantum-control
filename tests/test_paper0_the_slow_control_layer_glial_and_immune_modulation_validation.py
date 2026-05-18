# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 The Slow Control Layer - Glial and Immune Modulation validation tests
"""Tests for Paper 0 The Slow Control Layer - Glial and Immune Modulation source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.the_slow_control_layer_glial_and_immune_modulation_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    TheSlowControlLayerGlialAndImmuneModulationConfig,
    classify_the_slow_control_layer_glial_and_immune_modulation_component,
    the_slow_control_layer_glial_and_immune_modulation_labels,
    validate_the_slow_control_layer_glial_and_immune_modulation_fixture,
)


def test_the_slow_control_layer_glial_and_immune_modulation_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_the_slow_control_layer_glial_and_immune_modulation_fixture()
    assert result.source_ledger_span == ("P0R05347", "P0R05357")
    assert result.source_record_count == 11
    assert result.component_count == 2
    assert result.next_source_boundary == "P0R05358"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_the_slow_control_layer_glial_and_immune_modulation_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R05347"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R05357"


def test_the_slow_control_layer_glial_and_immune_modulation_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "the_slow_control_layer_glial_and_immune_modulation",
        "i_the_astrocyte_neuron_lattice_l2_l4_modulation",
    ):
        assert (
            classify_the_slow_control_layer_glial_and_immune_modulation_component(component)
            == f"{component}_source_boundary"
        )
    labels = the_slow_control_layer_glial_and_immune_modulation_labels()
    assert labels["section"] == "The Slow Control Layer - Glial and Immune Modulation"
    assert labels["next_boundary"] == "P0R05358"


def test_the_slow_control_layer_glial_and_immune_modulation_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 11"):
        TheSlowControlLayerGlialAndImmuneModulationConfig(expected_source_record_count=10)
    with pytest.raises(ValueError, match="expected_component_count must equal 2"):
        TheSlowControlLayerGlialAndImmuneModulationConfig(expected_component_count=3)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R05358"):
        TheSlowControlLayerGlialAndImmuneModulationConfig(next_source_boundary="P0R05357")
    with pytest.raises(
        ValueError, match="unknown the_slow_control_layer_glial_and_immune_modulation component"
    ):
        classify_the_slow_control_layer_glial_and_immune_modulation_component(
            "empirical_validation_claim"
        )
