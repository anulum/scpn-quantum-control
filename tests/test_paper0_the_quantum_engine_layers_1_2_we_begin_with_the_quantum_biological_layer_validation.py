# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 The Quantum Engine (Layers 1-2): We begin with the Quantum Biological (Layer 1) substrate, where biological systems create order from quantum potentiality, stabilised by mechanisms like QEC and neuroimmune quantum effects. This quantum information is then transduced by the Neurochemical-Neurological (Layer 2) layer, which acts as a dynamic filter, translating field intent into the biochemical language of the nervous system. validation tests
"""Tests for Paper 0 The Quantum Engine (Layers 1-2): We begin with the Quantum Biological (Layer 1) substrate, where biological systems create order from quantum potentiality, stabilised by mechanisms like QEC and neuroimmune quantum effects. This quantum information is then transduced by the Neurochemical-Neurological (Layer 2) layer, which acts as a dynamic filter, translating field intent into the biochemical language of the nervous system. source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.the_quantum_engine_layers_1_2_we_begin_with_the_quantum_biological_layer_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    TheQuantumEngineLayers12WeBeginWithTheQuantumBiologicalLayerConfig,
    classify_the_quantum_engine_layers_1_2_we_begin_with_the_quantum_biological_layer_component,
    the_quantum_engine_layers_1_2_we_begin_with_the_quantum_biological_layer_labels,
    validate_the_quantum_engine_layers_1_2_we_begin_with_the_quantum_biological_layer_fixture,
)


def test_the_quantum_engine_layers_1_2_we_begin_with_the_quantum_biological_layer_fixture_preserves_source_boundary() -> (
    None
):
    result = (
        validate_the_quantum_engine_layers_1_2_we_begin_with_the_quantum_biological_layer_fixture()
    )
    assert result.source_ledger_span == ("P0R05314", "P0R05322")
    assert result.source_record_count == 9
    assert result.component_count == 4
    assert result.next_source_boundary == "P0R05323"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_the_quantum_engine_layers_1_2_we_begin_with_the_quantum_biological_layer_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R05314"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R05322"


def test_the_quantum_engine_layers_1_2_we_begin_with_the_quantum_biological_layer_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "the_quantum_engine_layers_1_2_we_begin_with_the_quantum_biological_layer",
        "the_quantum_classical_bridge_selection_and_amplification",
        "i_guided_einselection_the_emergence_of_classicality",
        "ii_the_amplification_mechanism_quantum_stochastic_resonance_qsr_at_criti",
    ):
        assert (
            classify_the_quantum_engine_layers_1_2_we_begin_with_the_quantum_biological_layer_component(
                component
            )
            == f"{component}_source_boundary"
        )
    labels = the_quantum_engine_layers_1_2_we_begin_with_the_quantum_biological_layer_labels()
    assert (
        labels["section"]
        == "The Quantum Engine (Layers 1-2): We begin with the Quantum Biological (Layer 1) substrate, where biological systems create order from quantum potentiality, stabilised by mechanisms like QEC and neuroimmune quantum effects. This quantum information is then transduced by the Neurochemical-Neurological (Layer 2) layer, which acts as a dynamic filter, translating field intent into the biochemical language of the nervous system."
    )
    assert labels["next_boundary"] == "P0R05323"


def test_the_quantum_engine_layers_1_2_we_begin_with_the_quantum_biological_layer_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 9"):
        TheQuantumEngineLayers12WeBeginWithTheQuantumBiologicalLayerConfig(
            expected_source_record_count=8
        )
    with pytest.raises(ValueError, match="expected_component_count must equal 4"):
        TheQuantumEngineLayers12WeBeginWithTheQuantumBiologicalLayerConfig(
            expected_component_count=5
        )
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R05323"):
        TheQuantumEngineLayers12WeBeginWithTheQuantumBiologicalLayerConfig(
            next_source_boundary="P0R05322"
        )
    with pytest.raises(
        ValueError,
        match="unknown the_quantum_engine_layers_1_2_we_begin_with_the_quantum_biological_layer component",
    ):
        classify_the_quantum_engine_layers_1_2_we_begin_with_the_quantum_biological_layer_component(
            "empirical_validation_claim"
        )
