# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Glial Scaffolding: Astrocytic Regulation of Neural Synchrony and Criticality validation tests
"""Tests for Paper 0 Glial Scaffolding: Astrocytic Regulation of Neural Synchrony and Criticality source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.glial_scaffolding_astrocytic_regulation_of_neural_synchrony_and_critical_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    GlialScaffoldingAstrocyticRegulationOfNeuralSynchronyAndCriticalConfig,
    classify_glial_scaffolding_astrocytic_regulation_of_neural_synchrony_and_critical_component,
    glial_scaffolding_astrocytic_regulation_of_neural_synchrony_and_critical_labels,
    validate_glial_scaffolding_astrocytic_regulation_of_neural_synchrony_and_critical_fixture,
)


def test_glial_scaffolding_astrocytic_regulation_of_neural_synchrony_and_critical_fixture_preserves_source_boundary() -> (
    None
):
    result = (
        validate_glial_scaffolding_astrocytic_regulation_of_neural_synchrony_and_critical_fixture()
    )
    assert result.source_ledger_span == ("P0R05455", "P0R05478")
    assert result.source_record_count == 24
    assert result.component_count == 2
    assert result.next_source_boundary == "P0R05479"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_glial_scaffolding_astrocytic_regulation_of_neural_synchrony_and_critical_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R05455"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R05478"


def test_glial_scaffolding_astrocytic_regulation_of_neural_synchrony_and_critical_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "glial_scaffolding_astrocytic_regulation_of_neural_synchrony_and_critical",
        "the_neuro_immune_interface_state_space_geometry_and_embodied_coherence",
    ):
        assert (
            classify_glial_scaffolding_astrocytic_regulation_of_neural_synchrony_and_critical_component(
                component
            )
            == f"{component}_source_boundary"
        )
    labels = glial_scaffolding_astrocytic_regulation_of_neural_synchrony_and_critical_labels()
    assert (
        labels["section"]
        == "Glial Scaffolding: Astrocytic Regulation of Neural Synchrony and Criticality"
    )
    assert labels["next_boundary"] == "P0R05479"


def test_glial_scaffolding_astrocytic_regulation_of_neural_synchrony_and_critical_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 24"):
        GlialScaffoldingAstrocyticRegulationOfNeuralSynchronyAndCriticalConfig(
            expected_source_record_count=23
        )
    with pytest.raises(ValueError, match="expected_component_count must equal 2"):
        GlialScaffoldingAstrocyticRegulationOfNeuralSynchronyAndCriticalConfig(
            expected_component_count=3
        )
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R05479"):
        GlialScaffoldingAstrocyticRegulationOfNeuralSynchronyAndCriticalConfig(
            next_source_boundary="P0R05478"
        )
    with pytest.raises(
        ValueError,
        match="unknown glial_scaffolding_astrocytic_regulation_of_neural_synchrony_and_critical component",
    ):
        classify_glial_scaffolding_astrocytic_regulation_of_neural_synchrony_and_critical_component(
            "empirical_validation_claim"
        )
