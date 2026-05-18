# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 5.2 Embodied SCPN: Cellular, Neural, & Systemic Implementation validation tests
"""Tests for Paper 0 5.2 Embodied SCPN: Cellular, Neural, & Systemic Implementation source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.section_5_2_embodied_scpn_cellular_neural_systemic_implementation_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    Section52EmbodiedScpnCellularNeuralSystemicImplementationConfig,
    classify_section_5_2_embodied_scpn_cellular_neural_systemic_implementation_component,
    section_5_2_embodied_scpn_cellular_neural_systemic_implementation_labels,
    validate_section_5_2_embodied_scpn_cellular_neural_systemic_implementation_fixture,
)


def test_section_5_2_embodied_scpn_cellular_neural_systemic_implementation_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_section_5_2_embodied_scpn_cellular_neural_systemic_implementation_fixture()
    assert result.source_ledger_span == ("P0R04372", "P0R04379")
    assert result.source_record_count == 8
    assert result.component_count == 2
    assert result.next_source_boundary == "P0R04380"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_section_5_2_embodied_scpn_cellular_neural_systemic_implementation_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R04372"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R04379"


def test_section_5_2_embodied_scpn_cellular_neural_systemic_implementation_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "5_2_embodied_scpn_cellular_neural_systemic_implementation",
        "i_the_unified_geometric_principle_ugp_and_axiomatic_foundations",
    ):
        assert (
            classify_section_5_2_embodied_scpn_cellular_neural_systemic_implementation_component(
                component
            )
            == f"{component}_source_boundary"
        )
    labels = section_5_2_embodied_scpn_cellular_neural_systemic_implementation_labels()
    assert labels["section"] == "5.2 Embodied SCPN: Cellular, Neural, & Systemic Implementation"
    assert labels["next_boundary"] == "P0R04380"


def test_section_5_2_embodied_scpn_cellular_neural_systemic_implementation_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 8"):
        Section52EmbodiedScpnCellularNeuralSystemicImplementationConfig(
            expected_source_record_count=7
        )
    with pytest.raises(ValueError, match="expected_component_count must equal 2"):
        Section52EmbodiedScpnCellularNeuralSystemicImplementationConfig(expected_component_count=3)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R04380"):
        Section52EmbodiedScpnCellularNeuralSystemicImplementationConfig(
            next_source_boundary="P0R04379"
        )
    with pytest.raises(
        ValueError,
        match="unknown section_5_2_embodied_scpn_cellular_neural_systemic_implementation component",
    ):
        classify_section_5_2_embodied_scpn_cellular_neural_systemic_implementation_component(
            "empirical_validation_claim"
        )
