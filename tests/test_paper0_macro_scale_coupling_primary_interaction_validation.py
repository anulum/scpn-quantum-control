# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Macro-Scale Coupling (Primary Interaction): validation tests
"""Tests for Paper 0 Macro-Scale Coupling (Primary Interaction): source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.macro_scale_coupling_primary_interaction_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    MacroScaleCouplingPrimaryInteractionConfig,
    classify_macro_scale_coupling_primary_interaction_component,
    macro_scale_coupling_primary_interaction_labels,
    validate_macro_scale_coupling_primary_interaction_fixture,
)


def test_macro_scale_coupling_primary_interaction_fixture_preserves_source_boundary() -> None:
    result = validate_macro_scale_coupling_primary_interaction_fixture()
    assert result.source_ledger_span == ("P0R02107", "P0R02127")
    assert result.source_record_count == 21
    assert result.component_count == 4
    assert result.next_source_boundary == "P0R02128"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_macro_scale_coupling_primary_interaction_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R02107"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R02127"


def test_macro_scale_coupling_primary_interaction_classification_and_labels_are_explicit() -> None:
    for component in (
        "macro_scale_coupling_primary_interaction",
        "meso_scale_transduction",
        "quantum_scale_coupling_secondary_interaction",
        "domain_i_biological_substrate_layers_1_4",
    ):
        assert (
            classify_macro_scale_coupling_primary_interaction_component(component)
            == f"{component}_source_boundary"
        )
    labels = macro_scale_coupling_primary_interaction_labels()
    assert labels["section"] == "Macro-Scale Coupling (Primary Interaction):"
    assert labels["next_boundary"] == "P0R02128"


def test_macro_scale_coupling_primary_interaction_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 21"):
        MacroScaleCouplingPrimaryInteractionConfig(expected_source_record_count=20)
    with pytest.raises(ValueError, match="expected_component_count must equal 4"):
        MacroScaleCouplingPrimaryInteractionConfig(expected_component_count=5)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R02128"):
        MacroScaleCouplingPrimaryInteractionConfig(next_source_boundary="P0R02127")
    with pytest.raises(
        ValueError, match="unknown macro_scale_coupling_primary_interaction component"
    ):
        classify_macro_scale_coupling_primary_interaction_component("empirical_validation_claim")
