# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 V. The Interface Problem Synthesis (Mind-Body-Field) validation tests
"""Tests for Paper 0 V. The Interface Problem Synthesis (Mind-Body-Field) source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.v_the_interface_problem_synthesis_mind_body_field_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    VTheInterfaceProblemSynthesisMindBodyFieldConfig,
    classify_v_the_interface_problem_synthesis_mind_body_field_component,
    v_the_interface_problem_synthesis_mind_body_field_labels,
    validate_v_the_interface_problem_synthesis_mind_body_field_fixture,
)


def test_v_the_interface_problem_synthesis_mind_body_field_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_v_the_interface_problem_synthesis_mind_body_field_fixture()
    assert result.source_ledger_span == ("P0R03241", "P0R03249")
    assert result.source_record_count == 9
    assert result.component_count == 4
    assert result.next_source_boundary == "P0R03250"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_v_the_interface_problem_synthesis_mind_body_field_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R03241"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R03249"


def test_v_the_interface_problem_synthesis_mind_body_field_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "v_the_interface_problem_synthesis_mind_body_field",
        "downward_causation_the_psi_field_biases_physical_dynamics_via_qze_stabil",
        "upward_causation_the_physical_substrate_structures_the_psi_field_by_enco",
        "vi_formalising_emergence_phase_transitions_and_ginzburg_landau_theory",
    ):
        assert (
            classify_v_the_interface_problem_synthesis_mind_body_field_component(component)
            == f"{component}_source_boundary"
        )
    labels = v_the_interface_problem_synthesis_mind_body_field_labels()
    assert labels["section"] == "V. The Interface Problem Synthesis (Mind-Body-Field)"
    assert labels["next_boundary"] == "P0R03250"


def test_v_the_interface_problem_synthesis_mind_body_field_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 9"):
        VTheInterfaceProblemSynthesisMindBodyFieldConfig(expected_source_record_count=8)
    with pytest.raises(ValueError, match="expected_component_count must equal 4"):
        VTheInterfaceProblemSynthesisMindBodyFieldConfig(expected_component_count=5)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R03250"):
        VTheInterfaceProblemSynthesisMindBodyFieldConfig(next_source_boundary="P0R03249")
    with pytest.raises(
        ValueError, match="unknown v_the_interface_problem_synthesis_mind_body_field component"
    ):
        classify_v_the_interface_problem_synthesis_mind_body_field_component(
            "empirical_validation_claim"
        )
