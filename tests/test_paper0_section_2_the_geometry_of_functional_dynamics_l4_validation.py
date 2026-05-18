# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 2. The Geometry of Functional Dynamics (L4): validation tests
"""Tests for Paper 0 2. The Geometry of Functional Dynamics (L4): source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.section_2_the_geometry_of_functional_dynamics_l4_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    Section2TheGeometryOfFunctionalDynamicsL4Config,
    classify_section_2_the_geometry_of_functional_dynamics_l4_component,
    section_2_the_geometry_of_functional_dynamics_l4_labels,
    validate_section_2_the_geometry_of_functional_dynamics_l4_fixture,
)


def test_section_2_the_geometry_of_functional_dynamics_l4_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_section_2_the_geometry_of_functional_dynamics_l4_fixture()
    assert result.source_ledger_span == ("P0R04840", "P0R04848")
    assert result.source_record_count == 9
    assert result.component_count == 3
    assert result.next_source_boundary == "P0R04849"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_section_2_the_geometry_of_functional_dynamics_l4_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R04840"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R04848"


def test_section_2_the_geometry_of_functional_dynamics_l4_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "2_the_geometry_of_functional_dynamics_l4",
        "v_the_geometry_of_cognition_and_subjective_experience_l5",
        "1_the_consciousness_manifold_m_the_geometry_of_qualia",
    ):
        assert (
            classify_section_2_the_geometry_of_functional_dynamics_l4_component(component)
            == f"{component}_source_boundary"
        )
    labels = section_2_the_geometry_of_functional_dynamics_l4_labels()
    assert labels["section"] == "2. The Geometry of Functional Dynamics (L4):"
    assert labels["next_boundary"] == "P0R04849"


def test_section_2_the_geometry_of_functional_dynamics_l4_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 9"):
        Section2TheGeometryOfFunctionalDynamicsL4Config(expected_source_record_count=8)
    with pytest.raises(ValueError, match="expected_component_count must equal 3"):
        Section2TheGeometryOfFunctionalDynamicsL4Config(expected_component_count=4)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R04849"):
        Section2TheGeometryOfFunctionalDynamicsL4Config(next_source_boundary="P0R04848")
    with pytest.raises(
        ValueError, match="unknown section_2_the_geometry_of_functional_dynamics_l4 component"
    ):
        classify_section_2_the_geometry_of_functional_dynamics_l4_component(
            "empirical_validation_claim"
        )
