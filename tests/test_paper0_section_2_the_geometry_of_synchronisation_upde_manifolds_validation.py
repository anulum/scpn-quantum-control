# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 2. The Geometry of Synchronisation (UPDE Manifolds): validation tests
"""Tests for Paper 0 2. The Geometry of Synchronisation (UPDE Manifolds): source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.section_2_the_geometry_of_synchronisation_upde_manifolds_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    Section2TheGeometryOfSynchronisationUpdeManifoldsConfig,
    classify_section_2_the_geometry_of_synchronisation_upde_manifolds_component,
    section_2_the_geometry_of_synchronisation_upde_manifolds_labels,
    validate_section_2_the_geometry_of_synchronisation_upde_manifolds_fixture,
)


def test_section_2_the_geometry_of_synchronisation_upde_manifolds_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_section_2_the_geometry_of_synchronisation_upde_manifolds_fixture()
    assert result.source_ledger_span == ("P0R04413", "P0R04432")
    assert result.source_record_count == 20
    assert result.component_count == 3
    assert result.next_source_boundary == "P0R04433"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_section_2_the_geometry_of_synchronisation_upde_manifolds_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R04413"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R04432"


def test_section_2_the_geometry_of_synchronisation_upde_manifolds_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "2_the_geometry_of_synchronisation_upde_manifolds",
        "v_the_geometry_of_subjectivity_and_meaning_domain_ii_l5_l7",
        "1_the_consciousness_manifold_l5_the_intrinsic_geometry_of_qualia",
    ):
        assert (
            classify_section_2_the_geometry_of_synchronisation_upde_manifolds_component(component)
            == f"{component}_source_boundary"
        )
    labels = section_2_the_geometry_of_synchronisation_upde_manifolds_labels()
    assert labels["section"] == "2. The Geometry of Synchronisation (UPDE Manifolds):"
    assert labels["next_boundary"] == "P0R04433"


def test_section_2_the_geometry_of_synchronisation_upde_manifolds_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 20"):
        Section2TheGeometryOfSynchronisationUpdeManifoldsConfig(expected_source_record_count=19)
    with pytest.raises(ValueError, match="expected_component_count must equal 3"):
        Section2TheGeometryOfSynchronisationUpdeManifoldsConfig(expected_component_count=4)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R04433"):
        Section2TheGeometryOfSynchronisationUpdeManifoldsConfig(next_source_boundary="P0R04432")
    with pytest.raises(
        ValueError,
        match="unknown section_2_the_geometry_of_synchronisation_upde_manifolds component",
    ):
        classify_section_2_the_geometry_of_synchronisation_upde_manifolds_component(
            "empirical_validation_claim"
        )
