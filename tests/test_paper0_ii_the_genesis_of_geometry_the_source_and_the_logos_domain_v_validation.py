# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 II. The Genesis of Geometry: The Source and the Logos (Domain V) validation tests
"""Tests for Paper 0 II. The Genesis of Geometry: The Source and the Logos (Domain V) source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.ii_the_genesis_of_geometry_the_source_and_the_logos_domain_v_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    IiTheGenesisOfGeometryTheSourceAndTheLogosDomainVConfig,
    classify_ii_the_genesis_of_geometry_the_source_and_the_logos_domain_v_component,
    ii_the_genesis_of_geometry_the_source_and_the_logos_domain_v_labels,
    validate_ii_the_genesis_of_geometry_the_source_and_the_logos_domain_v_fixture,
)


def test_ii_the_genesis_of_geometry_the_source_and_the_logos_domain_v_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_ii_the_genesis_of_geometry_the_source_and_the_logos_domain_v_fixture()
    assert result.source_ledger_span == ("P0R04380", "P0R04387")
    assert result.source_record_count == 8
    assert result.component_count == 3
    assert result.next_source_boundary == "P0R04388"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_ii_the_genesis_of_geometry_the_source_and_the_logos_domain_v_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R04380"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R04387"


def test_ii_the_genesis_of_geometry_the_source_and_the_logos_domain_v_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "ii_the_genesis_of_geometry_the_source_and_the_logos_domain_v",
        "1_the_source_field_as_a_fiber_bundle_l13",
        "2_the_ethical_functional_as_the_principal_connection_l15",
    ):
        assert (
            classify_ii_the_genesis_of_geometry_the_source_and_the_logos_domain_v_component(
                component
            )
            == f"{component}_source_boundary"
        )
    labels = ii_the_genesis_of_geometry_the_source_and_the_logos_domain_v_labels()
    assert labels["section"] == "II. The Genesis of Geometry: The Source and the Logos (Domain V)"
    assert labels["next_boundary"] == "P0R04388"


def test_ii_the_genesis_of_geometry_the_source_and_the_logos_domain_v_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 8"):
        IiTheGenesisOfGeometryTheSourceAndTheLogosDomainVConfig(expected_source_record_count=7)
    with pytest.raises(ValueError, match="expected_component_count must equal 3"):
        IiTheGenesisOfGeometryTheSourceAndTheLogosDomainVConfig(expected_component_count=4)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R04388"):
        IiTheGenesisOfGeometryTheSourceAndTheLogosDomainVConfig(next_source_boundary="P0R04387")
    with pytest.raises(
        ValueError,
        match="unknown ii_the_genesis_of_geometry_the_source_and_the_logos_domain_v component",
    ):
        classify_ii_the_genesis_of_geometry_the_source_and_the_logos_domain_v_component(
            "empirical_validation_claim"
        )
