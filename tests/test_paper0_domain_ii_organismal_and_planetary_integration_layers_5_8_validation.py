# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Domain II: Organismal and Planetary Integration (Layers 5-8) validation tests
"""Tests for Paper 0 Domain II: Organismal and Planetary Integration (Layers 5-8) source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.domain_ii_organismal_and_planetary_integration_layers_5_8_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    DomainIiOrganismalAndPlanetaryIntegrationLayers58Config,
    classify_domain_ii_organismal_and_planetary_integration_layers_5_8_component,
    domain_ii_organismal_and_planetary_integration_layers_5_8_labels,
    validate_domain_ii_organismal_and_planetary_integration_layers_5_8_fixture,
)


def test_domain_ii_organismal_and_planetary_integration_layers_5_8_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_domain_ii_organismal_and_planetary_integration_layers_5_8_fixture()
    assert result.source_ledger_span == ("P0R05537", "P0R05550")
    assert result.source_record_count == 14
    assert result.component_count == 2
    assert result.next_source_boundary == "P0R05551"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_domain_ii_organismal_and_planetary_integration_layers_5_8_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R05537"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R05550"


def test_domain_ii_organismal_and_planetary_integration_layers_5_8_classification_and_labels_are_explicit() -> (
    None
):
    for component in ("domain_ii_organismal_and_planetary_integration_layers_5_8", "citations"):
        assert (
            classify_domain_ii_organismal_and_planetary_integration_layers_5_8_component(component)
            == f"{component}_source_boundary"
        )
    labels = domain_ii_organismal_and_planetary_integration_layers_5_8_labels()
    assert labels["section"] == "Domain II: Organismal and Planetary Integration (Layers 5-8)"
    assert labels["next_boundary"] == "P0R05551"


def test_domain_ii_organismal_and_planetary_integration_layers_5_8_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 14"):
        DomainIiOrganismalAndPlanetaryIntegrationLayers58Config(expected_source_record_count=13)
    with pytest.raises(ValueError, match="expected_component_count must equal 2"):
        DomainIiOrganismalAndPlanetaryIntegrationLayers58Config(expected_component_count=3)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R05551"):
        DomainIiOrganismalAndPlanetaryIntegrationLayers58Config(next_source_boundary="P0R05550")
    with pytest.raises(
        ValueError,
        match="unknown domain_ii_organismal_and_planetary_integration_layers_5_8 component",
    ):
        classify_domain_ii_organismal_and_planetary_integration_layers_5_8_component(
            "empirical_validation_claim"
        )
