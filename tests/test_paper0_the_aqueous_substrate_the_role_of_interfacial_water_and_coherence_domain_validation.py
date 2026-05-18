# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 The Aqueous Substrate - The Role of Interfacial Water and Coherence Domains validation tests
"""Tests for Paper 0 The Aqueous Substrate - The Role of Interfacial Water and Coherence Domains source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.the_aqueous_substrate_the_role_of_interfacial_water_and_coherence_domain_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    TheAqueousSubstrateTheRoleOfInterfacialWaterAndCoherenceDomainConfig,
    classify_the_aqueous_substrate_the_role_of_interfacial_water_and_coherence_domain_component,
    the_aqueous_substrate_the_role_of_interfacial_water_and_coherence_domain_labels,
    validate_the_aqueous_substrate_the_role_of_interfacial_water_and_coherence_domain_fixture,
)


def test_the_aqueous_substrate_the_role_of_interfacial_water_and_coherence_domain_fixture_preserves_source_boundary() -> (
    None
):
    result = (
        validate_the_aqueous_substrate_the_role_of_interfacial_water_and_coherence_domain_fixture()
    )
    assert result.source_ledger_span == ("P0R05293", "P0R05305")
    assert result.source_record_count == 13
    assert result.component_count == 2
    assert result.next_source_boundary == "P0R05306"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_the_aqueous_substrate_the_role_of_interfacial_water_and_coherence_domain_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R05293"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R05305"


def test_the_aqueous_substrate_the_role_of_interfacial_water_and_coherence_domain_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "the_aqueous_substrate_the_role_of_interfacial_water_and_coherence_domain",
        "p0r05299",
    ):
        assert (
            classify_the_aqueous_substrate_the_role_of_interfacial_water_and_coherence_domain_component(
                component
            )
            == f"{component}_source_boundary"
        )
    labels = the_aqueous_substrate_the_role_of_interfacial_water_and_coherence_domain_labels()
    assert (
        labels["section"]
        == "The Aqueous Substrate - The Role of Interfacial Water and Coherence Domains"
    )
    assert labels["next_boundary"] == "P0R05306"


def test_the_aqueous_substrate_the_role_of_interfacial_water_and_coherence_domain_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 13"):
        TheAqueousSubstrateTheRoleOfInterfacialWaterAndCoherenceDomainConfig(
            expected_source_record_count=12
        )
    with pytest.raises(ValueError, match="expected_component_count must equal 2"):
        TheAqueousSubstrateTheRoleOfInterfacialWaterAndCoherenceDomainConfig(
            expected_component_count=3
        )
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R05306"):
        TheAqueousSubstrateTheRoleOfInterfacialWaterAndCoherenceDomainConfig(
            next_source_boundary="P0R05305"
        )
    with pytest.raises(
        ValueError,
        match="unknown the_aqueous_substrate_the_role_of_interfacial_water_and_coherence_domain component",
    ):
        classify_the_aqueous_substrate_the_role_of_interfacial_water_and_coherence_domain_component(
            "empirical_validation_claim"
        )
