# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 The Aqueous Substrate (Domain I Interface) validation tests
"""Tests for Paper 0 The Aqueous Substrate (Domain I Interface) source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.the_aqueous_substrate_domain_i_interface_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    TheAqueousSubstrateDomainIInterfaceConfig,
    classify_the_aqueous_substrate_domain_i_interface_component,
    the_aqueous_substrate_domain_i_interface_labels,
    validate_the_aqueous_substrate_domain_i_interface_fixture,
)


def test_the_aqueous_substrate_domain_i_interface_fixture_preserves_source_boundary() -> None:
    result = validate_the_aqueous_substrate_domain_i_interface_fixture()
    assert result.source_ledger_span == ("P0R05331", "P0R05346")
    assert result.source_record_count == 16
    assert result.component_count == 4
    assert result.next_source_boundary == "P0R05347"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_the_aqueous_substrate_domain_i_interface_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R05331"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R05346"


def test_the_aqueous_substrate_domain_i_interface_classification_and_labels_are_explicit() -> None:
    for component in (
        "the_aqueous_substrate_domain_i_interface",
        "coherence_domains_cds_predicted_by_qed_interfacial_water_forms_cds_where",
        "integration_in_l1_cds_shield_microtubule_qubits_and_support_frhlich_cond",
        "the_genesis_of_life_abiogenesis_as_a_guided_phase_transition",
    ):
        assert (
            classify_the_aqueous_substrate_domain_i_interface_component(component)
            == f"{component}_source_boundary"
        )
    labels = the_aqueous_substrate_domain_i_interface_labels()
    assert labels["section"] == "The Aqueous Substrate (Domain I Interface)"
    assert labels["next_boundary"] == "P0R05347"


def test_the_aqueous_substrate_domain_i_interface_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 16"):
        TheAqueousSubstrateDomainIInterfaceConfig(expected_source_record_count=15)
    with pytest.raises(ValueError, match="expected_component_count must equal 4"):
        TheAqueousSubstrateDomainIInterfaceConfig(expected_component_count=5)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R05347"):
        TheAqueousSubstrateDomainIInterfaceConfig(next_source_boundary="P0R05346")
    with pytest.raises(
        ValueError, match="unknown the_aqueous_substrate_domain_i_interface component"
    ):
        classify_the_aqueous_substrate_domain_i_interface_component("empirical_validation_claim")
