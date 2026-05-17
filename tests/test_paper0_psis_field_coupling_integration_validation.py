# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Psis Field Coupling Integration validation tests
"""Tests for Paper 0 Psis Field Coupling Integration source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.psis_field_coupling_integration_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    PsisFieldCouplingIntegrationConfig,
    classify_psis_field_coupling_integration_component,
    psis_field_coupling_integration_labels,
    validate_psis_field_coupling_integration_fixture,
)


def test_psis_field_coupling_integration_fixture_preserves_source_boundary() -> None:
    result = validate_psis_field_coupling_integration_fixture()
    assert result.source_ledger_span == ("P0R02315", "P0R02366")
    assert result.source_record_count == 52
    assert result.component_count == 3
    assert result.next_source_boundary == "P0R02367"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_psis_field_coupling_integration_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R02315"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R02366"


def test_psis_field_coupling_integration_classification_and_labels_are_explicit() -> None:
    for component in (
        "psis_field_coupling_integration",
        "the_collective_state_variable_sigma",
        "the_coupling_mechanism",
    ):
        assert (
            classify_psis_field_coupling_integration_component(component)
            == f"{component}_source_boundary"
        )
    labels = psis_field_coupling_integration_labels()
    assert labels["section"] == "Psis Field Coupling Integration"
    assert labels["next_boundary"] == "P0R02367"


def test_psis_field_coupling_integration_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 52"):
        PsisFieldCouplingIntegrationConfig(expected_source_record_count=51)
    with pytest.raises(ValueError, match="expected_component_count must equal 3"):
        PsisFieldCouplingIntegrationConfig(expected_component_count=4)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R02367"):
        PsisFieldCouplingIntegrationConfig(next_source_boundary="P0R02366")
    with pytest.raises(ValueError, match="unknown psis_field_coupling_integration component"):
        classify_psis_field_coupling_integration_component("empirical_validation_claim")
