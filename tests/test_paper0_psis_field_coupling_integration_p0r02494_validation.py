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

from scpn_quantum_control.paper0.psis_field_coupling_integration_p0r02494_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    PsisFieldCouplingIntegrationP0r02494Config,
    classify_psis_field_coupling_integration_p0r02494_component,
    psis_field_coupling_integration_p0r02494_labels,
    validate_psis_field_coupling_integration_p0r02494_fixture,
)


def test_psis_field_coupling_integration_p0r02494_fixture_preserves_source_boundary() -> None:
    result = validate_psis_field_coupling_integration_p0r02494_fixture()
    assert result.source_ledger_span == ("P0R02494", "P0R02501")
    assert result.source_record_count == 8
    assert result.component_count == 4
    assert result.next_source_boundary == "P0R02502"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_psis_field_coupling_integration_p0r02494_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R02494"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R02501"


def test_psis_field_coupling_integration_p0r02494_classification_and_labels_are_explicit() -> None:
    for component in (
        "psis_field_coupling_integration",
        "upde_provides_the_coherent_sigma",
        "quasicriticality_makes_sigma_receptive",
        "ms_qec_makes_sigma_robust",
    ):
        assert (
            classify_psis_field_coupling_integration_p0r02494_component(component)
            == f"{component}_source_boundary"
        )
    labels = psis_field_coupling_integration_p0r02494_labels()
    assert labels["section"] == "Psis Field Coupling Integration"
    assert labels["next_boundary"] == "P0R02502"


def test_psis_field_coupling_integration_p0r02494_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 8"):
        PsisFieldCouplingIntegrationP0r02494Config(expected_source_record_count=7)
    with pytest.raises(ValueError, match="expected_component_count must equal 4"):
        PsisFieldCouplingIntegrationP0r02494Config(expected_component_count=5)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R02502"):
        PsisFieldCouplingIntegrationP0r02494Config(next_source_boundary="P0R02501")
    with pytest.raises(
        ValueError, match="unknown psis_field_coupling_integration_p0r02494 component"
    ):
        classify_psis_field_coupling_integration_p0r02494_component("empirical_validation_claim")
