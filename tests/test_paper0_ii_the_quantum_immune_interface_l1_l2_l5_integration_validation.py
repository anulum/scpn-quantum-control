# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 II. The Quantum-Immune Interface (L1/L2/L5 Integration) validation tests
"""Tests for Paper 0 II. The Quantum-Immune Interface (L1/L2/L5 Integration) source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.ii_the_quantum_immune_interface_l1_l2_l5_integration_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    IiTheQuantumImmuneInterfaceL1L2L5IntegrationConfig,
    classify_ii_the_quantum_immune_interface_l1_l2_l5_integration_component,
    ii_the_quantum_immune_interface_l1_l2_l5_integration_labels,
    validate_ii_the_quantum_immune_interface_l1_l2_l5_integration_fixture,
)


def test_ii_the_quantum_immune_interface_l1_l2_l5_integration_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_ii_the_quantum_immune_interface_l1_l2_l5_integration_fixture()
    assert result.source_ledger_span == ("P0R05358", "P0R05365")
    assert result.source_record_count == 8
    assert result.component_count == 1
    assert result.next_source_boundary == "P0R05366"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_ii_the_quantum_immune_interface_l1_l2_l5_integration_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R05358"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R05365"


def test_ii_the_quantum_immune_interface_l1_l2_l5_integration_classification_and_labels_are_explicit() -> (
    None
):
    for component in ("ii_the_quantum_immune_interface_l1_l2_l5_integration",):
        assert (
            classify_ii_the_quantum_immune_interface_l1_l2_l5_integration_component(component)
            == f"{component}_source_boundary"
        )
    labels = ii_the_quantum_immune_interface_l1_l2_l5_integration_labels()
    assert labels["section"] == "II. The Quantum-Immune Interface (L1/L2/L5 Integration)"
    assert labels["next_boundary"] == "P0R05366"


def test_ii_the_quantum_immune_interface_l1_l2_l5_integration_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 8"):
        IiTheQuantumImmuneInterfaceL1L2L5IntegrationConfig(expected_source_record_count=7)
    with pytest.raises(ValueError, match="expected_component_count must equal 1"):
        IiTheQuantumImmuneInterfaceL1L2L5IntegrationConfig(expected_component_count=2)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R05366"):
        IiTheQuantumImmuneInterfaceL1L2L5IntegrationConfig(next_source_boundary="P0R05365")
    with pytest.raises(
        ValueError, match="unknown ii_the_quantum_immune_interface_l1_l2_l5_integration component"
    ):
        classify_ii_the_quantum_immune_interface_l1_l2_l5_integration_component(
            "empirical_validation_claim"
        )
