# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 The Hierarchical Impedance Rescaling validation tests
"""Tests for Paper 0 The Hierarchical Impedance Rescaling source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.the_hierarchical_impedance_rescaling_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    TheHierarchicalImpedanceRescalingConfig,
    classify_the_hierarchical_impedance_rescaling_component,
    the_hierarchical_impedance_rescaling_labels,
    validate_the_hierarchical_impedance_rescaling_fixture,
)


def test_the_hierarchical_impedance_rescaling_fixture_preserves_source_boundary() -> None:
    result = validate_the_hierarchical_impedance_rescaling_fixture()
    assert result.source_ledger_span == ("P0R02655", "P0R02681")
    assert result.source_record_count == 27
    assert result.component_count == 3
    assert result.next_source_boundary == "P0R02682"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_the_hierarchical_impedance_rescaling_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R02655"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R02681"


def test_the_hierarchical_impedance_rescaling_classification_and_labels_are_explicit() -> None:
    for component in (
        "the_hierarchical_impedance_rescaling",
        "p0r02661",
        "formal_integration_of_the_enhanced_boundary_set_ebs_into_the_upde",
    ):
        assert (
            classify_the_hierarchical_impedance_rescaling_component(component)
            == f"{component}_source_boundary"
        )
    labels = the_hierarchical_impedance_rescaling_labels()
    assert labels["section"] == "The Hierarchical Impedance Rescaling"
    assert labels["next_boundary"] == "P0R02682"


def test_the_hierarchical_impedance_rescaling_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 27"):
        TheHierarchicalImpedanceRescalingConfig(expected_source_record_count=26)
    with pytest.raises(ValueError, match="expected_component_count must equal 3"):
        TheHierarchicalImpedanceRescalingConfig(expected_component_count=4)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R02682"):
        TheHierarchicalImpedanceRescalingConfig(next_source_boundary="P0R02681")
    with pytest.raises(ValueError, match="unknown the_hierarchical_impedance_rescaling component"):
        classify_the_hierarchical_impedance_rescaling_component("empirical_validation_claim")
