# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Information-Geometric Lift of UPDE validation tests
"""Tests for Paper 0 Information-Geometric Lift of UPDE source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.information_geometric_lift_of_upde_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    InformationGeometricLiftOfUpdeConfig,
    classify_information_geometric_lift_of_upde_component,
    information_geometric_lift_of_upde_labels,
    validate_information_geometric_lift_of_upde_fixture,
)


def test_information_geometric_lift_of_upde_fixture_preserves_source_boundary() -> None:
    result = validate_information_geometric_lift_of_upde_fixture()
    assert result.source_ledger_span == ("P0R02640", "P0R02654")
    assert result.source_record_count == 15
    assert result.component_count == 1
    assert result.next_source_boundary == "P0R02655"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_information_geometric_lift_of_upde_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R02640"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R02654"


def test_information_geometric_lift_of_upde_classification_and_labels_are_explicit() -> None:
    for component in ("information_geometric_lift_of_upde",):
        assert (
            classify_information_geometric_lift_of_upde_component(component)
            == f"{component}_source_boundary"
        )
    labels = information_geometric_lift_of_upde_labels()
    assert labels["section"] == "Information-Geometric Lift of UPDE"
    assert labels["next_boundary"] == "P0R02655"


def test_information_geometric_lift_of_upde_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 15"):
        InformationGeometricLiftOfUpdeConfig(expected_source_record_count=14)
    with pytest.raises(ValueError, match="expected_component_count must equal 1"):
        InformationGeometricLiftOfUpdeConfig(expected_component_count=2)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R02655"):
        InformationGeometricLiftOfUpdeConfig(next_source_boundary="P0R02654")
    with pytest.raises(ValueError, match="unknown information_geometric_lift_of_upde component"):
        classify_information_geometric_lift_of_upde_component("empirical_validation_claim")
