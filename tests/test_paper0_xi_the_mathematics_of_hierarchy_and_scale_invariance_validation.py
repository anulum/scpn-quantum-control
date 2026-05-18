# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 XI. The Mathematics of Hierarchy and Scale Invariance validation tests
"""Tests for Paper 0 XI. The Mathematics of Hierarchy and Scale Invariance source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.xi_the_mathematics_of_hierarchy_and_scale_invariance_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    XiTheMathematicsOfHierarchyAndScaleInvarianceConfig,
    classify_xi_the_mathematics_of_hierarchy_and_scale_invariance_component,
    validate_xi_the_mathematics_of_hierarchy_and_scale_invariance_fixture,
    xi_the_mathematics_of_hierarchy_and_scale_invariance_labels,
)


def test_xi_the_mathematics_of_hierarchy_and_scale_invariance_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_xi_the_mathematics_of_hierarchy_and_scale_invariance_fixture()
    assert result.source_ledger_span == ("P0R06057", "P0R06065")
    assert result.source_record_count == 9
    assert result.component_count == 2
    assert result.next_source_boundary == "P0R06066"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_xi_the_mathematics_of_hierarchy_and_scale_invariance_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R06057"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R06065"


def test_xi_the_mathematics_of_hierarchy_and_scale_invariance_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "xi_the_mathematics_of_hierarchy_and_scale_invariance",
        "xii_the_principle_of_fractal_self_similarity_pfss",
    ):
        assert (
            classify_xi_the_mathematics_of_hierarchy_and_scale_invariance_component(component)
            == f"{component}_source_boundary"
        )
    labels = xi_the_mathematics_of_hierarchy_and_scale_invariance_labels()
    assert labels["section"] == "XI. The Mathematics of Hierarchy and Scale Invariance"
    assert labels["next_boundary"] == "P0R06066"


def test_xi_the_mathematics_of_hierarchy_and_scale_invariance_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 9"):
        XiTheMathematicsOfHierarchyAndScaleInvarianceConfig(expected_source_record_count=8)
    with pytest.raises(ValueError, match="expected_component_count must equal 2"):
        XiTheMathematicsOfHierarchyAndScaleInvarianceConfig(expected_component_count=3)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R06066"):
        XiTheMathematicsOfHierarchyAndScaleInvarianceConfig(next_source_boundary="P0R06065")
    with pytest.raises(
        ValueError, match="unknown xi_the_mathematics_of_hierarchy_and_scale_invariance component"
    ):
        classify_xi_the_mathematics_of_hierarchy_and_scale_invariance_component(
            "empirical_validation_claim"
        )
