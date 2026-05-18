# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 One Spine, Many Couplings - UPDE Scope Constraint validation tests
"""Tests for Paper 0 One Spine, Many Couplings - UPDE Scope Constraint source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.one_spine_many_couplings_upde_scope_constraint_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    OneSpineManyCouplingsUpdeScopeConstraintConfig,
    classify_one_spine_many_couplings_upde_scope_constraint_component,
    one_spine_many_couplings_upde_scope_constraint_labels,
    validate_one_spine_many_couplings_upde_scope_constraint_fixture,
)


def test_one_spine_many_couplings_upde_scope_constraint_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_one_spine_many_couplings_upde_scope_constraint_fixture()
    assert result.source_ledger_span == ("P0R02682", "P0R02745")
    assert result.source_record_count == 64
    assert result.component_count == 1
    assert result.next_source_boundary == "P0R02746"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_one_spine_many_couplings_upde_scope_constraint_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R02682"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R02745"


def test_one_spine_many_couplings_upde_scope_constraint_classification_and_labels_are_explicit() -> (
    None
):
    for component in ("one_spine_many_couplings_upde_scope_constraint",):
        assert (
            classify_one_spine_many_couplings_upde_scope_constraint_component(component)
            == f"{component}_source_boundary"
        )
    labels = one_spine_many_couplings_upde_scope_constraint_labels()
    assert labels["section"] == "One Spine, Many Couplings - UPDE Scope Constraint"
    assert labels["next_boundary"] == "P0R02746"


def test_one_spine_many_couplings_upde_scope_constraint_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 64"):
        OneSpineManyCouplingsUpdeScopeConstraintConfig(expected_source_record_count=63)
    with pytest.raises(ValueError, match="expected_component_count must equal 1"):
        OneSpineManyCouplingsUpdeScopeConstraintConfig(expected_component_count=2)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R02746"):
        OneSpineManyCouplingsUpdeScopeConstraintConfig(next_source_boundary="P0R02745")
    with pytest.raises(
        ValueError, match="unknown one_spine_many_couplings_upde_scope_constraint component"
    ):
        classify_one_spine_many_couplings_upde_scope_constraint_component(
            "empirical_validation_claim"
        )
