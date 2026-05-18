# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Field Coupling (CField): validation tests
"""Tests for Paper 0 Field Coupling (CField): source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.field_coupling_cfield_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    FieldCouplingCfieldConfig,
    classify_field_coupling_cfield_component,
    field_coupling_cfield_labels,
    validate_field_coupling_cfield_fixture,
)


def test_field_coupling_cfield_fixture_preserves_source_boundary() -> None:
    result = validate_field_coupling_cfield_fixture()
    assert result.source_ledger_span == ("P0R02632", "P0R02639")
    assert result.source_record_count == 8
    assert result.component_count == 1
    assert result.next_source_boundary == "P0R02640"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_field_coupling_cfield_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R02632"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R02639"


def test_field_coupling_cfield_classification_and_labels_are_explicit() -> None:
    for component in ("field_coupling_cfield",):
        assert (
            classify_field_coupling_cfield_component(component) == f"{component}_source_boundary"
        )
    labels = field_coupling_cfield_labels()
    assert labels["section"] == "Field Coupling (CField):"
    assert labels["next_boundary"] == "P0R02640"


def test_field_coupling_cfield_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 8"):
        FieldCouplingCfieldConfig(expected_source_record_count=7)
    with pytest.raises(ValueError, match="expected_component_count must equal 1"):
        FieldCouplingCfieldConfig(expected_component_count=2)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R02640"):
        FieldCouplingCfieldConfig(next_source_boundary="P0R02639")
    with pytest.raises(ValueError, match="unknown field_coupling_cfield component"):
        classify_field_coupling_cfield_component("empirical_validation_claim")
