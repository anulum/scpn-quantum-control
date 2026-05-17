# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0  validation tests
"""Tests for Paper 0  source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.paper0_slice_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    Paper0SliceConfig,
    classify_paper0_slice_component,
    paper0_slice_labels,
    validate_paper0_slice_fixture,
)


def test_paper0_slice_fixture_preserves_source_boundary() -> None:
    result = validate_paper0_slice_fixture()
    assert result.source_ledger_span == ("P0R01959", "P0R01992")
    assert result.source_record_count == 34
    assert result.component_count == 3
    assert result.next_source_boundary == "P0R01993"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert result.problem_metadata["protocol_state"] == "source_paper0_slice_only_no_experiment"
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R01959"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R01992"


def test_paper0_slice_classification_and_labels_are_explicit() -> None:
    for component in (
        "source_component",
        "p0r01965",
        "2_6_8_formal_derivation_of_the_hierarchy_via_bulk_brane_overlap_integral",
    ):
        assert classify_paper0_slice_component(component) == f"{component}_source_boundary"
    labels = paper0_slice_labels()
    assert labels["section"] == ""
    assert labels["next_boundary"] == "P0R01993"


def test_paper0_slice_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 34"):
        Paper0SliceConfig(expected_source_record_count=33)
    with pytest.raises(ValueError, match="expected_component_count must equal 3"):
        Paper0SliceConfig(expected_component_count=4)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R01993"):
        Paper0SliceConfig(next_source_boundary="P0R01992")
    with pytest.raises(ValueError, match="unknown paper0_slice component"):
        classify_paper0_slice_component("empirical_validation_claim")
