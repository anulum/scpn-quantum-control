# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 15-Layer Summary Table validation tests
"""Tests for Paper 0 15-Layer Summary Table source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.section_15_layer_summary_table_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    Section15LayerSummaryTableConfig,
    classify_section_15_layer_summary_table_component,
    section_15_layer_summary_table_labels,
    validate_section_15_layer_summary_table_fixture,
)


def test_section_15_layer_summary_table_fixture_preserves_source_boundary() -> None:
    result = validate_section_15_layer_summary_table_fixture()
    assert result.source_ledger_span == ("P0R02061", "P0R02087")
    assert result.source_record_count == 27
    assert result.component_count == 1
    assert result.next_source_boundary == "P0R02088"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_section_15_layer_summary_table_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R02061"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R02087"


def test_section_15_layer_summary_table_classification_and_labels_are_explicit() -> None:
    for component in ("15_layer_summary_table",):
        assert (
            classify_section_15_layer_summary_table_component(component)
            == f"{component}_source_boundary"
        )
    labels = section_15_layer_summary_table_labels()
    assert labels["section"] == "15-Layer Summary Table"
    assert labels["next_boundary"] == "P0R02088"


def test_section_15_layer_summary_table_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 27"):
        Section15LayerSummaryTableConfig(expected_source_record_count=26)
    with pytest.raises(ValueError, match="expected_component_count must equal 1"):
        Section15LayerSummaryTableConfig(expected_component_count=2)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R02088"):
        Section15LayerSummaryTableConfig(next_source_boundary="P0R02087")
    with pytest.raises(ValueError, match="unknown section_15_layer_summary_table component"):
        classify_section_15_layer_summary_table_component("empirical_validation_claim")
