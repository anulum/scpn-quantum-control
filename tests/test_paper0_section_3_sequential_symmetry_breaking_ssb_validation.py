# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 3. Sequential Symmetry Breaking (SSB): validation tests
"""Tests for Paper 0 3. Sequential Symmetry Breaking (SSB): source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.section_3_sequential_symmetry_breaking_ssb_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    Section3SequentialSymmetryBreakingSsbConfig,
    classify_section_3_sequential_symmetry_breaking_ssb_component,
    section_3_sequential_symmetry_breaking_ssb_labels,
    validate_section_3_sequential_symmetry_breaking_ssb_fixture,
)


def test_section_3_sequential_symmetry_breaking_ssb_fixture_preserves_source_boundary() -> None:
    result = validate_section_3_sequential_symmetry_breaking_ssb_fixture()
    assert result.source_ledger_span == ("P0R04388", "P0R04395")
    assert result.source_record_count == 8
    assert result.component_count == 4
    assert result.next_source_boundary == "P0R04396"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_section_3_sequential_symmetry_breaking_ssb_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R04388"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R04395"


def test_section_3_sequential_symmetry_breaking_ssb_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "3_sequential_symmetry_breaking_ssb",
        "iii_the_geometry_of_the_quantum_biological_interface_domain_i_l1_l3",
        "1_quantum_geometry_and_topological_order_l1",
        "2_molecular_geometry_chirality_and_iet_l2_l3",
    ):
        assert (
            classify_section_3_sequential_symmetry_breaking_ssb_component(component)
            == f"{component}_source_boundary"
        )
    labels = section_3_sequential_symmetry_breaking_ssb_labels()
    assert labels["section"] == "3. Sequential Symmetry Breaking (SSB):"
    assert labels["next_boundary"] == "P0R04396"


def test_section_3_sequential_symmetry_breaking_ssb_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 8"):
        Section3SequentialSymmetryBreakingSsbConfig(expected_source_record_count=7)
    with pytest.raises(ValueError, match="expected_component_count must equal 4"):
        Section3SequentialSymmetryBreakingSsbConfig(expected_component_count=5)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R04396"):
        Section3SequentialSymmetryBreakingSsbConfig(next_source_boundary="P0R04395")
    with pytest.raises(
        ValueError, match="unknown section_3_sequential_symmetry_breaking_ssb component"
    ):
        classify_section_3_sequential_symmetry_breaking_ssb_component("empirical_validation_claim")
