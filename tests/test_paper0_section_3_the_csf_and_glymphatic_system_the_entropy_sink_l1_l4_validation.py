# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 3. The CSF and Glymphatic System: The Entropy Sink (L1-L4) validation tests
"""Tests for Paper 0 3. The CSF and Glymphatic System: The Entropy Sink (L1-L4) source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.section_3_the_csf_and_glymphatic_system_the_entropy_sink_l1_l4_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    Section3TheCsfAndGlymphaticSystemTheEntropySinkL1L4Config,
    classify_section_3_the_csf_and_glymphatic_system_the_entropy_sink_l1_l4_component,
    section_3_the_csf_and_glymphatic_system_the_entropy_sink_l1_l4_labels,
    validate_section_3_the_csf_and_glymphatic_system_the_entropy_sink_l1_l4_fixture,
)


def test_section_3_the_csf_and_glymphatic_system_the_entropy_sink_l1_l4_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_section_3_the_csf_and_glymphatic_system_the_entropy_sink_l1_l4_fixture()
    assert result.source_ledger_span == ("P0R04871", "P0R04878")
    assert result.source_record_count == 8
    assert result.component_count == 2
    assert result.next_source_boundary == "P0R04879"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_section_3_the_csf_and_glymphatic_system_the_entropy_sink_l1_l4_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R04871"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R04878"


def test_section_3_the_csf_and_glymphatic_system_the_entropy_sink_l1_l4_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "3_the_csf_and_glymphatic_system_the_entropy_sink_l1_l4",
        "ii_neuro_vascular_coupling_and_hemodynamics_the_energetics_of_consciousn",
    ):
        assert (
            classify_section_3_the_csf_and_glymphatic_system_the_entropy_sink_l1_l4_component(
                component
            )
            == f"{component}_source_boundary"
        )
    labels = section_3_the_csf_and_glymphatic_system_the_entropy_sink_l1_l4_labels()
    assert labels["section"] == "3. The CSF and Glymphatic System: The Entropy Sink (L1-L4)"
    assert labels["next_boundary"] == "P0R04879"


def test_section_3_the_csf_and_glymphatic_system_the_entropy_sink_l1_l4_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 8"):
        Section3TheCsfAndGlymphaticSystemTheEntropySinkL1L4Config(expected_source_record_count=7)
    with pytest.raises(ValueError, match="expected_component_count must equal 2"):
        Section3TheCsfAndGlymphaticSystemTheEntropySinkL1L4Config(expected_component_count=3)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R04879"):
        Section3TheCsfAndGlymphaticSystemTheEntropySinkL1L4Config(next_source_boundary="P0R04878")
    with pytest.raises(
        ValueError,
        match="unknown section_3_the_csf_and_glymphatic_system_the_entropy_sink_l1_l4 component",
    ):
        classify_section_3_the_csf_and_glymphatic_system_the_entropy_sink_l1_l4_component(
            "empirical_validation_claim"
        )
