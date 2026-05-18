# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 3. The Quasicritical Brain: validation tests
"""Tests for Paper 0 3. The Quasicritical Brain: source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.section_3_the_quasicritical_brain_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    Section3TheQuasicriticalBrainConfig,
    classify_section_3_the_quasicritical_brain_component,
    section_3_the_quasicritical_brain_labels,
    validate_section_3_the_quasicritical_brain_fixture,
)


def test_section_3_the_quasicritical_brain_fixture_preserves_source_boundary() -> None:
    result = validate_section_3_the_quasicritical_brain_fixture()
    assert result.source_ledger_span == ("P0R04499", "P0R04506")
    assert result.source_record_count == 8
    assert result.component_count == 3
    assert result.next_source_boundary == "P0R04507"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_section_3_the_quasicritical_brain_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R04499"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R04506"


def test_section_3_the_quasicritical_brain_classification_and_labels_are_explicit() -> None:
    for component in (
        "3_the_quasicritical_brain",
        "4_the_role_of_glia_astrocytes",
        "5_the_cerebellum_the_timing_and_prediction_engine_l4_l5_interface",
    ):
        assert (
            classify_section_3_the_quasicritical_brain_component(component)
            == f"{component}_source_boundary"
        )
    labels = section_3_the_quasicritical_brain_labels()
    assert labels["section"] == "3. The Quasicritical Brain:"
    assert labels["next_boundary"] == "P0R04507"


def test_section_3_the_quasicritical_brain_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 8"):
        Section3TheQuasicriticalBrainConfig(expected_source_record_count=7)
    with pytest.raises(ValueError, match="expected_component_count must equal 3"):
        Section3TheQuasicriticalBrainConfig(expected_component_count=4)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R04507"):
        Section3TheQuasicriticalBrainConfig(next_source_boundary="P0R04506")
    with pytest.raises(ValueError, match="unknown section_3_the_quasicritical_brain component"):
        classify_section_3_the_quasicritical_brain_component("empirical_validation_claim")
