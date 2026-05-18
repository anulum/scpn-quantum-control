# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 2. The Endocrine System and HPA Axis (Stress Response) validation tests
"""Tests for Paper 0 2. The Endocrine System and HPA Axis (Stress Response) source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.section_2_the_endocrine_system_and_hpa_axis_stress_response_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    Section2TheEndocrineSystemAndHpaAxisStressResponseConfig,
    classify_section_2_the_endocrine_system_and_hpa_axis_stress_response_component,
    section_2_the_endocrine_system_and_hpa_axis_stress_response_labels,
    validate_section_2_the_endocrine_system_and_hpa_axis_stress_response_fixture,
)


def test_section_2_the_endocrine_system_and_hpa_axis_stress_response_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_section_2_the_endocrine_system_and_hpa_axis_stress_response_fixture()
    assert result.source_ledger_span == ("P0R04935", "P0R04942")
    assert result.source_record_count == 8
    assert result.component_count == 1
    assert result.next_source_boundary == "P0R04943"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_section_2_the_endocrine_system_and_hpa_axis_stress_response_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R04935"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R04942"


def test_section_2_the_endocrine_system_and_hpa_axis_stress_response_classification_and_labels_are_explicit() -> (
    None
):
    for component in ("2_the_endocrine_system_and_hpa_axis_stress_response",):
        assert (
            classify_section_2_the_endocrine_system_and_hpa_axis_stress_response_component(
                component
            )
            == f"{component}_source_boundary"
        )
    labels = section_2_the_endocrine_system_and_hpa_axis_stress_response_labels()
    assert labels["section"] == "2. The Endocrine System and HPA Axis (Stress Response)"
    assert labels["next_boundary"] == "P0R04943"


def test_section_2_the_endocrine_system_and_hpa_axis_stress_response_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 8"):
        Section2TheEndocrineSystemAndHpaAxisStressResponseConfig(expected_source_record_count=7)
    with pytest.raises(ValueError, match="expected_component_count must equal 1"):
        Section2TheEndocrineSystemAndHpaAxisStressResponseConfig(expected_component_count=2)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R04943"):
        Section2TheEndocrineSystemAndHpaAxisStressResponseConfig(next_source_boundary="P0R04942")
    with pytest.raises(
        ValueError,
        match="unknown section_2_the_endocrine_system_and_hpa_axis_stress_response component",
    ):
        classify_section_2_the_endocrine_system_and_hpa_axis_stress_response_component(
            "empirical_validation_claim"
        )
