# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 1. The Generalised Second Law (GSL): validation tests
"""Tests for Paper 0 1. The Generalised Second Law (GSL): source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.section_1_the_generalised_second_law_gsl_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    Section1TheGeneralisedSecondLawGslConfig,
    classify_section_1_the_generalised_second_law_gsl_component,
    section_1_the_generalised_second_law_gsl_labels,
    validate_section_1_the_generalised_second_law_gsl_fixture,
)


def test_section_1_the_generalised_second_law_gsl_fixture_preserves_source_boundary() -> None:
    result = validate_section_1_the_generalised_second_law_gsl_fixture()
    assert result.source_ledger_span == ("P0R05944", "P0R05952")
    assert result.source_record_count == 9
    assert result.component_count == 3
    assert result.next_source_boundary == "P0R05953"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_section_1_the_generalised_second_law_gsl_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R05944"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R05952"


def test_section_1_the_generalised_second_law_gsl_classification_and_labels_are_explicit() -> None:
    for component in (
        "1_the_generalised_second_law_gsl",
        "2_the_psi_field_as_a_negentropy_source_information_thermodynamics",
        "the_rate_of_negentropy_injection_npsi_is_proportional_to_the_mutual_info",
    ):
        assert (
            classify_section_1_the_generalised_second_law_gsl_component(component)
            == f"{component}_source_boundary"
        )
    labels = section_1_the_generalised_second_law_gsl_labels()
    assert labels["section"] == "1. The Generalised Second Law (GSL):"
    assert labels["next_boundary"] == "P0R05953"


def test_section_1_the_generalised_second_law_gsl_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 9"):
        Section1TheGeneralisedSecondLawGslConfig(expected_source_record_count=8)
    with pytest.raises(ValueError, match="expected_component_count must equal 3"):
        Section1TheGeneralisedSecondLawGslConfig(expected_component_count=4)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R05953"):
        Section1TheGeneralisedSecondLawGslConfig(next_source_boundary="P0R05952")
    with pytest.raises(
        ValueError, match="unknown section_1_the_generalised_second_law_gsl component"
    ):
        classify_section_1_the_generalised_second_law_gsl_component("empirical_validation_claim")
