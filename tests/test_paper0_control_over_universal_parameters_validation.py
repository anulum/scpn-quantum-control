# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Control over Universal Parameters: validation tests
"""Tests for Paper 0 Control over Universal Parameters: source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.control_over_universal_parameters_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    ControlOverUniversalParametersConfig,
    classify_control_over_universal_parameters_component,
    control_over_universal_parameters_labels,
    validate_control_over_universal_parameters_fixture,
)


def test_control_over_universal_parameters_fixture_preserves_source_boundary() -> None:
    result = validate_control_over_universal_parameters_fixture()
    assert result.source_ledger_span == ("P0R02448", "P0R02467")
    assert result.source_record_count == 20
    assert result.component_count == 2
    assert result.next_source_boundary == "P0R02468"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_control_over_universal_parameters_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R02448"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R02467"


def test_control_over_universal_parameters_classification_and_labels_are_explicit() -> None:
    for component in (
        "control_over_universal_parameters",
        "the_gdelian_oracle_as_a_consistency_check_on_coupling_laws",
    ):
        assert (
            classify_control_over_universal_parameters_component(component)
            == f"{component}_source_boundary"
        )
    labels = control_over_universal_parameters_labels()
    assert labels["section"] == "Control over Universal Parameters:"
    assert labels["next_boundary"] == "P0R02468"


def test_control_over_universal_parameters_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 20"):
        ControlOverUniversalParametersConfig(expected_source_record_count=19)
    with pytest.raises(ValueError, match="expected_component_count must equal 2"):
        ControlOverUniversalParametersConfig(expected_component_count=3)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R02468"):
        ControlOverUniversalParametersConfig(next_source_boundary="P0R02467")
    with pytest.raises(ValueError, match="unknown control_over_universal_parameters component"):
        classify_control_over_universal_parameters_component("empirical_validation_claim")
