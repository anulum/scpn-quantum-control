# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0  as a Measure of Causal Efficacy: validation tests
"""Tests for Paper 0  as a Measure of Causal Efficacy: source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.as_a_measure_of_causal_efficacy_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    AsAMeasureOfCausalEfficacyConfig,
    as_a_measure_of_causal_efficacy_labels,
    classify_as_a_measure_of_causal_efficacy_component,
    validate_as_a_measure_of_causal_efficacy_fixture,
)


def test_as_a_measure_of_causal_efficacy_fixture_preserves_source_boundary() -> None:
    result = validate_as_a_measure_of_causal_efficacy_fixture()
    assert result.source_ledger_span == ("P0R03295", "P0R03306")
    assert result.source_record_count == 12
    assert result.component_count == 3
    assert result.next_source_boundary == "P0R03307"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_as_a_measure_of_causal_efficacy_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R03295"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R03306"


def test_as_a_measure_of_causal_efficacy_classification_and_labels_are_explicit() -> None:
    for component in (
        "as_a_measure_of_causal_efficacy",
        "the_quantum_gravity_interface_and_cigd",
        "consciousness_induced_gravitational_decoherence_cigd",
    ):
        assert (
            classify_as_a_measure_of_causal_efficacy_component(component)
            == f"{component}_source_boundary"
        )
    labels = as_a_measure_of_causal_efficacy_labels()
    assert labels["section"] == " as a Measure of Causal Efficacy:"
    assert labels["next_boundary"] == "P0R03307"


def test_as_a_measure_of_causal_efficacy_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 12"):
        AsAMeasureOfCausalEfficacyConfig(expected_source_record_count=11)
    with pytest.raises(ValueError, match="expected_component_count must equal 3"):
        AsAMeasureOfCausalEfficacyConfig(expected_component_count=4)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R03307"):
        AsAMeasureOfCausalEfficacyConfig(next_source_boundary="P0R03306")
    with pytest.raises(ValueError, match="unknown as_a_measure_of_causal_efficacy component"):
        classify_as_a_measure_of_causal_efficacy_component("empirical_validation_claim")
