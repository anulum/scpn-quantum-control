# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Schizophrenia (Dissonance) validation tests
"""Tests for Paper 0 Schizophrenia (Dissonance) source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.schizophrenia_dissonance_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    SchizophreniaDissonanceConfig,
    classify_schizophrenia_dissonance_component,
    schizophrenia_dissonance_labels,
    validate_schizophrenia_dissonance_fixture,
)


def test_schizophrenia_dissonance_fixture_preserves_source_boundary() -> None:
    result = validate_schizophrenia_dissonance_fixture()
    assert result.source_ledger_span == ("P0R04630", "P0R04639")
    assert result.source_record_count == 10
    assert result.component_count == 3
    assert result.next_source_boundary == "P0R04640"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_schizophrenia_dissonance_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R04630"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R04639"


def test_schizophrenia_dissonance_classification_and_labels_are_explicit() -> None:
    for component in (
        "schizophrenia_dissonance",
        "depression_dyscritia_dissonance",
        "alzheimer_s_disease_decoherence_cascade",
    ):
        assert (
            classify_schizophrenia_dissonance_component(component)
            == f"{component}_source_boundary"
        )
    labels = schizophrenia_dissonance_labels()
    assert labels["section"] == "Schizophrenia (Dissonance)"
    assert labels["next_boundary"] == "P0R04640"


def test_schizophrenia_dissonance_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 10"):
        SchizophreniaDissonanceConfig(expected_source_record_count=9)
    with pytest.raises(ValueError, match="expected_component_count must equal 3"):
        SchizophreniaDissonanceConfig(expected_component_count=4)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R04640"):
        SchizophreniaDissonanceConfig(next_source_boundary="P0R04639")
    with pytest.raises(ValueError, match="unknown schizophrenia_dissonance component"):
        classify_schizophrenia_dissonance_component("empirical_validation_claim")
