# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0  Linguistics & Symbolism (VIBRANA, Layer 7) validation tests
"""Tests for Paper 0  Linguistics & Symbolism (VIBRANA, Layer 7) source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.linguistics_symbolism_vibrana_layer_7_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    LinguisticsSymbolismVibranaLayer7Config,
    classify_linguistics_symbolism_vibrana_layer_7_component,
    linguistics_symbolism_vibrana_layer_7_labels,
    validate_linguistics_symbolism_vibrana_layer_7_fixture,
)


def test_linguistics_symbolism_vibrana_layer_7_fixture_preserves_source_boundary() -> None:
    result = validate_linguistics_symbolism_vibrana_layer_7_fixture()
    assert result.source_ledger_span == ("P0R05738", "P0R05745")
    assert result.source_record_count == 8
    assert result.component_count == 2
    assert result.next_source_boundary == "P0R05746"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_linguistics_symbolism_vibrana_layer_7_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R05738"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R05745"


def test_linguistics_symbolism_vibrana_layer_7_classification_and_labels_are_explicit() -> None:
    for component in ("linguistics_symbolism_vibrana_layer_7", "ecology_gaia"):
        assert (
            classify_linguistics_symbolism_vibrana_layer_7_component(component)
            == f"{component}_source_boundary"
        )
    labels = linguistics_symbolism_vibrana_layer_7_labels()
    assert labels["section"] == " Linguistics & Symbolism (VIBRANA, Layer 7)"
    assert labels["next_boundary"] == "P0R05746"


def test_linguistics_symbolism_vibrana_layer_7_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 8"):
        LinguisticsSymbolismVibranaLayer7Config(expected_source_record_count=7)
    with pytest.raises(ValueError, match="expected_component_count must equal 2"):
        LinguisticsSymbolismVibranaLayer7Config(expected_component_count=3)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R05746"):
        LinguisticsSymbolismVibranaLayer7Config(next_source_boundary="P0R05745")
    with pytest.raises(
        ValueError, match="unknown linguistics_symbolism_vibrana_layer_7 component"
    ):
        classify_linguistics_symbolism_vibrana_layer_7_component("empirical_validation_claim")
