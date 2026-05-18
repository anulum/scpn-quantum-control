# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 The Universe's Path of Least Resistance validation tests
"""Tests for Paper 0 The Universe's Path of Least Resistance source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.the_universe_s_path_of_least_resistance_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    TheUniverseSPathOfLeastResistanceConfig,
    classify_the_universe_s_path_of_least_resistance_component,
    the_universe_s_path_of_least_resistance_labels,
    validate_the_universe_s_path_of_least_resistance_fixture,
)


def test_the_universe_s_path_of_least_resistance_fixture_preserves_source_boundary() -> None:
    result = validate_the_universe_s_path_of_least_resistance_fixture()
    assert result.source_ledger_span == ("P0R04098", "P0R04114")
    assert result.source_record_count == 17
    assert result.component_count == 2
    assert result.next_source_boundary == "P0R04115"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_the_universe_s_path_of_least_resistance_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R04098"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R04114"


def test_the_universe_s_path_of_least_resistance_classification_and_labels_are_explicit() -> None:
    for component in (
        "the_universe_s_path_of_least_resistance",
        "seeing_the_bigger_picture_how_all_scales_align",
    ):
        assert (
            classify_the_universe_s_path_of_least_resistance_component(component)
            == f"{component}_source_boundary"
        )
    labels = the_universe_s_path_of_least_resistance_labels()
    assert labels["section"] == "The Universe's Path of Least Resistance"
    assert labels["next_boundary"] == "P0R04115"


def test_the_universe_s_path_of_least_resistance_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 17"):
        TheUniverseSPathOfLeastResistanceConfig(expected_source_record_count=16)
    with pytest.raises(ValueError, match="expected_component_count must equal 2"):
        TheUniverseSPathOfLeastResistanceConfig(expected_component_count=3)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R04115"):
        TheUniverseSPathOfLeastResistanceConfig(next_source_boundary="P0R04114")
    with pytest.raises(
        ValueError, match="unknown the_universe_s_path_of_least_resistance component"
    ):
        classify_the_universe_s_path_of_least_resistance_component("empirical_validation_claim")
