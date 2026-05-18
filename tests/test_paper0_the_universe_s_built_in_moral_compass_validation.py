# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 The Universe's Built-in Moral Compass validation tests
"""Tests for Paper 0 The Universe's Built-in Moral Compass source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.the_universe_s_built_in_moral_compass_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    TheUniverseSBuiltInMoralCompassConfig,
    classify_the_universe_s_built_in_moral_compass_component,
    the_universe_s_built_in_moral_compass_labels,
    validate_the_universe_s_built_in_moral_compass_fixture,
)


def test_the_universe_s_built_in_moral_compass_fixture_preserves_source_boundary() -> None:
    result = validate_the_universe_s_built_in_moral_compass_fixture()
    assert result.source_ledger_span == ("P0R03715", "P0R03722")
    assert result.source_record_count == 8
    assert result.component_count == 3
    assert result.next_source_boundary == "P0R03723"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_the_universe_s_built_in_moral_compass_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R03715"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R03722"


def test_the_universe_s_built_in_moral_compass_classification_and_labels_are_explicit() -> None:
    for component in (
        "the_universe_s_built_in_moral_compass",
        "the_pull_of_the_future_how_purpose_guides_the_present",
        "formalisation_of_the_causal_entropic_principle",
    ):
        assert (
            classify_the_universe_s_built_in_moral_compass_component(component)
            == f"{component}_source_boundary"
        )
    labels = the_universe_s_built_in_moral_compass_labels()
    assert labels["section"] == "The Universe's Built-in Moral Compass"
    assert labels["next_boundary"] == "P0R03723"


def test_the_universe_s_built_in_moral_compass_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 8"):
        TheUniverseSBuiltInMoralCompassConfig(expected_source_record_count=7)
    with pytest.raises(ValueError, match="expected_component_count must equal 3"):
        TheUniverseSBuiltInMoralCompassConfig(expected_component_count=4)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R03723"):
        TheUniverseSBuiltInMoralCompassConfig(next_source_boundary="P0R03722")
    with pytest.raises(
        ValueError, match="unknown the_universe_s_built_in_moral_compass component"
    ):
        classify_the_universe_s_built_in_moral_compass_component("empirical_validation_claim")
