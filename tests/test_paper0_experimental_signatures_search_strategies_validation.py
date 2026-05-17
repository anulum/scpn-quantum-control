# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 experimental signatures validation tests
"""Tests for Paper 0 experimental-signatures search-strategy validation."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.experimental_signatures_search_strategies_validation import (
    ExperimentalSignaturesSearchStrategiesConfig,
    classify_experimental_signatures_search_strategies_component,
    experimental_signatures_search_strategies_labels,
    validate_experimental_signatures_search_strategies_fixture,
)


def test_experimental_signatures_config_rejects_boundary_and_count_drift() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 8"):
        ExperimentalSignaturesSearchStrategiesConfig(expected_source_record_count=7)
    with pytest.raises(ValueError, match="expected_component_count must equal 4"):
        ExperimentalSignaturesSearchStrategiesConfig(expected_component_count=3)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R01655"):
        ExperimentalSignaturesSearchStrategiesConfig(next_source_boundary="P0R01654")


def test_experimental_signatures_classifiers_are_source_bounded() -> None:
    assert (
        classify_experimental_signatures_search_strategies_component("falsifiability_frame")
        == "two_particle_falsifiability_search_frame_boundary"
    )
    assert (
        classify_experimental_signatures_search_strategies_component("collider_channel")
        == "lhc_exotic_higgs_decay_search_channel_boundary"
    )
    assert (
        classify_experimental_signatures_search_strategies_component("cosmological_channel")
        == "superradiance_continuous_wave_search_channel_boundary"
    )
    assert (
        classify_experimental_signatures_search_strategies_component("complementary_test_boundary")
        == "complementary_falsifiable_hypothesis_boundary"
    )
    with pytest.raises(
        ValueError, match="unknown experimental-signatures search-strategy component"
    ):
        classify_experimental_signatures_search_strategies_component("mass_mixing")


def test_experimental_signatures_fixture_preserves_claim_boundary() -> None:
    result = validate_experimental_signatures_search_strategies_fixture()

    assert result.source_ledger_span == ("P0R01647", "P0R01654")
    assert result.hardware_status == "source_methodology_no_experiment"
    assert result.source_record_count == 8
    assert result.component_count == 4
    assert result.next_source_boundary == "P0R01655"
    assert result.null_controls == {
        "testability_framing_is_not_detection_evidence": 1.0,
        "lhc_search_channel_is_not_observed_excess": 1.0,
        "continuous_wave_search_channel_is_not_detected_boson_cloud": 1.0,
        "complementary_hypothesis_language_is_not_confirmation": 1.0,
    }
    assert (
        result.problem_metadata["protocol_state"]
        == "source_experimental_signatures_search_strategies_only_no_experiment"
    )
    assert result.problem_metadata["source_ledger_ids"][0] == "P0R01647"
    assert result.problem_metadata["source_ledger_ids"][-1] == "P0R01654"


def test_experimental_signatures_labels_name_next_lhc_phenomenology_boundary() -> None:
    labels = experimental_signatures_search_strategies_labels()

    assert labels["section"] == "Experimental Signatures and Search Strategies"
    assert labels["collider"] == "h_SM -> h_Psi h_Psi"
    assert labels["cosmology"] == "black-hole superradiance continuous gravitational waves"
    assert labels["detectors"] == "CMS, ATLAS, LISA, Einstein Telescope, Cosmic Explorer"
    assert (
        labels["next_boundary"]
        == "The Psi-Higgs Boson: Phenomenology and Experimental Signatures at the LHC"
    )
