# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 LHC search roadmap validation tests
"""Tests for Paper 0 LHC search-strategy roadmap validation."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.lhc_search_strategy_roadmap_validation import (
    LHCSearchStrategyRoadmapConfig,
    classify_lhc_search_strategy_roadmap_component,
    lhc_search_strategy_roadmap_labels,
    validate_lhc_search_strategy_roadmap_fixture,
)


def test_lhc_search_strategy_roadmap_config_rejects_boundary_and_count_drift() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 9"):
        LHCSearchStrategyRoadmapConfig(expected_source_record_count=8)
    with pytest.raises(ValueError, match="expected_component_count must equal 3"):
        LHCSearchStrategyRoadmapConfig(expected_component_count=4)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R01693"):
        LHCSearchStrategyRoadmapConfig(next_source_boundary="P0R01692")


def test_lhc_search_strategy_roadmap_classifiers_are_source_bounded() -> None:
    assert (
        classify_lhc_search_strategy_roadmap_component("search_signature_overview")
        == "lhc_search_signature_overview_boundary"
    )
    assert (
        classify_lhc_search_strategy_roadmap_component("table_roadmap")
        == "table_2_experimental_roadmap_source_boundary"
    )
    assert (
        classify_lhc_search_strategy_roadmap_component("ssb_cascade_transition")
        == "ssb_cascade_section_transition_boundary"
    )
    with pytest.raises(ValueError, match="unknown LHC search-strategy roadmap component"):
        classify_lhc_search_strategy_roadmap_component("hierarchy_genesis")


def test_lhc_search_strategy_roadmap_fixture_preserves_claim_boundary() -> None:
    result = validate_lhc_search_strategy_roadmap_fixture()

    assert result.source_ledger_span == ("P0R01684", "P0R01692")
    assert result.hardware_status == "source_methodology_no_experiment"
    assert result.source_record_count == 9
    assert result.component_count == 3
    assert result.next_source_boundary == "P0R01693"
    assert result.null_controls == {
        "lhc_search_channels_are_not_observed_psi_higgs_events": 1.0,
        "tbl003_is_source_roadmap_not_experimental_result": 1.0,
        "ssb_cascade_transition_has_no_empirical_claim": 1.0,
    }
    assert (
        result.problem_metadata["protocol_state"]
        == "source_lhc_search_strategy_roadmap_only_no_experiment"
    )
    assert result.problem_metadata["source_ledger_ids"][0] == "P0R01684"
    assert result.problem_metadata["source_ledger_ids"][-1] == "P0R01692"


def test_lhc_search_strategy_roadmap_labels_name_next_hierarchy_boundary() -> None:
    labels = lhc_search_strategy_roadmap_labels()

    assert labels["section"] == "Phenomenology and Search Strategies at the LHC"
    assert (
        labels["table"] == "TBL003 Proposed Experimental Search Parameters for the Psi-Higgs Boson"
    )
    assert (
        labels["channels"]
        == "exotic Higgs decays, resonant production, cascade decays, invisible decays"
    )
    assert (
        labels["next_boundary"]
        == "The Genesis of the Hierarchy: A Cascade of Sequential Symmetry Breaking"
    )
