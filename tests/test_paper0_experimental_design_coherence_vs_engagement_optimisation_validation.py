# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Experimental Design: Coherence vs. Engagement Optimisation validation tests
"""Tests for Paper 0 Experimental Design: Coherence vs. Engagement Optimisation source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.experimental_design_coherence_vs_engagement_optimisation_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    ExperimentalDesignCoherenceVsEngagementOptimisationConfig,
    classify_experimental_design_coherence_vs_engagement_optimisation_component,
    experimental_design_coherence_vs_engagement_optimisation_labels,
    validate_experimental_design_coherence_vs_engagement_optimisation_fixture,
)


def test_experimental_design_coherence_vs_engagement_optimisation_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_experimental_design_coherence_vs_engagement_optimisation_fixture()
    assert result.source_ledger_span == ("P0R05256", "P0R05263")
    assert result.source_record_count == 8
    assert result.component_count == 1
    assert result.next_source_boundary == "P0R05264"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_experimental_design_coherence_vs_engagement_optimisation_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R05256"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R05263"


def test_experimental_design_coherence_vs_engagement_optimisation_classification_and_labels_are_explicit() -> (
    None
):
    for component in ("experimental_design_coherence_vs_engagement_optimisation",):
        assert (
            classify_experimental_design_coherence_vs_engagement_optimisation_component(component)
            == f"{component}_source_boundary"
        )
    labels = experimental_design_coherence_vs_engagement_optimisation_labels()
    assert labels["section"] == "Experimental Design: Coherence vs. Engagement Optimisation"
    assert labels["next_boundary"] == "P0R05264"


def test_experimental_design_coherence_vs_engagement_optimisation_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 8"):
        ExperimentalDesignCoherenceVsEngagementOptimisationConfig(expected_source_record_count=7)
    with pytest.raises(ValueError, match="expected_component_count must equal 1"):
        ExperimentalDesignCoherenceVsEngagementOptimisationConfig(expected_component_count=2)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R05264"):
        ExperimentalDesignCoherenceVsEngagementOptimisationConfig(next_source_boundary="P0R05263")
    with pytest.raises(
        ValueError,
        match="unknown experimental_design_coherence_vs_engagement_optimisation component",
    ):
        classify_experimental_design_coherence_vs_engagement_optimisation_component(
            "empirical_validation_claim"
        )
