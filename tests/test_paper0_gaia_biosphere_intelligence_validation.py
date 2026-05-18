# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0  Gaia & Biosphere Intelligence validation tests
"""Tests for Paper 0  Gaia & Biosphere Intelligence source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.gaia_biosphere_intelligence_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    GaiaBiosphereIntelligenceConfig,
    classify_gaia_biosphere_intelligence_component,
    gaia_biosphere_intelligence_labels,
    validate_gaia_biosphere_intelligence_fixture,
)


def test_gaia_biosphere_intelligence_fixture_preserves_source_boundary() -> None:
    result = validate_gaia_biosphere_intelligence_fixture()
    assert result.source_ledger_span == ("P0R05910", "P0R05918")
    assert result.source_record_count == 9
    assert result.component_count == 2
    assert result.next_source_boundary == "P0R05919"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_gaia_biosphere_intelligence_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R05910"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R05918"


def test_gaia_biosphere_intelligence_classification_and_labels_are_explicit() -> None:
    for component in ("gaia_biosphere_intelligence", "metaphysical_foundational_crossovers"):
        assert (
            classify_gaia_biosphere_intelligence_component(component)
            == f"{component}_source_boundary"
        )
    labels = gaia_biosphere_intelligence_labels()
    assert labels["section"] == " Gaia & Biosphere Intelligence"
    assert labels["next_boundary"] == "P0R05919"


def test_gaia_biosphere_intelligence_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 9"):
        GaiaBiosphereIntelligenceConfig(expected_source_record_count=8)
    with pytest.raises(ValueError, match="expected_component_count must equal 2"):
        GaiaBiosphereIntelligenceConfig(expected_component_count=3)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R05919"):
        GaiaBiosphereIntelligenceConfig(next_source_boundary="P0R05918")
    with pytest.raises(ValueError, match="unknown gaia_biosphere_intelligence component"):
        classify_gaia_biosphere_intelligence_component("empirical_validation_claim")
