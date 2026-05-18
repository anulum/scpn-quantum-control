# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0  Consciousness Studies & Cognitive Models validation tests
"""Tests for Paper 0  Consciousness Studies & Cognitive Models source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.consciousness_studies_cognitive_models_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    ConsciousnessStudiesCognitiveModelsConfig,
    classify_consciousness_studies_cognitive_models_component,
    consciousness_studies_cognitive_models_labels,
    validate_consciousness_studies_cognitive_models_fixture,
)


def test_consciousness_studies_cognitive_models_fixture_preserves_source_boundary() -> None:
    result = validate_consciousness_studies_cognitive_models_fixture()
    assert result.source_ledger_span == ("P0R05860", "P0R05867")
    assert result.source_record_count == 8
    assert result.component_count == 2
    assert result.next_source_boundary == "P0R05868"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_consciousness_studies_cognitive_models_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R05860"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R05867"


def test_consciousness_studies_cognitive_models_classification_and_labels_are_explicit() -> None:
    for component in ("consciousness_studies_cognitive_models", "language_symbols_semiotics"):
        assert (
            classify_consciousness_studies_cognitive_models_component(component)
            == f"{component}_source_boundary"
        )
    labels = consciousness_studies_cognitive_models_labels()
    assert labels["section"] == " Consciousness Studies & Cognitive Models"
    assert labels["next_boundary"] == "P0R05868"


def test_consciousness_studies_cognitive_models_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 8"):
        ConsciousnessStudiesCognitiveModelsConfig(expected_source_record_count=7)
    with pytest.raises(ValueError, match="expected_component_count must equal 2"):
        ConsciousnessStudiesCognitiveModelsConfig(expected_component_count=3)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R05868"):
        ConsciousnessStudiesCognitiveModelsConfig(next_source_boundary="P0R05867")
    with pytest.raises(
        ValueError, match="unknown consciousness_studies_cognitive_models component"
    ):
        classify_consciousness_studies_cognitive_models_component("empirical_validation_claim")
