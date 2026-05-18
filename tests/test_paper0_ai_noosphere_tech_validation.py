# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0  AI, Noosphere & Tech validation tests
"""Tests for Paper 0  AI, Noosphere & Tech source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.ai_noosphere_tech_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    AiNoosphereTechConfig,
    ai_noosphere_tech_labels,
    classify_ai_noosphere_tech_component,
    validate_ai_noosphere_tech_fixture,
)


def test_ai_noosphere_tech_fixture_preserves_source_boundary() -> None:
    result = validate_ai_noosphere_tech_fixture()
    assert result.source_ledger_span == ("P0R05802", "P0R05809")
    assert result.source_record_count == 8
    assert result.component_count == 2
    assert result.next_source_boundary == "P0R05810"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"] == "source_ai_noosphere_tech_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R05802"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R05809"


def test_ai_noosphere_tech_classification_and_labels_are_explicit() -> None:
    for component in ("ai_noosphere_tech", "ai_culture_and_noosphere_dynamics"):
        assert classify_ai_noosphere_tech_component(component) == f"{component}_source_boundary"
    labels = ai_noosphere_tech_labels()
    assert labels["section"] == " AI, Noosphere & Tech"
    assert labels["next_boundary"] == "P0R05810"


def test_ai_noosphere_tech_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 8"):
        AiNoosphereTechConfig(expected_source_record_count=7)
    with pytest.raises(ValueError, match="expected_component_count must equal 2"):
        AiNoosphereTechConfig(expected_component_count=3)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R05810"):
        AiNoosphereTechConfig(next_source_boundary="P0R05809")
    with pytest.raises(ValueError, match="unknown ai_noosphere_tech component"):
        classify_ai_noosphere_tech_component("empirical_validation_claim")
