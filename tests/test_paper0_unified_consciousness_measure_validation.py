# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Unified Consciousness Measure: validation tests
"""Tests for Paper 0 Unified Consciousness Measure: source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.unified_consciousness_measure_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    UnifiedConsciousnessMeasureConfig,
    classify_unified_consciousness_measure_component,
    unified_consciousness_measure_labels,
    validate_unified_consciousness_measure_fixture,
)


def test_unified_consciousness_measure_fixture_preserves_source_boundary() -> None:
    result = validate_unified_consciousness_measure_fixture()
    assert result.source_ledger_span == ("P0R03564", "P0R03580")
    assert result.source_record_count == 17
    assert result.component_count == 2
    assert result.next_source_boundary == "P0R03581"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_unified_consciousness_measure_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R03564"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R03580"


def test_unified_consciousness_measure_classification_and_labels_are_explicit() -> None:
    for component in ("unified_consciousness_measure", "iit_axioms_in_scpn"):
        assert (
            classify_unified_consciousness_measure_component(component)
            == f"{component}_source_boundary"
        )
    labels = unified_consciousness_measure_labels()
    assert labels["section"] == "Unified Consciousness Measure:"
    assert labels["next_boundary"] == "P0R03581"


def test_unified_consciousness_measure_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 17"):
        UnifiedConsciousnessMeasureConfig(expected_source_record_count=16)
    with pytest.raises(ValueError, match="expected_component_count must equal 2"):
        UnifiedConsciousnessMeasureConfig(expected_component_count=3)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R03581"):
        UnifiedConsciousnessMeasureConfig(next_source_boundary="P0R03580")
    with pytest.raises(ValueError, match="unknown unified_consciousness_measure component"):
        classify_unified_consciousness_measure_component("empirical_validation_claim")
