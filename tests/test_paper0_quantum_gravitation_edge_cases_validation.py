# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0  Quantum & Gravitation Edge Cases validation tests
"""Tests for Paper 0  Quantum & Gravitation Edge Cases source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.quantum_gravitation_edge_cases_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    QuantumGravitationEdgeCasesConfig,
    classify_quantum_gravitation_edge_cases_component,
    quantum_gravitation_edge_cases_labels,
    validate_quantum_gravitation_edge_cases_fixture,
)


def test_quantum_gravitation_edge_cases_fixture_preserves_source_boundary() -> None:
    result = validate_quantum_gravitation_edge_cases_fixture()
    assert result.source_ledger_span == ("P0R05689", "P0R05696")
    assert result.source_record_count == 8
    assert result.component_count == 2
    assert result.next_source_boundary == "P0R05697"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_quantum_gravitation_edge_cases_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R05689"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R05696"


def test_quantum_gravitation_edge_cases_classification_and_labels_are_explicit() -> None:
    for component in ("quantum_gravitation_edge_cases", "quantum_foundations_info"):
        assert (
            classify_quantum_gravitation_edge_cases_component(component)
            == f"{component}_source_boundary"
        )
    labels = quantum_gravitation_edge_cases_labels()
    assert labels["section"] == " Quantum & Gravitation Edge Cases"
    assert labels["next_boundary"] == "P0R05697"


def test_quantum_gravitation_edge_cases_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 8"):
        QuantumGravitationEdgeCasesConfig(expected_source_record_count=7)
    with pytest.raises(ValueError, match="expected_component_count must equal 2"):
        QuantumGravitationEdgeCasesConfig(expected_component_count=3)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R05697"):
        QuantumGravitationEdgeCasesConfig(next_source_boundary="P0R05696")
    with pytest.raises(ValueError, match="unknown quantum_gravitation_edge_cases component"):
        classify_quantum_gravitation_edge_cases_component("empirical_validation_claim")
