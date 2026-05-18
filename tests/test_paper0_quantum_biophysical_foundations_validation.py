# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0  Quantum & Biophysical Foundations validation tests
"""Tests for Paper 0  Quantum & Biophysical Foundations source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.quantum_biophysical_foundations_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    QuantumBiophysicalFoundationsConfig,
    classify_quantum_biophysical_foundations_component,
    quantum_biophysical_foundations_labels,
    validate_quantum_biophysical_foundations_fixture,
)


def test_quantum_biophysical_foundations_fixture_preserves_source_boundary() -> None:
    result = validate_quantum_biophysical_foundations_fixture()
    assert result.source_ledger_span == ("P0R05673", "P0R05680")
    assert result.source_record_count == 8
    assert result.component_count == 2
    assert result.next_source_boundary == "P0R05681"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_quantum_biophysical_foundations_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R05673"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R05680"


def test_quantum_biophysical_foundations_classification_and_labels_are_explicit() -> None:
    for component in ("quantum_biophysical_foundations", "quantum_biology_biophysics"):
        assert (
            classify_quantum_biophysical_foundations_component(component)
            == f"{component}_source_boundary"
        )
    labels = quantum_biophysical_foundations_labels()
    assert labels["section"] == " Quantum & Biophysical Foundations"
    assert labels["next_boundary"] == "P0R05681"


def test_quantum_biophysical_foundations_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 8"):
        QuantumBiophysicalFoundationsConfig(expected_source_record_count=7)
    with pytest.raises(ValueError, match="expected_component_count must equal 2"):
        QuantumBiophysicalFoundationsConfig(expected_component_count=3)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R05681"):
        QuantumBiophysicalFoundationsConfig(next_source_boundary="P0R05680")
    with pytest.raises(ValueError, match="unknown quantum_biophysical_foundations component"):
        classify_quantum_biophysical_foundations_component("empirical_validation_claim")
