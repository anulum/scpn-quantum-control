# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0  Biophysics & Quantum Biology validation tests
"""Tests for Paper 0  Biophysics & Quantum Biology source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.biophysics_quantum_biology_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    BiophysicsQuantumBiologyConfig,
    biophysics_quantum_biology_labels,
    classify_biophysics_quantum_biology_component,
    validate_biophysics_quantum_biology_fixture,
)


def test_biophysics_quantum_biology_fixture_preserves_source_boundary() -> None:
    result = validate_biophysics_quantum_biology_fixture()
    assert result.source_ledger_span == ("P0R05650", "P0R05664")
    assert result.source_record_count == 15
    assert result.component_count == 2
    assert result.next_source_boundary == "P0R05665"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_biophysics_quantum_biology_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R05650"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R05664"


def test_biophysics_quantum_biology_classification_and_labels_are_explicit() -> None:
    for component in ("biophysics_quantum_biology", "cosmology_fundamental_physics"):
        assert (
            classify_biophysics_quantum_biology_component(component)
            == f"{component}_source_boundary"
        )
    labels = biophysics_quantum_biology_labels()
    assert labels["section"] == " Biophysics & Quantum Biology"
    assert labels["next_boundary"] == "P0R05665"


def test_biophysics_quantum_biology_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 15"):
        BiophysicsQuantumBiologyConfig(expected_source_record_count=14)
    with pytest.raises(ValueError, match="expected_component_count must equal 2"):
        BiophysicsQuantumBiologyConfig(expected_component_count=3)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R05665"):
        BiophysicsQuantumBiologyConfig(next_source_boundary="P0R05664")
    with pytest.raises(ValueError, match="unknown biophysics_quantum_biology component"):
        classify_biophysics_quantum_biology_component("empirical_validation_claim")
