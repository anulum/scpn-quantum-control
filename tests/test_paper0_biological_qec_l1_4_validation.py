# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Biological QEC (L1-4): validation tests
"""Tests for Paper 0 Biological QEC (L1-4): source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.biological_qec_l1_4_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    BiologicalQecL14Config,
    biological_qec_l1_4_labels,
    classify_biological_qec_l1_4_component,
    validate_biological_qec_l1_4_fixture,
)


def test_biological_qec_l1_4_fixture_preserves_source_boundary() -> None:
    result = validate_biological_qec_l1_4_fixture()
    assert result.source_ledger_span == ("P0R03042", "P0R03050")
    assert result.source_record_count == 9
    assert result.component_count == 4
    assert result.next_source_boundary == "P0R03051"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_biological_qec_l1_4_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R03042"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R03050"


def test_biological_qec_l1_4_classification_and_labels_are_explicit() -> None:
    for component in (
        "biological_qec_l1_4",
        "network_qec_l4_8",
        "holographic_qec_l9_10",
        "cosmological_qec_l13_15",
    ):
        assert classify_biological_qec_l1_4_component(component) == f"{component}_source_boundary"
    labels = biological_qec_l1_4_labels()
    assert labels["section"] == "Biological QEC (L1-4):"
    assert labels["next_boundary"] == "P0R03051"


def test_biological_qec_l1_4_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 9"):
        BiologicalQecL14Config(expected_source_record_count=8)
    with pytest.raises(ValueError, match="expected_component_count must equal 4"):
        BiologicalQecL14Config(expected_component_count=5)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R03051"):
        BiologicalQecL14Config(next_source_boundary="P0R03050")
    with pytest.raises(ValueError, match="unknown biological_qec_l1_4 component"):
        classify_biological_qec_l1_4_component("empirical_validation_claim")
