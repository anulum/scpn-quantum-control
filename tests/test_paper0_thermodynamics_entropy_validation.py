# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0  Thermodynamics & Entropy validation tests
"""Tests for Paper 0  Thermodynamics & Entropy source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.thermodynamics_entropy_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    ThermodynamicsEntropyConfig,
    classify_thermodynamics_entropy_component,
    thermodynamics_entropy_labels,
    validate_thermodynamics_entropy_fixture,
)


def test_thermodynamics_entropy_fixture_preserves_source_boundary() -> None:
    result = validate_thermodynamics_entropy_fixture()
    assert result.source_ledger_span == ("P0R05721", "P0R05729")
    assert result.source_record_count == 9
    assert result.component_count == 2
    assert result.next_source_boundary == "P0R05730"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_thermodynamics_entropy_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R05721"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R05729"


def test_thermodynamics_entropy_classification_and_labels_are_explicit() -> None:
    for component in ("thermodynamics_entropy", "gauge_field_theory_foundations"):
        assert (
            classify_thermodynamics_entropy_component(component) == f"{component}_source_boundary"
        )
    labels = thermodynamics_entropy_labels()
    assert labels["section"] == " Thermodynamics & Entropy"
    assert labels["next_boundary"] == "P0R05730"


def test_thermodynamics_entropy_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 9"):
        ThermodynamicsEntropyConfig(expected_source_record_count=8)
    with pytest.raises(ValueError, match="expected_component_count must equal 2"):
        ThermodynamicsEntropyConfig(expected_component_count=3)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R05730"):
        ThermodynamicsEntropyConfig(next_source_boundary="P0R05729")
    with pytest.raises(ValueError, match="unknown thermodynamics_entropy component"):
        classify_thermodynamics_entropy_component("empirical_validation_claim")
