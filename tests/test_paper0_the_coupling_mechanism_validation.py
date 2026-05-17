# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 The Coupling Mechanism: validation tests
"""Tests for Paper 0 The Coupling Mechanism: source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.the_coupling_mechanism_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    TheCouplingMechanismConfig,
    classify_the_coupling_mechanism_component,
    the_coupling_mechanism_labels,
    validate_the_coupling_mechanism_fixture,
)


def test_the_coupling_mechanism_fixture_preserves_source_boundary() -> None:
    result = validate_the_coupling_mechanism_fixture()
    assert result.source_ledger_span == ("P0R02206", "P0R02222")
    assert result.source_record_count == 17
    assert result.component_count == 1
    assert result.next_source_boundary == "P0R02223"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_the_coupling_mechanism_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R02206"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R02222"


def test_the_coupling_mechanism_classification_and_labels_are_explicit() -> None:
    for component in ("the_coupling_mechanism",):
        assert (
            classify_the_coupling_mechanism_component(component) == f"{component}_source_boundary"
        )
    labels = the_coupling_mechanism_labels()
    assert labels["section"] == "The Coupling Mechanism:"
    assert labels["next_boundary"] == "P0R02223"


def test_the_coupling_mechanism_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 17"):
        TheCouplingMechanismConfig(expected_source_record_count=16)
    with pytest.raises(ValueError, match="expected_component_count must equal 1"):
        TheCouplingMechanismConfig(expected_component_count=2)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R02223"):
        TheCouplingMechanismConfig(next_source_boundary="P0R02222")
    with pytest.raises(ValueError, match="unknown the_coupling_mechanism component"):
        classify_the_coupling_mechanism_component("empirical_validation_claim")
