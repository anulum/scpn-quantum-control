# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0  Mathematics of Dynamical Systems validation tests
"""Tests for Paper 0  Mathematics of Dynamical Systems source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.mathematics_of_dynamical_systems_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    MathematicsOfDynamicalSystemsConfig,
    classify_mathematics_of_dynamical_systems_component,
    mathematics_of_dynamical_systems_labels,
    validate_mathematics_of_dynamical_systems_fixture,
)


def test_mathematics_of_dynamical_systems_fixture_preserves_source_boundary() -> None:
    result = validate_mathematics_of_dynamical_systems_fixture()
    assert result.source_ledger_span == ("P0R05826", "P0R05835")
    assert result.source_record_count == 10
    assert result.component_count == 3
    assert result.next_source_boundary == "P0R05836"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_mathematics_of_dynamical_systems_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R05826"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R05835"


def test_mathematics_of_dynamical_systems_classification_and_labels_are_explicit() -> None:
    for component in (
        "mathematics_of_dynamical_systems",
        "source_component",
        "mathematics_of_geometry_topology",
    ):
        assert (
            classify_mathematics_of_dynamical_systems_component(component)
            == f"{component}_source_boundary"
        )
    labels = mathematics_of_dynamical_systems_labels()
    assert labels["section"] == " Mathematics of Dynamical Systems"
    assert labels["next_boundary"] == "P0R05836"


def test_mathematics_of_dynamical_systems_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 10"):
        MathematicsOfDynamicalSystemsConfig(expected_source_record_count=9)
    with pytest.raises(ValueError, match="expected_component_count must equal 3"):
        MathematicsOfDynamicalSystemsConfig(expected_component_count=4)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R05836"):
        MathematicsOfDynamicalSystemsConfig(next_source_boundary="P0R05835")
    with pytest.raises(ValueError, match="unknown mathematics_of_dynamical_systems component"):
        classify_mathematics_of_dynamical_systems_component("empirical_validation_claim")
