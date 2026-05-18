# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0  as a Coupling Affinity: validation tests
"""Tests for Paper 0  as a Coupling Affinity: source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.as_a_coupling_affinity_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    AsACouplingAffinityConfig,
    as_a_coupling_affinity_labels,
    classify_as_a_coupling_affinity_component,
    validate_as_a_coupling_affinity_fixture,
)


def test_as_a_coupling_affinity_fixture_preserves_source_boundary() -> None:
    result = validate_as_a_coupling_affinity_fixture()
    assert result.source_ledger_span == ("P0R03501", "P0R03509")
    assert result.source_record_count == 9
    assert result.component_count == 3
    assert result.next_source_boundary == "P0R03510"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_as_a_coupling_affinity_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R03501"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R03509"


def test_as_a_coupling_affinity_classification_and_labels_are_explicit() -> None:
    for component in (
        "as_a_coupling_affinity",
        "the_scaling_law_of_consciousness_slc",
        "formalisation_via_integrated_information",
    ):
        assert (
            classify_as_a_coupling_affinity_component(component) == f"{component}_source_boundary"
        )
    labels = as_a_coupling_affinity_labels()
    assert labels["section"] == " as a Coupling Affinity:"
    assert labels["next_boundary"] == "P0R03510"


def test_as_a_coupling_affinity_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 9"):
        AsACouplingAffinityConfig(expected_source_record_count=8)
    with pytest.raises(ValueError, match="expected_component_count must equal 3"):
        AsACouplingAffinityConfig(expected_component_count=4)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R03510"):
        AsACouplingAffinityConfig(next_source_boundary="P0R03509")
    with pytest.raises(ValueError, match="unknown as_a_coupling_affinity component"):
        classify_as_a_coupling_affinity_component("empirical_validation_claim")
