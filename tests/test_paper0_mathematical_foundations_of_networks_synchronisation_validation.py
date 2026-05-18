# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0  Mathematical Foundations of Networks & Synchronisation validation tests
"""Tests for Paper 0  Mathematical Foundations of Networks & Synchronisation source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.mathematical_foundations_of_networks_synchronisation_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    MathematicalFoundationsOfNetworksSynchronisationConfig,
    classify_mathematical_foundations_of_networks_synchronisation_component,
    mathematical_foundations_of_networks_synchronisation_labels,
    validate_mathematical_foundations_of_networks_synchronisation_fixture,
)


def test_mathematical_foundations_of_networks_synchronisation_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_mathematical_foundations_of_networks_synchronisation_fixture()
    assert result.source_ledger_span == ("P0R05836", "P0R05843")
    assert result.source_record_count == 8
    assert result.component_count == 2
    assert result.next_source_boundary == "P0R05844"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_mathematical_foundations_of_networks_synchronisation_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R05836"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R05843"


def test_mathematical_foundations_of_networks_synchronisation_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "mathematical_foundations_of_networks_synchronisation",
        "neuroscience_brain_rhythms",
    ):
        assert (
            classify_mathematical_foundations_of_networks_synchronisation_component(component)
            == f"{component}_source_boundary"
        )
    labels = mathematical_foundations_of_networks_synchronisation_labels()
    assert labels["section"] == " Mathematical Foundations of Networks & Synchronisation"
    assert labels["next_boundary"] == "P0R05844"


def test_mathematical_foundations_of_networks_synchronisation_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 8"):
        MathematicalFoundationsOfNetworksSynchronisationConfig(expected_source_record_count=7)
    with pytest.raises(ValueError, match="expected_component_count must equal 2"):
        MathematicalFoundationsOfNetworksSynchronisationConfig(expected_component_count=3)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R05844"):
        MathematicalFoundationsOfNetworksSynchronisationConfig(next_source_boundary="P0R05843")
    with pytest.raises(
        ValueError, match="unknown mathematical_foundations_of_networks_synchronisation component"
    ):
        classify_mathematical_foundations_of_networks_synchronisation_component(
            "empirical_validation_claim"
        )
