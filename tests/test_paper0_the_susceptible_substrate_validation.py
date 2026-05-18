# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 The Susceptible Substrate: validation tests
"""Tests for Paper 0 The Susceptible Substrate: source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.the_susceptible_substrate_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    TheSusceptibleSubstrateConfig,
    classify_the_susceptible_substrate_component,
    the_susceptible_substrate_labels,
    validate_the_susceptible_substrate_fixture,
)


def test_the_susceptible_substrate_fixture_preserves_source_boundary() -> None:
    result = validate_the_susceptible_substrate_fixture()
    assert result.source_ledger_span == ("P0R02848", "P0R02858")
    assert result.source_record_count == 11
    assert result.component_count == 4
    assert result.next_source_boundary == "P0R02859"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_the_susceptible_substrate_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R02848"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R02858"


def test_the_susceptible_substrate_classification_and_labels_are_explicit() -> None:
    for component in (
        "the_susceptible_substrate",
        "the_branching_parameter_as_sigma",
        "overarching_dynamic_principles",
        "a_the_universal_dynamic_regime_quasicriticality",
    ):
        assert (
            classify_the_susceptible_substrate_component(component)
            == f"{component}_source_boundary"
        )
    labels = the_susceptible_substrate_labels()
    assert labels["section"] == "The Susceptible Substrate:"
    assert labels["next_boundary"] == "P0R02859"


def test_the_susceptible_substrate_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 11"):
        TheSusceptibleSubstrateConfig(expected_source_record_count=10)
    with pytest.raises(ValueError, match="expected_component_count must equal 4"):
        TheSusceptibleSubstrateConfig(expected_component_count=5)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R02859"):
        TheSusceptibleSubstrateConfig(next_source_boundary="P0R02858")
    with pytest.raises(ValueError, match="unknown the_susceptible_substrate component"):
        classify_the_susceptible_substrate_component("empirical_validation_claim")
