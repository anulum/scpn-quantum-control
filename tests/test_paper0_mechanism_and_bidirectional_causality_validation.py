# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Mechanism and Bidirectional Causality validation tests
"""Tests for Paper 0 Mechanism and Bidirectional Causality source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.mechanism_and_bidirectional_causality_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    MechanismAndBidirectionalCausalityConfig,
    classify_mechanism_and_bidirectional_causality_component,
    mechanism_and_bidirectional_causality_labels,
    validate_mechanism_and_bidirectional_causality_fixture,
)


def test_mechanism_and_bidirectional_causality_fixture_preserves_source_boundary() -> None:
    result = validate_mechanism_and_bidirectional_causality_fixture()
    assert result.source_ledger_span == ("P0R04404", "P0R04412")
    assert result.source_record_count == 9
    assert result.component_count == 3
    assert result.next_source_boundary == "P0R04413"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_mechanism_and_bidirectional_causality_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R04404"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R04412"


def test_mechanism_and_bidirectional_causality_classification_and_labels_are_explicit() -> None:
    for component in (
        "mechanism_and_bidirectional_causality",
        "iv_the_geometry_of_networks_and_dynamics_domain_i_l4",
        "1_the_connectome_topology_the_optimised_scaffold",
    ):
        assert (
            classify_mechanism_and_bidirectional_causality_component(component)
            == f"{component}_source_boundary"
        )
    labels = mechanism_and_bidirectional_causality_labels()
    assert labels["section"] == "Mechanism and Bidirectional Causality"
    assert labels["next_boundary"] == "P0R04413"


def test_mechanism_and_bidirectional_causality_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 9"):
        MechanismAndBidirectionalCausalityConfig(expected_source_record_count=8)
    with pytest.raises(ValueError, match="expected_component_count must equal 3"):
        MechanismAndBidirectionalCausalityConfig(expected_component_count=4)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R04413"):
        MechanismAndBidirectionalCausalityConfig(next_source_boundary="P0R04412")
    with pytest.raises(
        ValueError, match="unknown mechanism_and_bidirectional_causality component"
    ):
        classify_mechanism_and_bidirectional_causality_component("empirical_validation_claim")
