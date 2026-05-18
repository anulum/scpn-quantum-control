# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 3. The Dynamic Connectome and Functional Connectivity: validation tests
"""Tests for Paper 0 3. The Dynamic Connectome and Functional Connectivity: source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.section_3_the_dynamic_connectome_and_functional_connectivity_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    Section3TheDynamicConnectomeAndFunctionalConnectivityConfig,
    classify_section_3_the_dynamic_connectome_and_functional_connectivity_component,
    section_3_the_dynamic_connectome_and_functional_connectivity_labels,
    validate_section_3_the_dynamic_connectome_and_functional_connectivity_fixture,
)


def test_section_3_the_dynamic_connectome_and_functional_connectivity_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_section_3_the_dynamic_connectome_and_functional_connectivity_fixture()
    assert result.source_ledger_span == ("P0R04674", "P0R04683")
    assert result.source_record_count == 10
    assert result.component_count == 4
    assert result.next_source_boundary == "P0R04684"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_section_3_the_dynamic_connectome_and_functional_connectivity_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R04674"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R04683"


def test_section_3_the_dynamic_connectome_and_functional_connectivity_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "3_the_dynamic_connectome_and_functional_connectivity",
        "iv_the_architecture_of_the_conscious_self_domain_ii_l5",
        "1_hierarchical_predictive_coding_hpc_and_the_canonical_microcircuit",
        "2_the_self_the_dmn_and_the_strange_loop",
    ):
        assert (
            classify_section_3_the_dynamic_connectome_and_functional_connectivity_component(
                component
            )
            == f"{component}_source_boundary"
        )
    labels = section_3_the_dynamic_connectome_and_functional_connectivity_labels()
    assert labels["section"] == "3. The Dynamic Connectome and Functional Connectivity:"
    assert labels["next_boundary"] == "P0R04684"


def test_section_3_the_dynamic_connectome_and_functional_connectivity_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 10"):
        Section3TheDynamicConnectomeAndFunctionalConnectivityConfig(expected_source_record_count=9)
    with pytest.raises(ValueError, match="expected_component_count must equal 4"):
        Section3TheDynamicConnectomeAndFunctionalConnectivityConfig(expected_component_count=5)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R04684"):
        Section3TheDynamicConnectomeAndFunctionalConnectivityConfig(
            next_source_boundary="P0R04683"
        )
    with pytest.raises(
        ValueError,
        match="unknown section_3_the_dynamic_connectome_and_functional_connectivity component",
    ):
        classify_section_3_the_dynamic_connectome_and_functional_connectivity_component(
            "empirical_validation_claim"
        )
