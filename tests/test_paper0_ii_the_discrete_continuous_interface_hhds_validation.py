# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 II. The Discrete-Continuous Interface (HHDS) validation tests
"""Tests for Paper 0 II. The Discrete-Continuous Interface (HHDS) source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.ii_the_discrete_continuous_interface_hhds_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    IiTheDiscreteContinuousInterfaceHhdsConfig,
    classify_ii_the_discrete_continuous_interface_hhds_component,
    ii_the_discrete_continuous_interface_hhds_labels,
    validate_ii_the_discrete_continuous_interface_hhds_fixture,
)


def test_ii_the_discrete_continuous_interface_hhds_fixture_preserves_source_boundary() -> None:
    result = validate_ii_the_discrete_continuous_interface_hhds_fixture()
    assert result.source_ledger_span == ("P0R03260", "P0R03268")
    assert result.source_record_count == 9
    assert result.component_count == 4
    assert result.next_source_boundary == "P0R03269"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_ii_the_discrete_continuous_interface_hhds_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R03260"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R03268"


def test_ii_the_discrete_continuous_interface_hhds_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "ii_the_discrete_continuous_interface_hhds",
        "iii_formalising_emergence_ginzburg_landau",
        "iv_the_physics_of_meaning_semantic_transduction",
        "v_the_observer_loop_and_the_strange_loop_of_self_l5",
    ):
        assert (
            classify_ii_the_discrete_continuous_interface_hhds_component(component)
            == f"{component}_source_boundary"
        )
    labels = ii_the_discrete_continuous_interface_hhds_labels()
    assert labels["section"] == "II. The Discrete-Continuous Interface (HHDS)"
    assert labels["next_boundary"] == "P0R03269"


def test_ii_the_discrete_continuous_interface_hhds_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 9"):
        IiTheDiscreteContinuousInterfaceHhdsConfig(expected_source_record_count=8)
    with pytest.raises(ValueError, match="expected_component_count must equal 4"):
        IiTheDiscreteContinuousInterfaceHhdsConfig(expected_component_count=5)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R03269"):
        IiTheDiscreteContinuousInterfaceHhdsConfig(next_source_boundary="P0R03268")
    with pytest.raises(
        ValueError, match="unknown ii_the_discrete_continuous_interface_hhds component"
    ):
        classify_ii_the_discrete_continuous_interface_hhds_component("empirical_validation_claim")
