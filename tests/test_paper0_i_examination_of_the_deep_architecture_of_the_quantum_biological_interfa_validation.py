# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 I. Examination of The Deep Architecture of the Quantum-Biological Interface (Domain I: L1-L2) validation tests
"""Tests for Paper 0 I. Examination of The Deep Architecture of the Quantum-Biological Interface (Domain I: L1-L2) source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.i_examination_of_the_deep_architecture_of_the_quantum_biological_interfa_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    IExaminationOfTheDeepArchitectureOfTheQuantumBiologicalInterfaConfig,
    classify_i_examination_of_the_deep_architecture_of_the_quantum_biological_interfa_component,
    i_examination_of_the_deep_architecture_of_the_quantum_biological_interfa_labels,
    validate_i_examination_of_the_deep_architecture_of_the_quantum_biological_interfa_fixture,
)


def test_i_examination_of_the_deep_architecture_of_the_quantum_biological_interfa_fixture_preserves_source_boundary() -> (
    None
):
    result = (
        validate_i_examination_of_the_deep_architecture_of_the_quantum_biological_interfa_fixture()
    )
    assert result.source_ledger_span == ("P0R04544", "P0R04551")
    assert result.source_record_count == 8
    assert result.component_count == 3
    assert result.next_source_boundary == "P0R04552"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_i_examination_of_the_deep_architecture_of_the_quantum_biological_interfa_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R04544"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R04551"


def test_i_examination_of_the_deep_architecture_of_the_quantum_biological_interfa_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "i_examination_of_the_deep_architecture_of_the_quantum_biological_interfa",
        "the_extended_cytoskeletal_network_l1_the_tensegrity_matrix_of_life",
        "neuromodulators_as_precision_controllers_l2_tuning_the_neural_orchestra",
    ):
        assert (
            classify_i_examination_of_the_deep_architecture_of_the_quantum_biological_interfa_component(
                component
            )
            == f"{component}_source_boundary"
        )
    labels = i_examination_of_the_deep_architecture_of_the_quantum_biological_interfa_labels()
    assert (
        labels["section"]
        == "I. Examination of The Deep Architecture of the Quantum-Biological Interface (Domain I: L1-L2)"
    )
    assert labels["next_boundary"] == "P0R04552"


def test_i_examination_of_the_deep_architecture_of_the_quantum_biological_interfa_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 8"):
        IExaminationOfTheDeepArchitectureOfTheQuantumBiologicalInterfaConfig(
            expected_source_record_count=7
        )
    with pytest.raises(ValueError, match="expected_component_count must equal 3"):
        IExaminationOfTheDeepArchitectureOfTheQuantumBiologicalInterfaConfig(
            expected_component_count=4
        )
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R04552"):
        IExaminationOfTheDeepArchitectureOfTheQuantumBiologicalInterfaConfig(
            next_source_boundary="P0R04551"
        )
    with pytest.raises(
        ValueError,
        match="unknown i_examination_of_the_deep_architecture_of_the_quantum_biological_interfa component",
    ):
        classify_i_examination_of_the_deep_architecture_of_the_quantum_biological_interfa_component(
            "empirical_validation_claim"
        )
