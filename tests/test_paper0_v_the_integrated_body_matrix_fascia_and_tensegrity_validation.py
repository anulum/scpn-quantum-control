# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 V. The Integrated Body Matrix (Fascia and Tensegrity) validation tests
"""Tests for Paper 0 V. The Integrated Body Matrix (Fascia and Tensegrity) source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.v_the_integrated_body_matrix_fascia_and_tensegrity_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    VTheIntegratedBodyMatrixFasciaAndTensegrityConfig,
    classify_v_the_integrated_body_matrix_fascia_and_tensegrity_component,
    v_the_integrated_body_matrix_fascia_and_tensegrity_labels,
    validate_v_the_integrated_body_matrix_fascia_and_tensegrity_fixture,
)


def test_v_the_integrated_body_matrix_fascia_and_tensegrity_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_v_the_integrated_body_matrix_fascia_and_tensegrity_fixture()
    assert result.source_ledger_span == ("P0R04943", "P0R04955")
    assert result.source_record_count == 13
    assert result.component_count == 2
    assert result.next_source_boundary == "P0R04956"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_v_the_integrated_body_matrix_fascia_and_tensegrity_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R04943"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R04955"


def test_v_the_integrated_body_matrix_fascia_and_tensegrity_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "v_the_integrated_body_matrix_fascia_and_tensegrity",
        "vi_synthesis_the_holistic_pathology_of_the_scpn",
    ):
        assert (
            classify_v_the_integrated_body_matrix_fascia_and_tensegrity_component(component)
            == f"{component}_source_boundary"
        )
    labels = v_the_integrated_body_matrix_fascia_and_tensegrity_labels()
    assert labels["section"] == "V. The Integrated Body Matrix (Fascia and Tensegrity)"
    assert labels["next_boundary"] == "P0R04956"


def test_v_the_integrated_body_matrix_fascia_and_tensegrity_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 13"):
        VTheIntegratedBodyMatrixFasciaAndTensegrityConfig(expected_source_record_count=12)
    with pytest.raises(ValueError, match="expected_component_count must equal 2"):
        VTheIntegratedBodyMatrixFasciaAndTensegrityConfig(expected_component_count=3)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R04956"):
        VTheIntegratedBodyMatrixFasciaAndTensegrityConfig(next_source_boundary="P0R04955")
    with pytest.raises(
        ValueError, match="unknown v_the_integrated_body_matrix_fascia_and_tensegrity component"
    ):
        classify_v_the_integrated_body_matrix_fascia_and_tensegrity_component(
            "empirical_validation_claim"
        )
