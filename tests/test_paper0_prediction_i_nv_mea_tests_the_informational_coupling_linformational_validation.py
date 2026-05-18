# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Prediction I (NV-MEA) Tests the Informational Coupling (LInformational): validation tests
"""Tests for Paper 0 Prediction I (NV-MEA) Tests the Informational Coupling (LInformational): source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.prediction_i_nv_mea_tests_the_informational_coupling_linformational_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    PredictionINvMeaTestsTheInformationalCouplingLinformationalConfig,
    classify_prediction_i_nv_mea_tests_the_informational_coupling_linformational_component,
    prediction_i_nv_mea_tests_the_informational_coupling_linformational_labels,
    validate_prediction_i_nv_mea_tests_the_informational_coupling_linformational_fixture,
)


def test_prediction_i_nv_mea_tests_the_informational_coupling_linformational_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_prediction_i_nv_mea_tests_the_informational_coupling_linformational_fixture()
    assert result.source_ledger_span == ("P0R05152", "P0R05161")
    assert result.source_record_count == 10
    assert result.component_count == 3
    assert result.next_source_boundary == "P0R05162"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_prediction_i_nv_mea_tests_the_informational_coupling_linformational_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R05152"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R05161"


def test_prediction_i_nv_mea_tests_the_informational_coupling_linformational_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "prediction_i_nv_mea_tests_the_informational_coupling_linformational",
        "prediction_ii_qrng_tests_the_geometric_coupling_lgeometric",
        "introduction",
    ):
        assert (
            classify_prediction_i_nv_mea_tests_the_informational_coupling_linformational_component(
                component
            )
            == f"{component}_source_boundary"
        )
    labels = prediction_i_nv_mea_tests_the_informational_coupling_linformational_labels()
    assert (
        labels["section"]
        == "Prediction I (NV-MEA) Tests the Informational Coupling (LInformational):"
    )
    assert labels["next_boundary"] == "P0R05162"


def test_prediction_i_nv_mea_tests_the_informational_coupling_linformational_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 10"):
        PredictionINvMeaTestsTheInformationalCouplingLinformationalConfig(
            expected_source_record_count=9
        )
    with pytest.raises(ValueError, match="expected_component_count must equal 3"):
        PredictionINvMeaTestsTheInformationalCouplingLinformationalConfig(
            expected_component_count=4
        )
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R05162"):
        PredictionINvMeaTestsTheInformationalCouplingLinformationalConfig(
            next_source_boundary="P0R05161"
        )
    with pytest.raises(
        ValueError,
        match="unknown prediction_i_nv_mea_tests_the_informational_coupling_linformational component",
    ):
        classify_prediction_i_nv_mea_tests_the_informational_coupling_linformational_component(
            "empirical_validation_claim"
        )
