# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Multi-Scale Quantum Error Correction (MS-QEC) validation tests
"""Tests for Paper 0 Multi-Scale Quantum Error Correction (MS-QEC) source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.multi_scale_quantum_error_correction_ms_qec_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    MultiScaleQuantumErrorCorrectionMsQecConfig,
    classify_multi_scale_quantum_error_correction_ms_qec_component,
    multi_scale_quantum_error_correction_ms_qec_labels,
    validate_multi_scale_quantum_error_correction_ms_qec_fixture,
)


def test_multi_scale_quantum_error_correction_ms_qec_fixture_preserves_source_boundary() -> None:
    result = validate_multi_scale_quantum_error_correction_ms_qec_fixture()
    assert result.source_ledger_span == ("P0R03010", "P0R03024")
    assert result.source_record_count == 15
    assert result.component_count == 1
    assert result.next_source_boundary == "P0R03025"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_multi_scale_quantum_error_correction_ms_qec_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R03010"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R03024"


def test_multi_scale_quantum_error_correction_ms_qec_classification_and_labels_are_explicit() -> (
    None
):
    for component in ("multi_scale_quantum_error_correction_ms_qec",):
        assert (
            classify_multi_scale_quantum_error_correction_ms_qec_component(component)
            == f"{component}_source_boundary"
        )
    labels = multi_scale_quantum_error_correction_ms_qec_labels()
    assert labels["section"] == "Multi-Scale Quantum Error Correction (MS-QEC)"
    assert labels["next_boundary"] == "P0R03025"


def test_multi_scale_quantum_error_correction_ms_qec_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 15"):
        MultiScaleQuantumErrorCorrectionMsQecConfig(expected_source_record_count=14)
    with pytest.raises(ValueError, match="expected_component_count must equal 1"):
        MultiScaleQuantumErrorCorrectionMsQecConfig(expected_component_count=2)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R03025"):
        MultiScaleQuantumErrorCorrectionMsQecConfig(next_source_boundary="P0R03024")
    with pytest.raises(
        ValueError, match="unknown multi_scale_quantum_error_correction_ms_qec component"
    ):
        classify_multi_scale_quantum_error_correction_ms_qec_component(
            "empirical_validation_claim"
        )
