# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 III. The Coherence Backbone: Multi-Scale Quantum Error Correction (MS-QEC) validation tests
"""Tests for Paper 0 III. The Coherence Backbone: Multi-Scale Quantum Error Correction (MS-QEC) source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.iii_the_coherence_backbone_multi_scale_quantum_error_correction_ms_qec_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    IiiTheCoherenceBackboneMultiScaleQuantumErrorCorrectionMsQecConfig,
    classify_iii_the_coherence_backbone_multi_scale_quantum_error_correction_ms_qec_component,
    iii_the_coherence_backbone_multi_scale_quantum_error_correction_ms_qec_labels,
    validate_iii_the_coherence_backbone_multi_scale_quantum_error_correction_ms_qec_fixture,
)


def test_iii_the_coherence_backbone_multi_scale_quantum_error_correction_ms_qec_fixture_preserves_source_boundary() -> (
    None
):
    result = (
        validate_iii_the_coherence_backbone_multi_scale_quantum_error_correction_ms_qec_fixture()
    )
    assert result.source_ledger_span == ("P0R02521", "P0R02531")
    assert result.source_record_count == 11
    assert result.component_count == 1
    assert result.next_source_boundary == "P0R02532"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_iii_the_coherence_backbone_multi_scale_quantum_error_correction_ms_qec_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R02521"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R02531"


def test_iii_the_coherence_backbone_multi_scale_quantum_error_correction_ms_qec_classification_and_labels_are_explicit() -> (
    None
):
    for component in ("iii_the_coherence_backbone_multi_scale_quantum_error_correction_ms_qec",):
        assert (
            classify_iii_the_coherence_backbone_multi_scale_quantum_error_correction_ms_qec_component(
                component
            )
            == f"{component}_source_boundary"
        )
    labels = iii_the_coherence_backbone_multi_scale_quantum_error_correction_ms_qec_labels()
    assert (
        labels["section"]
        == "III. The Coherence Backbone: Multi-Scale Quantum Error Correction (MS-QEC)"
    )
    assert labels["next_boundary"] == "P0R02532"


def test_iii_the_coherence_backbone_multi_scale_quantum_error_correction_ms_qec_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 11"):
        IiiTheCoherenceBackboneMultiScaleQuantumErrorCorrectionMsQecConfig(
            expected_source_record_count=10
        )
    with pytest.raises(ValueError, match="expected_component_count must equal 1"):
        IiiTheCoherenceBackboneMultiScaleQuantumErrorCorrectionMsQecConfig(
            expected_component_count=2
        )
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R02532"):
        IiiTheCoherenceBackboneMultiScaleQuantumErrorCorrectionMsQecConfig(
            next_source_boundary="P0R02531"
        )
    with pytest.raises(
        ValueError,
        match="unknown iii_the_coherence_backbone_multi_scale_quantum_error_correction_ms_qec component",
    ):
        classify_iii_the_coherence_backbone_multi_scale_quantum_error_correction_ms_qec_component(
            "empirical_validation_claim"
        )
