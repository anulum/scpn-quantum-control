# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 IV. The Coherence Backbone (MS-QEC and Symmetry) validation tests
"""Tests for Paper 0 IV. The Coherence Backbone (MS-QEC and Symmetry) source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.iv_the_coherence_backbone_ms_qec_and_symmetry_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    IvTheCoherenceBackboneMsQecAndSymmetryConfig,
    classify_iv_the_coherence_backbone_ms_qec_and_symmetry_component,
    iv_the_coherence_backbone_ms_qec_and_symmetry_labels,
    validate_iv_the_coherence_backbone_ms_qec_and_symmetry_fixture,
)


def test_iv_the_coherence_backbone_ms_qec_and_symmetry_fixture_preserves_source_boundary() -> None:
    result = validate_iv_the_coherence_backbone_ms_qec_and_symmetry_fixture()
    assert result.source_ledger_span == ("P0R06123", "P0R06131")
    assert result.source_record_count == 9
    assert result.component_count == 4
    assert result.next_source_boundary == "P0R06132"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_iv_the_coherence_backbone_ms_qec_and_symmetry_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R06123"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R06131"


def test_iv_the_coherence_backbone_ms_qec_and_symmetry_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "iv_the_coherence_backbone_ms_qec_and_symmetry",
        "v_the_architecture_of_time_mmc_tsvf_and_synchronicity",
        "vi_thermodynamics_and_energetics",
        "vii_the_scpn_measurement_postulate_intrinsic_measurement",
    ):
        assert (
            classify_iv_the_coherence_backbone_ms_qec_and_symmetry_component(component)
            == f"{component}_source_boundary"
        )
    labels = iv_the_coherence_backbone_ms_qec_and_symmetry_labels()
    assert labels["section"] == "IV. The Coherence Backbone (MS-QEC and Symmetry)"
    assert labels["next_boundary"] == "P0R06132"


def test_iv_the_coherence_backbone_ms_qec_and_symmetry_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 9"):
        IvTheCoherenceBackboneMsQecAndSymmetryConfig(expected_source_record_count=8)
    with pytest.raises(ValueError, match="expected_component_count must equal 4"):
        IvTheCoherenceBackboneMsQecAndSymmetryConfig(expected_component_count=5)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R06132"):
        IvTheCoherenceBackboneMsQecAndSymmetryConfig(next_source_boundary="P0R06131")
    with pytest.raises(
        ValueError, match="unknown iv_the_coherence_backbone_ms_qec_and_symmetry component"
    ):
        classify_iv_the_coherence_backbone_ms_qec_and_symmetry_component(
            "empirical_validation_claim"
        )
