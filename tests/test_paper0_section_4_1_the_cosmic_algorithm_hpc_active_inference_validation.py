# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 4.1 The Cosmic Algorithm: HPC & Active Inference validation tests
"""Tests for Paper 0 4.1 The Cosmic Algorithm: HPC & Active Inference source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.section_4_1_the_cosmic_algorithm_hpc_active_inference_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    Section41TheCosmicAlgorithmHpcActiveInferenceConfig,
    classify_section_4_1_the_cosmic_algorithm_hpc_active_inference_component,
    section_4_1_the_cosmic_algorithm_hpc_active_inference_labels,
    validate_section_4_1_the_cosmic_algorithm_hpc_active_inference_fixture,
)


def test_section_4_1_the_cosmic_algorithm_hpc_active_inference_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_section_4_1_the_cosmic_algorithm_hpc_active_inference_fixture()
    assert result.source_ledger_span == ("P0R03174", "P0R03196")
    assert result.source_record_count == 23
    assert result.component_count == 2
    assert result.next_source_boundary == "P0R03197"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_section_4_1_the_cosmic_algorithm_hpc_active_inference_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R03174"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R03196"


def test_section_4_1_the_cosmic_algorithm_hpc_active_inference_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "4_1_the_cosmic_algorithm_hpc_active_inference",
        "integrative_mechanisms_the_computational_and_physical_synthesis",
    ):
        assert (
            classify_section_4_1_the_cosmic_algorithm_hpc_active_inference_component(component)
            == f"{component}_source_boundary"
        )
    labels = section_4_1_the_cosmic_algorithm_hpc_active_inference_labels()
    assert labels["section"] == "4.1 The Cosmic Algorithm: HPC & Active Inference"
    assert labels["next_boundary"] == "P0R03197"


def test_section_4_1_the_cosmic_algorithm_hpc_active_inference_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 23"):
        Section41TheCosmicAlgorithmHpcActiveInferenceConfig(expected_source_record_count=22)
    with pytest.raises(ValueError, match="expected_component_count must equal 2"):
        Section41TheCosmicAlgorithmHpcActiveInferenceConfig(expected_component_count=3)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R03197"):
        Section41TheCosmicAlgorithmHpcActiveInferenceConfig(next_source_boundary="P0R03196")
    with pytest.raises(
        ValueError, match="unknown section_4_1_the_cosmic_algorithm_hpc_active_inference component"
    ):
        classify_section_4_1_the_cosmic_algorithm_hpc_active_inference_component(
            "empirical_validation_claim"
        )
