# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 1. The Lipid Bilayer and Lipid Rafts: validation tests
"""Tests for Paper 0 1. The Lipid Bilayer and Lipid Rafts: source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.section_1_the_lipid_bilayer_and_lipid_rafts_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    Section1TheLipidBilayerAndLipidRaftsConfig,
    classify_section_1_the_lipid_bilayer_and_lipid_rafts_component,
    section_1_the_lipid_bilayer_and_lipid_rafts_labels,
    validate_section_1_the_lipid_bilayer_and_lipid_rafts_fixture,
)


def test_section_1_the_lipid_bilayer_and_lipid_rafts_fixture_preserves_source_boundary() -> None:
    result = validate_section_1_the_lipid_bilayer_and_lipid_rafts_fixture()
    assert result.source_ledger_span == ("P0R04720", "P0R04727")
    assert result.source_record_count == 8
    assert result.component_count == 3
    assert result.next_source_boundary == "P0R04728"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_section_1_the_lipid_bilayer_and_lipid_rafts_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R04720"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R04727"


def test_section_1_the_lipid_bilayer_and_lipid_rafts_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "1_the_lipid_bilayer_and_lipid_rafts",
        "2_ion_channels_the_molecular_transistors_l1_l2_interface",
        "iv_the_internal_architecture_cytoskeleton_and_organelles_l1_l3",
    ):
        assert (
            classify_section_1_the_lipid_bilayer_and_lipid_rafts_component(component)
            == f"{component}_source_boundary"
        )
    labels = section_1_the_lipid_bilayer_and_lipid_rafts_labels()
    assert labels["section"] == "1. The Lipid Bilayer and Lipid Rafts:"
    assert labels["next_boundary"] == "P0R04728"


def test_section_1_the_lipid_bilayer_and_lipid_rafts_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 8"):
        Section1TheLipidBilayerAndLipidRaftsConfig(expected_source_record_count=7)
    with pytest.raises(ValueError, match="expected_component_count must equal 3"):
        Section1TheLipidBilayerAndLipidRaftsConfig(expected_component_count=4)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R04728"):
        Section1TheLipidBilayerAndLipidRaftsConfig(next_source_boundary="P0R04727")
    with pytest.raises(
        ValueError, match="unknown section_1_the_lipid_bilayer_and_lipid_rafts component"
    ):
        classify_section_1_the_lipid_bilayer_and_lipid_rafts_component(
            "empirical_validation_claim"
        )
