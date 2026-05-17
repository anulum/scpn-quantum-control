# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Part II: The Physical Sector (Field Theory & Quantization) > 2.4 The SSB Cascade: Origin of Mass & The Solitonic Self > The "Self" as a Soliton: Emergence of Localised Consciousness (Layer 5) > The Definition of the Self: The Triadic Solution validation tests
"""Tests for Paper 0 Part II: The Physical Sector (Field Theory & Quantization) > 2.4 The SSB Cascade: Origin of Mass & The Solitonic Self > The "Self" as a Soliton: Emergence of Localised Consciousness (Layer 5) > The Definition of the Self: The Triadic Solution source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    PartIiThePhysicalSectorFieldTheoryQuantization24TheSsbCascadConfig,
    classify_part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad_component,
    part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad_labels,
    validate_part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad_fixture,
)


def test_part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad_fixture_preserves_source_boundary() -> (
    None
):
    result = (
        validate_part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad_fixture()
    )
    assert result.source_ledger_span == ("P0R01895", "P0R01958")
    assert result.source_record_count == 64
    assert result.component_count == 1
    assert result.next_source_boundary == "P0R01959"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R01895"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R01958"


def test_part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad_classification_and_labels_are_explicit() -> (
    None
):
    for component in ("part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad",):
        assert (
            classify_part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad_component(
                component
            )
            == f"{component}_source_boundary"
        )
    labels = part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad_labels()
    assert (
        labels["section"]
        == 'Part II: The Physical Sector (Field Theory & Quantization) > 2.4 The SSB Cascade: Origin of Mass & The Solitonic Self > The "Self" as a Soliton: Emergence of Localised Consciousness (Layer 5) > The Definition of the Self: The Triadic Solution'
    )
    assert labels["next_boundary"] == "P0R01959"


def test_part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 64"):
        PartIiThePhysicalSectorFieldTheoryQuantization24TheSsbCascadConfig(
            expected_source_record_count=63
        )
    with pytest.raises(ValueError, match="expected_component_count must equal 1"):
        PartIiThePhysicalSectorFieldTheoryQuantization24TheSsbCascadConfig(
            expected_component_count=2
        )
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R01959"):
        PartIiThePhysicalSectorFieldTheoryQuantization24TheSsbCascadConfig(
            next_source_boundary="P0R01958"
        )
    with pytest.raises(
        ValueError,
        match="unknown part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad component",
    ):
        classify_part_ii_the_physical_sector_field_theory_quantization_2_4_the_ssb_cascad_component(
            "empirical_validation_claim"
        )
