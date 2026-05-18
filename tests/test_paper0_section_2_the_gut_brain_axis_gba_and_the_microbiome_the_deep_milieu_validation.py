# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 2. The Gut-Brain Axis (GBA) and the Microbiome: The Deep Milieu validation tests
"""Tests for Paper 0 2. The Gut-Brain Axis (GBA) and the Microbiome: The Deep Milieu source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.section_2_the_gut_brain_axis_gba_and_the_microbiome_the_deep_milieu_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    Section2TheGutBrainAxisGbaAndTheMicrobiomeTheDeepMilieuConfig,
    classify_section_2_the_gut_brain_axis_gba_and_the_microbiome_the_deep_milieu_component,
    section_2_the_gut_brain_axis_gba_and_the_microbiome_the_deep_milieu_labels,
    validate_section_2_the_gut_brain_axis_gba_and_the_microbiome_the_deep_milieu_fixture,
)


def test_section_2_the_gut_brain_axis_gba_and_the_microbiome_the_deep_milieu_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_section_2_the_gut_brain_axis_gba_and_the_microbiome_the_deep_milieu_fixture()
    assert result.source_ledger_span == ("P0R04911", "P0R04920")
    assert result.source_record_count == 10
    assert result.component_count == 1
    assert result.next_source_boundary == "P0R04921"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_section_2_the_gut_brain_axis_gba_and_the_microbiome_the_deep_milieu_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R04911"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R04920"


def test_section_2_the_gut_brain_axis_gba_and_the_microbiome_the_deep_milieu_classification_and_labels_are_explicit() -> (
    None
):
    for component in ("2_the_gut_brain_axis_gba_and_the_microbiome_the_deep_milieu",):
        assert (
            classify_section_2_the_gut_brain_axis_gba_and_the_microbiome_the_deep_milieu_component(
                component
            )
            == f"{component}_source_boundary"
        )
    labels = section_2_the_gut_brain_axis_gba_and_the_microbiome_the_deep_milieu_labels()
    assert labels["section"] == "2. The Gut-Brain Axis (GBA) and the Microbiome: The Deep Milieu"
    assert labels["next_boundary"] == "P0R04921"


def test_section_2_the_gut_brain_axis_gba_and_the_microbiome_the_deep_milieu_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 10"):
        Section2TheGutBrainAxisGbaAndTheMicrobiomeTheDeepMilieuConfig(
            expected_source_record_count=9
        )
    with pytest.raises(ValueError, match="expected_component_count must equal 1"):
        Section2TheGutBrainAxisGbaAndTheMicrobiomeTheDeepMilieuConfig(expected_component_count=2)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R04921"):
        Section2TheGutBrainAxisGbaAndTheMicrobiomeTheDeepMilieuConfig(
            next_source_boundary="P0R04920"
        )
    with pytest.raises(
        ValueError,
        match="unknown section_2_the_gut_brain_axis_gba_and_the_microbiome_the_deep_milieu component",
    ):
        classify_section_2_the_gut_brain_axis_gba_and_the_microbiome_the_deep_milieu_component(
            "empirical_validation_claim"
        )
