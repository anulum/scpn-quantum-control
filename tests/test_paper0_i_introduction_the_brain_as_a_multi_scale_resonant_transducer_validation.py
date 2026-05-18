# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 I. Introduction: The Brain as a Multi-Scale Resonant Transducer validation tests
"""Tests for Paper 0 I. Introduction: The Brain as a Multi-Scale Resonant Transducer source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.i_introduction_the_brain_as_a_multi_scale_resonant_transducer_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    IIntroductionTheBrainAsAMultiScaleResonantTransducerConfig,
    classify_i_introduction_the_brain_as_a_multi_scale_resonant_transducer_component,
    i_introduction_the_brain_as_a_multi_scale_resonant_transducer_labels,
    validate_i_introduction_the_brain_as_a_multi_scale_resonant_transducer_fixture,
)


def test_i_introduction_the_brain_as_a_multi_scale_resonant_transducer_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_i_introduction_the_brain_as_a_multi_scale_resonant_transducer_fixture()
    assert result.source_ledger_span == ("P0R04462", "P0R04469")
    assert result.source_record_count == 8
    assert result.component_count == 3
    assert result.next_source_boundary == "P0R04470"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_i_introduction_the_brain_as_a_multi_scale_resonant_transducer_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R04462"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R04469"


def test_i_introduction_the_brain_as_a_multi_scale_resonant_transducer_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "i_introduction_the_brain_as_a_multi_scale_resonant_transducer",
        "ii_the_quantum_neural_interface_l1_l2",
        "1_the_neuronal_quantum_substrate_l1",
    ):
        assert (
            classify_i_introduction_the_brain_as_a_multi_scale_resonant_transducer_component(
                component
            )
            == f"{component}_source_boundary"
        )
    labels = i_introduction_the_brain_as_a_multi_scale_resonant_transducer_labels()
    assert labels["section"] == "I. Introduction: The Brain as a Multi-Scale Resonant Transducer"
    assert labels["next_boundary"] == "P0R04470"


def test_i_introduction_the_brain_as_a_multi_scale_resonant_transducer_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 8"):
        IIntroductionTheBrainAsAMultiScaleResonantTransducerConfig(expected_source_record_count=7)
    with pytest.raises(ValueError, match="expected_component_count must equal 3"):
        IIntroductionTheBrainAsAMultiScaleResonantTransducerConfig(expected_component_count=4)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R04470"):
        IIntroductionTheBrainAsAMultiScaleResonantTransducerConfig(next_source_boundary="P0R04469")
    with pytest.raises(
        ValueError,
        match="unknown i_introduction_the_brain_as_a_multi_scale_resonant_transducer component",
    ):
        classify_i_introduction_the_brain_as_a_multi_scale_resonant_transducer_component(
            "empirical_validation_claim"
        )
