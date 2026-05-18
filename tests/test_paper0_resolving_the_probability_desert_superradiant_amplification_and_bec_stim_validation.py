# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Resolving the Probability Desert: Superradiant Amplification and BEC Stimulated Emission validation tests
"""Tests for Paper 0 Resolving the Probability Desert: Superradiant Amplification and BEC Stimulated Emission source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.resolving_the_probability_desert_superradiant_amplification_and_bec_stim_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    ResolvingTheProbabilityDesertSuperradiantAmplificationAndBecStimConfig,
    classify_resolving_the_probability_desert_superradiant_amplification_and_bec_stim_component,
    resolving_the_probability_desert_superradiant_amplification_and_bec_stim_labels,
    validate_resolving_the_probability_desert_superradiant_amplification_and_bec_stim_fixture,
)


def test_resolving_the_probability_desert_superradiant_amplification_and_bec_stim_fixture_preserves_source_boundary() -> (
    None
):
    result = (
        validate_resolving_the_probability_desert_superradiant_amplification_and_bec_stim_fixture()
    )
    assert result.source_ledger_span == ("P0R04257", "P0R04272")
    assert result.source_record_count == 16
    assert result.component_count == 1
    assert result.next_source_boundary == "P0R04273"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_resolving_the_probability_desert_superradiant_amplification_and_bec_stim_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R04257"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R04272"


def test_resolving_the_probability_desert_superradiant_amplification_and_bec_stim_classification_and_labels_are_explicit() -> (
    None
):
    for component in ("resolving_the_probability_desert_superradiant_amplification_and_bec_stim",):
        assert (
            classify_resolving_the_probability_desert_superradiant_amplification_and_bec_stim_component(
                component
            )
            == f"{component}_source_boundary"
        )
    labels = resolving_the_probability_desert_superradiant_amplification_and_bec_stim_labels()
    assert (
        labels["section"]
        == "Resolving the Probability Desert: Superradiant Amplification and BEC Stimulated Emission"
    )
    assert labels["next_boundary"] == "P0R04273"


def test_resolving_the_probability_desert_superradiant_amplification_and_bec_stim_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 16"):
        ResolvingTheProbabilityDesertSuperradiantAmplificationAndBecStimConfig(
            expected_source_record_count=15
        )
    with pytest.raises(ValueError, match="expected_component_count must equal 1"):
        ResolvingTheProbabilityDesertSuperradiantAmplificationAndBecStimConfig(
            expected_component_count=2
        )
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R04273"):
        ResolvingTheProbabilityDesertSuperradiantAmplificationAndBecStimConfig(
            next_source_boundary="P0R04272"
        )
    with pytest.raises(
        ValueError,
        match="unknown resolving_the_probability_desert_superradiant_amplification_and_bec_stim component",
    ):
        classify_resolving_the_probability_desert_superradiant_amplification_and_bec_stim_component(
            "empirical_validation_claim"
        )
