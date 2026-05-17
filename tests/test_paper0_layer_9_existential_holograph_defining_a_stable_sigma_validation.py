# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Layer 9 (Existential Holograph) - Defining a Stable sigma: validation tests
"""Tests for Paper 0 Layer 9 (Existential Holograph) - Defining a Stable sigma: source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.layer_9_existential_holograph_defining_a_stable_sigma_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    Layer9ExistentialHolographDefiningAStableSigmaConfig,
    classify_layer_9_existential_holograph_defining_a_stable_sigma_component,
    layer_9_existential_holograph_defining_a_stable_sigma_labels,
    validate_layer_9_existential_holograph_defining_a_stable_sigma_fixture,
)


def test_layer_9_existential_holograph_defining_a_stable_sigma_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_layer_9_existential_holograph_defining_a_stable_sigma_fixture()
    assert result.source_ledger_span == ("P0R02287", "P0R02305")
    assert result.source_record_count == 19
    assert result.component_count == 3
    assert result.next_source_boundary == "P0R02306"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_layer_9_existential_holograph_defining_a_stable_sigma_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R02287"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R02305"


def test_layer_9_existential_holograph_defining_a_stable_sigma_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "layer_9_existential_holograph_defining_a_stable_sigma",
        "layer_10_boundary_control_modulating_the_coupling_constant_lambda",
        "case_study_the_layer_11_noospheric_spin_glass_system",
    ):
        assert (
            classify_layer_9_existential_holograph_defining_a_stable_sigma_component(component)
            == f"{component}_source_boundary"
        )
    labels = layer_9_existential_holograph_defining_a_stable_sigma_labels()
    assert labels["section"] == "Layer 9 (Existential Holograph) - Defining a Stable sigma:"
    assert labels["next_boundary"] == "P0R02306"


def test_layer_9_existential_holograph_defining_a_stable_sigma_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 19"):
        Layer9ExistentialHolographDefiningAStableSigmaConfig(expected_source_record_count=18)
    with pytest.raises(ValueError, match="expected_component_count must equal 3"):
        Layer9ExistentialHolographDefiningAStableSigmaConfig(expected_component_count=4)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R02306"):
        Layer9ExistentialHolographDefiningAStableSigmaConfig(next_source_boundary="P0R02305")
    with pytest.raises(
        ValueError, match="unknown layer_9_existential_holograph_defining_a_stable_sigma component"
    ):
        classify_layer_9_existential_holograph_defining_a_stable_sigma_component(
            "empirical_validation_claim"
        )
