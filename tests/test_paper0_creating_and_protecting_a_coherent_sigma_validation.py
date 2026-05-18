# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Creating and Protecting a Coherent sigma: validation tests
"""Tests for Paper 0 Creating and Protecting a Coherent sigma: source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.creating_and_protecting_a_coherent_sigma_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    CreatingAndProtectingACoherentSigmaConfig,
    classify_creating_and_protecting_a_coherent_sigma_component,
    creating_and_protecting_a_coherent_sigma_labels,
    validate_creating_and_protecting_a_coherent_sigma_fixture,
)


def test_creating_and_protecting_a_coherent_sigma_fixture_preserves_source_boundary() -> None:
    result = validate_creating_and_protecting_a_coherent_sigma_fixture()
    assert result.source_ledger_span == ("P0R03034", "P0R03041")
    assert result.source_record_count == 8
    assert result.component_count == 3
    assert result.next_source_boundary == "P0R03042"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_creating_and_protecting_a_coherent_sigma_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R03034"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R03041"


def test_creating_and_protecting_a_coherent_sigma_classification_and_labels_are_explicit() -> None:
    for component in (
        "creating_and_protecting_a_coherent_sigma",
        "the_hierarchy_of_protection",
        "the_coherence_backbone_multi_scale_quantum_error_correction_ms_qec",
    ):
        assert (
            classify_creating_and_protecting_a_coherent_sigma_component(component)
            == f"{component}_source_boundary"
        )
    labels = creating_and_protecting_a_coherent_sigma_labels()
    assert labels["section"] == "Creating and Protecting a Coherent sigma:"
    assert labels["next_boundary"] == "P0R03042"


def test_creating_and_protecting_a_coherent_sigma_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 8"):
        CreatingAndProtectingACoherentSigmaConfig(expected_source_record_count=7)
    with pytest.raises(ValueError, match="expected_component_count must equal 3"):
        CreatingAndProtectingACoherentSigmaConfig(expected_component_count=4)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R03042"):
        CreatingAndProtectingACoherentSigmaConfig(next_source_boundary="P0R03041")
    with pytest.raises(
        ValueError, match="unknown creating_and_protecting_a_coherent_sigma component"
    ):
        classify_creating_and_protecting_a_coherent_sigma_component("empirical_validation_claim")
