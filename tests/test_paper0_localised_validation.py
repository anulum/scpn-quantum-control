# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Localised: validation tests
"""Tests for Paper 0 Localised: source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.localised_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    LocalisedConfig,
    classify_localised_component,
    localised_labels,
    validate_localised_fixture,
)


def test_localised_fixture_preserves_source_boundary() -> None:
    result = validate_localised_fixture()
    assert result.source_ledger_span == ("P0R01820", "P0R01830")
    assert result.source_record_count == 11
    assert result.component_count == 4
    assert result.next_source_boundary == "P0R01831"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert result.problem_metadata["protocol_state"] == "source_localised_only_no_experiment"
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R01820"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R01830"


def test_localised_classification_and_labels_are_explicit() -> None:
    for component in (
        "localised",
        "persistent",
        "the_nature_of_the_interaction",
        "the_metaphysical_stance_hierarchical_field_monism_hfm",
    ):
        assert classify_localised_component(component) == f"{component}_source_boundary"
    labels = localised_labels()
    assert labels["section"] == "Localised:"
    assert labels["next_boundary"] == "P0R01831"


def test_localised_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 11"):
        LocalisedConfig(expected_source_record_count=10)
    with pytest.raises(ValueError, match="expected_component_count must equal 4"):
        LocalisedConfig(expected_component_count=5)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R01831"):
        LocalisedConfig(next_source_boundary="P0R01830")
    with pytest.raises(ValueError, match="unknown localised component"):
        classify_localised_component("empirical_validation_claim")
