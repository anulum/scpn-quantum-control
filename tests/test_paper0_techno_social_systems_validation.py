# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0  Techno-Social Systems validation tests
"""Tests for Paper 0  Techno-Social Systems source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.techno_social_systems_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    TechnoSocialSystemsConfig,
    classify_techno_social_systems_component,
    techno_social_systems_labels,
    validate_techno_social_systems_fixture,
)


def test_techno_social_systems_fixture_preserves_source_boundary() -> None:
    result = validate_techno_social_systems_fixture()
    assert result.source_ledger_span == ("P0R05868", "P0R05875")
    assert result.source_record_count == 8
    assert result.component_count == 2
    assert result.next_source_boundary == "P0R05876"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_techno_social_systems_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R05868"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R05875"


def test_techno_social_systems_classification_and_labels_are_explicit() -> None:
    for component in ("techno_social_systems", "neurophenomenology_first_person_science"):
        assert (
            classify_techno_social_systems_component(component) == f"{component}_source_boundary"
        )
    labels = techno_social_systems_labels()
    assert labels["section"] == " Techno-Social Systems"
    assert labels["next_boundary"] == "P0R05876"


def test_techno_social_systems_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 8"):
        TechnoSocialSystemsConfig(expected_source_record_count=7)
    with pytest.raises(ValueError, match="expected_component_count must equal 2"):
        TechnoSocialSystemsConfig(expected_component_count=3)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R05876"):
        TechnoSocialSystemsConfig(next_source_boundary="P0R05875")
    with pytest.raises(ValueError, match="unknown techno_social_systems component"):
        classify_techno_social_systems_component("empirical_validation_claim")
