# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 SCPN-IIT Correspondence: validation tests
"""Tests for Paper 0 SCPN-IIT Correspondence: source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.scpn_iit_correspondence_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    ScpnIitCorrespondenceConfig,
    classify_scpn_iit_correspondence_component,
    scpn_iit_correspondence_labels,
    validate_scpn_iit_correspondence_fixture,
)


def test_scpn_iit_correspondence_fixture_preserves_source_boundary() -> None:
    result = validate_scpn_iit_correspondence_fixture()
    assert result.source_ledger_span == ("P0R03555", "P0R03563")
    assert result.source_record_count == 9
    assert result.component_count == 1
    assert result.next_source_boundary == "P0R03564"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_scpn_iit_correspondence_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R03555"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R03563"


def test_scpn_iit_correspondence_classification_and_labels_are_explicit() -> None:
    for component in ("scpn_iit_correspondence",):
        assert (
            classify_scpn_iit_correspondence_component(component) == f"{component}_source_boundary"
        )
    labels = scpn_iit_correspondence_labels()
    assert labels["section"] == "SCPN-IIT Correspondence:"
    assert labels["next_boundary"] == "P0R03564"


def test_scpn_iit_correspondence_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 9"):
        ScpnIitCorrespondenceConfig(expected_source_record_count=8)
    with pytest.raises(ValueError, match="expected_component_count must equal 1"):
        ScpnIitCorrespondenceConfig(expected_component_count=2)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R03564"):
        ScpnIitCorrespondenceConfig(next_source_boundary="P0R03563")
    with pytest.raises(ValueError, match="unknown scpn_iit_correspondence component"):
        classify_scpn_iit_correspondence_component("empirical_validation_claim")
