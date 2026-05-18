# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Citations: validation tests
"""Tests for Paper 0 Citations: source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.citations_p0r05625_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    CitationsP0r05625Config,
    citations_p0r05625_labels,
    classify_citations_p0r05625_component,
    validate_citations_p0r05625_fixture,
)


def test_citations_p0r05625_fixture_preserves_source_boundary() -> None:
    result = validate_citations_p0r05625_fixture()
    assert result.source_ledger_span == ("P0R05625", "P0R05632")
    assert result.source_record_count == 8
    assert result.component_count == 1
    assert result.next_source_boundary == "P0R05633"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"] == "source_citations_p0r05625_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R05625"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R05632"


def test_citations_p0r05625_classification_and_labels_are_explicit() -> None:
    for component in ("citations",):
        assert classify_citations_p0r05625_component(component) == f"{component}_source_boundary"
    labels = citations_p0r05625_labels()
    assert labels["section"] == "Citations:"
    assert labels["next_boundary"] == "P0R05633"


def test_citations_p0r05625_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 8"):
        CitationsP0r05625Config(expected_source_record_count=7)
    with pytest.raises(ValueError, match="expected_component_count must equal 1"):
        CitationsP0r05625Config(expected_component_count=2)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R05633"):
        CitationsP0r05625Config(next_source_boundary="P0R05632")
    with pytest.raises(ValueError, match="unknown citations_p0r05625 component"):
        classify_citations_p0r05625_component("empirical_validation_claim")
