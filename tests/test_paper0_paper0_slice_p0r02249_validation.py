# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0  validation tests
"""Tests for Paper 0  source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.paper0_slice_p0r02249_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    Paper0SliceP0r02249Config,
    classify_paper0_slice_p0r02249_component,
    paper0_slice_p0r02249_labels,
    validate_paper0_slice_p0r02249_fixture,
)


def test_paper0_slice_p0r02249_fixture_preserves_source_boundary() -> None:
    result = validate_paper0_slice_p0r02249_fixture()
    assert result.source_ledger_span == ("P0R02249", "P0R02256")
    assert result.source_record_count == 8
    assert result.component_count == 3
    assert result.next_source_boundary == "P0R02257"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_paper0_slice_p0r02249_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R02249"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R02256"


def test_paper0_slice_p0r02249_classification_and_labels_are_explicit() -> None:
    for component in ("p0r02249", "2_memory_integrity_stabiliser_transfer_lemma", "p0r02256"):
        assert (
            classify_paper0_slice_p0r02249_component(component) == f"{component}_source_boundary"
        )
    labels = paper0_slice_p0r02249_labels()
    assert labels["section"] == ""
    assert labels["next_boundary"] == "P0R02257"


def test_paper0_slice_p0r02249_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 8"):
        Paper0SliceP0r02249Config(expected_source_record_count=7)
    with pytest.raises(ValueError, match="expected_component_count must equal 3"):
        Paper0SliceP0r02249Config(expected_component_count=4)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R02257"):
        Paper0SliceP0r02249Config(next_source_boundary="P0R02256")
    with pytest.raises(ValueError, match="unknown paper0_slice_p0r02249 component"):
        classify_paper0_slice_p0r02249_component("empirical_validation_claim")
