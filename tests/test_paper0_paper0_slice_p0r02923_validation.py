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

from scpn_quantum_control.paper0.paper0_slice_p0r02923_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    Paper0SliceP0r02923Config,
    classify_paper0_slice_p0r02923_component,
    paper0_slice_p0r02923_labels,
    validate_paper0_slice_p0r02923_fixture,
)


def test_paper0_slice_p0r02923_fixture_preserves_source_boundary() -> None:
    result = validate_paper0_slice_p0r02923_fixture()
    assert result.source_ledger_span == ("P0R02923", "P0R02930")
    assert result.source_record_count == 8
    assert result.component_count == 8
    assert result.next_source_boundary == "P0R02931"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_paper0_slice_p0r02923_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R02923"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R02930"


def test_paper0_slice_p0r02923_classification_and_labels_are_explicit() -> None:
    for component in (
        "p0r02923",
        "p0r02924",
        "p0r02925",
        "p0r02926",
        "p0r02927",
        "p0r02928",
        "p0r02929",
        "p0r02930",
    ):
        assert (
            classify_paper0_slice_p0r02923_component(component) == f"{component}_source_boundary"
        )
    labels = paper0_slice_p0r02923_labels()
    assert labels["section"] == ""
    assert labels["next_boundary"] == "P0R02931"


def test_paper0_slice_p0r02923_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 8"):
        Paper0SliceP0r02923Config(expected_source_record_count=7)
    with pytest.raises(ValueError, match="expected_component_count must equal 8"):
        Paper0SliceP0r02923Config(expected_component_count=9)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R02931"):
        Paper0SliceP0r02923Config(next_source_boundary="P0R02930")
    with pytest.raises(ValueError, match="unknown paper0_slice_p0r02923 component"):
        classify_paper0_slice_p0r02923_component("empirical_validation_claim")
