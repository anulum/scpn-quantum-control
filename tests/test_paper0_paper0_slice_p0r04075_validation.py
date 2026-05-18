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

from scpn_quantum_control.paper0.paper0_slice_p0r04075_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    Paper0SliceP0r04075Config,
    classify_paper0_slice_p0r04075_component,
    paper0_slice_p0r04075_labels,
    validate_paper0_slice_p0r04075_fixture,
)


def test_paper0_slice_p0r04075_fixture_preserves_source_boundary() -> None:
    result = validate_paper0_slice_p0r04075_fixture()
    assert result.source_ledger_span == ("P0R04075", "P0R04088")
    assert result.source_record_count == 14
    assert result.component_count == 2
    assert result.next_source_boundary == "P0R04089"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_paper0_slice_p0r04075_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R04075"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R04088"


def test_paper0_slice_p0r04075_classification_and_labels_are_explicit() -> None:
    for component in (
        "p0r04075",
        "resolving_the_attractor_freeze_paradox_the_strange_attractor_of_sec",
    ):
        assert (
            classify_paper0_slice_p0r04075_component(component) == f"{component}_source_boundary"
        )
    labels = paper0_slice_p0r04075_labels()
    assert labels["section"] == ""
    assert labels["next_boundary"] == "P0R04089"


def test_paper0_slice_p0r04075_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 14"):
        Paper0SliceP0r04075Config(expected_source_record_count=13)
    with pytest.raises(ValueError, match="expected_component_count must equal 2"):
        Paper0SliceP0r04075Config(expected_component_count=3)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R04089"):
        Paper0SliceP0r04075Config(next_source_boundary="P0R04088")
    with pytest.raises(ValueError, match="unknown paper0_slice_p0r04075 component"):
        classify_paper0_slice_p0r04075_component("empirical_validation_claim")
