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

from scpn_quantum_control.paper0.paper0_slice_p0r04247_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    Paper0SliceP0r04247Config,
    classify_paper0_slice_p0r04247_component,
    paper0_slice_p0r04247_labels,
    validate_paper0_slice_p0r04247_fixture,
)


def test_paper0_slice_p0r04247_fixture_preserves_source_boundary() -> None:
    result = validate_paper0_slice_p0r04247_fixture()
    assert result.source_ledger_span == ("P0R04247", "P0R04256")
    assert result.source_record_count == 10
    assert result.component_count == 3
    assert result.next_source_boundary == "P0R04257"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_paper0_slice_p0r04247_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R04247"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R04256"


def test_paper0_slice_p0r04247_classification_and_labels_are_explicit() -> None:
    for component in (
        "p0r04247",
        "5_1_the_em_interface_an_alp_mediated_bridge",
        "an_alp_mediated_bridge",
    ):
        assert (
            classify_paper0_slice_p0r04247_component(component) == f"{component}_source_boundary"
        )
    labels = paper0_slice_p0r04247_labels()
    assert labels["section"] == ""
    assert labels["next_boundary"] == "P0R04257"


def test_paper0_slice_p0r04247_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 10"):
        Paper0SliceP0r04247Config(expected_source_record_count=9)
    with pytest.raises(ValueError, match="expected_component_count must equal 3"):
        Paper0SliceP0r04247Config(expected_component_count=4)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R04257"):
        Paper0SliceP0r04247Config(next_source_boundary="P0R04256")
    with pytest.raises(ValueError, match="unknown paper0_slice_p0r04247 component"):
        classify_paper0_slice_p0r04247_component("empirical_validation_claim")
