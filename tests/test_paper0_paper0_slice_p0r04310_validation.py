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

from scpn_quantum_control.paper0.paper0_slice_p0r04310_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    Paper0SliceP0r04310Config,
    classify_paper0_slice_p0r04310_component,
    paper0_slice_p0r04310_labels,
    validate_paper0_slice_p0r04310_fixture,
)


def test_paper0_slice_p0r04310_fixture_preserves_source_boundary() -> None:
    result = validate_paper0_slice_p0r04310_fixture()
    assert result.source_ledger_span == ("P0R04310", "P0R04321")
    assert result.source_record_count == 12
    assert result.component_count == 2
    assert result.next_source_boundary == "P0R04322"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_paper0_slice_p0r04310_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R04310"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R04321"


def test_paper0_slice_p0r04310_classification_and_labels_are_explicit() -> None:
    for component in ("p0r04310", "the_two_scalar_sector_and_the_pseudoscalar_coupling"):
        assert (
            classify_paper0_slice_p0r04310_component(component) == f"{component}_source_boundary"
        )
    labels = paper0_slice_p0r04310_labels()
    assert labels["section"] == ""
    assert labels["next_boundary"] == "P0R04322"


def test_paper0_slice_p0r04310_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 12"):
        Paper0SliceP0r04310Config(expected_source_record_count=11)
    with pytest.raises(ValueError, match="expected_component_count must equal 2"):
        Paper0SliceP0r04310Config(expected_component_count=3)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R04322"):
        Paper0SliceP0r04310Config(next_source_boundary="P0R04321")
    with pytest.raises(ValueError, match="unknown paper0_slice_p0r04310 component"):
        classify_paper0_slice_p0r04310_component("empirical_validation_claim")
