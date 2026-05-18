# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Scale-Invariant Cybernetic Principle validation tests
"""Tests for Paper 0 Scale-Invariant Cybernetic Principle source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.scale_invariant_cybernetic_principle_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    ScaleInvariantCyberneticPrincipleConfig,
    classify_scale_invariant_cybernetic_principle_component,
    scale_invariant_cybernetic_principle_labels,
    validate_scale_invariant_cybernetic_principle_fixture,
)


def test_scale_invariant_cybernetic_principle_fixture_preserves_source_boundary() -> None:
    result = validate_scale_invariant_cybernetic_principle_fixture()
    assert result.source_ledger_span == ("P0R05493", "P0R05507")
    assert result.source_record_count == 15
    assert result.component_count == 2
    assert result.next_source_boundary == "P0R05508"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_scale_invariant_cybernetic_principle_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R05493"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R05507"


def test_scale_invariant_cybernetic_principle_classification_and_labels_are_explicit() -> None:
    for component in ("scale_invariant_cybernetic_principle", "citations"):
        assert (
            classify_scale_invariant_cybernetic_principle_component(component)
            == f"{component}_source_boundary"
        )
    labels = scale_invariant_cybernetic_principle_labels()
    assert labels["section"] == "Scale-Invariant Cybernetic Principle"
    assert labels["next_boundary"] == "P0R05508"


def test_scale_invariant_cybernetic_principle_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 15"):
        ScaleInvariantCyberneticPrincipleConfig(expected_source_record_count=14)
    with pytest.raises(ValueError, match="expected_component_count must equal 2"):
        ScaleInvariantCyberneticPrincipleConfig(expected_component_count=3)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R05508"):
        ScaleInvariantCyberneticPrincipleConfig(next_source_boundary="P0R05507")
    with pytest.raises(ValueError, match="unknown scale_invariant_cybernetic_principle component"):
        classify_scale_invariant_cybernetic_principle_component("empirical_validation_claim")
