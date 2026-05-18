# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 VI. The Unified Geometric Principle (UGP) validation tests
"""Tests for Paper 0 VI. The Unified Geometric Principle (UGP) source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.vi_the_unified_geometric_principle_ugp_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    ViTheUnifiedGeometricPrincipleUgpConfig,
    classify_vi_the_unified_geometric_principle_ugp_component,
    validate_vi_the_unified_geometric_principle_ugp_fixture,
    vi_the_unified_geometric_principle_ugp_labels,
)


def test_vi_the_unified_geometric_principle_ugp_fixture_preserves_source_boundary() -> None:
    result = validate_vi_the_unified_geometric_principle_ugp_fixture()
    assert result.source_ledger_span == ("P0R06039", "P0R06046")
    assert result.source_record_count == 8
    assert result.component_count == 3
    assert result.next_source_boundary == "P0R06047"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_vi_the_unified_geometric_principle_ugp_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R06039"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R06046"


def test_vi_the_unified_geometric_principle_ugp_classification_and_labels_are_explicit() -> None:
    for component in (
        "vi_the_unified_geometric_principle_ugp",
        "vii_symmetry_principles_preservation_and_breaking",
        "viii_energetics_and_metabolism_of_the_scpn",
    ):
        assert (
            classify_vi_the_unified_geometric_principle_ugp_component(component)
            == f"{component}_source_boundary"
        )
    labels = vi_the_unified_geometric_principle_ugp_labels()
    assert labels["section"] == "VI. The Unified Geometric Principle (UGP)"
    assert labels["next_boundary"] == "P0R06047"


def test_vi_the_unified_geometric_principle_ugp_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 8"):
        ViTheUnifiedGeometricPrincipleUgpConfig(expected_source_record_count=7)
    with pytest.raises(ValueError, match="expected_component_count must equal 3"):
        ViTheUnifiedGeometricPrincipleUgpConfig(expected_component_count=4)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R06047"):
        ViTheUnifiedGeometricPrincipleUgpConfig(next_source_boundary="P0R06046")
    with pytest.raises(
        ValueError, match="unknown vi_the_unified_geometric_principle_ugp component"
    ):
        classify_vi_the_unified_geometric_principle_ugp_component("empirical_validation_claim")
