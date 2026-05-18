# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 I. The Unified Mathematical Description (The Dynamic Fiber Bundle) validation tests
"""Tests for Paper 0 I. The Unified Mathematical Description (The Dynamic Fiber Bundle) source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.i_the_unified_mathematical_description_the_dynamic_fiber_bundle_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    ITheUnifiedMathematicalDescriptionTheDynamicFiberBundleConfig,
    classify_i_the_unified_mathematical_description_the_dynamic_fiber_bundle_component,
    i_the_unified_mathematical_description_the_dynamic_fiber_bundle_labels,
    validate_i_the_unified_mathematical_description_the_dynamic_fiber_bundle_fixture,
)


def test_i_the_unified_mathematical_description_the_dynamic_fiber_bundle_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_i_the_unified_mathematical_description_the_dynamic_fiber_bundle_fixture()
    assert result.source_ledger_span == ("P0R06115", "P0R06122")
    assert result.source_record_count == 8
    assert result.component_count == 3
    assert result.next_source_boundary == "P0R06123"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_i_the_unified_mathematical_description_the_dynamic_fiber_bundle_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R06115"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R06122"


def test_i_the_unified_mathematical_description_the_dynamic_fiber_bundle_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "i_the_unified_mathematical_description_the_dynamic_fiber_bundle",
        "ii_the_unified_phase_dynamics_equation_upde_the_spine",
        "iii_the_universal_dynamic_regime_quasicriticality",
    ):
        assert (
            classify_i_the_unified_mathematical_description_the_dynamic_fiber_bundle_component(
                component
            )
            == f"{component}_source_boundary"
        )
    labels = i_the_unified_mathematical_description_the_dynamic_fiber_bundle_labels()
    assert (
        labels["section"] == "I. The Unified Mathematical Description (The Dynamic Fiber Bundle)"
    )
    assert labels["next_boundary"] == "P0R06123"


def test_i_the_unified_mathematical_description_the_dynamic_fiber_bundle_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 8"):
        ITheUnifiedMathematicalDescriptionTheDynamicFiberBundleConfig(
            expected_source_record_count=7
        )
    with pytest.raises(ValueError, match="expected_component_count must equal 3"):
        ITheUnifiedMathematicalDescriptionTheDynamicFiberBundleConfig(expected_component_count=4)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R06123"):
        ITheUnifiedMathematicalDescriptionTheDynamicFiberBundleConfig(
            next_source_boundary="P0R06122"
        )
    with pytest.raises(
        ValueError,
        match="unknown i_the_unified_mathematical_description_the_dynamic_fiber_bundle component",
    ):
        classify_i_the_unified_mathematical_description_the_dynamic_fiber_bundle_component(
            "empirical_validation_claim"
        )
