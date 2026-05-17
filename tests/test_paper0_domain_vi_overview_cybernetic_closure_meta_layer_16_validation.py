# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Domain VI Overview: Cybernetic Closure (Meta-Layer 16) validation tests
"""Tests for Paper 0 Domain VI Overview: Cybernetic Closure (Meta-Layer 16) source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.domain_vi_overview_cybernetic_closure_meta_layer_16_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    DomainViOverviewCyberneticClosureMetaLayer16Config,
    classify_domain_vi_overview_cybernetic_closure_meta_layer_16_component,
    domain_vi_overview_cybernetic_closure_meta_layer_16_labels,
    validate_domain_vi_overview_cybernetic_closure_meta_layer_16_fixture,
)


def test_domain_vi_overview_cybernetic_closure_meta_layer_16_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_domain_vi_overview_cybernetic_closure_meta_layer_16_fixture()
    assert result.source_ledger_span == ("P0R02408", "P0R02438")
    assert result.source_record_count == 31
    assert result.component_count == 1
    assert result.next_source_boundary == "P0R02439"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_domain_vi_overview_cybernetic_closure_meta_layer_16_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R02408"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R02438"


def test_domain_vi_overview_cybernetic_closure_meta_layer_16_classification_and_labels_are_explicit() -> (
    None
):
    for component in ("domain_vi_overview_cybernetic_closure_meta_layer_16",):
        assert (
            classify_domain_vi_overview_cybernetic_closure_meta_layer_16_component(component)
            == f"{component}_source_boundary"
        )
    labels = domain_vi_overview_cybernetic_closure_meta_layer_16_labels()
    assert labels["section"] == "Domain VI Overview: Cybernetic Closure (Meta-Layer 16)"
    assert labels["next_boundary"] == "P0R02439"


def test_domain_vi_overview_cybernetic_closure_meta_layer_16_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 31"):
        DomainViOverviewCyberneticClosureMetaLayer16Config(expected_source_record_count=30)
    with pytest.raises(ValueError, match="expected_component_count must equal 1"):
        DomainViOverviewCyberneticClosureMetaLayer16Config(expected_component_count=2)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R02439"):
        DomainViOverviewCyberneticClosureMetaLayer16Config(next_source_boundary="P0R02438")
    with pytest.raises(
        ValueError, match="unknown domain_vi_overview_cybernetic_closure_meta_layer_16 component"
    ):
        classify_domain_vi_overview_cybernetic_closure_meta_layer_16_component(
            "empirical_validation_claim"
        )
