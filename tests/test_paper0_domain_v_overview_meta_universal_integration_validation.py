# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Domain V Overview: Meta-Universal Integration validation tests
"""Tests for Paper 0 Domain V Overview: Meta-Universal Integration source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.domain_v_overview_meta_universal_integration_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    DomainVOverviewMetaUniversalIntegrationConfig,
    classify_domain_v_overview_meta_universal_integration_component,
    domain_v_overview_meta_universal_integration_labels,
    validate_domain_v_overview_meta_universal_integration_fixture,
)


def test_domain_v_overview_meta_universal_integration_fixture_preserves_source_boundary() -> None:
    result = validate_domain_v_overview_meta_universal_integration_fixture()
    assert result.source_ledger_span == ("P0R02367", "P0R02407")
    assert result.source_record_count == 41
    assert result.component_count == 2
    assert result.next_source_boundary == "P0R02408"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_domain_v_overview_meta_universal_integration_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R02367"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R02407"


def test_domain_v_overview_meta_universal_integration_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "domain_v_overview_meta_universal_integration",
        "the_spatial_boundary_of_the_psi_field_the_holographic_bekenstein_bound",
    ):
        assert (
            classify_domain_v_overview_meta_universal_integration_component(component)
            == f"{component}_source_boundary"
        )
    labels = domain_v_overview_meta_universal_integration_labels()
    assert labels["section"] == "Domain V Overview: Meta-Universal Integration"
    assert labels["next_boundary"] == "P0R02408"


def test_domain_v_overview_meta_universal_integration_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 41"):
        DomainVOverviewMetaUniversalIntegrationConfig(expected_source_record_count=40)
    with pytest.raises(ValueError, match="expected_component_count must equal 2"):
        DomainVOverviewMetaUniversalIntegrationConfig(expected_component_count=3)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R02408"):
        DomainVOverviewMetaUniversalIntegrationConfig(next_source_boundary="P0R02407")
    with pytest.raises(
        ValueError, match="unknown domain_v_overview_meta_universal_integration component"
    ):
        classify_domain_v_overview_meta_universal_integration_component(
            "empirical_validation_claim"
        )
