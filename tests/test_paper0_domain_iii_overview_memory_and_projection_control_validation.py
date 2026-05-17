# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Domain III Overview: Memory and Projection Control validation tests
"""Tests for Paper 0 Domain III Overview: Memory and Projection Control source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.domain_iii_overview_memory_and_projection_control_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    DomainIiiOverviewMemoryAndProjectionControlConfig,
    classify_domain_iii_overview_memory_and_projection_control_component,
    domain_iii_overview_memory_and_projection_control_labels,
    validate_domain_iii_overview_memory_and_projection_control_fixture,
)


def test_domain_iii_overview_memory_and_projection_control_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_domain_iii_overview_memory_and_projection_control_fixture()
    assert result.source_ledger_span == ("P0R02237", "P0R02248")
    assert result.source_record_count == 12
    assert result.component_count == 3
    assert result.next_source_boundary == "P0R02249"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_domain_iii_overview_memory_and_projection_control_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R02237"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R02248"


def test_domain_iii_overview_memory_and_projection_control_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "domain_iii_overview_memory_and_projection_control",
        "p0r02242",
        "1_memory_retrieval_retrocausality_via_abl_rule",
    ):
        assert (
            classify_domain_iii_overview_memory_and_projection_control_component(component)
            == f"{component}_source_boundary"
        )
    labels = domain_iii_overview_memory_and_projection_control_labels()
    assert labels["section"] == "Domain III Overview: Memory and Projection Control"
    assert labels["next_boundary"] == "P0R02249"


def test_domain_iii_overview_memory_and_projection_control_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 12"):
        DomainIiiOverviewMemoryAndProjectionControlConfig(expected_source_record_count=11)
    with pytest.raises(ValueError, match="expected_component_count must equal 3"):
        DomainIiiOverviewMemoryAndProjectionControlConfig(expected_component_count=4)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R02249"):
        DomainIiiOverviewMemoryAndProjectionControlConfig(next_source_boundary="P0R02248")
    with pytest.raises(
        ValueError, match="unknown domain_iii_overview_memory_and_projection_control component"
    ):
        classify_domain_iii_overview_memory_and_projection_control_component(
            "empirical_validation_claim"
        )
