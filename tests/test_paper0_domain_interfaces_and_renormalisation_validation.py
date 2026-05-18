# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Domain Interfaces and Renormalisation validation tests
"""Tests for Paper 0 Domain Interfaces and Renormalisation source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.domain_interfaces_and_renormalisation_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    DomainInterfacesAndRenormalisationConfig,
    classify_domain_interfaces_and_renormalisation_component,
    domain_interfaces_and_renormalisation_labels,
    validate_domain_interfaces_and_renormalisation_fixture,
)


def test_domain_interfaces_and_renormalisation_fixture_preserves_source_boundary() -> None:
    result = validate_domain_interfaces_and_renormalisation_fixture()
    assert result.source_ledger_span == ("P0R05633", "P0R05640")
    assert result.source_record_count == 8
    assert result.component_count == 2
    assert result.next_source_boundary == "P0R05641"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_domain_interfaces_and_renormalisation_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R05633"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R05640"


def test_domain_interfaces_and_renormalisation_classification_and_labels_are_explicit() -> None:
    for component in (
        "domain_interfaces_and_renormalisation",
        "1_renormalisation_group_rg_flow_across_domains",
    ):
        assert (
            classify_domain_interfaces_and_renormalisation_component(component)
            == f"{component}_source_boundary"
        )
    labels = domain_interfaces_and_renormalisation_labels()
    assert labels["section"] == "Domain Interfaces and Renormalisation"
    assert labels["next_boundary"] == "P0R05641"


def test_domain_interfaces_and_renormalisation_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 8"):
        DomainInterfacesAndRenormalisationConfig(expected_source_record_count=7)
    with pytest.raises(ValueError, match="expected_component_count must equal 2"):
        DomainInterfacesAndRenormalisationConfig(expected_component_count=3)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R05641"):
        DomainInterfacesAndRenormalisationConfig(next_source_boundary="P0R05640")
    with pytest.raises(
        ValueError, match="unknown domain_interfaces_and_renormalisation component"
    ):
        classify_domain_interfaces_and_renormalisation_component("empirical_validation_claim")
