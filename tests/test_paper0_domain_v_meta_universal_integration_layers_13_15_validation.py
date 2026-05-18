# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Domain V: Meta-Universal Integration (Layers 13-15) validation tests
"""Tests for Paper 0 Domain V: Meta-Universal Integration (Layers 13-15) source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.domain_v_meta_universal_integration_layers_13_15_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    DomainVMetaUniversalIntegrationLayers1315Config,
    classify_domain_v_meta_universal_integration_layers_13_15_component,
    domain_v_meta_universal_integration_layers_13_15_labels,
    validate_domain_v_meta_universal_integration_layers_13_15_fixture,
)


def test_domain_v_meta_universal_integration_layers_13_15_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_domain_v_meta_universal_integration_layers_13_15_fixture()
    assert result.source_ledger_span == ("P0R05571", "P0R05583")
    assert result.source_record_count == 13
    assert result.component_count == 2
    assert result.next_source_boundary == "P0R05584"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_domain_v_meta_universal_integration_layers_13_15_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R05571"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R05583"


def test_domain_v_meta_universal_integration_layers_13_15_classification_and_labels_are_explicit() -> (
    None
):
    for component in ("domain_v_meta_universal_integration_layers_13_15", "citations"):
        assert (
            classify_domain_v_meta_universal_integration_layers_13_15_component(component)
            == f"{component}_source_boundary"
        )
    labels = domain_v_meta_universal_integration_layers_13_15_labels()
    assert labels["section"] == "Domain V: Meta-Universal Integration (Layers 13-15)"
    assert labels["next_boundary"] == "P0R05584"


def test_domain_v_meta_universal_integration_layers_13_15_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 13"):
        DomainVMetaUniversalIntegrationLayers1315Config(expected_source_record_count=12)
    with pytest.raises(ValueError, match="expected_component_count must equal 2"):
        DomainVMetaUniversalIntegrationLayers1315Config(expected_component_count=3)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R05584"):
        DomainVMetaUniversalIntegrationLayers1315Config(next_source_boundary="P0R05583")
    with pytest.raises(
        ValueError, match="unknown domain_v_meta_universal_integration_layers_13_15 component"
    ):
        classify_domain_v_meta_universal_integration_layers_13_15_component(
            "empirical_validation_claim"
        )
