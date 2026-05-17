# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Meta-Framework Integrations validation tests
"""Tests for Paper 0 Meta-Framework Integrations source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.meta_framework_integrations_p0r01803_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    MetaFrameworkIntegrationsP0r01803Config,
    classify_meta_framework_integrations_p0r01803_component,
    meta_framework_integrations_p0r01803_labels,
    validate_meta_framework_integrations_p0r01803_fixture,
)


def test_meta_framework_integrations_p0r01803_fixture_preserves_source_boundary() -> None:
    result = validate_meta_framework_integrations_p0r01803_fixture()
    assert result.source_ledger_span == ("P0R01803", "P0R01811")
    assert result.source_record_count == 9
    assert result.component_count == 5
    assert result.next_source_boundary == "P0R01812"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_meta_framework_integrations_p0r01803_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R01803"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R01811"


def test_meta_framework_integrations_p0r01803_classification_and_labels_are_explicit() -> None:
    for component in (
        "meta_framework_integrations",
        "predictive_coding_integration",
        "the_soliton_as_the_physical_prior",
        "charge_conservation_as_model_integrity",
        "psis_field_coupling_integration",
    ):
        assert (
            classify_meta_framework_integrations_p0r01803_component(component)
            == f"{component}_source_boundary"
        )
    labels = meta_framework_integrations_p0r01803_labels()
    assert labels["section"] == "Meta-Framework Integrations"
    assert labels["next_boundary"] == "P0R01812"


def test_meta_framework_integrations_p0r01803_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 9"):
        MetaFrameworkIntegrationsP0r01803Config(expected_source_record_count=8)
    with pytest.raises(ValueError, match="expected_component_count must equal 5"):
        MetaFrameworkIntegrationsP0r01803Config(expected_component_count=6)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R01812"):
        MetaFrameworkIntegrationsP0r01803Config(next_source_boundary="P0R01811")
    with pytest.raises(ValueError, match="unknown meta_framework_integrations_p0r01803 component"):
        classify_meta_framework_integrations_p0r01803_component("empirical_validation_claim")
