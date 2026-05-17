# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 meta-framework integrations validation tests
"""Tests for Paper 0 meta-framework integrations source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.meta_framework_integrations_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    MetaFrameworkIntegrationsConfig,
    classify_meta_framework_integrations_component,
    meta_framework_integrations_labels,
    validate_meta_framework_integrations_fixture,
)


def test_meta_framework_integrations_fixture_preserves_source_boundary() -> None:
    result = validate_meta_framework_integrations_fixture()

    assert result.source_ledger_span == ("P0R01714", "P0R01726")
    assert result.source_record_count == 13
    assert result.component_count == 3
    assert result.next_source_boundary == "P0R01727"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_meta_framework_integrations_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R01714"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R01726"


def test_meta_framework_integrations_classification_and_labels_are_explicit() -> None:
    assert (
        classify_meta_framework_integrations_component("predictive_coding_flat_prior")
        == "predictive_coding_flat_prior_hierarchy_boundary"
    )
    assert (
        classify_meta_framework_integrations_component("psi_s_field_coupling")
        == "psi_s_h_int_coupling_source_boundary"
    )
    assert (
        classify_meta_framework_integrations_component("differentiated_sigma_interface")
        == "differentiated_sigma_interface_source_boundary"
    )
    labels = meta_framework_integrations_labels()
    assert labels["section"] == "Meta-Framework Integrations"
    assert labels["coupling"] == "H_int = -lambda * Psi_s * sigma"


def test_meta_framework_integrations_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 13"):
        MetaFrameworkIntegrationsConfig(expected_source_record_count=12)
    with pytest.raises(ValueError, match="expected_component_count must equal 3"):
        MetaFrameworkIntegrationsConfig(expected_component_count=4)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R01727"):
        MetaFrameworkIntegrationsConfig(next_source_boundary="P0R01726")
    with pytest.raises(ValueError, match="unknown meta-framework integrations component"):
        classify_meta_framework_integrations_component("empirical_cosmic_inference")
