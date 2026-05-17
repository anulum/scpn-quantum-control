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

from scpn_quantum_control.paper0.meta_framework_integrations_p0r02189_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    MetaFrameworkIntegrationsP0r02189Config,
    classify_meta_framework_integrations_p0r02189_component,
    meta_framework_integrations_p0r02189_labels,
    validate_meta_framework_integrations_p0r02189_fixture,
)


def test_meta_framework_integrations_p0r02189_fixture_preserves_source_boundary() -> None:
    result = validate_meta_framework_integrations_p0r02189_fixture()
    assert result.source_ledger_span == ("P0R02189", "P0R02197")
    assert result.source_record_count == 9
    assert result.component_count == 5
    assert result.next_source_boundary == "P0R02198"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_meta_framework_integrations_p0r02189_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R02189"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R02197"


def test_meta_framework_integrations_p0r02189_classification_and_labels_are_explicit() -> None:
    for component in (
        "meta_framework_integrations",
        "predictive_coding_integration",
        "policy_selection_basal_ganglia",
        "prediction_generation_cerebellum",
        "error_processing_cortex",
    ):
        assert (
            classify_meta_framework_integrations_p0r02189_component(component)
            == f"{component}_source_boundary"
        )
    labels = meta_framework_integrations_p0r02189_labels()
    assert labels["section"] == "Meta-Framework Integrations"
    assert labels["next_boundary"] == "P0R02198"


def test_meta_framework_integrations_p0r02189_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 9"):
        MetaFrameworkIntegrationsP0r02189Config(expected_source_record_count=8)
    with pytest.raises(ValueError, match="expected_component_count must equal 5"):
        MetaFrameworkIntegrationsP0r02189Config(expected_component_count=6)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R02198"):
        MetaFrameworkIntegrationsP0r02189Config(next_source_boundary="P0R02197")
    with pytest.raises(ValueError, match="unknown meta_framework_integrations_p0r02189 component"):
        classify_meta_framework_integrations_p0r02189_component("empirical_validation_claim")
