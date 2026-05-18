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

from scpn_quantum_control.paper0.meta_framework_integrations_p0r02600_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    MetaFrameworkIntegrationsP0r02600Config,
    classify_meta_framework_integrations_p0r02600_component,
    meta_framework_integrations_p0r02600_labels,
    validate_meta_framework_integrations_p0r02600_fixture,
)


def test_meta_framework_integrations_p0r02600_fixture_preserves_source_boundary() -> None:
    result = validate_meta_framework_integrations_p0r02600_fixture()
    assert result.source_ledger_span == ("P0R02600", "P0R02607")
    assert result.source_record_count == 8
    assert result.component_count == 4
    assert result.next_source_boundary == "P0R02608"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_meta_framework_integrations_p0r02600_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R02600"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R02607"


def test_meta_framework_integrations_p0r02600_classification_and_labels_are_explicit() -> None:
    for component in (
        "meta_framework_integrations",
        "predictive_coding_integration",
        "phase_coherence_as_prediction",
        "phase_slips_as_prediction_error",
    ):
        assert (
            classify_meta_framework_integrations_p0r02600_component(component)
            == f"{component}_source_boundary"
        )
    labels = meta_framework_integrations_p0r02600_labels()
    assert labels["section"] == "Meta-Framework Integrations"
    assert labels["next_boundary"] == "P0R02608"


def test_meta_framework_integrations_p0r02600_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 8"):
        MetaFrameworkIntegrationsP0r02600Config(expected_source_record_count=7)
    with pytest.raises(ValueError, match="expected_component_count must equal 4"):
        MetaFrameworkIntegrationsP0r02600Config(expected_component_count=5)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R02608"):
        MetaFrameworkIntegrationsP0r02600Config(next_source_boundary="P0R02607")
    with pytest.raises(ValueError, match="unknown meta_framework_integrations_p0r02600 component"):
        classify_meta_framework_integrations_p0r02600_component("empirical_validation_claim")
