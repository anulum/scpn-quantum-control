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

from scpn_quantum_control.paper0.meta_framework_integrations_p0r03139_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    MetaFrameworkIntegrationsP0r03139Config,
    classify_meta_framework_integrations_p0r03139_component,
    meta_framework_integrations_p0r03139_labels,
    validate_meta_framework_integrations_p0r03139_fixture,
)


def test_meta_framework_integrations_p0r03139_fixture_preserves_source_boundary() -> None:
    result = validate_meta_framework_integrations_p0r03139_fixture()
    assert result.source_ledger_span == ("P0R03139", "P0R03147")
    assert result.source_record_count == 9
    assert result.component_count == 5
    assert result.next_source_boundary == "P0R03148"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_meta_framework_integrations_p0r03139_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R03139"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R03147"


def test_meta_framework_integrations_p0r03139_classification_and_labels_are_explicit() -> None:
    for component in (
        "meta_framework_integrations",
        "predictive_coding_integration",
        "protecting_the_priors",
        "psis_field_coupling_integration",
        "ensuring_a_high_fidelity_sigma",
    ):
        assert (
            classify_meta_framework_integrations_p0r03139_component(component)
            == f"{component}_source_boundary"
        )
    labels = meta_framework_integrations_p0r03139_labels()
    assert labels["section"] == "Meta-Framework Integrations"
    assert labels["next_boundary"] == "P0R03148"


def test_meta_framework_integrations_p0r03139_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 9"):
        MetaFrameworkIntegrationsP0r03139Config(expected_source_record_count=8)
    with pytest.raises(ValueError, match="expected_component_count must equal 5"):
        MetaFrameworkIntegrationsP0r03139Config(expected_component_count=6)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R03148"):
        MetaFrameworkIntegrationsP0r03139Config(next_source_boundary="P0R03147")
    with pytest.raises(ValueError, match="unknown meta_framework_integrations_p0r03139 component"):
        classify_meta_framework_integrations_p0r03139_component("empirical_validation_claim")
