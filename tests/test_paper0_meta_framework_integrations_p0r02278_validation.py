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

from scpn_quantum_control.paper0.meta_framework_integrations_p0r02278_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    MetaFrameworkIntegrationsP0r02278Config,
    classify_meta_framework_integrations_p0r02278_component,
    meta_framework_integrations_p0r02278_labels,
    validate_meta_framework_integrations_p0r02278_fixture,
)


def test_meta_framework_integrations_p0r02278_fixture_preserves_source_boundary() -> None:
    result = validate_meta_framework_integrations_p0r02278_fixture()
    assert result.source_ledger_span == ("P0R02278", "P0R02286")
    assert result.source_record_count == 9
    assert result.component_count == 5
    assert result.next_source_boundary == "P0R02287"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_meta_framework_integrations_p0r02278_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R02278"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R02286"


def test_meta_framework_integrations_p0r02278_classification_and_labels_are_explicit() -> None:
    for component in (
        "meta_framework_integrations",
        "predictive_coding_integration",
        "layer_9_existential_holograph_as_the_deep_priors",
        "layer_10_boundary_control_as_precision_weighting_at_the_self_world_inter",
        "psis_field_coupling_integration",
    ):
        assert (
            classify_meta_framework_integrations_p0r02278_component(component)
            == f"{component}_source_boundary"
        )
    labels = meta_framework_integrations_p0r02278_labels()
    assert labels["section"] == "Meta-Framework Integrations"
    assert labels["next_boundary"] == "P0R02287"


def test_meta_framework_integrations_p0r02278_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 9"):
        MetaFrameworkIntegrationsP0r02278Config(expected_source_record_count=8)
    with pytest.raises(ValueError, match="expected_component_count must equal 5"):
        MetaFrameworkIntegrationsP0r02278Config(expected_component_count=6)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R02287"):
        MetaFrameworkIntegrationsP0r02278Config(next_source_boundary="P0R02286")
    with pytest.raises(ValueError, match="unknown meta_framework_integrations_p0r02278 component"):
        classify_meta_framework_integrations_p0r02278_component("empirical_validation_claim")
