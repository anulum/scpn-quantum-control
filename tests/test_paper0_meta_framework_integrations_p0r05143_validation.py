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

from scpn_quantum_control.paper0.meta_framework_integrations_p0r05143_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    MetaFrameworkIntegrationsP0r05143Config,
    classify_meta_framework_integrations_p0r05143_component,
    meta_framework_integrations_p0r05143_labels,
    validate_meta_framework_integrations_p0r05143_fixture,
)


def test_meta_framework_integrations_p0r05143_fixture_preserves_source_boundary() -> None:
    result = validate_meta_framework_integrations_p0r05143_fixture()
    assert result.source_ledger_span == ("P0R05143", "P0R05151")
    assert result.source_record_count == 9
    assert result.component_count == 5
    assert result.next_source_boundary == "P0R05152"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_meta_framework_integrations_p0r05143_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R05143"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R05151"


def test_meta_framework_integrations_p0r05143_classification_and_labels_are_explicit() -> None:
    for component in (
        "meta_framework_integrations",
        "predictive_coding_integration",
        "prediction_i_nv_mea_tests_the_model_s_geometry",
        "prediction_ii_qrng_tests_the_model_s_deepest_prior",
        "psis_field_coupling_integration",
    ):
        assert (
            classify_meta_framework_integrations_p0r05143_component(component)
            == f"{component}_source_boundary"
        )
    labels = meta_framework_integrations_p0r05143_labels()
    assert labels["section"] == "Meta-Framework Integrations"
    assert labels["next_boundary"] == "P0R05152"


def test_meta_framework_integrations_p0r05143_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 9"):
        MetaFrameworkIntegrationsP0r05143Config(expected_source_record_count=8)
    with pytest.raises(ValueError, match="expected_component_count must equal 5"):
        MetaFrameworkIntegrationsP0r05143Config(expected_component_count=6)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R05152"):
        MetaFrameworkIntegrationsP0r05143Config(next_source_boundary="P0R05151")
    with pytest.raises(ValueError, match="unknown meta_framework_integrations_p0r05143 component"):
        classify_meta_framework_integrations_p0r05143_component("empirical_validation_claim")
