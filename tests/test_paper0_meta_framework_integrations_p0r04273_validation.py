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

from scpn_quantum_control.paper0.meta_framework_integrations_p0r04273_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    MetaFrameworkIntegrationsP0r04273Config,
    classify_meta_framework_integrations_p0r04273_component,
    meta_framework_integrations_p0r04273_labels,
    validate_meta_framework_integrations_p0r04273_fixture,
)


def test_meta_framework_integrations_p0r04273_fixture_preserves_source_boundary() -> None:
    result = validate_meta_framework_integrations_p0r04273_fixture()
    assert result.source_ledger_span == ("P0R04273", "P0R04282")
    assert result.source_record_count == 10
    assert result.component_count == 5
    assert result.next_source_boundary == "P0R04283"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_meta_framework_integrations_p0r04273_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R04273"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R04282"


def test_meta_framework_integrations_p0r04273_classification_and_labels_are_explicit() -> None:
    for component in (
        "meta_framework_integrations",
        "predictive_coding_integration",
        "downward_causation_psi_em_is_the_write_operation",
        "upward_causation_em_psi_is_the_read_operation",
        "psis_field_coupling_integration",
    ):
        assert (
            classify_meta_framework_integrations_p0r04273_component(component)
            == f"{component}_source_boundary"
        )
    labels = meta_framework_integrations_p0r04273_labels()
    assert labels["section"] == "Meta-Framework Integrations"
    assert labels["next_boundary"] == "P0R04283"


def test_meta_framework_integrations_p0r04273_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 10"):
        MetaFrameworkIntegrationsP0r04273Config(expected_source_record_count=9)
    with pytest.raises(ValueError, match="expected_component_count must equal 5"):
        MetaFrameworkIntegrationsP0r04273Config(expected_component_count=6)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R04283"):
        MetaFrameworkIntegrationsP0r04273Config(next_source_boundary="P0R04282")
    with pytest.raises(ValueError, match="unknown meta_framework_integrations_p0r04273 component"):
        classify_meta_framework_integrations_p0r04273_component("empirical_validation_claim")
