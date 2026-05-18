# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0  Nonlinear Dynamics & Criticality validation tests
"""Tests for Paper 0  Nonlinear Dynamics & Criticality source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.nonlinear_dynamics_criticality_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    NonlinearDynamicsCriticalityConfig,
    classify_nonlinear_dynamics_criticality_component,
    nonlinear_dynamics_criticality_labels,
    validate_nonlinear_dynamics_criticality_fixture,
)


def test_nonlinear_dynamics_criticality_fixture_preserves_source_boundary() -> None:
    result = validate_nonlinear_dynamics_criticality_fixture()
    assert result.source_ledger_span == ("P0R05730", "P0R05737")
    assert result.source_record_count == 8
    assert result.component_count == 2
    assert result.next_source_boundary == "P0R05738"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_nonlinear_dynamics_criticality_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R05730"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R05737"


def test_nonlinear_dynamics_criticality_classification_and_labels_are_explicit() -> None:
    for component in ("nonlinear_dynamics_criticality", "computational_complexity_science"):
        assert (
            classify_nonlinear_dynamics_criticality_component(component)
            == f"{component}_source_boundary"
        )
    labels = nonlinear_dynamics_criticality_labels()
    assert labels["section"] == " Nonlinear Dynamics & Criticality"
    assert labels["next_boundary"] == "P0R05738"


def test_nonlinear_dynamics_criticality_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 8"):
        NonlinearDynamicsCriticalityConfig(expected_source_record_count=7)
    with pytest.raises(ValueError, match="expected_component_count must equal 2"):
        NonlinearDynamicsCriticalityConfig(expected_component_count=3)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R05738"):
        NonlinearDynamicsCriticalityConfig(next_source_boundary="P0R05737")
    with pytest.raises(ValueError, match="unknown nonlinear_dynamics_criticality component"):
        classify_nonlinear_dynamics_criticality_component("empirical_validation_claim")
