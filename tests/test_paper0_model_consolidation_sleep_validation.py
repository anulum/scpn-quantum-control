# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Model Consolidation (Sleep): validation tests
"""Tests for Paper 0 Model Consolidation (Sleep): source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.model_consolidation_sleep_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    ModelConsolidationSleepConfig,
    classify_model_consolidation_sleep_component,
    model_consolidation_sleep_labels,
    validate_model_consolidation_sleep_fixture,
)


def test_model_consolidation_sleep_fixture_preserves_source_boundary() -> None:
    result = validate_model_consolidation_sleep_fixture()
    assert result.source_ledger_span == ("P0R02198", "P0R02205")
    assert result.source_record_count == 8
    assert result.component_count == 3
    assert result.next_source_boundary == "P0R02206"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_model_consolidation_sleep_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R02198"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R02205"


def test_model_consolidation_sleep_classification_and_labels_are_explicit() -> None:
    for component in (
        "model_consolidation_sleep",
        "psis_field_coupling_integration",
        "the_collective_state_variable_sigma",
    ):
        assert (
            classify_model_consolidation_sleep_component(component)
            == f"{component}_source_boundary"
        )
    labels = model_consolidation_sleep_labels()
    assert labels["section"] == "Model Consolidation (Sleep):"
    assert labels["next_boundary"] == "P0R02206"


def test_model_consolidation_sleep_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 8"):
        ModelConsolidationSleepConfig(expected_source_record_count=7)
    with pytest.raises(ValueError, match="expected_component_count must equal 3"):
        ModelConsolidationSleepConfig(expected_component_count=4)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R02206"):
        ModelConsolidationSleepConfig(next_source_boundary="P0R02205")
    with pytest.raises(ValueError, match="unknown model_consolidation_sleep component"):
        classify_model_consolidation_sleep_component("empirical_validation_claim")
