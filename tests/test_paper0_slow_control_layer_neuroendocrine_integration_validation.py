# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Slow Control Layer: Neuroendocrine Integration validation tests
"""Tests for Paper 0 Slow Control Layer: Neuroendocrine Integration source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.slow_control_layer_neuroendocrine_integration_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    SlowControlLayerNeuroendocrineIntegrationConfig,
    classify_slow_control_layer_neuroendocrine_integration_component,
    slow_control_layer_neuroendocrine_integration_labels,
    validate_slow_control_layer_neuroendocrine_integration_fixture,
)


def test_slow_control_layer_neuroendocrine_integration_fixture_preserves_source_boundary() -> None:
    result = validate_slow_control_layer_neuroendocrine_integration_fixture()
    assert result.source_ledger_span == ("P0R05430", "P0R05444")
    assert result.source_record_count == 15
    assert result.component_count == 2
    assert result.next_source_boundary == "P0R05445"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_slow_control_layer_neuroendocrine_integration_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R05430"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R05444"


def test_slow_control_layer_neuroendocrine_integration_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "slow_control_layer_neuroendocrine_integration",
        "hormonal_modulation_as_a_third_control_layer",
    ):
        assert (
            classify_slow_control_layer_neuroendocrine_integration_component(component)
            == f"{component}_source_boundary"
        )
    labels = slow_control_layer_neuroendocrine_integration_labels()
    assert labels["section"] == "Slow Control Layer: Neuroendocrine Integration"
    assert labels["next_boundary"] == "P0R05445"


def test_slow_control_layer_neuroendocrine_integration_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 15"):
        SlowControlLayerNeuroendocrineIntegrationConfig(expected_source_record_count=14)
    with pytest.raises(ValueError, match="expected_component_count must equal 2"):
        SlowControlLayerNeuroendocrineIntegrationConfig(expected_component_count=3)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R05445"):
        SlowControlLayerNeuroendocrineIntegrationConfig(next_source_boundary="P0R05444")
    with pytest.raises(
        ValueError, match="unknown slow_control_layer_neuroendocrine_integration component"
    ):
        classify_slow_control_layer_neuroendocrine_integration_component(
            "empirical_validation_claim"
        )
