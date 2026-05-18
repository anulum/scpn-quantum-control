# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Domain III & IV: Memory, Control, and Collective Coherence (Layers 9-12) validation tests
"""Tests for Paper 0 Domain III & IV: Memory, Control, and Collective Coherence (Layers 9-12) source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.domain_iii_iv_memory_control_and_collective_coherence_layers_9_12_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    DomainIiiIvMemoryControlAndCollectiveCoherenceLayers912Config,
    classify_domain_iii_iv_memory_control_and_collective_coherence_layers_9_12_component,
    domain_iii_iv_memory_control_and_collective_coherence_layers_9_12_labels,
    validate_domain_iii_iv_memory_control_and_collective_coherence_layers_9_12_fixture,
)


def test_domain_iii_iv_memory_control_and_collective_coherence_layers_9_12_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_domain_iii_iv_memory_control_and_collective_coherence_layers_9_12_fixture()
    assert result.source_ledger_span == ("P0R05551", "P0R05559")
    assert result.source_record_count == 9
    assert result.component_count == 1
    assert result.next_source_boundary == "P0R05560"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_domain_iii_iv_memory_control_and_collective_coherence_layers_9_12_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R05551"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R05559"


def test_domain_iii_iv_memory_control_and_collective_coherence_layers_9_12_classification_and_labels_are_explicit() -> (
    None
):
    for component in ("domain_iii_iv_memory_control_and_collective_coherence_layers_9_12",):
        assert (
            classify_domain_iii_iv_memory_control_and_collective_coherence_layers_9_12_component(
                component
            )
            == f"{component}_source_boundary"
        )
    labels = domain_iii_iv_memory_control_and_collective_coherence_layers_9_12_labels()
    assert (
        labels["section"]
        == "Domain III & IV: Memory, Control, and Collective Coherence (Layers 9-12)"
    )
    assert labels["next_boundary"] == "P0R05560"


def test_domain_iii_iv_memory_control_and_collective_coherence_layers_9_12_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 9"):
        DomainIiiIvMemoryControlAndCollectiveCoherenceLayers912Config(
            expected_source_record_count=8
        )
    with pytest.raises(ValueError, match="expected_component_count must equal 1"):
        DomainIiiIvMemoryControlAndCollectiveCoherenceLayers912Config(expected_component_count=2)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R05560"):
        DomainIiiIvMemoryControlAndCollectiveCoherenceLayers912Config(
            next_source_boundary="P0R05559"
        )
    with pytest.raises(
        ValueError,
        match="unknown domain_iii_iv_memory_control_and_collective_coherence_layers_9_12 component",
    ):
        classify_domain_iii_iv_memory_control_and_collective_coherence_layers_9_12_component(
            "empirical_validation_claim"
        )
