# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Resolving the Erasure Paradox: Lossy Compression and the Heat Sink Boundary validation tests
"""Tests for Paper 0 Resolving the Erasure Paradox: Lossy Compression and the Heat Sink Boundary source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.resolving_the_erasure_paradox_lossy_compression_and_the_heat_sink_bounda_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    ResolvingTheErasureParadoxLossyCompressionAndTheHeatSinkBoundaConfig,
    classify_resolving_the_erasure_paradox_lossy_compression_and_the_heat_sink_bounda_component,
    resolving_the_erasure_paradox_lossy_compression_and_the_heat_sink_bounda_labels,
    validate_resolving_the_erasure_paradox_lossy_compression_and_the_heat_sink_bounda_fixture,
)


def test_resolving_the_erasure_paradox_lossy_compression_and_the_heat_sink_bounda_fixture_preserves_source_boundary() -> (
    None
):
    result = (
        validate_resolving_the_erasure_paradox_lossy_compression_and_the_heat_sink_bounda_fixture()
    )
    assert result.source_ledger_span == ("P0R05964", "P0R05985")
    assert result.source_record_count == 22
    assert result.component_count == 1
    assert result.next_source_boundary == "P0R05986"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_resolving_the_erasure_paradox_lossy_compression_and_the_heat_sink_bounda_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R05964"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R05985"


def test_resolving_the_erasure_paradox_lossy_compression_and_the_heat_sink_bounda_classification_and_labels_are_explicit() -> (
    None
):
    for component in ("resolving_the_erasure_paradox_lossy_compression_and_the_heat_sink_bounda",):
        assert (
            classify_resolving_the_erasure_paradox_lossy_compression_and_the_heat_sink_bounda_component(
                component
            )
            == f"{component}_source_boundary"
        )
    labels = resolving_the_erasure_paradox_lossy_compression_and_the_heat_sink_bounda_labels()
    assert (
        labels["section"]
        == "Resolving the Erasure Paradox: Lossy Compression and the Heat Sink Boundary"
    )
    assert labels["next_boundary"] == "P0R05986"


def test_resolving_the_erasure_paradox_lossy_compression_and_the_heat_sink_bounda_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 22"):
        ResolvingTheErasureParadoxLossyCompressionAndTheHeatSinkBoundaConfig(
            expected_source_record_count=21
        )
    with pytest.raises(ValueError, match="expected_component_count must equal 1"):
        ResolvingTheErasureParadoxLossyCompressionAndTheHeatSinkBoundaConfig(
            expected_component_count=2
        )
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R05986"):
        ResolvingTheErasureParadoxLossyCompressionAndTheHeatSinkBoundaConfig(
            next_source_boundary="P0R05985"
        )
    with pytest.raises(
        ValueError,
        match="unknown resolving_the_erasure_paradox_lossy_compression_and_the_heat_sink_bounda component",
    ):
        classify_resolving_the_erasure_paradox_lossy_compression_and_the_heat_sink_bounda_component(
            "empirical_validation_claim"
        )
