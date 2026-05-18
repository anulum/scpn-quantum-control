# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 The Biological Syndrome Measurement and Recovery Protocol validation tests
"""Tests for Paper 0 The Biological Syndrome Measurement and Recovery Protocol source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.the_biological_syndrome_measurement_and_recovery_protocol_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    TheBiologicalSyndromeMeasurementAndRecoveryProtocolConfig,
    classify_the_biological_syndrome_measurement_and_recovery_protocol_component,
    the_biological_syndrome_measurement_and_recovery_protocol_labels,
    validate_the_biological_syndrome_measurement_and_recovery_protocol_fixture,
)


def test_the_biological_syndrome_measurement_and_recovery_protocol_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_the_biological_syndrome_measurement_and_recovery_protocol_fixture()
    assert result.source_ledger_span == ("P0R03076", "P0R03098")
    assert result.source_record_count == 23
    assert result.component_count == 1
    assert result.next_source_boundary == "P0R03099"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_the_biological_syndrome_measurement_and_recovery_protocol_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R03076"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R03098"


def test_the_biological_syndrome_measurement_and_recovery_protocol_classification_and_labels_are_explicit() -> (
    None
):
    for component in ("the_biological_syndrome_measurement_and_recovery_protocol",):
        assert (
            classify_the_biological_syndrome_measurement_and_recovery_protocol_component(component)
            == f"{component}_source_boundary"
        )
    labels = the_biological_syndrome_measurement_and_recovery_protocol_labels()
    assert labels["section"] == "The Biological Syndrome Measurement and Recovery Protocol"
    assert labels["next_boundary"] == "P0R03099"


def test_the_biological_syndrome_measurement_and_recovery_protocol_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 23"):
        TheBiologicalSyndromeMeasurementAndRecoveryProtocolConfig(expected_source_record_count=22)
    with pytest.raises(ValueError, match="expected_component_count must equal 1"):
        TheBiologicalSyndromeMeasurementAndRecoveryProtocolConfig(expected_component_count=2)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R03099"):
        TheBiologicalSyndromeMeasurementAndRecoveryProtocolConfig(next_source_boundary="P0R03098")
    with pytest.raises(
        ValueError,
        match="unknown the_biological_syndrome_measurement_and_recovery_protocol component",
    ):
        classify_the_biological_syndrome_measurement_and_recovery_protocol_component(
            "empirical_validation_claim"
        )
