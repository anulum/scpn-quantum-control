# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Proposed Experimental Protocol: Correlating QRNG Statistical Deviations with Collective Coherence Metrics validation tests
"""Tests for Paper 0 Proposed Experimental Protocol: Correlating QRNG Statistical Deviations with Collective Coherence Metrics source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.proposed_experimental_protocol_correlating_qrng_statistical_deviations_w_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    ProposedExperimentalProtocolCorrelatingQrngStatisticalDeviationsWConfig,
    classify_proposed_experimental_protocol_correlating_qrng_statistical_deviations_w_component,
    proposed_experimental_protocol_correlating_qrng_statistical_deviations_w_labels,
    validate_proposed_experimental_protocol_correlating_qrng_statistical_deviations_w_fixture,
)


def test_proposed_experimental_protocol_correlating_qrng_statistical_deviations_w_fixture_preserves_source_boundary() -> (
    None
):
    result = (
        validate_proposed_experimental_protocol_correlating_qrng_statistical_deviations_w_fixture()
    )
    assert result.source_ledger_span == ("P0R05217", "P0R05227")
    assert result.source_record_count == 11
    assert result.component_count == 3
    assert result.next_source_boundary == "P0R05228"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_proposed_experimental_protocol_correlating_qrng_statistical_deviations_w_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R05217"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R05227"


def test_proposed_experimental_protocol_correlating_qrng_statistical_deviations_w_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "proposed_experimental_protocol_correlating_qrng_statistical_deviations_w",
        "apparatus",
        "protocol",
    ):
        assert (
            classify_proposed_experimental_protocol_correlating_qrng_statistical_deviations_w_component(
                component
            )
            == f"{component}_source_boundary"
        )
    labels = proposed_experimental_protocol_correlating_qrng_statistical_deviations_w_labels()
    assert (
        labels["section"]
        == "Proposed Experimental Protocol: Correlating QRNG Statistical Deviations with Collective Coherence Metrics"
    )
    assert labels["next_boundary"] == "P0R05228"


def test_proposed_experimental_protocol_correlating_qrng_statistical_deviations_w_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 11"):
        ProposedExperimentalProtocolCorrelatingQrngStatisticalDeviationsWConfig(
            expected_source_record_count=10
        )
    with pytest.raises(ValueError, match="expected_component_count must equal 3"):
        ProposedExperimentalProtocolCorrelatingQrngStatisticalDeviationsWConfig(
            expected_component_count=4
        )
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R05228"):
        ProposedExperimentalProtocolCorrelatingQrngStatisticalDeviationsWConfig(
            next_source_boundary="P0R05227"
        )
    with pytest.raises(
        ValueError,
        match="unknown proposed_experimental_protocol_correlating_qrng_statistical_deviations_w component",
    ):
        classify_proposed_experimental_protocol_correlating_qrng_statistical_deviations_w_component(
            "empirical_validation_claim"
        )
