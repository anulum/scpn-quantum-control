# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Quasicriticality with MS-QEC: Two-Timescale Control and Stability Certificates (revision 11.07) validation tests
"""Tests for Paper 0 Quasicriticality with MS-QEC: Two-Timescale Control and Stability Certificates (revision 11.07) source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.quasicriticality_with_ms_qec_two_timescale_control_and_stability_certifi_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    QuasicriticalityWithMsQecTwoTimescaleControlAndStabilityCertifiConfig,
    classify_quasicriticality_with_ms_qec_two_timescale_control_and_stability_certifi_component,
    quasicriticality_with_ms_qec_two_timescale_control_and_stability_certifi_labels,
    validate_quasicriticality_with_ms_qec_two_timescale_control_and_stability_certifi_fixture,
)


def test_quasicriticality_with_ms_qec_two_timescale_control_and_stability_certifi_fixture_preserves_source_boundary() -> (
    None
):
    result = (
        validate_quasicriticality_with_ms_qec_two_timescale_control_and_stability_certifi_fixture()
    )
    assert result.source_ledger_span == ("P0R02983", "P0R02990")
    assert result.source_record_count == 8
    assert result.component_count == 3
    assert result.next_source_boundary == "P0R02991"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_quasicriticality_with_ms_qec_two_timescale_control_and_stability_certifi_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R02983"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R02990"


def test_quasicriticality_with_ms_qec_two_timescale_control_and_stability_certifi_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "quasicriticality_with_ms_qec_two_timescale_control_and_stability_certifi",
        "two_timescale_structure",
        "gain_scheduling_via_affective_field_sensitivity",
    ):
        assert (
            classify_quasicriticality_with_ms_qec_two_timescale_control_and_stability_certifi_component(
                component
            )
            == f"{component}_source_boundary"
        )
    labels = quasicriticality_with_ms_qec_two_timescale_control_and_stability_certifi_labels()
    assert (
        labels["section"]
        == "Quasicriticality with MS-QEC: Two-Timescale Control and Stability Certificates (revision 11.07)"
    )
    assert labels["next_boundary"] == "P0R02991"


def test_quasicriticality_with_ms_qec_two_timescale_control_and_stability_certifi_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 8"):
        QuasicriticalityWithMsQecTwoTimescaleControlAndStabilityCertifiConfig(
            expected_source_record_count=7
        )
    with pytest.raises(ValueError, match="expected_component_count must equal 3"):
        QuasicriticalityWithMsQecTwoTimescaleControlAndStabilityCertifiConfig(
            expected_component_count=4
        )
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R02991"):
        QuasicriticalityWithMsQecTwoTimescaleControlAndStabilityCertifiConfig(
            next_source_boundary="P0R02990"
        )
    with pytest.raises(
        ValueError,
        match="unknown quasicriticality_with_ms_qec_two_timescale_control_and_stability_certifi component",
    ):
        classify_quasicriticality_with_ms_qec_two_timescale_control_and_stability_certifi_component(
            "empirical_validation_claim"
        )
