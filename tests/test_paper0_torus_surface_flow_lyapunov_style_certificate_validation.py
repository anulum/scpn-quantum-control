# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Torus surface flow: Lyapunov-style certificate. validation tests
"""Tests for Paper 0 Torus surface flow: Lyapunov-style certificate. source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.torus_surface_flow_lyapunov_style_certificate_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    TorusSurfaceFlowLyapunovStyleCertificateConfig,
    classify_torus_surface_flow_lyapunov_style_certificate_component,
    torus_surface_flow_lyapunov_style_certificate_labels,
    validate_torus_surface_flow_lyapunov_style_certificate_fixture,
)


def test_torus_surface_flow_lyapunov_style_certificate_fixture_preserves_source_boundary() -> None:
    result = validate_torus_surface_flow_lyapunov_style_certificate_fixture()
    assert result.source_ledger_span == ("P0R02975", "P0R02982")
    assert result.source_record_count == 8
    assert result.component_count == 3
    assert result.next_source_boundary == "P0R02983"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_torus_surface_flow_lyapunov_style_certificate_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R02975"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R02982"


def test_torus_surface_flow_lyapunov_style_certificate_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "torus_surface_flow_lyapunov_style_certificate",
        "ms_qec_integration_fast_channel_realiser",
        "implementation_notes",
    ):
        assert (
            classify_torus_surface_flow_lyapunov_style_certificate_component(component)
            == f"{component}_source_boundary"
        )
    labels = torus_surface_flow_lyapunov_style_certificate_labels()
    assert labels["section"] == "Torus surface flow: Lyapunov-style certificate."
    assert labels["next_boundary"] == "P0R02983"


def test_torus_surface_flow_lyapunov_style_certificate_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 8"):
        TorusSurfaceFlowLyapunovStyleCertificateConfig(expected_source_record_count=7)
    with pytest.raises(ValueError, match="expected_component_count must equal 3"):
        TorusSurfaceFlowLyapunovStyleCertificateConfig(expected_component_count=4)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R02983"):
        TorusSurfaceFlowLyapunovStyleCertificateConfig(next_source_boundary="P0R02982")
    with pytest.raises(
        ValueError, match="unknown torus_surface_flow_lyapunov_style_certificate component"
    ):
        classify_torus_surface_flow_lyapunov_style_certificate_component(
            "empirical_validation_claim"
        )
