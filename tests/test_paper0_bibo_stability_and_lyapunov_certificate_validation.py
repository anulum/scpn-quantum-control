# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 BIBO Stability and Lyapunov Certificate: validation tests
"""Tests for Paper 0 BIBO Stability and Lyapunov Certificate: source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.bibo_stability_and_lyapunov_certificate_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    BiboStabilityAndLyapunovCertificateConfig,
    bibo_stability_and_lyapunov_certificate_labels,
    classify_bibo_stability_and_lyapunov_certificate_component,
    validate_bibo_stability_and_lyapunov_certificate_fixture,
)


def test_bibo_stability_and_lyapunov_certificate_fixture_preserves_source_boundary() -> None:
    result = validate_bibo_stability_and_lyapunov_certificate_fixture()
    assert result.source_ledger_span == ("P0R02991", "P0R03009")
    assert result.source_record_count == 19
    assert result.component_count == 1
    assert result.next_source_boundary == "P0R03010"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_bibo_stability_and_lyapunov_certificate_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R02991"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R03009"


def test_bibo_stability_and_lyapunov_certificate_classification_and_labels_are_explicit() -> None:
    for component in ("bibo_stability_and_lyapunov_certificate",):
        assert (
            classify_bibo_stability_and_lyapunov_certificate_component(component)
            == f"{component}_source_boundary"
        )
    labels = bibo_stability_and_lyapunov_certificate_labels()
    assert labels["section"] == "BIBO Stability and Lyapunov Certificate:"
    assert labels["next_boundary"] == "P0R03010"


def test_bibo_stability_and_lyapunov_certificate_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 19"):
        BiboStabilityAndLyapunovCertificateConfig(expected_source_record_count=18)
    with pytest.raises(ValueError, match="expected_component_count must equal 1"):
        BiboStabilityAndLyapunovCertificateConfig(expected_component_count=2)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R03010"):
        BiboStabilityAndLyapunovCertificateConfig(next_source_boundary="P0R03009")
    with pytest.raises(
        ValueError, match="unknown bibo_stability_and_lyapunov_certificate component"
    ):
        classify_bibo_stability_and_lyapunov_certificate_component("empirical_validation_claim")
