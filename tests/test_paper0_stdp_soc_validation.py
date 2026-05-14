# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 STDP/SOC fixture tests
"""Tests for Paper 0 STDP/SOC simulator fixtures."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.stdp_soc_validation import (
    STDPSOCConfig,
    avalanche_power_law_density,
    branching_parameter_relaxation_derivative,
    stdp_weight_update,
    validate_stdp_soc_fixture,
)


def test_stdp_weight_update_preserves_ltp_and_ltd_signs() -> None:
    ltp = stdp_weight_update(delta_t=0.012, amplitude_ltp=0.08, amplitude_ltd=0.07)
    ltd = stdp_weight_update(delta_t=-0.012, amplitude_ltp=0.08, amplitude_ltd=0.07)

    assert ltp > 0.0
    assert ltd < 0.0
    assert abs(ltp) > abs(ltd)


def test_power_law_density_decreases_and_rejects_invalid_parameters() -> None:
    small = avalanche_power_law_density(size=4.0, tau=1.5)
    large = avalanche_power_law_density(size=16.0, tau=1.5)

    assert small > large
    with pytest.raises(ValueError, match="size must be finite and positive"):
        avalanche_power_law_density(size=0.0, tau=1.5)
    with pytest.raises(ValueError, match="tau must be finite and positive"):
        avalanche_power_law_density(size=4.0, tau=0.0)


def test_branching_relaxation_drives_sigma_towards_one() -> None:
    above = branching_parameter_relaxation_derivative(sigma_l=1.2, kappa_l=0.4, eta_l=0.0)
    below = branching_parameter_relaxation_derivative(sigma_l=0.8, kappa_l=0.4, eta_l=0.0)

    assert above < 0.0
    assert below > 0.0


def test_stdp_soc_fixture_preserves_boundaries_and_controls() -> None:
    with pytest.raises(ValueError, match="criticality_threshold must be finite and positive"):
        STDPSOCConfig(criticality_threshold=0.0)

    result = validate_stdp_soc_fixture()

    assert result.spec_keys == (
        "stdp_soc.asymmetric_learning_window",
        "stdp_soc.avalanche_power_law_signature",
        "stdp_soc.quasicritical_relaxation_mapping",
        "stdp_soc.l4_microscopic_engine_boundary",
    )
    assert result.hardware_status == "simulator_only_no_provider_submission"
    assert result.source_ledger_span == ("P0R06402", "P0R06413")
    assert result.ltp_update > 0.0
    assert result.ltd_update < 0.0
    assert result.relaxation_above_critical < 0.0
    assert result.relaxation_below_critical > 0.0
    assert result.power_law_ratio > result.config_thresholds["criticality_threshold"]
    assert result.null_controls["wrong_stdp_sign_rejection_label"] == 1.0
    assert result.null_controls["missing_relaxation_rejection_label"] == 1.0
    assert result.null_controls["unsupported_empirical_evidence_rejection_label"] == 1.0
    assert "not empirical evidence" in result.claim_boundary
