# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 three-channel coupling fixtures
"""Tests for Paper 0 three-channel coupling parameter scan fixtures."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.three_channel_coupling_validation import (
    ThreeChannelCouplingConfig,
    classify_three_channel_outcome,
    constraint_catalogue,
    coupling_factors,
    coupling_ratios,
    propagate_gravitational_bound,
    sweet_spot_predictions,
    validate_three_channel_coupling_fixture,
)


def test_coupling_factors_and_ratios_preserve_source_values() -> None:
    factors = coupling_factors()
    ratios = coupling_ratios(factors)

    assert factors["G"] == pytest.approx(1.1e-122)
    assert factors["EM"] == pytest.approx(9.3e-2)
    assert factors["Q"] == pytest.approx(8.0e-2)
    assert factors["S"] == pytest.approx(1.3e-6)
    assert ratios["EM_over_G"] == pytest.approx(8.5e120, rel=0.02)
    assert ratios["Q_over_EM"] == pytest.approx(0.86, rel=0.02)
    assert ratios["S_over_EM"] == pytest.approx(1.4e-5, rel=0.02)


def test_constraints_and_sweet_spot_predictions_preserve_three_channel_window() -> None:
    constraints = constraint_catalogue()
    predictions = sweet_spot_predictions(lambda0=1.0e-5)

    assert tuple(item.channel for item in constraints) == ("G", "EM", "Q")
    assert constraints[1].lambda0_at_limit == pytest.approx(1.8e-5)
    assert constraints[2].lambda0_at_limit == pytest.approx(2.0e-5)
    assert predictions["extra_acceleration_m_s2"] == pytest.approx(1.0e-9)
    assert predictions["alpha_drift_per_year"] == pytest.approx(5.0e-18)
    assert predictions["decoherence_fraction"] == pytest.approx(5.0e-6)


def test_cross_channel_propagation_preserves_source_bounds() -> None:
    propagated = propagate_gravitational_bound(lambda_psi_g_bound=1.0e-126)

    assert propagated["lambda_psi_EM_bound"] == pytest.approx(1.2e-6, rel=0.05)
    assert propagated["lambda_psi_Q_bound"] == pytest.approx(9.4e-7, rel=0.05)


def test_outcome_classifier_preserves_falsification_boundary() -> None:
    assert classify_three_channel_outcome({"G": True, "EM": True, "Q": True}) == (
        "single-lambda0-correlation-supported"
    )
    assert classify_three_channel_outcome({"G": False, "EM": True, "Q": False}) == (
        "single-channel-signal-falsifies-unified-coupling"
    )
    assert classify_three_channel_outcome({"G": False, "EM": False, "Q": False}) == (
        "all-null-window-tightened"
    )


def test_three_channel_coupling_fixture_preserves_claim_boundary() -> None:
    result = validate_three_channel_coupling_fixture()

    assert result.hardware_status == "parameter_scan_protocol_no_execution"
    assert result.source_ledger_span == ("P0R07081", "P0R07129")
    assert result.channel_count == 3
    assert result.spec_count == 6
    assert result.null_controls["single_channel_overclaim_rejection_label"] == 1.0
    assert result.null_controls["ratio_mismatch_rejection_label"] == 1.0
    assert result.null_controls["missing_constraint_propagation_rejection_label"] == 1.0
    assert "not empirical support" in result.claim_boundary

    with pytest.raises(ValueError, match="lambda0 must be positive"):
        sweet_spot_predictions(lambda0=0.0)
    with pytest.raises(ValueError, match="expected_channel_count must be at least 1"):
        ThreeChannelCouplingConfig(expected_channel_count=0)
