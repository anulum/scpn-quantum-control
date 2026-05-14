# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 two-timescale quasicritical fixture tests
"""Tests for Paper 0 two-timescale quasicritical controller fixtures."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.two_timescale_quasicritical_validation import (
    TwoTimescaleQuasicriticalConfig,
    fast_stabilizer_gain,
    lyapunov_drift_bound,
    lyapunov_total,
    slow_explorer_gain,
    validate_two_timescale_quasicritical_fixture,
)


def test_affective_gain_scheduling_matches_exploit_explore_direction() -> None:
    cfg = TwoTimescaleQuasicriticalConfig()
    near_critical_sigma = 1.02

    high_surprise_fast = fast_stabilizer_gain(
        sigma=near_critical_sigma,
        affective_gradient=2.0,
        config=cfg,
    )
    low_surprise_fast = fast_stabilizer_gain(
        sigma=near_critical_sigma,
        affective_gradient=0.05,
        config=cfg,
    )
    high_surprise_slow = slow_explorer_gain(
        sigma=near_critical_sigma,
        affective_gradient=2.0,
        config=cfg,
    )
    low_surprise_slow = slow_explorer_gain(
        sigma=near_critical_sigma,
        affective_gradient=0.05,
        config=cfg,
    )

    assert high_surprise_fast > low_surprise_fast
    assert high_surprise_slow < low_surprise_slow
    assert slow_explorer_gain(sigma=1.4, affective_gradient=0.05, config=cfg) == 0.0


def test_lyapunov_certificate_is_positive_and_has_dissipative_bound() -> None:
    cfg = TwoTimescaleQuasicriticalConfig(beta=0.75, alpha_f=1.2, alpha_s=0.4)
    value = lyapunov_total(sigma=1.2, coherence=0.7, target_coherence=0.9, beta=cfg.beta)
    bound = lyapunov_drift_bound(
        sigma=1.2,
        coherence=0.7,
        target_coherence=0.9,
        bounded_noise=0.001,
        config=cfg,
    )

    assert value > 0.0
    assert bound < 0.0


def test_two_timescale_guards_and_fixture_boundary() -> None:
    with pytest.raises(ValueError, match="tau_s must be greater than tau_f"):
        TwoTimescaleQuasicriticalConfig(tau_f=1.0, tau_s=1.0)
    with pytest.raises(ValueError, match="delta must be finite and positive"):
        TwoTimescaleQuasicriticalConfig(delta=0.0)
    with pytest.raises(ValueError, match="sigma must be finite"):
        fast_stabilizer_gain(
            sigma=float("nan"),
            affective_gradient=0.1,
            config=TwoTimescaleQuasicriticalConfig(),
        )

    result = validate_two_timescale_quasicritical_fixture()

    assert result.spec_keys == (
        "two_timescale_quasicritical.block_framing",
        "two_timescale_quasicritical.dual_channel_architecture",
        "two_timescale_quasicritical.affective_gain_scheduling",
        "two_timescale_quasicritical.bibo_stability_certificate",
        "two_timescale_quasicritical.operational_consequence",
    )
    assert result.hardware_status == "simulator_only_no_provider_submission"
    assert result.source_ledger_span == ("P0R06646", "P0R06676")
    assert result.timescale_ratio < result.config_thresholds["max_timescale_ratio"]
    assert result.high_surprise_fast_gain > result.low_surprise_fast_gain
    assert result.high_surprise_slow_gain < result.low_surprise_slow_gain
    assert result.outside_band_slow_gain == 0.0
    assert result.lyapunov_value > 0.0
    assert result.lyapunov_drift_upper_bound < 0.0
    assert result.null_controls["invalid_timescale_rejection_label"] == 1.0
    assert result.null_controls["invalid_delta_rejection_label"] == 1.0
    assert result.null_controls["unsupported_bibo_empirical_claim_rejection_label"] == 1.0
    assert "not empirical evidence" in result.claim_boundary
