# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 NV quantum sensing fixture tests
"""Tests for Paper 0 NV-center quantum sensing protocol fixtures."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.paper0.nv_quantum_sensing_validation import (
    NVQuantumSensingConfig,
    decoherence_excess,
    expected_effect_size_ratio,
    falsification_decision,
    fit_decoherence_regression,
    validate_nv_quantum_sensing_fixture,
)


def test_decoherence_excess_and_effect_size_match_source_definitions() -> None:
    delta = decoherence_excess(gamma_spontaneous=1.12, gamma_replay=1.0)
    ratio = expected_effect_size_ratio(delta_gamma=delta, gamma_baseline=1.0)

    assert delta == pytest.approx(0.12)
    assert ratio == pytest.approx(0.12)


def test_regression_recovers_positive_fim_coefficient_independent_of_classical_field() -> None:
    b_classical = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64)
    fim_proxy = np.array([0.0, 0.7, 0.2, 1.0, 0.5], dtype=np.float64)
    gamma = 0.9 + 0.2 * b_classical + 0.45 * fim_proxy

    fit = fit_decoherence_regression(
        gamma=gamma,
        b_classical=b_classical,
        fim_proxy=fim_proxy,
    )

    assert fit.beta_0 == pytest.approx(0.9)
    assert fit.beta_1 == pytest.approx(0.2)
    assert fit.beta_2 == pytest.approx(0.45)
    assert fit.residual_norm < 1.0e-12


def test_falsification_decision_and_fixture_boundaries() -> None:
    assert falsification_decision(delta_gamma=0.01, beta_2=0.2, beta_2_p_value=0.01) is False
    assert falsification_decision(delta_gamma=0.0, beta_2=0.2, beta_2_p_value=0.01) is True
    assert falsification_decision(delta_gamma=0.01, beta_2=0.2, beta_2_p_value=0.06) is True
    assert falsification_decision(delta_gamma=0.01, beta_2=-0.1, beta_2_p_value=0.01) is True

    with pytest.raises(ValueError, match="gamma_baseline must be finite and positive"):
        expected_effect_size_ratio(delta_gamma=0.1, gamma_baseline=0.0)
    with pytest.raises(ValueError, match="all regression inputs must have the same shape"):
        fit_decoherence_regression(
            gamma=np.array([1.0, 1.1], dtype=np.float64),
            b_classical=np.array([0.1], dtype=np.float64),
            fim_proxy=np.array([0.2, 0.3], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="culture_count must be at least 1"):
        NVQuantumSensingConfig(culture_count=0)

    result = validate_nv_quantum_sensing_fixture()

    assert result.spec_keys == (
        "nv_quantum_sensing.block_framing",
        "nv_quantum_sensing.apparatus",
        "nv_quantum_sensing.protocol_steps",
        "nv_quantum_sensing.isomorphic_replay_control",
        "nv_quantum_sensing.analysis_and_falsification",
        "nv_quantum_sensing.controls_effect_size_timeline",
    )
    assert result.hardware_status == "protocol_design_no_lab_execution"
    assert result.source_ledger_span == ("P0R06677", "P0R06729")
    assert result.delta_gamma > 0.0
    assert 0.05 <= result.effect_size_ratio <= 0.15
    assert result.regression_beta_2 > 0.0
    assert result.falsification_rejected is False
    assert result.total_protocol_days == 30
    assert result.null_controls["shape_mismatch_rejection_label"] == 1.0
    assert result.null_controls["invalid_baseline_rejection_label"] == 1.0
    assert result.null_controls["unsupported_empirical_protocol_claim_rejection_label"] == 1.0
    assert "not empirical evidence" in result.claim_boundary
