# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — adaptive FIM feedback tests
# scpn-quantum-control -- adaptive FIM feedback tests
"""Tests for adaptive FIM lambda update rules."""

from __future__ import annotations

import pytest

from scpn_quantum_control.analysis.adaptive_fim_feedback import (
    AdaptiveFIMConfig,
    FIMWitness,
    adaptive_lambda_schedule,
    propose_next_lambda,
)


def test_leakage_suppression_reduces_lambda_when_leakage_is_high() -> None:
    config = AdaptiveFIMConfig(step_gain=2.0, target_leakage=0.05)
    step = propose_next_lambda(4.0, FIMWitness(leakage=0.15, retention=0.8), config)

    assert step.lambda_out == pytest.approx(3.8)
    assert step.error_signal == pytest.approx(0.10)
    assert step.clipped is False


def test_deadband_holds_lambda_fixed() -> None:
    config = AdaptiveFIMConfig(step_gain=5.0, target_leakage=0.10, deadband=0.02)
    step = propose_next_lambda(2.0, FIMWitness(leakage=0.11, retention=0.9), config)

    assert step.lambda_out == pytest.approx(2.0)


def test_retention_recovery_reduces_lambda_when_retention_is_low() -> None:
    config = AdaptiveFIMConfig(
        mode="retention_recovery",
        target_retention=0.95,
        step_gain=1.5,
    )
    step = propose_next_lambda(3.0, FIMWitness(leakage=0.2, retention=0.75), config)

    assert step.lambda_out == pytest.approx(2.7)
    assert step.rationale == "reduce lambda when retention is below target"


def test_lambda_update_clips_to_bounds() -> None:
    config = AdaptiveFIMConfig(lambda_min=0.5, lambda_max=2.0, step_gain=10.0)
    step = propose_next_lambda(1.0, FIMWitness(leakage=0.9, retention=0.1), config)

    assert step.lambda_out == pytest.approx(0.5)
    assert step.clipped is True


def test_schedule_threads_lambda_between_witnesses() -> None:
    config = AdaptiveFIMConfig(step_gain=1.0, target_leakage=0.0)
    steps = adaptive_lambda_schedule(
        2.0,
        [
            FIMWitness(leakage=0.2, retention=0.9, depth=2),
            FIMWitness(leakage=0.1, retention=0.95, depth=4),
        ],
        config,
    )

    assert [step.index for step in steps] == [0, 1]
    assert steps[0].lambda_out == pytest.approx(1.8)
    assert steps[1].lambda_in == pytest.approx(1.8)
    assert steps[1].lambda_out == pytest.approx(1.7)


def test_invalid_probability_is_rejected() -> None:
    with pytest.raises(ValueError, match="leakage"):
        FIMWitness(leakage=1.1, retention=0.5)


def test_invalid_config_is_rejected() -> None:
    with pytest.raises(ValueError, match="lambda_max"):
        AdaptiveFIMConfig(lambda_min=2.0, lambda_max=1.0)
