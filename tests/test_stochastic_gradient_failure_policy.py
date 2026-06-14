# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Stochastic Gradient Failure Policy Tests
"""Tests for stochastic-gradient confidence intervals and fail-closed policy metadata."""

from __future__ import annotations

import numpy as np
import pytest

import scpn_quantum_control as scpn
from scpn_quantum_control import Parameter
from scpn_quantum_control.differentiable import (
    GradientFailurePolicy,
    StochasticGradientConfidenceInterval,
    gradient_confidence_interval,
    parameter_shift_gradient_with_uncertainty,
    score_function_gradient_estimate,
)


def test_parameter_shift_uncertainty_reports_interval_and_passed_policy() -> None:
    result = parameter_shift_gradient_with_uncertainty(
        plus_values=[0.8, 0.1],
        minus_values=[0.2, -0.3],
        plus_variances=[0.36, 0.25],
        minus_variances=[0.16, 0.09],
        plus_shots=[900, 400],
        minus_shots=[400, 100],
        parameters=[Parameter("theta"), Parameter("frozen", trainable=False)],
        failure_policy=GradientFailurePolicy(
            max_standard_error=0.1,
            max_confidence_radius=0.2,
        ),
    )

    interval = result.confidence_interval
    assert isinstance(interval, StochasticGradientConfidenceInterval)
    assert interval.status == "passed"
    assert interval.failure_reasons == ()
    np.testing.assert_allclose(interval.lower, result.gradient - result.confidence_radius)
    np.testing.assert_allclose(interval.upper, result.gradient + result.confidence_radius)
    assert result.failure_policy_status == "passed"
    assert result.failure_reasons == ()


def test_score_function_policy_fails_closed_on_uncertainty_threshold() -> None:
    result = score_function_gradient_estimate(
        [2.0, 0.0, 4.0],
        [[1.0, 2.0], [-1.0, 0.0], [0.0, 1.0]],
        baseline=1.0,
        failure_policy=GradientFailurePolicy(max_standard_error=0.01),
    )

    assert result.confidence_interval.status == "failed"
    assert result.failure_policy_status == "failed"
    assert any("standard_error" in reason for reason in result.failure_reasons)


def test_gradient_confidence_interval_rejects_invalid_policy_contracts() -> None:
    with pytest.raises(ValueError, match="confidence_z"):
        gradient_confidence_interval([1.0], [0.1], confidence_z=0.0)
    with pytest.raises(ValueError, match="trainable"):
        gradient_confidence_interval([1.0], [0.1], trainable=[False])
    with pytest.raises(ValueError, match="failure_policy"):
        GradientFailurePolicy(max_standard_error=0.0)


def test_failure_policy_exports_from_package_root() -> None:
    assert scpn.GradientFailurePolicy is GradientFailurePolicy
    assert scpn.StochasticGradientConfidenceInterval is StochasticGradientConfidenceInterval
    assert scpn.gradient_confidence_interval is gradient_confidence_interval
