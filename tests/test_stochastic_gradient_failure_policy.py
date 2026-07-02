# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Stochastic Gradient Failure Policy Tests
"""Tests for stochastic-gradient confidence intervals and fail-closed policy metadata."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import numpy as np
import pytest

import scpn_quantum_control as scpn
from scpn_quantum_control import Parameter, differentiable
from scpn_quantum_control.differentiable import (
    STOCHASTIC_PARAMETER_SHIFT_CLAIM_BOUNDARY,
    GradientFailurePolicy,
    StochasticGradientConfidenceInterval,
    gradient_confidence_interval,
    parameter_shift_gradient_with_uncertainty,
    score_function_gradient_estimate,
)
from scpn_quantum_control.differentiable_stochastic_policy import (
    STOCHASTIC_PARAMETER_SHIFT_CLAIM_BOUNDARY as DIRECT_STOCHASTIC_BOUNDARY,
)
from scpn_quantum_control.differentiable_stochastic_policy import (
    GradientFailurePolicy as DirectGradientFailurePolicy,
)
from scpn_quantum_control.differentiable_stochastic_policy import (
    StochasticGradientConfidenceInterval as DirectStochasticGradientConfidenceInterval,
)
from scpn_quantum_control.differentiable_stochastic_policy import (
    gradient_confidence_interval as direct_gradient_confidence_interval,
)

SAMPLE_PROVENANCE = {
    "sample_seed": "stochastic-policy-test-seed",
    "shot_batch_id": "stochastic-policy-test-batch",
    "source_class": "caller_supplied",
}


def test_stochastic_policy_direct_facade_and_root_exports_match() -> None:
    """The extracted policy module should preserve public import identities."""

    assert GradientFailurePolicy is DirectGradientFailurePolicy
    assert StochasticGradientConfidenceInterval is DirectStochasticGradientConfidenceInterval
    assert gradient_confidence_interval is direct_gradient_confidence_interval
    assert STOCHASTIC_PARAMETER_SHIFT_CLAIM_BOUNDARY == DIRECT_STOCHASTIC_BOUNDARY
    assert differentiable.GradientFailurePolicy is DirectGradientFailurePolicy
    assert differentiable.StochasticGradientConfidenceInterval is (
        DirectStochasticGradientConfidenceInterval
    )
    assert differentiable.gradient_confidence_interval is direct_gradient_confidence_interval
    assert differentiable.STOCHASTIC_PARAMETER_SHIFT_CLAIM_BOUNDARY == (DIRECT_STOCHASTIC_BOUNDARY)
    assert scpn.GradientFailurePolicy is DirectGradientFailurePolicy
    assert scpn.StochasticGradientConfidenceInterval is DirectStochasticGradientConfidenceInterval
    assert scpn.gradient_confidence_interval is direct_gradient_confidence_interval
    assert "no provider callback or hardware execution" in DIRECT_STOCHASTIC_BOUNDARY


def test_parameter_shift_uncertainty_reports_interval_and_passed_policy() -> None:
    result = parameter_shift_gradient_with_uncertainty(
        plus_values=[0.8, 0.1],
        minus_values=[0.2, -0.3],
        plus_variances=[0.36, 0.25],
        minus_variances=[0.16, 0.09],
        plus_shots=[900, 400],
        minus_shots=[400, 100],
        sample_provenance=SAMPLE_PROVENANCE,
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

    interval = result.confidence_interval
    assert isinstance(interval, StochasticGradientConfidenceInterval)
    assert interval.status == "failed"
    assert result.failure_policy_status == "failed"
    assert any("standard_error" in reason for reason in result.failure_reasons)


def test_gradient_confidence_interval_rejects_invalid_policy_contracts() -> None:
    with pytest.raises(ValueError, match="confidence_z"):
        gradient_confidence_interval([1.0], [0.1], confidence_z=0.0)
    with pytest.raises(ValueError, match="standard_error shape"):
        gradient_confidence_interval([1.0], [0.1, 0.2])
    with pytest.raises(ValueError, match="standard_error"):
        gradient_confidence_interval([1.0], [-0.1])
    with pytest.raises(ValueError, match="confidence_level"):
        gradient_confidence_interval([1.0], [0.1], confidence_level=1.0)
    with pytest.raises(ValueError, match="trainable mask"):
        gradient_confidence_interval([1.0], [0.1], trainable=[True, False])
    with pytest.raises(ValueError, match="trainable"):
        gradient_confidence_interval([1.0], [0.1], trainable=[False])
    with pytest.raises(ValueError, match="failure_policy"):
        GradientFailurePolicy(max_standard_error=0.0)
    with pytest.raises(ValueError, match="max_confidence_radius"):
        GradientFailurePolicy(max_confidence_radius=0.0)
    with pytest.raises(ValueError, match="require_trainable"):
        GradientFailurePolicy(require_trainable=cast(Any, np.bool_(True)))


def test_stochastic_confidence_interval_rejects_invalid_interval_metadata() -> None:
    """Interval records should fail closed on inconsistent policy metadata."""

    policy = GradientFailurePolicy(require_trainable=False)
    interval = StochasticGradientConfidenceInterval(
        lower=np.array([0.8]),
        upper=np.array([1.2]),
        confidence_z=2.0,
        confidence_level=None,
        policy=policy,
        status="passed",
        failure_reasons=(),
    )
    assert interval.to_dict()["policy"] == policy.to_dict()

    invalid_cases: tuple[tuple[Callable[[], object], str], ...] = (
        (
            lambda: StochasticGradientConfidenceInterval(
                lower=np.array([1.0, 2.0]),
                upper=np.array([1.0]),
                confidence_z=2.0,
                confidence_level=None,
                policy=policy,
                status="passed",
                failure_reasons=(),
            ),
            "lower/upper shapes",
        ),
        (
            lambda: StochasticGradientConfidenceInterval(
                lower=np.array([2.0]),
                upper=np.array([1.0]),
                confidence_z=2.0,
                confidence_level=None,
                policy=policy,
                status="passed",
                failure_reasons=(),
            ),
            "lower bounds",
        ),
        (
            lambda: StochasticGradientConfidenceInterval(
                lower=np.array([1.0]),
                upper=np.array([2.0]),
                confidence_z=0.0,
                confidence_level=None,
                policy=policy,
                status="passed",
                failure_reasons=(),
            ),
            "confidence_z",
        ),
        (
            lambda: StochasticGradientConfidenceInterval(
                lower=np.array([1.0]),
                upper=np.array([2.0]),
                confidence_z=2.0,
                confidence_level=1.0,
                policy=policy,
                status="passed",
                failure_reasons=(),
            ),
            "confidence_level",
        ),
        (
            lambda: StochasticGradientConfidenceInterval(
                lower=np.array([1.0]),
                upper=np.array([2.0]),
                confidence_z=2.0,
                confidence_level=None,
                policy=policy,
                status="unknown",
                failure_reasons=(),
            ),
            "status",
        ),
        (
            lambda: StochasticGradientConfidenceInterval(
                lower=np.array([1.0]),
                upper=np.array([2.0]),
                confidence_z=2.0,
                confidence_level=None,
                policy=policy,
                status="passed",
                failure_reasons=("unexpected",),
            ),
            "cannot contain failure reasons",
        ),
    )
    for factory, message in invalid_cases:
        with pytest.raises(ValueError, match=message):
            factory()


def test_gradient_confidence_interval_reports_threshold_failures() -> None:
    """Both policy thresholds should produce auditable failure reasons."""

    default_interval = gradient_confidence_interval([1.0], [0.1])
    assert default_interval.status == "passed"

    interval = gradient_confidence_interval(
        [1.0, 2.0],
        [0.2, 0.3],
        confidence_z=2.0,
        confidence_level=0.95,
        trainable=[True, True],
        failure_policy=GradientFailurePolicy(
            max_standard_error=0.1,
            max_confidence_radius=0.5,
        ),
    )

    assert interval.status == "failed"
    assert interval.confidence_level == pytest.approx(0.95)
    assert any("standard_error" in reason for reason in interval.failure_reasons)
    assert any("confidence_radius" in reason for reason in interval.failure_reasons)


def test_failure_policy_exports_from_package_root() -> None:
    assert scpn.GradientFailurePolicy is GradientFailurePolicy
    assert scpn.StochasticGradientConfidenceInterval is StochasticGradientConfidenceInterval
    assert scpn.gradient_confidence_interval is gradient_confidence_interval
