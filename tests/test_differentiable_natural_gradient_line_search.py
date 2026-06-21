# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- natural-gradient optimizer tests
"""Tests for natural-gradient optimizer integration."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from scpn_quantum_control.differentiable import (
    NaturalGradientOptimizationResult,
    NaturalGradientOptimizer,
    Parameter,
    ParameterBounds,
)


def _assert_allclose(
    actual: object,
    expected: object,
    *,
    rtol: float = 1.0e-7,
    atol: float = 0.0,
) -> None:
    """Assert NumPy-close equality while preserving strict test typing."""

    cast(Any, np.testing.assert_allclose)(actual, expected, rtol=rtol, atol=atol)


def test_natural_gradient_optimizer_converges_with_explicit_metric() -> None:
    """Natural-gradient optimization should compose gradients, metrics, and bounds."""

    optimizer = NaturalGradientOptimizer(learning_rate=0.5, damping=0.0, max_step_norm=1.0)
    result = optimizer.minimize(
        lambda values: float(0.5 * (4.0 * values[0] ** 2 + values[1] ** 2)),
        [2.0, -2.0],
        lambda _gradient, _values: np.diag([4.0, 1.0]),
        gradient_method="finite_difference",
        bounds=[ParameterBounds(lower=-3.0, upper=3.0), ParameterBounds(lower=-3.0, upper=3.0)],
        max_steps=80,
        gradient_tolerance=1.0e-7,
    )

    assert isinstance(result, NaturalGradientOptimizationResult)
    assert result.converged
    assert result.reason == "gradient_tolerance"
    assert result.best_value <= result.value_history[0]
    assert len(result.natural_step_norm_history) == result.steps
    _assert_allclose(result.values, [0.0, 0.0], atol=1.0e-5)


def test_natural_gradient_optimizer_respects_frozen_parameters() -> None:
    """Frozen parameters should not move even when the metric includes them."""

    optimizer = NaturalGradientOptimizer(learning_rate=0.5)
    result = optimizer.minimize(
        lambda values: values[0] ** 2 + values[1] ** 2,
        [2.0, 10.0],
        lambda _gradient, _values: np.eye(2),
        parameters=[Parameter("x"), Parameter("frozen", trainable=False)],
        gradient_method="finite_difference",
        max_steps=40,
        gradient_tolerance=1.0e-7,
    )

    assert result.converged
    _assert_allclose(result.values, [0.0, 10.0], atol=1.0e-5)
    assert result.final_natural_gradient.natural_gradient[1] == pytest.approx(0.0)


def test_natural_gradient_optimizer_rejects_invalid_controls() -> None:
    """Natural-gradient optimizer controls and metric callback must fail closed."""

    with pytest.raises(ValueError, match="learning_rate"):
        NaturalGradientOptimizer(learning_rate=-1.0)
    with pytest.raises(ValueError, match="damping"):
        NaturalGradientOptimizer(damping=-1.0)
    with pytest.raises(ValueError, match="rcond"):
        NaturalGradientOptimizer(rcond=0.0)
    with pytest.raises(ValueError, match="max_step_norm"):
        NaturalGradientOptimizer(max_step_norm=0.0)

    optimizer = NaturalGradientOptimizer()
    with pytest.raises(ValueError, match="gradient_method"):
        optimizer.minimize(
            lambda values: float(values[0] ** 2),
            [1.0],
            lambda _gradient, _values: np.eye(1),
            gradient_method="unknown",
        )
    with pytest.raises(ValueError, match="positive definite"):
        optimizer.minimize(
            lambda values: float(values[0] ** 2),
            [1.0],
            lambda _gradient, _values: np.array([[0.0]]),
            gradient_method="finite_difference",
            max_steps=1,
        )
