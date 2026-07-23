# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — differentiable natural gradient line search tests
# scpn-quantum-control -- natural-gradient optimizer tests
"""Tests for natural-gradient optimizer integration."""

from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
import pytest

import scpn_quantum_control as scpn
from scpn_quantum_control import differentiable as differentiable_module
from scpn_quantum_control import differentiable_natural_gradient as natural_gradient_module
from scpn_quantum_control.differentiable import (
    GradientResult,
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


def test_facade_and_package_root_reuse_extracted_natural_gradient_optimizer() -> None:
    """Facade and package-root exports should reuse the extracted optimizer."""
    assert (
        differentiable_module.NaturalGradientOptimizer
        is natural_gradient_module.NaturalGradientOptimizer
    )
    assert scpn.NaturalGradientOptimizer is natural_gradient_module.NaturalGradientOptimizer


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


def test_natural_gradient_optimizer_uses_parameter_shift_backend() -> None:
    """The optimizer should dispatch through parameter-shift gradients on request."""
    optimizer = NaturalGradientOptimizer(learning_rate=0.1)
    result = optimizer.minimize(
        lambda values: math.sin(float(values[0])),
        [0.25],
        lambda _gradient, _values: np.eye(1),
        max_steps=0,
        gradient_tolerance=0.0,
    )

    assert result.converged is False
    assert result.reason == "max_steps"
    assert result.final_gradient.method == "parameter_shift"


def test_natural_gradient_optimizer_reports_max_steps_without_update() -> None:
    """A zero-step optimizer run should report a bounded max-steps result."""
    optimizer = NaturalGradientOptimizer(learning_rate=0.1)
    result = optimizer.minimize(
        lambda values: float(values[0] ** 2),
        [1.0],
        lambda _gradient, _values: np.eye(1),
        gradient_method="finite_difference",
        max_steps=0,
        gradient_tolerance=0.0,
    )

    assert result.converged is False
    assert result.reason == "max_steps"
    assert result.steps == 0
    assert result.natural_step_norm_history == ()


def test_natural_gradient_optimizer_reports_step_tolerance() -> None:
    """A zero learning-rate update should stop at the step-tolerance gate."""
    optimizer = NaturalGradientOptimizer(learning_rate=0.0)
    result = optimizer.minimize(
        lambda values: float(values[0] ** 2),
        [1.0],
        lambda _gradient, _values: np.eye(1),
        gradient_method="finite_difference",
        max_steps=2,
        gradient_tolerance=0.0,
        step_tolerance=0.0,
    )

    assert result.converged is True
    assert result.reason == "step_tolerance"
    assert result.steps == 0


def test_natural_gradient_optimizer_reports_value_tolerance() -> None:
    """A small accepted update should stop at the value-tolerance gate."""
    optimizer = NaturalGradientOptimizer(learning_rate=1.0e-8)
    result = optimizer.minimize(
        lambda values: float(values[0]),
        [1.0],
        lambda _gradient, _values: np.eye(1),
        gradient_method="finite_difference",
        max_steps=3,
        gradient_tolerance=0.0,
        step_tolerance=0.0,
        value_tolerance=1.0e-6,
    )

    assert result.converged is True
    assert result.reason == "value_tolerance"
    assert result.steps == 1


def test_natural_gradient_optimizer_preserves_existing_best_on_equal_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Equal objective values should not replace the first best iterate."""

    def constant_gradient(
        _objective: object,
        values: object,
        *,
        parameters: object | None = None,
        step: float = 1.0e-6,
    ) -> GradientResult:
        del _objective, parameters, step
        return GradientResult(
            value=1.0,
            gradient=np.array([1.0]),
            method="finite_difference_central",
            shift=1.0e-6,
            coefficient=5.0e5,
            evaluations=3,
            parameter_names=("theta",),
            trainable=(True,),
        )

    monkeypatch.setattr(
        natural_gradient_module,
        "value_and_finite_difference_grad",
        constant_gradient,
    )
    optimizer = NaturalGradientOptimizer(learning_rate=0.25)
    result = optimizer.minimize(
        lambda values: float(values[0]),
        [1.0],
        lambda _gradient, _values: np.eye(1),
        gradient_method="finite_difference",
        max_steps=1,
        gradient_tolerance=0.0,
        step_tolerance=0.0,
    )

    assert result.reason == "max_steps"
    _assert_allclose(result.best_values, [1.0])
    assert result.best_value == pytest.approx(1.0)


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
    with pytest.raises(ValueError, match="finite_difference_step"):
        optimizer.minimize(
            lambda values: float(values[0] ** 2),
            [1.0],
            lambda _gradient, _values: np.eye(1),
            gradient_method="finite_difference",
            finite_difference_step=0.0,
        )
    with pytest.raises(ValueError, match="max_steps"):
        optimizer.minimize(
            lambda values: float(values[0] ** 2),
            [1.0],
            lambda _gradient, _values: np.eye(1),
            max_steps=-1,
        )
    with pytest.raises(ValueError, match="tolerances"):
        optimizer.minimize(
            lambda values: float(values[0] ** 2),
            [1.0],
            lambda _gradient, _values: np.eye(1),
            gradient_tolerance=-1.0,
        )
    with pytest.raises(ValueError, match="value_tolerance"):
        optimizer.minimize(
            lambda values: float(values[0] ** 2),
            [1.0],
            lambda _gradient, _values: np.eye(1),
            value_tolerance=-1.0,
        )
    with pytest.raises(ValueError, match="positive definite"):
        optimizer.minimize(
            lambda values: float(values[0] ** 2),
            [1.0],
            lambda _gradient, _values: np.array([[0.0]]),
            gradient_method="finite_difference",
            max_steps=1,
        )
