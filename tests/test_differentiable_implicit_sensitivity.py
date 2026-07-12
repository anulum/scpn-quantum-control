# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — differentiable implicit sensitivity tests
# scpn-quantum-control -- implicit sensitivity tests
"""Tests for implicit stationary and fixed-point sensitivity solvers."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

import scpn_quantum_control as scpn
import scpn_quantum_control.differentiable as differentiable
from scpn_quantum_control.differentiable import (
    FixedPointSensitivityResult,
    ImplicitSensitivityResult,
    Parameter,
)
from scpn_quantum_control.differentiable_implicit_sensitivity import (
    implicit_fixed_point_sensitivity,
    implicit_stationary_sensitivity,
)


def _assert_allclose(actual: object, expected: object) -> None:
    """Assert NumPy-close equality while preserving strict test typing."""

    cast(Any, np.testing.assert_allclose)(actual, expected)


def test_facade_and_package_root_reuse_extracted_implicit_sensitivity_helpers() -> None:
    """Facade and package-root exports should point at the extracted solvers."""

    assert differentiable.implicit_fixed_point_sensitivity is implicit_fixed_point_sensitivity
    assert differentiable.implicit_stationary_sensitivity is implicit_stationary_sensitivity
    assert scpn.implicit_fixed_point_sensitivity is implicit_fixed_point_sensitivity
    assert scpn.implicit_stationary_sensitivity is implicit_stationary_sensitivity


def test_implicit_stationary_sensitivity_solves_trainable_system() -> None:
    """Implicit stationary sensitivities should solve -H^{-1}B on trainable parameters."""

    result = implicit_stationary_sensitivity(
        hessian=np.diag([2.0, 4.0, 9.0]),
        cross_derivative=np.array([[4.0, -2.0], [8.0, 4.0], [9.0, 9.0]]),
        parameters=[Parameter("x"), Parameter("y", trainable=False), Parameter("z")],
        hyperparameter_names=("alpha", "beta"),
    )

    assert isinstance(result, ImplicitSensitivityResult)
    assert result.method == "implicit_stationary_sensitivity"
    assert result.parameter_names == ("x", "y", "z")
    assert result.hyperparameter_names == ("alpha", "beta")
    assert result.trainable == (True, False, True)
    _assert_allclose(
        result.sensitivity,
        [[-2.0, 1.0], [0.0, 0.0], [-1.0, -1.0]],
    )
    assert result.condition_number == pytest.approx(9.0 / 2.0)


def test_implicit_stationary_sensitivity_applies_damping() -> None:
    """Implicit sensitivity should expose damped positive-definite solves."""

    result = implicit_stationary_sensitivity(
        hessian=np.diag([1.0, 3.0]),
        cross_derivative=[2.0, 6.0],
        damping=1.0,
    )

    _assert_allclose(result.sensitivity, [[-1.0], [-1.5]])
    assert result.damping == pytest.approx(1.0)
    assert result.hyperparameter_names == ("alpha0",)


def test_implicit_stationary_sensitivity_handles_fully_frozen_parameters() -> None:
    """Frozen stationary parameters should be reported with zero sensitivity."""

    result = implicit_stationary_sensitivity(
        hessian=np.diag([2.0, 3.0]),
        cross_derivative=np.array([[4.0], [9.0]]),
        parameters=[Parameter("x", trainable=False), Parameter("y", trainable=False)],
    )

    _assert_allclose(result.sensitivity, [[0.0], [0.0]])
    assert result.condition_number == pytest.approx(1.0)
    assert result.trainable == (False, False)


def test_implicit_stationary_sensitivity_rejects_invalid_contracts() -> None:
    """Implicit solves must fail closed on invalid stationary systems."""

    with pytest.raises(ValueError, match="square"):
        implicit_stationary_sensitivity([[1.0, 0.0]], [[1.0]])
    with pytest.raises(ValueError, match="row count"):
        implicit_stationary_sensitivity(np.eye(2), [[1.0, 2.0, 3.0]])
    with pytest.raises(ValueError, match="symmetric"):
        implicit_stationary_sensitivity([[1.0, 2.0], [0.0, 1.0]], [[1.0], [1.0]])
    with pytest.raises(ValueError, match="positive definite"):
        implicit_stationary_sensitivity([[0.0]], [[1.0]])
    with pytest.raises(ValueError, match="hyperparameter_names"):
        implicit_stationary_sensitivity(np.eye(2), np.ones((2, 2)), hyperparameter_names=("a",))
    with pytest.raises(ValueError, match="implicit rcond"):
        implicit_stationary_sensitivity(np.eye(1), [[1.0]], rcond=0.0)
    with pytest.raises(ValueError, match="finite values"):
        implicit_stationary_sensitivity([[float("nan")]], [[1.0]])
    with pytest.raises(ValueError, match="non-negative"):
        implicit_stationary_sensitivity(np.eye(1), [[1.0]], damping=-1.0)
    with pytest.raises(ValueError, match="ill-conditioned"):
        implicit_stationary_sensitivity(np.diag([1.0, 1.0e-14]), [[1.0], [1.0]])


def test_implicit_fixed_point_sensitivity_solves_trainable_system() -> None:
    """Fixed-point sensitivities should solve (I - dT/dx)^{-1}dT/dalpha."""

    result = implicit_fixed_point_sensitivity(
        state_jacobian=np.diag([0.5, 0.2, 0.0]),
        parameter_jacobian=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        parameters=[Parameter("x"), Parameter("y", trainable=False), Parameter("z")],
        hyperparameter_names=("alpha", "beta"),
    )

    assert isinstance(result, FixedPointSensitivityResult)
    assert result.method == "implicit_fixed_point_sensitivity"
    assert result.parameter_names == ("x", "y", "z")
    assert result.hyperparameter_names == ("alpha", "beta")
    assert result.trainable == (True, False, True)
    _assert_allclose(
        result.system_matrix,
        np.diag([0.5, 0.8, 1.0]),
    )
    _assert_allclose(
        result.sensitivity,
        [[2.0, 4.0], [0.0, 0.0], [5.0, 6.0]],
    )
    assert result.condition_number == pytest.approx(2.0)


def test_implicit_fixed_point_sensitivity_applies_damping() -> None:
    """Fixed-point sensitivities should expose damped nonsingular solves."""

    result = implicit_fixed_point_sensitivity(
        state_jacobian=[[0.5]],
        parameter_jacobian=[2.0],
        damping=0.5,
    )

    _assert_allclose(result.system_matrix, [[1.0]])
    _assert_allclose(result.sensitivity, [[2.0]])
    assert result.damping == pytest.approx(0.5)
    assert result.hyperparameter_names == ("alpha0",)


def test_implicit_fixed_point_sensitivity_handles_fully_frozen_parameters() -> None:
    """Frozen fixed-point parameters should be reported with zero sensitivity."""

    result = implicit_fixed_point_sensitivity(
        state_jacobian=np.diag([0.5, 0.25]),
        parameter_jacobian=np.array([[2.0], [4.0]]),
        parameters=[Parameter("x", trainable=False), Parameter("y", trainable=False)],
    )

    _assert_allclose(result.sensitivity, [[0.0], [0.0]])
    assert result.condition_number == pytest.approx(1.0)
    assert result.trainable == (False, False)


def test_implicit_fixed_point_sensitivity_rejects_invalid_contracts() -> None:
    """Fixed-point implicit differentiation must fail closed on bad systems."""

    with pytest.raises(ValueError, match="square"):
        implicit_fixed_point_sensitivity([[0.1, 0.0]], [[1.0]])
    with pytest.raises(ValueError, match="row count"):
        implicit_fixed_point_sensitivity(np.eye(2), [[1.0, 2.0, 3.0]])
    with pytest.raises(ValueError, match="ill-conditioned"):
        implicit_fixed_point_sensitivity([[1.0]], [[1.0]])
    with pytest.raises(ValueError, match="hyperparameter_names"):
        implicit_fixed_point_sensitivity(np.eye(2), np.ones((2, 2)), hyperparameter_names=("a",))
    with pytest.raises(ValueError, match="fixed-point rcond"):
        implicit_fixed_point_sensitivity(np.eye(1), [[1.0]], rcond=0.0)
    with pytest.raises(ValueError, match="fixed-point damping"):
        implicit_fixed_point_sensitivity(np.eye(1), [[1.0]], damping=-1.0)
    with pytest.raises(ValueError, match="finite values"):
        implicit_fixed_point_sensitivity([[0.5]], [[float("inf")]])
