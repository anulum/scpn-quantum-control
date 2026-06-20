# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Program AD selection-fold tests
"""Tests for Program AD static selection-fold adjoint semantics."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from scpn_quantum_control.differentiable import (
    Parameter,
    program_adjoint_gradient,
    whole_program_value_and_grad,
)


def _assert_allclose(
    actual: object, expected: object, *, rtol: float = 1.0e-7, atol: float = 0.0
) -> None:
    """Assert NumPy closeness across dynamically typed Program AD payloads."""

    cast(Any, np.testing.assert_allclose)(actual, expected, rtol=rtol, atol=atol)


def test_program_ad_select_folds_branch_adjoint_semantics() -> None:
    """Program AD np.select should route adjoints through selected branches."""

    def objective(values: Any) -> object:
        selected = np.select(
            [values < -0.5, values > 0.75],
            [values * values, 3.0 * values + 1.0],
            default=-2.0 * values,
        )
        return np.sum(selected)

    values = np.array([-1.0, -0.25, 0.5, 1.0, 2.0], dtype=np.float64)
    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=tuple(Parameter(f"theta_{index}") for index in range(values.size)),
    )

    expected_gradient = np.array([-2.0, -2.0, -2.0, 3.0, 3.0], dtype=np.float64)
    assert result.value == pytest.approx(float(cast(Any, objective(values))))
    _assert_allclose(result.gradient, expected_gradient, rtol=1.0e-12, atol=1.0e-12)
    _assert_allclose(
        program_adjoint_gradient(result), expected_gradient, rtol=1.0e-12, atol=1.0e-12
    )


def test_program_ad_piecewise_callable_folds_branch_adjoint_semantics() -> None:
    """Program AD np.piecewise should support callable branch transforms."""

    def objective(values: Any) -> object:
        selected = np.piecewise(
            values,
            [values < -0.5, values > 0.75],
            [lambda item: item * item, lambda item: 3.0 * item + 1.0, lambda item: -2.0 * item],
        )
        return np.sum(selected)

    values = np.array([-1.0, -0.25, 0.5, 1.0, 2.0], dtype=np.float64)
    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=tuple(Parameter(f"theta_{index}") for index in range(values.size)),
    )

    expected_gradient = np.array([-2.0, -2.0, -2.0, 3.0, 3.0], dtype=np.float64)
    assert result.value == pytest.approx(float(cast(Any, objective(values))))
    _assert_allclose(result.gradient, expected_gradient, rtol=1.0e-12, atol=1.0e-12)
    _assert_allclose(
        program_adjoint_gradient(result), expected_gradient, rtol=1.0e-12, atol=1.0e-12
    )


def test_program_ad_piecewise_matches_numpy_overwrite_and_default_semantics() -> None:
    """Program AD np.piecewise should preserve default-zero and later-branch semantics."""

    def objective(values: Any) -> object:
        selected = np.piecewise(
            values,
            [values > -0.5, values > 0.75],
            [lambda item: item + 1.0, lambda item: 3.0 * item],
        )
        return np.sum(selected)

    values = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=tuple(Parameter(f"theta_{index}") for index in range(values.size)),
    )

    expected_gradient = np.array([0.0, 1.0, 3.0], dtype=np.float64)
    assert result.value == pytest.approx(float(cast(Any, objective(values))))
    _assert_allclose(result.gradient, expected_gradient, rtol=1.0e-12, atol=1.0e-12)
    _assert_allclose(
        program_adjoint_gradient(result), expected_gradient, rtol=1.0e-12, atol=1.0e-12
    )


def test_program_ad_choose_routes_static_selector_adjoint_semantics() -> None:
    """Program AD np.choose should route adjoints through static selected choices."""

    selector = np.array([0, 1, 2, 0], dtype=np.int64)

    def objective(values: Any) -> object:
        selected = np.choose(
            selector,
            [values * values, -values, 3.0 * values + 1.0],
        )
        return np.sum(selected)

    values = np.array([-1.0, 0.25, 1.0, 2.0], dtype=np.float64)
    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=tuple(Parameter(f"theta_{index}") for index in range(values.size)),
    )

    expected_gradient = np.array([-2.0, -1.0, 3.0, 4.0], dtype=np.float64)
    assert result.value == pytest.approx(float(cast(Any, objective(values))))
    _assert_allclose(result.gradient, expected_gradient, rtol=1.0e-12, atol=1.0e-12)
    _assert_allclose(
        program_adjoint_gradient(result), expected_gradient, rtol=1.0e-12, atol=1.0e-12
    )


def test_program_ad_choose_matches_static_clip_and_wrap_modes() -> None:
    """Program AD np.choose should preserve NumPy static selector policies."""

    selector = np.array([-1, 0, 3], dtype=np.int64)

    def clip_objective(values: Any) -> object:
        return np.sum(np.choose(selector, [values, 2.0 * values], mode="clip"))

    def wrap_objective(values: Any) -> object:
        return np.sum(np.choose(selector, [values, 2.0 * values], mode="wrap"))

    values = np.array([0.2, 0.4, 0.6], dtype=np.float64)
    clip_result = whole_program_value_and_grad(
        clip_objective,
        values,
        parameters=tuple(Parameter(f"clip_{index}") for index in range(values.size)),
    )
    wrap_result = whole_program_value_and_grad(
        wrap_objective,
        values,
        parameters=tuple(Parameter(f"wrap_{index}") for index in range(values.size)),
    )

    assert clip_result.value == pytest.approx(float(cast(Any, clip_objective(values))))
    _assert_allclose(
        clip_result.gradient,
        np.array([1.0, 1.0, 2.0], dtype=np.float64),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert wrap_result.value == pytest.approx(float(cast(Any, wrap_objective(values))))
    _assert_allclose(
        wrap_result.gradient,
        np.array([2.0, 1.0, 2.0], dtype=np.float64),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_program_ad_compress_routes_flat_static_mask_adjoint_semantics() -> None:
    """Program AD np.compress should route flat static mask adjoints as gathers."""

    mask = np.array([True, False, True, False], dtype=np.bool_)

    def objective(values: Any) -> object:
        compressed = np.compress(mask, values)
        return np.sum(compressed * np.array([2.0, -3.0], dtype=np.float64))

    values = np.array([-1.0, 0.25, 1.5, 2.0], dtype=np.float64)
    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=tuple(Parameter(f"theta_{index}") for index in range(values.size)),
    )

    expected_gradient = np.array([2.0, 0.0, -3.0, 0.0], dtype=np.float64)
    assert result.value == pytest.approx(float(cast(Any, objective(values))))
    _assert_allclose(result.gradient, expected_gradient, rtol=1.0e-12, atol=1.0e-12)
    _assert_allclose(
        program_adjoint_gradient(result), expected_gradient, rtol=1.0e-12, atol=1.0e-12
    )


def test_program_ad_compress_routes_axis_static_mask_adjoint_semantics() -> None:
    """Program AD np.compress should preserve axis-specific static mask gathers."""

    mask = np.array([True, False, True], dtype=np.bool_)

    def objective(values: Any) -> object:
        matrix = values.reshape((2, 3))
        compressed = np.compress(mask, matrix, axis=1)
        return np.sum(compressed * np.array([[1.0, -2.0], [3.0, -4.0]], dtype=np.float64))

    values = np.array([0.5, -0.25, 1.0, 1.5, 0.75, -2.0], dtype=np.float64)
    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=tuple(Parameter(f"theta_{index}") for index in range(values.size)),
    )

    expected_gradient = np.array([1.0, 0.0, -2.0, 3.0, 0.0, -4.0], dtype=np.float64)
    assert result.value == pytest.approx(float(cast(Any, objective(values))))
    _assert_allclose(result.gradient, expected_gradient, rtol=1.0e-12, atol=1.0e-12)
    _assert_allclose(
        program_adjoint_gradient(result), expected_gradient, rtol=1.0e-12, atol=1.0e-12
    )


def test_program_ad_extract_routes_static_mask_adjoint_semantics() -> None:
    """Program AD np.extract should route same-size static mask adjoints as gathers."""

    mask = np.array([True, False, True, False], dtype=np.bool_)

    def objective(values: Any) -> object:
        extracted = np.extract(mask, values)
        return np.sum(extracted * np.array([1.5, -2.0], dtype=np.float64))

    values = np.array([-1.0, 0.25, 1.5, 2.0], dtype=np.float64)
    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=tuple(Parameter(f"theta_{index}") for index in range(values.size)),
    )

    expected_gradient = np.array([1.5, 0.0, -2.0, 0.0], dtype=np.float64)
    assert result.value == pytest.approx(float(cast(Any, objective(values))))
    _assert_allclose(result.gradient, expected_gradient, rtol=1.0e-12, atol=1.0e-12)
    _assert_allclose(
        program_adjoint_gradient(result), expected_gradient, rtol=1.0e-12, atol=1.0e-12
    )


def test_program_ad_extract_routes_matrix_static_mask_adjoint_semantics() -> None:
    """Program AD np.extract should flatten same-shape matrix masks like NumPy."""

    mask = np.array([[True, False, True], [False, True, False]], dtype=np.bool_)

    def objective(values: Any) -> object:
        matrix = values.reshape((2, 3))
        extracted = np.extract(mask, matrix)
        return np.sum(extracted * np.array([1.0, -2.0, 3.0], dtype=np.float64))

    values = np.array([0.5, -0.25, 1.0, 1.5, 0.75, -2.0], dtype=np.float64)
    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=tuple(Parameter(f"theta_{index}") for index in range(values.size)),
    )

    expected_gradient = np.array([1.0, 0.0, -2.0, 0.0, 3.0, 0.0], dtype=np.float64)
    assert result.value == pytest.approx(float(cast(Any, objective(values))))
    _assert_allclose(result.gradient, expected_gradient, rtol=1.0e-12, atol=1.0e-12)
    _assert_allclose(
        program_adjoint_gradient(result), expected_gradient, rtol=1.0e-12, atol=1.0e-12
    )


def test_program_ad_select_and_piecewise_reject_invalid_contracts() -> None:
    """Program AD selection folds should fail closed on malformed branch contracts."""

    with pytest.raises(ValueError, match="matching condition and choice counts"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.select([values > 0.0], [values, -values])),
            np.array([-1.0, 1.0], dtype=np.float64),
        )

    with pytest.raises(ValueError, match="one function per condition"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.piecewise(values, [values > 0.0], [])),
            np.array([-1.0, 1.0], dtype=np.float64),
        )


def test_program_ad_choose_rejects_dynamic_and_invalid_selector_contracts() -> None:
    """Program AD np.choose should fail closed on nondifferentiable selector contracts."""

    with pytest.raises(ValueError, match="static integer selector"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.choose(values > 0.0, [values, -values])),
            np.array([-1.0, 1.0], dtype=np.float64),
        )

    with pytest.raises(ValueError, match="out of bounds"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.choose(np.array([0, 2]), [values, -values])),
            np.array([-1.0, 1.0], dtype=np.float64),
        )


def test_program_ad_compress_rejects_dynamic_and_invalid_mask_contracts() -> None:
    """Program AD np.compress should fail closed on nondifferentiable mask contracts."""

    with pytest.raises(ValueError, match="static boolean condition"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.compress(values > 0.0, values)),
            np.array([-1.0, 1.0], dtype=np.float64),
        )

    with pytest.raises(ValueError, match="one-dimensional condition"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.compress(np.array([[True, False]]), values)),
            np.array([-1.0, 1.0], dtype=np.float64),
        )


def test_program_ad_extract_rejects_dynamic_and_size_mismatched_mask_contracts() -> None:
    """Program AD np.extract should fail closed on nondifferentiable mask contracts."""

    with pytest.raises(ValueError, match="static boolean condition"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.extract(values > 0.0, values)),
            np.array([-1.0, 1.0], dtype=np.float64),
        )

    with pytest.raises(ValueError, match="condition size must match array size"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.extract(np.array([True, False]), values)),
            np.array([-1.0, 1.0, 2.0], dtype=np.float64),
        )
