# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — differentiable transform helpers tests
# scpn-quantum-control -- differentiable transform helper tests
"""Regression tests for extracted differentiable transform helper contracts."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

import scpn_quantum_control.differentiable as differentiable
import scpn_quantum_control.differentiable_transform_helpers as helpers
from scpn_quantum_control.differentiable_parameter_contracts import Parameter, ParameterBounds
from scpn_quantum_control.differentiable_scalar_kernels import DualNumber, ReverseNode
from scpn_quantum_control.differentiable_transform_helpers import (
    _as_complex_step_scalar,
    _as_forward_mode_scalar,
    _as_reverse_mode_scalar,
    _as_scalar,
    _as_vector_output,
    _clip_gradient,
    _normalise_bounds,
    _normalise_parameters,
    _project_bounds,
    _reverse_topological_order,
    _validate_max_gradient_norm,
)


def _assert_allclose(actual: object, expected: object, *, atol: float = 0.0) -> None:
    """Assert NumPy closeness across helper payloads."""

    cast(Any, np.testing.assert_allclose)(actual, expected, atol=atol)


def test_facade_reuses_extracted_transform_helpers() -> None:
    """The differentiable facade should expose the extracted helper objects."""

    facade_symbols = vars(differentiable)
    assert facade_symbols["_as_scalar"] is _as_scalar
    assert facade_symbols["_as_vector_output"] is _as_vector_output
    assert facade_symbols["_clip_gradient"] is _clip_gradient


def test_scalar_helpers_accept_and_reject_scalar_objectives() -> None:
    """Scalar coercion helpers should fail closed on non-scalar or non-finite values."""

    assert _as_scalar(np.array(1.5, dtype=np.float64)) == pytest.approx(1.5)
    assert _as_forward_mode_scalar(DualNumber(2.0, 3.0)).tangent == pytest.approx(3.0)
    assert _as_forward_mode_scalar(2.5).primal == pytest.approx(2.5)
    reverse_node = ReverseNode(4.0)
    assert _as_reverse_mode_scalar(reverse_node) is reverse_node
    assert _as_reverse_mode_scalar(5.0).primal == pytest.approx(5.0)
    assert _as_complex_step_scalar(np.array(1.0 + 2.0j)) == pytest.approx(1.0 + 2.0j)

    with pytest.raises(ValueError, match="differentiable objective must return a scalar"):
        _as_scalar(cast(Any, np.array([1.0], dtype=np.float64)))
    with pytest.raises(ValueError, match="must be finite"):
        _as_scalar(cast(Any, np.array(np.nan, dtype=np.float64)))
    with pytest.raises(ValueError, match="forward-mode objective must return a scalar"):
        _as_forward_mode_scalar(np.array([1.0], dtype=np.float64))
    with pytest.raises(ValueError, match="reverse-mode objective must return a scalar"):
        _as_reverse_mode_scalar(np.array([1.0], dtype=np.float64))
    with pytest.raises(ValueError, match="complex-step objective must return a scalar"):
        _as_complex_step_scalar(np.array([1.0 + 0.0j], dtype=np.complex128))
    with pytest.raises(ValueError, match="complex-step objective must return a scalar"):
        _as_complex_step_scalar(cast(Any, "1.0"))
    with pytest.raises(ValueError, match="complex-step objective returned a non-finite scalar"):
        _as_complex_step_scalar(complex(np.inf, 0.0))


def test_scalar_helpers_preserve_defensive_lower_level_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scalar helpers should preserve non-scalar lower-level errors exactly."""

    def finite_error(_name: str, _value: object) -> float:
        raise ValueError("synthetic finite validation failure")

    monkeypatch.setattr(helpers, "_as_real_scalar", finite_error)
    with pytest.raises(ValueError, match="synthetic finite validation failure"):
        _as_forward_mode_scalar(1.0)
    with pytest.raises(ValueError, match="synthetic finite validation failure"):
        _as_reverse_mode_scalar(1.0)


def test_scalar_helper_rejects_defensive_nonfinite_return(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scalar helper should reject a lower-level scalar helper returning NaN."""

    def nonfinite_scalar(_name: str, _value: object) -> float:
        return float("nan")

    monkeypatch.setattr(helpers, "_as_real_scalar", nonfinite_scalar)
    with pytest.raises(ValueError, match="non-finite scalar"):
        _as_scalar(1.0)


def test_complex_step_scalar_preserves_item_conversion_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Complex-step helper should wrap scalar item conversion errors."""

    class _SyntheticDType:
        kind = "f"

    class _SyntheticScalarArray:
        shape: tuple[()] = ()
        dtype = _SyntheticDType()

        def item(self) -> object:
            raise ValueError("synthetic item conversion failure")

    def synthetic_asarray(_value: object) -> _SyntheticScalarArray:
        return _SyntheticScalarArray()

    numpy_namespace = cast(Any, vars(helpers)["np"])
    monkeypatch.setattr(numpy_namespace, "asarray", synthetic_asarray)
    with pytest.raises(ValueError, match="numeric scalar"):
        _as_complex_step_scalar(1.0)


def test_reverse_topological_order_deduplicates_shared_parents() -> None:
    """Reverse-mode tape traversal should visit shared parents once before children."""

    lhs = ReverseNode(2.0)
    rhs = ReverseNode(3.0)
    product = lhs * rhs
    root = product + lhs

    ordered = _reverse_topological_order(root)

    assert ordered[-1] is root
    assert ordered.count(lhs) == 1
    assert ordered.count(rhs) == 1
    assert ordered.index(lhs) < ordered.index(product)
    assert ordered.index(rhs) < ordered.index(product)


def test_vector_output_and_parameter_metadata_contracts() -> None:
    """Vector output and parameter metadata helpers should validate static shape."""

    _assert_allclose(_as_vector_output([1.0, 2.0]), [1.0, 2.0])
    default_parameters = _normalise_parameters(np.array([1.0, 2.0], dtype=np.float64), None)
    assert tuple(parameter.name for parameter in default_parameters) == ("theta_0", "theta_1")
    explicit = _normalise_parameters(
        np.array([1.0], dtype=np.float64), [Parameter("theta", trainable=False)]
    )
    assert explicit[0].trainable is False

    with pytest.raises(ValueError, match="one-dimensional"):
        _as_vector_output(np.array([[1.0]], dtype=np.float64))
    with pytest.raises(ValueError, match="non-finite"):
        _as_vector_output(np.array([np.nan], dtype=np.float64))
    with pytest.raises(ValueError, match="parameters length"):
        _normalise_parameters(np.array([1.0, 2.0], dtype=np.float64), [Parameter("x")])
    with pytest.raises(ValueError, match="parameter names must be unique"):
        _normalise_parameters(
            np.array([1.0, 2.0], dtype=np.float64), [Parameter("x"), Parameter("x")]
        )


def test_bounds_projection_and_gradient_clipping_contracts() -> None:
    """Bounds and gradient-norm helpers should preserve trainable masks."""

    values = np.array([-2.0, 2.0, 5.0], dtype=np.float64)
    bounds = _normalise_bounds(
        values,
        [
            ParameterBounds(lower=-1.0, upper=1.0),
            ParameterBounds(lower=-0.5, upper=0.5),
            ParameterBounds(lower=-np.pi, upper=np.pi, periodic=True),
        ],
    )

    _assert_allclose(_project_bounds(values, bounds), [-1.0, 0.5, 5.0 - 2.0 * np.pi])
    default_bounds = _normalise_bounds(np.array([1.0, 2.0], dtype=np.float64), None)
    assert len(default_bounds) == 2
    assert _validate_max_gradient_norm(None) is None
    assert _validate_max_gradient_norm(2.0) == pytest.approx(2.0)

    clipped = _clip_gradient(
        np.array([3.0, 4.0, 100.0], dtype=np.float64),
        np.array([True, True, False], dtype=np.bool_),
        max_gradient_norm=1.0,
    )
    _assert_allclose(clipped, [0.6, 0.8, 100.0], atol=1.0e-12)
    unchanged = _clip_gradient(
        np.array([3.0, 4.0], dtype=np.float64),
        np.array([False, False], dtype=np.bool_),
        max_gradient_norm=1.0,
    )
    _assert_allclose(unchanged, [3.0, 4.0])
    no_cap = _clip_gradient(
        np.array([3.0, 4.0], dtype=np.float64),
        np.array([True, True], dtype=np.bool_),
        max_gradient_norm=None,
    )
    _assert_allclose(no_cap, [3.0, 4.0])

    with pytest.raises(ValueError, match="bounds length"):
        _normalise_bounds(np.array([1.0], dtype=np.float64), [])
    with pytest.raises(ValueError, match="ParameterBounds"):
        _normalise_bounds(np.array([1.0], dtype=np.float64), [cast(Any, object())])
    with pytest.raises(ValueError, match="max_gradient_norm must be finite and positive"):
        _validate_max_gradient_norm(0.0)


def test_clip_gradient_leaves_small_trainable_norm_unchanged() -> None:
    """Gradient clipping should be a no-op below the configured norm cap."""

    gradient: NDArray[np.float64] = np.array([0.3, 0.4], dtype=np.float64)
    clipped = _clip_gradient(
        gradient,
        np.array([True, True], dtype=np.bool_),
        max_gradient_norm=2.0,
    )

    _assert_allclose(clipped, [0.3, 0.4])
    assert clipped is not gradient
