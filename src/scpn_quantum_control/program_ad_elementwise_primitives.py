# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — program AD elementwise primitives module
"""Program AD elementwise primitive contracts and direct derivative factories."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import NoReturn

import numpy as np
from numpy.typing import NDArray

from .program_ad_registry import (
    _PROGRAM_AD_ELEMENTWISE_BINARY_NAMES,
    _PROGRAM_AD_ELEMENTWISE_DISCONTINUOUS_NAMES,
    _PROGRAM_AD_ELEMENTWISE_IDENTITIES,
    _PROGRAM_AD_ELEMENTWISE_NAMES,
    _PROGRAM_AD_ELEMENTWISE_POLICY,
    _PROGRAM_AD_ELEMENTWISE_UNARY_NAMES,
    DEFAULT_CUSTOM_DERIVATIVE_REGISTRY,
    CustomDerivativeRule,
    CustomJVPRule,
    CustomVJPRule,
    PrimitiveContract,
    PrimitiveShapeRule,
    PrimitiveTransformRule,
    VectorObjective,
)


def _as_real_numeric_array(name: str, value: object) -> NDArray[np.float64]:
    """Return ``value`` as a real-valued float64 array."""
    array = np.asarray(value)
    if array.dtype.kind not in {"b", "i", "u", "f"}:
        raise ValueError(f"{name} must be a real numeric array")
    return np.asarray(array, dtype=np.float64)


def _normalise_axis(name: str, axis: int, ndim: int) -> int:
    """Return a non-negative axis for an array with ``ndim`` dimensions."""
    if ndim == 0:
        raise ValueError(f"{name} cannot map over a scalar")
    if axis < 0:
        axis += ndim
    if axis < 0 or axis >= ndim:
        raise ValueError(f"{name} is out of bounds for argument rank {ndim}")
    return axis


def _program_ad_float64_vector_result(values: object) -> NDArray[np.float64]:
    """Flatten a direct-rule result into the canonical float64 vector form."""
    return np.asarray(values, dtype=np.float64).reshape(-1)


def _program_ad_shape_static_size(source_shape: Sequence[int]) -> int:
    """Return the element count for a static tensor shape."""
    size = 1
    for dimension in source_shape:
        size *= int(dimension)
    return int(size)


def _program_ad_shape_signature(source_shape: Sequence[int]) -> str:
    """Return a stable shape token for direct derivative rule names."""
    if not source_shape:
        return "scalar"
    return "x".join(str(int(dimension)) for dimension in source_shape)


def _program_ad_array_shape_of(value: object) -> tuple[int, ...]:
    """Return the static array shape recorded by a trace value or array-like input."""
    shape = getattr(value, "shape", None)
    if isinstance(shape, tuple):
        return tuple(int(dimension) for dimension in shape)
    if isinstance(shape, Sequence):
        return tuple(int(dimension) for dimension in shape)
    return tuple(int(dimension) for dimension in np.asarray(value).shape)


def _program_ad_array_dtype_of(value: object) -> str:
    """Return the dtype name recorded by a trace value or array-like input."""
    dtype = getattr(value, "dtype", None)
    if dtype is not None:
        return str(np.dtype(dtype))
    if hasattr(value, "_items") and hasattr(value, "context"):
        return "float64"
    array = np.asarray(value)
    if array.dtype.kind in {"O", "S", "U", "c"}:
        raise ValueError("program AD elementwise dtype rule requires real numeric arrays")
    return str(array.dtype)


def _broadcast_shape(*shapes: tuple[int, ...]) -> tuple[int, ...]:
    """Return a NumPy-compatible broadcast shape or fail closed."""
    try:
        shape: tuple[int, ...] = np.broadcast_shapes(*shapes)
        return shape
    except ValueError as exc:
        raise ValueError(
            "program AD elementwise operands must follow NumPy broadcasting rules"
        ) from exc


def _program_ad_elementwise_unbroadcast(
    values: NDArray[np.float64],
    *,
    target_shape: tuple[int, ...],
) -> NDArray[np.float64]:
    """Reduce a broadcasted adjoint back to ``target_shape``."""
    result = np.asarray(values, dtype=np.float64)
    if target_shape == ():
        return np.array([float(np.sum(result))], dtype=np.float64)
    while result.ndim > len(target_shape):
        result = np.sum(result, axis=0)
    for axis, dimension in enumerate(target_shape):
        if dimension == 1 and result.shape[axis] != 1:
            result = np.sum(result, axis=axis, keepdims=True)
    return _program_ad_float64_vector_result(result.reshape(target_shape))


def _validate_program_ad_elementwise_contract_dispatch(
    contract: PrimitiveContract,
    args: tuple[object, ...],
) -> None:
    """Validate elementwise primitive dispatch helpers against concrete arguments."""
    if contract.static_argument_rule is None:
        raise ValueError(
            f"program AD primitive {contract.identity.key} missing static argument rule"
        )
    if contract.shape_rule is None:
        raise ValueError(f"program AD primitive {contract.identity.key} missing shape rule")
    if contract.dtype_rule is None:
        raise ValueError(f"program AD primitive {contract.identity.key} missing dtype rule")
    static_arguments = contract.static_argument_rule(args)
    if not isinstance(static_arguments, tuple):
        raise ValueError(
            f"program AD primitive {contract.identity.key} static rule must return a tuple"
        )
    shape = contract.shape_rule(args)
    if not isinstance(shape, tuple) or any(
        not isinstance(dimension, int) or dimension < 0 for dimension in shape
    ):
        raise ValueError(
            f"program AD primitive {contract.identity.key} shape rule must return "
            "non-negative integer dimensions"
        )
    dtype = contract.dtype_rule(args)
    if not isinstance(dtype, str) or not dtype:
        raise ValueError(
            f"program AD primitive {contract.identity.key} dtype rule must return a dtype name"
        )


def _program_ad_elementwise_direct_value(_values: NDArray[np.float64]) -> NDArray[np.float64]:
    raise ValueError(
        "program AD elementwise primitive contracts are executable only through "
        "operator-intercepted trace dispatch"
    )


def _program_ad_elementwise_direct_jvp(
    _values: NDArray[np.float64],
    _tangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    raise ValueError(
        "program AD elementwise primitive contracts are executable only through "
        "operator-intercepted trace dispatch"
    )


def _raise_program_ad_derivative_losing_elementwise(name: str) -> NoReturn:
    raise ValueError(
        f"program AD {name} is derivative-losing and fails closed under the registered "
        "nondifferentiability policy"
    )


def _program_ad_elementwise_derivative_losing_value_for(name: str) -> VectorObjective:
    def value_fn(_values: NDArray[np.float64]) -> NDArray[np.float64]:
        _raise_program_ad_derivative_losing_elementwise(name)

    return value_fn


def _program_ad_elementwise_derivative_losing_jvp_for(name: str) -> CustomJVPRule:
    def jvp_rule(
        _values: NDArray[np.float64],
        _tangent: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        _raise_program_ad_derivative_losing_elementwise(name)

    return jvp_rule


def _program_ad_elementwise_unary_vector(
    name: str, values: NDArray[np.float64]
) -> NDArray[np.float64]:
    return _as_real_numeric_array(f"program AD elementwise {name} values", values).reshape(-1)


def _program_ad_elementwise_unary_tangent_pair(
    name: str,
    values: NDArray[np.float64],
    tangent: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    vector = _program_ad_elementwise_unary_vector(name, values)
    tangent_vector = _as_real_numeric_array(
        f"program AD elementwise {name} tangent", tangent
    ).reshape(-1)
    if tangent_vector.shape != vector.shape:
        raise ValueError(f"program AD elementwise {name} tangent shape must match values shape")
    return vector, tangent_vector


def _program_ad_elementwise_require_domain(
    name: str,
    vector: NDArray[np.float64],
    *,
    derivative: bool,
) -> None:
    if name == "log" and np.any(vector <= 0.0):
        raise ValueError(
            "program AD elementwise log direct rule requires values greater than zero"
        )
    if name == "log1p" and np.any(vector <= -1.0):
        raise ValueError(
            "program AD elementwise log1p direct rule requires values greater than -1"
        )
    if name == "sqrt" and np.any(vector < 0.0):
        raise ValueError("program AD elementwise sqrt direct rule requires non-negative values")
    if name == "sqrt" and derivative and np.any(vector <= 0.0):
        raise ValueError(
            "program AD elementwise sqrt derivative is singular at non-positive values"
        )
    if name in {"arcsin", "arccos"} and np.any(np.abs(vector) > 1.0):
        raise ValueError(f"program AD elementwise {name} direct rule requires values in [-1, 1]")
    if name in {"arcsin", "arccos"} and derivative and np.any(np.abs(vector) >= 1.0):
        raise ValueError(
            f"program AD elementwise {name} derivative is singular at boundary values"
        )
    if name == "reciprocal" and np.any(vector == 0.0):
        raise ValueError("program AD elementwise reciprocal direct rule requires non-zero values")
    if name == "tan" and derivative and np.any(np.cos(vector) == 0.0):
        raise ValueError("program AD elementwise tan derivative is singular at odd pi/2 values")
    if name == "abs" and derivative and np.any(vector == 0.0):
        raise ValueError("program AD elementwise abs derivative is undefined at zero")


def _program_ad_elementwise_unary_value(
    name: str, values: NDArray[np.float64]
) -> NDArray[np.float64]:
    vector = _program_ad_elementwise_unary_vector(name, values)
    _program_ad_elementwise_require_domain(name, vector, derivative=False)
    if name == "sin":
        return _program_ad_float64_vector_result(np.sin(vector))
    if name == "cos":
        return _program_ad_float64_vector_result(np.cos(vector))
    if name == "exp":
        return _program_ad_float64_vector_result(np.exp(vector))
    if name == "expm1":
        return _program_ad_float64_vector_result(np.expm1(vector))
    if name == "log":
        return _program_ad_float64_vector_result(np.log(vector))
    if name == "log1p":
        return _program_ad_float64_vector_result(np.log1p(vector))
    if name == "sqrt":
        return _program_ad_float64_vector_result(np.sqrt(vector))
    if name == "tan":
        return _program_ad_float64_vector_result(np.tan(vector))
    if name == "tanh":
        return _program_ad_float64_vector_result(np.tanh(vector))
    if name == "arcsin":
        return _program_ad_float64_vector_result(np.arcsin(vector))
    if name == "arccos":
        return _program_ad_float64_vector_result(np.arccos(vector))
    if name == "reciprocal":
        return _program_ad_float64_vector_result(np.reciprocal(vector))
    if name == "square":
        return _program_ad_float64_vector_result(np.square(vector))
    if name == "abs":
        return _program_ad_float64_vector_result(np.abs(vector))
    if name == "negative":
        return _program_ad_float64_vector_result(np.negative(vector))
    raise ValueError(f"unsupported program AD elementwise primitive {name}")


def _program_ad_elementwise_unary_jvp(
    name: str,
    values: NDArray[np.float64],
    tangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    vector, tangent_vector = _program_ad_elementwise_unary_tangent_pair(name, values, tangent)
    _program_ad_elementwise_require_domain(name, vector, derivative=True)
    if name == "sin":
        return _program_ad_float64_vector_result(np.cos(vector) * tangent_vector)
    if name == "cos":
        return _program_ad_float64_vector_result(-np.sin(vector) * tangent_vector)
    if name == "exp":
        return _program_ad_float64_vector_result(np.exp(vector) * tangent_vector)
    if name == "expm1":
        return _program_ad_float64_vector_result(np.exp(vector) * tangent_vector)
    if name == "log":
        return _program_ad_float64_vector_result(tangent_vector / vector)
    if name == "log1p":
        return _program_ad_float64_vector_result(tangent_vector / (1.0 + vector))
    if name == "sqrt":
        return _program_ad_float64_vector_result(tangent_vector / (2.0 * np.sqrt(vector)))
    if name == "tan":
        return _program_ad_float64_vector_result(tangent_vector / np.cos(vector) ** 2)
    if name == "tanh":
        return _program_ad_float64_vector_result(tangent_vector * (1.0 - np.tanh(vector) ** 2))
    if name == "arcsin":
        return _program_ad_float64_vector_result(tangent_vector / np.sqrt(1.0 - vector**2))
    if name == "arccos":
        return _program_ad_float64_vector_result(-tangent_vector / np.sqrt(1.0 - vector**2))
    if name == "reciprocal":
        return _program_ad_float64_vector_result(-tangent_vector / vector**2)
    if name == "square":
        return _program_ad_float64_vector_result(2.0 * vector * tangent_vector)
    if name == "abs":
        return _program_ad_float64_vector_result(np.sign(vector) * tangent_vector)
    if name == "negative":
        return _program_ad_float64_vector_result(np.negative(tangent_vector))
    raise ValueError(f"unsupported program AD elementwise primitive {name}")


def _program_ad_elementwise_direct_value_for(name: str) -> VectorObjective:
    return lambda values: _program_ad_elementwise_unary_value(name, values)


def _program_ad_elementwise_direct_jvp_for(name: str) -> CustomJVPRule:
    return lambda values, tangent: _program_ad_elementwise_unary_jvp(name, values, tangent)


def _program_ad_elementwise_direct_vjp_for(name: str) -> CustomVJPRule:
    return lambda values, cotangent: _program_ad_elementwise_unary_jvp(name, values, cotangent)


def _program_ad_elementwise_binary_pair(
    name: str,
    values: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    vector = _as_real_numeric_array(f"program AD elementwise {name} values", values).reshape(-1)
    if vector.size == 0 or vector.size % 2 != 0:
        raise ValueError(
            f"program AD elementwise {name} direct rule requires two equal flat operands"
        )
    midpoint = vector.size // 2
    return vector[:midpoint], vector[midpoint:]


def _program_ad_elementwise_binary_tangent_pair(
    name: str,
    values: NDArray[np.float64],
    tangent: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    left, right = _program_ad_elementwise_binary_pair(name, values)
    tangent_left, tangent_right = _program_ad_elementwise_binary_pair(f"{name} tangent", tangent)
    if tangent_left.shape != left.shape or tangent_right.shape != right.shape:
        raise ValueError(f"program AD elementwise {name} tangent shape must match values shape")
    return left, right, tangent_left, tangent_right


def _program_ad_elementwise_require_binary_domain(
    name: str,
    left: NDArray[np.float64],
    right: NDArray[np.float64],
    *,
    derivative: bool,
) -> None:
    if name == "divide" and np.any(right == 0.0):
        raise ValueError(
            "program AD elementwise divide direct rule requires non-zero right operand"
        )
    if name == "power" and np.any(left <= 0.0):
        raise ValueError("program AD elementwise power direct rule requires positive left operand")
    if name in {"maximum", "minimum"} and derivative and np.any(left == right):
        raise ValueError(
            f"program AD elementwise {name} derivative is undefined at equal operands"
        )


def _program_ad_elementwise_binary_value(
    name: str,
    values: NDArray[np.float64],
) -> NDArray[np.float64]:
    left, right = _program_ad_elementwise_binary_pair(name, values)
    _program_ad_elementwise_require_binary_domain(name, left, right, derivative=False)
    if name == "add":
        return _program_ad_float64_vector_result(left + right)
    if name == "subtract":
        return _program_ad_float64_vector_result(left - right)
    if name == "multiply":
        return _program_ad_float64_vector_result(left * right)
    if name == "divide":
        return _program_ad_float64_vector_result(left / right)
    if name == "power":
        return _program_ad_float64_vector_result(left**right)
    if name == "maximum":
        return _program_ad_float64_vector_result(np.maximum(left, right))
    if name == "minimum":
        return _program_ad_float64_vector_result(np.minimum(left, right))
    raise ValueError(f"unsupported program AD elementwise primitive {name}")


def _program_ad_elementwise_binary_jvp(
    name: str,
    values: NDArray[np.float64],
    tangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    left, right, tangent_left, tangent_right = _program_ad_elementwise_binary_tangent_pair(
        name, values, tangent
    )
    _program_ad_elementwise_require_binary_domain(name, left, right, derivative=True)
    if name == "add":
        return _program_ad_float64_vector_result(tangent_left + tangent_right)
    if name == "subtract":
        return _program_ad_float64_vector_result(tangent_left - tangent_right)
    if name == "multiply":
        return _program_ad_float64_vector_result(tangent_left * right + left * tangent_right)
    if name == "divide":
        return _program_ad_float64_vector_result(
            (tangent_left * right - left * tangent_right) / right**2
        )
    if name == "power":
        return _program_ad_float64_vector_result(
            left**right * (tangent_right * np.log(left) + right * tangent_left / left)
        )
    if name == "maximum":
        return _program_ad_float64_vector_result(
            np.where(left > right, tangent_left, tangent_right)
        )
    if name == "minimum":
        return _program_ad_float64_vector_result(
            np.where(left < right, tangent_left, tangent_right)
        )
    raise ValueError(f"unsupported program AD elementwise primitive {name}")


def _program_ad_elementwise_binary_vjp(
    name: str,
    values: NDArray[np.float64],
    cotangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    left, right = _program_ad_elementwise_binary_pair(name, values)
    cotangent_vector = _as_real_numeric_array(
        f"program AD elementwise {name} cotangent", cotangent
    ).reshape(-1)
    if cotangent_vector.shape != left.shape:
        raise ValueError(
            f"program AD elementwise {name} VJP cotangent shape must match output shape"
        )
    _program_ad_elementwise_require_binary_domain(name, left, right, derivative=True)
    left_vjp: NDArray[np.float64]
    right_vjp: NDArray[np.float64]
    if name == "add":
        left_vjp = cotangent_vector
        right_vjp = cotangent_vector
    elif name == "subtract":
        left_vjp = cotangent_vector
        right_vjp = -cotangent_vector
    elif name == "multiply":
        left_vjp = cotangent_vector * right
        right_vjp = cotangent_vector * left
    elif name == "divide":
        left_vjp = cotangent_vector / right
        right_vjp = -cotangent_vector * left / right**2
    elif name == "power":
        left_vjp = cotangent_vector * right * left ** (right - 1.0)
        right_vjp = cotangent_vector * left**right * np.log(left)
    elif name == "maximum":
        left_vjp = _program_ad_float64_vector_result(np.where(left > right, cotangent_vector, 0.0))
        right_vjp = _program_ad_float64_vector_result(
            np.where(left > right, 0.0, cotangent_vector)
        )
    elif name == "minimum":
        left_vjp = _program_ad_float64_vector_result(np.where(left < right, cotangent_vector, 0.0))
        right_vjp = _program_ad_float64_vector_result(
            np.where(left < right, 0.0, cotangent_vector)
        )
    else:
        raise ValueError(f"unsupported program AD elementwise primitive {name}")
    return _program_ad_float64_vector_result(np.concatenate([left_vjp, right_vjp]))


def _program_ad_elementwise_binary_value_for(name: str) -> VectorObjective:
    return lambda values: _program_ad_elementwise_binary_value(name, values)


def _program_ad_elementwise_binary_jvp_for(name: str) -> CustomJVPRule:
    return lambda values, tangent: _program_ad_elementwise_binary_jvp(name, values, tangent)


def _program_ad_elementwise_binary_vjp_for(name: str) -> CustomVJPRule:
    return lambda values, cotangent: _program_ad_elementwise_binary_vjp(name, values, cotangent)


def _program_ad_elementwise_normalise_binary_static_shapes(
    name: str,
    left_shape: Sequence[int],
    right_shape: Sequence[int],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    if name not in _PROGRAM_AD_ELEMENTWISE_BINARY_NAMES:
        raise ValueError(f"unsupported program AD elementwise binary primitive {name}")
    left = tuple(int(dimension) for dimension in left_shape)
    right = tuple(int(dimension) for dimension in right_shape)
    if any(dimension < 0 for dimension in (*left, *right)):
        raise ValueError(
            f"program AD elementwise {name} direct rule requires non-negative dimensions"
        )
    try:
        output = np.broadcast_shapes(left, right)
    except ValueError as exc:
        raise ValueError(
            f"program AD elementwise {name} direct rule requires broadcast-compatible shapes"
        ) from exc
    return left, right, tuple(int(dimension) for dimension in output)


def _program_ad_elementwise_binary_static_split(
    name: str,
    role: str,
    values: NDArray[np.float64],
    *,
    left_shape: tuple[int, ...],
    right_shape: tuple[int, ...],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    vector = _as_real_numeric_array(f"program AD elementwise {name} {role}", values).reshape(-1)
    left_size = _program_ad_shape_static_size(left_shape)
    right_size = _program_ad_shape_static_size(right_shape)
    if vector.size != left_size + right_size:
        raise ValueError(
            f"program AD elementwise {name} direct rule requires flattened left operand "
            "followed by right operand"
        )
    return (
        vector[:left_size].reshape(left_shape),
        vector[left_size:].reshape(right_shape),
    )


def _program_ad_elementwise_binary_static_value_array(
    name: str,
    left: NDArray[np.float64],
    right: NDArray[np.float64],
) -> NDArray[np.float64]:
    _program_ad_elementwise_require_binary_domain(name, left, right, derivative=False)
    if name == "add":
        return _program_ad_float64_vector_result(left + right)
    if name == "subtract":
        return _program_ad_float64_vector_result(left - right)
    if name == "multiply":
        return _program_ad_float64_vector_result(left * right)
    if name == "divide":
        return _program_ad_float64_vector_result(left / right)
    if name == "power":
        return _program_ad_float64_vector_result(left**right)
    if name == "maximum":
        return _program_ad_float64_vector_result(np.maximum(left, right))
    if name == "minimum":
        return _program_ad_float64_vector_result(np.minimum(left, right))
    raise ValueError(f"unsupported program AD elementwise primitive {name}")


def _program_ad_elementwise_binary_static_jvp_array(
    name: str,
    left: NDArray[np.float64],
    right: NDArray[np.float64],
    tangent_left: NDArray[np.float64],
    tangent_right: NDArray[np.float64],
) -> NDArray[np.float64]:
    _program_ad_elementwise_require_binary_domain(name, left, right, derivative=True)
    if name == "add":
        return _program_ad_float64_vector_result(tangent_left + tangent_right)
    if name == "subtract":
        return _program_ad_float64_vector_result(tangent_left - tangent_right)
    if name == "multiply":
        return _program_ad_float64_vector_result(tangent_left * right + left * tangent_right)
    if name == "divide":
        return _program_ad_float64_vector_result(
            (tangent_left * right - left * tangent_right) / right**2
        )
    if name == "power":
        return _program_ad_float64_vector_result(
            left**right * (tangent_right * np.log(left) + right * tangent_left / left)
        )
    if name == "maximum":
        return _program_ad_float64_vector_result(
            np.where(left > right, tangent_left, tangent_right)
        )
    if name == "minimum":
        return _program_ad_float64_vector_result(
            np.where(left < right, tangent_left, tangent_right)
        )
    raise ValueError(f"unsupported program AD elementwise primitive {name}")


def _program_ad_elementwise_binary_static_adjoint_arrays(
    name: str,
    left: NDArray[np.float64],
    right: NDArray[np.float64],
    cotangent: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    _program_ad_elementwise_require_binary_domain(name, left, right, derivative=True)
    if name == "add":
        return cotangent, cotangent
    if name == "subtract":
        return cotangent, -cotangent
    if name == "multiply":
        return cotangent * right, cotangent * left
    if name == "divide":
        return cotangent / right, -cotangent * left / right**2
    if name == "power":
        return (
            cotangent * right * left ** (right - 1.0),
            cotangent * left**right * np.log(left),
        )
    if name == "maximum":
        return (
            _program_ad_float64_vector_result(np.where(left > right, cotangent, 0.0)).reshape(
                cotangent.shape
            ),
            _program_ad_float64_vector_result(np.where(left > right, 0.0, cotangent)).reshape(
                cotangent.shape
            ),
        )
    if name == "minimum":
        return (
            _program_ad_float64_vector_result(np.where(left < right, cotangent, 0.0)).reshape(
                cotangent.shape
            ),
            _program_ad_float64_vector_result(np.where(left < right, 0.0, cotangent)).reshape(
                cotangent.shape
            ),
        )
    raise ValueError(f"unsupported program AD elementwise primitive {name}")


def program_ad_elementwise_binary_derivative_rule(
    name: str,
    left_shape: Sequence[int],
    right_shape: Sequence[int],
) -> CustomDerivativeRule:
    """Build a direct value/JVP/VJP rule for a fixed broadcasted binary primitive."""
    left_static_shape, right_static_shape, output_shape = (
        _program_ad_elementwise_normalise_binary_static_shapes(name, left_shape, right_shape)
    )

    def value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
        left, right = _program_ad_elementwise_binary_static_split(
            name, "values", values, left_shape=left_static_shape, right_shape=right_static_shape
        )
        return _program_ad_elementwise_binary_static_value_array(name, left, right)

    def jvp_rule(values: NDArray[np.float64], tangent: NDArray[np.float64]) -> NDArray[np.float64]:
        left, right = _program_ad_elementwise_binary_static_split(
            name, "values", values, left_shape=left_static_shape, right_shape=right_static_shape
        )
        tangent_left, tangent_right = _program_ad_elementwise_binary_static_split(
            name,
            "tangent",
            tangent,
            left_shape=left_static_shape,
            right_shape=right_static_shape,
        )
        return _program_ad_elementwise_binary_static_jvp_array(
            name, left, right, tangent_left, tangent_right
        )

    def vjp_rule(
        values: NDArray[np.float64], cotangent: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        left, right = _program_ad_elementwise_binary_static_split(
            name, "values", values, left_shape=left_static_shape, right_shape=right_static_shape
        )
        cotangent_vector = _as_real_numeric_array(
            f"program AD elementwise {name} cotangent", cotangent
        ).reshape(-1)
        if cotangent_vector.size != _program_ad_shape_static_size(output_shape):
            raise ValueError(
                f"program AD elementwise {name} VJP cotangent shape must match output shape"
            )
        cotangent_array = cotangent_vector.reshape(output_shape)
        left_adjoint, right_adjoint = _program_ad_elementwise_binary_static_adjoint_arrays(
            name, left, right, cotangent_array
        )
        return _program_ad_float64_vector_result(
            np.concatenate(
                (
                    _program_ad_elementwise_unbroadcast(
                        left_adjoint, target_shape=left_static_shape
                    ),
                    _program_ad_elementwise_unbroadcast(
                        right_adjoint, target_shape=right_static_shape
                    ),
                )
            )
        )

    return CustomDerivativeRule(
        name=(
            f"program_ad_elementwise_{name}_{_program_ad_shape_signature(left_static_shape)}_by_"
            f"{_program_ad_shape_signature(right_static_shape)}_broadcast_direct_rule"
        ),
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
    )


def _program_ad_elementwise_derivative_rule(name: str) -> CustomDerivativeRule:
    if name in _PROGRAM_AD_ELEMENTWISE_DISCONTINUOUS_NAMES:
        return CustomDerivativeRule(
            name=f"program_ad_elementwise_{name}_fail_closed_rule",
            value_fn=_program_ad_elementwise_derivative_losing_value_for(name),
            jvp_rule=_program_ad_elementwise_derivative_losing_jvp_for(name),
            vjp_rule=_program_ad_elementwise_derivative_losing_jvp_for(name),
        )
    if name in _PROGRAM_AD_ELEMENTWISE_UNARY_NAMES:
        return CustomDerivativeRule(
            name=f"program_ad_elementwise_{name}_direct_rule",
            value_fn=_program_ad_elementwise_direct_value_for(name),
            jvp_rule=_program_ad_elementwise_direct_jvp_for(name),
            vjp_rule=_program_ad_elementwise_direct_vjp_for(name),
        )
    if name in _PROGRAM_AD_ELEMENTWISE_BINARY_NAMES:
        return CustomDerivativeRule(
            name=f"program_ad_elementwise_{name}_direct_rule",
            value_fn=_program_ad_elementwise_binary_value_for(name),
            jvp_rule=_program_ad_elementwise_binary_jvp_for(name),
            vjp_rule=_program_ad_elementwise_binary_vjp_for(name),
        )
    return CustomDerivativeRule(
        name=f"program_ad_elementwise_{name}_trace_contract",
        value_fn=_program_ad_elementwise_direct_value,
        jvp_rule=_program_ad_elementwise_direct_jvp,
    )


def _program_ad_elementwise_name(ufunc: np.ufunc) -> str:
    if ufunc is np.absolute:
        return "abs"
    return str(ufunc.__name__)


def _program_ad_elementwise_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    if len(args) == 1:
        return _program_ad_array_shape_of(args[0])
    if len(args) == 2:
        return _broadcast_shape(
            _program_ad_array_shape_of(args[0]), _program_ad_array_shape_of(args[1])
        )
    raise ValueError("program AD elementwise shape rule requires one or two operands")


def _program_ad_elementwise_dtype_rule(args: tuple[object, ...]) -> str:
    if len(args) not in {1, 2}:
        raise ValueError("program AD elementwise dtype rule requires one or two operands")
    dtypes = tuple(np.dtype(_program_ad_array_dtype_of(arg)) for arg in args)
    return str(np.result_type(*dtypes))


def _program_ad_elementwise_static_arguments(args: tuple[object, ...]) -> tuple[object, ...]:
    if len(args) not in {1, 2}:
        raise ValueError("program AD elementwise static rule requires one or two operands")
    return ()


_PROGRAM_AD_ELEMENTWISE_SHAPE_RULES: Mapping[str, PrimitiveShapeRule] = {
    name: _program_ad_elementwise_shape for name in _PROGRAM_AD_ELEMENTWISE_NAMES
}


def _program_ad_elementwise_batching_rule(
    function: Callable[..., object],
    args: tuple[object, ...],
    axes: tuple[int | None, ...],
    out_axes: int,
) -> object:
    if len(args) != len(axes):
        raise ValueError("program AD elementwise batching axes must match argument count")
    if len(args) not in {1, 2}:
        raise ValueError("program AD elementwise batching requires one or two operands")
    arrays = tuple(
        _as_real_numeric_array(f"program AD elementwise batched operand {index}", arg)
        for index, arg in enumerate(args)
    )
    if all(axis is None for axis in axes):
        result = _as_real_numeric_array("program AD elementwise batched output", function(*arrays))
        return result
    mapped_axes: list[int | None] = [
        None if axis is None else _normalise_axis(f"axes[{index}]", axis, array.ndim)
        for index, (axis, array) in enumerate(zip(axes, arrays, strict=True))
    ]
    batch_sizes = {
        int(array.shape[axis])
        for array, axis in zip(arrays, mapped_axes, strict=True)
        if axis is not None
    }
    if len(batch_sizes) != 1:
        raise ValueError("program AD elementwise batching axes must share one batch size")
    batch_size = batch_sizes.pop()
    outputs = []
    for batch_index in range(batch_size):
        sliced_args = tuple(
            array if axis is None else np.take(array, batch_index, axis=axis)
            for array, axis in zip(arrays, mapped_axes, strict=True)
        )
        outputs.append(
            _as_real_numeric_array("program AD elementwise batched output", function(*sliced_args))
        )
    stacked = np.stack(outputs, axis=0)
    return np.moveaxis(stacked, 0, _normalise_axis("out_axes", out_axes, stacked.ndim))


def _program_ad_elementwise_lowering_metadata(name: str) -> Mapping[str, str]:
    is_binary = name in _PROGRAM_AD_ELEMENTWISE_BINARY_NAMES
    is_derivative_losing = name in _PROGRAM_AD_ELEMENTWISE_DISCONTINUOUS_NAMES
    metadata = {
        "program_ad": "operator_intercepted_trace",
        "mlir": "available: scpn_diff elementwise dialect interchange; executable lowering blocked",
        "mlir_op": f"scpn_diff.elementwise.{name}",
        "llvm": "blocked_until_executable_elementwise_lowering",
        "rust": "blocked_until_polyglot_elementwise_ad",
        "static_argument_rule": "none",
        "static_derivative_factory": (
            "blocked_derivative_losing"
            if is_derivative_losing
            else "program_ad_elementwise_binary_derivative_rule"
            if is_binary
            else "not_required"
        ),
        "static_signature": (
            "left_shape:ranked_tensor_shape;right_shape:ranked_tensor_shape"
            if is_binary
            else "source_shape:ranked_tensor_shape;step_value"
            if name == "heaviside"
            else "source_shape:ranked_tensor_shape"
            if is_derivative_losing
            else "none"
        ),
    }
    nondifferentiable_boundaries = {
        "log": "positive_domain",
        "log1p": "greater_than_minus_one_domain",
        "sqrt": "nonnegative_domain_with_singular_zero_derivative",
        "arcsin": "closed_unit_interval_with_singular_endpoints",
        "arccos": "closed_unit_interval_with_singular_endpoints",
        "reciprocal": "nonzero_domain",
        "abs": "zero_cusp",
        "divide": "nonzero_denominator",
        "power": "positive_base_for_variable_exponent",
        "maximum": "equal_operand_tie",
        "minimum": "equal_operand_tie",
        "sign": "sign_step_derivative_losing_boundary",
        "heaviside": "heaviside_step_derivative_losing_boundary",
    }
    boundary = nondifferentiable_boundaries.get(name)
    if boundary is not None:
        metadata["nondifferentiable_boundary"] = boundary
    else:
        metadata["nondifferentiable_boundary"] = "none"
    metadata["nondifferentiable_boundary_policy"] = "fail_closed"
    return metadata


def _register_program_ad_elementwise_primitive_contracts() -> None:
    for name, identity in _PROGRAM_AD_ELEMENTWISE_IDENTITIES.items():
        if DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.contract_for(identity) is not None:
            continue
        DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.register_transform(
            PrimitiveTransformRule(
                identity=identity,
                derivative_rule=_program_ad_elementwise_derivative_rule(name),
                batching_rule=_program_ad_elementwise_batching_rule,
                lowering_metadata=_program_ad_elementwise_lowering_metadata(name),
                shape_rule=_PROGRAM_AD_ELEMENTWISE_SHAPE_RULES[name],
                dtype_rule=_program_ad_elementwise_dtype_rule,
                static_argument_rule=_program_ad_elementwise_static_arguments,
                nondifferentiable_policy=_PROGRAM_AD_ELEMENTWISE_POLICY,
                effect="pure",
            )
        )


def _require_program_ad_elementwise_contract(
    name: str,
    args: tuple[object, ...] | None = None,
) -> PrimitiveContract:
    """Return a validated elementwise primitive runtime contract."""
    identity = _PROGRAM_AD_ELEMENTWISE_IDENTITIES.get(name)
    if identity is None:
        raise ValueError(f"no program AD elementwise primitive identity registered for {name}")
    contract = DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.require_contract(identity)
    if contract.nondifferentiable_policy != _PROGRAM_AD_ELEMENTWISE_POLICY:
        raise ValueError(f"invalid program AD elementwise primitive policy for {identity.key}")
    if contract.effect != "pure":
        raise ValueError(f"invalid program AD elementwise primitive effect for {identity.key}")

    missing: list[str] = []
    if contract.batching_rule is None:
        missing.append("batching_rule")
    if not contract.lowering_metadata:
        missing.append("lowering_metadata")
    if not contract.lowering_metadata.get("mlir_op"):
        missing.append("mlir_op")
    if not contract.lowering_metadata.get("nondifferentiable_boundary"):
        missing.append("nondifferentiable_boundary")
    if contract.lowering_metadata.get("nondifferentiable_boundary_policy") != "fail_closed":
        missing.append("nondifferentiable_boundary_policy")
    if contract.shape_rule is None:
        missing.append("shape_rule")
    if contract.dtype_rule is None:
        missing.append("dtype_rule")
    if contract.static_argument_rule is None:
        missing.append("static_argument_rule")
    if missing:
        joined = ", ".join(missing)
        raise ValueError(
            f"incomplete program AD elementwise primitive runtime contract for "
            f"{identity.key}: missing {joined}"
        )
    if args is not None:
        _validate_program_ad_elementwise_contract_dispatch(contract, args)
    return contract
