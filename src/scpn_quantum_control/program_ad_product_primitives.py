# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — program AD product primitives module
# scpn-quantum-control -- Program AD product primitive rules
"""Program AD product/contraction derivative factories and registry contracts."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import cast

import numpy as np
from numpy.typing import NDArray

from .differentiable_parameter_contracts import _as_real_numeric_array
from .program_ad_array_indexing import _normalise_axis
from .program_ad_registry import (
    _PROGRAM_AD_PRODUCT_IDENTITIES,
    _PROGRAM_AD_PRODUCT_POLICY,
    DEFAULT_CUSTOM_DERIVATIVE_REGISTRY,
    CustomDerivativeRule,
    PrimitiveContract,
    PrimitiveShapeRule,
    PrimitiveTransformRule,
)
from .program_ad_shape_transforms import (
    _program_ad_float64_vector_result,
    _program_ad_shape_signature,
    _program_ad_shape_static_size,
)


def _is_trace_array(value: object) -> bool:
    """Return whether ``value`` behaves like a whole-program trace array."""
    return type(value).__name__ == "TraceADArray" and hasattr(value, "context")


def _program_ad_array_shape_of(value: object) -> tuple[int, ...]:
    """Return the static array shape recorded by a trace value or array-like input."""
    if _is_trace_array(value):
        shape = getattr(value, "shape", None)
        if not isinstance(shape, tuple):
            raise ValueError("program AD product trace array shape must be static")
        return tuple(int(dimension) for dimension in shape)
    return tuple(int(dim) for dim in np.asarray(value).shape)


def _program_ad_array_dtype_of(value: object) -> str:
    """Return the dtype name recorded by a trace value or array-like input."""
    if _is_trace_array(value):
        return "float64"
    array = np.asarray(value)
    if array.dtype.kind in {"O", "S", "U", "c"}:
        raise ValueError("program AD product primitive dtype rule requires real numeric arrays")
    return str(array.dtype)


def _validate_program_ad_product_contract_dispatch(
    contract: PrimitiveContract,
    args: tuple[object, ...],
) -> None:
    """Validate product primitive dispatch helpers against concrete arguments."""
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


def _parse_static_einsum_subscripts(
    subscripts: str,
    operand_shapes: Sequence[Sequence[int]],
) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...], dict[str, int]]:
    normalised = subscripts.replace(" ", "")
    if "..." in normalised:
        raise ValueError("whole-program AD np.einsum ellipsis forms require explicit expansion")
    if "->" not in normalised:
        raise ValueError("whole-program AD np.einsum requires an explicit output subscript")
    if normalised.count("->") != 1:
        raise ValueError("whole-program AD np.einsum requires one explicit output separator")
    input_spec, output_spec = normalised.split("->", 1)
    input_labels = tuple(tuple(part) for part in input_spec.split(","))
    output_labels = tuple(output_spec)
    if len(input_labels) != len(operand_shapes):
        raise ValueError("whole-program AD np.einsum operand count must match subscripts")
    if len(set(output_labels)) != len(output_labels):
        raise ValueError("whole-program AD np.einsum output labels must be unique")
    if any(not label.isalpha() for labels in input_labels for label in labels) or any(
        not label.isalpha() for label in output_labels
    ):
        raise ValueError("whole-program AD np.einsum supports alphabetic labels only")

    dimensions: dict[str, int] = {}
    seen_labels: set[str] = set()
    for labels, raw_shape in zip(input_labels, operand_shapes, strict=True):
        shape = tuple(int(dimension) for dimension in raw_shape)
        if len(labels) != len(shape):
            raise ValueError("whole-program AD np.einsum labels must match operand rank")
        local_dimensions: dict[str, int] = {}
        for label, dimension in zip(labels, shape, strict=True):
            if dimension <= 0:
                raise ValueError("whole-program AD np.einsum operand dimensions must be positive")
            seen_labels.add(label)
            previous = dimensions.get(label)
            if previous is not None and previous != dimension:
                raise ValueError("whole-program AD np.einsum label dimensions must agree")
            local_previous = local_dimensions.get(label)
            if local_previous is not None and local_previous != dimension:
                raise ValueError("whole-program AD np.einsum repeated-label dimensions must agree")
            dimensions[label] = dimension
            local_dimensions[label] = dimension
    missing_output = set(output_labels) - seen_labels
    if missing_output:
        raise ValueError("whole-program AD np.einsum output labels must appear in operands")
    return output_labels, input_labels, dimensions


def _program_ad_product_split_pair(
    primitive_name: str,
    values: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    vector = _as_real_numeric_array(f"program AD product {primitive_name} values", values).reshape(
        -1
    )
    if vector.size == 0 or vector.size % 2 != 0:
        raise ValueError(
            f"program AD product {primitive_name} direct rule requires two equal flat operands"
        )
    midpoint = vector.size // 2
    return vector[:midpoint], vector[midpoint:]


def _program_ad_product_dot_value(values: NDArray[np.float64]) -> NDArray[np.float64]:
    left, right = _program_ad_product_split_pair("dot", values)
    return np.array([float(np.dot(left, right))], dtype=np.float64)


def _program_ad_product_dot_jvp(
    values: NDArray[np.float64],
    tangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    left, right = _program_ad_product_split_pair("dot", values)
    tangent_left, tangent_right = _program_ad_product_split_pair("dot tangent", tangent)
    if tangent_left.shape != left.shape or tangent_right.shape != right.shape:
        raise ValueError("program AD product dot tangent shape must match values shape")
    return np.array(
        [float(np.dot(tangent_left, right) + np.dot(left, tangent_right))], dtype=np.float64
    )


def _program_ad_product_scalar_cotangent(
    primitive_name: str,
    cotangent: NDArray[np.float64],
) -> float:
    cotangent_vector = _as_real_numeric_array(
        f"program AD product {primitive_name} cotangent", cotangent
    ).reshape(-1)
    if cotangent_vector.shape != (1,):
        raise ValueError(f"program AD product {primitive_name} VJP requires one scalar cotangent")
    return float(cotangent_vector[0])


def _program_ad_product_dot_vjp(
    values: NDArray[np.float64],
    cotangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    left, right = _program_ad_product_split_pair("dot", values)
    scalar_cotangent = _program_ad_product_scalar_cotangent("dot", cotangent)
    return _program_ad_float64_vector_result(
        np.concatenate((scalar_cotangent * right, scalar_cotangent * left))
    )


def _program_ad_product_vdot_value(values: NDArray[np.float64]) -> NDArray[np.float64]:
    left, right = _program_ad_product_split_pair("vdot", values)
    return np.array([float(np.vdot(left, right))], dtype=np.float64)


def _program_ad_product_vdot_jvp(
    values: NDArray[np.float64],
    tangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    left, right = _program_ad_product_split_pair("vdot", values)
    tangent_left, tangent_right = _program_ad_product_split_pair("vdot tangent", tangent)
    if tangent_left.shape != left.shape or tangent_right.shape != right.shape:
        raise ValueError("program AD product vdot tangent shape must match values shape")
    return np.array(
        [float(np.vdot(tangent_left, right) + np.vdot(left, tangent_right))],
        dtype=np.float64,
    )


def _program_ad_product_vdot_vjp(
    values: NDArray[np.float64],
    cotangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    left, right = _program_ad_product_split_pair("vdot", values)
    scalar_cotangent = _program_ad_product_scalar_cotangent("vdot", cotangent)
    return _program_ad_float64_vector_result(
        np.concatenate((scalar_cotangent * right, scalar_cotangent * left))
    )


def _program_ad_product_inner_value(values: NDArray[np.float64]) -> NDArray[np.float64]:
    left, right = _program_ad_product_split_pair("inner", values)
    return _program_ad_float64_vector_result([float(np.inner(left, right))])


def _program_ad_product_inner_jvp(
    values: NDArray[np.float64],
    tangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    left, right = _program_ad_product_split_pair("inner", values)
    tangent_left, tangent_right = _program_ad_product_split_pair("inner tangent", tangent)
    if tangent_left.shape != left.shape or tangent_right.shape != right.shape:
        raise ValueError("program AD product inner tangent shape must match values shape")
    return _program_ad_float64_vector_result(
        [float(np.inner(tangent_left, right) + np.inner(left, tangent_right))]
    )


def _program_ad_product_inner_vjp(
    values: NDArray[np.float64],
    cotangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    left, right = _program_ad_product_split_pair("inner", values)
    scalar_cotangent = _program_ad_product_scalar_cotangent("inner", cotangent)
    return _program_ad_float64_vector_result(
        np.concatenate((scalar_cotangent * right, scalar_cotangent * left))
    )


def _program_ad_product_outer_value(values: NDArray[np.float64]) -> NDArray[np.float64]:
    left, right = _program_ad_product_split_pair("outer", values)
    return _program_ad_float64_vector_result(np.outer(left, right))


def _program_ad_product_outer_jvp(
    values: NDArray[np.float64],
    tangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    left, right = _program_ad_product_split_pair("outer", values)
    tangent_left, tangent_right = _program_ad_product_split_pair("outer tangent", tangent)
    if tangent_left.shape != left.shape or tangent_right.shape != right.shape:
        raise ValueError("program AD product outer tangent shape must match values shape")
    return _program_ad_float64_vector_result(
        np.outer(tangent_left, right) + np.outer(left, tangent_right)
    )


def _program_ad_product_outer_vjp(
    values: NDArray[np.float64],
    cotangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    left, right = _program_ad_product_split_pair("outer", values)
    cotangent_vector = _as_real_numeric_array(
        "program AD product outer cotangent", cotangent
    ).reshape(-1)
    if cotangent_vector.size != left.size * right.size:
        raise ValueError("program AD product outer VJP cotangent shape must match output shape")
    cotangent_matrix = cotangent_vector.reshape(left.size, right.size)
    return _program_ad_float64_vector_result(
        np.concatenate((cotangent_matrix @ right, cotangent_matrix.T @ left))
    )


def _program_ad_product_square_matrix_pair(
    primitive_name: str,
    values: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    left, right = _program_ad_product_split_pair(primitive_name, values)
    rows = int(math.isqrt(left.size))
    if rows * rows != left.size:
        raise ValueError(
            f"program AD product {primitive_name} direct rule requires two square matrices"
        )
    return left.reshape(rows, rows), right.reshape(rows, rows)


def _program_ad_product_matmul_value(values: NDArray[np.float64]) -> NDArray[np.float64]:
    left, right = _program_ad_product_square_matrix_pair("matmul", values)
    return (left @ right).reshape(-1).astype(np.float64)


def _program_ad_product_matmul_jvp(
    values: NDArray[np.float64],
    tangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    left, right = _program_ad_product_square_matrix_pair("matmul", values)
    tangent_left, tangent_right = _program_ad_product_square_matrix_pair("matmul tangent", tangent)
    if tangent_left.shape != left.shape or tangent_right.shape != right.shape:
        raise ValueError("program AD product matmul tangent shape must match values shape")
    return (tangent_left @ right + left @ tangent_right).reshape(-1).astype(np.float64)


def _program_ad_product_matmul_vjp(
    values: NDArray[np.float64],
    cotangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    left, right = _program_ad_product_square_matrix_pair("matmul", values)
    cotangent_matrix = _as_real_numeric_array(
        "program AD product matmul cotangent", cotangent
    ).reshape(-1)
    if cotangent_matrix.shape != (left.size,):
        raise ValueError("program AD product matmul VJP cotangent shape must match output shape")
    cotangent_square = cotangent_matrix.reshape(left.shape)
    return _program_ad_float64_vector_result(
        np.concatenate(
            (
                (cotangent_square @ right.T).reshape(-1),
                (left.T @ cotangent_square).reshape(-1),
            )
        )
    )


def _program_ad_product_normalise_matmul_shapes(
    left_shape: Sequence[int],
    right_shape: Sequence[int],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    left = tuple(int(dimension) for dimension in left_shape)
    right = tuple(int(dimension) for dimension in right_shape)
    if any(dimension < 0 for dimension in (*left, *right)):
        raise ValueError("program AD product matmul direct rule requires non-negative dimensions")
    if len(left) not in {1, 2} or len(right) not in {1, 2}:
        raise ValueError(
            "program AD product matmul direct rule supports rank-1 or rank-2 operands"
        )
    if len(left) == 1 and len(right) == 1:
        if left[0] != right[0]:
            raise ValueError("program AD product matmul direct rule dimensions must align")
        return left, right, ()
    if len(left) == 2 and len(right) == 1:
        if left[1] != right[0]:
            raise ValueError("program AD product matmul direct rule dimensions must align")
        return left, right, (left[0],)
    if len(left) == 1 and len(right) == 2:
        if left[0] != right[0]:
            raise ValueError("program AD product matmul direct rule dimensions must align")
        return left, right, (right[1],)
    if left[1] != right[0]:
        raise ValueError("program AD product matmul direct rule dimensions must align")
    return left, right, (left[0], right[1])


def _program_ad_product_matmul_static_split(
    role: str,
    values: NDArray[np.float64],
    *,
    left_shape: tuple[int, ...],
    right_shape: tuple[int, ...],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    vector = _as_real_numeric_array(f"program AD product matmul {role}", values).reshape(-1)
    left_size = _program_ad_shape_static_size(left_shape)
    right_size = _program_ad_shape_static_size(right_shape)
    if vector.size != left_size + right_size:
        raise ValueError(
            "program AD product matmul direct rule requires flattened left operand "
            "followed by right operand"
        )
    return (
        vector[:left_size].reshape(left_shape),
        vector[left_size:].reshape(right_shape),
    )


def program_ad_product_matmul_derivative_rule(
    left_shape: Sequence[int],
    right_shape: Sequence[int],
) -> CustomDerivativeRule:
    """Build a direct value/JVP/VJP rule for a fixed matmul primitive signature."""
    left_static_shape, right_static_shape, output_shape = (
        _program_ad_product_normalise_matmul_shapes(left_shape, right_shape)
    )

    def value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
        left, right = _program_ad_product_matmul_static_split(
            "values", values, left_shape=left_static_shape, right_shape=right_static_shape
        )
        return _program_ad_float64_vector_result(left @ right)

    def jvp_rule(values: NDArray[np.float64], tangent: NDArray[np.float64]) -> NDArray[np.float64]:
        left, right = _program_ad_product_matmul_static_split(
            "values", values, left_shape=left_static_shape, right_shape=right_static_shape
        )
        tangent_left, tangent_right = _program_ad_product_matmul_static_split(
            "tangent", tangent, left_shape=left_static_shape, right_shape=right_static_shape
        )
        return _program_ad_float64_vector_result(tangent_left @ right + left @ tangent_right)

    def vjp_rule(
        values: NDArray[np.float64], cotangent: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        left, right = _program_ad_product_matmul_static_split(
            "values", values, left_shape=left_static_shape, right_shape=right_static_shape
        )
        cotangent_vector = _as_real_numeric_array(
            "program AD product matmul cotangent", cotangent
        ).reshape(-1)
        expected_size = _program_ad_shape_static_size(output_shape)
        if cotangent_vector.size != expected_size:
            raise ValueError(
                "program AD product matmul VJP cotangent shape must match output shape"
            )
        cotangent_value = (
            float(cotangent_vector[0])
            if output_shape == ()
            else cotangent_vector.reshape(output_shape)
        )
        left_adjoint: NDArray[np.float64]
        right_adjoint: NDArray[np.float64]
        if left.ndim == 1 and right.ndim == 1:
            scalar = float(cotangent_value)
            left_adjoint = scalar * right
            right_adjoint = scalar * left
        elif left.ndim == 2 and right.ndim == 1:
            cotangent_array = cast(NDArray[np.float64], cotangent_value)
            left_adjoint = _program_ad_float64_vector_result(
                np.outer(cotangent_array, right)
            ).reshape(left.shape)
            right_adjoint = left.T @ cotangent_array
        elif left.ndim == 1 and right.ndim == 2:
            cotangent_array = cast(NDArray[np.float64], cotangent_value)
            left_adjoint = right @ cotangent_array
            right_adjoint = _program_ad_float64_vector_result(
                np.outer(left, cotangent_array)
            ).reshape(right.shape)
        else:
            cotangent_array = cast(NDArray[np.float64], cotangent_value)
            left_adjoint = cotangent_array @ right.T
            right_adjoint = left.T @ cotangent_array
        return _program_ad_float64_vector_result(
            np.concatenate(
                (np.asarray(left_adjoint).reshape(-1), np.asarray(right_adjoint).reshape(-1))
            )
        )

    return CustomDerivativeRule(
        name=(
            "program_ad_product_matmul_"
            f"{_program_ad_shape_signature(left_static_shape)}_by_"
            f"{_program_ad_shape_signature(right_static_shape)}_direct_rule"
        ),
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
    )


def _program_ad_product_normalise_inner_shapes(
    left_shape: Sequence[int],
    right_shape: Sequence[int],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    left = tuple(int(dimension) for dimension in left_shape)
    right = tuple(int(dimension) for dimension in right_shape)
    if not left or not right:
        raise ValueError("program AD product inner direct rule requires non-scalar operands")
    if any(dimension <= 0 for dimension in (*left, *right)):
        raise ValueError("program AD product inner direct rule dimensions must be positive")
    if left[-1] != right[-1]:
        raise ValueError("program AD product inner direct rule last dimensions must align")
    return left, right, left[:-1] + right[:-1]


def _program_ad_product_static_split_pair(
    primitive_name: str,
    values: NDArray[np.float64],
    *,
    left_shape: tuple[int, ...],
    right_shape: tuple[int, ...],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    vector = _as_real_numeric_array(f"program AD product {primitive_name} values", values).reshape(
        -1
    )
    left_size = _program_ad_shape_static_size(left_shape)
    right_size = _program_ad_shape_static_size(right_shape)
    if vector.size != left_size + right_size:
        raise ValueError(
            f"program AD product {primitive_name} direct rule requires flattened left "
            "operand followed by right operand"
        )
    return (
        vector[:left_size].reshape(left_shape),
        vector[left_size:].reshape(right_shape),
    )


def _normalise_program_ad_product_tensordot_axis_sequence(
    label: str,
    axes: object,
    rank: int,
) -> tuple[int, ...]:
    if isinstance(axes, (bool, np.bool_)):
        raise ValueError(f"program AD product tensordot {label} axes must be static integers")
    if isinstance(axes, (int, np.integer)):
        return (_normalise_axis(label, int(axes), rank),)
    if isinstance(axes, np.ndarray):
        raw_axes = tuple(axes.reshape(-1).tolist())
    elif isinstance(axes, Sequence) and not isinstance(axes, (str, bytes)):
        raw_axes = tuple(axes)
    else:
        raise ValueError(f"program AD product tensordot {label} axes must be static integers")
    normalised: list[int] = []
    for raw_axis in raw_axes:
        if isinstance(raw_axis, (bool, np.bool_)) or not isinstance(raw_axis, (int, np.integer)):
            raise ValueError(f"program AD product tensordot {label} axes must be static integers")
        normalised.append(_normalise_axis(label, int(raw_axis), rank))
    if len(set(normalised)) != len(normalised):
        raise ValueError(f"program AD product tensordot {label} axes must be unique")
    return tuple(normalised)


def _normalise_program_ad_product_tensordot_shape(shape: Sequence[int]) -> tuple[int, ...]:
    static_shape = tuple(int(dimension) for dimension in shape)
    if any(dimension <= 0 for dimension in static_shape):
        raise ValueError("program AD product tensordot dimensions must be positive")
    return static_shape


def _normalise_program_ad_product_tensordot_signature(
    left_shape: Sequence[int],
    right_shape: Sequence[int],
    axes: object,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    left = _normalise_program_ad_product_tensordot_shape(left_shape)
    right = _normalise_program_ad_product_tensordot_shape(right_shape)
    if isinstance(axes, (bool, np.bool_)):
        raise ValueError("program AD product tensordot axes must be a static integer or pair")
    if isinstance(axes, (int, np.integer)):
        axis_count = int(axes)
        if axis_count < 0:
            raise ValueError("program AD product tensordot axis count must be non-negative")
        if axis_count > min(len(left), len(right)):
            raise ValueError("program AD product tensordot axis count exceeds operand rank")
        left_axes = tuple(range(len(left) - axis_count, len(left)))
        right_axes = tuple(range(axis_count))
    elif isinstance(axes, Sequence) and not isinstance(axes, (str, bytes)) and len(axes) == 2:
        left_axes = _normalise_program_ad_product_tensordot_axis_sequence(
            "left",
            axes[0],
            len(left),
        )
        right_axes = _normalise_program_ad_product_tensordot_axis_sequence(
            "right",
            axes[1],
            len(right),
        )
    else:
        raise ValueError("program AD product tensordot axes must be a static integer or pair")
    if len(left_axes) != len(right_axes):
        raise ValueError("program AD product tensordot axis lists must have equal length")
    for left_axis, right_axis in zip(left_axes, right_axes, strict=True):
        if left[left_axis] != right[right_axis]:
            raise ValueError("program AD product tensordot contracted dimensions must align")
    left_free = tuple(axis for axis in range(len(left)) if axis not in left_axes)
    right_free = tuple(axis for axis in range(len(right)) if axis not in right_axes)
    output_shape = tuple(left[axis] for axis in left_free) + tuple(
        right[axis] for axis in right_free
    )
    return left, right, left_axes, right_axes, output_shape


def _program_ad_product_tensordot_axes_signature(
    left_axes: tuple[int, ...],
    right_axes: tuple[int, ...],
) -> str:
    left = "_".join(str(axis) for axis in left_axes) if left_axes else "none"
    right = "_".join(str(axis) for axis in right_axes) if right_axes else "none"
    return f"{left}_by_{right}"


def program_ad_product_tensordot_derivative_rule(
    left_shape: Sequence[int],
    right_shape: Sequence[int],
    *,
    axes: object = 2,
) -> CustomDerivativeRule:
    """Build a direct value/JVP/VJP rule for a fixed ``np.tensordot`` signature."""
    left_static_shape, right_static_shape, left_axes, right_axes, output_shape = (
        _normalise_program_ad_product_tensordot_signature(left_shape, right_shape, axes)
    )
    output_size = _program_ad_shape_static_size(output_shape)
    normalised_axes = (left_axes, right_axes)

    def value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
        left, right = _program_ad_product_static_split_pair(
            "tensordot", values, left_shape=left_static_shape, right_shape=right_static_shape
        )
        return _program_ad_float64_vector_result(np.tensordot(left, right, axes=normalised_axes))

    def jvp_rule(values: NDArray[np.float64], tangent: NDArray[np.float64]) -> NDArray[np.float64]:
        left, right = _program_ad_product_static_split_pair(
            "tensordot", values, left_shape=left_static_shape, right_shape=right_static_shape
        )
        tangent_left, tangent_right = _program_ad_product_static_split_pair(
            "tensordot tangent",
            tangent,
            left_shape=left_static_shape,
            right_shape=right_static_shape,
        )
        return _program_ad_float64_vector_result(
            np.tensordot(tangent_left, right, axes=normalised_axes)
            + np.tensordot(left, tangent_right, axes=normalised_axes)
        )

    def vjp_rule(
        values: NDArray[np.float64],
        cotangent: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        left, right = _program_ad_product_static_split_pair(
            "tensordot", values, left_shape=left_static_shape, right_shape=right_static_shape
        )
        cotangent_vector = _as_real_numeric_array(
            "program AD product tensordot cotangent", cotangent
        ).reshape(-1)
        if cotangent_vector.size != output_size:
            raise ValueError(
                "program AD product tensordot VJP cotangent size must match output shape"
            )
        left_adjoint = np.zeros_like(left, dtype=np.float64)
        right_adjoint = np.zeros_like(right, dtype=np.float64)
        for element_index in np.ndindex(left.shape):
            basis = np.zeros_like(left, dtype=np.float64)
            basis[element_index] = 1.0
            left_adjoint[element_index] = float(
                np.dot(
                    cotangent_vector,
                    _program_ad_float64_vector_result(
                        np.tensordot(basis, right, axes=normalised_axes)
                    ),
                )
            )
        for element_index in np.ndindex(right.shape):
            basis = np.zeros_like(right, dtype=np.float64)
            basis[element_index] = 1.0
            right_adjoint[element_index] = float(
                np.dot(
                    cotangent_vector,
                    _program_ad_float64_vector_result(
                        np.tensordot(left, basis, axes=normalised_axes)
                    ),
                )
            )
        return _program_ad_float64_vector_result(
            np.concatenate((left_adjoint.reshape(-1), right_adjoint.reshape(-1)))
        )

    return CustomDerivativeRule(
        name=(
            "program_ad_product_tensordot_"
            f"{_program_ad_shape_signature(left_static_shape)}_by_"
            f"{_program_ad_shape_signature(right_static_shape)}_axes_"
            f"{_program_ad_product_tensordot_axes_signature(left_axes, right_axes)}_direct_rule"
        ),
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
    )


def program_ad_product_inner_derivative_rule(
    left_shape: Sequence[int],
    right_shape: Sequence[int],
) -> CustomDerivativeRule:
    """Build a direct value/JVP/VJP rule for a fixed inner-product signature."""
    left_static_shape, right_static_shape, output_shape = (
        _program_ad_product_normalise_inner_shapes(left_shape, right_shape)
    )
    expected_output_size = _program_ad_shape_static_size(output_shape)

    def value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
        left, right = _program_ad_product_static_split_pair(
            "inner", values, left_shape=left_static_shape, right_shape=right_static_shape
        )
        return _program_ad_float64_vector_result(np.inner(left, right))

    def jvp_rule(values: NDArray[np.float64], tangent: NDArray[np.float64]) -> NDArray[np.float64]:
        left, right = _program_ad_product_static_split_pair(
            "inner", values, left_shape=left_static_shape, right_shape=right_static_shape
        )
        tangent_left, tangent_right = _program_ad_product_static_split_pair(
            "inner tangent",
            tangent,
            left_shape=left_static_shape,
            right_shape=right_static_shape,
        )
        return _program_ad_float64_vector_result(
            np.inner(tangent_left, right) + np.inner(left, tangent_right)
        )

    def vjp_rule(
        values: NDArray[np.float64], cotangent: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        left, right = _program_ad_product_static_split_pair(
            "inner", values, left_shape=left_static_shape, right_shape=right_static_shape
        )
        cotangent_vector = _as_real_numeric_array(
            "program AD product inner cotangent", cotangent
        ).reshape(-1)
        if cotangent_vector.size != expected_output_size:
            raise ValueError(
                "program AD product inner VJP cotangent shape must match output shape"
            )
        cotangent_array = (
            cotangent_vector.reshape(output_shape)
            if output_shape
            else np.asarray(cotangent_vector[0], dtype=np.float64)
        )
        left_outer_rank = left.ndim - 1
        right_outer_rank = right.ndim - 1
        left_adjoint = np.tensordot(
            cotangent_array,
            right,
            axes=(
                tuple(range(left_outer_rank, left_outer_rank + right_outer_rank)),
                tuple(range(right_outer_rank)),
            ),
        )
        right_adjoint = np.tensordot(
            cotangent_array,
            left,
            axes=(tuple(range(left_outer_rank)), tuple(range(left_outer_rank))),
        )
        return _program_ad_float64_vector_result(
            np.concatenate(
                (np.asarray(left_adjoint).reshape(-1), np.asarray(right_adjoint).reshape(-1))
            )
        )

    return CustomDerivativeRule(
        name=(
            "program_ad_product_inner_"
            f"{_program_ad_shape_signature(left_static_shape)}_by_"
            f"{_program_ad_shape_signature(right_static_shape)}_direct_rule"
        ),
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
    )


def _program_ad_product_normalise_outer_shapes(
    left_shape: Sequence[int],
    right_shape: Sequence[int],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    left = tuple(int(dimension) for dimension in left_shape)
    right = tuple(int(dimension) for dimension in right_shape)
    if any(dimension <= 0 for dimension in (*left, *right)):
        raise ValueError("program AD product outer direct rule dimensions must be positive")
    return left, right


def program_ad_product_outer_derivative_rule(
    left_shape: Sequence[int],
    right_shape: Sequence[int],
) -> CustomDerivativeRule:
    """Build a direct value/JVP/VJP rule for a fixed outer-product signature."""
    left_static_shape, right_static_shape = _program_ad_product_normalise_outer_shapes(
        left_shape, right_shape
    )
    left_size = _program_ad_shape_static_size(left_static_shape)
    right_size = _program_ad_shape_static_size(right_static_shape)

    def value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
        left, right = _program_ad_product_static_split_pair(
            "outer", values, left_shape=left_static_shape, right_shape=right_static_shape
        )
        return _program_ad_float64_vector_result(np.outer(left.reshape(-1), right.reshape(-1)))

    def jvp_rule(values: NDArray[np.float64], tangent: NDArray[np.float64]) -> NDArray[np.float64]:
        left, right = _program_ad_product_static_split_pair(
            "outer", values, left_shape=left_static_shape, right_shape=right_static_shape
        )
        tangent_left, tangent_right = _program_ad_product_static_split_pair(
            "outer tangent",
            tangent,
            left_shape=left_static_shape,
            right_shape=right_static_shape,
        )
        return _program_ad_float64_vector_result(
            np.outer(tangent_left.reshape(-1), right.reshape(-1))
            + np.outer(left.reshape(-1), tangent_right.reshape(-1))
        )

    def vjp_rule(
        values: NDArray[np.float64], cotangent: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        left, right = _program_ad_product_static_split_pair(
            "outer", values, left_shape=left_static_shape, right_shape=right_static_shape
        )
        cotangent_vector = _as_real_numeric_array(
            "program AD product outer cotangent", cotangent
        ).reshape(-1)
        if cotangent_vector.size != left_size * right_size:
            raise ValueError(
                "program AD product outer VJP cotangent shape must match output shape"
            )
        cotangent_matrix = cotangent_vector.reshape(left_size, right_size)
        left_adjoint = (cotangent_matrix @ right.reshape(-1)).reshape(left_static_shape)
        right_adjoint = (cotangent_matrix.T @ left.reshape(-1)).reshape(right_static_shape)
        return _program_ad_float64_vector_result(
            np.concatenate((left_adjoint.reshape(-1), right_adjoint.reshape(-1)))
        )

    return CustomDerivativeRule(
        name=(
            "program_ad_product_outer_"
            f"{_program_ad_shape_signature(left_static_shape)}_by_"
            f"{_program_ad_shape_signature(right_static_shape)}_direct_rule"
        ),
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
    )


def _normalise_program_ad_product_einsum_shapes(
    operand_shapes: Sequence[Sequence[int]],
) -> tuple[tuple[int, ...], ...]:
    shapes = tuple(tuple(int(dimension) for dimension in shape) for shape in operand_shapes)
    if not shapes:
        raise ValueError("program AD product einsum derivative rule requires operands")
    if any(any(dimension <= 0 for dimension in shape) for shape in shapes):
        raise ValueError("program AD product einsum derivative rule dimensions must be positive")
    return shapes


def _split_program_ad_product_einsum_operands(
    name: str,
    values: NDArray[np.float64],
    shapes: tuple[tuple[int, ...], ...],
) -> tuple[NDArray[np.float64], ...]:
    vector = _as_real_numeric_array(f"program AD product einsum {name}", values).reshape(-1)
    expected = sum(_program_ad_shape_static_size(shape) for shape in shapes)
    if vector.size != expected:
        raise ValueError("program AD product einsum direct rule values size must match shapes")
    operands: list[NDArray[np.float64]] = []
    offset = 0
    for shape in shapes:
        size = _program_ad_shape_static_size(shape)
        operands.append(vector[offset : offset + size].reshape(shape))
        offset += size
    return tuple(operands)


def program_ad_product_einsum_derivative_rule(
    subscripts: str,
    operand_shapes: Sequence[Sequence[int]],
) -> CustomDerivativeRule:
    """Build a direct value/JVP/VJP rule for fixed explicit ``np.einsum`` signatures."""
    shapes = _normalise_program_ad_product_einsum_shapes(operand_shapes)
    normalised = subscripts.replace(" ", "")
    output_labels, input_labels, dimensions = _parse_static_einsum_subscripts(
        normalised,
        shapes,
    )
    output_shape = tuple(dimensions[label] for label in output_labels)
    output_size = _program_ad_shape_static_size(output_shape)

    def flat_einsum(operands: tuple[NDArray[np.float64], ...]) -> NDArray[np.float64]:
        result = np.einsum(normalised, *operands)
        return _program_ad_float64_vector_result(result)

    def value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
        operands = _split_program_ad_product_einsum_operands("values", values, shapes)
        return flat_einsum(operands)

    def jvp_rule(
        values: NDArray[np.float64],
        tangent: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        operands = _split_program_ad_product_einsum_operands("values", values, shapes)
        tangent_operands = _split_program_ad_product_einsum_operands("tangent", tangent, shapes)
        total = np.zeros(output_size, dtype=np.float64)
        for operand_index, tangent_operand in enumerate(tangent_operands):
            varied = operands[:operand_index] + (tangent_operand,) + operands[operand_index + 1 :]
            total += flat_einsum(varied)
        return _program_ad_float64_vector_result(total)

    def vjp_rule(
        values: NDArray[np.float64],
        cotangent: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        operands = _split_program_ad_product_einsum_operands("values", values, shapes)
        cotangent_vector = _as_real_numeric_array(
            "program AD product einsum cotangent", cotangent
        ).reshape(-1)
        if cotangent_vector.size != output_size:
            raise ValueError(
                "program AD product einsum VJP cotangent size must match output shape"
            )
        adjoints: list[NDArray[np.float64]] = []
        for operand_index, operand in enumerate(operands):
            operand_adjoint = np.zeros_like(operand, dtype=np.float64)
            for element_index in np.ndindex(operand.shape):
                basis = np.zeros_like(operand, dtype=np.float64)
                basis[element_index] = 1.0
                varied = operands[:operand_index] + (basis,) + operands[operand_index + 1 :]
                operand_adjoint[element_index] = float(
                    np.dot(cotangent_vector, flat_einsum(varied))
                )
            adjoints.append(operand_adjoint.reshape(-1))
        return _program_ad_float64_vector_result(np.concatenate(adjoints))

    label_signature = "_".join(
        "".join(labels) for labels in (*input_labels, ("".join(output_labels),))
    )
    shape_signature = "_by_".join(_program_ad_shape_signature(shape) for shape in shapes)
    return CustomDerivativeRule(
        name=f"program_ad_product_einsum_{label_signature}_{shape_signature}_direct_rule",
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
    )


def _program_ad_product_einsum_unconfigured_value(
    values: NDArray[np.float64],
) -> NDArray[np.float64]:
    del values
    raise ValueError(
        "program AD product einsum direct rule requires fixed subscripts and operand shapes"
    )


def _program_ad_product_einsum_unconfigured_jvp(
    values: NDArray[np.float64],
    tangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    del values, tangent
    raise ValueError(
        "program AD product einsum JVP rule requires fixed subscripts and operand shapes"
    )


def _program_ad_product_einsum_unconfigured_vjp(
    values: NDArray[np.float64],
    cotangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    del values, cotangent
    raise ValueError(
        "program AD product einsum VJP rule requires fixed subscripts and operand shapes"
    )


def _program_ad_product_tensordot_unconfigured_value(
    values: NDArray[np.float64],
) -> NDArray[np.float64]:
    del values
    raise ValueError("program AD product tensordot direct rule requires fixed shapes and axes")


def _program_ad_product_tensordot_unconfigured_jvp(
    values: NDArray[np.float64],
    tangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    del values, tangent
    raise ValueError("program AD product tensordot JVP rule requires fixed shapes and axes")


def _program_ad_product_tensordot_unconfigured_vjp(
    values: NDArray[np.float64],
    cotangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    del values, cotangent
    raise ValueError("program AD product tensordot VJP rule requires fixed shapes and axes")


def _program_ad_product_derivative_rule(name: str) -> CustomDerivativeRule:
    if name == "dot":
        return CustomDerivativeRule(
            name="program_ad_product_dot_direct_rule",
            value_fn=_program_ad_product_dot_value,
            jvp_rule=_program_ad_product_dot_jvp,
            vjp_rule=_program_ad_product_dot_vjp,
        )
    if name == "vdot":
        return CustomDerivativeRule(
            name="program_ad_product_vdot_direct_rule",
            value_fn=_program_ad_product_vdot_value,
            jvp_rule=_program_ad_product_vdot_jvp,
            vjp_rule=_program_ad_product_vdot_vjp,
        )
    if name == "inner":
        return CustomDerivativeRule(
            name="program_ad_product_inner_direct_rule",
            value_fn=_program_ad_product_inner_value,
            jvp_rule=_program_ad_product_inner_jvp,
            vjp_rule=_program_ad_product_inner_vjp,
        )
    if name == "outer":
        return CustomDerivativeRule(
            name="program_ad_product_outer_direct_rule",
            value_fn=_program_ad_product_outer_value,
            jvp_rule=_program_ad_product_outer_jvp,
            vjp_rule=_program_ad_product_outer_vjp,
        )
    if name == "matmul":
        return CustomDerivativeRule(
            name="program_ad_product_matmul_direct_rule",
            value_fn=_program_ad_product_matmul_value,
            jvp_rule=_program_ad_product_matmul_jvp,
            vjp_rule=_program_ad_product_matmul_vjp,
        )
    if name == "tensordot":
        return CustomDerivativeRule(
            name="program_ad_product_tensordot_static_signature_required_rule",
            value_fn=_program_ad_product_tensordot_unconfigured_value,
            jvp_rule=_program_ad_product_tensordot_unconfigured_jvp,
            vjp_rule=_program_ad_product_tensordot_unconfigured_vjp,
        )
    if name == "einsum":
        return CustomDerivativeRule(
            name="program_ad_product_einsum_static_signature_required_rule",
            value_fn=_program_ad_product_einsum_unconfigured_value,
            jvp_rule=_program_ad_product_einsum_unconfigured_jvp,
            vjp_rule=_program_ad_product_einsum_unconfigured_vjp,
        )
    raise ValueError(f"unsupported program AD product primitive {name}")


def _program_ad_product_matmul_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    if len(args) != 2:
        raise ValueError("program AD product matmul shape rule requires two operands")
    lhs_shape = _program_ad_array_shape_of(args[0])
    rhs_shape = _program_ad_array_shape_of(args[1])
    if len(lhs_shape) == 1 and len(rhs_shape) == 1:
        if lhs_shape != rhs_shape:
            raise ValueError("program AD product vector dimensions must align")
        return ()
    if len(lhs_shape) == 2 and len(rhs_shape) == 1:
        if lhs_shape[1] != rhs_shape[0]:
            raise ValueError("program AD product matrix-vector dimensions must align")
        return (lhs_shape[0],)
    if len(lhs_shape) == 1 and len(rhs_shape) == 2:
        if lhs_shape[0] != rhs_shape[0]:
            raise ValueError("program AD product vector-matrix dimensions must align")
        return (rhs_shape[1],)
    if len(lhs_shape) == 2 and len(rhs_shape) == 2:
        if lhs_shape[1] != rhs_shape[0]:
            raise ValueError("program AD product matrix-matrix dimensions must align")
        return (lhs_shape[0], rhs_shape[1])
    raise ValueError("program AD product matmul supports rank-1 and rank-2 operands")


def _program_ad_product_dot_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    shape = _program_ad_product_matmul_shape(args)
    if shape != ():
        raise ValueError("program AD product dot contract supports scalar dot results only")
    return shape


def _program_ad_product_vdot_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    if len(args) != 2:
        raise ValueError("program AD product vdot shape rule requires two operands")
    lhs_size = int(np.prod(_program_ad_array_shape_of(args[0])))
    rhs_size = int(np.prod(_program_ad_array_shape_of(args[1])))
    if lhs_size != rhs_size:
        raise ValueError("program AD np.vdot flattened operands must have matching size")
    return ()


def _program_ad_product_inner_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    if len(args) != 2:
        raise ValueError("program AD product inner shape rule requires two operands")
    lhs_shape = _program_ad_array_shape_of(args[0])
    rhs_shape = _program_ad_array_shape_of(args[1])
    if not lhs_shape or not rhs_shape:
        raise ValueError("program AD product inner shape rule requires non-scalar operands")
    if lhs_shape[-1] != rhs_shape[-1]:
        raise ValueError("program AD product inner last dimensions must align")
    return lhs_shape[:-1] + rhs_shape[:-1]


def _program_ad_product_outer_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    if len(args) != 2:
        raise ValueError("program AD product outer shape rule requires two operands")
    lhs_size = int(np.prod(_program_ad_array_shape_of(args[0])))
    rhs_size = int(np.prod(_program_ad_array_shape_of(args[1])))
    if lhs_size <= 0 or rhs_size <= 0:
        raise ValueError("program AD product outer shape rule requires non-empty operands")
    return (lhs_size, rhs_size)


def _program_ad_product_einsum_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    if len(args) < 2:
        raise ValueError("program AD product einsum shape rule requires subscripts and operands")
    if not isinstance(args[0], str):
        raise ValueError("program AD product einsum shape rule requires static subscripts")
    operand_shapes = tuple(_program_ad_array_shape_of(arg) for arg in args[1:])
    output_labels, _input_labels, dimensions = _parse_static_einsum_subscripts(
        args[0],
        operand_shapes,
    )
    return tuple(dimensions[label] for label in output_labels)


def _program_ad_product_tensordot_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    if len(args) != 3:
        raise ValueError("program AD product tensordot shape rule requires two operands and axes")
    _left, _right, _left_axes, _right_axes, output_shape = (
        _normalise_program_ad_product_tensordot_signature(
            _program_ad_array_shape_of(args[0]),
            _program_ad_array_shape_of(args[1]),
            args[2],
        )
    )
    return output_shape


def _program_ad_product_dtype_rule(args: tuple[object, ...]) -> str:
    if args and isinstance(args[0], str):
        product_args = args[1:]
    elif len(args) == 3:
        product_args = args[:2]
    else:
        product_args = args
    if not product_args:
        raise ValueError("program AD product dtype rule requires operands")
    dtypes = tuple(np.dtype(_program_ad_array_dtype_of(arg)) for arg in product_args)
    return str(np.result_type(*dtypes))


def _program_ad_product_static_arguments(args: tuple[object, ...]) -> tuple[object, ...]:
    if args and isinstance(args[0], str):
        operand_shapes = tuple(_program_ad_array_shape_of(arg) for arg in args[1:])
        output_labels, input_labels, _dimensions = _parse_static_einsum_subscripts(
            args[0],
            operand_shapes,
        )
        return (
            args[0].replace(" ", ""),
            operand_shapes,
            tuple("".join(labels) for labels in input_labels),
            "".join(output_labels),
        )
    if len(args) == 3:
        left_shape = _program_ad_array_shape_of(args[0])
        right_shape = _program_ad_array_shape_of(args[1])
        _left, _right, left_axes, right_axes, _output_shape = (
            _normalise_program_ad_product_tensordot_signature(
                left_shape,
                right_shape,
                args[2],
            )
        )
        return (left_shape, right_shape, (left_axes, right_axes))
    if len(args) != 2:
        raise ValueError("program AD product static rule requires two operands")
    return ()


_PROGRAM_AD_PRODUCT_SHAPE_RULES: Mapping[str, PrimitiveShapeRule] = {
    "dot": _program_ad_product_dot_shape,
    "vdot": _program_ad_product_vdot_shape,
    "inner": _program_ad_product_inner_shape,
    "outer": _program_ad_product_outer_shape,
    "matmul": _program_ad_product_matmul_shape,
    "tensordot": _program_ad_product_tensordot_shape,
    "einsum": _program_ad_product_einsum_shape,
}


def _program_ad_product_batching_rule(
    function: Callable[..., object],
    args: tuple[object, ...],
    axes: tuple[int | None, ...],
    out_axes: int,
) -> object:
    if args and isinstance(args[0], str):
        if len(args) != len(axes):
            raise ValueError("program AD product einsum batching axes must match arguments")
        if axes[0] is not None:
            raise ValueError("program AD product einsum batching requires static subscripts")
        arrays = tuple(
            _as_real_numeric_array(f"program AD product einsum batched operand {index}", arg)
            for index, arg in enumerate(args[1:])
        )
        if not arrays:
            raise ValueError("program AD product einsum batching requires operands")
        if all(axis is None for axis in axes[1:]):
            return _as_real_numeric_array(
                "program AD product einsum batched output", function(*args)
            )
        einsum_mapped_axes: list[int | None] = [
            None if axis is None else _normalise_axis(f"axes[{index + 1}]", axis, array.ndim)
            for index, (axis, array) in enumerate(zip(axes[1:], arrays, strict=True))
        ]
        batch_sizes = {
            int(array.shape[axis])
            for array, axis in zip(arrays, einsum_mapped_axes, strict=True)
            if axis is not None
        }
        if len(batch_sizes) != 1:
            raise ValueError("program AD product einsum batching axes must share one batch size")
        batch_size = batch_sizes.pop()
        outputs = []
        for batch_index in range(batch_size):
            sliced_args = (
                args[0],
                *(
                    array if axis is None else np.take(array, batch_index, axis=axis)
                    for array, axis in zip(arrays, einsum_mapped_axes, strict=True)
                ),
            )
            outputs.append(
                _as_real_numeric_array(
                    "program AD product einsum batched output",
                    function(*sliced_args),
                )
            )
        stacked = np.stack(outputs, axis=0)
        return np.moveaxis(stacked, 0, _normalise_axis("out_axes", out_axes, stacked.ndim))
    if len(args) == 3:
        if len(axes) != 3:
            raise ValueError("program AD product tensordot batching axes must match arguments")
        if axes[2] is not None:
            raise ValueError("program AD product tensordot batching requires static axes")
        arrays = tuple(
            _as_real_numeric_array(f"program AD product tensordot batched operand {index}", arg)
            for index, arg in enumerate(args[:2])
        )
        if all(axis is None for axis in axes[:2]):
            return _as_real_numeric_array(
                "program AD product tensordot batched output",
                function(*arrays, args[2]),
            )
        tensordot_mapped_axes: list[int | None] = [
            None if axis is None else _normalise_axis(f"axes[{index}]", axis, array.ndim)
            for index, (axis, array) in enumerate(zip(axes[:2], arrays, strict=True))
        ]
        batch_sizes = {
            int(array.shape[axis])
            for array, axis in zip(arrays, tensordot_mapped_axes, strict=True)
            if axis is not None
        }
        if len(batch_sizes) != 1:
            raise ValueError(
                "program AD product tensordot batching axes must share one batch size"
            )
        batch_size = batch_sizes.pop()
        outputs = []
        for batch_index in range(batch_size):
            sliced_args = (
                *(
                    array if axis is None else np.take(array, batch_index, axis=axis)
                    for array, axis in zip(arrays, tensordot_mapped_axes, strict=True)
                ),
                args[2],
            )
            outputs.append(
                _as_real_numeric_array(
                    "program AD product tensordot batched output",
                    function(*sliced_args),
                )
            )
        stacked = np.stack(outputs, axis=0)
        return np.moveaxis(stacked, 0, _normalise_axis("out_axes", out_axes, stacked.ndim))
    if len(args) != 2 or len(axes) != 2:
        raise ValueError("program AD product batching requires two operands and two axes")
    arrays = tuple(
        _as_real_numeric_array(f"program AD product batched operand {index}", arg)
        for index, arg in enumerate(args)
    )
    if all(axis is None for axis in axes):
        return _as_real_numeric_array("program AD product batched output", function(*arrays))
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
        raise ValueError("program AD product batching axes must share one batch size")
    batch_size = batch_sizes.pop()
    outputs = []
    for batch_index in range(batch_size):
        sliced_args = tuple(
            array if axis is None else np.take(array, batch_index, axis=axis)
            for array, axis in zip(arrays, mapped_axes, strict=True)
        )
        outputs.append(
            _as_real_numeric_array("program AD product batched output", function(*sliced_args))
        )
    stacked = np.stack(outputs, axis=0)
    return np.moveaxis(stacked, 0, _normalise_axis("out_axes", out_axes, stacked.ndim))


def _program_ad_product_lowering_metadata(name: str) -> Mapping[str, str]:
    static_factories = {
        "dot": "not_required",
        "vdot": "not_required",
        "inner": "program_ad_product_inner_derivative_rule",
        "outer": "program_ad_product_outer_derivative_rule",
        "matmul": "program_ad_product_matmul_derivative_rule",
        "tensordot": "program_ad_product_tensordot_derivative_rule",
        "einsum": "program_ad_product_einsum_derivative_rule",
    }
    static_signatures = {
        "dot": "none",
        "vdot": "none",
        "inner": "left_shape:ranked_tensor_shape;right_shape:ranked_tensor_shape",
        "outer": "left_shape:ranked_tensor_shape;right_shape:ranked_tensor_shape",
        "matmul": "left_shape:ranked_tensor_shape;right_shape:ranked_tensor_shape",
        "tensordot": "left_shape:ranked_tensor_shape;right_shape:ranked_tensor_shape;axes",
        "einsum": "subscripts:explicit_static;operand_shapes:ranked_tensor_shapes",
    }
    nondifferentiable_boundaries = {
        "dot": "inner_dimension_alignment",
        "vdot": "flattened_size_alignment",
        "inner": "last_dimension_alignment",
        "outer": "flattened_outer_product",
        "matmul": "core_dimension_alignment",
        "tensordot": "static_axes_tensor_contraction",
        "einsum": "explicit_static_tensor_contraction",
    }
    return {
        "program_ad": "operator_intercepted_trace",
        "mlir": "available: scpn_diff product dialect interchange; executable lowering blocked",
        "mlir_op": f"scpn_diff.product.{name}",
        "llvm": "blocked_until_executable_product_lowering",
        "rust": "blocked_until_polyglot_product_ad",
        "static_argument_rule": "required" if name in {"einsum", "tensordot"} else "none",
        "static_derivative_factory": static_factories[name],
        "static_signature": static_signatures[name],
        "nondifferentiable_boundary": nondifferentiable_boundaries[name],
        "nondifferentiable_boundary_policy": "fail_closed",
    }


def _register_program_ad_product_primitive_contracts() -> None:
    for name, identity in _PROGRAM_AD_PRODUCT_IDENTITIES.items():
        if DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.contract_for(identity) is not None:
            continue
        DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.register_transform(
            PrimitiveTransformRule(
                identity=identity,
                derivative_rule=_program_ad_product_derivative_rule(name),
                batching_rule=_program_ad_product_batching_rule,
                lowering_metadata=_program_ad_product_lowering_metadata(name),
                shape_rule=_PROGRAM_AD_PRODUCT_SHAPE_RULES[name],
                dtype_rule=_program_ad_product_dtype_rule,
                static_argument_rule=_program_ad_product_static_arguments,
                nondifferentiable_policy=_PROGRAM_AD_PRODUCT_POLICY,
                effect="pure",
            )
        )


def _require_program_ad_product_contract(
    name: str,
    args: tuple[object, ...] | None = None,
) -> PrimitiveContract:
    """Return and validate a registered Program AD product primitive contract."""
    identity = _PROGRAM_AD_PRODUCT_IDENTITIES.get(name)
    if identity is None:
        raise ValueError(f"no program AD product primitive identity registered for {name}")
    contract = DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.require_contract(identity)
    if contract.nondifferentiable_policy != _PROGRAM_AD_PRODUCT_POLICY:
        raise ValueError(f"invalid program AD product primitive policy for {identity.key}")
    if contract.effect != "pure":
        raise ValueError(f"invalid program AD product primitive effect for {identity.key}")
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
            "incomplete program AD product primitive runtime contract for "
            f"{identity.key}: missing {joined}"
        )
    if args is not None:
        _validate_program_ad_product_contract_dispatch(contract, args)
    return contract
