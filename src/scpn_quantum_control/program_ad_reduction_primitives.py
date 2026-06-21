# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Program AD reduction primitive contracts and direct derivative factories."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .program_ad_registry import (
    _PROGRAM_AD_REDUCTION_IDENTITIES,
    _PROGRAM_AD_REDUCTION_POLICY,
    DEFAULT_CUSTOM_DERIVATIVE_REGISTRY,
    CustomDerivativeRule,
    PrimitiveContract,
    PrimitiveShapeRule,
    PrimitiveStaticArgumentRule,
    PrimitiveTransformRule,
)
from .program_ad_trapezoid_primitives import (
    _normalise_trapezoid_axis,
    _program_ad_reduction_trapezoid_jvp,
    _program_ad_reduction_trapezoid_static_widths,
    _program_ad_reduction_trapezoid_value,
    _program_ad_reduction_trapezoid_vjp,
)


def _as_real_numeric_array(name: str, value: object) -> NDArray[np.float64]:
    """Return ``value`` as a real-valued float64 array."""

    array = np.asarray(value)
    if array.dtype.kind not in {"b", "i", "u", "f"}:
        raise ValueError(f"{name} must be a real numeric array")
    return np.asarray(array, dtype=np.float64)


def _as_real_scalar(name: str, value: object) -> float:
    """Return ``value`` as a real scalar."""

    array = np.asarray(value)
    if array.shape != ():
        raise ValueError(f"{name} must be scalar")
    if array.dtype.kind not in {"b", "i", "u", "f"}:
        raise ValueError(f"{name} must be real numeric")
    return float(array)


def _normalise_axis(name: str, axis: int, rank: int) -> int:
    """Normalise a possibly negative axis against a tensor rank."""

    axis_int = int(axis)
    if axis_int < -rank or axis_int >= rank:
        raise ValueError(f"{name} out of bounds for rank {rank}")
    return axis_int % rank


def _program_ad_float64_vector_result(value: object) -> NDArray[np.float64]:
    """Flatten a direct-rule result into the canonical float64 vector form."""

    return np.asarray(value, dtype=np.float64).reshape(-1)


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


def _program_ad_shape_vector(
    name: str,
    role: str,
    values: NDArray[np.float64],
    *,
    expected_size: int,
) -> NDArray[np.float64]:
    """Validate a flattened static-shape operand for direct derivative rules."""

    vector = _as_real_numeric_array(f"program AD {name} {role}", values).reshape(-1)
    if vector.size != expected_size:
        raise ValueError(
            f"program AD {name} {role} requires {expected_size} values, got {vector.size}"
        )
    return vector


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
        raise ValueError("program AD reduction dtype rule requires real numeric arrays")
    return str(array.dtype)


def _is_program_ad_trace_value(value: object) -> bool:
    """Return whether ``value`` follows the lightweight trace-value protocol."""

    return hasattr(value, "value") and hasattr(value, "node")


def _validate_program_ad_reduction_contract_dispatch(
    contract: PrimitiveContract,
    args: tuple[object, ...],
) -> None:
    """Validate reduction primitive dispatch helpers against concrete arguments."""

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


def _require_program_ad_reduction_runtime_contract(
    name: str,
    *,
    args: tuple[object, ...] | None = None,
) -> PrimitiveContract:
    """Return a validated reduction primitive runtime contract."""

    identity = _PROGRAM_AD_REDUCTION_IDENTITIES.get(name)
    if identity is None:
        raise ValueError(f"no program AD reduction primitive identity registered for {name}")
    contract = DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.require_contract(identity)
    if contract.nondifferentiable_policy != _PROGRAM_AD_REDUCTION_POLICY:
        raise ValueError(f"invalid program AD reduction primitive policy for {identity.key}")
    if contract.effect != "pure":
        raise ValueError(f"invalid program AD reduction primitive effect for {identity.key}")

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
            f"incomplete program AD reduction primitive runtime contract for "
            f"{identity.key}: missing {joined}"
        )
    if args is not None:
        _validate_program_ad_reduction_contract_dispatch(contract, args)
    return contract


def _normalise_order_statistic_axis(axis: object, rank: int) -> int | None:
    if axis is None:
        return None
    if isinstance(axis, (bool, np.bool_)) or not isinstance(axis, (int, np.integer)):
        raise ValueError("program AD order-statistic axis must be a static integer or None")
    try:
        return _normalise_axis("axis", int(axis), rank)
    except ValueError as exc:
        if "out of bounds" in str(exc):
            raise ValueError("program AD order-statistic axis out of bounds") from exc
        raise


def _normalise_order_statistic_method(method: object) -> None:
    if not isinstance(method, str):
        raise ValueError("program AD order-statistic method must be static string")
    if method != "linear":
        raise ValueError("program AD order-statistic reductions only supports method='linear'")


def _normalise_order_statistic_q(q: object, *, percentile: bool) -> float:
    if _is_program_ad_trace_value(q):
        raise ValueError("program AD order-statistic q must be static")
    raw = np.asarray(q)
    if raw.shape != ():
        raise ValueError("program AD order-statistic reductions require scalar q")
    if raw.dtype.kind in {"b", "O", "S", "U", "c"}:
        raise ValueError("program AD order-statistic q must be static real numeric")
    q_value = _as_real_scalar("program AD order-statistic q", raw.item())
    if not math.isfinite(q_value):
        raise ValueError("program AD order-statistic q must be finite")
    if percentile:
        if q_value < 0.0 or q_value > 100.0:
            raise ValueError("program AD np.percentile q must be in [0, 100]")
        return q_value / 100.0
    if q_value < 0.0 or q_value > 1.0:
        raise ValueError("program AD np.quantile q must be in [0, 1]")
    return q_value


def _require_strict_order_statistic_values(values: NDArray[np.float64], op_name: str) -> None:
    if values.size == 0:
        raise ValueError(f"program AD {op_name} requires at least one element")
    if not bool(np.all(np.isfinite(values))):
        raise ValueError(f"program AD {op_name} requires finite values")
    if values.size <= 1:
        return
    sorted_values = np.sort(values.reshape(-1))
    if bool(np.any(np.diff(sorted_values) == 0.0)):
        raise ValueError(
            "program AD order-statistic reductions require strictly ordered values; "
            "equal values form a nondifferentiable selection boundary"
        )


def _normalise_ddof(ddof: object, count: int) -> int:
    if not isinstance(ddof, (int, np.integer)):
        raise ValueError("program AD variance/std reductions require integer ddof")
    ddof_int = int(ddof)
    if ddof_int < 0:
        raise ValueError("program AD variance/std reductions require non-negative ddof")
    if count - ddof_int <= 0:
        raise ValueError("program AD variance/std ddof must leave a positive denominator")
    return ddof_int


def _program_ad_reduction_vector(name: str, values: NDArray[np.float64]) -> NDArray[np.float64]:
    vector = _as_real_numeric_array(f"program AD reduction {name} values", values).reshape(-1)
    if vector.size == 0:
        raise ValueError(f"program AD reduction {name} direct rule requires at least one value")
    return vector


def _program_ad_reduction_tangent_pair(
    name: str,
    values: NDArray[np.float64],
    tangent: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    vector = _program_ad_reduction_vector(name, values)
    tangent_vector = _as_real_numeric_array(
        f"program AD reduction {name} tangent", tangent
    ).reshape(-1)
    if tangent_vector.shape != vector.shape:
        raise ValueError(f"program AD reduction {name} tangent shape must match values shape")
    return vector, tangent_vector


def _program_ad_reduction_sum_value(values: NDArray[np.float64]) -> NDArray[np.float64]:
    vector = _program_ad_reduction_vector("sum", values)
    return np.array([float(np.sum(vector))], dtype=np.float64)


def _program_ad_reduction_sum_jvp(
    values: NDArray[np.float64],
    tangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    _vector, tangent_vector = _program_ad_reduction_tangent_pair("sum", values, tangent)
    return np.array([float(np.sum(tangent_vector))], dtype=np.float64)


def _program_ad_reduction_scalar_cotangent(
    name: str,
    cotangent: NDArray[np.float64],
) -> float:
    cotangent_vector = _as_real_numeric_array(
        f"program AD reduction {name} cotangent", cotangent
    ).reshape(-1)
    if cotangent_vector.shape != (1,):
        raise ValueError(f"program AD reduction {name} VJP requires one scalar cotangent")
    return float(cotangent_vector[0])


def _program_ad_reduction_sum_vjp(
    values: NDArray[np.float64],
    cotangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    vector = _program_ad_reduction_vector("sum", values)
    scalar_cotangent = _program_ad_reduction_scalar_cotangent("sum", cotangent)
    return np.full(vector.shape, scalar_cotangent, dtype=np.float64)


def _program_ad_reduction_prod_value(values: NDArray[np.float64]) -> NDArray[np.float64]:
    vector = _program_ad_reduction_vector("prod", values)
    return np.array([float(np.prod(vector))], dtype=np.float64)


def _program_ad_reduction_prod_jvp(
    values: NDArray[np.float64],
    tangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    vector, tangent_vector = _program_ad_reduction_tangent_pair("prod", values, tangent)
    total = 0.0
    for tangent_index in range(vector.size):
        product = 1.0
        for factor_index in range(vector.size):
            product *= (
                tangent_vector[factor_index]
                if factor_index == tangent_index
                else vector[factor_index]
            )
        total += product
    return np.array([float(total)], dtype=np.float64)


def _program_ad_reduction_prod_vjp(
    values: NDArray[np.float64],
    cotangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    vector = _program_ad_reduction_vector("prod", values)
    scalar_cotangent = _program_ad_reduction_scalar_cotangent("prod", cotangent)
    result = np.empty_like(vector, dtype=np.float64)
    for tangent_index in range(vector.size):
        product = 1.0
        for factor_index in range(vector.size):
            if factor_index != tangent_index:
                product *= vector[factor_index]
        result[tangent_index] = scalar_cotangent * product
    return result


def _program_ad_reduction_mean_value(values: NDArray[np.float64]) -> NDArray[np.float64]:
    vector = _program_ad_reduction_vector("mean", values)
    return np.array([float(np.mean(vector))], dtype=np.float64)


def _program_ad_reduction_mean_jvp(
    values: NDArray[np.float64],
    tangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    _vector, tangent_vector = _program_ad_reduction_tangent_pair("mean", values, tangent)
    return np.array([float(np.mean(tangent_vector))], dtype=np.float64)


def _program_ad_reduction_mean_vjp(
    values: NDArray[np.float64],
    cotangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    vector = _program_ad_reduction_vector("mean", values)
    scalar_cotangent = _program_ad_reduction_scalar_cotangent("mean", cotangent)
    return np.full(vector.shape, scalar_cotangent / float(vector.size), dtype=np.float64)


def _program_ad_reduction_variance_gradient(
    name: str,
    values: NDArray[np.float64],
    *,
    ddof: int,
) -> NDArray[np.float64]:
    vector = _program_ad_reduction_vector(name, values)
    ddof_int = _normalise_ddof(ddof, vector.size)
    mean = float(np.mean(vector))
    return (2.0 / float(vector.size - ddof_int)) * (vector - mean)


def _program_ad_reduction_var_value(values: NDArray[np.float64]) -> NDArray[np.float64]:
    vector = _program_ad_reduction_vector("var", values)
    _normalise_ddof(0, vector.size)
    return np.array([float(np.var(vector))], dtype=np.float64)


def _program_ad_reduction_var_jvp(
    values: NDArray[np.float64],
    tangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    vector, tangent_vector = _program_ad_reduction_tangent_pair("var", values, tangent)
    gradient = _program_ad_reduction_variance_gradient("var", vector, ddof=0)
    return np.array([float(np.dot(gradient, tangent_vector))], dtype=np.float64)


def _program_ad_reduction_var_vjp(
    values: NDArray[np.float64],
    cotangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    vector = _program_ad_reduction_vector("var", values)
    scalar_cotangent = _program_ad_reduction_scalar_cotangent("var", cotangent)
    return scalar_cotangent * _program_ad_reduction_variance_gradient("var", vector, ddof=0)


def _program_ad_reduction_std_value(values: NDArray[np.float64]) -> NDArray[np.float64]:
    vector = _program_ad_reduction_vector("std", values)
    _normalise_ddof(0, vector.size)
    return np.array([float(np.std(vector))], dtype=np.float64)


def _program_ad_reduction_std_gradient(
    values: NDArray[np.float64],
    *,
    ddof: int,
) -> NDArray[np.float64]:
    vector = _program_ad_reduction_vector("std", values)
    standard_deviation = float(np.std(vector, ddof=ddof))
    if standard_deviation == 0.0:
        raise ValueError("program AD reduction std direct rule is undefined at zero variance")
    return _program_ad_reduction_variance_gradient("std", vector, ddof=ddof) / (
        2.0 * standard_deviation
    )


def _program_ad_reduction_std_jvp(
    values: NDArray[np.float64],
    tangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    vector, tangent_vector = _program_ad_reduction_tangent_pair("std", values, tangent)
    gradient = _program_ad_reduction_std_gradient(vector, ddof=0)
    return np.array([float(np.dot(gradient, tangent_vector))], dtype=np.float64)


def _program_ad_reduction_std_vjp(
    values: NDArray[np.float64],
    cotangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    vector = _program_ad_reduction_vector("std", values)
    scalar_cotangent = _program_ad_reduction_scalar_cotangent("std", cotangent)
    return scalar_cotangent * _program_ad_reduction_std_gradient(vector, ddof=0)


def _program_ad_reduction_order_statistic_value(
    name: str,
    values: NDArray[np.float64],
    *,
    q: float,
) -> NDArray[np.float64]:
    vector = _program_ad_reduction_vector(name, values)
    _require_strict_order_statistic_values(vector, f"np.{name}")
    order = np.argsort(vector, kind="stable")
    position = q * float(vector.size - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    upper_weight = position - float(lower)
    value = vector[int(order[lower])]
    if lower != upper:
        value = value * (1.0 - upper_weight) + vector[int(order[upper])] * upper_weight
    return np.array([float(value)], dtype=np.float64)


def _program_ad_reduction_order_statistic_jvp(
    name: str,
    values: NDArray[np.float64],
    tangent: NDArray[np.float64],
    *,
    q: float,
) -> NDArray[np.float64]:
    vector, tangent_vector = _program_ad_reduction_tangent_pair(name, values, tangent)
    _require_strict_order_statistic_values(vector, f"np.{name}")
    order = np.argsort(vector, kind="stable")
    position = q * float(vector.size - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    upper_weight = position - float(lower)
    value = tangent_vector[int(order[lower])]
    if lower != upper:
        value = value * (1.0 - upper_weight) + tangent_vector[int(order[upper])] * upper_weight
    return np.array([float(value)], dtype=np.float64)


def _program_ad_reduction_order_statistic_vjp(
    name: str,
    values: NDArray[np.float64],
    cotangent: NDArray[np.float64],
    *,
    q: float,
) -> NDArray[np.float64]:
    vector = _program_ad_reduction_vector(name, values)
    scalar_cotangent = _program_ad_reduction_scalar_cotangent(name, cotangent)
    _require_strict_order_statistic_values(vector, f"np.{name}")
    order = np.argsort(vector, kind="stable")
    position = q * float(vector.size - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    upper_weight = position - float(lower)
    result = np.zeros_like(vector, dtype=np.float64)
    result[int(order[lower])] += scalar_cotangent * (1.0 - upper_weight)
    if lower != upper:
        result[int(order[upper])] += scalar_cotangent * upper_weight
    return result


def _program_ad_reduction_derivative_rule(name: str) -> CustomDerivativeRule:
    if name == "sum":
        return CustomDerivativeRule(
            name="program_ad_reduction_sum_direct_rule",
            value_fn=_program_ad_reduction_sum_value,
            jvp_rule=_program_ad_reduction_sum_jvp,
            vjp_rule=_program_ad_reduction_sum_vjp,
        )
    if name == "prod":
        return CustomDerivativeRule(
            name="program_ad_reduction_prod_direct_rule",
            value_fn=_program_ad_reduction_prod_value,
            jvp_rule=_program_ad_reduction_prod_jvp,
            vjp_rule=_program_ad_reduction_prod_vjp,
        )
    if name == "mean":
        return CustomDerivativeRule(
            name="program_ad_reduction_mean_direct_rule",
            value_fn=_program_ad_reduction_mean_value,
            jvp_rule=_program_ad_reduction_mean_jvp,
            vjp_rule=_program_ad_reduction_mean_vjp,
        )
    if name == "var":
        return CustomDerivativeRule(
            name="program_ad_reduction_var_direct_rule",
            value_fn=_program_ad_reduction_var_value,
            jvp_rule=_program_ad_reduction_var_jvp,
            vjp_rule=_program_ad_reduction_var_vjp,
        )
    if name == "std":
        return CustomDerivativeRule(
            name="program_ad_reduction_std_direct_rule",
            value_fn=_program_ad_reduction_std_value,
            jvp_rule=_program_ad_reduction_std_jvp,
            vjp_rule=_program_ad_reduction_std_vjp,
        )
    if name == "max":
        return _program_ad_reduction_order_statistic_rule(name, q=1.0)
    if name == "min":
        return _program_ad_reduction_order_statistic_rule(name, q=0.0)
    if name == "median":
        return _program_ad_reduction_order_statistic_rule(name, q=0.5)
    if name == "quantile":
        return _program_ad_reduction_order_statistic_rule(name, q=0.5)
    if name == "percentile":
        return _program_ad_reduction_order_statistic_rule(name, q=0.5)
    if name == "trapezoid":
        return CustomDerivativeRule(
            name="program_ad_reduction_trapezoid_direct_rule",
            value_fn=_program_ad_reduction_trapezoid_value,
            jvp_rule=_program_ad_reduction_trapezoid_jvp,
            vjp_rule=_program_ad_reduction_trapezoid_vjp,
        )
    raise ValueError(f"unsupported program AD reduction primitive {name}")


def _program_ad_reduction_normalise_static_shape(
    name: str,
    source_shape: Sequence[int],
) -> tuple[int, ...]:
    shape = tuple(int(dimension) for dimension in source_shape)
    if any(dimension < 0 for dimension in shape):
        raise ValueError(
            f"program AD reduction {name} direct rule requires non-negative dimensions"
        )
    if _program_ad_shape_static_size(shape) == 0:
        raise ValueError(f"program AD reduction {name} direct rule requires at least one value")
    return shape


def _program_ad_reduction_axis_signature(axis: int | None) -> str:
    return "flat" if axis is None else str(axis)


def _program_ad_reduction_output_shape(
    source_shape: tuple[int, ...],
    axis: int | None,
) -> tuple[int, ...]:
    if axis is None:
        return ()
    normalised_axis = _normalise_axis("axis", axis, len(source_shape))
    return source_shape[:normalised_axis] + source_shape[normalised_axis + 1 :]


def _program_ad_reduction_q_signature(q: float) -> str:
    return str(float(q)).replace("-", "neg_").replace(".", "_")


def _program_ad_reduction_order_statistic_static_value(
    name: str,
    values: NDArray[np.float64],
    *,
    source_shape: tuple[int, ...],
    axis: int | None,
    q: float,
) -> NDArray[np.float64]:
    vector = _program_ad_reduction_source_vector(name, "values", values, source_shape=source_shape)
    value_array = vector.reshape(source_shape)
    if axis is None:
        return _program_ad_reduction_order_statistic_value(name, vector, q=q)
    output = np.empty(_program_ad_reduction_output_shape(source_shape, axis), dtype=np.float64)
    for reduced_index in np.ndindex(output.shape):
        source_values = value_array[
            reduced_index[:axis] + (slice(None),) + reduced_index[axis:]
        ].reshape(-1)
        output[reduced_index] = _program_ad_reduction_order_statistic_value(
            name, source_values, q=q
        )[0]
    return _program_ad_float64_vector_result(output)


def _program_ad_reduction_var_static_value(
    values: NDArray[np.float64],
    *,
    source_shape: tuple[int, ...],
    axis: int | None,
    ddof: int,
) -> NDArray[np.float64]:
    vector = _program_ad_reduction_source_vector(
        "var", "values", values, source_shape=source_shape
    )
    count = vector.size if axis is None else source_shape[axis]
    _normalise_ddof(ddof, count)
    return _program_ad_float64_vector_result(
        np.var(vector.reshape(source_shape), axis=axis, ddof=ddof)
    )


def _program_ad_reduction_var_static_jvp(
    values: NDArray[np.float64],
    tangent: NDArray[np.float64],
    *,
    source_shape: tuple[int, ...],
    axis: int | None,
    ddof: int,
) -> NDArray[np.float64]:
    value_array = _program_ad_reduction_source_vector(
        "var", "values", values, source_shape=source_shape
    ).reshape(source_shape)
    tangent_array = _program_ad_reduction_source_vector(
        "var", "tangent", tangent, source_shape=source_shape
    ).reshape(source_shape)
    count = value_array.size if axis is None else source_shape[axis]
    ddof_int = _normalise_ddof(ddof, count)
    mean = np.mean(value_array, axis=axis, keepdims=True)
    gradient = (2.0 / float(count - ddof_int)) * (value_array - mean)
    return _program_ad_float64_vector_result(np.sum(gradient * tangent_array, axis=axis))


def _program_ad_reduction_var_static_vjp(
    values: NDArray[np.float64],
    cotangent: NDArray[np.float64],
    *,
    source_shape: tuple[int, ...],
    axis: int | None,
    ddof: int,
) -> NDArray[np.float64]:
    value_array = _program_ad_reduction_source_vector(
        "var", "values", values, source_shape=source_shape
    ).reshape(source_shape)
    output_shape = _program_ad_reduction_output_shape(source_shape, axis)
    cotangent_array = _program_ad_reduction_cotangent_array(
        "var", cotangent, output_shape=output_shape
    )
    count = value_array.size if axis is None else source_shape[axis]
    ddof_int = _normalise_ddof(ddof, count)
    mean = np.mean(value_array, axis=axis, keepdims=True)
    gradient = (2.0 / float(count - ddof_int)) * (value_array - mean)
    if axis is None:
        return _program_ad_float64_vector_result(gradient * float(cotangent_array))
    expanded_cotangent = np.expand_dims(cotangent_array, axis=axis)
    return _program_ad_float64_vector_result(gradient * expanded_cotangent)


def _program_ad_reduction_std_static_value(
    values: NDArray[np.float64],
    *,
    source_shape: tuple[int, ...],
    axis: int | None,
    ddof: int,
) -> NDArray[np.float64]:
    vector = _program_ad_reduction_source_vector(
        "std", "values", values, source_shape=source_shape
    )
    count = vector.size if axis is None else source_shape[axis]
    _normalise_ddof(ddof, count)
    result = np.std(vector.reshape(source_shape), axis=axis, ddof=ddof)
    if bool(np.any(np.asarray(result) == 0.0)):
        raise ValueError("program AD reduction std direct rule is undefined at zero variance")
    return _program_ad_float64_vector_result(result)


def _program_ad_reduction_std_static_jvp(
    values: NDArray[np.float64],
    tangent: NDArray[np.float64],
    *,
    source_shape: tuple[int, ...],
    axis: int | None,
    ddof: int,
) -> NDArray[np.float64]:
    value_array = _program_ad_reduction_source_vector(
        "std", "values", values, source_shape=source_shape
    ).reshape(source_shape)
    tangent_array = _program_ad_reduction_source_vector(
        "std", "tangent", tangent, source_shape=source_shape
    ).reshape(source_shape)
    count = value_array.size if axis is None else source_shape[axis]
    ddof_int = _normalise_ddof(ddof, count)
    standard_deviation = np.std(value_array, axis=axis, ddof=ddof_int, keepdims=True)
    if bool(np.any(standard_deviation == 0.0)):
        raise ValueError("program AD reduction std direct rule is undefined at zero variance")
    mean = np.mean(value_array, axis=axis, keepdims=True)
    gradient = (value_array - mean) / (float(count - ddof_int) * standard_deviation)
    return _program_ad_float64_vector_result(np.sum(gradient * tangent_array, axis=axis))


def _program_ad_reduction_std_static_vjp(
    values: NDArray[np.float64],
    cotangent: NDArray[np.float64],
    *,
    source_shape: tuple[int, ...],
    axis: int | None,
    ddof: int,
) -> NDArray[np.float64]:
    value_array = _program_ad_reduction_source_vector(
        "std", "values", values, source_shape=source_shape
    ).reshape(source_shape)
    output_shape = _program_ad_reduction_output_shape(source_shape, axis)
    cotangent_array = _program_ad_reduction_cotangent_array(
        "std", cotangent, output_shape=output_shape
    )
    count = value_array.size if axis is None else source_shape[axis]
    ddof_int = _normalise_ddof(ddof, count)
    standard_deviation = np.std(value_array, axis=axis, ddof=ddof_int, keepdims=True)
    if bool(np.any(standard_deviation == 0.0)):
        raise ValueError("program AD reduction std direct rule is undefined at zero variance")
    mean = np.mean(value_array, axis=axis, keepdims=True)
    gradient = (value_array - mean) / (float(count - ddof_int) * standard_deviation)
    if axis is None:
        return _program_ad_float64_vector_result(gradient * float(cotangent_array))
    expanded_cotangent = np.expand_dims(cotangent_array, axis=axis)
    return _program_ad_float64_vector_result(gradient * expanded_cotangent)


def _program_ad_reduction_order_statistic_static_jvp(
    name: str,
    values: NDArray[np.float64],
    tangent: NDArray[np.float64],
    *,
    source_shape: tuple[int, ...],
    axis: int | None,
    q: float,
) -> NDArray[np.float64]:
    value_array = _program_ad_reduction_source_vector(
        name, "values", values, source_shape=source_shape
    ).reshape(source_shape)
    tangent_array = _program_ad_reduction_source_vector(
        name, "tangent", tangent, source_shape=source_shape
    ).reshape(source_shape)
    if axis is None:
        return _program_ad_reduction_order_statistic_jvp(
            name, value_array.reshape(-1), tangent_array.reshape(-1), q=q
        )
    output = np.empty(_program_ad_reduction_output_shape(source_shape, axis), dtype=np.float64)
    for reduced_index in np.ndindex(output.shape):
        selector = reduced_index[:axis] + (slice(None),) + reduced_index[axis:]
        output[reduced_index] = _program_ad_reduction_order_statistic_jvp(
            name,
            value_array[selector].reshape(-1),
            tangent_array[selector].reshape(-1),
            q=q,
        )[0]
    return _program_ad_float64_vector_result(output)


def _program_ad_reduction_order_statistic_static_vjp(
    name: str,
    values: NDArray[np.float64],
    cotangent: NDArray[np.float64],
    *,
    source_shape: tuple[int, ...],
    axis: int | None,
    q: float,
) -> NDArray[np.float64]:
    value_array = _program_ad_reduction_source_vector(
        name, "values", values, source_shape=source_shape
    ).reshape(source_shape)
    output_shape = _program_ad_reduction_output_shape(source_shape, axis)
    cotangent_array = _program_ad_reduction_cotangent_array(
        name, cotangent, output_shape=output_shape
    )
    if axis is None:
        return _program_ad_reduction_order_statistic_vjp(
            name, value_array.reshape(-1), cotangent_array.reshape(-1), q=q
        )
    result = np.zeros_like(value_array, dtype=np.float64)
    for reduced_index in np.ndindex(output_shape):
        selector = reduced_index[:axis] + (slice(None),) + reduced_index[axis:]
        result[selector] += _program_ad_reduction_order_statistic_vjp(
            name,
            value_array[selector].reshape(-1),
            np.array([float(cotangent_array[reduced_index])], dtype=np.float64),
            q=q,
        ).reshape(result[selector].shape)
    return _program_ad_float64_vector_result(result)


def _program_ad_reduction_order_statistic_rule(
    name: str,
    *,
    q: float,
    source_shape: Sequence[int] | None = None,
    axis: int | None = None,
) -> CustomDerivativeRule:
    if source_shape is None:

        def value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
            return _program_ad_reduction_order_statistic_value(name, values, q=q)

        def jvp_rule(
            values: NDArray[np.float64],
            tangent: NDArray[np.float64],
        ) -> NDArray[np.float64]:
            return _program_ad_reduction_order_statistic_jvp(name, values, tangent, q=q)

        def vjp_rule(
            values: NDArray[np.float64],
            cotangent: NDArray[np.float64],
        ) -> NDArray[np.float64]:
            return _program_ad_reduction_order_statistic_vjp(name, values, cotangent, q=q)

        return CustomDerivativeRule(
            name=f"program_ad_reduction_{name}_q_{_program_ad_reduction_q_signature(q)}_direct_rule",
            value_fn=value_fn,
            jvp_rule=jvp_rule,
            vjp_rule=vjp_rule,
        )

    source = _program_ad_reduction_normalise_static_shape(name, source_shape)
    normalised_axis = None if axis is None else _normalise_axis("axis", axis, len(source))

    def static_value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
        return _program_ad_reduction_order_statistic_static_value(
            name, values, source_shape=source, axis=normalised_axis, q=q
        )

    def static_jvp_rule(
        values: NDArray[np.float64],
        tangent: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        return _program_ad_reduction_order_statistic_static_jvp(
            name, values, tangent, source_shape=source, axis=normalised_axis, q=q
        )

    def static_vjp_rule(
        values: NDArray[np.float64],
        cotangent: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        return _program_ad_reduction_order_statistic_static_vjp(
            name, values, cotangent, source_shape=source, axis=normalised_axis, q=q
        )

    return CustomDerivativeRule(
        name=(
            f"program_ad_reduction_{name}_{_program_ad_shape_signature(source)}_axis_"
            f"{_program_ad_reduction_axis_signature(normalised_axis)}_q_"
            f"{_program_ad_reduction_q_signature(q)}_direct_rule"
        ),
        value_fn=static_value_fn,
        jvp_rule=static_jvp_rule,
        vjp_rule=static_vjp_rule,
    )


def _program_ad_reduction_source_vector(
    name: str,
    role: str,
    values: NDArray[np.float64],
    *,
    source_shape: tuple[int, ...],
) -> NDArray[np.float64]:
    return _program_ad_shape_vector(
        f"reduction {name}",
        role,
        values,
        expected_size=_program_ad_shape_static_size(source_shape),
    )


def _program_ad_reduction_cotangent_array(
    name: str,
    cotangent: NDArray[np.float64],
    *,
    output_shape: tuple[int, ...],
) -> NDArray[np.float64]:
    cotangent_vector = _as_real_numeric_array(
        f"program AD reduction {name} cotangent", cotangent
    ).reshape(-1)
    expected_size = _program_ad_shape_static_size(output_shape)
    if cotangent_vector.size != expected_size:
        raise ValueError(
            f"program AD reduction {name} VJP requires cotangent with {expected_size} values"
        )
    return cotangent_vector.reshape(output_shape)


def _program_ad_reduction_sum_static_value(
    values: NDArray[np.float64],
    *,
    source_shape: tuple[int, ...],
    axis: int | None,
) -> NDArray[np.float64]:
    vector = _program_ad_reduction_source_vector(
        "sum", "values", values, source_shape=source_shape
    )
    return _program_ad_float64_vector_result(np.sum(vector.reshape(source_shape), axis=axis))


def _program_ad_reduction_sum_static_jvp(
    values: NDArray[np.float64],
    tangent: NDArray[np.float64],
    *,
    source_shape: tuple[int, ...],
    axis: int | None,
) -> NDArray[np.float64]:
    _program_ad_reduction_source_vector("sum", "values", values, source_shape=source_shape)
    tangent_vector = _program_ad_reduction_source_vector(
        "sum", "tangent", tangent, source_shape=source_shape
    )
    return _program_ad_float64_vector_result(
        np.sum(tangent_vector.reshape(source_shape), axis=axis)
    )


def _program_ad_reduction_sum_static_vjp(
    values: NDArray[np.float64],
    cotangent: NDArray[np.float64],
    *,
    source_shape: tuple[int, ...],
    axis: int | None,
) -> NDArray[np.float64]:
    _program_ad_reduction_source_vector("sum", "values", values, source_shape=source_shape)
    output_shape = _program_ad_reduction_output_shape(source_shape, axis)
    cotangent_array = _program_ad_reduction_cotangent_array(
        "sum", cotangent, output_shape=output_shape
    )
    if axis is None:
        return np.full(_program_ad_shape_static_size(source_shape), float(cotangent_array))
    expanded = np.expand_dims(cotangent_array, axis=axis)
    return _program_ad_float64_vector_result(np.broadcast_to(expanded, source_shape))


def _program_ad_reduction_mean_static_value(
    values: NDArray[np.float64],
    *,
    source_shape: tuple[int, ...],
    axis: int | None,
) -> NDArray[np.float64]:
    vector = _program_ad_reduction_source_vector(
        "mean", "values", values, source_shape=source_shape
    )
    return _program_ad_float64_vector_result(np.mean(vector.reshape(source_shape), axis=axis))


def _program_ad_reduction_mean_static_jvp(
    values: NDArray[np.float64],
    tangent: NDArray[np.float64],
    *,
    source_shape: tuple[int, ...],
    axis: int | None,
) -> NDArray[np.float64]:
    _program_ad_reduction_source_vector("mean", "values", values, source_shape=source_shape)
    tangent_vector = _program_ad_reduction_source_vector(
        "mean", "tangent", tangent, source_shape=source_shape
    )
    return _program_ad_float64_vector_result(
        np.mean(tangent_vector.reshape(source_shape), axis=axis)
    )


def _program_ad_reduction_mean_static_vjp(
    values: NDArray[np.float64],
    cotangent: NDArray[np.float64],
    *,
    source_shape: tuple[int, ...],
    axis: int | None,
) -> NDArray[np.float64]:
    scale = float(
        _program_ad_shape_static_size(source_shape) if axis is None else source_shape[axis]
    )
    return (
        _program_ad_reduction_sum_static_vjp(
            values, cotangent, source_shape=source_shape, axis=axis
        )
        / scale
    )


def _program_ad_reduction_prod_static_value(
    values: NDArray[np.float64],
    *,
    source_shape: tuple[int, ...],
    axis: int | None,
) -> NDArray[np.float64]:
    vector = _program_ad_reduction_source_vector(
        "prod", "values", values, source_shape=source_shape
    )
    return _program_ad_float64_vector_result(np.prod(vector.reshape(source_shape), axis=axis))


def _program_ad_reduction_prod_static_jvp(
    values: NDArray[np.float64],
    tangent: NDArray[np.float64],
    *,
    source_shape: tuple[int, ...],
    axis: int | None,
) -> NDArray[np.float64]:
    value_array = _program_ad_reduction_source_vector(
        "prod", "values", values, source_shape=source_shape
    ).reshape(source_shape)
    tangent_array = _program_ad_reduction_source_vector(
        "prod", "tangent", tangent, source_shape=source_shape
    ).reshape(source_shape)
    derivative = np.zeros_like(value_array, dtype=np.float64)
    for flat_index in range(value_array.size):
        multi_index = np.unravel_index(flat_index, source_shape)
        basis = np.zeros_like(value_array, dtype=np.float64)
        basis[multi_index] = tangent_array[multi_index]
        derivative[multi_index] = np.sum(
            np.prod(np.where(basis != 0.0, basis, value_array), axis=axis)
        )
    if axis is None:
        total = 0.0
        flat_values = value_array.reshape(-1)
        flat_tangent = tangent_array.reshape(-1)
        for tangent_index in range(flat_values.size):
            product = 1.0
            for factor_index in range(flat_values.size):
                product *= (
                    flat_tangent[factor_index]
                    if factor_index == tangent_index
                    else flat_values[factor_index]
                )
            total += product
        return np.array([float(total)], dtype=np.float64)
    output = np.zeros(_program_ad_reduction_output_shape(source_shape, axis), dtype=np.float64)
    for flat_index in range(value_array.size):
        multi_index = np.unravel_index(flat_index, source_shape)
        output_index = multi_index[:axis] + multi_index[axis + 1 :]
        product = 1.0
        for factor_index in range(source_shape[axis]):
            candidate_index = multi_index[:axis] + (factor_index,) + multi_index[axis + 1 :]
            product *= (
                tangent_array[candidate_index]
                if factor_index == multi_index[axis]
                else value_array[candidate_index]
            )
        output[output_index] += product
    return _program_ad_float64_vector_result(output)


def _program_ad_reduction_prod_static_vjp(
    values: NDArray[np.float64],
    cotangent: NDArray[np.float64],
    *,
    source_shape: tuple[int, ...],
    axis: int | None,
) -> NDArray[np.float64]:
    value_array = _program_ad_reduction_source_vector(
        "prod", "values", values, source_shape=source_shape
    ).reshape(source_shape)
    output_shape = _program_ad_reduction_output_shape(source_shape, axis)
    cotangent_array = _program_ad_reduction_cotangent_array(
        "prod", cotangent, output_shape=output_shape
    )
    result = np.zeros_like(value_array, dtype=np.float64)
    if axis is None:
        scalar_cotangent = float(cotangent_array)
        flat_values = value_array.reshape(-1)
        flat_result = result.reshape(-1)
        for tangent_index in range(flat_values.size):
            product = 1.0
            for factor_index in range(flat_values.size):
                if factor_index != tangent_index:
                    product *= flat_values[factor_index]
            flat_result[tangent_index] = scalar_cotangent * product
        return flat_result
    for flat_index in range(value_array.size):
        multi_index = np.unravel_index(flat_index, source_shape)
        output_index = multi_index[:axis] + multi_index[axis + 1 :]
        product = 1.0
        for factor_index in range(source_shape[axis]):
            candidate_index = multi_index[:axis] + (factor_index,) + multi_index[axis + 1 :]
            if factor_index != multi_index[axis]:
                product *= value_array[candidate_index]
        result[multi_index] = cotangent_array[output_index] * product
    return _program_ad_float64_vector_result(result)


def _program_ad_reduction_static_rule(
    name: str,
    source_shape: Sequence[int],
    axis: int | None,
) -> CustomDerivativeRule:
    source = _program_ad_reduction_normalise_static_shape(name, source_shape)
    normalised_axis = None if axis is None else _normalise_axis("axis", axis, len(source))

    def value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
        if name == "sum":
            return _program_ad_reduction_sum_static_value(
                values, source_shape=source, axis=normalised_axis
            )
        if name == "mean":
            return _program_ad_reduction_mean_static_value(
                values, source_shape=source, axis=normalised_axis
            )
        if name == "prod":
            return _program_ad_reduction_prod_static_value(
                values, source_shape=source, axis=normalised_axis
            )
        raise ValueError(f"unsupported program AD reduction primitive {name}")

    def jvp_rule(values: NDArray[np.float64], tangent: NDArray[np.float64]) -> NDArray[np.float64]:
        if name == "sum":
            return _program_ad_reduction_sum_static_jvp(
                values, tangent, source_shape=source, axis=normalised_axis
            )
        if name == "mean":
            return _program_ad_reduction_mean_static_jvp(
                values, tangent, source_shape=source, axis=normalised_axis
            )
        if name == "prod":
            return _program_ad_reduction_prod_static_jvp(
                values, tangent, source_shape=source, axis=normalised_axis
            )
        raise ValueError(f"unsupported program AD reduction primitive {name}")

    def vjp_rule(
        values: NDArray[np.float64], cotangent: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        if name == "sum":
            return _program_ad_reduction_sum_static_vjp(
                values, cotangent, source_shape=source, axis=normalised_axis
            )
        if name == "mean":
            return _program_ad_reduction_mean_static_vjp(
                values, cotangent, source_shape=source, axis=normalised_axis
            )
        if name == "prod":
            return _program_ad_reduction_prod_static_vjp(
                values, cotangent, source_shape=source, axis=normalised_axis
            )
        raise ValueError(f"unsupported program AD reduction primitive {name}")

    return CustomDerivativeRule(
        name=(
            f"program_ad_reduction_{name}_{_program_ad_shape_signature(source)}_axis_"
            f"{_program_ad_reduction_axis_signature(normalised_axis)}_direct_rule"
        ),
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
    )


def program_ad_reduction_sum_derivative_rule(
    source_shape: Sequence[int],
    axis: int | None = None,
) -> CustomDerivativeRule:
    """Build an exact direct derivative rule for a fixed sum reduction signature."""

    return _program_ad_reduction_static_rule("sum", source_shape, axis)


def program_ad_reduction_mean_derivative_rule(
    source_shape: Sequence[int],
    axis: int | None = None,
) -> CustomDerivativeRule:
    """Build an exact direct derivative rule for a fixed mean reduction signature."""

    return _program_ad_reduction_static_rule("mean", source_shape, axis)


def _program_ad_reduction_var_std_static_rule(
    name: Literal["var", "std"],
    source_shape: Sequence[int],
    *,
    axis: int | None,
    ddof: int,
) -> CustomDerivativeRule:
    source = _program_ad_reduction_normalise_static_shape(name, source_shape)
    normalised_axis = None if axis is None else _normalise_axis("axis", axis, len(source))
    count = (
        _program_ad_shape_static_size(source)
        if normalised_axis is None
        else source[normalised_axis]
    )
    ddof_int = _normalise_ddof(ddof, count)

    def value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
        if name == "var":
            return _program_ad_reduction_var_static_value(
                values, source_shape=source, axis=normalised_axis, ddof=ddof_int
            )
        return _program_ad_reduction_std_static_value(
            values, source_shape=source, axis=normalised_axis, ddof=ddof_int
        )

    def jvp_rule(values: NDArray[np.float64], tangent: NDArray[np.float64]) -> NDArray[np.float64]:
        if name == "var":
            return _program_ad_reduction_var_static_jvp(
                values, tangent, source_shape=source, axis=normalised_axis, ddof=ddof_int
            )
        return _program_ad_reduction_std_static_jvp(
            values, tangent, source_shape=source, axis=normalised_axis, ddof=ddof_int
        )

    def vjp_rule(
        values: NDArray[np.float64], cotangent: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        if name == "var":
            return _program_ad_reduction_var_static_vjp(
                values, cotangent, source_shape=source, axis=normalised_axis, ddof=ddof_int
            )
        return _program_ad_reduction_std_static_vjp(
            values, cotangent, source_shape=source, axis=normalised_axis, ddof=ddof_int
        )

    return CustomDerivativeRule(
        name=(
            f"program_ad_reduction_{name}_{_program_ad_shape_signature(source)}_axis_"
            f"{_program_ad_reduction_axis_signature(normalised_axis)}_ddof_"
            f"{ddof_int}_direct_rule"
        ),
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
    )


def program_ad_reduction_var_derivative_rule(
    source_shape: Sequence[int],
    axis: int | None = None,
    *,
    ddof: int = 0,
) -> CustomDerivativeRule:
    """Build an exact direct derivative rule for a fixed variance signature."""

    return _program_ad_reduction_var_std_static_rule("var", source_shape, axis=axis, ddof=ddof)


def program_ad_reduction_std_derivative_rule(
    source_shape: Sequence[int],
    axis: int | None = None,
    *,
    ddof: int = 0,
) -> CustomDerivativeRule:
    """Build an exact direct derivative rule for a fixed standard-deviation signature."""

    return _program_ad_reduction_var_std_static_rule("std", source_shape, axis=axis, ddof=ddof)


def program_ad_reduction_max_derivative_rule(
    source_shape: Sequence[int],
    axis: int | None = None,
) -> CustomDerivativeRule:
    """Build an exact direct derivative rule for a fixed maximum reduction signature."""

    return _program_ad_reduction_order_statistic_rule(
        "max", q=1.0, source_shape=source_shape, axis=axis
    )


def program_ad_reduction_min_derivative_rule(
    source_shape: Sequence[int],
    axis: int | None = None,
) -> CustomDerivativeRule:
    """Build an exact direct derivative rule for a fixed minimum reduction signature."""

    return _program_ad_reduction_order_statistic_rule(
        "min", q=0.0, source_shape=source_shape, axis=axis
    )


def program_ad_reduction_median_derivative_rule(
    source_shape: Sequence[int],
    axis: int | None = None,
) -> CustomDerivativeRule:
    """Build an exact direct derivative rule for a fixed median reduction signature."""

    return _program_ad_reduction_order_statistic_rule(
        "median", q=0.5, source_shape=source_shape, axis=axis
    )


def program_ad_reduction_quantile_derivative_rule(
    source_shape: Sequence[int],
    *,
    q: object = 0.5,
    axis: int | None = None,
) -> CustomDerivativeRule:
    """Build an exact direct derivative rule for a fixed scalar-quantile signature."""

    return _program_ad_reduction_order_statistic_rule(
        "quantile",
        q=_normalise_order_statistic_q(q, percentile=False),
        source_shape=source_shape,
        axis=axis,
    )


def program_ad_reduction_percentile_derivative_rule(
    source_shape: Sequence[int],
    *,
    q: object = 50.0,
    axis: int | None = None,
) -> CustomDerivativeRule:
    """Build an exact direct derivative rule for a fixed scalar-percentile signature."""

    return _program_ad_reduction_order_statistic_rule(
        "percentile",
        q=_normalise_order_statistic_q(q, percentile=True),
        source_shape=source_shape,
        axis=axis,
    )


def program_ad_reduction_prod_derivative_rule(
    source_shape: Sequence[int],
    axis: int | None = None,
) -> CustomDerivativeRule:
    """Build an exact direct derivative rule for a fixed product reduction signature."""

    return _program_ad_reduction_static_rule("prod", source_shape, axis)


def _program_ad_reduction_axis(args: tuple[object, ...]) -> int | None:
    if len(args) not in {1, 2}:
        raise ValueError("program AD reduction rule requires array and optional axis")
    if len(args) == 1 or args[1] is None:
        return None
    axis = args[1]
    if isinstance(axis, bool) or not isinstance(axis, (int, np.integer)):
        raise ValueError("program AD reduction axis must be a static integer or None")
    return int(axis)


def _program_ad_reduction_axis_ddof(args: tuple[object, ...]) -> tuple[int | None, int]:
    if len(args) != 3:
        raise ValueError("program AD variance/std rule requires array, axis, and ddof")
    source_shape = _program_ad_array_shape_of(args[0])
    raw_axis = args[1]
    if raw_axis is None:
        axis = None
    elif isinstance(raw_axis, bool) or not isinstance(raw_axis, (int, np.integer)):
        raise ValueError("program AD variance/std axis must be a static integer or None")
    else:
        axis = _normalise_axis("axis", int(raw_axis), len(source_shape))
    count = int(np.prod(source_shape)) if axis is None else source_shape[axis]
    return axis, _normalise_ddof(args[2], count)


def _program_ad_order_statistic_reduction_axis(args: tuple[object, ...]) -> int | None:
    if len(args) == 2:
        axis = args[1]
    elif len(args) == 4:
        _normalise_order_statistic_method(args[3])
        axis = args[2]
    else:
        raise ValueError("program AD order-statistic reduction rule requires static arguments")
    return _normalise_order_statistic_axis(axis, len(_program_ad_array_shape_of(args[0])))


def _program_ad_reduction_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    source_shape = _program_ad_array_shape_of(args[0])
    if int(np.prod(source_shape)) == 0:
        raise ValueError("program AD reduction shape rule requires at least one element")
    axis = _program_ad_reduction_axis(args)
    if axis is None:
        return ()
    normalised_axis = _normalise_axis("axis", axis, len(source_shape))
    return source_shape[:normalised_axis] + source_shape[normalised_axis + 1 :]


def _program_ad_reduction_var_std_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    source_shape = _program_ad_array_shape_of(args[0])
    if int(np.prod(source_shape)) == 0:
        raise ValueError("program AD variance/std shape rule requires at least one element")
    axis, _ddof = _program_ad_reduction_axis_ddof(args)
    if axis is None:
        return ()
    return source_shape[:axis] + source_shape[axis + 1 :]


def _program_ad_order_statistic_reduction_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    source_shape = _program_ad_array_shape_of(args[0])
    if int(np.prod(source_shape)) == 0:
        raise ValueError(
            "program AD order-statistic reduction shape rule requires at least one element"
        )
    axis = _program_ad_order_statistic_reduction_axis(args)
    if axis is None:
        return ()
    return source_shape[:axis] + source_shape[axis + 1 :]


def _program_ad_reduction_trapezoid_axis(args: tuple[object, ...]) -> int:
    if len(args) != 4:
        raise ValueError("program AD trapezoid rule requires y, x, dx, and axis")
    return _normalise_trapezoid_axis(args[3], len(_program_ad_array_shape_of(args[0])))


def _program_ad_reduction_trapezoid_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    source_shape = _program_ad_array_shape_of(args[0])
    axis = _program_ad_reduction_trapezoid_axis(args)
    _program_ad_reduction_trapezoid_static_widths(source_shape, x=args[1], dx=args[2], axis=axis)
    return source_shape[:axis] + source_shape[axis + 1 :]


def _program_ad_reduction_dtype_rule(args: tuple[object, ...]) -> str:
    if not args:
        raise ValueError("program AD reduction dtype rule requires an array operand")
    return _program_ad_array_dtype_of(args[0])


def _program_ad_reduction_static_arguments(args: tuple[object, ...]) -> tuple[object, ...]:
    if len(args) == 4:
        source_shape = _program_ad_array_shape_of(args[0])
        axis = _program_ad_reduction_trapezoid_axis(args)
        _program_ad_reduction_trapezoid_static_widths(
            source_shape, x=args[1], dx=args[2], axis=axis
        )
        x = args[1]
        if x is None:
            return (None, _as_real_scalar("program AD trapezoid dx", args[2]), axis)
        x_array = _as_real_numeric_array("program AD trapezoid x", x)
        return (
            (
                "x",
                tuple(int(dimension) for dimension in x_array.shape),
                tuple(float(item) for item in x_array.reshape(-1)),
            ),
            1.0,
            axis,
        )
    return (_program_ad_reduction_axis(args),)


def _program_ad_reduction_order_statistic_axis_static_arguments(
    args: tuple[object, ...],
) -> tuple[object, ...]:
    return (_program_ad_order_statistic_reduction_axis(args),)


def _program_ad_reduction_var_std_static_arguments(
    args: tuple[object, ...],
) -> tuple[object, ...]:
    return _program_ad_reduction_axis_ddof(args)


def _program_ad_reduction_quantile_static_arguments(
    args: tuple[object, ...],
) -> tuple[object, ...]:
    if len(args) != 4:
        raise ValueError("program AD quantile rule requires array, q, axis, and method")
    _normalise_order_statistic_method(args[3])
    return (
        _normalise_order_statistic_q(args[1], percentile=False),
        _program_ad_order_statistic_reduction_axis(args),
        "linear",
    )


def _program_ad_reduction_percentile_static_arguments(
    args: tuple[object, ...],
) -> tuple[object, ...]:
    if len(args) != 4:
        raise ValueError("program AD percentile rule requires array, q, axis, and method")
    _normalise_order_statistic_method(args[3])
    return (
        _normalise_order_statistic_q(args[1], percentile=True),
        _program_ad_order_statistic_reduction_axis(args),
        "linear",
    )


def _program_ad_reduction_batching_rule(
    function: Callable[..., object],
    args: tuple[object, ...],
    axes: tuple[int | None, ...],
    out_axes: int,
) -> object:
    if len(args) != len(axes):
        raise ValueError("program AD reduction batching axes must match argument count")
    if not args:
        raise ValueError("program AD reduction batching requires an array operand")
    array = _as_real_numeric_array("program AD reduction batched operand", args[0])
    batch_axis = axes[0]
    if batch_axis is None:
        return function(*args)
    if any(item is not None for item in axes[1:]):
        raise ValueError("program AD reduction batching supports static axes only")
    batch_axis = _normalise_axis("axes[0]", batch_axis, array.ndim)
    order_statistic = len(args) == 4 and isinstance(args[3], str)
    if order_statistic:
        reduction_axis = _program_ad_order_statistic_reduction_axis(args)
    elif len(args) == 3:
        reduction_axis, ddof = _program_ad_reduction_axis_ddof(args)
    elif len(args) == 4:
        reduction_axis = _program_ad_reduction_trapezoid_axis(args)
    else:
        reduction_axis = _program_ad_reduction_axis(args)
    if reduction_axis is not None:
        reduction_axis = _normalise_axis("reduction axis", reduction_axis, array.ndim)
        if reduction_axis == batch_axis:
            raise ValueError("program AD reduction batching cannot reduce the mapped batch axis")
        if reduction_axis > batch_axis:
            reduction_axis -= 1
    static_tail: tuple[object, ...] = (reduction_axis,)
    if order_statistic:
        q = args[1]
        method = args[3]
        outputs = [
            _as_real_numeric_array(
                "program AD reduction batched output",
                function(
                    np.take(array, batch_index, axis=batch_axis),
                    q,
                    axis=reduction_axis,
                    method=method,
                ),
            )
            for batch_index in range(int(array.shape[batch_axis]))
        ]
        stacked = np.stack(outputs, axis=0)
        return np.moveaxis(stacked, 0, _normalise_axis("out_axes", out_axes, stacked.ndim))
    if len(args) == 3:
        outputs = [
            _as_real_numeric_array(
                "program AD reduction batched output",
                function(
                    np.take(array, batch_index, axis=batch_axis),
                    axis=reduction_axis,
                    ddof=ddof,
                ),
            )
            for batch_index in range(int(array.shape[batch_axis]))
        ]
        stacked = np.stack(outputs, axis=0)
        return np.moveaxis(stacked, 0, _normalise_axis("out_axes", out_axes, stacked.ndim))
    if len(args) == 4 and args[1] is not None:
        x_array = _as_real_numeric_array("program AD trapezoid batched x", args[1])
        if tuple(x_array.shape) == tuple(array.shape):
            raise ValueError(
                "program AD trapezoid batching requires scalar dx or one-dimensional static x"
            )
        static_tail = (args[1], args[2], reduction_axis)
    outputs = [
        _as_real_numeric_array(
            "program AD reduction batched output",
            function(np.take(array, batch_index, axis=batch_axis), *static_tail),
        )
        for batch_index in range(int(array.shape[batch_axis]))
    ]
    stacked = np.stack(outputs, axis=0)
    return np.moveaxis(stacked, 0, _normalise_axis("out_axes", out_axes, stacked.ndim))


def _program_ad_reduction_lowering_metadata(name: str) -> Mapping[str, str]:
    static_factory = {
        "sum": "program_ad_reduction_sum_derivative_rule",
        "prod": "program_ad_reduction_prod_derivative_rule",
        "mean": "program_ad_reduction_mean_derivative_rule",
        "var": "program_ad_reduction_var_derivative_rule",
        "std": "program_ad_reduction_std_derivative_rule",
        "max": "program_ad_reduction_max_derivative_rule",
        "min": "program_ad_reduction_min_derivative_rule",
        "median": "program_ad_reduction_median_derivative_rule",
        "quantile": "program_ad_reduction_quantile_derivative_rule",
        "percentile": "program_ad_reduction_percentile_derivative_rule",
        "trapezoid": "program_ad_reduction_trapezoid_derivative_rule",
    }[name]
    nondifferentiable_boundaries = {
        "sum": "static_axis_and_stable_output_shape",
        "prod": "static_axis_zero_factor_sensitive",
        "mean": "static_axis_nonempty_reduction",
        "var": "static_axis_ddof_positive_denominator",
        "std": "static_axis_ddof_positive_denominator_nonzero_variance",
        "max": "static_axis_unique_max_selector",
        "min": "static_axis_unique_min_selector",
        "median": "static_axis_strict_order_selection",
        "quantile": "static_scalar_q_axis_method_strict_order_selection",
        "percentile": "static_scalar_q_axis_method_strict_order_selection",
        "trapezoid": "static_axis_and_static_grid_spacing",
    }
    return {
        "program_ad": "operator_intercepted_trace",
        "mlir": "available: scpn_diff reduction dialect interchange; executable lowering blocked",
        "mlir_op": f"scpn_diff.reduction.{name}",
        "llvm": "blocked_until_executable_reduction_lowering",
        "rust": "blocked_until_polyglot_reduction_ad",
        "static_argument_rule": "required",
        "static_derivative_factory": static_factory,
        "static_signature": (
            "source_shape:ranked_tensor_shape;x_or_dx;axis"
            if name == "trapezoid"
            else (
                "source_shape:ranked_tensor_shape;q;axis;method"
                if name in {"quantile", "percentile"}
                else (
                    "source_shape:ranked_tensor_shape;axis;ddof"
                    if name in {"var", "std"}
                    else "source_shape:ranked_tensor_shape;axis"
                )
            )
        ),
        "nondifferentiable_boundary": nondifferentiable_boundaries[name],
        "nondifferentiable_boundary_policy": "fail_closed",
    }


_PROGRAM_AD_REDUCTION_SHAPE_RULES: Mapping[str, PrimitiveShapeRule] = {
    "sum": _program_ad_reduction_shape,
    "prod": _program_ad_reduction_shape,
    "mean": _program_ad_reduction_shape,
    "var": _program_ad_reduction_var_std_shape,
    "std": _program_ad_reduction_var_std_shape,
    "max": _program_ad_order_statistic_reduction_shape,
    "min": _program_ad_order_statistic_reduction_shape,
    "median": _program_ad_order_statistic_reduction_shape,
    "quantile": _program_ad_order_statistic_reduction_shape,
    "percentile": _program_ad_order_statistic_reduction_shape,
    "trapezoid": _program_ad_reduction_trapezoid_shape,
}


_PROGRAM_AD_REDUCTION_STATIC_ARGUMENT_RULES: Mapping[str, PrimitiveStaticArgumentRule] = {
    "sum": _program_ad_reduction_static_arguments,
    "prod": _program_ad_reduction_static_arguments,
    "mean": _program_ad_reduction_static_arguments,
    "var": _program_ad_reduction_var_std_static_arguments,
    "std": _program_ad_reduction_var_std_static_arguments,
    "max": _program_ad_reduction_order_statistic_axis_static_arguments,
    "min": _program_ad_reduction_order_statistic_axis_static_arguments,
    "median": _program_ad_reduction_order_statistic_axis_static_arguments,
    "quantile": _program_ad_reduction_quantile_static_arguments,
    "percentile": _program_ad_reduction_percentile_static_arguments,
    "trapezoid": _program_ad_reduction_static_arguments,
}


def _register_program_ad_reduction_primitive_contracts() -> None:
    for name, identity in _PROGRAM_AD_REDUCTION_IDENTITIES.items():
        if DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.contract_for(identity) is not None:
            continue
        DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.register_transform(
            PrimitiveTransformRule(
                identity=identity,
                derivative_rule=_program_ad_reduction_derivative_rule(name),
                batching_rule=_program_ad_reduction_batching_rule,
                lowering_metadata=_program_ad_reduction_lowering_metadata(name),
                shape_rule=_PROGRAM_AD_REDUCTION_SHAPE_RULES[name],
                dtype_rule=_program_ad_reduction_dtype_rule,
                static_argument_rule=_PROGRAM_AD_REDUCTION_STATIC_ARGUMENT_RULES[name],
                nondifferentiable_policy=_PROGRAM_AD_REDUCTION_POLICY,
                effect="pure",
            )
        )


def _require_program_ad_reduction_contract(
    name: str,
    args: tuple[object, ...] | None = None,
) -> PrimitiveContract:
    return _require_program_ad_reduction_runtime_contract(name, args=args)


__all__ = [
    "_normalise_ddof",
    "_normalise_order_statistic_axis",
    "_normalise_order_statistic_method",
    "_normalise_order_statistic_q",
    "_program_ad_reduction_derivative_rule",
    "_program_ad_reduction_batching_rule",
    "_program_ad_reduction_lowering_metadata",
    "_register_program_ad_reduction_primitive_contracts",
    "_require_program_ad_reduction_contract",
    "_require_strict_order_statistic_values",
    "program_ad_reduction_max_derivative_rule",
    "program_ad_reduction_mean_derivative_rule",
    "program_ad_reduction_median_derivative_rule",
    "program_ad_reduction_min_derivative_rule",
    "program_ad_reduction_percentile_derivative_rule",
    "program_ad_reduction_prod_derivative_rule",
    "program_ad_reduction_quantile_derivative_rule",
    "program_ad_reduction_std_derivative_rule",
    "program_ad_reduction_sum_derivative_rule",
    "program_ad_reduction_var_derivative_rule",
]
