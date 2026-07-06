# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Program AD signal primitive rules
"""Static convolution and correlation derivative rules for Program AD."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

from .differentiable_parameter_contracts import _as_real_numeric_array
from .program_ad_array_indexing import (
    _normalise_axis,
    _program_ad_array_normalise_static_shape,
    _program_ad_array_static_size,
)
from .program_ad_registry import (
    _PROGRAM_AD_SIGNAL_IDENTITIES,
    _PROGRAM_AD_SIGNAL_POLICY,
    DEFAULT_CUSTOM_DERIVATIVE_REGISTRY,
    CustomDerivativeRule,
    PrimitiveContract,
    PrimitiveIdentity,
    PrimitiveShapeRule,
    PrimitiveStaticArgumentRule,
    PrimitiveTransformRule,
)


def _is_trace_array(value: object) -> bool:
    """Return whether ``value`` behaves like a whole-program trace array."""

    return type(value).__name__ == "TraceADArray" and hasattr(value, "context")


def _program_ad_array_shape_of(value: object) -> tuple[int, ...]:
    """Return the static shape recorded by a trace value or array-like input."""

    if _is_trace_array(value):
        shape = getattr(value, "shape", None)
        if not isinstance(shape, tuple):
            raise ValueError("program AD signal trace array shape must be static")
        return tuple(int(dimension) for dimension in shape)
    return tuple(int(dimension) for dimension in np.asarray(value).shape)


def _program_ad_array_dtype_of(value: object) -> str:
    """Return the dtype name recorded by a trace value or array-like input."""

    if _is_trace_array(value):
        return "float64"
    array = np.asarray(value)
    if array.dtype.kind in {"O", "S", "U", "c"}:
        raise ValueError("program AD signal primitive dtype rule requires real numeric arrays")
    return str(array.dtype)


def _normalise_convolve_mode(mode: object) -> Literal["full", "same", "valid"]:
    if not isinstance(mode, str) or mode not in {"full", "same", "valid"}:
        raise ValueError("program AD np.convolve mode must be 'full', 'same', or 'valid'")
    return cast(Literal["full", "same", "valid"], mode)


def _normalise_correlate_mode(mode: object) -> Literal["full", "same", "valid"]:
    if not isinstance(mode, str) or mode not in {"full", "same", "valid"}:
        raise ValueError("program AD np.correlate mode must be 'full', 'same', or 'valid'")
    return cast(Literal["full", "same", "valid"], mode)


def _convolve_output_window(
    left_size: int, right_size: int, mode: Literal["full", "same", "valid"]
) -> tuple[int, int]:
    if mode == "full":
        return 0, left_size + right_size - 1
    if mode == "same":
        output_size = max(left_size, right_size)
        start = (min(left_size, right_size) - 1) // 2
        return start, start + output_size
    output_size = max(left_size, right_size) - min(left_size, right_size) + 1
    start = min(left_size, right_size) - 1
    return start, start + output_size


def _program_ad_signal_convolve_static_shape(role: str, shape: Sequence[int]) -> tuple[int, ...]:
    normalised = _program_ad_array_normalise_static_shape(f"signal convolve {role}", shape)
    if len(normalised) != 1:
        raise ValueError("program AD signal convolve direct rule requires rank-1 operands")
    if normalised[0] <= 0:
        raise ValueError("program AD signal convolve direct rule requires non-empty operands")
    return normalised


def _program_ad_signal_convolve_source_size(
    left_shape: tuple[int, ...], right_shape: tuple[int, ...]
) -> int:
    return _program_ad_array_static_size(left_shape) + _program_ad_array_static_size(right_shape)


def _program_ad_signal_convolve_split_source(
    role: str,
    values: NDArray[np.float64],
    *,
    left_shape: tuple[int, ...],
    right_shape: tuple[int, ...],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    vector = _as_real_numeric_array(f"program AD signal convolve {role}", values).reshape(-1)
    left_size = _program_ad_array_static_size(left_shape)
    expected_size = _program_ad_signal_convolve_source_size(left_shape, right_shape)
    if vector.size != expected_size:
        raise ValueError(
            f"program AD signal convolve direct rule requires {expected_size} {role} values"
        )
    return vector[:left_size], vector[left_size:]


def _program_ad_signal_convolve_output_size(
    left_shape: tuple[int, ...],
    right_shape: tuple[int, ...],
    mode: Literal["full", "same", "valid"],
) -> int:
    left_size = _program_ad_array_static_size(left_shape)
    right_size = _program_ad_array_static_size(right_shape)
    start, stop = _convolve_output_window(left_size, right_size, mode)
    return stop - start


def _program_ad_signal_convolve_direct_value(
    values: NDArray[np.float64],
    *,
    left_shape: tuple[int, ...],
    right_shape: tuple[int, ...],
    mode: Literal["full", "same", "valid"],
) -> NDArray[np.float64]:
    left, right = _program_ad_signal_convolve_split_source(
        "values", values, left_shape=left_shape, right_shape=right_shape
    )
    return np.convolve(left, right, mode=mode).astype(np.float64, copy=False)


def _program_ad_signal_convolve_direct_jvp(
    values: NDArray[np.float64],
    tangent: NDArray[np.float64],
    *,
    left_shape: tuple[int, ...],
    right_shape: tuple[int, ...],
    mode: Literal["full", "same", "valid"],
) -> NDArray[np.float64]:
    left, right = _program_ad_signal_convolve_split_source(
        "values", values, left_shape=left_shape, right_shape=right_shape
    )
    tangent_left, tangent_right = _program_ad_signal_convolve_split_source(
        "tangent", tangent, left_shape=left_shape, right_shape=right_shape
    )
    return (
        np.convolve(tangent_left, right, mode=mode) + np.convolve(left, tangent_right, mode=mode)
    ).astype(np.float64, copy=False)


def _program_ad_signal_convolve_direct_vjp(
    values: NDArray[np.float64],
    cotangent: NDArray[np.float64],
    *,
    left_shape: tuple[int, ...],
    right_shape: tuple[int, ...],
    mode: Literal["full", "same", "valid"],
) -> NDArray[np.float64]:
    left, right = _program_ad_signal_convolve_split_source(
        "values", values, left_shape=left_shape, right_shape=right_shape
    )
    cotangent_vector = _as_real_numeric_array(
        "program AD signal convolve cotangent", cotangent
    ).reshape(-1)
    output_size = _program_ad_signal_convolve_output_size(left_shape, right_shape, mode)
    if cotangent_vector.size != output_size:
        raise ValueError("program AD signal convolve VJP requires cotangent matching output size")

    left_adjoint = np.zeros(left.size, dtype=np.float64)
    right_adjoint = np.zeros(right.size, dtype=np.float64)
    for index in range(left.size):
        basis = np.zeros(left.size, dtype=np.float64)
        basis[index] = 1.0
        left_adjoint[index] = float(
            np.sum(np.convolve(basis, right, mode=mode) * cotangent_vector)
        )
    for index in range(right.size):
        basis = np.zeros(right.size, dtype=np.float64)
        basis[index] = 1.0
        right_adjoint[index] = float(
            np.sum(np.convolve(left, basis, mode=mode) * cotangent_vector)
        )
    return np.concatenate([left_adjoint, right_adjoint])


def program_ad_signal_convolve_derivative_rule(
    left_shape: Sequence[int],
    right_shape: Sequence[int],
    *,
    mode: object = "full",
) -> CustomDerivativeRule:
    """Build an exact direct derivative rule for fixed static ``np.convolve`` operands."""

    left = _program_ad_signal_convolve_static_shape("left", left_shape)
    right = _program_ad_signal_convolve_static_shape("right", right_shape)
    mode_value = _normalise_convolve_mode(mode)

    def value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
        return _program_ad_signal_convolve_direct_value(
            values, left_shape=left, right_shape=right, mode=mode_value
        )

    def jvp_rule(values: NDArray[np.float64], tangent: NDArray[np.float64]) -> NDArray[np.float64]:
        return _program_ad_signal_convolve_direct_jvp(
            values, tangent, left_shape=left, right_shape=right, mode=mode_value
        )

    def vjp_rule(
        values: NDArray[np.float64], cotangent: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return _program_ad_signal_convolve_direct_vjp(
            values, cotangent, left_shape=left, right_shape=right, mode=mode_value
        )

    return CustomDerivativeRule(
        name=(
            "program_ad_signal_convolve_"
            f"left{left[0]}_right{right[0]}_mode_{mode_value}_direct_rule"
        ),
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
    )


def _program_ad_signal_correlate_static_shape(role: str, shape: Sequence[int]) -> tuple[int, ...]:
    normalised = _program_ad_array_normalise_static_shape(f"signal correlate {role}", shape)
    if len(normalised) != 1:
        raise ValueError("program AD signal correlate direct rule requires rank-1 operands")
    if normalised[0] <= 0:
        raise ValueError("program AD signal correlate direct rule requires non-empty operands")
    return normalised


def _program_ad_signal_correlate_split_source(
    role: str,
    values: NDArray[np.float64],
    *,
    left_shape: tuple[int, ...],
    right_shape: tuple[int, ...],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    vector = _as_real_numeric_array(f"program AD signal correlate {role}", values).reshape(-1)
    left_size = _program_ad_array_static_size(left_shape)
    expected_size = _program_ad_signal_convolve_source_size(left_shape, right_shape)
    if vector.size != expected_size:
        raise ValueError(
            f"program AD signal correlate direct rule requires {expected_size} {role} values"
        )
    return vector[:left_size], vector[left_size:]


def _program_ad_signal_correlate_output_size(
    left_shape: tuple[int, ...],
    right_shape: tuple[int, ...],
    mode: Literal["full", "same", "valid"],
) -> int:
    left = np.zeros(left_shape[0], dtype=np.float64)
    right = np.zeros(right_shape[0], dtype=np.float64)
    return int(np.correlate(left, right, mode=mode).size)


def _program_ad_signal_correlate_direct_value(
    values: NDArray[np.float64],
    *,
    left_shape: tuple[int, ...],
    right_shape: tuple[int, ...],
    mode: Literal["full", "same", "valid"],
) -> NDArray[np.float64]:
    left, right = _program_ad_signal_correlate_split_source(
        "values", values, left_shape=left_shape, right_shape=right_shape
    )
    return np.correlate(left, right, mode=mode).astype(np.float64, copy=False)


def _program_ad_signal_correlate_direct_jvp(
    values: NDArray[np.float64],
    tangent: NDArray[np.float64],
    *,
    left_shape: tuple[int, ...],
    right_shape: tuple[int, ...],
    mode: Literal["full", "same", "valid"],
) -> NDArray[np.float64]:
    left, right = _program_ad_signal_correlate_split_source(
        "values", values, left_shape=left_shape, right_shape=right_shape
    )
    tangent_left, tangent_right = _program_ad_signal_correlate_split_source(
        "tangent", tangent, left_shape=left_shape, right_shape=right_shape
    )
    return (
        np.correlate(tangent_left, right, mode=mode) + np.correlate(left, tangent_right, mode=mode)
    ).astype(np.float64, copy=False)


def _program_ad_signal_correlate_direct_vjp(
    values: NDArray[np.float64],
    cotangent: NDArray[np.float64],
    *,
    left_shape: tuple[int, ...],
    right_shape: tuple[int, ...],
    mode: Literal["full", "same", "valid"],
) -> NDArray[np.float64]:
    left, right = _program_ad_signal_correlate_split_source(
        "values", values, left_shape=left_shape, right_shape=right_shape
    )
    cotangent_vector = _as_real_numeric_array(
        "program AD signal correlate cotangent", cotangent
    ).reshape(-1)
    output_size = _program_ad_signal_correlate_output_size(left_shape, right_shape, mode)
    if cotangent_vector.size != output_size:
        raise ValueError("program AD signal correlate VJP requires cotangent matching output size")

    left_adjoint = np.zeros(left.size, dtype=np.float64)
    right_adjoint = np.zeros(right.size, dtype=np.float64)
    for index in range(left.size):
        basis = np.zeros(left.size, dtype=np.float64)
        basis[index] = 1.0
        left_adjoint[index] = float(
            np.sum(np.correlate(basis, right, mode=mode) * cotangent_vector)
        )
    for index in range(right.size):
        basis = np.zeros(right.size, dtype=np.float64)
        basis[index] = 1.0
        right_adjoint[index] = float(
            np.sum(np.correlate(left, basis, mode=mode) * cotangent_vector)
        )
    return np.concatenate([left_adjoint, right_adjoint])


def program_ad_signal_correlate_derivative_rule(
    left_shape: Sequence[int],
    right_shape: Sequence[int],
    *,
    mode: object = "valid",
) -> CustomDerivativeRule:
    """Build an exact direct derivative rule for fixed static ``np.correlate`` operands."""

    left = _program_ad_signal_correlate_static_shape("left", left_shape)
    right = _program_ad_signal_correlate_static_shape("right", right_shape)
    mode_value = _normalise_correlate_mode(mode)

    def value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
        return _program_ad_signal_correlate_direct_value(
            values, left_shape=left, right_shape=right, mode=mode_value
        )

    def jvp_rule(values: NDArray[np.float64], tangent: NDArray[np.float64]) -> NDArray[np.float64]:
        return _program_ad_signal_correlate_direct_jvp(
            values, tangent, left_shape=left, right_shape=right, mode=mode_value
        )

    def vjp_rule(
        values: NDArray[np.float64], cotangent: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return _program_ad_signal_correlate_direct_vjp(
            values, cotangent, left_shape=left, right_shape=right, mode=mode_value
        )

    return CustomDerivativeRule(
        name=(
            "program_ad_signal_correlate_"
            f"left{left[0]}_right{right[0]}_mode_{mode_value}_direct_rule"
        ),
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
    )


def _program_ad_signal_direct_value(_values: NDArray[np.float64]) -> NDArray[np.float64]:
    raise ValueError(
        "program AD signal primitive contracts are executable only through "
        "operator-intercepted trace dispatch or fixed-shape derivative factories"
    )


def _program_ad_signal_direct_jvp(
    _values: NDArray[np.float64],
    _tangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    raise ValueError(
        "program AD signal primitive contracts are executable only through "
        "operator-intercepted trace dispatch or fixed-shape derivative factories"
    )


def _program_ad_signal_derivative_rule(name: str) -> CustomDerivativeRule:
    if name in {"convolve", "correlate"}:
        return CustomDerivativeRule(
            name=f"program_ad_signal_{name}_trace_contract",
            value_fn=_program_ad_signal_direct_value,
            jvp_rule=_program_ad_signal_direct_jvp,
        )
    raise ValueError(f"unsupported program AD signal primitive {name}")


def _program_ad_signal_shape_of(name: str, value: object) -> tuple[int, ...]:
    shape = _program_ad_array_shape_of(value)
    if len(shape) != 1:
        raise ValueError(f"program AD signal {name} shape rule requires rank-1 operands")
    if shape[0] <= 0:
        raise ValueError(f"program AD signal {name} shape rule requires non-empty operands")
    return shape


def _program_ad_signal_convolve_static_parts(
    args: tuple[object, ...],
) -> tuple[tuple[int, ...], tuple[int, ...], Literal["full", "same", "valid"]]:
    if len(args) != 3:
        raise ValueError("program AD signal convolve requires left, right, and mode")
    left_shape = _program_ad_signal_shape_of("convolve", args[0])
    right_shape = _program_ad_signal_shape_of("convolve", args[1])
    mode = _normalise_convolve_mode(args[2])
    return left_shape, right_shape, mode


def _program_ad_signal_convolve_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    left_shape, right_shape, mode = _program_ad_signal_convolve_static_parts(args)
    return (_program_ad_signal_convolve_output_size(left_shape, right_shape, mode),)


def _program_ad_signal_convolve_dtype_rule(args: tuple[object, ...]) -> str:
    if len(args) != 3:
        raise ValueError("program AD signal convolve dtype rule requires left, right, and mode")
    return str(
        np.result_type(
            np.dtype(_program_ad_array_dtype_of(args[0])),
            np.dtype(_program_ad_array_dtype_of(args[1])),
        )
    )


def _program_ad_signal_convolve_static_arguments(
    args: tuple[object, ...],
) -> tuple[object, ...]:
    left_shape, right_shape, mode = _program_ad_signal_convolve_static_parts(args)
    return left_shape, right_shape, mode


def _program_ad_signal_correlate_static_parts(
    args: tuple[object, ...],
) -> tuple[tuple[int, ...], tuple[int, ...], Literal["full", "same", "valid"]]:
    if len(args) != 3:
        raise ValueError("program AD signal correlate requires left, right, and mode")
    left_shape = _program_ad_signal_shape_of("correlate", args[0])
    right_shape = _program_ad_signal_shape_of("correlate", args[1])
    mode = _normalise_correlate_mode(args[2])
    return left_shape, right_shape, mode


def _program_ad_signal_correlate_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    left_shape, right_shape, mode = _program_ad_signal_correlate_static_parts(args)
    return (_program_ad_signal_correlate_output_size(left_shape, right_shape, mode),)


def _program_ad_signal_correlate_static_arguments(
    args: tuple[object, ...],
) -> tuple[object, ...]:
    left_shape, right_shape, mode = _program_ad_signal_correlate_static_parts(args)
    return left_shape, right_shape, mode


def _program_ad_signal_batching_rule(
    function: Callable[..., object],
    args: tuple[object, ...],
    axes: tuple[int | None, ...],
    out_axes: int,
) -> object:
    if len(args) != 3 or len(axes) != 3:
        raise ValueError("program AD signal convolve batching requires left, right, and mode")
    if any(axis is not None for axis in axes[1:]):
        raise ValueError("program AD signal convolve batching keeps right operand and mode static")
    if axes[0] is None:
        return _as_real_numeric_array("program AD signal convolve batched output", function(*args))

    left_batch = _as_real_numeric_array("program AD signal convolve batched left operand", args[0])
    batch_axis = _normalise_axis("axes[0]", axes[0], left_batch.ndim)
    outputs = [
        _as_real_numeric_array(
            "program AD signal convolve batched output",
            function(np.take(left_batch, batch_index, axis=batch_axis), args[1], args[2]),
        )
        for batch_index in range(left_batch.shape[batch_axis])
    ]
    stacked = np.stack(outputs, axis=0)
    return np.moveaxis(stacked, 0, _normalise_axis("out_axes", out_axes, stacked.ndim))


def _program_ad_signal_lowering_metadata(name: str) -> Mapping[str, str]:
    factory_names = {
        "convolve": "program_ad_signal_convolve_derivative_rule",
        "correlate": "program_ad_signal_correlate_derivative_rule",
    }
    if name not in factory_names:
        raise ValueError(f"unsupported program AD signal primitive {name}")
    return {
        "program_ad": "operator_intercepted_trace",
        "mlir": "available: scpn_diff signal dialect interchange; executable lowering blocked",
        "mlir_op": f"scpn_diff.signal.{name}",
        "llvm": "blocked_until_executable_signal_lowering",
        "rust": "available: bounded compact Program AD Rust value+gradient replay",
        "static_argument_rule": "required",
        "static_derivative_factory": factory_names[name],
        "static_signature": "left_shape:rank1;right_shape:rank1;mode",
        "nondifferentiable_boundary": "rank1_nonempty_static_mode_window",
        "nondifferentiable_boundary_policy": "fail_closed",
    }


_PROGRAM_AD_SIGNAL_SHAPE_RULES: Mapping[str, PrimitiveShapeRule] = {
    "convolve": _program_ad_signal_convolve_shape,
    "correlate": _program_ad_signal_correlate_shape,
}

_PROGRAM_AD_SIGNAL_STATIC_ARGUMENT_RULES: Mapping[str, PrimitiveStaticArgumentRule] = {
    "convolve": _program_ad_signal_convolve_static_arguments,
    "correlate": _program_ad_signal_correlate_static_arguments,
}


def _register_program_ad_signal_primitive_contracts() -> None:
    """Register fail-closed Program AD signal primitive contracts."""

    for name, identity in _PROGRAM_AD_SIGNAL_IDENTITIES.items():
        if DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.contract_for(identity) is not None:
            continue
        DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.register_transform(
            PrimitiveTransformRule(
                identity=identity,
                derivative_rule=_program_ad_signal_derivative_rule(name),
                batching_rule=_program_ad_signal_batching_rule,
                lowering_metadata=_program_ad_signal_lowering_metadata(name),
                shape_rule=_PROGRAM_AD_SIGNAL_SHAPE_RULES[name],
                dtype_rule=_program_ad_signal_convolve_dtype_rule,
                static_argument_rule=_PROGRAM_AD_SIGNAL_STATIC_ARGUMENT_RULES[name],
                nondifferentiable_policy=_PROGRAM_AD_SIGNAL_POLICY,
                effect="pure",
            )
        )


def _validate_program_ad_signal_contract_dispatch(
    contract: PrimitiveContract,
    args: tuple[object, ...],
) -> None:
    """Validate signal primitive dispatch helpers against concrete arguments."""

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


def _require_program_ad_signal_contract(
    name: str,
    args: tuple[object, ...] | None = None,
) -> PrimitiveContract:
    """Return and validate a registered signal primitive runtime contract."""

    identity: PrimitiveIdentity | None = _PROGRAM_AD_SIGNAL_IDENTITIES.get(name)
    if identity is None:
        raise ValueError(f"no program AD signal primitive identity registered for {name}")
    contract = DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.require_contract(identity)
    if contract.nondifferentiable_policy != _PROGRAM_AD_SIGNAL_POLICY:
        raise ValueError(f"invalid program AD signal primitive policy for {identity.key}")
    if contract.effect != "pure":
        raise ValueError(f"invalid program AD signal primitive effect for {identity.key}")

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
            f"incomplete program AD signal primitive runtime contract for "
            f"{identity.key}: missing {joined}"
        )
    if args is not None:
        _validate_program_ad_signal_contract_dispatch(contract, args)
    return contract


__all__ = (
    "_convolve_output_window",
    "_normalise_convolve_mode",
    "_normalise_correlate_mode",
    "_program_ad_signal_convolve_output_size",
    "_program_ad_signal_correlate_output_size",
    "_register_program_ad_signal_primitive_contracts",
    "_require_program_ad_signal_contract",
    "program_ad_signal_convolve_derivative_rule",
    "program_ad_signal_correlate_derivative_rule",
)
