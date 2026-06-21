# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Program AD interpolation primitive rules
"""Static interpolation derivative rules for Program AD."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .differentiable_parameter_contracts import _as_real_numeric_array, _as_real_scalar
from .program_ad_array_indexing import (
    _normalise_axis,
    _program_ad_array_direct_jvp,
    _program_ad_array_direct_value,
    _program_ad_array_normalise_static_shape,
    _program_ad_array_signature,
    _program_ad_array_static_size,
)
from .program_ad_registry import (
    _PROGRAM_AD_INTERPOLATION_IDENTITIES,
    _PROGRAM_AD_INTERPOLATION_POLICY,
    DEFAULT_CUSTOM_DERIVATIVE_REGISTRY,
    CustomDerivativeRule,
    PrimitiveContract,
    PrimitiveIdentity,
    PrimitiveTransformRule,
)


def _is_program_ad_trace_value(value: object) -> bool:
    """Return whether ``value`` behaves like a whole-program trace value."""

    return type(value).__name__ in {"TraceADArray", "TraceADScalar"}


def _is_program_ad_trace_array(value: object) -> bool:
    """Return whether ``value`` behaves like a whole-program trace array."""

    return type(value).__name__ == "TraceADArray"


def _is_program_ad_trace_scalar(value: object) -> bool:
    """Return whether ``value`` behaves like a whole-program trace scalar."""

    return type(value).__name__ == "TraceADScalar"


def _program_ad_trace_shape(value: object) -> tuple[int, ...]:
    """Return a static shape from a structural trace-array value."""

    shape = getattr(value, "shape", None)
    if not isinstance(shape, tuple):
        raise ValueError("program AD interpolation trace array shape must be static")
    return tuple(int(dimension) for dimension in shape)


def _normalise_interp_grid(xp: object) -> NDArray[np.float64]:
    if _is_program_ad_trace_value(xp):
        raise ValueError("program AD np.interp xp grid must be static real numeric")
    grid = _as_real_numeric_array("program AD np.interp xp grid", xp)
    if grid.ndim != 1:
        raise ValueError("program AD np.interp xp grid must be one-dimensional")
    if grid.size < 2:
        raise ValueError("program AD np.interp xp grid requires at least two samples")
    if not bool(np.all(np.isfinite(grid))):
        raise ValueError("program AD np.interp xp grid must contain only finite values")
    if not bool(np.all(np.diff(grid) > 0.0)):
        raise ValueError("program AD np.interp xp grid must be strictly increasing")
    return grid


def _program_ad_interp_static_boundary(name: str, value: object) -> float | None:
    if value is None:
        return None
    if _is_program_ad_trace_value(value):
        raise ValueError(f"program AD np.interp {name} boundary must be static real numeric")
    return _as_real_scalar(f"program AD interpolation interp {name}", value)


def _program_ad_interp_split_source(
    primitive_name: str,
    role: str,
    values: NDArray[np.float64],
    *,
    sample_shape: tuple[int, ...],
    grid_size: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    vector = _as_real_numeric_array(
        f"program AD interpolation {primitive_name} {role}", values
    ).reshape(-1)
    sample_size = _program_ad_array_static_size(sample_shape)
    expected_size = sample_size + grid_size
    if vector.size != expected_size:
        raise ValueError(
            f"program AD interpolation {primitive_name} direct rule requires "
            f"{expected_size} {role} values"
        )
    return vector[:sample_size], vector[sample_size:]


def _program_ad_interp_segment(
    sample: float,
    grid: NDArray[np.float64],
) -> tuple[Literal["left", "right", "interior"], int, float]:
    if not math.isfinite(sample):
        raise ValueError("program AD interpolation interp samples must be finite")
    if bool(np.any(grid == sample)):
        raise ValueError("program AD interpolation interp samples must avoid grid knots")
    if sample < float(grid[0]):
        return ("left", 0, 0.0)
    if sample > float(grid[-1]):
        return ("right", grid.size - 1, 0.0)
    segment = int(np.searchsorted(grid, sample, side="right") - 1)
    lower = float(grid[segment])
    upper = float(grid[segment + 1])
    weight = (sample - lower) / (upper - lower)
    return ("interior", segment, weight)


def _program_ad_interp_direct_value(
    values: NDArray[np.float64],
    *,
    sample_shape: tuple[int, ...],
    grid: NDArray[np.float64],
    left: float | None,
    right: float | None,
) -> NDArray[np.float64]:
    samples, fp_values = _program_ad_interp_split_source(
        "interp", "values", values, sample_shape=sample_shape, grid_size=grid.size
    )
    outputs = np.zeros(samples.size, dtype=np.float64)
    for index, sample in enumerate(samples):
        region, segment, weight = _program_ad_interp_segment(float(sample), grid)
        if region == "left":
            outputs[index] = fp_values[0] if left is None else left
        elif region == "right":
            outputs[index] = fp_values[-1] if right is None else right
        else:
            outputs[index] = (1.0 - weight) * fp_values[segment] + weight * fp_values[segment + 1]
    return outputs


def _program_ad_interp_direct_jvp(
    values: NDArray[np.float64],
    tangent: NDArray[np.float64],
    *,
    sample_shape: tuple[int, ...],
    grid: NDArray[np.float64],
    left: float | None,
    right: float | None,
) -> NDArray[np.float64]:
    samples, fp_values = _program_ad_interp_split_source(
        "interp", "values", values, sample_shape=sample_shape, grid_size=grid.size
    )
    sample_tangent, fp_tangent = _program_ad_interp_split_source(
        "interp", "tangent", tangent, sample_shape=sample_shape, grid_size=grid.size
    )
    outputs = np.zeros(samples.size, dtype=np.float64)
    for index, sample in enumerate(samples):
        region, segment, weight = _program_ad_interp_segment(float(sample), grid)
        if region == "left":
            outputs[index] = fp_tangent[0] if left is None else 0.0
        elif region == "right":
            outputs[index] = fp_tangent[-1] if right is None else 0.0
        else:
            width = float(grid[segment + 1] - grid[segment])
            slope = (fp_values[segment + 1] - fp_values[segment]) / width
            outputs[index] = (
                slope * sample_tangent[index]
                + (1.0 - weight) * fp_tangent[segment]
                + weight * fp_tangent[segment + 1]
            )
    return outputs


def _program_ad_interp_direct_vjp(
    values: NDArray[np.float64],
    cotangent: NDArray[np.float64],
    *,
    sample_shape: tuple[int, ...],
    grid: NDArray[np.float64],
    left: float | None,
    right: float | None,
) -> NDArray[np.float64]:
    samples, fp_values = _program_ad_interp_split_source(
        "interp", "values", values, sample_shape=sample_shape, grid_size=grid.size
    )
    cotangent_vector = _as_real_numeric_array(
        "program AD interpolation interp cotangent", cotangent
    ).reshape(-1)
    if cotangent_vector.size != samples.size:
        raise ValueError(
            "program AD interpolation interp VJP requires cotangent matching sample size"
        )
    sample_adjoint = np.zeros(samples.size, dtype=np.float64)
    fp_adjoint = np.zeros(grid.size, dtype=np.float64)
    for index, sample in enumerate(samples):
        region, segment, weight = _program_ad_interp_segment(float(sample), grid)
        scalar_cotangent = float(cotangent_vector[index])
        if region == "left":
            if left is None:
                fp_adjoint[0] += scalar_cotangent
        elif region == "right":
            if right is None:
                fp_adjoint[-1] += scalar_cotangent
        else:
            width = float(grid[segment + 1] - grid[segment])
            slope = (fp_values[segment + 1] - fp_values[segment]) / width
            sample_adjoint[index] += scalar_cotangent * slope
            fp_adjoint[segment] += scalar_cotangent * (1.0 - weight)
            fp_adjoint[segment + 1] += scalar_cotangent * weight
    return np.concatenate([sample_adjoint, fp_adjoint])


def program_ad_interpolation_interp_derivative_rule(
    sample_shape: Sequence[int],
    xp: object,
    value_shape: Sequence[int],
    *,
    left: object = None,
    right: object = None,
    period: object = None,
) -> CustomDerivativeRule:
    """Build an exact direct derivative rule for fixed static ``np.interp`` grids."""

    if period is not None:
        raise ValueError("program AD interpolation interp direct rule does not support period")
    samples = _program_ad_array_normalise_static_shape("interp", sample_shape)
    grid = _normalise_interp_grid(xp)
    fp_shape = tuple(int(dimension) for dimension in value_shape)
    if fp_shape != (grid.size,):
        raise ValueError(
            "program AD interpolation interp direct rule requires fp shape to match xp"
        )
    left_value = _program_ad_interp_static_boundary("left", left)
    right_value = _program_ad_interp_static_boundary("right", right)

    def value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
        return _program_ad_interp_direct_value(
            values,
            sample_shape=samples,
            grid=grid,
            left=left_value,
            right=right_value,
        )

    def jvp_rule(values: NDArray[np.float64], tangent: NDArray[np.float64]) -> NDArray[np.float64]:
        return _program_ad_interp_direct_jvp(
            values,
            tangent,
            sample_shape=samples,
            grid=grid,
            left=left_value,
            right=right_value,
        )

    def vjp_rule(
        values: NDArray[np.float64], cotangent: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return _program_ad_interp_direct_vjp(
            values,
            cotangent,
            sample_shape=samples,
            grid=grid,
            left=left_value,
            right=right_value,
        )

    return CustomDerivativeRule(
        name=(
            "program_ad_interpolation_interp_"
            f"x{_program_ad_array_signature(samples)}_grid{grid.size}_static_direct_rule"
        ),
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
    )


def _program_ad_interpolation_sample_shape(value: object) -> tuple[int, ...]:
    """Return the static sample shape for a trace or concrete interpolation input."""

    if _is_program_ad_trace_array(value):
        return _program_ad_trace_shape(value)
    if _is_program_ad_trace_scalar(value):
        return ()
    return tuple(int(dimension) for dimension in np.asarray(value).shape)


def _program_ad_interpolation_fp_shape(value: object) -> tuple[int, ...]:
    """Return the static interpolation value-grid shape."""

    if _is_program_ad_trace_array(value):
        return _program_ad_trace_shape(value)
    if _is_program_ad_trace_scalar(value):
        raise ValueError("program AD interpolation interp fp must be one-dimensional")
    return tuple(int(dimension) for dimension in np.asarray(value).shape)


def _program_ad_interpolation_static_parts(
    args: tuple[object, ...],
) -> tuple[tuple[int, ...], NDArray[np.float64], tuple[int, ...], float | None, float | None]:
    """Validate and normalise static ``np.interp`` dispatch arguments."""

    if len(args) != 6:
        raise ValueError(
            "program AD interpolation interp rule requires x, xp, fp, left, right, and period"
        )
    if args[5] is not None:
        raise ValueError("program AD interpolation interp period is not supported")
    sample_shape = _program_ad_interpolation_sample_shape(args[0])
    grid = _normalise_interp_grid(args[1])
    fp_shape = _program_ad_interpolation_fp_shape(args[2])
    if fp_shape != (grid.size,):
        raise ValueError("program AD np.interp fp values must match xp grid")
    left = _program_ad_interp_static_boundary("left", args[3])
    right = _program_ad_interp_static_boundary("right", args[4])
    return sample_shape, grid, fp_shape, left, right


def _program_ad_interpolation_interp_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    """Return the output shape for an intercepted ``np.interp`` primitive."""

    sample_shape, _grid, _fp_shape, _left, _right = _program_ad_interpolation_static_parts(args)
    return sample_shape


def _program_ad_interpolation_interp_dtype_rule(_args: tuple[object, ...]) -> str:
    """Return the dtype emitted by Program AD interpolation primitives."""

    return "float64"


def _program_ad_interpolation_interp_static_arguments(
    args: tuple[object, ...],
) -> tuple[object, ...]:
    """Return canonical static arguments for an intercepted ``np.interp`` primitive."""

    sample_shape, grid, fp_shape, left, right = _program_ad_interpolation_static_parts(args)
    return (
        sample_shape,
        ("xp", tuple(int(dimension) for dimension in grid.shape), tuple(float(x) for x in grid)),
        fp_shape,
        left,
        right,
        None,
    )


def _program_ad_interpolation_derivative_rule(name: str) -> CustomDerivativeRule:
    """Return the trace-dispatched derivative contract for an interpolation primitive."""

    if name == "interp":
        return CustomDerivativeRule(
            name="program_ad_interpolation_interp_trace_contract",
            value_fn=_program_ad_array_direct_value,
            jvp_rule=_program_ad_array_direct_jvp,
        )
    raise ValueError(f"unsupported program AD interpolation primitive {name}")


def _program_ad_interpolation_batching_rule(
    function: Callable[..., object],
    args: tuple[object, ...],
    axes: tuple[int | None, ...],
    out_axes: int,
) -> object:
    """Map interpolation samples over a batch axis while static data is shared."""

    if len(args) != 6 or len(axes) != 6:
        raise ValueError(
            "program AD interpolation interp batching requires x, xp, fp, left, right, and period"
        )
    if any(axis is not None for axis in axes[1:]):
        raise ValueError(
            "program AD interpolation interp batching keeps xp, fp, left, right, and period static"
        )
    if axes[0] is None:
        return _as_real_numeric_array(
            "program AD interpolation interp batched output", function(*args)
        )
    samples = _as_real_numeric_array("program AD interpolation interp batched samples", args[0])
    batch_axis = _normalise_axis("axes[0]", axes[0], samples.ndim)
    outputs = [
        _as_real_numeric_array(
            "program AD interpolation interp batched output",
            function(np.take(samples, batch_index, axis=batch_axis), *args[1:]),
        )
        for batch_index in range(samples.shape[batch_axis])
    ]
    stacked = np.stack(outputs, axis=0)
    return np.moveaxis(stacked, 0, _normalise_axis("out_axes", out_axes, stacked.ndim))


def _program_ad_interpolation_lowering_metadata(name: str) -> Mapping[str, str]:
    """Return lowering metadata for a Program AD interpolation primitive."""

    if name != "interp":
        raise ValueError(f"unsupported program AD interpolation primitive {name}")
    return {
        "program_ad": "operator_intercepted_trace",
        "mlir": "available: scpn_diff interpolation dialect interchange; executable lowering blocked",
        "mlir_op": "scpn_diff.interpolation.interp",
        "llvm": "blocked_until_executable_interpolation_lowering",
        "rust": "blocked_until_polyglot_interpolation_ad",
        "static_argument_rule": "required",
        "static_derivative_factory": "program_ad_interpolation_interp_derivative_rule",
        "static_signature": "sample_shape:ranked_tensor_shape;xp_grid;fp_shape;left_right_period",
        "nondifferentiable_boundary": "static_grid_knot_and_period_boundary",
        "nondifferentiable_boundary_policy": "fail_closed",
    }


def _register_program_ad_interpolation_primitive_contracts() -> None:
    """Register fail-closed Program AD interpolation primitive contracts."""

    for name, identity in _PROGRAM_AD_INTERPOLATION_IDENTITIES.items():
        if DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.contract_for(identity) is not None:
            continue
        DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.register_transform(
            PrimitiveTransformRule(
                identity=identity,
                derivative_rule=_program_ad_interpolation_derivative_rule(name),
                batching_rule=_program_ad_interpolation_batching_rule,
                lowering_metadata=_program_ad_interpolation_lowering_metadata(name),
                shape_rule=_program_ad_interpolation_interp_shape,
                dtype_rule=_program_ad_interpolation_interp_dtype_rule,
                static_argument_rule=_program_ad_interpolation_interp_static_arguments,
                nondifferentiable_policy=_PROGRAM_AD_INTERPOLATION_POLICY,
                effect="pure",
            )
        )


def _validate_program_ad_interpolation_contract_dispatch(
    contract: PrimitiveContract,
    args: tuple[object, ...],
) -> None:
    """Validate interpolation primitive dispatch helpers against concrete arguments."""

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


def _require_program_ad_interpolation_contract(
    name: str,
    args: tuple[object, ...] | None = None,
) -> PrimitiveContract:
    """Return and validate a registered interpolation primitive runtime contract."""

    identity: PrimitiveIdentity | None = _PROGRAM_AD_INTERPOLATION_IDENTITIES.get(name)
    if identity is None:
        raise ValueError(f"no program AD interpolation primitive identity registered for {name}")
    contract = DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.require_contract(identity)
    if contract.nondifferentiable_policy != _PROGRAM_AD_INTERPOLATION_POLICY:
        raise ValueError(f"invalid program AD interpolation primitive policy for {identity.key}")
    if contract.effect != "pure":
        raise ValueError(f"invalid program AD interpolation primitive effect for {identity.key}")

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
            f"incomplete program AD interpolation primitive runtime contract for "
            f"{identity.key}: missing {joined}"
        )
    if args is not None:
        _validate_program_ad_interpolation_contract_dispatch(contract, args)
    return contract


__all__ = (
    "_normalise_interp_grid",
    "_program_ad_interp_static_boundary",
    "_register_program_ad_interpolation_primitive_contracts",
    "_require_program_ad_interpolation_contract",
    "program_ad_interpolation_interp_derivative_rule",
)
