# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Program AD stencil primitive rules
"""Static finite-difference stencil derivative rules for Program AD."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .differentiable_parameter_contracts import _as_real_numeric_array, _as_real_scalar
from .program_ad_array_indexing import _normalise_axis, _program_ad_array_shape_of
from .program_ad_registry import (
    _PROGRAM_AD_STENCIL_IDENTITIES,
    _PROGRAM_AD_STENCIL_POLICY,
    DEFAULT_CUSTOM_DERIVATIVE_REGISTRY,
    CustomDerivativeRule,
    PrimitiveContract,
    PrimitiveIdentity,
    PrimitiveTransformRule,
)
from .program_ad_shape_transforms import _program_ad_shape_signature

_GradientSpacing = (
    tuple[Literal["scalar"], float] | tuple[Literal["coordinates"], NDArray[np.float64]]
)


def _normalise_gradient_edge_order(edge_order: object) -> int:
    if isinstance(edge_order, (bool, np.bool_)) or not isinstance(edge_order, (int, np.integer)):
        raise ValueError("program AD np.gradient edge_order must be 1 or 2")
    edge = int(edge_order)
    if edge not in {1, 2}:
        raise ValueError("program AD np.gradient edge_order must be 1 or 2")
    return edge


def _normalise_gradient_axis(axis: int, ndim: int) -> int:
    value = axis
    if value < 0:
        value += ndim
    if value < 0 or value >= ndim:
        raise ValueError("program AD np.gradient axis out of bounds")
    return value


def _normalise_gradient_axes(axis: object, ndim: int) -> tuple[int, ...]:
    if axis is None:
        return tuple(range(ndim))
    if isinstance(axis, (bool, np.bool_)):
        raise ValueError("program AD np.gradient axis must be a static integer")
    if isinstance(axis, (int, np.integer)):
        return (_normalise_gradient_axis(int(axis), ndim),)
    if not isinstance(axis, tuple) or not axis:
        raise ValueError("program AD np.gradient axis must be a static integer")
    axes: list[int] = []
    for item in axis:
        if isinstance(item, (bool, np.bool_)) or not isinstance(item, (int, np.integer)):
            raise ValueError("program AD np.gradient axis must be a static integer")
        axes.append(_normalise_gradient_axis(int(item), ndim))
    if len(set(axes)) != len(axes):
        raise ValueError("program AD np.gradient axes must be unique")
    return tuple(axes)


def _is_program_ad_trace_value(value: object) -> bool:
    return type(value).__name__ in {"TraceADArray", "TraceADScalar"}


def _is_static_gradient_scalar_spacing(spacing: object) -> bool:
    if _is_program_ad_trace_value(spacing):
        return False
    try:
        raw = np.asarray(spacing)
    except ValueError:
        return False
    return raw.shape == () and raw.dtype.kind not in {"b", "O", "S", "U", "c"}


def _normalise_gradient_spacing(spacing: object, axis_size: int) -> _GradientSpacing:
    if _is_program_ad_trace_value(spacing):
        raise ValueError("program AD np.gradient spacing must be static real numeric")
    raw = np.asarray(spacing)
    if raw.shape == ():
        scalar = _as_real_scalar("program AD np.gradient spacing", spacing)
        if scalar == 0.0:
            raise ValueError("program AD np.gradient spacing must be non-zero")
        return ("scalar", scalar)
    coordinates = _as_real_numeric_array("program AD np.gradient coordinates", spacing)
    if coordinates.ndim != 1 or coordinates.shape[0] != axis_size:
        raise ValueError("program AD np.gradient coordinates must match the differentiation axis")
    if not bool(np.all(np.isfinite(coordinates))):
        raise ValueError("program AD np.gradient coordinates must contain only finite values")
    deltas = np.diff(coordinates)
    if not bool(np.all(deltas > 0.0)) and not bool(np.all(deltas < 0.0)):
        raise ValueError("program AD np.gradient coordinates must be strictly monotonic")
    return ("coordinates", coordinates)


def _normalise_gradient_spacings(
    spacings: tuple[object, ...], axes: tuple[int, ...], shape: tuple[int, ...]
) -> tuple[_GradientSpacing, ...]:
    if not spacings:
        raw_spacings: tuple[object, ...] = tuple(1.0 for _ in axes)
    elif len(spacings) == 1:
        if len(axes) == 1 or _is_static_gradient_scalar_spacing(spacings[0]):
            raw_spacings = tuple(spacings[0] for _ in axes)
        else:
            raise ValueError("program AD np.gradient spacing count must match axes")
    elif len(spacings) == len(axes):
        raw_spacings = spacings
    else:
        raise ValueError("program AD np.gradient spacing count must match axes")
    return tuple(
        _normalise_gradient_spacing(spacing, shape[axis])
        for spacing, axis in zip(raw_spacings, axes, strict=True)
    )


def _gradient_axis_coefficients(
    position: int,
    axis_size: int,
    spacing: _GradientSpacing,
    edge_order: int,
) -> tuple[tuple[int, float], ...]:
    if axis_size < edge_order + 1:
        raise ValueError(
            f"program AD np.gradient edge_order {edge_order} requires "
            f"at least {edge_order + 1} samples"
        )
    if spacing[0] == "scalar":
        dx = spacing[1]
        if position == 0:
            if edge_order == 1:
                return ((0, -1.0 / dx), (1, 1.0 / dx))
            return ((0, -1.5 / dx), (1, 2.0 / dx), (2, -0.5 / dx))
        if position == axis_size - 1:
            if edge_order == 1:
                return ((axis_size - 2, -1.0 / dx), (axis_size - 1, 1.0 / dx))
            return (
                (axis_size - 3, 0.5 / dx),
                (axis_size - 2, -2.0 / dx),
                (axis_size - 1, 1.5 / dx),
            )
        return ((position - 1, -0.5 / dx), (position + 1, 0.5 / dx))

    coordinates = spacing[1]
    if position == 0:
        dx1 = float(coordinates[1] - coordinates[0])
        if edge_order == 1:
            return ((0, -1.0 / dx1), (1, 1.0 / dx1))
        dx2 = float(coordinates[2] - coordinates[1])
        return (
            (0, -(2.0 * dx1 + dx2) / (dx1 * (dx1 + dx2))),
            (1, (dx1 + dx2) / (dx1 * dx2)),
            (2, -dx1 / (dx2 * (dx1 + dx2))),
        )
    if position == axis_size - 1:
        dx1 = float(coordinates[-2] - coordinates[-3])
        dx2 = float(coordinates[-1] - coordinates[-2])
        if edge_order == 1:
            return ((axis_size - 2, -1.0 / dx2), (axis_size - 1, 1.0 / dx2))
        return (
            (axis_size - 3, dx2 / (dx1 * (dx1 + dx2))),
            (axis_size - 2, -(dx1 + dx2) / (dx1 * dx2)),
            (axis_size - 1, (dx1 + 2.0 * dx2) / (dx2 * (dx1 + dx2))),
        )
    dx1 = float(coordinates[position] - coordinates[position - 1])
    dx2 = float(coordinates[position + 1] - coordinates[position])
    return (
        (position - 1, -dx2 / (dx1 * (dx1 + dx2))),
        (position, (dx2 - dx1) / (dx1 * dx2)),
        (position + 1, dx1 / (dx2 * (dx1 + dx2))),
    )


def _program_ad_stencil_gradient_source_vector(
    role: str,
    values: NDArray[np.float64],
    *,
    source_shape: tuple[int, ...],
) -> NDArray[np.float64]:
    vector = _as_real_numeric_array(f"program AD stencil gradient {role}", values).reshape(-1)
    expected_size = int(np.prod(source_shape, dtype=np.int64))
    if vector.size != expected_size:
        raise ValueError(
            f"program AD stencil gradient direct rule requires {expected_size} {role} values"
        )
    return vector


def _program_ad_stencil_gradient_flat_components(
    source: NDArray[np.float64],
    *,
    axes: tuple[int, ...],
    spacings: tuple[_GradientSpacing, ...],
    edge_order: int,
) -> tuple[NDArray[np.float64], ...]:
    components: list[NDArray[np.float64]] = []
    for axis_index, spacing in zip(axes, spacings, strict=True):
        component = np.zeros_like(source, dtype=np.float64)
        for flat_index in range(source.size):
            target_index = np.unravel_index(flat_index, source.shape)
            total = 0.0
            for source_axis_index, coefficient in _gradient_axis_coefficients(
                int(target_index[axis_index]),
                int(source.shape[axis_index]),
                spacing,
                edge_order,
            ):
                source_index = (
                    target_index[:axis_index]
                    + (source_axis_index,)
                    + target_index[axis_index + 1 :]
                )
                total += float(source[source_index]) * coefficient
            component[target_index] = total
        components.append(component.reshape(-1))
    return tuple(components)


def _program_ad_stencil_gradient_flat_value(
    values: NDArray[np.float64],
    *,
    source_shape: tuple[int, ...],
    axes: tuple[int, ...],
    spacings: tuple[_GradientSpacing, ...],
    edge_order: int,
) -> NDArray[np.float64]:
    source = _program_ad_stencil_gradient_source_vector(
        "values", values, source_shape=source_shape
    ).reshape(source_shape)
    components = _program_ad_stencil_gradient_flat_components(
        source,
        axes=axes,
        spacings=spacings,
        edge_order=edge_order,
    )
    return np.concatenate(components).astype(np.float64, copy=False)


def _program_ad_stencil_gradient_flat_vjp(
    values: NDArray[np.float64],
    cotangent: NDArray[np.float64],
    *,
    source_shape: tuple[int, ...],
    axes: tuple[int, ...],
    spacings: tuple[_GradientSpacing, ...],
    edge_order: int,
) -> NDArray[np.float64]:
    _program_ad_stencil_gradient_source_vector("values", values, source_shape=source_shape)
    source_size = int(np.prod(source_shape, dtype=np.int64))
    cotangent_vector = _as_real_numeric_array(
        "program AD stencil gradient cotangent", cotangent
    ).reshape(-1)
    expected_cotangent_size = source_size * len(axes)
    if cotangent_vector.size != expected_cotangent_size:
        raise ValueError(
            f"program AD stencil gradient VJP requires {expected_cotangent_size} cotangent values"
        )
    adjoint = np.zeros(source_shape, dtype=np.float64)
    cotangent_components = cotangent_vector.reshape((len(axes), source_size))
    for component_cotangent, axis_index, spacing in zip(
        cotangent_components,
        axes,
        spacings,
        strict=True,
    ):
        cotangent_array = component_cotangent.reshape(source_shape)
        for flat_index in range(source_size):
            target_index = np.unravel_index(flat_index, source_shape)
            scalar_cotangent = float(cotangent_array[target_index])
            for source_axis_index, coefficient in _gradient_axis_coefficients(
                int(target_index[axis_index]),
                int(source_shape[axis_index]),
                spacing,
                edge_order,
            ):
                source_index = (
                    target_index[:axis_index]
                    + (source_axis_index,)
                    + target_index[axis_index + 1 :]
                )
                adjoint[source_index] += scalar_cotangent * coefficient
    return adjoint.reshape(-1)


def _program_ad_gradient_spacing_signature(
    spacing: _GradientSpacing,
) -> tuple[str, float] | tuple[str, tuple[int, ...], tuple[float, ...]]:
    if spacing[0] == "scalar":
        return ("scalar", float(spacing[1]))
    coordinates = np.asarray(spacing[1], dtype=np.float64)
    return (
        "coordinates",
        tuple(int(dimension) for dimension in coordinates.shape),
        tuple(float(value) for value in coordinates.reshape(-1)),
    )


def program_ad_stencil_gradient_derivative_rule(
    source_shape: Sequence[int],
    spacings: Sequence[object] = (),
    *,
    axis: object = None,
    edge_order: object = 1,
) -> CustomDerivativeRule:
    """Build an exact direct derivative rule for a fixed ``np.gradient`` signature."""
    source = tuple(int(dimension) for dimension in source_shape)
    if any(dimension <= 0 for dimension in source):
        raise ValueError(
            "program AD stencil gradient direct rule requires positive source dimensions"
        )
    edge = _normalise_gradient_edge_order(edge_order)
    axes = _normalise_gradient_axes(axis, len(source))
    spacing_specs = _normalise_gradient_spacings(tuple(spacings), axes, source)

    def value_fn(values: NDArray[np.float64]) -> NDArray[np.float64]:
        return _program_ad_stencil_gradient_flat_value(
            values,
            source_shape=source,
            axes=axes,
            spacings=spacing_specs,
            edge_order=edge,
        )

    def jvp_rule(values: NDArray[np.float64], tangent: NDArray[np.float64]) -> NDArray[np.float64]:
        _program_ad_stencil_gradient_source_vector("values", values, source_shape=source)
        return _program_ad_stencil_gradient_flat_value(
            tangent,
            source_shape=source,
            axes=axes,
            spacings=spacing_specs,
            edge_order=edge,
        )

    def vjp_rule(
        values: NDArray[np.float64], cotangent: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return _program_ad_stencil_gradient_flat_vjp(
            values,
            cotangent,
            source_shape=source,
            axes=axes,
            spacings=spacing_specs,
            edge_order=edge,
        )

    axes_signature = "_".join(str(axis_index) for axis_index in axes)
    return CustomDerivativeRule(
        name=(
            "program_ad_stencil_gradient_"
            f"{_program_ad_shape_signature(source)}_axes_{axes_signature}_edge_{edge}_direct_rule"
        ),
        value_fn=value_fn,
        jvp_rule=jvp_rule,
        vjp_rule=vjp_rule,
    )


def _validate_program_ad_stencil_contract_dispatch(
    contract: PrimitiveContract,
    args: tuple[object, ...],
) -> None:
    """Validate dispatch-time stencil static, shape, and dtype contract hooks."""
    if contract.static_argument_rule is None:
        raise ValueError(
            f"program AD stencil primitive {contract.identity.key} missing static argument rule"
        )
    if contract.shape_rule is None:
        raise ValueError(
            f"program AD stencil primitive {contract.identity.key} missing shape rule"
        )
    if contract.dtype_rule is None:
        raise ValueError(
            f"program AD stencil primitive {contract.identity.key} missing dtype rule"
        )
    static_arguments = contract.static_argument_rule(args)
    if not isinstance(static_arguments, tuple):
        raise ValueError(
            f"program AD stencil primitive {contract.identity.key} static rule must return a tuple"
        )
    shape = contract.shape_rule(args)
    if not isinstance(shape, tuple) or any(
        not isinstance(dimension, int) or dimension < 0 for dimension in shape
    ):
        raise ValueError(
            f"program AD stencil primitive {contract.identity.key} shape rule must return "
            "non-negative integer dimensions"
        )
    dtype = contract.dtype_rule(args)
    if not isinstance(dtype, str) or not dtype:
        raise ValueError(
            f"program AD stencil primitive {contract.identity.key} dtype rule must return a dtype name"
        )


def _require_program_ad_stencil_runtime_contract(
    name: str,
    *,
    identities: Mapping[str, PrimitiveIdentity],
    expected_policy: str,
    args: tuple[object, ...] | None = None,
) -> PrimitiveContract:
    """Return a fully dispatchable stencil primitive contract or fail closed."""
    identity = identities.get(name)
    if identity is None:
        raise ValueError(f"no program AD stencil primitive identity registered for {name}")
    contract = DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.require_contract(identity)
    if contract.nondifferentiable_policy != expected_policy:
        raise ValueError(f"invalid program AD stencil primitive policy for {identity.key}")
    if contract.effect != "pure":
        raise ValueError(f"invalid program AD stencil primitive effect for {identity.key}")

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
            f"incomplete program AD stencil primitive runtime contract for "
            f"{identity.key}: missing {joined}"
        )
    if args is not None:
        _validate_program_ad_stencil_contract_dispatch(contract, args)
    return contract


def _program_ad_stencil_direct_value(_values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Reject direct execution for trace-only stencil primitive contracts."""
    raise ValueError(
        "program AD stencil primitive contracts are executable only through "
        "operator-intercepted trace dispatch or fixed-shape derivative factories"
    )


def _program_ad_stencil_direct_jvp(
    _values: NDArray[np.float64],
    _tangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Reject direct JVP execution for trace-only stencil primitive contracts."""
    raise ValueError(
        "program AD stencil primitive contracts are executable only through "
        "operator-intercepted trace dispatch or fixed-shape derivative factories"
    )


def _program_ad_stencil_derivative_rule(name: str) -> CustomDerivativeRule:
    """Return the trace-only derivative contract for a stencil primitive."""
    if name == "gradient":
        return CustomDerivativeRule(
            name="program_ad_stencil_gradient_trace_contract",
            value_fn=_program_ad_stencil_direct_value,
            jvp_rule=_program_ad_stencil_direct_jvp,
        )
    raise ValueError(f"unsupported program AD stencil primitive {name}")


def _program_ad_stencil_shape_of(arg: object) -> tuple[int, ...]:
    """Return a positive static shape for a concrete or trace stencil source."""
    shape = _program_ad_array_shape_of(arg)
    if any(dimension <= 0 for dimension in shape):
        raise ValueError("program AD stencil gradient requires positive source dimensions")
    return shape


def _program_ad_stencil_spacings_arg(arg: object) -> tuple[object, ...]:
    """Normalise a static stencil spacing bundle."""
    if isinstance(arg, tuple):
        return arg
    if isinstance(arg, list):
        return tuple(arg)
    raise ValueError("program AD stencil gradient static rule requires spacing values as a tuple")


def _program_ad_stencil_gradient_static_parts(
    args: tuple[object, ...],
) -> tuple[tuple[int, ...], tuple[_GradientSpacing, ...], tuple[int, ...], int]:
    """Return the fully validated static signature for ``np.gradient``."""
    if len(args) != 4:
        raise ValueError(
            "program AD stencil gradient static rule requires source, spacings, axis, and edge_order"
        )
    source_shape = _program_ad_stencil_shape_of(args[0])
    edge = _normalise_gradient_edge_order(args[3])
    axes = _normalise_gradient_axes(args[2], len(source_shape))
    spacing_specs = _normalise_gradient_spacings(
        _program_ad_stencil_spacings_arg(args[1]),
        axes,
        source_shape,
    )
    return source_shape, spacing_specs, axes, edge


def _program_ad_stencil_gradient_shape(args: tuple[object, ...]) -> tuple[int, ...]:
    """Return the static output shape for a stencil gradient contract."""
    source_shape, _spacing_specs, axes, _edge = _program_ad_stencil_gradient_static_parts(args)
    if len(axes) == 1:
        return source_shape
    return (len(axes), *source_shape)


def _program_ad_stencil_dtype_rule(_args: tuple[object, ...]) -> str:
    """Return the dtype emitted by Program AD stencil gradient traces."""
    return "float64"


def _program_ad_stencil_gradient_static_arguments(
    args: tuple[object, ...],
) -> tuple[object, ...]:
    """Return the hashable static signature for a stencil gradient contract."""
    source_shape, spacing_specs, axes, edge = _program_ad_stencil_gradient_static_parts(args)
    return (
        source_shape,
        tuple(_program_ad_gradient_spacing_signature(spacing) for spacing in spacing_specs),
        axes,
        edge,
    )


def _program_ad_stencil_array_output(value: object) -> NDArray[np.float64]:
    """Convert a concrete stencil output or multi-axis output bundle to an array."""
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("program AD stencil batched output must not be empty")
        return np.stack(
            [
                _as_real_numeric_array("program AD stencil batched output", component)
                for component in value
            ],
            axis=0,
        )
    return _as_real_numeric_array("program AD stencil batched output", value)


def _program_ad_stencil_batching_rule(
    function: Callable[..., object],
    args: tuple[object, ...],
    axes: tuple[int | None, ...],
    out_axes: int,
) -> object:
    """Map a stencil gradient over one non-differentiated source axis."""
    if len(args) != 4 or len(axes) != 4:
        raise ValueError(
            "program AD stencil gradient batching requires source, spacings, axis, and edge_order"
        )
    if any(axis is not None for axis in axes[1:]):
        raise ValueError(
            "program AD stencil gradient batching keeps spacing, axis, and edge_order static"
        )
    if axes[0] is None:
        raise ValueError("program AD stencil gradient batching requires a mapped source axis")

    source = _as_real_numeric_array("program AD stencil batched source", args[0])
    batch_axis = _normalise_axis("axes[0]", axes[0], source.ndim)
    gradient_axes = _normalise_gradient_axes(args[2], source.ndim)
    if batch_axis in gradient_axes:
        raise ValueError("program AD stencil gradient cannot batch over a differentiated axis")
    adjusted_gradient_axes = tuple(
        axis_index - 1 if axis_index > batch_axis else axis_index for axis_index in gradient_axes
    )
    adjusted_axis_arg: object = (
        adjusted_gradient_axes[0] if len(adjusted_gradient_axes) == 1 else adjusted_gradient_axes
    )

    outputs = [
        _program_ad_stencil_array_output(
            function(
                np.take(source, batch_index, axis=batch_axis),
                args[1],
                adjusted_axis_arg,
                args[3],
            )
        )
        for batch_index in range(source.shape[batch_axis])
    ]
    stacked = np.stack(outputs, axis=0)
    axis_index = _normalise_axis("out_axes", out_axes, stacked.ndim)
    return np.moveaxis(stacked, 0, axis_index)


def _program_ad_stencil_lowering_metadata(name: str) -> Mapping[str, str]:
    """Return lowering metadata for the stencil primitive contract."""
    if name != "gradient":
        raise ValueError(f"unsupported program AD stencil primitive {name}")
    return {
        "program_ad": "operator_intercepted_trace",
        "mlir": "available: scpn_diff stencil dialect interchange; executable lowering blocked",
        "mlir_op": "scpn_diff.stencil.gradient",
        "llvm": "blocked_until_executable_stencil_lowering",
        "rust": "blocked_until_polyglot_stencil_ad",
        "nondifferentiable_boundary": "static_spacing_axis_edge_order",
        "nondifferentiable_boundary_policy": "fail_closed",
        "static_argument_rule": "required",
        "static_derivative_factory": "program_ad_stencil_gradient_derivative_rule",
        "static_signature": "source_shape:ranked_tensor_shape;spacing_axis_edge_order",
    }


def _register_program_ad_stencil_primitive_contracts() -> None:
    """Register Program AD stencil primitive contracts in the shared registry."""
    for name, identity in _PROGRAM_AD_STENCIL_IDENTITIES.items():
        if DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.contract_for(identity) is not None:
            continue
        DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.register_transform(
            PrimitiveTransformRule(
                identity=identity,
                derivative_rule=_program_ad_stencil_derivative_rule(name),
                batching_rule=_program_ad_stencil_batching_rule,
                lowering_metadata=_program_ad_stencil_lowering_metadata(name),
                shape_rule=_program_ad_stencil_gradient_shape,
                dtype_rule=_program_ad_stencil_dtype_rule,
                static_argument_rule=_program_ad_stencil_gradient_static_arguments,
                nondifferentiable_policy=_PROGRAM_AD_STENCIL_POLICY,
                effect="pure",
            )
        )


def _require_program_ad_stencil_contract(
    name: str,
    args: tuple[object, ...] | None = None,
) -> PrimitiveContract:
    """Return a dispatch-ready Program AD stencil primitive contract."""
    return _require_program_ad_stencil_runtime_contract(
        name,
        identities=_PROGRAM_AD_STENCIL_IDENTITIES,
        expected_policy=_PROGRAM_AD_STENCIL_POLICY,
        args=args,
    )
