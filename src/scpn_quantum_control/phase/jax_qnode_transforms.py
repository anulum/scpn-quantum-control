# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — JAX Registered-QNode Transforms
"""Native JAX execution and transforms for registered local Phase-QNodes.

This one-way leaf owns deterministic statevector, flat/PyTree transform,
PMAP-sharding, and AOT/export routes. The compatibility facade injects its
optional-JAX loader and retains later compatibility and maturity orchestration.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .jax_bridge_contracts import (
    PhaseJAXPhaseQNodeAOTExportResult,
    PhaseJAXPhaseQNodeNativeTransformResult,
    PhaseJAXPhaseQNodePyTreeTransformResult,
    PhaseJAXPhaseQNodeShardingTransformResult,
    PhaseJAXPhaseQNodeStatevectorResult,
)
from .jax_gradients import (
    JAXLoader,
    _as_non_negative_tolerance,
    _as_parameter_vector,
    _as_scalar,
    _load_jax,
)
from .qnode_circuit import (
    DenseHermitianObservable,
    PauliCovarianceObservable,
    PauliTerm,
    PhaseQNodeCircuit,
    PhaseQNodeOperation,
    PhaseQNodeSupportError,
    SparsePauliHamiltonian,
    parameter_shift_phase_qnode_gradient,
    phase_qnode_support_report,
)

FloatArray: TypeAlias = NDArray[np.float64]
JAXCallable = Callable[[object], object]


def _as_parameter_batch(name: str, values: object, *, width: int) -> FloatArray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    if matrix.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one parameter row")
    if matrix.shape[1] != width:
        raise ValueError(f"{name} must have shape (batch, {width}), got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    return matrix.astype(np.float64, copy=True)


def _require_jax_pmap_support(jax_module: Any) -> None:
    pmap = getattr(jax_module, "pmap", None)
    if not callable(pmap):
        raise RuntimeError("JAX pmap is required for bounded-QNN sharding compatibility")
    value_and_grad = getattr(jax_module, "value_and_grad", None)
    if not callable(value_and_grad):
        raise RuntimeError("JAX value_and_grad is required for bounded-QNN sharding compatibility")
    local_device_count = getattr(jax_module, "local_device_count", None)
    if not callable(local_device_count):
        raise RuntimeError(
            "JAX local_device_count is required for bounded-QNN sharding compatibility"
        )


def _require_jax_phase_qnode_transform_support(jax_module: Any) -> None:
    required = (
        "grad",
        "value_and_grad",
        "jacfwd",
        "jacrev",
        "hessian",
        "jvp",
        "vjp",
        "vmap",
        "jit",
    )
    missing = tuple(name for name in required if not callable(getattr(jax_module, name, None)))
    if missing:
        missing_names = ", ".join(missing)
        raise RuntimeError(
            "JAX native transforms are required for registered Phase-QNode lowering: "
            f"{missing_names}"
        )


def _require_jax_phase_qnode_aot_export_support(jax_module: Any) -> Any:
    jit = getattr(jax_module, "jit", None)
    if not callable(jit):
        raise RuntimeError("JAX JIT is required for registered Phase-QNode AOT lowering")
    if not callable(getattr(jax_module, "ShapeDtypeStruct", None)):
        raise RuntimeError(
            "JAX ShapeDtypeStruct is required for registered Phase-QNode AOT export"
        )
    export_module = getattr(jax_module, "export", None)
    if export_module is None:
        raise RuntimeError("JAX export is required for registered Phase-QNode AOT export")
    for name in ("export", "deserialize"):
        if not callable(getattr(export_module, name, None)):
            raise RuntimeError(f"JAX export.{name} is required for Phase-QNode AOT export")
    for name in (
        "minimum_supported_calling_convention_version",
        "maximum_supported_calling_convention_version",
    ):
        if getattr(export_module, name, None) is None:
            raise RuntimeError(f"JAX export.{name} is required for Phase-QNode AOT export")
    return export_module


def _jax_export_version(export_module: Any, name: str) -> int:
    """Return a JAX export calling-convention version from attr or function APIs."""
    value = getattr(export_module, name)
    if callable(value):
        value = value()
    version = int(value)
    if version < 0:
        raise RuntimeError(f"JAX export.{name} must be non-negative")
    return version


def _require_jax_phase_qnode_pytree_transform_support(jax_module: Any) -> None:
    required = (
        "grad",
        "value_and_grad",
        "jacfwd",
        "jacrev",
        "hessian",
        "jvp",
        "vjp",
        "vmap",
        "jit",
    )
    missing = tuple(name for name in required if not callable(getattr(jax_module, name, None)))
    if missing:
        missing_names = ", ".join(missing)
        raise RuntimeError(
            "JAX PyTree transforms are required for registered Phase-QNode lowering: "
            f"{missing_names}"
        )


def _require_jax_pytree_support(jax_module: Any) -> None:
    tree_util = getattr(jax_module, "tree_util", None)
    if tree_util is None:
        raise RuntimeError("JAX PyTree tree_util support is required for JAX PyTree parameters")
    if not callable(getattr(tree_util, "tree_flatten", None)):
        raise RuntimeError("JAX PyTree tree_flatten is required for JAX PyTree parameters")
    if not callable(getattr(tree_util, "tree_unflatten", None)):
        raise RuntimeError("JAX PyTree tree_unflatten is required for JAX PyTree parameters")
    value_and_grad = getattr(jax_module, "value_and_grad", None)
    if not callable(value_and_grad):
        raise RuntimeError("JAX value_and_grad is required for JAX PyTree parameters")


def _jax_local_devices(jax_module: Any, local_device_count: int) -> tuple[str, ...]:
    local_devices = getattr(jax_module, "local_devices", None)
    if not callable(local_devices):
        return tuple(f"local-device-{index}" for index in range(local_device_count))
    devices = tuple(str(device) for device in local_devices())
    if len(devices) != local_device_count:
        return tuple(f"local-device-{index}" for index in range(local_device_count))
    return devices


def _as_pytree_parameter_vector(
    jax_module: Any,
    name: str,
    params_pytree: object,
) -> tuple[FloatArray, object, tuple[tuple[int, ...], ...], tuple[int, ...]]:
    """Validate and flatten a numeric parameter PyTree with reconstruction metadata."""
    leaves, treedef = jax_module.tree_util.tree_flatten(params_pytree)
    if not leaves:
        raise ValueError(f"{name} must contain at least one numeric leaf")
    arrays = []
    shapes = []
    sizes = []
    for index, leaf in enumerate(leaves):
        array = np.asarray(leaf, dtype=float)
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} leaf {index} must contain only finite values")
        arrays.append(array.astype(np.float64, copy=True))
        shapes.append(tuple(int(axis) for axis in array.shape))
        sizes.append(int(array.size))
    vector = np.concatenate([array.reshape(-1) for array in arrays])
    return cast(FloatArray, vector), treedef, tuple(shapes), tuple(sizes)


def _jax_flatten_pytree(jax_module: Any, jnp: Any, params_pytree: object) -> object:
    leaves, _treedef = jax_module.tree_util.tree_flatten(params_pytree)
    parts = [jnp.ravel(jnp.asarray(leaf)) for leaf in leaves]
    if len(parts) == 1:
        return parts[0]
    return jnp.concatenate(parts)


def _jax_unflatten_vector_to_pytree(
    jax_module: Any,
    jnp: Any,
    vector: object,
    treedef: object,
    leaf_shapes: tuple[tuple[int, ...], ...],
    leaf_sizes: tuple[int, ...],
) -> object:
    flat = jnp.ravel(jnp.asarray(vector))
    leaves = []
    offset = 0
    for shape, size in zip(leaf_shapes, leaf_sizes, strict=True):
        leaves.append(jnp.reshape(flat[offset : offset + size], shape))
        offset += size
    return jax_module.tree_util.tree_unflatten(treedef, leaves)


def _jax_unflatten_batch_to_pytree(
    jax_module: Any,
    jnp: Any,
    batch: object,
    treedef: object,
    leaf_shapes: tuple[tuple[int, ...], ...],
    leaf_sizes: tuple[int, ...],
) -> object:
    batch_values = jnp.asarray(batch)
    leaves = []
    offset = 0
    for shape, size in zip(leaf_shapes, leaf_sizes, strict=True):
        target_shape = (batch_values.shape[0], *shape)
        leaves.append(jnp.reshape(batch_values[:, offset : offset + size], target_shape))
        offset += size
    return jax_module.tree_util.tree_unflatten(treedef, leaves)


def _flatten_runtime_pytree_gradient(
    jax_module: Any,
    name: str,
    gradient_pytree: object,
    *,
    width: int,
) -> FloatArray:
    leaves, _treedef = jax_module.tree_util.tree_flatten(gradient_pytree)
    if not leaves:
        raise ValueError(f"{name} must contain at least one gradient leaf")
    vector = np.concatenate([np.asarray(leaf, dtype=float).reshape(-1) for leaf in leaves])
    if vector.shape != (width,):
        raise ValueError(f"{name} must flatten to shape ({width},), got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return cast(FloatArray, vector.astype(np.float64, copy=True))


def _flatten_batched_runtime_pytree_gradient(
    jax_module: Any,
    name: str,
    gradient_pytree: object,
    *,
    batch_size: int,
    width: int,
) -> FloatArray:
    leaves, _treedef = jax_module.tree_util.tree_flatten(gradient_pytree)
    if not leaves:
        raise ValueError(f"{name} must contain at least one gradient leaf")
    parts = []
    for index, leaf in enumerate(leaves):
        array = np.asarray(leaf, dtype=float)
        if array.shape[0] != batch_size:
            raise ValueError(f"{name} leaf {index} must have leading batch size {batch_size}")
        parts.append(array.reshape(batch_size, -1))
    matrix = np.concatenate(parts, axis=1)
    if matrix.shape != (batch_size, width):
        raise ValueError(
            f"{name} must flatten to shape ({batch_size}, {width}), got {matrix.shape}"
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    return cast(FloatArray, matrix.astype(np.float64, copy=True))


def _flatten_runtime_pytree_hessian(
    jax_module: Any,
    name: str,
    hessian_pytree: object,
    *,
    leaf_shapes: tuple[tuple[int, ...], ...],
    leaf_sizes: tuple[int, ...],
) -> FloatArray:
    blocks, _treedef = jax_module.tree_util.tree_flatten(hessian_pytree)
    leaf_count = len(leaf_sizes)
    expected_blocks = leaf_count * leaf_count
    if len(blocks) != expected_blocks:
        raise ValueError(f"{name} must contain {expected_blocks} Hessian blocks")
    width = int(sum(leaf_sizes))
    matrix = np.zeros((width, width), dtype=np.float64)
    row_offset = 0
    block_index = 0
    for row_shape, row_size in zip(leaf_shapes, leaf_sizes, strict=True):
        col_offset = 0
        for col_shape, col_size in zip(leaf_shapes, leaf_sizes, strict=True):
            block = np.asarray(blocks[block_index], dtype=float)
            expected_shape = (*row_shape, *col_shape)
            if block.shape != expected_shape:
                raise ValueError(
                    f"{name} block {block_index} must have shape {expected_shape}, "
                    f"got {block.shape}"
                )
            matrix[
                row_offset : row_offset + row_size,
                col_offset : col_offset + col_size,
            ] = block.reshape(row_size, col_size)
            col_offset += col_size
            block_index += 1
        row_offset += row_size
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    return cast(FloatArray, matrix)


def jax_phase_qnode_value_and_grad(
    circuit: PhaseQNodeCircuit,
    params: ArrayLike,
    *,
    tolerance: float = 1e-6,
    jit: bool = False,
    _jax_loader: JAXLoader = _load_jax,
) -> PhaseJAXPhaseQNodeStatevectorResult:
    """Lower a registered deterministic Phase-QNode statevector route into JAX.

    The accepted surface is the local pure-state ``PhaseQNodeCircuit`` gate and
    observable family. It deliberately excludes finite-shot sampling, provider
    callbacks, hardware execution, density/noise channels, and dynamic circuits.
    """
    jax_module, jnp = _jax_loader()
    _enable_jax_x64(jax_module)
    tolerance_value = _as_non_negative_tolerance(tolerance)
    parameter_values = _as_parameter_vector("params", params)
    report = phase_qnode_support_report(circuit, parameter_values)
    if not report.supported:
        raise PhaseQNodeSupportError(report)
    parameter_shift = parameter_shift_phase_qnode_gradient(circuit, parameter_values)

    def value_function(raw_params: object) -> object:
        value, _state = _jax_phase_qnode_value_and_state(jnp, circuit, raw_params)
        return value

    value_and_grad = jax_module.value_and_grad(value_function)
    if jit:
        value_and_grad = jax_module.jit(value_and_grad)
        state_function = jax_module.jit(
            lambda raw_params: _jax_phase_qnode_value_and_state(jnp, circuit, raw_params)[1]
        )
        jitted = True
    else:

        def state_function(raw_params: object) -> object:
            return _jax_phase_qnode_value_and_state(jnp, circuit, raw_params)[1]

        jitted = False

    value_obj, gradient_obj = value_and_grad(jnp.asarray(parameter_values))
    state_obj = state_function(jnp.asarray(parameter_values))
    gradient = _as_parameter_vector(
        "JAX Phase-QNode gradient", gradient_obj, width=parameter_values.size
    )
    state = np.asarray(state_obj, dtype=np.complex128)
    value = _as_scalar("JAX Phase-QNode value", value_obj)
    max_abs_error = float(np.max(np.abs(gradient - parameter_shift.gradient), initial=0.0))
    l2_error = float(np.linalg.norm(gradient - parameter_shift.gradient))
    passed = bool(
        abs(value - parameter_shift.value) <= tolerance_value and max_abs_error <= tolerance_value
    )
    return PhaseJAXPhaseQNodeStatevectorResult(
        value=value,
        gradient=gradient,
        state=state,
        parameter_shift_value=parameter_shift.value,
        parameter_shift_gradient=parameter_shift.gradient.copy(),
        max_abs_error=max_abs_error,
        l2_error=l2_error,
        tolerance=tolerance_value,
        passed=passed,
        native_framework_autodiff=True,
        host_callback=False,
        jit_requested=jit,
        jitted=jitted,
    )


def jax_phase_qnode_native_transform_audit(
    circuit: PhaseQNodeCircuit,
    params: ArrayLike,
    *,
    tangent: ArrayLike | None = None,
    batch_offsets: ArrayLike | None = None,
    tolerance: float = 1e-6,
    _jax_loader: JAXLoader = _load_jax,
) -> PhaseJAXPhaseQNodeNativeTransformResult:
    """Audit native JAX transforms for a registered local Phase-QNode.

    The executable route is the same deterministic statevector lowering used by
    :func:`jax_phase_qnode_value_and_grad`, but it is exercised through JAX
    ``grad``, ``value_and_grad``, ``jacfwd``, ``jacrev``, ``hessian``, ``jvp``,
    ``vjp``, ``vmap``, and ``jit``. It deliberately does not use host callbacks
    and does not promote finite-shot, provider, hardware, density/noise, or
    dynamic-circuit lowering.
    """
    jax_module, jnp = _jax_loader()
    _enable_jax_x64(jax_module)
    _require_jax_phase_qnode_transform_support(jax_module)
    tolerance_value = _as_non_negative_tolerance(tolerance)
    parameter_values = _as_parameter_vector("params", params)
    report = phase_qnode_support_report(circuit, parameter_values)
    if not report.supported:
        raise PhaseQNodeSupportError(report)

    if tangent is None:
        tangent_values = np.linspace(
            0.25,
            0.25 + 0.05 * (parameter_values.size - 1),
            parameter_values.size,
            dtype=np.float64,
        )
    else:
        tangent_values = _as_parameter_vector("tangent", tangent, width=parameter_values.size)
    if batch_offsets is None:
        offsets = np.vstack(
            (
                np.zeros(parameter_values.size, dtype=np.float64),
                np.eye(parameter_values.size, dtype=np.float64) * 0.03,
                -np.eye(parameter_values.size, dtype=np.float64) * 0.02,
            )
        )
        batch_values = parameter_values + offsets.reshape(-1, parameter_values.size)
    else:
        offset_values = _as_parameter_batch(
            "batch_offsets",
            batch_offsets,
            width=parameter_values.size,
        )
        batch_values = parameter_values + offset_values

    parameter_shift = parameter_shift_phase_qnode_gradient(circuit, parameter_values)

    def value_function(raw_params: object) -> object:
        value, _state = _jax_phase_qnode_value_and_state(jnp, circuit, raw_params)
        return value

    raw_params = jnp.asarray(parameter_values)
    raw_tangent = jnp.asarray(tangent_values)
    raw_batch = jnp.asarray(batch_values)
    gradient_obj = jax_module.grad(value_function)(raw_params)
    value_and_grad_fn = jax_module.value_and_grad(value_function)
    value_and_grad_value_obj, value_and_grad_gradient_obj = value_and_grad_fn(raw_params)
    jit_value_obj, jit_gradient_obj = jax_module.jit(value_and_grad_fn)(raw_params)
    jacfwd_obj = jax_module.jacfwd(value_function)(raw_params)
    jacrev_obj = jax_module.jacrev(value_function)(raw_params)
    hessian_obj = jax_module.hessian(value_function)(raw_params)
    jvp_value_obj, jvp_tangent_obj = jax_module.jvp(value_function, (raw_params,), (raw_tangent,))
    vjp_value_obj, pullback = jax_module.vjp(value_function, raw_params)
    (vjp_cotangent_obj,) = pullback(jnp.asarray(1.0))
    vmap_value_and_grad = jax_module.vmap(value_and_grad_fn)
    vmap_values_obj, vmap_gradients_obj = vmap_value_and_grad(raw_batch)

    gradient = _as_parameter_vector(
        "JAX grad Phase-QNode gradient", gradient_obj, width=parameter_values.size
    )
    value_and_grad_gradient = _as_parameter_vector(
        "JAX value_and_grad Phase-QNode gradient",
        value_and_grad_gradient_obj,
        width=parameter_values.size,
    )
    jit_gradient = _as_parameter_vector(
        "JAX jit value_and_grad Phase-QNode gradient",
        jit_gradient_obj,
        width=parameter_values.size,
    )
    jacfwd_gradient = _as_parameter_vector(
        "JAX jacfwd Phase-QNode gradient",
        jacfwd_obj,
        width=parameter_values.size,
    )
    jacrev_gradient = _as_parameter_vector(
        "JAX jacrev Phase-QNode gradient",
        jacrev_obj,
        width=parameter_values.size,
    )
    vjp_cotangent_gradient = _as_parameter_vector(
        "JAX vjp Phase-QNode cotangent gradient",
        vjp_cotangent_obj,
        width=parameter_values.size,
    )
    vmap_values = _as_parameter_vector(
        "JAX vmap Phase-QNode values",
        vmap_values_obj,
        width=batch_values.shape[0],
    )
    vmap_gradients = _as_parameter_batch(
        "JAX vmap Phase-QNode gradients",
        vmap_gradients_obj,
        width=parameter_values.size,
    )
    hessian_matrix = np.asarray(hessian_obj, dtype=np.float64)
    if hessian_matrix.shape != (parameter_values.size, parameter_values.size):
        raise RuntimeError("JAX Phase-QNode hessian has an unexpected shape")
    batch_parameter_shift_gradients = np.vstack(
        [parameter_shift_phase_qnode_gradient(circuit, row).gradient for row in batch_values]
    ).astype(np.float64, copy=False)
    reference_errors = (
        gradient - parameter_shift.gradient,
        value_and_grad_gradient - parameter_shift.gradient,
        jit_gradient - parameter_shift.gradient,
        jacfwd_gradient - parameter_shift.gradient,
        jacrev_gradient - parameter_shift.gradient,
        vjp_cotangent_gradient - parameter_shift.gradient,
        vmap_gradients - batch_parameter_shift_gradients,
    )
    max_abs_gradient_error = max(
        float(np.max(np.abs(error), initial=0.0)) for error in reference_errors
    )
    value_errors = (
        abs(
            _as_scalar("JAX value_and_grad Phase-QNode value", value_and_grad_value_obj)
            - parameter_shift.value
        ),
        abs(_as_scalar("JAX jit Phase-QNode value", jit_value_obj) - parameter_shift.value),
        abs(_as_scalar("JAX jvp Phase-QNode value", jvp_value_obj) - parameter_shift.value),
        abs(_as_scalar("JAX vjp Phase-QNode value", vjp_value_obj) - parameter_shift.value),
    )
    expected_jvp = float(np.dot(parameter_shift.gradient, tangent_values))
    max_abs_transform_error = max(
        *(float(error) for error in value_errors),
        abs(_as_scalar("JAX jvp Phase-QNode tangent", jvp_tangent_obj) - expected_jvp),
        float(
            np.max(
                np.abs(
                    vmap_values
                    - np.asarray(
                        [
                            parameter_shift_phase_qnode_gradient(circuit, row).value
                            for row in batch_values
                        ],
                        dtype=np.float64,
                    )
                ),
                initial=0.0,
            )
        ),
    )
    max_abs_hessian_symmetry_error = float(
        np.max(np.abs(hessian_matrix - hessian_matrix.T), initial=0.0)
    )
    passed = bool(
        max_abs_gradient_error <= tolerance_value
        and max_abs_transform_error <= tolerance_value
        and max_abs_hessian_symmetry_error <= tolerance_value
    )
    return PhaseJAXPhaseQNodeNativeTransformResult(
        value=_as_scalar("JAX grad Phase-QNode value", value_function(raw_params)),
        gradient=gradient,
        value_and_grad_value=_as_scalar(
            "JAX value_and_grad Phase-QNode value",
            value_and_grad_value_obj,
        ),
        value_and_grad_gradient=value_and_grad_gradient,
        jacfwd_gradient=jacfwd_gradient,
        jacrev_gradient=jacrev_gradient,
        hessian=hessian_matrix.copy(),
        jvp_value=_as_scalar("JAX jvp Phase-QNode value", jvp_value_obj),
        jvp_tangent=_as_scalar("JAX jvp Phase-QNode tangent", jvp_tangent_obj),
        vjp_value=_as_scalar("JAX vjp Phase-QNode value", vjp_value_obj),
        vjp_cotangent_gradient=vjp_cotangent_gradient,
        vmap_values=vmap_values,
        vmap_gradients=vmap_gradients,
        parameter_shift_value=parameter_shift.value,
        parameter_shift_gradient=parameter_shift.gradient.copy(),
        tangent=tangent_values.copy(),
        batch_params=batch_values.copy(),
        batch_parameter_shift_gradients=batch_parameter_shift_gradients.copy(),
        max_abs_gradient_error=max_abs_gradient_error,
        max_abs_transform_error=max_abs_transform_error,
        max_abs_hessian_symmetry_error=max_abs_hessian_symmetry_error,
        tolerance=tolerance_value,
        passed=passed,
        native_framework_autodiff=True,
        host_callback=False,
        jit_value_and_grad=True,
        vmap_value_and_grad=True,
        transform_names=(
            "grad",
            "value_and_grad",
            "jacfwd",
            "jacrev",
            "hessian",
            "jvp",
            "vjp",
            "vmap",
            "jit",
        ),
    )


def jax_phase_qnode_pytree_transform_audit(
    circuit: PhaseQNodeCircuit,
    params_pytree: object,
    *,
    tangent: ArrayLike | None = None,
    batch_offsets: ArrayLike | None = None,
    tolerance: float = 1e-6,
    _jax_loader: JAXLoader = _load_jax,
) -> PhaseJAXPhaseQNodePyTreeTransformResult:
    """Audit native JAX PyTree transforms for a registered local Phase-QNode.

    The route accepts nested numeric PyTree parameter containers, lowers them
    into the registered deterministic Phase-QNode statevector value function,
    and validates native JAX ``grad``, ``value_and_grad``, ``jacfwd``,
    ``jacrev``, ``hessian``, ``jvp``, ``vjp``, ``vmap``, and ``jit`` against the
    canonical SCPN parameter-shift gradient. It keeps the same fail-closed
    boundary as the flat transform audit: no host callbacks, no finite-shot
    lowering, no provider execution, no hardware submission, and no
    dynamic-circuit claim.
    """
    jax_module, jnp = _jax_loader()
    _enable_jax_x64(jax_module)
    _require_jax_pytree_support(jax_module)
    _require_jax_phase_qnode_pytree_transform_support(jax_module)
    tolerance_value = _as_non_negative_tolerance(tolerance)
    parameter_values, treedef, leaf_shapes, leaf_sizes = _as_pytree_parameter_vector(
        jax_module,
        "params_pytree",
        params_pytree,
    )
    report = phase_qnode_support_report(circuit, parameter_values)
    if not report.supported:
        raise PhaseQNodeSupportError(report)

    if tangent is None:
        tangent_values = np.linspace(
            0.25,
            0.25 + 0.05 * (parameter_values.size - 1),
            parameter_values.size,
            dtype=np.float64,
        )
    else:
        tangent_values = _as_parameter_vector("tangent", tangent, width=parameter_values.size)
    if batch_offsets is None:
        offsets = np.vstack(
            (
                np.zeros(parameter_values.size, dtype=np.float64),
                np.eye(parameter_values.size, dtype=np.float64) * 0.03,
                -np.eye(parameter_values.size, dtype=np.float64) * 0.02,
            )
        )
        batch_values = parameter_values + offsets.reshape(-1, parameter_values.size)
    else:
        offset_values = _as_parameter_batch(
            "batch_offsets",
            batch_offsets,
            width=parameter_values.size,
        )
        batch_values = parameter_values + offset_values

    parameter_shift = parameter_shift_phase_qnode_gradient(circuit, parameter_values)
    raw_params = _jax_unflatten_vector_to_pytree(
        jax_module,
        jnp,
        jnp.asarray(parameter_values),
        treedef,
        leaf_shapes,
        leaf_sizes,
    )
    raw_tangent = _jax_unflatten_vector_to_pytree(
        jax_module,
        jnp,
        jnp.asarray(tangent_values),
        treedef,
        leaf_shapes,
        leaf_sizes,
    )
    raw_batch = _jax_unflatten_batch_to_pytree(
        jax_module,
        jnp,
        jnp.asarray(batch_values),
        treedef,
        leaf_shapes,
        leaf_sizes,
    )

    def value_function(raw_tree: object) -> object:
        vector = _jax_flatten_pytree(jax_module, jnp, raw_tree)
        value, _state = _jax_phase_qnode_value_and_state(jnp, circuit, vector)
        return value

    gradient_pytree_obj = jax_module.grad(value_function)(raw_params)
    value_and_grad_fn = jax_module.value_and_grad(value_function)
    value_and_grad_value_obj, value_and_grad_gradient_obj = value_and_grad_fn(raw_params)
    jit_value_obj, jit_gradient_obj = jax_module.jit(value_and_grad_fn)(raw_params)
    jacfwd_obj = jax_module.jacfwd(value_function)(raw_params)
    jacrev_obj = jax_module.jacrev(value_function)(raw_params)
    hessian_obj = jax_module.hessian(value_function)(raw_params)
    jvp_value_obj, jvp_tangent_obj = jax_module.jvp(value_function, (raw_params,), (raw_tangent,))
    vjp_value_obj, pullback = jax_module.vjp(value_function, raw_params)
    (vjp_cotangent_obj,) = pullback(jnp.asarray(1.0))
    vmap_value_and_grad = jax_module.vmap(value_and_grad_fn)
    vmap_values_obj, vmap_gradients_obj = vmap_value_and_grad(raw_batch)

    gradient = _flatten_runtime_pytree_gradient(
        jax_module,
        "JAX grad Phase-QNode PyTree gradient",
        gradient_pytree_obj,
        width=parameter_values.size,
    )
    value_and_grad_gradient = _flatten_runtime_pytree_gradient(
        jax_module,
        "JAX value_and_grad Phase-QNode PyTree gradient",
        value_and_grad_gradient_obj,
        width=parameter_values.size,
    )
    jit_gradient = _flatten_runtime_pytree_gradient(
        jax_module,
        "JAX jit Phase-QNode PyTree gradient",
        jit_gradient_obj,
        width=parameter_values.size,
    )
    jacfwd_gradient = _flatten_runtime_pytree_gradient(
        jax_module,
        "JAX jacfwd Phase-QNode PyTree gradient",
        jacfwd_obj,
        width=parameter_values.size,
    )
    jacrev_gradient = _flatten_runtime_pytree_gradient(
        jax_module,
        "JAX jacrev Phase-QNode PyTree gradient",
        jacrev_obj,
        width=parameter_values.size,
    )
    hessian_matrix = _flatten_runtime_pytree_hessian(
        jax_module,
        "JAX hessian Phase-QNode PyTree matrix",
        hessian_obj,
        leaf_shapes=leaf_shapes,
        leaf_sizes=leaf_sizes,
    )
    vjp_cotangent_gradient = _flatten_runtime_pytree_gradient(
        jax_module,
        "JAX vjp Phase-QNode PyTree cotangent gradient",
        vjp_cotangent_obj,
        width=parameter_values.size,
    )
    vmap_values = _as_parameter_vector(
        "JAX vmap Phase-QNode PyTree values",
        vmap_values_obj,
        width=batch_values.shape[0],
    )
    vmap_gradients = _flatten_batched_runtime_pytree_gradient(
        jax_module,
        "JAX vmap Phase-QNode PyTree gradients",
        vmap_gradients_obj,
        batch_size=batch_values.shape[0],
        width=parameter_values.size,
    )
    batch_parameter_shift_gradients = np.vstack(
        [parameter_shift_phase_qnode_gradient(circuit, row).gradient for row in batch_values]
    ).astype(np.float64, copy=False)
    reference_errors = (
        gradient - parameter_shift.gradient,
        value_and_grad_gradient - parameter_shift.gradient,
        jit_gradient - parameter_shift.gradient,
        jacfwd_gradient - parameter_shift.gradient,
        jacrev_gradient - parameter_shift.gradient,
        vjp_cotangent_gradient - parameter_shift.gradient,
        vmap_gradients - batch_parameter_shift_gradients,
    )
    max_abs_gradient_error = max(
        float(np.max(np.abs(error), initial=0.0)) for error in reference_errors
    )
    expected_jvp = float(np.dot(parameter_shift.gradient, tangent_values))
    value_errors = (
        abs(
            _as_scalar("JAX value_and_grad Phase-QNode PyTree value", value_and_grad_value_obj)
            - parameter_shift.value
        ),
        abs(_as_scalar("JAX jit Phase-QNode PyTree value", jit_value_obj) - parameter_shift.value),
        abs(_as_scalar("JAX jvp Phase-QNode PyTree value", jvp_value_obj) - parameter_shift.value),
        abs(_as_scalar("JAX vjp Phase-QNode PyTree value", vjp_value_obj) - parameter_shift.value),
        abs(_as_scalar("JAX jvp Phase-QNode PyTree tangent", jvp_tangent_obj) - expected_jvp),
        float(
            np.max(
                np.abs(
                    vmap_values
                    - np.asarray(
                        [
                            parameter_shift_phase_qnode_gradient(circuit, row).value
                            for row in batch_values
                        ],
                        dtype=np.float64,
                    )
                ),
                initial=0.0,
            )
        ),
    )
    max_abs_transform_error = max(float(error) for error in value_errors)
    max_abs_hessian_symmetry_error = float(
        np.max(np.abs(hessian_matrix - hessian_matrix.T), initial=0.0)
    )
    passed = bool(
        max_abs_gradient_error <= tolerance_value
        and max_abs_transform_error <= tolerance_value
        and max_abs_hessian_symmetry_error <= tolerance_value
    )
    return PhaseJAXPhaseQNodePyTreeTransformResult(
        value=_as_scalar("JAX grad Phase-QNode PyTree value", value_function(raw_params)),
        gradient=gradient,
        gradient_pytree=gradient_pytree_obj,
        value_and_grad_value=_as_scalar(
            "JAX value_and_grad Phase-QNode PyTree value",
            value_and_grad_value_obj,
        ),
        value_and_grad_gradient=value_and_grad_gradient,
        value_and_grad_gradient_pytree=value_and_grad_gradient_obj,
        jacfwd_gradient=jacfwd_gradient,
        jacrev_gradient=jacrev_gradient,
        hessian=hessian_matrix.copy(),
        hessian_pytree=hessian_obj,
        jvp_value=_as_scalar("JAX jvp Phase-QNode PyTree value", jvp_value_obj),
        jvp_tangent=_as_scalar("JAX jvp Phase-QNode PyTree tangent", jvp_tangent_obj),
        vjp_value=_as_scalar("JAX vjp Phase-QNode PyTree value", vjp_value_obj),
        vjp_cotangent_gradient=vjp_cotangent_gradient,
        vmap_values=vmap_values,
        vmap_gradients=vmap_gradients,
        parameter_shift_value=parameter_shift.value,
        parameter_shift_gradient=parameter_shift.gradient.copy(),
        parameter_vector=parameter_values.copy(),
        tangent=tangent_values.copy(),
        batch_params=batch_values.copy(),
        batch_parameter_shift_gradients=batch_parameter_shift_gradients.copy(),
        leaf_shapes=leaf_shapes,
        max_abs_gradient_error=max_abs_gradient_error,
        max_abs_transform_error=max_abs_transform_error,
        max_abs_hessian_symmetry_error=max_abs_hessian_symmetry_error,
        tolerance=tolerance_value,
        passed=passed,
        native_framework_autodiff=True,
        host_callback=False,
        jit_value_and_grad=True,
        vmap_value_and_grad=True,
        transform_names=(
            "grad",
            "value_and_grad",
            "jacfwd",
            "jacrev",
            "hessian",
            "jvp",
            "vjp",
            "vmap",
            "jit",
        ),
    )


def jax_phase_qnode_sharding_transform_audit(
    circuit: PhaseQNodeCircuit,
    params_batch: ArrayLike,
    *,
    tolerance: float = 1e-6,
    _jax_loader: JAXLoader = _load_jax,
) -> PhaseJAXPhaseQNodeShardingTransformResult:
    """Audit native JAX PMAP lowering for registered local Phase-QNode batches.

    The audit maps one deterministic statevector value-and-gradient row per
    local JAX device via ``jax.pmap`` and compares every row against the SCPN
    gate-aware parameter-shift reference. It is local sharding evidence only:
    no host callbacks, no finite-shot lowering, no provider execution, no
    hardware submission, and no wall-clock performance promotion.
    """
    jax_module, jnp = _jax_loader()
    _enable_jax_x64(jax_module)
    _require_jax_pmap_support(jax_module)
    tolerance_value = _as_non_negative_tolerance(tolerance)
    local_device_count = int(jax_module.local_device_count())
    if local_device_count <= 0:
        raise RuntimeError("JAX local_device_count must be positive for PMAP lowering")
    parameter_batch = np.asarray(params_batch, dtype=float)
    if parameter_batch.ndim != 2:
        raise ValueError("params_batch must be a two-dimensional array")
    if parameter_batch.shape[0] != local_device_count:
        raise ValueError(
            "params_batch must contain exactly one row per local JAX device, "
            f"got {parameter_batch.shape[0]} rows for {local_device_count} devices"
        )
    parameter_batch = _as_parameter_batch(
        "params_batch",
        parameter_batch,
        width=parameter_batch.shape[1],
    )
    for row in parameter_batch:
        report = phase_qnode_support_report(circuit, row)
        if not report.supported:
            raise PhaseQNodeSupportError(report)

    def value_function(raw_params: object) -> object:
        value, _state = _jax_phase_qnode_value_and_state(jnp, circuit, raw_params)
        return value

    value_and_grad_fn = jax_module.value_and_grad(value_function)
    values_obj, gradients_obj = jax_module.pmap(value_and_grad_fn)(jnp.asarray(parameter_batch))
    values = _as_parameter_vector(
        "JAX pmap Phase-QNode values",
        values_obj,
        width=local_device_count,
    )
    gradients = _as_parameter_batch(
        "JAX pmap Phase-QNode gradients",
        gradients_obj,
        width=parameter_batch.shape[1],
    )
    references = tuple(
        parameter_shift_phase_qnode_gradient(circuit, row) for row in parameter_batch
    )
    parameter_shift_values = np.asarray([result.value for result in references], dtype=np.float64)
    parameter_shift_gradients = np.vstack([result.gradient for result in references]).astype(
        np.float64,
        copy=False,
    )
    max_abs_value_error = float(np.max(np.abs(values - parameter_shift_values), initial=0.0))
    max_abs_gradient_error = float(
        np.max(np.abs(gradients - parameter_shift_gradients), initial=0.0)
    )
    passed = bool(
        max_abs_value_error <= tolerance_value and max_abs_gradient_error <= tolerance_value
    )
    sharding_mode = "single_device_pmap_smoke" if local_device_count == 1 else "multi_device_pmap"
    return PhaseJAXPhaseQNodeShardingTransformResult(
        values=values,
        gradients=gradients,
        parameter_shift_values=parameter_shift_values,
        parameter_shift_gradients=parameter_shift_gradients,
        batch_params=parameter_batch.copy(),
        batch_size=int(parameter_batch.shape[0]),
        local_device_count=local_device_count,
        device_descriptions=_jax_local_devices(jax_module, local_device_count),
        sharding_mode=sharding_mode,
        max_abs_value_error=max_abs_value_error,
        max_abs_gradient_error=max_abs_gradient_error,
        tolerance=tolerance_value,
        passed=passed,
        native_framework_autodiff=True,
        host_callback=False,
        pmapped=True,
    )


def jax_phase_qnode_aot_export_audit(
    circuit: PhaseQNodeCircuit,
    params: ArrayLike,
    *,
    tolerance: float = 1e-6,
    _jax_loader: JAXLoader = _load_jax,
) -> PhaseJAXPhaseQNodeAOTExportResult:
    """Audit JAX AOT lowering and export metadata for a registered Phase-QNode.

    The audit stages the deterministic local statevector value route through
    ``jax.jit(...).lower(...)``, records StableHLO/compiler metadata, exports
    the same jitted route through ``jax.export.export(...)``, serializes and
    deserializes it, and compares the compiled/exported values against the
    canonical SCPN parameter-shift value. The result is diagnostic evidence
    only: it does not promote persistent cross-platform execution, VJP export,
    provider callbacks, hardware execution, or performance claims.
    """
    jax_module, jnp = _jax_loader()
    _enable_jax_x64(jax_module)
    export_module = _require_jax_phase_qnode_aot_export_support(jax_module)
    tolerance_value = _as_non_negative_tolerance(tolerance)
    parameter_values = _as_parameter_vector("params", params)
    report = phase_qnode_support_report(circuit, parameter_values)
    if not report.supported:
        raise PhaseQNodeSupportError(report)
    parameter_shift = parameter_shift_phase_qnode_gradient(circuit, parameter_values)

    def value_function(raw_params: object) -> object:
        value, _state = _jax_phase_qnode_value_and_state(jnp, circuit, raw_params)
        return value

    parameter_shape = tuple(int(axis) for axis in parameter_values.shape)
    parameter_dtype = np.dtype(parameter_values.dtype)
    shape_dtype = jax_module.ShapeDtypeStruct(parameter_shape, parameter_dtype)
    jitted_value = jax_module.jit(value_function)
    lower = getattr(jitted_value, "lower", None)
    if not callable(lower):
        raise RuntimeError("JAX AOT lowering requires jitted functions with lower(...)")
    lowered = lower(shape_dtype)
    as_text = getattr(lowered, "as_text", None)
    if not callable(as_text):
        raise RuntimeError("JAX AOT lowering requires lowered.as_text() metadata")
    lowered_text = str(as_text())
    compiler_ir = getattr(lowered, "compiler_ir", None)
    if not callable(compiler_ir):
        raise RuntimeError("JAX AOT lowering requires lowered.compiler_ir(...) metadata")
    _stablehlo_ir = str(compiler_ir(dialect="stablehlo"))
    compile_lowered = getattr(lowered, "compile", None)
    if not callable(compile_lowered):
        raise RuntimeError("JAX AOT lowering requires lowered.compile()")
    compiled_executable = compile_lowered()
    compiled_value = _as_scalar(
        "JAX compiled Phase-QNode value",
        compiled_executable(jnp.asarray(parameter_values)),
    )

    exported = export_module.export(jax_module.jit(value_function))(shape_dtype)
    mlir_module = str(exported.mlir_module())
    serialized_blob = exported.serialize()
    rehydrated = export_module.deserialize(serialized_blob)
    deserialized_value = _as_scalar(
        "JAX exported Phase-QNode value",
        rehydrated.call(jnp.asarray(parameter_values)),
    )
    export_platforms = tuple(str(platform) for platform in getattr(exported, "platforms", ()))
    if not export_platforms:
        raise RuntimeError("JAX export did not report lowering platforms")
    calling_convention_version = int(exported.calling_convention_version)
    minimum_supported = _jax_export_version(
        export_module,
        "minimum_supported_calling_convention_version",
    )
    maximum_supported = _jax_export_version(
        export_module,
        "maximum_supported_calling_convention_version",
    )
    disabled_safety_checks = tuple(
        str(check) for check in getattr(exported, "disabled_safety_checks", ())
    )
    value_errors = (
        abs(compiled_value - parameter_shift.value),
        abs(deserialized_value - parameter_shift.value),
    )
    max_abs_value_error = max(float(error) for error in value_errors)
    passed = bool(
        max_abs_value_error <= tolerance_value
        and not disabled_safety_checks
        and minimum_supported <= calling_convention_version <= maximum_supported
    )
    return PhaseJAXPhaseQNodeAOTExportResult(
        value=compiled_value,
        compiled_value=compiled_value,
        deserialized_value=deserialized_value,
        parameter_shift_value=parameter_shift.value,
        max_abs_value_error=max_abs_value_error,
        tolerance=tolerance_value,
        passed=passed,
        lowered=True,
        compiled=True,
        exported=True,
        serialized=bool(serialized_blob),
        deserialized_call=True,
        host_callback=False,
        parameter_shape=parameter_shape,
        parameter_dtype=str(parameter_dtype),
        compiler_ir_dialects=("stablehlo",),
        lowered_text_bytes=len(lowered_text.encode("utf-8")),
        mlir_module_bytes=len(mlir_module.encode("utf-8")),
        serialized_bytes=len(serialized_blob),
        export_platforms=export_platforms,
        calling_convention_version=calling_convention_version,
        minimum_supported_calling_convention_version=minimum_supported,
        maximum_supported_calling_convention_version=maximum_supported,
        disabled_safety_checks=disabled_safety_checks,
        uses_global_constants=bool(getattr(exported, "uses_global_constants", False)),
    )


def _enable_jax_x64(jax_module: Any) -> None:
    config = getattr(jax_module, "config", None)
    update = getattr(config, "update", None)
    if callable(update):
        update("jax_enable_x64", True)


def _jax_phase_qnode_value_and_state(
    jnp: Any,
    circuit: PhaseQNodeCircuit,
    params: object,
) -> tuple[object, object]:
    parameter_tensor = jnp.asarray(params)
    state = jnp.zeros((2**circuit.n_qubits,), dtype=jnp.complex128)
    state = state.at[0].set(1.0 + 0.0j)
    operations = cast(tuple[PhaseQNodeOperation, ...], circuit.operations)
    for operation in operations:
        matrix = _jax_gate_matrix(
            jnp, operation.gate, _jax_operation_theta(operation, parameter_tensor)
        )
        state = _jax_apply_gate_matrix(jnp, state, circuit.n_qubits, operation.qubits, matrix)
    return _jax_expectation_value(jnp, state, circuit.n_qubits, circuit.observable), state


def _jax_operation_theta(operation: PhaseQNodeOperation, parameter_tensor: Any) -> Any:
    if operation.parameter_index is None:
        return 0.0
    return parameter_tensor[operation.parameter_index]


def _jax_gate_matrix(jnp: Any, gate: str, theta: Any) -> Any:
    complex_dtype = jnp.complex128
    one = jnp.asarray(1.0, dtype=complex_dtype)
    zero = jnp.asarray(0.0, dtype=complex_dtype)
    imag = jnp.asarray(1.0j, dtype=complex_dtype)
    identity = jnp.eye(2, dtype=complex_dtype)
    x_matrix = jnp.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=complex_dtype)
    y_matrix = jnp.asarray([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex_dtype)
    z_matrix = jnp.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=complex_dtype)
    h_matrix = (1.0 / jnp.sqrt(jnp.asarray(2.0))) * jnp.asarray(
        [[1.0, 1.0], [1.0, -1.0]],
        dtype=complex_dtype,
    )
    s_matrix = jnp.asarray([[1.0, 0.0], [0.0, 1.0j]], dtype=complex_dtype)
    t_matrix = jnp.asarray(
        [[1.0, 0.0], [0.0, np.exp(1.0j * np.pi / 4.0)]],
        dtype=complex_dtype,
    )
    sx_matrix = 0.5 * jnp.asarray(
        [[1.0 + 1.0j, 1.0 - 1.0j], [1.0 - 1.0j, 1.0 + 1.0j]],
        dtype=complex_dtype,
    )
    if gate == "h":
        return h_matrix
    if gate == "x":
        return x_matrix
    if gate == "y":
        return y_matrix
    if gate == "z":
        return z_matrix
    if gate == "s":
        return s_matrix
    if gate == "t":
        return t_matrix
    if gate == "sx":
        return sx_matrix
    if gate == "rx":
        return jnp.cos(theta / 2.0) * identity - imag * jnp.sin(theta / 2.0) * x_matrix
    if gate == "ry":
        return jnp.cos(theta / 2.0) * identity - imag * jnp.sin(theta / 2.0) * y_matrix
    if gate == "rz":
        return jnp.cos(theta / 2.0) * identity - imag * jnp.sin(theta / 2.0) * z_matrix
    if gate == "phase":
        return jnp.asarray([[one, zero], [zero, jnp.exp(imag * theta)]], dtype=complex_dtype)
    if gate == "cnot":
        return jnp.asarray(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex_dtype
        )
    if gate == "cz":
        return jnp.diag(jnp.asarray([1.0, 1.0, 1.0, -1.0], dtype=complex_dtype))
    if gate == "cy":
        return jnp.asarray(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1.0j], [0, 0, 1.0j, 0]],
            dtype=complex_dtype,
        )
    if gate == "swap":
        return jnp.asarray(
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex_dtype
        )
    if gate == "ch":
        return _jax_controlled(jnp, h_matrix)
    if gate == "cs":
        return _jax_controlled(jnp, s_matrix)
    if gate == "ct":
        return _jax_controlled(jnp, t_matrix)
    if gate == "ccnot":
        return _jax_ccnot_matrix(jnp)
    if gate == "ccz":
        return jnp.diag(
            jnp.asarray([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0], dtype=complex_dtype)
        )
    if gate == "cswap":
        return _jax_cswap_matrix(jnp)
    if gate == "crx":
        return _jax_controlled(jnp, _jax_gate_matrix(jnp, "rx", theta))
    if gate == "cry":
        return _jax_controlled(jnp, _jax_gate_matrix(jnp, "ry", theta))
    if gate == "crz":
        return _jax_controlled(jnp, _jax_gate_matrix(jnp, "rz", theta))
    if gate == "rxx":
        return jnp.cos(theta / 2.0) * jnp.eye(4, dtype=complex_dtype) - imag * jnp.sin(
            theta / 2.0
        ) * jnp.kron(x_matrix, x_matrix)
    if gate == "ryy":
        return jnp.cos(theta / 2.0) * jnp.eye(4, dtype=complex_dtype) - imag * jnp.sin(
            theta / 2.0
        ) * jnp.kron(y_matrix, y_matrix)
    rzz_matrix = jnp.cos(theta / 2.0) * jnp.eye(4, dtype=complex_dtype) - imag * jnp.sin(
        theta / 2.0
    ) * jnp.kron(z_matrix, z_matrix)
    return {"rzz": rzz_matrix}[gate]


def _jax_controlled(jnp: Any, target: Any) -> Any:
    matrix = jnp.eye(4, dtype=jnp.complex128)
    return matrix.at[2:4, 2:4].set(target)


def _jax_ccnot_matrix(jnp: Any) -> Any:
    matrix = jnp.eye(8, dtype=jnp.complex128)
    matrix = matrix.at[6, 6].set(0.0)
    matrix = matrix.at[7, 7].set(0.0)
    matrix = matrix.at[6, 7].set(1.0)
    return matrix.at[7, 6].set(1.0)


def _jax_cswap_matrix(jnp: Any) -> Any:
    matrix = jnp.eye(8, dtype=jnp.complex128)
    matrix = matrix.at[5, 5].set(0.0)
    matrix = matrix.at[6, 6].set(0.0)
    matrix = matrix.at[5, 6].set(1.0)
    return matrix.at[6, 5].set(1.0)


def _jax_apply_gate_matrix(
    jnp: Any,
    state: Any,
    n_qubits: int,
    qubits: tuple[int, ...],
    matrix: Any,
) -> Any:
    width = len(qubits)
    axes = list(qubits) + [axis for axis in range(n_qubits) if axis not in qubits]
    inverse = np.argsort(axes)
    tensor = jnp.transpose(jnp.reshape(state, (2,) * n_qubits), axes)
    front = jnp.reshape(tensor, (2**width, -1))
    updated = jnp.reshape(matrix @ front, (2,) * n_qubits)
    return jnp.reshape(jnp.transpose(updated, tuple(int(axis) for axis in inverse)), (-1,))


def _jax_expectation_value(
    jnp: Any,
    state: Any,
    n_qubits: int,
    observable: object,
) -> Any:
    if isinstance(observable, DenseHermitianObservable):
        matrix = jnp.asarray(observable.matrix, dtype=jnp.complex128)
        return jnp.real(jnp.vdot(state, matrix @ state))
    if isinstance(observable, PauliCovarianceObservable):
        symmetrized = _jax_symmetrized_product_expectation(
            jnp, state, n_qubits, observable.left, observable.right
        )
        left_mean = _jax_term_expectation(jnp, state, n_qubits, observable.left)
        right_mean = _jax_term_expectation(jnp, state, n_qubits, observable.right)
        return symmetrized - left_mean * right_mean
    if isinstance(observable, SparsePauliHamiltonian):
        total = jnp.asarray(0.0)
        for term in observable.terms:
            total = total + _jax_term_expectation(jnp, state, n_qubits, term)
        return total
    return _jax_term_expectation(jnp, state, n_qubits, cast(PauliTerm, observable))


def _jax_term_expectation(jnp: Any, state: Any, n_qubits: int, term: PauliTerm) -> Any:
    transformed = _jax_apply_term_operator(jnp, state, n_qubits, term)
    return term.coefficient * jnp.real(jnp.vdot(state, transformed))


def _jax_symmetrized_product_expectation(
    jnp: Any,
    state: Any,
    n_qubits: int,
    left: PauliTerm,
    right: PauliTerm,
) -> Any:
    left_right = _jax_term_product_expectation(jnp, state, n_qubits, left, right)
    right_left = _jax_term_product_expectation(jnp, state, n_qubits, right, left)
    return jnp.real(0.5 * (left_right + right_left))


def _jax_term_product_expectation(
    jnp: Any,
    state: Any,
    n_qubits: int,
    left: PauliTerm,
    right: PauliTerm,
) -> Any:
    transformed = _jax_apply_term_operator(jnp, state, n_qubits, right)
    transformed = _jax_apply_term_operator(jnp, transformed, n_qubits, left)
    return left.coefficient * right.coefficient * jnp.vdot(state, transformed)


def _jax_apply_term_operator(jnp: Any, state: Any, n_qubits: int, term: PauliTerm) -> Any:
    transformed = state
    for qubit, label in term.factors:
        transformed = _jax_apply_gate_matrix(
            jnp,
            transformed,
            n_qubits,
            (qubit,),
            _jax_pauli_matrix(jnp, label),
        )
    return transformed


def _jax_pauli_matrix(jnp: Any, label: str) -> Any:
    if label == "x":
        return jnp.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.complex128)
    if label == "y":
        return jnp.asarray([[0.0, -1.0j], [1.0j, 0.0]], dtype=jnp.complex128)
    z_matrix = jnp.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=jnp.complex128)
    return {"z": z_matrix}[label]


__all__ = [
    "jax_phase_qnode_aot_export_audit",
    "jax_phase_qnode_native_transform_audit",
    "jax_phase_qnode_pytree_transform_audit",
    "jax_phase_qnode_sharding_transform_audit",
    "jax_phase_qnode_value_and_grad",
]
