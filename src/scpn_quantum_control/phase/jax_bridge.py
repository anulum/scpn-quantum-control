# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Phase JAX Bridge
"""Optional JAX execution and compatibility facade for phase gradients.

Immutable JAX result records live in :mod:`.jax_bridge_contracts`, and bounded
gradient implementations live in :mod:`.jax_gradients`. This module retains
compatibility wrappers, registered-QNode lowering, transforms, and maturity audits.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..differentiable import (
    Parameter,
    ParameterShiftRule,
)
from .jax_bridge_contracts import (
    PhaseJAXCloudValidationRunSpec,
    PhaseJAXCustomVJPQNNGradientResult,
    PhaseJAXGradientAgreementResult,
    PhaseJAXJITCompatibilityResult,
    PhaseJAXMaturityAuditResult,
    PhaseJAXNativeQNNGradientResult,
    PhaseJAXNestedTransformAlgebraResult,
    PhaseJAXNestedTransformRoute,
    PhaseJAXParameterShiftResult,
    PhaseJAXPhaseQNodeAOTExportResult,
    PhaseJAXPhaseQNodeLoweringMatrixResult,
    PhaseJAXPhaseQNodeLoweringRoute,
    PhaseJAXPhaseQNodeNativeTransformResult,
    PhaseJAXPhaseQNodePyTreeTransformResult,
    PhaseJAXPhaseQNodeShardingTransformResult,
    PhaseJAXPhaseQNodeStatevectorResult,
    PhaseJAXPyTreeCompatibilityResult,
    PhaseJAXShardingCompatibilityResult,
    PhaseJAXVMAPCompatibilityResult,
)
from .jax_gradients import (
    _as_feature_matrix,
    _as_label_vector,
    _as_non_negative_tolerance,
    _as_parameter_vector,
    _as_scalar,
    _custom_vjp_function,
    _jax_bounded_qnn_gradient,
    _jax_bounded_qnn_loss,
    _load_jax,
    _require_jax_custom_vjp_support,
)
from .jax_gradients import (
    check_jax_parameter_shift_agreement as _check_jax_parameter_shift_agreement,
)
from .jax_gradients import (
    jax_custom_vjp_qnn_value_and_grad as _jax_custom_vjp_qnn_value_and_grad,
)
from .jax_gradients import (
    jax_native_qnn_value_and_grad as _jax_native_qnn_value_and_grad,
)
from .jax_gradients import (
    jax_parameter_shift_value_and_grad as _jax_parameter_shift_value_and_grad,
)
from .qnn_training import (
    parameter_shift_qnn_classifier_gradient,
    parameter_shift_qnn_classifier_loss,
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
ScalarObjective = Callable[[FloatArray], float]
GradientCallable = Callable[[FloatArray], ArrayLike]
JAXCallable = Callable[[object], object]


def is_phase_jax_available() -> bool:
    """Return whether the optional phase JAX adapter can import JAX."""
    try:
        _load_jax()
    except (AttributeError, ImportError, RuntimeError, ValueError):
        return False
    return True


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


def _require_jax_vmap_support(jax_module: Any) -> None:
    vmap = getattr(jax_module, "vmap", None)
    if not callable(vmap):
        raise RuntimeError("JAX vmap is required for bounded-QNN parameter-batch VMAP")
    value_and_grad = getattr(jax_module, "value_and_grad", None)
    if not callable(value_and_grad):
        raise RuntimeError("JAX value_and_grad is required for bounded-QNN parameter-batch VMAP")


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


def _require_jax_nested_transform_support(jax_module: Any) -> None:
    _require_jax_vmap_support(jax_module)
    _require_jax_pytree_support(jax_module)
    jit = getattr(jax_module, "jit", None)
    if not callable(jit):
        raise RuntimeError("JAX JIT is required for bounded nested-transform algebra")


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
    *,
    width: int,
) -> tuple[FloatArray, object, tuple[tuple[int, ...], ...], tuple[int, ...]]:
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
    if vector.shape != (width,):
        raise ValueError(f"{name} must flatten to shape ({width},), got {vector.shape}")
    return vector, treedef, tuple(shapes), tuple(sizes)


def _jax_flatten_pytree(jax_module: Any, jnp: Any, params_pytree: object) -> object:
    leaves, _treedef = jax_module.tree_util.tree_flatten(params_pytree)
    parts = [jnp.ravel(jnp.asarray(leaf)) for leaf in leaves]
    if not parts:
        raise ValueError("params_pytree must contain at least one numeric leaf")
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


def jax_parameter_shift_value_and_grad(
    objective: ScalarObjective,
    values: ArrayLike,
    *,
    jit: bool = False,
    parameters: Sequence[Parameter] | None = None,
    rule: ParameterShiftRule | None = None,
) -> PhaseJAXParameterShiftResult:
    """Return parameter-shift value and gradient through the optional JAX bridge.

    This adapter is an interop boundary, not a native quantum-kernel compiler.
    Without ``jit`` it imports JAX, accepts JAX-like inputs, and executes the
    repository's deterministic parameter-shift rule on the host. With ``jit`` it
    wraps that host execution in ``jax.pure_callback`` and ``jax.jit`` so JAX
    workflows can stage the call explicitly while provenance still reports
    ``host_callback=True``.
    """
    return _jax_parameter_shift_value_and_grad(
        objective,
        values,
        jit=jit,
        parameters=parameters,
        rule=rule,
        _jax_loader=_load_jax,
    )


def check_jax_parameter_shift_agreement(
    objective: ScalarObjective,
    jax_gradient: GradientCallable,
    values: ArrayLike,
    *,
    tolerance: float = 1e-6,
    parameters: Sequence[Parameter] | None = None,
    rule: ParameterShiftRule | None = None,
) -> PhaseJAXGradientAgreementResult:
    """Compare SCPN parameter-shift gradients with a JAX-derived gradient callable.

    ``jax_gradient`` is caller-supplied so the bridge can verify agreement with
    ``jax.grad`` or equivalent JAX code without claiming automatic conversion of
    every SCPN objective into a native JAX quantum kernel.
    """
    return _check_jax_parameter_shift_agreement(
        objective,
        jax_gradient,
        values,
        tolerance=tolerance,
        parameters=parameters,
        rule=rule,
        _jax_loader=_load_jax,
    )


def jax_native_qnn_value_and_grad(
    features: ArrayLike,
    labels: ArrayLike,
    params: ArrayLike,
    *,
    tolerance: float = 1e-6,
    jit: bool = False,
) -> PhaseJAXNativeQNNGradientResult:
    """Differentiate the bounded phase-QNN loss with native JAX autodiff.

    This route is intentionally narrower than arbitrary simulator autodiff. It
    expresses the repository's bounded phase-QNN classifier loss directly in
    JAX tensor operations, obtains a native ``value_and_grad`` result, and
    records agreement against the existing multi-frequency parameter-shift
    gradient. It does not use ``pure_callback`` and does not claim conversion of
    arbitrary quantum programs into JAX kernels.
    """
    return _jax_native_qnn_value_and_grad(
        features,
        labels,
        params,
        tolerance=tolerance,
        jit=jit,
        _jax_loader=_load_jax,
    )


def jax_custom_vjp_qnn_value_and_grad(
    features: ArrayLike,
    labels: ArrayLike,
    params: ArrayLike,
    *,
    tolerance: float = 1e-6,
    jit: bool = False,
) -> PhaseJAXCustomVJPQNNGradientResult:
    """Differentiate the bounded phase-QNN loss through a JAX custom VJP.

    The primal is the same bounded phase-QNN MSE loss used by
    ``jax_native_qnn_value_and_grad``. The VJP rule is registered explicitly
    and returns the mathematically equivalent bounded-QNN derivative, then the
    result is checked against the repository's multi-frequency parameter-shift
    reference. This route is still intentionally narrow: it does not expose
    arbitrary simulator autodiff, provider callbacks, or hardware execution.
    """
    return _jax_custom_vjp_qnn_value_and_grad(
        features,
        labels,
        params,
        tolerance=tolerance,
        jit=jit,
        _jax_loader=_load_jax,
    )


def jax_phase_qnode_value_and_grad(
    circuit: PhaseQNodeCircuit,
    params: ArrayLike,
    *,
    tolerance: float = 1e-6,
    jit: bool = False,
) -> PhaseJAXPhaseQNodeStatevectorResult:
    """Lower a registered deterministic Phase-QNode statevector route into JAX.

    The accepted surface is the local pure-state ``PhaseQNodeCircuit`` gate and
    observable family. It deliberately excludes finite-shot sampling, provider
    callbacks, hardware execution, density/noise channels, and dynamic circuits.
    """
    jax_module, jnp = _load_jax()
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
) -> PhaseJAXPhaseQNodeNativeTransformResult:
    """Audit native JAX transforms for a registered local Phase-QNode.

    The executable route is the same deterministic statevector lowering used by
    :func:`jax_phase_qnode_value_and_grad`, but it is exercised through JAX
    ``grad``, ``value_and_grad``, ``jacfwd``, ``jacrev``, ``hessian``, ``jvp``,
    ``vjp``, ``vmap``, and ``jit``. It deliberately does not use host callbacks
    and does not promote finite-shot, provider, hardware, density/noise, or
    dynamic-circuit lowering.
    """
    jax_module, jnp = _load_jax()
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
    jax_module, jnp = _load_jax()
    _enable_jax_x64(jax_module)
    _require_jax_phase_qnode_pytree_transform_support(jax_module)
    _require_jax_pytree_support(jax_module)
    tolerance_value = _as_non_negative_tolerance(tolerance)
    raw_parameter_vector = _jax_flatten_pytree(jax_module, jnp, params_pytree)
    parameter_values = _as_parameter_vector("params_pytree", raw_parameter_vector)
    (
        parameter_values,
        treedef,
        leaf_shapes,
        leaf_sizes,
    ) = _as_pytree_parameter_vector(
        jax_module,
        "params_pytree",
        params_pytree,
        width=parameter_values.size,
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
) -> PhaseJAXPhaseQNodeShardingTransformResult:
    """Audit native JAX PMAP lowering for registered local Phase-QNode batches.

    The audit maps one deterministic statevector value-and-gradient row per
    local JAX device via ``jax.pmap`` and compares every row against the SCPN
    gate-aware parameter-shift reference. It is local sharding evidence only:
    no host callbacks, no finite-shot lowering, no provider execution, no
    hardware submission, and no wall-clock performance promotion.
    """
    jax_module, jnp = _load_jax()
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
    jax_module, jnp = _load_jax()
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
    if gate == "rzz":
        return jnp.cos(theta / 2.0) * jnp.eye(4, dtype=complex_dtype) - imag * jnp.sin(
            theta / 2.0
        ) * jnp.kron(z_matrix, z_matrix)
    raise ValueError(f"unsupported JAX Phase-QNode gate: {gate}")


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
    if isinstance(observable, PauliTerm):
        return _jax_term_expectation(jnp, state, n_qubits, observable)
    raise ValueError(f"unsupported JAX Phase-QNode observable: {observable}")


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
    if label == "z":
        return jnp.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=jnp.complex128)
    raise ValueError(f"unsupported JAX Pauli label: {label}")


def run_jax_jit_compatibility_audit(
    *,
    features: ArrayLike,
    labels: ArrayLike,
    params: ArrayLike,
    tolerance: float = 1e-6,
) -> PhaseJAXJITCompatibilityResult:
    """Audit bounded JAX JIT support without promoting host callbacks.

    The audit exercises three JIT-facing routes:

    - bounded native QNN ``value_and_grad`` with ``host_callback=False``;
    - bounded QNN ``custom_vjp`` with ``host_callback=False``;
    - parameter-shift interop under ``jax.pure_callback``, recorded as a
      host-callback route and therefore excluded from native-JIT promotion.
    """
    tolerance_value = _as_non_negative_tolerance(tolerance)
    feature_matrix = _as_feature_matrix(features)
    label_vector = _as_label_vector(labels, n_samples=feature_matrix.shape[0])
    parameter_values = _as_parameter_vector(
        "params",
        params,
        width=feature_matrix.shape[1],
    )

    def objective(values: FloatArray) -> float:
        return parameter_shift_qnn_classifier_loss(feature_matrix, label_vector, values)

    native = jax_native_qnn_value_and_grad(
        feature_matrix,
        label_vector,
        parameter_values,
        tolerance=tolerance_value,
        jit=True,
    )
    custom = jax_custom_vjp_qnn_value_and_grad(
        feature_matrix,
        label_vector,
        parameter_values,
        tolerance=tolerance_value,
        jit=True,
    )
    parameter_shift = jax_parameter_shift_value_and_grad(
        objective,
        parameter_values,
        jit=True,
    )

    max_abs_error = max(native.max_abs_error, custom.max_abs_error)
    unsupported_native_routes: list[str] = []
    if parameter_shift.host_callback:
        unsupported_native_routes.append("parameter_shift_host_callback")
    passed = (
        native.passed
        and custom.passed
        and native.jitted
        and custom.jitted
        and not native.host_callback
        and not custom.host_callback
        and max_abs_error <= tolerance_value
    )
    return PhaseJAXJITCompatibilityResult(
        passed=passed,
        native_qnn_jitted=native.jitted,
        native_qnn_host_callback=native.host_callback,
        custom_vjp_qnn_jitted=custom.jitted,
        custom_vjp_qnn_host_callback=custom.host_callback,
        custom_vjp_registered=custom.custom_vjp,
        parameter_shift_jitted=parameter_shift.jitted,
        parameter_shift_host_callback=parameter_shift.host_callback,
        max_abs_error=max_abs_error,
        tolerance=tolerance_value,
        unsupported_native_routes=tuple(unsupported_native_routes),
    )


def run_jax_vmap_compatibility_audit(
    *,
    features: ArrayLike,
    labels: ArrayLike,
    params_batch: ArrayLike,
    tolerance: float = 1e-6,
) -> PhaseJAXVMAPCompatibilityResult:
    """Audit bounded JAX VMAP support for parameter batches.

    The audit vectorises the bounded phase-QNN native and custom-VJP loss routes
    over a two-dimensional parameter batch. SCPN parameter-shift results are
    used as host-side references only and are reported as such rather than
    promoted as native VMAP.
    """
    jax_module, jnp = _load_jax()
    _require_jax_vmap_support(jax_module)
    _require_jax_custom_vjp_support(jax_module)
    tolerance_value = _as_non_negative_tolerance(tolerance)
    feature_matrix = _as_feature_matrix(features)
    label_vector = _as_label_vector(labels, n_samples=feature_matrix.shape[0])
    parameter_batch = _as_parameter_batch(
        "params_batch",
        params_batch,
        width=feature_matrix.shape[1],
    )
    feature_tensor = jnp.asarray(feature_matrix)
    label_tensor = jnp.asarray(label_vector)

    def native_loss_fn(raw_params: object) -> object:
        return _jax_bounded_qnn_loss(jnp, feature_tensor, label_tensor, jnp.asarray(raw_params))

    def custom_loss_fn(raw_params: object) -> object:
        return _jax_bounded_qnn_loss(jnp, feature_tensor, label_tensor, jnp.asarray(raw_params))

    custom_loss: Any = _custom_vjp_function(jax_module, custom_loss_fn)

    def custom_loss_fwd(raw_params: object) -> tuple[object, object]:
        parameter_tensor = jnp.asarray(raw_params)
        return (
            _jax_bounded_qnn_loss(jnp, feature_tensor, label_tensor, parameter_tensor),
            parameter_tensor,
        )

    def custom_loss_bwd(parameter_tensor: object, cotangent: object) -> tuple[object]:
        gradient = _jax_bounded_qnn_gradient(
            jnp,
            feature_tensor,
            label_tensor,
            jnp.asarray(parameter_tensor),
        )
        return (jnp.asarray(cotangent) * gradient,)

    custom_loss.defvjp(custom_loss_fwd, custom_loss_bwd)

    native_losses_obj, native_gradients_obj = jax_module.vmap(
        jax_module.value_and_grad(native_loss_fn)
    )(jnp.asarray(parameter_batch))
    custom_losses_obj, custom_gradients_obj = jax_module.vmap(
        jax_module.value_and_grad(custom_loss)
    )(jnp.asarray(parameter_batch))

    native_losses = np.asarray(native_losses_obj, dtype=float)
    custom_losses = np.asarray(custom_losses_obj, dtype=float)
    native_gradients = np.asarray(native_gradients_obj, dtype=float)
    custom_gradients = np.asarray(custom_gradients_obj, dtype=float)
    if native_losses.shape != (parameter_batch.shape[0],):
        raise RuntimeError("JAX native VMAP loss batch has an unexpected shape")
    if custom_losses.shape != (parameter_batch.shape[0],):
        raise RuntimeError("JAX custom-VJP VMAP loss batch has an unexpected shape")
    if native_gradients.shape != parameter_batch.shape:
        raise RuntimeError("JAX native VMAP gradient batch has an unexpected shape")
    if custom_gradients.shape != parameter_batch.shape:
        raise RuntimeError("JAX custom-VJP VMAP gradient batch has an unexpected shape")

    reference_losses = np.asarray(
        [
            parameter_shift_qnn_classifier_loss(feature_matrix, label_vector, row)
            for row in parameter_batch
        ],
        dtype=np.float64,
    )
    reference_gradients = np.vstack(
        [
            parameter_shift_qnn_classifier_gradient(feature_matrix, label_vector, row)
            for row in parameter_batch
        ]
    ).astype(np.float64, copy=False)
    loss_error = max(
        float(np.max(np.abs(native_losses - reference_losses))),
        float(np.max(np.abs(custom_losses - reference_losses))),
    )
    gradient_error = max(
        float(np.max(np.abs(native_gradients - reference_gradients))),
        float(np.max(np.abs(custom_gradients - reference_gradients))),
    )
    max_abs_error = max(loss_error, gradient_error)
    unsupported_native_routes = ("parameter_shift_host_loop_reference",)
    passed = max_abs_error <= tolerance_value
    return PhaseJAXVMAPCompatibilityResult(
        passed=passed,
        batch_size=int(parameter_batch.shape[0]),
        native_qnn_vmapped=True,
        native_qnn_host_callback=False,
        custom_vjp_qnn_vmapped=True,
        custom_vjp_qnn_host_callback=False,
        custom_vjp_registered=True,
        native_losses=native_losses.astype(np.float64, copy=True),
        native_gradients=native_gradients.astype(np.float64, copy=True),
        custom_vjp_losses=custom_losses.astype(np.float64, copy=True),
        custom_vjp_gradients=custom_gradients.astype(np.float64, copy=True),
        parameter_shift_losses=reference_losses.copy(),
        parameter_shift_gradients=reference_gradients.copy(),
        max_abs_error=max_abs_error,
        tolerance=tolerance_value,
        unsupported_native_routes=unsupported_native_routes,
    )


def run_jax_sharding_compatibility_audit(
    *,
    features: ArrayLike,
    labels: ArrayLike,
    params_batch: ArrayLike,
    tolerance: float = 1e-6,
) -> PhaseJAXShardingCompatibilityResult:
    """Audit bounded JAX PMAP/sharding support for local-device batches.

    The audit maps one parameter row per local JAX device with ``jax.pmap``.
    It promotes only the bounded native and custom-VJP phase-QNN loss routes.
    SCPN parameter-shift references stay host-side validation rows and are not
    reported as sharded/native execution.
    """
    jax_module, jnp = _load_jax()
    _require_jax_pmap_support(jax_module)
    _require_jax_custom_vjp_support(jax_module)
    tolerance_value = _as_non_negative_tolerance(tolerance)
    feature_matrix = _as_feature_matrix(features)
    label_vector = _as_label_vector(labels, n_samples=feature_matrix.shape[0])
    local_device_count = int(jax_module.local_device_count())
    if local_device_count < 1:
        raise RuntimeError("JAX pmap requires at least one local device")
    parameter_batch = _as_parameter_batch(
        "params_batch",
        params_batch,
        width=feature_matrix.shape[1],
    )
    if parameter_batch.shape[0] != local_device_count:
        raise ValueError(
            "params_batch row count must match JAX local_device_count "
            f"({local_device_count}), got {parameter_batch.shape[0]}"
        )
    feature_tensor = jnp.asarray(feature_matrix)
    label_tensor = jnp.asarray(label_vector)

    def native_loss_fn(raw_params: object) -> object:
        return _jax_bounded_qnn_loss(jnp, feature_tensor, label_tensor, jnp.asarray(raw_params))

    def custom_loss_fn(raw_params: object) -> object:
        return _jax_bounded_qnn_loss(jnp, feature_tensor, label_tensor, jnp.asarray(raw_params))

    custom_loss: Any = _custom_vjp_function(jax_module, custom_loss_fn)

    def custom_loss_fwd(raw_params: object) -> tuple[object, object]:
        parameter_tensor = jnp.asarray(raw_params)
        return (
            _jax_bounded_qnn_loss(jnp, feature_tensor, label_tensor, parameter_tensor),
            parameter_tensor,
        )

    def custom_loss_bwd(parameter_tensor: object, cotangent: object) -> tuple[object]:
        gradient = _jax_bounded_qnn_gradient(
            jnp,
            feature_tensor,
            label_tensor,
            jnp.asarray(parameter_tensor),
        )
        return (jnp.asarray(cotangent) * gradient,)

    custom_loss.defvjp(custom_loss_fwd, custom_loss_bwd)

    native_losses_obj, native_gradients_obj = jax_module.pmap(
        jax_module.value_and_grad(native_loss_fn)
    )(jnp.asarray(parameter_batch))
    custom_losses_obj, custom_gradients_obj = jax_module.pmap(
        jax_module.value_and_grad(custom_loss)
    )(jnp.asarray(parameter_batch))

    native_losses = np.asarray(native_losses_obj, dtype=float)
    custom_losses = np.asarray(custom_losses_obj, dtype=float)
    native_gradients = np.asarray(native_gradients_obj, dtype=float)
    custom_gradients = np.asarray(custom_gradients_obj, dtype=float)
    if native_losses.shape != (parameter_batch.shape[0],):
        raise RuntimeError("JAX native PMAP loss batch has an unexpected shape")
    if custom_losses.shape != (parameter_batch.shape[0],):
        raise RuntimeError("JAX custom-VJP PMAP loss batch has an unexpected shape")
    if native_gradients.shape != parameter_batch.shape:
        raise RuntimeError("JAX native PMAP gradient batch has an unexpected shape")
    if custom_gradients.shape != parameter_batch.shape:
        raise RuntimeError("JAX custom-VJP PMAP gradient batch has an unexpected shape")

    reference_losses = np.asarray(
        [
            parameter_shift_qnn_classifier_loss(feature_matrix, label_vector, row)
            for row in parameter_batch
        ],
        dtype=np.float64,
    )
    reference_gradients = np.vstack(
        [
            parameter_shift_qnn_classifier_gradient(feature_matrix, label_vector, row)
            for row in parameter_batch
        ]
    ).astype(np.float64, copy=False)
    loss_error = max(
        float(np.max(np.abs(native_losses - reference_losses))),
        float(np.max(np.abs(custom_losses - reference_losses))),
    )
    gradient_error = max(
        float(np.max(np.abs(native_gradients - reference_gradients))),
        float(np.max(np.abs(custom_gradients - reference_gradients))),
    )
    max_abs_error = max(loss_error, gradient_error)
    sharding_mode = "pmap_single_device" if local_device_count == 1 else "pmap_multi_device"
    unsupported_native_routes = ("parameter_shift_host_loop_reference",)
    passed = max_abs_error <= tolerance_value
    return PhaseJAXShardingCompatibilityResult(
        passed=passed,
        batch_size=int(parameter_batch.shape[0]),
        local_device_count=local_device_count,
        device_descriptions=_jax_local_devices(jax_module, local_device_count),
        sharding_mode=sharding_mode,
        native_qnn_pmapped=True,
        native_qnn_host_callback=False,
        custom_vjp_qnn_pmapped=True,
        custom_vjp_qnn_host_callback=False,
        custom_vjp_registered=True,
        native_losses=native_losses.astype(np.float64, copy=True),
        native_gradients=native_gradients.astype(np.float64, copy=True),
        custom_vjp_losses=custom_losses.astype(np.float64, copy=True),
        custom_vjp_gradients=custom_gradients.astype(np.float64, copy=True),
        parameter_shift_losses=reference_losses.copy(),
        parameter_shift_gradients=reference_gradients.copy(),
        max_abs_error=max_abs_error,
        tolerance=tolerance_value,
        unsupported_native_routes=unsupported_native_routes,
    )


def run_jax_pytree_compatibility_audit(
    *,
    features: ArrayLike,
    labels: ArrayLike,
    params_pytree: object,
    tolerance: float = 1e-6,
) -> PhaseJAXPyTreeCompatibilityResult:
    """Audit bounded JAX PyTree parameter support.

    The audit accepts a JAX PyTree of numeric parameter leaves, flattens it into
    the bounded phase-QNN parameter vector, and restores gradients into the same
    PyTree structure. It does not claim arbitrary simulator PyTree lowering.
    """
    jax_module, jnp = _load_jax()
    _require_jax_pytree_support(jax_module)
    _require_jax_custom_vjp_support(jax_module)
    tolerance_value = _as_non_negative_tolerance(tolerance)
    feature_matrix = _as_feature_matrix(features)
    label_vector = _as_label_vector(labels, n_samples=feature_matrix.shape[0])
    parameter_vector, treedef, leaf_shapes, leaf_sizes = _as_pytree_parameter_vector(
        jax_module,
        "params_pytree",
        params_pytree,
        width=feature_matrix.shape[1],
    )
    feature_tensor = jnp.asarray(feature_matrix)
    label_tensor = jnp.asarray(label_vector)

    def native_loss_fn(raw_tree: object) -> object:
        parameter_tensor = _jax_flatten_pytree(jax_module, jnp, raw_tree)
        return _jax_bounded_qnn_loss(jnp, feature_tensor, label_tensor, parameter_tensor)

    def custom_loss_fn(raw_tree: object) -> object:
        parameter_tensor = _jax_flatten_pytree(jax_module, jnp, raw_tree)
        return _jax_bounded_qnn_loss(jnp, feature_tensor, label_tensor, parameter_tensor)

    custom_loss: Any = _custom_vjp_function(jax_module, custom_loss_fn)

    def custom_loss_fwd(raw_tree: object) -> tuple[object, object]:
        parameter_tensor = _jax_flatten_pytree(jax_module, jnp, raw_tree)
        return (
            _jax_bounded_qnn_loss(jnp, feature_tensor, label_tensor, parameter_tensor),
            parameter_tensor,
        )

    def custom_loss_bwd(parameter_tensor: object, cotangent: object) -> tuple[object]:
        gradient_vector = _jax_bounded_qnn_gradient(
            jnp,
            feature_tensor,
            label_tensor,
            jnp.asarray(parameter_tensor),
        )
        gradient_tree = _jax_unflatten_vector_to_pytree(
            jax_module,
            jnp,
            jnp.asarray(cotangent) * gradient_vector,
            treedef,
            leaf_shapes,
            leaf_sizes,
        )
        return (gradient_tree,)

    custom_loss.defvjp(custom_loss_fwd, custom_loss_bwd)

    native_loss_obj, native_gradient_pytree = jax_module.value_and_grad(native_loss_fn)(
        params_pytree
    )
    custom_loss_obj, custom_gradient_pytree = jax_module.value_and_grad(custom_loss)(params_pytree)
    native_gradient_vector = _flatten_runtime_pytree_gradient(
        jax_module,
        "JAX native PyTree gradient",
        native_gradient_pytree,
        width=parameter_vector.size,
    )
    custom_gradient_vector = _flatten_runtime_pytree_gradient(
        jax_module,
        "JAX custom-VJP PyTree gradient",
        custom_gradient_pytree,
        width=parameter_vector.size,
    )
    reference_gradient = parameter_shift_qnn_classifier_gradient(
        feature_matrix,
        label_vector,
        parameter_vector,
    )
    reference_loss = parameter_shift_qnn_classifier_loss(
        feature_matrix,
        label_vector,
        parameter_vector,
    )
    native_loss = _as_scalar("JAX native PyTree QNN loss", native_loss_obj)
    custom_loss = _as_scalar("JAX custom-VJP PyTree QNN loss", custom_loss_obj)
    loss_error = max(abs(native_loss - reference_loss), abs(custom_loss - reference_loss))
    gradient_error = max(
        float(np.max(np.abs(native_gradient_vector - reference_gradient))),
        float(np.max(np.abs(custom_gradient_vector - reference_gradient))),
    )
    max_abs_error = max(float(loss_error), gradient_error)
    unsupported_native_routes = ("arbitrary_simulator_pytree_lowering",)
    passed = max_abs_error <= tolerance_value
    return PhaseJAXPyTreeCompatibilityResult(
        passed=passed,
        leaf_count=len(leaf_shapes),
        parameter_count=int(parameter_vector.size),
        leaf_shapes=leaf_shapes,
        parameter_vector=parameter_vector.copy(),
        native_qnn_pytree=True,
        native_qnn_host_callback=False,
        custom_vjp_qnn_pytree=True,
        custom_vjp_qnn_host_callback=False,
        custom_vjp_registered=True,
        native_loss=native_loss,
        custom_vjp_loss=custom_loss,
        parameter_shift_loss=reference_loss,
        native_gradient_vector=native_gradient_vector,
        custom_vjp_gradient_vector=custom_gradient_vector,
        parameter_shift_gradient=reference_gradient.copy(),
        native_gradient_pytree=native_gradient_pytree,
        custom_vjp_gradient_pytree=custom_gradient_pytree,
        max_abs_error=max_abs_error,
        tolerance=tolerance_value,
        unsupported_native_routes=unsupported_native_routes,
    )


def run_jax_nested_transform_algebra_audit(
    *,
    features: ArrayLike,
    labels: ArrayLike,
    params_batch: ArrayLike,
    params_pytree: object,
    tolerance: float = 1e-6,
) -> PhaseJAXNestedTransformAlgebraResult:
    """Audit bounded JAX nested-transform algebra for the phase-QNN route.

    This verifies only the implemented bounded classifier path. It does not
    promote arbitrary Phase-QNode lowering, full `jacfwd`/`jacrev` coverage,
    hardware/provider callbacks, or isolated benchmark evidence.
    """
    jax_module, jnp = _load_jax()
    _require_jax_nested_transform_support(jax_module)
    tolerance_value = _as_non_negative_tolerance(tolerance)
    feature_matrix = _as_feature_matrix(features)
    label_vector = _as_label_vector(labels, n_samples=feature_matrix.shape[0])
    parameter_batch = _as_parameter_batch(
        "params_batch",
        params_batch,
        width=feature_matrix.shape[1],
    )
    parameter_vector, treedef, leaf_shapes, leaf_sizes = _as_pytree_parameter_vector(
        jax_module,
        "params_pytree",
        params_pytree,
        width=feature_matrix.shape[1],
    )
    feature_tensor = jnp.asarray(feature_matrix)
    label_tensor = jnp.asarray(label_vector)

    def loss_fn(raw_params: object) -> object:
        return _jax_bounded_qnn_loss(jnp, feature_tensor, label_tensor, jnp.asarray(raw_params))

    value_and_grad_fn = jax_module.value_and_grad(loss_fn)
    jit_value_and_grad = jax_module.jit(value_and_grad_fn)
    _jit_values, jit_under_vmap_obj = jax_module.vmap(jit_value_and_grad)(
        jnp.asarray(parameter_batch)
    )
    _vmap_values, jit_vmap_obj = jax_module.jit(jax_module.vmap(value_and_grad_fn))(
        jnp.asarray(parameter_batch)
    )

    def pytree_loss_fn(raw_tree: object) -> object:
        raw_vector = _jax_flatten_pytree(jax_module, jnp, raw_tree)
        return _jax_bounded_qnn_loss(jnp, feature_tensor, label_tensor, raw_vector)

    _pytree_value, pytree_gradient_obj = jax_module.jit(jax_module.value_and_grad(pytree_loss_fn))(
        params_pytree
    )
    jit_under_vmap_gradients = _as_parameter_batch(
        "JAX jit-under-vmap gradients",
        jit_under_vmap_obj,
        width=feature_matrix.shape[1],
    )
    jit_vmap_gradients = _as_parameter_batch(
        "JAX jit-vmap gradients",
        jit_vmap_obj,
        width=feature_matrix.shape[1],
    )
    pytree_gradient_vector = _flatten_runtime_pytree_gradient(
        jax_module,
        "JAX nested-transform PyTree gradient",
        pytree_gradient_obj,
        width=feature_matrix.shape[1],
    )
    parameter_shift_batch = np.vstack(
        [
            parameter_shift_qnn_classifier_gradient(feature_matrix, label_vector, row)
            for row in parameter_batch
        ],
    )
    parameter_shift_pytree = parameter_shift_qnn_classifier_gradient(
        feature_matrix,
        label_vector,
        parameter_vector,
    )
    del treedef, leaf_shapes, leaf_sizes
    deltas = (
        jit_under_vmap_gradients - parameter_shift_batch,
        jit_vmap_gradients - parameter_shift_batch,
        pytree_gradient_vector - parameter_shift_pytree,
    )
    max_abs_error = max(float(np.max(np.abs(delta))) if delta.size else 0.0 for delta in deltas)
    l2_error = float(np.sqrt(sum(float(np.sum(delta * delta)) for delta in deltas)))
    routes = (
        PhaseJAXNestedTransformRoute(
            name="jit_value_and_grad_under_vmap",
            status="passed",
            reason="bounded phase-QNN JIT(value_and_grad) route agrees under VMAP",
        ),
        PhaseJAXNestedTransformRoute(
            name="jit_vmap_value_and_grad",
            status="passed",
            reason="bounded phase-QNN JIT(VMAP(value_and_grad)) route agrees",
        ),
        PhaseJAXNestedTransformRoute(
            name="jit_value_and_grad_pytree",
            status="passed",
            reason="bounded phase-QNN PyTree value-and-gradient route agrees under JIT",
        ),
        PhaseJAXNestedTransformRoute(
            name="arbitrary_quantum_kernel_jax_lowering",
            status="blocked",
            reason="arbitrary Phase-QNode circuits do not lower into native JAX kernels",
            requires=("jax_lowering_rules", "gate_observable_coverage_matrix"),
        ),
        PhaseJAXNestedTransformRoute(
            name="arbitrary_phase_qnode_jacfwd_jacrev",
            status="blocked",
            reason="full vector-output jacfwd/jacrev algebra is not promoted for arbitrary QNodes",
            requires=("jacfwd_jacrev_parity_artifact", "shape_dtype_transform_matrix"),
        ),
        PhaseJAXNestedTransformRoute(
            name="arbitrary_phase_qnode_hessian",
            status="blocked",
            reason="arbitrary QNode Hessian algebra remains outside the bounded JAX route",
            requires=("hessian_parity_artifact", "conditioning_policy"),
        ),
        PhaseJAXNestedTransformRoute(
            name="hardware_provider_callback_transform_safety",
            status="blocked",
            reason="provider callbacks and hardware execution are not transform-safe JAX routes",
            requires=("provider_allowlist", "live_ticket", "callback_safety_audit"),
        ),
        PhaseJAXNestedTransformRoute(
            name="isolated_benchmark_artifact",
            status="blocked",
            reason="provider-exceedance promotion requires isolated benchmark evidence",
            requires=("isolated_affinity_benchmark_id",),
        ),
    )
    return PhaseJAXNestedTransformAlgebraResult(
        jit_under_vmap_gradients=jit_under_vmap_gradients,
        jit_vmap_gradients=jit_vmap_gradients,
        pytree_gradient_vector=pytree_gradient_vector,
        parameter_shift_batch_gradients=parameter_shift_batch,
        parameter_shift_pytree_gradient=parameter_shift_pytree,
        max_abs_error=max_abs_error,
        l2_error=l2_error,
        tolerance=tolerance_value,
        routes=routes,
    )


def run_jax_phase_qnode_lowering_matrix() -> PhaseJAXPhaseQNodeLoweringMatrixResult:
    """Return the JAX parity matrix for registered Phase-QNode lowering.

    The bounded phase-QNN JAX routes are no-host-callback native framework
    evidence. Arbitrary registered Phase-QNode circuit lowering remains blocked
    until native JAX lowering rules and parity artefacts exist.
    """
    routes = (
        PhaseJAXPhaseQNodeLoweringRoute(
            name="bounded_qnn_native_value_and_grad",
            status="passed",
            reason="bounded phase-QNN loss is expressed directly in JAX operations",
            host_callback=False,
        ),
        PhaseJAXPhaseQNodeLoweringRoute(
            name="bounded_qnn_custom_vjp",
            status="passed",
            reason="bounded phase-QNN custom VJP route is registered without host callbacks",
            host_callback=False,
        ),
        PhaseJAXPhaseQNodeLoweringRoute(
            name="bounded_qnn_jit_value_and_grad",
            status="passed",
            reason="bounded phase-QNN JIT value-and-gradient route is no-host-callback",
            host_callback=False,
        ),
        PhaseJAXPhaseQNodeLoweringRoute(
            name="bounded_qnn_vmap_value_and_grad",
            status="passed",
            reason="bounded phase-QNN VMAP value-and-gradient route is no-host-callback",
            host_callback=False,
        ),
        PhaseJAXPhaseQNodeLoweringRoute(
            name="bounded_qnn_pytree_value_and_grad",
            status="passed",
            reason="bounded phase-QNN PyTree value-and-gradient route is no-host-callback",
            host_callback=False,
        ),
        PhaseJAXPhaseQNodeLoweringRoute(
            name="registered_phase_qnode_statevector_lowering",
            status="passed",
            reason=(
                "registered deterministic Phase-QNode statevector circuits lower into "
                "native JAX value-and-gradient execution without host callbacks"
            ),
            host_callback=False,
        ),
        PhaseJAXPhaseQNodeLoweringRoute(
            name="registered_phase_qnode_native_transform_lowering",
            status="passed",
            reason=(
                "registered deterministic Phase-QNode statevector circuits execute "
                "through native JAX grad, value_and_grad, jacfwd, jacrev, hessian, "
                "jvp, vjp, vmap, and jit routes without host callbacks"
            ),
            host_callback=False,
        ),
        PhaseJAXPhaseQNodeLoweringRoute(
            name="registered_phase_qnode_pytree_transform_lowering",
            status="passed",
            reason=(
                "registered deterministic Phase-QNode statevector circuits execute "
                "structured PyTree parameters through native JAX grad, value_and_grad, "
                "jacfwd, jacrev, jvp, vjp, vmap, and jit routes without host callbacks"
            ),
            host_callback=False,
        ),
        PhaseJAXPhaseQNodeLoweringRoute(
            name="registered_phase_qnode_pmap_sharding_lowering",
            status="passed",
            reason=(
                "registered deterministic Phase-QNode statevector circuit batches execute "
                "one row per local JAX device through native pmap value-and-gradient "
                "routes without host callbacks"
            ),
            host_callback=False,
        ),
        PhaseJAXPhaseQNodeLoweringRoute(
            name="registered_phase_qnode_aot_export_lowering",
            status="passed",
            reason=(
                "registered deterministic Phase-QNode statevector value routes can be "
                "staged through JAX AOT lowering and jax.export serialization diagnostics "
                "without host callbacks"
            ),
            host_callback=False,
        ),
        PhaseJAXPhaseQNodeLoweringRoute(
            name="registered_phase_qnode_finite_shot_lowering",
            status="blocked",
            reason="finite-shot JAX lowering needs sampler, seed, and uncertainty provenance",
            host_callback=False,
            requires=(
                "shot_policy",
                "rng_seed_provenance",
                "uncertainty_artifact",
            ),
        ),
        PhaseJAXPhaseQNodeLoweringRoute(
            name="registered_phase_qnode_provider_lowering",
            status="blocked",
            reason="provider callbacks are not native JAX transform-safe routes",
            host_callback=False,
            requires=(
                "provider_allowlist",
                "callback_transform_safety_audit",
                "provider_execution_artifact",
            ),
        ),
        PhaseJAXPhaseQNodeLoweringRoute(
            name="registered_phase_qnode_hardware_lowering",
            status="blocked",
            reason="live hardware JAX lowering requires ticketed execution evidence",
            host_callback=False,
            requires=(
                "live_ticket",
                "provider_allowlist",
                "shot_budget",
                "hardware_evidence_id",
            ),
        ),
        PhaseJAXPhaseQNodeLoweringRoute(
            name="registered_phase_qnode_dynamic_circuit_lowering",
            status="blocked",
            reason="mid-circuit measurement and feedback are outside the native JAX lowering boundary",
            host_callback=False,
            requires=(
                "dynamic_circuit_semantics",
                "classical_feedback_contract",
                "gradient_policy",
            ),
        ),
        PhaseJAXPhaseQNodeLoweringRoute(
            name="isolated_benchmark_artifact",
            status="blocked",
            reason="provider-exceedance promotion requires isolated benchmark evidence",
            host_callback=False,
            requires=("isolated_affinity_benchmark_id",),
        ),
    )
    return PhaseJAXPhaseQNodeLoweringMatrixResult(routes=routes)


def plan_jax_cloud_validation_batch(
    *,
    runner: str = "jarvislabs",
    accelerator_backend: str = "cuda",
) -> PhaseJAXCloudValidationRunSpec:
    """Plan the JAX cloud validation batch for blocked local accelerator routes.

    Parameters
    ----------
    runner:
        Human-readable runner label used in the downstream validation queue.
    accelerator_backend:
        Accelerator runtime requested for the cloud rerun. The differentiable
        lane currently accepts ``"cuda"`` and ``"rocm"`` plans.

    Returns
    -------
    PhaseJAXCloudValidationRunSpec
        JSON-ready scheduling metadata with local skip status, required cloud
        artefacts, environment constraints, and reproduction commands.
    """
    clean_runner = runner.strip()
    if not clean_runner:
        raise ValueError("runner must be a non-empty string")
    clean_backend = accelerator_backend.strip().lower()
    if clean_backend not in {"cuda", "rocm"}:
        raise ValueError("accelerator_backend must be 'cuda' or 'rocm'")

    jax_module, _ = _load_jax()
    local_device_count_fn = getattr(jax_module, "local_device_count", None)
    local_device_count = int(local_device_count_fn()) if callable(local_device_count_fn) else 0
    device_descriptions = _jax_local_devices(jax_module, max(local_device_count, 0))
    local_skip_reason = _jax_cloud_local_skip_reason(
        accelerator_backend=clean_backend,
        local_device_count=local_device_count,
        device_descriptions=device_descriptions,
    )
    blocked_local_routes = _jax_cloud_blocked_routes(
        accelerator_backend=clean_backend,
        local_skip_reason=local_skip_reason,
        local_device_count=local_device_count,
    )
    local_execution_status = (
        "local_accelerator_ready"
        if not blocked_local_routes
        else "skipped_incompatible_local_hardware"
    )
    commands = (
        ".venv/bin/python -m pytest "
        "tests/test_phase_jax_bridge.py::"
        "test_phase_jax_registered_qnode_sharding_transform_audit_uses_no_callback "
        "tests/test_phase_jax_bridge.py::"
        "test_phase_jax_sharding_compatibility_audit_batches_native_and_custom_vjp "
        "tests/test_phase_jax_bridge.py::"
        "test_phase_jax_phase_qnode_lowering_matrix_fails_closed_for_arbitrary_qnodes -q",
        ".venv/bin/python -m pytest "
        "tests/test_differentiable_programming_benchmarks.py::"
        "test_quantum_gradient_benchmark_suite_matches_analytic_references -q",
        ".venv/bin/python - <<'PY'\n"
        "from scpn_quantum_control.phase import plan_jax_cloud_validation_batch\n"
        "print(plan_jax_cloud_validation_batch().to_dict())\n"
        "PY",
    )
    return PhaseJAXCloudValidationRunSpec(
        runner=clean_runner,
        local_execution_status=local_execution_status,
        local_skip_reason=local_skip_reason,
        accelerator_backend=clean_backend,
        local_device_count=local_device_count,
        device_descriptions=device_descriptions,
        blocked_local_routes=blocked_local_routes,
        required_artifacts=(
            "jax_cuda_device_metadata_artifact",
            "jax_xla_gpu_compile_artifact",
            "registered_phase_qnode_jax_pmap_sharding_artifact",
            "jax_multi_device_value_and_gradient_artifact",
            "isolated_benchmark_artifact",
            "host_load_and_affinity_metadata",
        ),
        required_environment={
            "accelerator_backend": clean_backend,
            "minimum_cuda_compute_capability": "7.5" if clean_backend == "cuda" else None,
            "minimum_visible_device_count": 2,
            "blocked_local_device_patterns": ("GTX 1060",),
            "visible_device_metadata_required": True,
            "host_load_metadata_required": True,
            "isolated_affinity_required_for_promotion": True,
            "network_required": False,
            "hardware_submission_allowed": False,
        },
        commands=commands,
        ready_for_cloud_dispatch=bool(blocked_local_routes),
    )


def _jax_cloud_local_skip_reason(
    *,
    accelerator_backend: str,
    local_device_count: int,
    device_descriptions: tuple[str, ...],
) -> str:
    joined_devices = " ".join(device_descriptions)
    lower_devices = joined_devices.lower()
    reasons: list[str] = []
    if local_device_count < 2:
        reasons.append(
            "JAX PMAP promotion requires at least two visible accelerator devices; "
            f"local_device_count={local_device_count}"
        )
    if accelerator_backend == "cuda":
        if "gtx 1060" in lower_devices:
            reasons.append(
                "local GTX 1060 does not satisfy the CUDA cloud validation floor "
                "or current JAX CUDA wheel route"
            )
        elif not any(token in lower_devices for token in ("cuda", "gpu", "nvidia")):
            reasons.append("no CUDA/GPU JAX device metadata is visible locally")
    elif not any(token in lower_devices for token in ("rocm", "amd", "gpu")):
        reasons.append("no ROCm/GPU JAX device metadata is visible locally")
    return "; ".join(reasons)


def _jax_cloud_blocked_routes(
    *,
    accelerator_backend: str,
    local_skip_reason: str,
    local_device_count: int,
) -> tuple[str, ...]:
    routes: list[str] = []
    if local_skip_reason:
        routes.append(f"jax_{accelerator_backend}_accelerator_device")
    if local_device_count < 2 or local_skip_reason:
        routes.append("registered_phase_qnode_pmap_multi_device_lowering")
    if routes:
        routes.append("isolated_benchmark_artifact")
    return tuple(dict.fromkeys(routes))


def run_jax_maturity_audit(
    *,
    features: ArrayLike,
    labels: ArrayLike,
    params: ArrayLike,
    params_batch: ArrayLike,
    params_pytree: object,
    tolerance: float = 1e-6,
) -> PhaseJAXMaturityAuditResult:
    """Aggregate bounded JAX evidence and provider-level parity blockers.

    The audit intentionally separates the bounded phase-QNN evidence that is
    implemented today from the larger JAX ecosystem maturity target. It does
    not promote arbitrary quantum kernels, provider callbacks, hardware
    gradients, or benchmark claims until those routes have their own artefacts.
    """
    tolerance_value = _as_non_negative_tolerance(tolerance)
    feature_matrix = _as_feature_matrix(features)
    label_vector = _as_label_vector(labels, n_samples=feature_matrix.shape[0])
    parameter_values = _as_parameter_vector(
        "params",
        params,
        width=feature_matrix.shape[1],
    )
    parameter_batch = _as_parameter_batch(
        "params_batch",
        params_batch,
        width=feature_matrix.shape[1],
    )

    custom_vjp = jax_custom_vjp_qnn_value_and_grad(
        feature_matrix,
        label_vector,
        parameter_values,
        tolerance=tolerance_value,
    )
    jit = run_jax_jit_compatibility_audit(
        features=feature_matrix,
        labels=label_vector,
        params=parameter_values,
        tolerance=tolerance_value,
    )
    vmap = run_jax_vmap_compatibility_audit(
        features=feature_matrix,
        labels=label_vector,
        params_batch=parameter_batch,
        tolerance=tolerance_value,
    )
    sharding = run_jax_sharding_compatibility_audit(
        features=feature_matrix,
        labels=label_vector,
        params_batch=parameter_batch,
        tolerance=tolerance_value,
    )
    pytree = run_jax_pytree_compatibility_audit(
        features=feature_matrix,
        labels=label_vector,
        params_pytree=params_pytree,
        tolerance=tolerance_value,
    )
    nested_transform_algebra = run_jax_nested_transform_algebra_audit(
        features=feature_matrix,
        labels=label_vector,
        params_batch=parameter_batch,
        params_pytree=params_pytree,
        tolerance=tolerance_value,
    )
    phase_qnode_lowering_matrix = run_jax_phase_qnode_lowering_matrix()
    cloud_validation_batch = plan_jax_cloud_validation_batch()

    evidence: dict[str, object] = {
        "custom_vjp": custom_vjp,
        "jit": jit,
        "vmap": vmap,
        "pmap_sharding": sharding,
        "pytree": pytree,
        "nested_transform_algebra": nested_transform_algebra,
        "phase_qnode_lowering_matrix": phase_qnode_lowering_matrix,
        "cloud_validation_batch": cloud_validation_batch,
    }
    bounded_model_ready = all(
        bool(getattr(result, "passed", False))
        for name, result in evidence.items()
        if name
        not in {
            "phase_qnode_lowering_matrix",
            "cloud_validation_batch",
        }
    )
    required_capabilities = {
        "custom_vjp": "passed" if custom_vjp.passed else "failed",
        "jit": "passed" if jit.passed else "failed",
        "vmap": "passed" if vmap.passed else "failed",
        "pmap_sharding": "passed" if sharding.passed else "failed",
        "pytree": "passed" if pytree.passed else "failed",
        "nested_transform_algebra": "passed" if nested_transform_algebra.passed else "failed",
        "phase_qnode_lowering_matrix": (
            "passed" if phase_qnode_lowering_matrix.ready_for_provider_exceedance else "blocked"
        ),
        "cloud_validation_batch": (
            "scheduled" if cloud_validation_batch.ready_for_cloud_dispatch else "not_required"
        ),
        "arbitrary_quantum_kernel_jax_lowering": "blocked",
        "hardware_or_provider_callback_transform_safety": "blocked",
        "promotion_grade_isolated_benchmarks": "blocked",
    }
    required_capabilities.update(
        {
            f"nested_transform:{route.name}": route.status
            for route in nested_transform_algebra.routes
            if route.status != "passed"
        }
    )
    required_capabilities.update(
        {
            f"phase_qnode_lowering:{route.name}": route.status
            for route in phase_qnode_lowering_matrix.routes
            if route.status != "passed"
        }
    )
    open_gaps = tuple(name for name, status in required_capabilities.items() if status != "passed")
    return PhaseJAXMaturityAuditResult(
        bounded_model_ready=bounded_model_ready,
        ready_for_provider_exceedance=bounded_model_ready and not open_gaps,
        evidence=evidence,
        required_capabilities=required_capabilities,
        open_gaps=open_gaps,
    )


__all__ = [
    "PhaseJAXCloudValidationRunSpec",
    "PhaseJAXCustomVJPQNNGradientResult",
    "PhaseJAXGradientAgreementResult",
    "PhaseJAXJITCompatibilityResult",
    "PhaseJAXMaturityAuditResult",
    "PhaseJAXNativeQNNGradientResult",
    "PhaseJAXNestedTransformAlgebraResult",
    "PhaseJAXNestedTransformRoute",
    "PhaseJAXParameterShiftResult",
    "PhaseJAXPhaseQNodeLoweringMatrixResult",
    "PhaseJAXPhaseQNodeLoweringRoute",
    "PhaseJAXPhaseQNodeAOTExportResult",
    "PhaseJAXPhaseQNodeNativeTransformResult",
    "PhaseJAXPhaseQNodePyTreeTransformResult",
    "PhaseJAXPhaseQNodeShardingTransformResult",
    "PhaseJAXPhaseQNodeStatevectorResult",
    "PhaseJAXPyTreeCompatibilityResult",
    "PhaseJAXShardingCompatibilityResult",
    "PhaseJAXVMAPCompatibilityResult",
    "check_jax_parameter_shift_agreement",
    "is_phase_jax_available",
    "jax_custom_vjp_qnn_value_and_grad",
    "jax_native_qnn_value_and_grad",
    "jax_parameter_shift_value_and_grad",
    "jax_phase_qnode_aot_export_audit",
    "jax_phase_qnode_native_transform_audit",
    "jax_phase_qnode_pytree_transform_audit",
    "jax_phase_qnode_sharding_transform_audit",
    "jax_phase_qnode_value_and_grad",
    "plan_jax_cloud_validation_batch",
    "run_jax_jit_compatibility_audit",
    "run_jax_maturity_audit",
    "run_jax_nested_transform_algebra_audit",
    "run_jax_phase_qnode_lowering_matrix",
    "run_jax_pytree_compatibility_audit",
    "run_jax_sharding_compatibility_audit",
    "run_jax_vmap_compatibility_audit",
]
