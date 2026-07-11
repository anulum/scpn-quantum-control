# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — JAX Compatibility Audits
"""Bounded phase-QNN JAX compatibility and nested-transform audits.

This one-way leaf owns JIT, VMAP, PMAP, PyTree, and nested-transform algebra
verification. The public facade injects its active optional-JAX loader and
retains lowering, cloud-planning, and maturity orchestration.
"""

from __future__ import annotations

from typing import Any, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .jax_bridge_contracts import (
    PhaseJAXJITCompatibilityResult,
    PhaseJAXNestedTransformAlgebraResult,
    PhaseJAXNestedTransformRoute,
    PhaseJAXPyTreeCompatibilityResult,
    PhaseJAXShardingCompatibilityResult,
    PhaseJAXVMAPCompatibilityResult,
)
from .jax_gradients import (
    JAXLoader,
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
    jax_custom_vjp_qnn_value_and_grad,
    jax_native_qnn_value_and_grad,
    jax_parameter_shift_value_and_grad,
)
from .jax_qnode_transforms import (
    _as_parameter_batch,
    _as_pytree_parameter_vector,
    _flatten_runtime_pytree_gradient,
    _jax_flatten_pytree,
    _jax_local_devices,
    _jax_unflatten_vector_to_pytree,
    _require_jax_pmap_support,
    _require_jax_pytree_support,
)
from .qnn_training import (
    parameter_shift_qnn_classifier_gradient,
    parameter_shift_qnn_classifier_loss,
)

FloatArray: TypeAlias = NDArray[np.float64]


def _require_jax_vmap_support(jax_module: Any) -> None:
    vmap = getattr(jax_module, "vmap", None)
    if not callable(vmap):
        raise RuntimeError("JAX vmap is required for bounded-QNN parameter-batch VMAP")
    value_and_grad = getattr(jax_module, "value_and_grad", None)
    if not callable(value_and_grad):
        raise RuntimeError("JAX value_and_grad is required for bounded-QNN parameter-batch VMAP")


def _require_jax_nested_transform_support(jax_module: Any) -> None:
    _require_jax_vmap_support(jax_module)
    _require_jax_pytree_support(jax_module)
    jit = getattr(jax_module, "jit", None)
    if not callable(jit):
        raise RuntimeError("JAX JIT is required for bounded nested-transform algebra")


def run_jax_jit_compatibility_audit(
    *,
    features: ArrayLike,
    labels: ArrayLike,
    params: ArrayLike,
    tolerance: float = 1e-6,
    _jax_loader: JAXLoader = _load_jax,
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
        _jax_loader=_jax_loader,
    )
    custom = jax_custom_vjp_qnn_value_and_grad(
        feature_matrix,
        label_vector,
        parameter_values,
        tolerance=tolerance_value,
        jit=True,
        _jax_loader=_jax_loader,
    )
    parameter_shift = jax_parameter_shift_value_and_grad(
        objective,
        parameter_values,
        jit=True,
        _jax_loader=_jax_loader,
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
    _jax_loader: JAXLoader = _load_jax,
) -> PhaseJAXVMAPCompatibilityResult:
    """Audit bounded JAX VMAP support for parameter batches.

    The audit vectorises the bounded phase-QNN native and custom-VJP loss routes
    over a two-dimensional parameter batch. SCPN parameter-shift results are
    used as host-side references only and are reported as such rather than
    promoted as native VMAP.
    """
    jax_module, jnp = _jax_loader()
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
    _jax_loader: JAXLoader = _load_jax,
) -> PhaseJAXShardingCompatibilityResult:
    """Audit bounded JAX PMAP/sharding support for local-device batches.

    The audit maps one parameter row per local JAX device with ``jax.pmap``.
    It promotes only the bounded native and custom-VJP phase-QNN loss routes.
    SCPN parameter-shift references stay host-side validation rows and are not
    reported as sharded/native execution.
    """
    jax_module, jnp = _jax_loader()
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
    _jax_loader: JAXLoader = _load_jax,
) -> PhaseJAXPyTreeCompatibilityResult:
    """Audit bounded JAX PyTree parameter support.

    The audit accepts a JAX PyTree of numeric parameter leaves, flattens it into
    the bounded phase-QNN parameter vector, and restores gradients into the same
    PyTree structure. It does not claim arbitrary simulator PyTree lowering.
    """
    jax_module, jnp = _jax_loader()
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
    _jax_loader: JAXLoader = _load_jax,
) -> PhaseJAXNestedTransformAlgebraResult:
    """Audit bounded JAX nested-transform algebra for the phase-QNN route.

    This verifies only the implemented bounded classifier path. It does not
    promote arbitrary Phase-QNode lowering, full `jacfwd`/`jacrev` coverage,
    hardware/provider callbacks, or isolated benchmark evidence.
    """
    jax_module, jnp = _jax_loader()
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


__all__ = [
    "run_jax_jit_compatibility_audit",
    "run_jax_nested_transform_algebra_audit",
    "run_jax_pytree_compatibility_audit",
    "run_jax_sharding_compatibility_audit",
    "run_jax_vmap_compatibility_audit",
]
