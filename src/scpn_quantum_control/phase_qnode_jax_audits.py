# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — JAX Phase-QNode capability audits
"""Execute JAX Phase-QNode capabilities and compare them with parameter shift.

Each audit runs real JAX capabilities on an admitted registered circuit: the
native transforms (``grad``, ``value_and_grad``, ``jacfwd``, ``jacrev``,
``hessian``, ``jvp``, ``vjp``, ``vmap`` and ``jit``), the same transforms over
parameter PyTrees, ``pmap`` sharding, and AOT lowering with ``jax.export``
serialisation. The caller supplies the JAX modules, admitted parameters and the
differentiable value function; this module never loads a framework. Every
admission and conversion operation is reached through a resolver called where
the audit needs it.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from importlib import import_module
from typing import Any, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from .differentiable_native_curvature import jax_hessian, jax_jacfwd, jax_jacrev
from .differentiable_native_gradient import jax_gradient
from .differentiable_native_linearization import jax_jvp, jax_vjp
from .differentiable_tree_vectors import (
    batched_gradient_tree_matrix,
    gradient_tree_vector,
    hessian_tree_matrix,
    jax_batch_tree,
    jax_tree_vector,
    jax_vector_tree,
    parameter_tree_vector,
)
from .phase.jax_bridge_contracts import (
    PhaseJAXPhaseQNodeAOTExportResult,
    PhaseJAXPhaseQNodeNativeTransformResult,
    PhaseJAXPhaseQNodePyTreeTransformResult,
    PhaseJAXPhaseQNodeShardingTransformResult,
)
from .phase.qnode_circuit import (
    PhaseQNodeCircuit,
    PhaseQNodeSupportError,
    parameter_shift_phase_qnode_gradient,
    phase_qnode_support_report,
)
from .phase_qnode_audit_admission import (
    admitted_jax_tolerance,
    admitted_parameter_batch,
    admitted_parameter_vector,
    admitted_scalar,
)
from .phase_qnode_native_statevector import jax_statevector_value_and_state

FloatArray: TypeAlias = NDArray[np.float64]
Resolver: TypeAlias = Callable[[], Callable[..., Any]]
"""Zero-argument callable returning the operation to call at that moment."""

TRANSFORM_NAMES: tuple[str, ...] = (
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
"""Native JAX transforms exercised by the flat and PyTree audits."""


@dataclass(frozen=True)
class JAXAuditOperations:
    """Resolvers for the admission and conversion operations of the JAX audits.

    Each field returns the operation when called; the audits call it exactly
    where the operation is needed.

    Parameters
    ----------
    support_report, parameter_shift
        Circuit support report and parameter-shift reference.
    support_error
        Exception class raised with an unsupported circuit's report.
    parameter_vector, parameter_batch, scalar
        Admission of host vectors, batches and scalars.
    unflatten_vector, unflatten_batch
        Rebuild native PyTrees from a parameter vector or batch.
    flatten_gradient, flatten_batched_gradient, flatten_hessian
        Flatten native PyTree derivatives into host arrays.
    local_devices
        ``(jax, count) -> tuple[str, ...]`` device descriptions.
    export_version
        ``(export_module, name) -> int`` calling-convention version reader.

    """

    support_report: Resolver
    parameter_shift: Resolver
    support_error: Resolver
    parameter_vector: Resolver
    parameter_batch: Resolver
    scalar: Resolver
    unflatten_vector: Resolver
    unflatten_batch: Resolver
    flatten_gradient: Resolver
    flatten_batched_gradient: Resolver
    flatten_hessian: Resolver
    local_devices: Resolver
    export_version: Resolver


def enable_jax_x64(jax_module: Any) -> None:
    """Enable JAX 64-bit mode when the runtime exposes ``config.update``.

    Parameters
    ----------
    jax_module
        Supplied JAX module.

    """
    config = getattr(jax_module, "config", None)
    update = getattr(config, "update", None)
    if callable(update):
        update("jax_enable_x64", True)


def require_jax_transform_support(jax_module: Any, *, surface: str) -> None:
    """Refuse a JAX module that lacks any of the audited transforms.

    Parameters
    ----------
    jax_module
        Supplied JAX module.
    surface
        Audited surface named in the refusal, such as ``"native transforms"``.

    Raises
    ------
    RuntimeError
        If any transform in :data:`TRANSFORM_NAMES` is missing.

    """
    missing = tuple(
        name for name in TRANSFORM_NAMES if not callable(getattr(jax_module, name, None))
    )
    if missing:
        missing_names = ", ".join(missing)
        raise RuntimeError(
            f"JAX {surface} are required for registered Phase-QNode lowering: {missing_names}"
        )


def require_jax_pmap_support(jax_module: Any) -> None:
    """Refuse a JAX module without ``pmap``, ``value_and_grad`` or device counting.

    Parameters
    ----------
    jax_module
        Supplied JAX module.

    Raises
    ------
    RuntimeError
        If one of the required entry points is missing.

    """
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


def require_jax_pytree_support(jax_module: Any) -> None:
    """Refuse a JAX module without PyTree flattening or ``value_and_grad``.

    Parameters
    ----------
    jax_module
        Supplied JAX module.

    Raises
    ------
    RuntimeError
        If ``tree_util`` or one of the required entry points is missing.

    """
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


def require_jax_aot_export_support(
    jax_module: Any, *, importer: Resolver, suppressor: Resolver = lambda: suppress
) -> Any:
    """Return ``jax.export`` after checking every AOT lowering entry point.

    Parameters
    ----------
    jax_module
        Supplied JAX module.
    importer
        Resolver of the module importer, called only when ``jax.export`` must
        be imported explicitly.
    suppressor
        Resolver of the context manager that tolerates a failed import.

    Returns
    -------
    Any
        The JAX export module.

    Raises
    ------
    RuntimeError
        If JIT, ``ShapeDtypeStruct``, the export module or one of its entry
        points or calling-convention bounds is missing.

    """
    jit = getattr(jax_module, "jit", None)
    if not callable(jit):
        raise RuntimeError("JAX JIT is required for registered Phase-QNode AOT lowering")
    if not callable(getattr(jax_module, "ShapeDtypeStruct", None)):
        raise RuntimeError(
            "JAX ShapeDtypeStruct is required for registered Phase-QNode AOT export"
        )
    export_module = getattr(jax_module, "export", None)
    if export_module is None and getattr(jax_module, "__name__", None) == "jax":
        with suppressor()(ImportError):
            export_module = importer()("jax.export")
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


def jax_export_version(export_module: Any, name: str) -> int:
    """Return a JAX export calling-convention version from attr or function APIs.

    Parameters
    ----------
    export_module
        JAX export module.
    name
        Attribute name of the version bound.

    Returns
    -------
    int
        Non-negative version number.

    Raises
    ------
    RuntimeError
        If the version is negative.

    """
    value = getattr(export_module, name)
    if callable(value):
        value = value()
    version = int(value)
    if version < 0:
        raise RuntimeError(f"JAX export.{name} must be non-negative")
    return version


def jax_local_devices(jax_module: Any, local_device_count: int) -> tuple[str, ...]:
    """Return local device descriptions, or numbered placeholders.

    Parameters
    ----------
    jax_module
        Supplied JAX module.
    local_device_count
        Number of local devices reported by the runtime.

    Returns
    -------
    tuple[str, ...]
        One description per local device.

    """
    local_devices = getattr(jax_module, "local_devices", None)
    if not callable(local_devices):
        return tuple(f"local-device-{index}" for index in range(local_device_count))
    devices = tuple(str(device) for device in local_devices())
    if len(devices) != local_device_count:
        return tuple(f"local-device-{index}" for index in range(local_device_count))
    return devices


def _ensure_supported(
    operations: JAXAuditOperations, circuit: PhaseQNodeCircuit, values: FloatArray
) -> None:
    report = operations.support_report()(circuit, values)
    if not report.supported:
        raise operations.support_error()(report)


def _tangent_and_batch(
    operations: JAXAuditOperations,
    parameter_values: FloatArray,
    tangent: object | None,
    batch_offsets: object | None,
) -> tuple[FloatArray, FloatArray]:
    if tangent is None:
        tangent_values = np.linspace(
            0.25,
            0.25 + 0.05 * (parameter_values.size - 1),
            parameter_values.size,
            dtype=np.float64,
        )
    else:
        tangent_values = operations.parameter_vector()(
            "tangent", tangent, width=parameter_values.size
        )
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
        offset_values = operations.parameter_batch()(
            "batch_offsets",
            batch_offsets,
            width=parameter_values.size,
        )
        batch_values = parameter_values + offset_values
    return tangent_values, batch_values


def _batch_references(
    operations: JAXAuditOperations, circuit: PhaseQNodeCircuit, batch_values: FloatArray
) -> FloatArray:
    return np.vstack(
        [operations.parameter_shift()(circuit, row).gradient for row in batch_values]
    ).astype(np.float64, copy=False)


def _batch_value_error(
    operations: JAXAuditOperations,
    circuit: PhaseQNodeCircuit,
    batch_values: FloatArray,
    vmap_values: FloatArray,
) -> float:
    return float(
        np.max(
            np.abs(
                vmap_values
                - np.asarray(
                    [operations.parameter_shift()(circuit, row).value for row in batch_values],
                    dtype=np.float64,
                )
            ),
            initial=0.0,
        )
    )


def jax_native_transform_audit(
    jax_module: Any,
    jnp: Any,
    circuit: PhaseQNodeCircuit,
    parameter_values: FloatArray,
    tolerance: float,
    value_function: Callable[[object], object],
    operations: JAXAuditOperations,
    *,
    tangent: object | None,
    batch_offsets: object | None,
) -> PhaseJAXPhaseQNodeNativeTransformResult:
    """Run every native JAX transform on the value function and compare with references.

    Parameters
    ----------
    jax_module, jnp
        Supplied JAX module and ``jax.numpy`` namespace.
    circuit
        Registered circuit; refused with ``PhaseQNodeSupportError`` if unsupported.
    parameter_values
        Admitted float64 parameters.
    tolerance
        Admitted non-negative tolerance.
    value_function
        Differentiable ``parameters -> value`` of the circuit.
    operations
        Admission and conversion resolvers.
    tangent
        Optional JVP tangent; a fixed ramp by default.
    batch_offsets
        Optional ``vmap`` offsets; zero, ``+0.03`` and ``-0.02`` steps by default.

    Returns
    -------
    PhaseJAXPhaseQNodeNativeTransformResult
        Transform outputs with gradient, transform and Hessian-symmetry errors.

    Raises
    ------
    RuntimeError
        If the Hessian has an unexpected shape.

    """
    _ensure_supported(operations, circuit, parameter_values)
    tangent_values, batch_values = _tangent_and_batch(
        operations, parameter_values, tangent, batch_offsets
    )
    parameter_shift = operations.parameter_shift()(circuit, parameter_values)

    raw_params = jnp.asarray(parameter_values)
    raw_tangent = jnp.asarray(tangent_values)
    raw_batch = jnp.asarray(batch_values)
    gradient_obj = jax_gradient(jax_module, value_function)(raw_params)
    value_and_grad_fn = jax_module.value_and_grad(value_function)
    value_and_grad_value_obj, value_and_grad_gradient_obj = value_and_grad_fn(raw_params)
    jit_value_obj, jit_gradient_obj = jax_module.jit(value_and_grad_fn)(raw_params)
    jacfwd_obj = jax_jacfwd(jax_module, value_function, raw_params)
    jacrev_obj = jax_jacrev(jax_module, value_function, raw_params)
    hessian_obj = jax_hessian(jax_module, value_function, raw_params)
    jvp_value_obj, jvp_tangent_obj = jax_jvp(jax_module, value_function, raw_params, raw_tangent)
    vjp_value_obj, pullback = jax_vjp(jax_module, value_function, raw_params)
    (vjp_cotangent_obj,) = pullback(jnp.asarray(1.0))
    vmap_value_and_grad = jax_module.vmap(value_and_grad_fn)
    vmap_values_obj, vmap_gradients_obj = vmap_value_and_grad(raw_batch)

    width = parameter_values.size
    vector = operations.parameter_vector
    gradient = vector()("JAX grad Phase-QNode gradient", gradient_obj, width=width)
    value_and_grad_gradient = vector()(
        "JAX value_and_grad Phase-QNode gradient", value_and_grad_gradient_obj, width=width
    )
    jit_gradient = vector()(
        "JAX jit value_and_grad Phase-QNode gradient", jit_gradient_obj, width=width
    )
    jacfwd_gradient = vector()("JAX jacfwd Phase-QNode gradient", jacfwd_obj, width=width)
    jacrev_gradient = vector()("JAX jacrev Phase-QNode gradient", jacrev_obj, width=width)
    vjp_cotangent_gradient = vector()(
        "JAX vjp Phase-QNode cotangent gradient", vjp_cotangent_obj, width=width
    )
    vmap_values = vector()(
        "JAX vmap Phase-QNode values", vmap_values_obj, width=batch_values.shape[0]
    )
    vmap_gradients = operations.parameter_batch()(
        "JAX vmap Phase-QNode gradients", vmap_gradients_obj, width=width
    )
    hessian_matrix = np.asarray(hessian_obj, dtype=np.float64)
    if hessian_matrix.shape != (width, width):
        raise RuntimeError("JAX Phase-QNode hessian has an unexpected shape")
    batch_parameter_shift_gradients = _batch_references(operations, circuit, batch_values)
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
    scalar = operations.scalar
    value_errors = (
        abs(
            scalar()("JAX value_and_grad Phase-QNode value", value_and_grad_value_obj)
            - parameter_shift.value
        ),
        abs(scalar()("JAX jit Phase-QNode value", jit_value_obj) - parameter_shift.value),
        abs(scalar()("JAX jvp Phase-QNode value", jvp_value_obj) - parameter_shift.value),
        abs(scalar()("JAX vjp Phase-QNode value", vjp_value_obj) - parameter_shift.value),
    )
    expected_jvp = float(np.dot(parameter_shift.gradient, tangent_values))
    max_abs_transform_error = max(
        *(float(error) for error in value_errors),
        abs(scalar()("JAX jvp Phase-QNode tangent", jvp_tangent_obj) - expected_jvp),
        _batch_value_error(operations, circuit, batch_values, vmap_values),
    )
    max_abs_hessian_symmetry_error = float(
        np.max(np.abs(hessian_matrix - hessian_matrix.T), initial=0.0)
    )
    passed = bool(
        max_abs_gradient_error <= tolerance
        and max_abs_transform_error <= tolerance
        and max_abs_hessian_symmetry_error <= tolerance
    )
    return PhaseJAXPhaseQNodeNativeTransformResult(
        value=scalar()("JAX grad Phase-QNode value", value_function(raw_params)),
        gradient=gradient,
        value_and_grad_value=scalar()(
            "JAX value_and_grad Phase-QNode value", value_and_grad_value_obj
        ),
        value_and_grad_gradient=value_and_grad_gradient,
        jacfwd_gradient=jacfwd_gradient,
        jacrev_gradient=jacrev_gradient,
        hessian=hessian_matrix.copy(),
        jvp_value=scalar()("JAX jvp Phase-QNode value", jvp_value_obj),
        jvp_tangent=scalar()("JAX jvp Phase-QNode tangent", jvp_tangent_obj),
        vjp_value=scalar()("JAX vjp Phase-QNode value", vjp_value_obj),
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
        tolerance=tolerance,
        passed=passed,
        native_framework_autodiff=True,
        host_callback=False,
        jit_value_and_grad=True,
        vmap_value_and_grad=True,
        transform_names=TRANSFORM_NAMES,
    )


def jax_pytree_transform_audit(
    jax_module: Any,
    jnp: Any,
    circuit: PhaseQNodeCircuit,
    flattened: tuple[FloatArray, object, tuple[tuple[int, ...], ...], tuple[int, ...]],
    tolerance: float,
    value_function: Callable[[object], object],
    operations: JAXAuditOperations,
    *,
    tangent: object | None,
    batch_offsets: object | None,
) -> PhaseJAXPhaseQNodePyTreeTransformResult:
    """Run every native JAX transform over a parameter PyTree and compare with references.

    Parameters
    ----------
    jax_module, jnp
        Supplied JAX module and ``jax.numpy`` namespace.
    circuit
        Registered circuit; refused with ``PhaseQNodeSupportError`` if unsupported.
    flattened
        Admitted parameter vector, tree definition, leaf shapes and leaf sizes.
    tolerance
        Admitted non-negative tolerance.
    value_function
        Differentiable ``parameter tree -> value`` of the circuit.
    operations
        Admission, PyTree and conversion resolvers.
    tangent
        Optional JVP tangent vector; a fixed ramp by default.
    batch_offsets
        Optional ``vmap`` offsets; zero, ``+0.03`` and ``-0.02`` steps by default.

    Returns
    -------
    PhaseJAXPhaseQNodePyTreeTransformResult
        Transform outputs and native PyTrees with their reference errors.

    """
    parameter_values, treedef, leaf_shapes, leaf_sizes = flattened
    _ensure_supported(operations, circuit, parameter_values)
    tangent_values, batch_values = _tangent_and_batch(
        operations, parameter_values, tangent, batch_offsets
    )
    parameter_shift = operations.parameter_shift()(circuit, parameter_values)
    raw_params = operations.unflatten_vector()(
        jax_module, jnp, jnp.asarray(parameter_values), treedef, leaf_shapes, leaf_sizes
    )
    raw_tangent = operations.unflatten_vector()(
        jax_module, jnp, jnp.asarray(tangent_values), treedef, leaf_shapes, leaf_sizes
    )
    raw_batch = operations.unflatten_batch()(
        jax_module, jnp, jnp.asarray(batch_values), treedef, leaf_shapes, leaf_sizes
    )

    gradient_pytree_obj = jax_gradient(jax_module, value_function)(raw_params)
    value_and_grad_fn = jax_module.value_and_grad(value_function)
    value_and_grad_value_obj, value_and_grad_gradient_obj = value_and_grad_fn(raw_params)
    jit_value_obj, jit_gradient_obj = jax_module.jit(value_and_grad_fn)(raw_params)
    jacfwd_obj = jax_jacfwd(jax_module, value_function, raw_params)
    jacrev_obj = jax_jacrev(jax_module, value_function, raw_params)
    hessian_obj = jax_hessian(jax_module, value_function, raw_params)
    jvp_value_obj, jvp_tangent_obj = jax_jvp(jax_module, value_function, raw_params, raw_tangent)
    vjp_value_obj, pullback = jax_vjp(jax_module, value_function, raw_params)
    (vjp_cotangent_obj,) = pullback(jnp.asarray(1.0))
    vmap_value_and_grad = jax_module.vmap(value_and_grad_fn)
    vmap_values_obj, vmap_gradients_obj = vmap_value_and_grad(raw_batch)

    width = parameter_values.size
    flatten = operations.flatten_gradient
    gradient = flatten()(
        jax_module, "JAX grad Phase-QNode PyTree gradient", gradient_pytree_obj, width=width
    )
    value_and_grad_gradient = flatten()(
        jax_module,
        "JAX value_and_grad Phase-QNode PyTree gradient",
        value_and_grad_gradient_obj,
        width=width,
    )
    jit_gradient = flatten()(
        jax_module, "JAX jit Phase-QNode PyTree gradient", jit_gradient_obj, width=width
    )
    jacfwd_gradient = flatten()(
        jax_module, "JAX jacfwd Phase-QNode PyTree gradient", jacfwd_obj, width=width
    )
    jacrev_gradient = flatten()(
        jax_module, "JAX jacrev Phase-QNode PyTree gradient", jacrev_obj, width=width
    )
    hessian_matrix = operations.flatten_hessian()(
        jax_module,
        "JAX hessian Phase-QNode PyTree matrix",
        hessian_obj,
        leaf_shapes=leaf_shapes,
        leaf_sizes=leaf_sizes,
    )
    vjp_cotangent_gradient = flatten()(
        jax_module,
        "JAX vjp Phase-QNode PyTree cotangent gradient",
        vjp_cotangent_obj,
        width=width,
    )
    vmap_values = operations.parameter_vector()(
        "JAX vmap Phase-QNode PyTree values", vmap_values_obj, width=batch_values.shape[0]
    )
    vmap_gradients = operations.flatten_batched_gradient()(
        jax_module,
        "JAX vmap Phase-QNode PyTree gradients",
        vmap_gradients_obj,
        batch_size=batch_values.shape[0],
        width=width,
    )
    batch_parameter_shift_gradients = _batch_references(operations, circuit, batch_values)
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
    scalar = operations.scalar
    value_errors = (
        abs(
            scalar()("JAX value_and_grad Phase-QNode PyTree value", value_and_grad_value_obj)
            - parameter_shift.value
        ),
        abs(scalar()("JAX jit Phase-QNode PyTree value", jit_value_obj) - parameter_shift.value),
        abs(scalar()("JAX jvp Phase-QNode PyTree value", jvp_value_obj) - parameter_shift.value),
        abs(scalar()("JAX vjp Phase-QNode PyTree value", vjp_value_obj) - parameter_shift.value),
        abs(scalar()("JAX jvp Phase-QNode PyTree tangent", jvp_tangent_obj) - expected_jvp),
        _batch_value_error(operations, circuit, batch_values, vmap_values),
    )
    max_abs_transform_error = max(float(error) for error in value_errors)
    max_abs_hessian_symmetry_error = float(
        np.max(np.abs(hessian_matrix - hessian_matrix.T), initial=0.0)
    )
    passed = bool(
        max_abs_gradient_error <= tolerance
        and max_abs_transform_error <= tolerance
        and max_abs_hessian_symmetry_error <= tolerance
    )
    return PhaseJAXPhaseQNodePyTreeTransformResult(
        value=scalar()("JAX grad Phase-QNode PyTree value", value_function(raw_params)),
        gradient=gradient,
        gradient_pytree=gradient_pytree_obj,
        value_and_grad_value=scalar()(
            "JAX value_and_grad Phase-QNode PyTree value", value_and_grad_value_obj
        ),
        value_and_grad_gradient=value_and_grad_gradient,
        value_and_grad_gradient_pytree=value_and_grad_gradient_obj,
        jacfwd_gradient=jacfwd_gradient,
        jacrev_gradient=jacrev_gradient,
        hessian=hessian_matrix.copy(),
        hessian_pytree=hessian_obj,
        jvp_value=scalar()("JAX jvp Phase-QNode PyTree value", jvp_value_obj),
        jvp_tangent=scalar()("JAX jvp Phase-QNode PyTree tangent", jvp_tangent_obj),
        vjp_value=scalar()("JAX vjp Phase-QNode PyTree value", vjp_value_obj),
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
        tolerance=tolerance,
        passed=passed,
        native_framework_autodiff=True,
        host_callback=False,
        jit_value_and_grad=True,
        vmap_value_and_grad=True,
        transform_names=TRANSFORM_NAMES,
    )


def jax_sharding_transform_audit(
    jax_module: Any,
    jnp: Any,
    circuit: PhaseQNodeCircuit,
    params_batch: object,
    tolerance: float,
    value_function: Callable[[object], object],
    operations: JAXAuditOperations,
) -> PhaseJAXPhaseQNodeShardingTransformResult:
    """Map one value-and-gradient row per local device with ``pmap`` and compare.

    Parameters
    ----------
    jax_module, jnp
        Supplied JAX module and ``jax.numpy`` namespace.
    circuit
        Registered circuit; each row is refused with ``PhaseQNodeSupportError``
        if unsupported.
    params_batch
        Host batch with exactly one parameter row per local device.
    tolerance
        Admitted non-negative tolerance.
    value_function
        Differentiable ``parameters -> value`` of the circuit.
    operations
        Admission and device-description resolvers.

    Returns
    -------
    PhaseJAXPhaseQNodeShardingTransformResult
        Per-device values and gradients with their reference errors.

    Raises
    ------
    RuntimeError
        If the runtime reports no local device.
    ValueError
        If the batch is not two-dimensional or its rows do not match the devices.

    """
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
    parameter_batch = operations.parameter_batch()(
        "params_batch", parameter_batch, width=parameter_batch.shape[1]
    )
    for row in parameter_batch:
        _ensure_supported(operations, circuit, row)

    value_and_grad_fn = jax_module.value_and_grad(value_function)
    values_obj, gradients_obj = jax_module.pmap(value_and_grad_fn)(jnp.asarray(parameter_batch))
    values = operations.parameter_vector()(
        "JAX pmap Phase-QNode values", values_obj, width=local_device_count
    )
    gradients = operations.parameter_batch()(
        "JAX pmap Phase-QNode gradients", gradients_obj, width=parameter_batch.shape[1]
    )
    references = tuple(operations.parameter_shift()(circuit, row) for row in parameter_batch)
    parameter_shift_values = np.asarray([result.value for result in references], dtype=np.float64)
    parameter_shift_gradients = np.vstack([result.gradient for result in references]).astype(
        np.float64, copy=False
    )
    max_abs_value_error = float(np.max(np.abs(values - parameter_shift_values), initial=0.0))
    max_abs_gradient_error = float(
        np.max(np.abs(gradients - parameter_shift_gradients), initial=0.0)
    )
    passed = bool(max_abs_value_error <= tolerance and max_abs_gradient_error <= tolerance)
    sharding_mode = "single_device_pmap_smoke" if local_device_count == 1 else "multi_device_pmap"
    return PhaseJAXPhaseQNodeShardingTransformResult(
        values=values,
        gradients=gradients,
        parameter_shift_values=parameter_shift_values,
        parameter_shift_gradients=parameter_shift_gradients,
        batch_params=parameter_batch.copy(),
        batch_size=int(parameter_batch.shape[0]),
        local_device_count=local_device_count,
        device_descriptions=operations.local_devices()(jax_module, local_device_count),
        sharding_mode=sharding_mode,
        max_abs_value_error=max_abs_value_error,
        max_abs_gradient_error=max_abs_gradient_error,
        tolerance=tolerance,
        passed=passed,
        native_framework_autodiff=True,
        host_callback=False,
        pmapped=True,
    )


def jax_aot_export_audit(
    jax_module: Any,
    jnp: Any,
    export_module: Any,
    circuit: PhaseQNodeCircuit,
    parameter_values: FloatArray,
    tolerance: float,
    value_function: Callable[[object], object],
    operations: JAXAuditOperations,
) -> PhaseJAXPhaseQNodeAOTExportResult:
    """Lower, compile, export, serialise and deserialise the value route and compare.

    Parameters
    ----------
    jax_module, jnp
        Supplied JAX module and ``jax.numpy`` namespace.
    export_module
        Checked ``jax.export`` module.
    circuit
        Registered circuit; refused with ``PhaseQNodeSupportError`` if unsupported.
    parameter_values
        Admitted float64 parameters.
    tolerance
        Admitted non-negative tolerance.
    value_function
        ``parameters -> value`` of the circuit.
    operations
        Admission and export-version resolvers.

    Returns
    -------
    PhaseJAXPhaseQNodeAOTExportResult
        Compiled and deserialised values, export metadata and the verdict.

    Raises
    ------
    RuntimeError
        If a lowering, compiler metadata, compile or export platform step is
        unavailable.

    """
    _ensure_supported(operations, circuit, parameter_values)
    parameter_shift = operations.parameter_shift()(circuit, parameter_values)

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
    compiled_value = operations.scalar()(
        "JAX compiled Phase-QNode value",
        compiled_executable(jnp.asarray(parameter_values)),
    )

    exported = export_module.export(jax_module.jit(value_function))(shape_dtype)
    mlir_module = str(exported.mlir_module())
    serialized_blob = exported.serialize()
    rehydrated = export_module.deserialize(serialized_blob)
    deserialized_value = operations.scalar()(
        "JAX exported Phase-QNode value",
        rehydrated.call(jnp.asarray(parameter_values)),
    )
    export_platforms = tuple(str(platform) for platform in getattr(exported, "platforms", ()))
    if not export_platforms:
        raise RuntimeError("JAX export did not report lowering platforms")
    calling_convention_version = int(exported.calling_convention_version)
    minimum_supported = operations.export_version()(
        export_module, "minimum_supported_calling_convention_version"
    )
    maximum_supported = operations.export_version()(
        export_module, "maximum_supported_calling_convention_version"
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
        max_abs_value_error <= tolerance
        and not disabled_safety_checks
        and minimum_supported <= calling_convention_version <= maximum_supported
    )
    return PhaseJAXPhaseQNodeAOTExportResult(
        value=compiled_value,
        compiled_value=compiled_value,
        deserialized_value=deserialized_value,
        parameter_shift_value=parameter_shift.value,
        max_abs_value_error=max_abs_value_error,
        tolerance=tolerance,
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


def admitted_parameter_tree(
    jax_module: Any, name: str, params_pytree: object
) -> tuple[FloatArray, object, tuple[tuple[int, ...], ...], tuple[int, ...]]:
    """Flatten a numeric parameter PyTree with its reconstruction metadata.

    Parameters
    ----------
    jax_module
        Supplied JAX module providing ``tree_util``.
    name
        Name used in refusal messages.
    params_pytree
        Nested numeric parameter containers.

    Returns
    -------
    tuple[FloatArray, object, tuple[tuple[int, ...], ...], tuple[int, ...]]
        Parameter vector, tree definition, leaf shapes and leaf sizes.

    """
    return parameter_tree_vector(
        jax_module,
        name,
        params_pytree,
        numpy=lambda: np,
        scalar=lambda: float,
        integer=lambda: int,
        sequence=lambda: tuple,
        indexed=lambda: enumerate,
        invalid=lambda: ValueError,
        caster=lambda: cast,
        array_type=lambda: FloatArray,
    )


def _unflatten_vector(
    jax_module: Any,
    jnp: Any,
    vector: object,
    treedef: object,
    leaf_shapes: tuple[tuple[int, ...], ...],
    leaf_sizes: tuple[int, ...],
) -> object:
    return jax_vector_tree(
        jax_module, jnp, vector, treedef, leaf_shapes, leaf_sizes, paired=lambda: zip
    )


def _unflatten_batch(
    jax_module: Any,
    jnp: Any,
    batch: object,
    treedef: object,
    leaf_shapes: tuple[tuple[int, ...], ...],
    leaf_sizes: tuple[int, ...],
) -> object:
    return jax_batch_tree(
        jax_module, jnp, batch, treedef, leaf_shapes, leaf_sizes, paired=lambda: zip
    )


def _flatten_gradient(jax_module: Any, name: str, gradient: object, *, width: int) -> FloatArray:
    return gradient_tree_vector(
        jax_module,
        name,
        gradient,
        width=width,
        numpy=lambda: np,
        scalar=lambda: float,
        invalid=lambda: ValueError,
        caster=lambda: cast,
        array_type=lambda: FloatArray,
    )


def _flatten_batched_gradient(
    jax_module: Any, name: str, gradient: object, *, batch_size: int, width: int
) -> FloatArray:
    return batched_gradient_tree_matrix(
        jax_module,
        name,
        gradient,
        batch_size=batch_size,
        width=width,
        numpy=lambda: np,
        scalar=lambda: float,
        indexed=lambda: enumerate,
        invalid=lambda: ValueError,
        caster=lambda: cast,
        array_type=lambda: FloatArray,
    )


def _flatten_hessian(
    jax_module: Any,
    name: str,
    hessian: object,
    *,
    leaf_shapes: tuple[tuple[int, ...], ...],
    leaf_sizes: tuple[int, ...],
) -> FloatArray:
    return hessian_tree_matrix(
        jax_module,
        name,
        hessian,
        leaf_shapes=leaf_shapes,
        leaf_sizes=leaf_sizes,
        numpy=lambda: np,
        scalar=lambda: float,
        integer=lambda: int,
        length=lambda: len,
        paired=lambda: zip,
        accumulate=lambda: sum,
        invalid=lambda: ValueError,
        caster=lambda: cast,
        array_type=lambda: FloatArray,
    )


NATIVE_JAX_AUDIT_OPERATIONS = JAXAuditOperations(
    support_report=lambda: phase_qnode_support_report,
    parameter_shift=lambda: parameter_shift_phase_qnode_gradient,
    support_error=lambda: PhaseQNodeSupportError,
    parameter_vector=lambda: admitted_parameter_vector,
    parameter_batch=lambda: admitted_parameter_batch,
    scalar=lambda: admitted_scalar,
    unflatten_vector=lambda: _unflatten_vector,
    unflatten_batch=lambda: _unflatten_batch,
    flatten_gradient=lambda: _flatten_gradient,
    flatten_batched_gradient=lambda: _flatten_batched_gradient,
    flatten_hessian=lambda: _flatten_hessian,
    local_devices=lambda: jax_local_devices,
    export_version=lambda: jax_export_version,
)
"""Operations of the lower admission and PyTree leaves."""


def native_jax_value_function(jnp: Any, circuit: PhaseQNodeCircuit) -> Callable[[object], object]:
    """Return the engine's differentiable ``parameters -> value`` for a circuit.

    Parameters
    ----------
    jnp
        Supplied ``jax.numpy`` namespace.
    circuit
        Admitted registered circuit.

    Returns
    -------
    Callable[[object], object]
        Value function over a JAX parameter vector.

    """

    def value_function(raw_params: object) -> object:
        value, _state = jax_statevector_value_and_state(jnp, circuit, raw_params)
        return cast(object, value)

    return value_function


def native_jax_tree_value_function(
    jax_module: Any, jnp: Any, circuit: PhaseQNodeCircuit
) -> Callable[[object], object]:
    """Return the engine's differentiable ``parameter tree -> value`` for a circuit.

    Parameters
    ----------
    jax_module, jnp
        Supplied JAX module and ``jax.numpy`` namespace.
    circuit
        Admitted registered circuit.

    Returns
    -------
    Callable[[object], object]
        Value function over a JAX parameter PyTree.

    """

    def value_function(raw_tree: object) -> object:
        vector = jax_tree_vector(jax_module, jnp, raw_tree, length=lambda: len)
        value, _state = jax_statevector_value_and_state(jnp, circuit, vector)
        return cast(object, value)

    return value_function


def native_jax_native_transform_audit(
    jax_module: Any,
    jnp: Any,
    circuit: PhaseQNodeCircuit,
    params: object,
    *,
    tangent: object | None = None,
    batch_offsets: object | None = None,
    tolerance: float = 1e-6,
) -> PhaseJAXPhaseQNodeNativeTransformResult:
    """Run the native transform audit on the lower statevector engine.

    Parameters
    ----------
    jax_module, jnp
        Supplied JAX module and ``jax.numpy`` namespace; 64-bit mode is enabled.
    circuit
        Registered circuit.
    params
        Parameter values, admitted as a finite real vector.
    tangent, batch_offsets
        Optional JVP tangent and ``vmap`` offsets.
    tolerance
        Non-negative tolerance, admitted as on the JAX route.

    Returns
    -------
    PhaseJAXPhaseQNodeNativeTransformResult
        The audit result.

    """
    enable_jax_x64(jax_module)
    require_jax_transform_support(jax_module, surface="native transforms")
    tolerance_value = admitted_jax_tolerance(tolerance)
    parameter_values = admitted_parameter_vector("params", params)
    return jax_native_transform_audit(
        jax_module,
        jnp,
        circuit,
        parameter_values,
        tolerance_value,
        native_jax_value_function(jnp, circuit),
        NATIVE_JAX_AUDIT_OPERATIONS,
        tangent=tangent,
        batch_offsets=batch_offsets,
    )


def native_jax_pytree_transform_audit(
    jax_module: Any,
    jnp: Any,
    circuit: PhaseQNodeCircuit,
    params_pytree: object,
    *,
    tangent: object | None = None,
    batch_offsets: object | None = None,
    tolerance: float = 1e-6,
) -> PhaseJAXPhaseQNodePyTreeTransformResult:
    """Run the PyTree transform audit on the lower statevector engine.

    Parameters
    ----------
    jax_module, jnp
        Supplied JAX module and ``jax.numpy`` namespace; 64-bit mode is enabled.
    circuit
        Registered circuit.
    params_pytree
        Nested numeric parameter containers.
    tangent, batch_offsets
        Optional JVP tangent and ``vmap`` offsets.
    tolerance
        Non-negative tolerance, admitted as on the JAX route.

    Returns
    -------
    PhaseJAXPhaseQNodePyTreeTransformResult
        The audit result.

    """
    enable_jax_x64(jax_module)
    require_jax_pytree_support(jax_module)
    require_jax_transform_support(jax_module, surface="PyTree transforms")
    tolerance_value = admitted_jax_tolerance(tolerance)
    flattened = admitted_parameter_tree(jax_module, "params_pytree", params_pytree)
    return jax_pytree_transform_audit(
        jax_module,
        jnp,
        circuit,
        flattened,
        tolerance_value,
        native_jax_tree_value_function(jax_module, jnp, circuit),
        NATIVE_JAX_AUDIT_OPERATIONS,
        tangent=tangent,
        batch_offsets=batch_offsets,
    )


def native_jax_sharding_transform_audit(
    jax_module: Any,
    jnp: Any,
    circuit: PhaseQNodeCircuit,
    params_batch: object,
    *,
    tolerance: float = 1e-6,
) -> PhaseJAXPhaseQNodeShardingTransformResult:
    """Run the ``pmap`` sharding audit on the lower statevector engine.

    Parameters
    ----------
    jax_module, jnp
        Supplied JAX module and ``jax.numpy`` namespace; 64-bit mode is enabled.
    circuit
        Registered circuit.
    params_batch
        One parameter row per local device.
    tolerance
        Non-negative tolerance, admitted as on the JAX route.

    Returns
    -------
    PhaseJAXPhaseQNodeShardingTransformResult
        The audit result.

    """
    enable_jax_x64(jax_module)
    require_jax_pmap_support(jax_module)
    tolerance_value = admitted_jax_tolerance(tolerance)
    return jax_sharding_transform_audit(
        jax_module,
        jnp,
        circuit,
        params_batch,
        tolerance_value,
        native_jax_value_function(jnp, circuit),
        NATIVE_JAX_AUDIT_OPERATIONS,
    )


def native_jax_aot_export_audit(
    jax_module: Any,
    jnp: Any,
    circuit: PhaseQNodeCircuit,
    params: object,
    *,
    tolerance: float = 1e-6,
) -> PhaseJAXPhaseQNodeAOTExportResult:
    """Run the AOT lowering and ``jax.export`` audit on the lower statevector engine.

    Parameters
    ----------
    jax_module, jnp
        Supplied JAX module and ``jax.numpy`` namespace; 64-bit mode is enabled.
    circuit
        Registered circuit.
    params
        Parameter values, admitted as a finite real vector.
    tolerance
        Non-negative tolerance, admitted as on the JAX route.

    Returns
    -------
    PhaseJAXPhaseQNodeAOTExportResult
        The audit result, including real serialisation and deserialisation.

    """
    enable_jax_x64(jax_module)
    export_module = require_jax_aot_export_support(jax_module, importer=lambda: import_module)
    tolerance_value = admitted_jax_tolerance(tolerance)
    parameter_values = admitted_parameter_vector("params", params)
    return jax_aot_export_audit(
        jax_module,
        jnp,
        export_module,
        circuit,
        parameter_values,
        tolerance_value,
        native_jax_value_function(jnp, circuit),
        NATIVE_JAX_AUDIT_OPERATIONS,
    )


__all__ = [
    "NATIVE_JAX_AUDIT_OPERATIONS",
    "TRANSFORM_NAMES",
    "JAXAuditOperations",
    "admitted_parameter_tree",
    "enable_jax_x64",
    "jax_aot_export_audit",
    "jax_export_version",
    "jax_local_devices",
    "jax_native_transform_audit",
    "jax_pytree_transform_audit",
    "jax_sharding_transform_audit",
    "native_jax_aot_export_audit",
    "native_jax_native_transform_audit",
    "native_jax_pytree_transform_audit",
    "native_jax_sharding_transform_audit",
    "native_jax_tree_value_function",
    "native_jax_value_function",
    "require_jax_aot_export_support",
    "require_jax_pmap_support",
    "require_jax_pytree_support",
    "require_jax_transform_support",
]
