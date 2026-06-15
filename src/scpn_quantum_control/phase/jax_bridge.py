# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Phase JAX Bridge
"""Optional JAX interop for phase parameter-shift gradients."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..differentiable import (
    GradientResult,
    Parameter,
    ParameterShiftRule,
    value_and_parameter_shift_grad,
)
from .qnn_training import (
    parameter_shift_qnn_classifier_gradient,
    parameter_shift_qnn_classifier_loss,
)

FloatArray: TypeAlias = NDArray[np.float64]
ScalarObjective = Callable[[FloatArray], float]
GradientCallable = Callable[[FloatArray], ArrayLike]


@dataclass(frozen=True)
class PhaseJAXParameterShiftResult:
    """Result from the optional JAX phase parameter-shift bridge."""

    value: float
    gradient: FloatArray
    method: str
    evaluations: int
    jit_requested: bool
    jitted: bool
    host_callback: bool
    shift_terms: int = 1

    def to_dict(self) -> dict[str, object]:
        """Return JSON-serialisable JAX interop metadata."""
        return {
            "value": self.value,
            "gradient": self.gradient.copy(),
            "method": self.method,
            "evaluations": self.evaluations,
            "jit_requested": self.jit_requested,
            "jitted": self.jitted,
            "host_callback": self.host_callback,
            "shift_terms": self.shift_terms,
        }


@dataclass(frozen=True)
class PhaseJAXGradientAgreementResult:
    """Agreement report between SCPN and JAX-style gradient callables."""

    value: float
    scpn_gradient: FloatArray
    jax_gradient: FloatArray
    max_abs_error: float
    l2_error: float
    tolerance: float
    passed: bool
    evaluations: int
    method: str = "parameter_shift"
    shift_terms: int = 1

    def to_dict(self) -> dict[str, object]:
        """Return JSON-serialisable JAX gradient agreement metadata."""
        return {
            "value": self.value,
            "scpn_gradient": self.scpn_gradient.tolist(),
            "jax_gradient": self.jax_gradient.tolist(),
            "max_abs_error": self.max_abs_error,
            "l2_error": self.l2_error,
            "tolerance": self.tolerance,
            "passed": self.passed,
            "evaluations": self.evaluations,
            "method": self.method,
            "shift_terms": self.shift_terms,
        }


@dataclass(frozen=True)
class PhaseJAXNativeQNNGradientResult:
    """Native JAX autodiff agreement for the bounded phase-QNN classifier."""

    loss: float
    gradient: FloatArray
    parameter_shift_gradient: FloatArray
    max_abs_error: float
    l2_error: float
    tolerance: float
    passed: bool
    native_framework_autodiff: bool
    host_callback: bool
    jit_requested: bool
    jitted: bool
    method: str = "jax_native_bounded_phase_qnn_value_and_grad"

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready native JAX bounded-QNN gradient evidence."""
        return {
            "loss": self.loss,
            "gradient": self.gradient.tolist(),
            "parameter_shift_gradient": self.parameter_shift_gradient.tolist(),
            "max_abs_error": self.max_abs_error,
            "l2_error": self.l2_error,
            "tolerance": self.tolerance,
            "passed": self.passed,
            "native_framework_autodiff": self.native_framework_autodiff,
            "host_callback": self.host_callback,
            "jit_requested": self.jit_requested,
            "jitted": self.jitted,
            "method": self.method,
        }


@dataclass(frozen=True)
class PhaseJAXCustomVJPQNNGradientResult:
    """JAX custom-VJP evidence for the bounded phase-QNN classifier."""

    loss: float
    gradient: FloatArray
    parameter_shift_gradient: FloatArray
    max_abs_error: float
    l2_error: float
    tolerance: float
    passed: bool
    custom_vjp: bool
    native_framework_autodiff: bool
    host_callback: bool
    jit_requested: bool
    jitted: bool
    method: str = "jax_custom_vjp_bounded_phase_qnn_value_and_grad"

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready custom-VJP bounded-QNN gradient evidence."""
        return {
            "loss": self.loss,
            "gradient": self.gradient.tolist(),
            "parameter_shift_gradient": self.parameter_shift_gradient.tolist(),
            "max_abs_error": self.max_abs_error,
            "l2_error": self.l2_error,
            "tolerance": self.tolerance,
            "passed": self.passed,
            "custom_vjp": self.custom_vjp,
            "native_framework_autodiff": self.native_framework_autodiff,
            "host_callback": self.host_callback,
            "jit_requested": self.jit_requested,
            "jitted": self.jitted,
            "method": self.method,
        }


@dataclass(frozen=True)
class PhaseJAXJITCompatibilityResult:
    """Audited JAX JIT compatibility for bounded phase-QNN gradient routes."""

    passed: bool
    native_qnn_jitted: bool
    native_qnn_host_callback: bool
    custom_vjp_qnn_jitted: bool
    custom_vjp_qnn_host_callback: bool
    custom_vjp_registered: bool
    parameter_shift_jitted: bool
    parameter_shift_host_callback: bool
    max_abs_error: float
    tolerance: float
    unsupported_native_routes: tuple[str, ...]
    claim_boundary: str = "bounded_jax_jit_compatibility"

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready JAX JIT compatibility evidence."""
        return {
            "passed": self.passed,
            "native_qnn_jitted": self.native_qnn_jitted,
            "native_qnn_host_callback": self.native_qnn_host_callback,
            "custom_vjp_qnn_jitted": self.custom_vjp_qnn_jitted,
            "custom_vjp_qnn_host_callback": self.custom_vjp_qnn_host_callback,
            "custom_vjp_registered": self.custom_vjp_registered,
            "parameter_shift_jitted": self.parameter_shift_jitted,
            "parameter_shift_host_callback": self.parameter_shift_host_callback,
            "max_abs_error": self.max_abs_error,
            "tolerance": self.tolerance,
            "unsupported_native_routes": list(self.unsupported_native_routes),
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class PhaseJAXVMAPCompatibilityResult:
    """Audited JAX VMAP compatibility for bounded phase-QNN parameter batches."""

    passed: bool
    batch_size: int
    native_qnn_vmapped: bool
    native_qnn_host_callback: bool
    custom_vjp_qnn_vmapped: bool
    custom_vjp_qnn_host_callback: bool
    custom_vjp_registered: bool
    native_losses: FloatArray
    native_gradients: FloatArray
    custom_vjp_losses: FloatArray
    custom_vjp_gradients: FloatArray
    parameter_shift_losses: FloatArray
    parameter_shift_gradients: FloatArray
    max_abs_error: float
    tolerance: float
    unsupported_native_routes: tuple[str, ...]
    claim_boundary: str = "bounded_jax_vmap_compatibility"

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready JAX VMAP compatibility evidence."""
        return {
            "passed": self.passed,
            "batch_size": self.batch_size,
            "native_qnn_vmapped": self.native_qnn_vmapped,
            "native_qnn_host_callback": self.native_qnn_host_callback,
            "custom_vjp_qnn_vmapped": self.custom_vjp_qnn_vmapped,
            "custom_vjp_qnn_host_callback": self.custom_vjp_qnn_host_callback,
            "custom_vjp_registered": self.custom_vjp_registered,
            "native_losses": self.native_losses.tolist(),
            "native_gradients": self.native_gradients.tolist(),
            "custom_vjp_losses": self.custom_vjp_losses.tolist(),
            "custom_vjp_gradients": self.custom_vjp_gradients.tolist(),
            "parameter_shift_losses": self.parameter_shift_losses.tolist(),
            "parameter_shift_gradients": self.parameter_shift_gradients.tolist(),
            "max_abs_error": self.max_abs_error,
            "tolerance": self.tolerance,
            "unsupported_native_routes": list(self.unsupported_native_routes),
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class PhaseJAXShardingCompatibilityResult:
    """Audited JAX PMAP/sharding compatibility for bounded phase-QNN batches."""

    passed: bool
    batch_size: int
    local_device_count: int
    device_descriptions: tuple[str, ...]
    sharding_mode: str
    native_qnn_pmapped: bool
    native_qnn_host_callback: bool
    custom_vjp_qnn_pmapped: bool
    custom_vjp_qnn_host_callback: bool
    custom_vjp_registered: bool
    native_losses: FloatArray
    native_gradients: FloatArray
    custom_vjp_losses: FloatArray
    custom_vjp_gradients: FloatArray
    parameter_shift_losses: FloatArray
    parameter_shift_gradients: FloatArray
    max_abs_error: float
    tolerance: float
    unsupported_native_routes: tuple[str, ...]
    claim_boundary: str = "bounded_jax_pmap_sharding_compatibility"

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready JAX PMAP/sharding compatibility evidence."""
        return {
            "passed": self.passed,
            "batch_size": self.batch_size,
            "local_device_count": self.local_device_count,
            "device_descriptions": list(self.device_descriptions),
            "sharding_mode": self.sharding_mode,
            "native_qnn_pmapped": self.native_qnn_pmapped,
            "native_qnn_host_callback": self.native_qnn_host_callback,
            "custom_vjp_qnn_pmapped": self.custom_vjp_qnn_pmapped,
            "custom_vjp_qnn_host_callback": self.custom_vjp_qnn_host_callback,
            "custom_vjp_registered": self.custom_vjp_registered,
            "native_losses": self.native_losses.tolist(),
            "native_gradients": self.native_gradients.tolist(),
            "custom_vjp_losses": self.custom_vjp_losses.tolist(),
            "custom_vjp_gradients": self.custom_vjp_gradients.tolist(),
            "parameter_shift_losses": self.parameter_shift_losses.tolist(),
            "parameter_shift_gradients": self.parameter_shift_gradients.tolist(),
            "max_abs_error": self.max_abs_error,
            "tolerance": self.tolerance,
            "unsupported_native_routes": list(self.unsupported_native_routes),
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class PhaseJAXPyTreeCompatibilityResult:
    """Audited JAX PyTree parameter support for bounded phase-QNN gradients."""

    passed: bool
    leaf_count: int
    parameter_count: int
    leaf_shapes: tuple[tuple[int, ...], ...]
    parameter_vector: FloatArray
    native_qnn_pytree: bool
    native_qnn_host_callback: bool
    custom_vjp_qnn_pytree: bool
    custom_vjp_qnn_host_callback: bool
    custom_vjp_registered: bool
    native_loss: float
    custom_vjp_loss: float
    parameter_shift_loss: float
    native_gradient_vector: FloatArray
    custom_vjp_gradient_vector: FloatArray
    parameter_shift_gradient: FloatArray
    native_gradient_pytree: object
    custom_vjp_gradient_pytree: object
    max_abs_error: float
    tolerance: float
    unsupported_native_routes: tuple[str, ...]
    claim_boundary: str = "bounded_jax_pytree_compatibility"

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready JAX PyTree compatibility evidence."""
        return {
            "passed": self.passed,
            "leaf_count": self.leaf_count,
            "parameter_count": self.parameter_count,
            "leaf_shapes": [list(shape) for shape in self.leaf_shapes],
            "parameter_vector": self.parameter_vector.tolist(),
            "native_qnn_pytree": self.native_qnn_pytree,
            "native_qnn_host_callback": self.native_qnn_host_callback,
            "custom_vjp_qnn_pytree": self.custom_vjp_qnn_pytree,
            "custom_vjp_qnn_host_callback": self.custom_vjp_qnn_host_callback,
            "custom_vjp_registered": self.custom_vjp_registered,
            "native_loss": self.native_loss,
            "custom_vjp_loss": self.custom_vjp_loss,
            "parameter_shift_loss": self.parameter_shift_loss,
            "native_gradient_vector": self.native_gradient_vector.tolist(),
            "custom_vjp_gradient_vector": self.custom_vjp_gradient_vector.tolist(),
            "parameter_shift_gradient": self.parameter_shift_gradient.tolist(),
            "native_gradient_pytree": _json_ready_pytree(self.native_gradient_pytree),
            "custom_vjp_gradient_pytree": _json_ready_pytree(self.custom_vjp_gradient_pytree),
            "max_abs_error": self.max_abs_error,
            "tolerance": self.tolerance,
            "unsupported_native_routes": list(self.unsupported_native_routes),
            "claim_boundary": self.claim_boundary,
        }


def _load_jax() -> tuple[Any, Any]:
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:
        raise ImportError("JAX is unavailable; install scpn-quantum-control[jax]") from exc
    return jax, jnp


def is_phase_jax_available() -> bool:
    """Return whether the optional phase JAX adapter can import JAX."""
    try:
        _load_jax()
    except ImportError:
        return False
    return True


def _as_parameter_vector(name: str, values: object, *, width: int | None = None) -> FloatArray:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    if width is not None and vector.shape != (width,):
        raise ValueError(f"{name} must have shape ({width},), got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return cast(FloatArray, vector.astype(np.float64, copy=True))


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


def _json_ready_pytree(tree: object) -> object:
    if isinstance(tree, dict):
        return {str(key): _json_ready_pytree(value) for key, value in tree.items()}
    if isinstance(tree, tuple):
        return [_json_ready_pytree(value) for value in tree]
    if isinstance(tree, list):
        return [_json_ready_pytree(value) for value in tree]
    return np.asarray(tree, dtype=float).tolist()


def _as_scalar(name: str, value: object) -> float:
    array = np.asarray(value, dtype=float)
    if array.shape != ():
        raise ValueError(f"{name} must be a scalar")
    scalar = float(array)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def _as_non_negative_tolerance(value: float) -> float:
    tolerance = float(value)
    if tolerance < 0.0 or not np.isfinite(tolerance):
        raise ValueError("tolerance must be finite and non-negative")
    return tolerance


def _as_feature_matrix(features: ArrayLike) -> FloatArray:
    matrix = np.asarray(features, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("features must be a two-dimensional array")
    if matrix.shape[0] == 0:
        raise ValueError("features must contain at least one sample")
    if matrix.shape[1] == 0:
        raise ValueError("features must contain at least one feature column")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("features must contain only finite values")
    return matrix.astype(np.float64, copy=True)


def _as_label_vector(labels: ArrayLike, *, n_samples: int) -> FloatArray:
    vector = np.asarray(labels, dtype=float)
    if vector.ndim == 2 and vector.shape[1] == 1:
        vector = vector[:, 0]
    if vector.ndim != 1:
        raise ValueError("labels must be a one-dimensional array or a single-column matrix")
    if vector.shape != (n_samples,):
        raise ValueError("features and labels must have the same sample count")
    if not np.all(np.isfinite(vector)):
        raise ValueError("labels must contain only finite values")
    if np.any((vector < 0.0) | (vector > 1.0)):
        raise ValueError("labels must lie in the closed interval [0, 1]")
    return cast(FloatArray, vector.astype(np.float64, copy=True))


def _require_jax_callback_support(jax_module: Any) -> None:
    if not hasattr(jax_module, "jit"):
        raise RuntimeError("JAX JIT is unavailable in the active JAX module")
    if not hasattr(jax_module, "pure_callback"):
        raise RuntimeError("JAX pure_callback is required for JIT-wrapped host gradients")
    if not hasattr(jax_module, "ShapeDtypeStruct"):
        raise RuntimeError("JAX ShapeDtypeStruct is required for JIT-wrapped host gradients")


def _require_jax_custom_vjp_support(jax_module: Any) -> None:
    custom_vjp = getattr(jax_module, "custom_vjp", None)
    if not callable(custom_vjp):
        raise RuntimeError("JAX custom_vjp is required for the bounded-QNN custom VJP route")
    value_and_grad = getattr(jax_module, "value_and_grad", None)
    if not callable(value_and_grad):
        raise RuntimeError("JAX value_and_grad is required for the bounded-QNN custom VJP route")


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


def _require_jax_pytree_support(jax_module: Any) -> None:
    tree_util = getattr(jax_module, "tree_util", None)
    if tree_util is None:
        raise RuntimeError(
            "JAX PyTree tree_util support is required for bounded-QNN PyTree parameters"
        )
    if not callable(getattr(tree_util, "tree_flatten", None)):
        raise RuntimeError("JAX PyTree tree_flatten is required for bounded-QNN PyTree parameters")
    if not callable(getattr(tree_util, "tree_unflatten", None)):
        raise RuntimeError(
            "JAX PyTree tree_unflatten is required for bounded-QNN PyTree parameters"
        )
    value_and_grad = getattr(jax_module, "value_and_grad", None)
    if not callable(value_and_grad):
        raise RuntimeError("JAX value_and_grad is required for bounded-QNN PyTree parameters")


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


def _jax_bounded_qnn_loss(
    jnp: Any,
    feature_tensor: Any,
    label_tensor: Any,
    parameter_tensor: Any,
) -> Any:
    probabilities = 0.5 * (1.0 - jnp.cos(feature_tensor + parameter_tensor[None, :]))
    predictions = jnp.mean(probabilities, axis=1)
    residual = predictions - label_tensor
    return jnp.mean(residual * residual)


def _jax_bounded_qnn_gradient(
    jnp: Any,
    feature_tensor: Any,
    label_tensor: Any,
    parameter_tensor: Any,
) -> Any:
    feature_count = feature_tensor.shape[1]
    probabilities = 0.5 * (1.0 - jnp.cos(feature_tensor + parameter_tensor[None, :]))
    predictions = jnp.mean(probabilities, axis=1)
    residual = predictions - label_tensor
    probability_derivative = 0.5 * jnp.sin(feature_tensor + parameter_tensor[None, :])
    prediction_derivative = probability_derivative / feature_count
    return jnp.mean(2.0 * residual[:, None] * prediction_derivative, axis=0)


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
    parameter_values = _as_parameter_vector("values", values)
    jax_module, jnp = _load_jax()
    shift_terms = len((rule or ParameterShiftRule()).terms)
    last_result: GradientResult | None = None

    def evaluate(raw_values: object) -> tuple[np.ndarray, np.ndarray]:
        nonlocal last_result
        raw_array = _as_parameter_vector(
            "JAX callback values", raw_values, width=parameter_values.size
        )
        result: GradientResult = value_and_parameter_shift_grad(
            objective,
            raw_array,
            parameters=parameters,
            rule=rule,
        )
        last_result = result
        return (
            np.asarray(result.value, dtype=callback_dtype),
            result.gradient.astype(callback_dtype, copy=False),
        )

    parameter_tensor = jnp.asarray(parameter_values)
    callback_dtype = np.dtype(np.asarray(parameter_tensor).dtype)
    if jit:
        _require_jax_callback_support(jax_module)
        value_shape = jax_module.ShapeDtypeStruct((), callback_dtype)
        gradient_shape = jax_module.ShapeDtypeStruct(parameter_values.shape, callback_dtype)

        def wrapped(raw_values: object) -> tuple[object, object]:
            return cast(
                tuple[object, object],
                jax_module.pure_callback(
                    evaluate,
                    (value_shape, gradient_shape),
                    raw_values,
                ),
            )

        value_obj, gradient_obj = jax_module.jit(wrapped)(parameter_tensor)
        jitted = True
        host_callback = True
    else:
        value_obj, gradient_obj = evaluate(parameter_tensor)
        jitted = False
        host_callback = False

    gradient = _as_parameter_vector(
        "JAX parameter-shift gradient",
        gradient_obj,
        width=parameter_values.size,
    )
    if last_result is None:
        raise RuntimeError("JAX parameter-shift callback did not execute")
    return PhaseJAXParameterShiftResult(
        value=_as_scalar("JAX parameter-shift value", value_obj),
        gradient=gradient,
        method=last_result.method,
        evaluations=last_result.evaluations,
        jit_requested=jit,
        jitted=jitted,
        host_callback=host_callback,
        shift_terms=shift_terms,
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
    _load_jax()
    tolerance_value = _as_non_negative_tolerance(tolerance)
    parameter_values = _as_parameter_vector("values", values)
    scpn = value_and_parameter_shift_grad(
        objective,
        parameter_values,
        parameters=parameters,
        rule=rule,
    )
    shift_terms = len((rule or ParameterShiftRule()).terms)
    external_gradient = _as_parameter_vector(
        "JAX gradient",
        jax_gradient(parameter_values.copy()),
        width=parameter_values.size,
    )
    delta = scpn.gradient - external_gradient
    max_abs_error = float(np.max(np.abs(delta))) if delta.size else 0.0
    l2_error = float(np.linalg.norm(delta, ord=2))
    return PhaseJAXGradientAgreementResult(
        value=float(scpn.value),
        scpn_gradient=scpn.gradient.copy(),
        jax_gradient=external_gradient,
        max_abs_error=max_abs_error,
        l2_error=l2_error,
        tolerance=tolerance_value,
        passed=max_abs_error <= tolerance_value,
        evaluations=scpn.evaluations,
        method=scpn.method,
        shift_terms=shift_terms,
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
    jax_module, jnp = _load_jax()
    tolerance_value = _as_non_negative_tolerance(tolerance)
    feature_matrix = _as_feature_matrix(features)
    label_vector = _as_label_vector(labels, n_samples=feature_matrix.shape[0])
    parameter_values = _as_parameter_vector(
        "params",
        params,
        width=feature_matrix.shape[1],
    )
    feature_tensor = jnp.asarray(feature_matrix)
    label_tensor = jnp.asarray(label_vector)

    def loss_fn(raw_params: object) -> object:
        return _jax_bounded_qnn_loss(jnp, feature_tensor, label_tensor, jnp.asarray(raw_params))

    value_and_grad_fn = getattr(jax_module, "value_and_grad", None)
    if not callable(value_and_grad_fn):
        raise RuntimeError("JAX value_and_grad is required for native QNN autodiff")
    executable = value_and_grad_fn(loss_fn)
    jitted = False
    if jit:
        jit_fn = getattr(jax_module, "jit", None)
        if not callable(jit_fn):
            raise RuntimeError("JAX JIT is unavailable in the active JAX module")
        executable = jit_fn(executable)
        jitted = True

    loss_obj, gradient_obj = executable(jnp.asarray(parameter_values))
    gradient = _as_parameter_vector(
        "JAX native QNN gradient",
        gradient_obj,
        width=parameter_values.size,
    )
    reference_gradient = parameter_shift_qnn_classifier_gradient(
        feature_matrix,
        label_vector,
        parameter_values,
    )
    reference_loss = parameter_shift_qnn_classifier_loss(
        feature_matrix,
        label_vector,
        parameter_values,
    )
    loss = _as_scalar("JAX native QNN loss", loss_obj)
    if abs(loss - reference_loss) > max(tolerance_value, 1e-12):
        raise RuntimeError("JAX native QNN loss disagrees with parameter-shift loss")
    delta = gradient - reference_gradient
    max_abs_error = float(np.max(np.abs(delta))) if delta.size else 0.0
    l2_error = float(np.linalg.norm(delta, ord=2))
    return PhaseJAXNativeQNNGradientResult(
        loss=loss,
        gradient=gradient,
        parameter_shift_gradient=reference_gradient.copy(),
        max_abs_error=max_abs_error,
        l2_error=l2_error,
        tolerance=tolerance_value,
        passed=max_abs_error <= tolerance_value,
        native_framework_autodiff=True,
        host_callback=False,
        jit_requested=jit,
        jitted=jitted,
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
    jax_module, jnp = _load_jax()
    _require_jax_custom_vjp_support(jax_module)
    tolerance_value = _as_non_negative_tolerance(tolerance)
    feature_matrix = _as_feature_matrix(features)
    label_vector = _as_label_vector(labels, n_samples=feature_matrix.shape[0])
    parameter_values = _as_parameter_vector(
        "params",
        params,
        width=feature_matrix.shape[1],
    )
    feature_tensor = jnp.asarray(feature_matrix)
    label_tensor = jnp.asarray(label_vector)

    @jax_module.custom_vjp
    def loss_fn(raw_params: object) -> object:
        return _jax_bounded_qnn_loss(jnp, feature_tensor, label_tensor, jnp.asarray(raw_params))

    def loss_fwd(raw_params: object) -> tuple[object, object]:
        parameter_tensor = jnp.asarray(raw_params)
        return (
            _jax_bounded_qnn_loss(jnp, feature_tensor, label_tensor, parameter_tensor),
            parameter_tensor,
        )

    def loss_bwd(parameter_tensor: object, cotangent: object) -> tuple[object]:
        gradient = _jax_bounded_qnn_gradient(
            jnp,
            feature_tensor,
            label_tensor,
            jnp.asarray(parameter_tensor),
        )
        return (jnp.asarray(cotangent) * gradient,)

    loss_fn.defvjp(loss_fwd, loss_bwd)

    executable = jax_module.value_and_grad(loss_fn)
    jitted = False
    if jit:
        jit_fn = getattr(jax_module, "jit", None)
        if not callable(jit_fn):
            raise RuntimeError("JAX JIT is unavailable in the active JAX module")
        executable = jit_fn(executable)
        jitted = True

    loss_obj, gradient_obj = executable(jnp.asarray(parameter_values))
    gradient = _as_parameter_vector(
        "JAX custom VJP QNN gradient",
        gradient_obj,
        width=parameter_values.size,
    )
    reference_gradient = parameter_shift_qnn_classifier_gradient(
        feature_matrix,
        label_vector,
        parameter_values,
    )
    reference_loss = parameter_shift_qnn_classifier_loss(
        feature_matrix,
        label_vector,
        parameter_values,
    )
    loss = _as_scalar("JAX custom VJP QNN loss", loss_obj)
    if abs(loss - reference_loss) > max(tolerance_value, 1e-12):
        raise RuntimeError("JAX custom VJP QNN loss disagrees with parameter-shift loss")
    delta = gradient - reference_gradient
    max_abs_error = float(np.max(np.abs(delta))) if delta.size else 0.0
    l2_error = float(np.linalg.norm(delta, ord=2))
    return PhaseJAXCustomVJPQNNGradientResult(
        loss=loss,
        gradient=gradient,
        parameter_shift_gradient=reference_gradient.copy(),
        max_abs_error=max_abs_error,
        l2_error=l2_error,
        tolerance=tolerance_value,
        passed=max_abs_error <= tolerance_value,
        custom_vjp=True,
        native_framework_autodiff=True,
        host_callback=False,
        jit_requested=jit,
        jitted=jitted,
    )


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

    @jax_module.custom_vjp
    def custom_loss_fn(raw_params: object) -> object:
        return _jax_bounded_qnn_loss(jnp, feature_tensor, label_tensor, jnp.asarray(raw_params))

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

    custom_loss_fn.defvjp(custom_loss_fwd, custom_loss_bwd)

    native_losses_obj, native_gradients_obj = jax_module.vmap(
        jax_module.value_and_grad(native_loss_fn)
    )(jnp.asarray(parameter_batch))
    custom_losses_obj, custom_gradients_obj = jax_module.vmap(
        jax_module.value_and_grad(custom_loss_fn)
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

    @jax_module.custom_vjp
    def custom_loss_fn(raw_params: object) -> object:
        return _jax_bounded_qnn_loss(jnp, feature_tensor, label_tensor, jnp.asarray(raw_params))

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

    custom_loss_fn.defvjp(custom_loss_fwd, custom_loss_bwd)

    native_losses_obj, native_gradients_obj = jax_module.pmap(
        jax_module.value_and_grad(native_loss_fn)
    )(jnp.asarray(parameter_batch))
    custom_losses_obj, custom_gradients_obj = jax_module.pmap(
        jax_module.value_and_grad(custom_loss_fn)
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

    @jax_module.custom_vjp
    def custom_loss_fn(raw_tree: object) -> object:
        parameter_tensor = _jax_flatten_pytree(jax_module, jnp, raw_tree)
        return _jax_bounded_qnn_loss(jnp, feature_tensor, label_tensor, parameter_tensor)

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

    custom_loss_fn.defvjp(custom_loss_fwd, custom_loss_bwd)

    native_loss_obj, native_gradient_pytree = jax_module.value_and_grad(native_loss_fn)(
        params_pytree
    )
    custom_loss_obj, custom_gradient_pytree = jax_module.value_and_grad(custom_loss_fn)(
        params_pytree
    )
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


__all__ = [
    "PhaseJAXCustomVJPQNNGradientResult",
    "PhaseJAXGradientAgreementResult",
    "PhaseJAXJITCompatibilityResult",
    "PhaseJAXNativeQNNGradientResult",
    "PhaseJAXParameterShiftResult",
    "PhaseJAXPyTreeCompatibilityResult",
    "PhaseJAXShardingCompatibilityResult",
    "PhaseJAXVMAPCompatibilityResult",
    "check_jax_parameter_shift_agreement",
    "is_phase_jax_available",
    "jax_custom_vjp_qnn_value_and_grad",
    "jax_native_qnn_value_and_grad",
    "jax_parameter_shift_value_and_grad",
    "run_jax_jit_compatibility_audit",
    "run_jax_pytree_compatibility_audit",
    "run_jax_sharding_compatibility_audit",
    "run_jax_vmap_compatibility_audit",
]
