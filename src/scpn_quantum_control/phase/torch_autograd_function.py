# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — PyTorch Autograd Function Utilities
"""Bounded PyTorch ``autograd.Function`` utilities for phase-QNN gradients.

The surface in this module is deliberately narrow: it exposes a scalar loss for
the bounded phase-QNN classifier and a local audit proving that
``Tensor.backward()`` and a standard PyTorch optimizer consume the custom
backward rule. The backward rule is checked against the canonical SCPN
parameter-shift gradient and does not claim higher-order autograd, CUDA,
provider, hardware, arbitrary-simulator, isolated-benchmark, or performance
promotion.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .qnn_training import (
    parameter_shift_qnn_classifier_gradient,
    parameter_shift_qnn_classifier_loss,
)

FloatArray: TypeAlias = NDArray[np.float64]

TORCH_AUTOGRAD_FUNCTION_SCHEMA = (
    "scpn_quantum_control.phase.torch_bounded_qnn_autograd_function.v1"
)
TORCH_AUTOGRAD_FUNCTION_CLAIM_BOUNDARY = (
    "bounded PyTorch torch.autograd.Function route for the local phase-QNN "
    "classifier loss only; Tensor.backward and SGD optimizer integration are "
    "checked against SCPN parameter-shift references with no higher-order "
    "autograd, CUDA, provider, hardware, arbitrary-simulator, isolated "
    "benchmark, or performance claim"
)


@dataclass(frozen=True)
class PhaseTorchAutogradFunctionRoute:
    """One route in the bounded PyTorch autograd-function audit."""

    name: str
    status: str
    reason: str
    requires: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready autograd-function route metadata."""
        return {
            "name": self.name,
            "status": self.status,
            "reason": self.reason,
            "requires": list(self.requires),
        }


@dataclass(frozen=True)
class PhaseTorchAutogradFunctionResult:
    """Audit evidence for a bounded PyTorch custom autograd function."""

    matrix_schema: str
    feature_shape: tuple[int, int]
    gradient_shape: tuple[int, ...]
    loss: float
    reference_loss: float
    gradient: FloatArray
    parameter_shift_gradient: FloatArray
    optimizer_loss_before: float
    optimizer_loss_after: float
    optimizer_step_delta_norm: float
    max_abs_error: float
    l2_error: float
    tolerance: float
    torch_version: str
    torch_loss: Any
    torch_gradient: Any
    torch_parameter_shift_gradient: Any
    routes: tuple[PhaseTorchAutogradFunctionRoute, ...]
    custom_autograd_function_claim: bool = True
    higher_order_claim: bool = False
    provider_claim: bool = False
    hardware_claim: bool = False
    performance_claim: bool = False
    method: str = "torch_bounded_phase_qnn_autograd_function_audit"
    claim_boundary: str = TORCH_AUTOGRAD_FUNCTION_CLAIM_BOUNDARY

    @property
    def passed(self) -> bool:
        """Return whether the bounded local custom-autograd audit passed."""
        return (
            self.route_status("custom_autograd_backward") == "passed"
            and self.route_status("tensor_backward_integration") == "passed"
            and self.route_status("optimizer_step_integration") == "passed"
            and self.max_abs_error <= self.tolerance
            and all(route.status != "failed" for route in self.routes)
        )

    @property
    def open_gaps(self) -> tuple[str, ...]:
        """Return routes that remain blocked or failed."""
        return tuple(route.name for route in self.routes if route.status != "passed")

    def route_status(self, name: str) -> str:
        """Return the status for a named autograd-function route."""
        for route in self.routes:
            if route.name == name:
                return route.status
        raise KeyError(f"unknown PyTorch autograd-function route: {name}")

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready autograd-function audit evidence."""
        return {
            "matrix_schema": self.matrix_schema,
            "feature_shape": list(self.feature_shape),
            "gradient_shape": list(self.gradient_shape),
            "loss": self.loss,
            "reference_loss": self.reference_loss,
            "gradient": self.gradient.tolist(),
            "parameter_shift_gradient": self.parameter_shift_gradient.tolist(),
            "optimizer_loss_before": self.optimizer_loss_before,
            "optimizer_loss_after": self.optimizer_loss_after,
            "optimizer_step_delta_norm": self.optimizer_step_delta_norm,
            "max_abs_error": self.max_abs_error,
            "l2_error": self.l2_error,
            "tolerance": self.tolerance,
            "torch_version": self.torch_version,
            "torch_loss_type": type(self.torch_loss).__name__,
            "torch_gradient_type": type(self.torch_gradient).__name__,
            "torch_parameter_shift_gradient_type": type(
                self.torch_parameter_shift_gradient,
            ).__name__,
            "routes": {route.name: route.to_dict() for route in self.routes},
            "open_gaps": list(self.open_gaps),
            "passed": self.passed,
            "custom_autograd_function_claim": self.custom_autograd_function_claim,
            "higher_order_claim": self.higher_order_claim,
            "provider_claim": self.provider_claim,
            "hardware_claim": self.hardware_claim,
            "performance_claim": self.performance_claim,
            "method": self.method,
            "claim_boundary": self.claim_boundary,
        }


def torch_autograd_function_qnn_loss(
    features: ArrayLike | object,
    labels: ArrayLike | object,
    params: ArrayLike | object,
    *,
    tolerance: float = 1e-6,
) -> Any:
    """Return a scalar PyTorch loss with a bounded custom backward rule.

    Parameters
    ----------
    features:
        Real-valued feature matrix with shape ``(n_samples, n_parameters)``.
    labels:
        Binary label vector with shape ``(n_samples,)``.
    params:
        Trainable phase parameters. Passing a leaf PyTorch tensor with
        ``requires_grad=True`` allows callers to use ``loss.backward()`` and
        inspect ``params.grad``.
    tolerance:
        Non-negative absolute tolerance used to reject a custom backward rule
        that no longer matches the SCPN parameter-shift reference.

    Returns
    -------
    Any
        A scalar PyTorch tensor whose custom backward returns the bounded
        phase-QNN parameter-shift gradient for ``params`` and no gradients for
        ``features`` or ``labels``.
    """
    torch_module = _load_torch()
    feature_matrix, label_vector, parameter_values = _validate_inputs(features, labels, params)
    tolerance_value = _as_non_negative_tolerance(tolerance)
    parameter_tensor = _parameter_tensor(torch_module, params, parameter_values)
    feature_tensor = _torch_tensor_like(torch_module, feature_matrix, parameter_tensor)
    label_tensor = _torch_tensor_like(torch_module, label_vector, parameter_tensor)
    function_cls = _autograd_function_class(torch_module, tolerance=tolerance_value)
    return function_cls.apply(feature_tensor, label_tensor, parameter_tensor)


def run_torch_autograd_function_audit(
    *,
    features: ArrayLike,
    labels: ArrayLike,
    initial_params: ArrayLike | object,
    learning_rate: float = 0.05,
    tolerance: float = 1e-6,
) -> PhaseTorchAutogradFunctionResult:
    """Audit bounded PyTorch custom-autograd backward and optimizer integration.

    Parameters
    ----------
    features:
        Real-valued feature matrix with shape ``(n_samples, n_parameters)``.
    labels:
        Binary label vector with shape ``(n_samples,)``.
    initial_params:
        Initial trainable phase vector with one parameter per feature column.
    learning_rate:
        Positive SGD learning rate used for the local optimizer-integration
        evidence row.
    tolerance:
        Non-negative absolute tolerance for loss and gradient parity against
        SCPN parameter-shift references.

    Returns
    -------
    PhaseTorchAutogradFunctionResult
        JSON-ready evidence for local ``Tensor.backward()`` and optimizer-step
        integration plus explicit blocked route classifications.
    """
    torch_module = _load_torch()
    feature_matrix, label_vector, parameter_values = _validate_inputs(
        features,
        labels,
        initial_params,
        parameter_name="initial_params",
    )
    learning_rate_value = _as_positive_learning_rate(learning_rate)
    tolerance_value = _as_non_negative_tolerance(tolerance)

    trainable_params = _trainable_parameter_tensor(torch_module, parameter_values)
    torch_loss = torch_autograd_function_qnn_loss(
        feature_matrix,
        label_vector,
        trainable_params,
        tolerance=tolerance_value,
    )
    _torch_backward(torch_loss)
    torch_gradient = _require_tensor_gradient(trainable_params)
    gradient = _as_parameter_vector(
        "PyTorch autograd-function gradient",
        _values_to_numpy(torch_gradient),
        width=parameter_values.size,
    )
    reference_loss = parameter_shift_qnn_classifier_loss(
        feature_matrix,
        label_vector,
        parameter_values,
    )
    reference_gradient = _as_parameter_vector(
        "SCPN bounded phase-QNN parameter-shift gradient",
        parameter_shift_qnn_classifier_gradient(feature_matrix, label_vector, parameter_values),
        width=parameter_values.size,
    )
    max_abs_error = _max_abs_error(gradient, reference_gradient)
    l2_error = float(np.linalg.norm(gradient - reference_gradient))

    optimizer_params = _trainable_parameter_tensor(torch_module, parameter_values)
    optimizer = _torch_sgd_optimizer(
        torch_module,
        (optimizer_params,),
        learning_rate=learning_rate_value,
    )
    _torch_optimizer_zero_grad(optimizer)
    optimizer_loss_before_tensor = torch_autograd_function_qnn_loss(
        feature_matrix,
        label_vector,
        optimizer_params,
        tolerance=tolerance_value,
    )
    optimizer_loss_before = _scalar_to_float(optimizer_loss_before_tensor)
    _torch_backward(optimizer_loss_before_tensor)
    _torch_optimizer_step(optimizer)
    updated_params = _as_parameter_vector(
        "PyTorch autograd-function optimizer parameters",
        _values_to_numpy(optimizer_params),
        width=parameter_values.size,
    )
    optimizer_loss_after_tensor = torch_autograd_function_qnn_loss(
        feature_matrix,
        label_vector,
        optimizer_params,
        tolerance=tolerance_value,
    )
    optimizer_loss_after = _scalar_to_float(optimizer_loss_after_tensor)
    optimizer_step_delta_norm = float(np.linalg.norm(updated_params - parameter_values))

    routes = _classify_routes(
        gradient=gradient,
        reference_gradient=reference_gradient,
        optimizer_loss_before=optimizer_loss_before,
        optimizer_loss_after=optimizer_loss_after,
        optimizer_step_delta_norm=optimizer_step_delta_norm,
        tolerance=tolerance_value,
    )
    return PhaseTorchAutogradFunctionResult(
        matrix_schema=TORCH_AUTOGRAD_FUNCTION_SCHEMA,
        feature_shape=(int(feature_matrix.shape[0]), int(feature_matrix.shape[1])),
        gradient_shape=gradient.shape,
        loss=_scalar_to_float(torch_loss),
        reference_loss=float(reference_loss),
        gradient=gradient,
        parameter_shift_gradient=reference_gradient,
        optimizer_loss_before=optimizer_loss_before,
        optimizer_loss_after=optimizer_loss_after,
        optimizer_step_delta_norm=optimizer_step_delta_norm,
        max_abs_error=max_abs_error,
        l2_error=l2_error,
        tolerance=tolerance_value,
        torch_version=str(getattr(torch_module, "__version__", "unknown")),
        torch_loss=torch_loss,
        torch_gradient=torch_gradient,
        torch_parameter_shift_gradient=_torch_tensor_like(
            torch_module,
            reference_gradient,
            trainable_params,
        ),
        routes=routes,
    )


def _autograd_function_class(torch_module: Any, *, tolerance: float) -> Any:
    """Return a custom Function class bound to one checked tolerance."""
    function_base = _torch_autograd_function(torch_module)

    class _BoundedPhaseQNNAutogradFunction(function_base):  # type: ignore[misc, valid-type]
        @staticmethod
        def forward(
            ctx: Any, feature_tensor: Any, label_tensor: Any, parameter_tensor: Any
        ) -> Any:
            feature_matrix, label_vector, parameter_values = _validate_inputs(
                feature_tensor,
                label_tensor,
                parameter_tensor,
            )
            loss = parameter_shift_qnn_classifier_loss(
                feature_matrix,
                label_vector,
                parameter_values,
            )
            reference_gradient = _as_parameter_vector(
                "SCPN bounded phase-QNN parameter-shift gradient",
                parameter_shift_qnn_classifier_gradient(
                    feature_matrix,
                    label_vector,
                    parameter_values,
                ),
                width=parameter_values.size,
            )
            analytic_gradient = _bounded_analytic_qnn_gradient(
                feature_matrix,
                label_vector,
                parameter_values,
            )
            max_abs_error = _max_abs_error(analytic_gradient, reference_gradient)
            if max_abs_error > tolerance:
                raise RuntimeError(
                    "PyTorch custom autograd gradient diverged from the SCPN "
                    "parameter-shift reference",
                )
            set_materialize_grads = getattr(ctx, "set_materialize_grads", None)
            if callable(set_materialize_grads):
                set_materialize_grads(False)
            ctx.save_for_backward(
                _torch_tensor_like(torch_module, analytic_gradient, parameter_tensor)
            )
            return _torch_tensor_like(torch_module, loss, parameter_tensor)

        @staticmethod
        def backward(ctx: Any, grad_output: Any) -> tuple[None, None, Any]:
            (gradient_tensor,) = ctx.saved_tensors
            return None, None, gradient_tensor * grad_output

    return _BoundedPhaseQNNAutogradFunction


def _classify_routes(
    *,
    gradient: FloatArray,
    reference_gradient: FloatArray,
    optimizer_loss_before: float,
    optimizer_loss_after: float,
    optimizer_step_delta_norm: float,
    tolerance: float,
) -> tuple[PhaseTorchAutogradFunctionRoute, ...]:
    """Classify local and blocked autograd-function routes."""
    gradient_error = _max_abs_error(gradient, reference_gradient)
    gradient_shape_ok = gradient.shape == reference_gradient.shape and np.all(
        np.isfinite(gradient)
    )
    gradient_passed = gradient_error <= tolerance
    optimizer_passed = (
        optimizer_step_delta_norm > 0.0
        and np.isfinite(optimizer_loss_before)
        and np.isfinite(optimizer_loss_after)
        and optimizer_loss_after <= optimizer_loss_before + tolerance
    )
    return (
        PhaseTorchAutogradFunctionRoute(
            name="custom_autograd_backward",
            status="passed" if gradient_passed else "failed",
            reason="custom backward matches the SCPN parameter-shift gradient",
            requires=() if gradient_passed else ("matching_parameter_shift_gradient",),
        ),
        PhaseTorchAutogradFunctionRoute(
            name="tensor_backward_integration",
            status="passed" if gradient_shape_ok else "failed",
            reason="Tensor.backward populated a finite gradient with the parameter shape",
            requires=() if gradient_shape_ok else ("finite_parameter_tensor_grad",),
        ),
        PhaseTorchAutogradFunctionRoute(
            name="optimizer_step_integration",
            status="passed" if optimizer_passed else "failed",
            reason="torch.optim.SGD consumed the custom backward and produced a loss-safe update",
            requires=() if optimizer_passed else ("nonzero_optimizer_step", "nonincreasing_loss"),
        ),
        PhaseTorchAutogradFunctionRoute(
            name="higher_order_autograd",
            status="blocked",
            reason="the bounded custom backward returns a checked analytic tensor, not a higher-order transformable graph",
            requires=("higher_order_transformable_backward_rule",),
        ),
        PhaseTorchAutogradFunctionRoute(
            name="cuda_autograd_function",
            status="blocked",
            reason="CUDA custom-autograd execution requires compatible device artefacts",
            requires=("compatible_cuda_autograd_function_artifact",),
        ),
        PhaseTorchAutogradFunctionRoute(
            name="provider_hardware_autograd_function",
            status="blocked",
            reason="provider and hardware autograd routes require live-ticketed execution evidence",
            requires=("provider_job_policy", "hardware_execution_ticket"),
        ),
        PhaseTorchAutogradFunctionRoute(
            name="isolated_benchmark_autograd_function",
            status="blocked",
            reason="performance promotion requires isolated benchmark artefacts",
            requires=("isolated_affinity_autograd_function_benchmark",),
        ),
        PhaseTorchAutogradFunctionRoute(
            name="arbitrary_simulator_autograd_function",
            status="blocked",
            reason="the route covers the bounded phase-QNN classifier only",
            requires=("arbitrary_simulator_autograd_function_lowering",),
        ),
    )


def _validate_inputs(
    features: object,
    labels: object,
    params: object,
    *,
    parameter_name: str = "params",
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Validate bounded phase-QNN feature, label, and parameter arrays."""
    feature_matrix = _as_feature_matrix(_values_to_numpy(features))
    label_vector = _as_label_vector(_values_to_numpy(labels), n_samples=feature_matrix.shape[0])
    parameter_values = _as_parameter_vector(
        parameter_name,
        _values_to_numpy(params),
        width=feature_matrix.shape[1],
    )
    return feature_matrix, label_vector, parameter_values


def _bounded_analytic_qnn_gradient(
    feature_matrix: FloatArray,
    label_vector: FloatArray,
    parameter_values: FloatArray,
) -> FloatArray:
    """Return the exact bounded phase-QNN MSE gradient."""
    shifted = feature_matrix + parameter_values[None, :]
    probabilities = 0.5 * (1.0 - np.cos(shifted))
    predictions = np.mean(probabilities, axis=1)
    residual = predictions - label_vector
    scale = 1.0 / float(feature_matrix.shape[1])
    gradient = (2.0 / float(feature_matrix.shape[0])) * np.sum(
        residual[:, None] * (0.5 * np.sin(shifted) * scale),
        axis=0,
    )
    return _as_parameter_vector("bounded phase-QNN analytic gradient", gradient)


def _load_torch() -> Any:
    """Import PyTorch or raise the optional-dependency error used by this surface."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch is unavailable; install scpn-quantum-control[torch]") from exc
    return torch


def _torch_autograd_function(torch_module: Any) -> Any:
    """Return ``torch.autograd.Function`` from an imported PyTorch module."""
    autograd = getattr(torch_module, "autograd", None)
    function = getattr(autograd, "Function", None)
    apply = getattr(function, "apply", None)
    if autograd is None or function is None or not callable(apply):
        raise RuntimeError("PyTorch module does not expose torch.autograd.Function.apply")
    return function


def _torch_tensor_like(torch_module: Any, values: object, like: object) -> Any:
    """Create a PyTorch tensor on the same device and dtype as a reference tensor."""
    new_tensor = getattr(like, "new_tensor", None)
    if callable(new_tensor):
        return new_tensor(values)
    dtype = getattr(torch_module, "float64", None)
    as_tensor = getattr(torch_module, "as_tensor", None)
    if callable(as_tensor):
        if dtype is None:
            return as_tensor(values)
        return as_tensor(values, dtype=dtype)
    tensor = getattr(torch_module, "tensor", None)
    if callable(tensor):
        if dtype is None:
            return tensor(values)
        return tensor(values, dtype=dtype)
    raise RuntimeError("PyTorch module does not expose as_tensor or tensor")


def _parameter_tensor(torch_module: Any, params: object, parameter_values: FloatArray) -> Any:
    """Return a parameter tensor, preserving caller-owned PyTorch tensors."""
    if _looks_like_torch_tensor(params):
        return params
    return _torch_tensor_like(
        torch_module, parameter_values, _torch_tensor_like(torch_module, 0.0, 0.0)
    )


def _trainable_parameter_tensor(torch_module: Any, values: FloatArray) -> Any:
    """Return a leaf trainable PyTorch parameter tensor for audit execution."""
    tensor = _torch_tensor_like(torch_module, values, _torch_tensor_like(torch_module, 0.0, 0.0))
    detach = getattr(tensor, "detach", None)
    if callable(detach):
        tensor = detach()
    clone = getattr(tensor, "clone", None)
    if callable(clone):
        tensor = clone()
    requires_grad = getattr(tensor, "requires_grad_", None)
    if not callable(requires_grad):
        raise RuntimeError("PyTorch tensor does not expose requires_grad_")
    return requires_grad(True)


def _looks_like_torch_tensor(value: object) -> bool:
    """Return true when an object exposes the tensor methods this route needs."""
    return callable(getattr(value, "detach", None)) and callable(
        getattr(value, "new_tensor", None)
    )


def _values_to_numpy(values: object) -> FloatArray:
    """Convert tensors or array-like objects into a finite float64 NumPy array."""
    candidate = values
    detach = getattr(candidate, "detach", None)
    if callable(detach):
        candidate = detach()
    cpu = getattr(candidate, "cpu", None)
    if callable(cpu):
        candidate = cpu()
    numpy_method = getattr(candidate, "numpy", None)
    if callable(numpy_method):
        candidate = numpy_method()
    array = np.asarray(candidate, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError("values must contain only finite entries")
    return array.astype(np.float64, copy=True)


def _as_feature_matrix(values: object) -> FloatArray:
    """Return a validated feature matrix."""
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("features must be a two-dimensional array")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("features must not be empty")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("features must contain only finite values")
    return matrix.astype(np.float64, copy=True)


def _as_label_vector(values: object, *, n_samples: int) -> FloatArray:
    """Return a validated label vector."""
    vector = np.asarray(values, dtype=float)
    if vector.ndim == 2 and vector.shape[1] == 1:
        vector = vector[:, 0]
    if vector.ndim != 1:
        raise ValueError("labels must be one-dimensional or a single-column matrix")
    if vector.shape != (n_samples,):
        raise ValueError(f"labels must have shape ({n_samples},), got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError("labels must contain only finite values")
    if np.any((vector < 0.0) | (vector > 1.0)):
        raise ValueError("labels must lie in the closed interval [0, 1]")
    return vector.astype(np.float64, copy=True)


def _as_parameter_vector(
    name: str,
    values: object,
    *,
    width: int | None = None,
) -> FloatArray:
    """Return a validated parameter vector."""
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    if width is not None and vector.shape != (width,):
        raise ValueError(f"{name} must have shape ({width},), got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector.astype(np.float64, copy=True)


def _as_non_negative_tolerance(value: float) -> float:
    """Return a finite non-negative tolerance."""
    tolerance = float(value)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be a finite non-negative number")
    return tolerance


def _as_positive_learning_rate(value: float) -> float:
    """Return a finite positive learning rate."""
    learning_rate = float(value)
    if not np.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("learning_rate must be a finite positive number")
    return learning_rate


def _torch_backward(tensor: object) -> None:
    """Call ``Tensor.backward`` with a targeted error when unavailable."""
    backward = getattr(tensor, "backward", None)
    if not callable(backward):
        raise RuntimeError("PyTorch scalar loss does not expose backward")
    backward()


def _require_tensor_gradient(tensor: object) -> Any:
    """Return ``tensor.grad`` or fail closed when backward did not populate it."""
    gradient = getattr(tensor, "grad", None)
    if gradient is None:
        raise RuntimeError("PyTorch backward did not populate params.grad")
    return gradient


def _torch_sgd_optimizer(
    torch_module: Any,
    parameters: Iterable[object],
    *,
    learning_rate: float,
) -> Any:
    """Build a ``torch.optim.SGD`` optimizer for audit integration evidence."""
    optim = getattr(torch_module, "optim", None)
    sgd = getattr(optim, "SGD", None)
    if optim is None or not callable(sgd):
        raise RuntimeError("PyTorch module does not expose torch.optim.SGD")
    return sgd(list(parameters), lr=learning_rate)


def _torch_optimizer_zero_grad(optimizer: object) -> None:
    """Clear gradients through a PyTorch optimizer."""
    zero_grad = getattr(optimizer, "zero_grad", None)
    if not callable(zero_grad):
        raise RuntimeError("PyTorch optimizer does not expose zero_grad")
    zero_grad()


def _torch_optimizer_step(optimizer: object) -> None:
    """Apply a PyTorch optimizer step."""
    step = getattr(optimizer, "step", None)
    if not callable(step):
        raise RuntimeError("PyTorch optimizer does not expose step")
    step()


def _scalar_to_float(value: object) -> float:
    """Convert a scalar tensor or array-like object to a finite float."""
    array = np.asarray(_values_to_numpy(value), dtype=float)
    if array.shape not in ((), (1,)):
        raise ValueError(f"PyTorch scalar value must be scalar-like, got {array.shape}")
    return float(array.reshape(-1)[0])


def _max_abs_error(left: FloatArray, right: FloatArray) -> float:
    """Return maximum absolute error, preserving shape-mismatch failures."""
    if left.shape != right.shape:
        return float("inf")
    if left.size == 0:
        return 0.0
    return float(np.max(np.abs(left - right)))


__all__ = [
    "PhaseTorchAutogradFunctionResult",
    "PhaseTorchAutogradFunctionRoute",
    "TORCH_AUTOGRAD_FUNCTION_CLAIM_BOUNDARY",
    "TORCH_AUTOGRAD_FUNCTION_SCHEMA",
    "run_torch_autograd_function_audit",
    "torch_autograd_function_qnn_loss",
]
