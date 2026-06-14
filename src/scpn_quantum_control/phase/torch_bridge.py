# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Phase PyTorch Bridge
"""Optional PyTorch interop for phase parameter-shift gradients."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias

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
AutogradAuditRecord: TypeAlias = tuple[float, FloatArray, FloatArray, float, float, bool]


@dataclass(frozen=True)
class PhaseTorchParameterShiftResult:
    """Result from the optional PyTorch phase parameter-shift bridge."""

    value: float
    gradient: FloatArray
    torch_value: Any
    torch_gradient: Any
    method: str
    evaluations: int
    host_boundary: bool
    shift_terms: int = 1

    def to_dict(self) -> dict[str, object]:
        """Return JSON-serialisable PyTorch interop metadata."""
        return {
            "value": self.value,
            "gradient": self.gradient.copy(),
            "method": self.method,
            "evaluations": self.evaluations,
            "host_boundary": self.host_boundary,
            "shift_terms": self.shift_terms,
            "torch_value_type": type(self.torch_value).__name__,
            "torch_gradient_type": type(self.torch_gradient).__name__,
        }


@dataclass(frozen=True)
class PhaseTorchQNNGradientResult:
    """Tensor-ready bounded phase-QNN gradient evidence for PyTorch workflows."""

    loss: float
    gradient: FloatArray
    parameter_shift_gradient: FloatArray
    torch_loss: Any
    torch_gradient: Any
    torch_parameter_shift_gradient: Any
    max_abs_error: float
    l2_error: float
    tolerance: float
    passed: bool
    method: str = "torch_bounded_phase_qnn_analytic_value_and_grad"
    host_boundary: bool = False
    native_framework_autodiff: bool = False
    analytic_framework_gradient: bool = True

    def to_dict(self) -> dict[str, object]:
        """Return JSON-serialisable PyTorch bounded-QNN gradient metadata."""
        return {
            "loss": self.loss,
            "gradient": self.gradient.copy(),
            "parameter_shift_gradient": self.parameter_shift_gradient.copy(),
            "max_abs_error": self.max_abs_error,
            "l2_error": self.l2_error,
            "tolerance": self.tolerance,
            "passed": self.passed,
            "method": self.method,
            "host_boundary": self.host_boundary,
            "native_framework_autodiff": self.native_framework_autodiff,
            "analytic_framework_gradient": self.analytic_framework_gradient,
            "torch_loss_type": type(self.torch_loss).__name__,
            "torch_gradient_type": type(self.torch_gradient).__name__,
            "torch_parameter_shift_gradient_type": type(
                self.torch_parameter_shift_gradient,
            ).__name__,
        }


@dataclass(frozen=True)
class PhaseTorchAutogradQNNGradientResult:
    """Bounded phase-QNN gradient evidence from a PyTorch custom autograd function."""

    loss: float
    gradient: FloatArray
    parameter_shift_gradient: FloatArray
    torch_loss: Any
    torch_gradient: Any
    torch_parameter_shift_gradient: Any
    max_abs_error: float
    l2_error: float
    tolerance: float
    passed: bool
    method: str = "torch_bounded_phase_qnn_custom_autograd_function"
    host_boundary: bool = False
    native_framework_autodiff: bool = True
    custom_autograd_function: bool = True
    analytic_framework_gradient: bool = False

    def to_dict(self) -> dict[str, object]:
        """Return JSON-serialisable PyTorch custom-autograd QNN metadata."""
        return {
            "loss": self.loss,
            "gradient": self.gradient.copy(),
            "parameter_shift_gradient": self.parameter_shift_gradient.copy(),
            "max_abs_error": self.max_abs_error,
            "l2_error": self.l2_error,
            "tolerance": self.tolerance,
            "passed": self.passed,
            "method": self.method,
            "host_boundary": self.host_boundary,
            "native_framework_autodiff": self.native_framework_autodiff,
            "custom_autograd_function": self.custom_autograd_function,
            "analytic_framework_gradient": self.analytic_framework_gradient,
            "torch_loss_type": type(self.torch_loss).__name__,
            "torch_gradient_type": type(self.torch_gradient).__name__,
            "torch_parameter_shift_gradient_type": type(
                self.torch_parameter_shift_gradient,
            ).__name__,
        }


def _load_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch is unavailable; install scpn-quantum-control[torch]") from exc
    return torch


def is_phase_torch_available() -> bool:
    """Return whether the optional phase PyTorch bridge can import PyTorch."""
    try:
        _load_torch()
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
    return vector.astype(np.float64, copy=True)


def _as_feature_matrix(features: ArrayLike) -> FloatArray:
    matrix = np.asarray(features, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("features must be a two-dimensional array")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("features must not be empty")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("features must contain only finite values")
    return matrix.astype(np.float64, copy=True)


def _as_label_vector(labels: ArrayLike, *, n_samples: int) -> FloatArray:
    vector = np.asarray(labels, dtype=float)
    if vector.ndim != 1:
        raise ValueError("labels must be a one-dimensional array")
    if vector.shape != (n_samples,):
        raise ValueError(f"labels must have shape ({n_samples},), got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError("labels must contain only finite values")
    return vector.astype(np.float64, copy=True)


def _as_non_negative_tolerance(value: float) -> float:
    tolerance = float(value)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be a non-negative finite float")
    return tolerance


def _torch_values_to_numpy(values: object) -> FloatArray:
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
    return _as_parameter_vector("values", candidate)


def _torch_tensor(torch_module: Any, values: object) -> Any:
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


def _torch_autograd_function(torch_module: Any) -> Any:
    autograd = getattr(torch_module, "autograd", None)
    function = getattr(autograd, "Function", None)
    apply = getattr(function, "apply", None)
    if autograd is None or function is None or not callable(apply):
        raise RuntimeError("PyTorch module does not expose torch.autograd.Function.apply")
    return function


def _torch_autograd_grad(torch_module: Any) -> Any:
    autograd = getattr(torch_module, "autograd", None)
    grad = getattr(autograd, "grad", None)
    if not callable(grad):
        raise RuntimeError("PyTorch module does not expose torch.autograd.grad")
    return grad


def _torch_trainable_tensor(torch_module: Any, values: FloatArray) -> Any:
    tensor = _torch_tensor(torch_module, values)
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


def _bounded_qnn_loss_gradient_reference(
    features: ArrayLike,
    labels: ArrayLike,
    params: ArrayLike | object,
    *,
    tolerance: float,
) -> tuple[FloatArray, FloatArray, FloatArray, float, float, float, bool]:
    feature_matrix = _as_feature_matrix(features)
    label_vector = _as_label_vector(labels, n_samples=feature_matrix.shape[0])
    parameter_values = _as_parameter_vector("params", params)
    if parameter_values.shape != (feature_matrix.shape[1],):
        raise ValueError(
            "params width must match feature width: "
            f"{parameter_values.shape[0]} != {feature_matrix.shape[1]}",
        )
    tolerance_value = _as_non_negative_tolerance(tolerance)

    shifted = feature_matrix + parameter_values[None, :]
    probabilities = 0.5 * (1.0 - np.cos(shifted))
    predictions = np.mean(probabilities, axis=1)
    residual = predictions - label_vector
    loss = float(np.mean(residual * residual))
    scale = 1.0 / float(feature_matrix.shape[1])
    gradient = (2.0 / float(feature_matrix.shape[0])) * np.sum(
        residual[:, None] * (0.5 * np.sin(shifted) * scale),
        axis=0,
    )
    gradient = _as_parameter_vector(
        "PyTorch bounded phase-QNN gradient",
        gradient,
        width=parameter_values.size,
    )

    reference_loss = parameter_shift_qnn_classifier_loss(
        feature_matrix,
        label_vector,
        parameter_values,
    )
    if abs(loss - reference_loss) > tolerance_value:
        raise RuntimeError(
            "PyTorch bounded phase-QNN tensor loss disagrees with SCPN "
            f"parameter-shift loss: {loss} != {reference_loss}",
        )
    reference_gradient = parameter_shift_qnn_classifier_gradient(
        feature_matrix,
        label_vector,
        parameter_values,
    )
    reference_gradient = _as_parameter_vector(
        "SCPN bounded phase-QNN parameter-shift gradient",
        reference_gradient,
        width=parameter_values.size,
    )
    delta = gradient - reference_gradient
    max_abs_error = float(np.max(np.abs(delta))) if delta.size else 0.0
    l2_error = float(np.linalg.norm(delta))
    passed = bool(max_abs_error <= tolerance_value)
    return (
        np.asarray(loss, dtype=np.float64),
        gradient,
        reference_gradient,
        max_abs_error,
        l2_error,
        tolerance_value,
        passed,
    )


def torch_parameter_shift_value_and_grad(
    objective: ScalarObjective,
    values: ArrayLike | object,
    *,
    parameters: Sequence[Parameter] | None = None,
    rule: ParameterShiftRule | None = None,
) -> PhaseTorchParameterShiftResult:
    """Return phase parameter-shift value and gradient as NumPy and PyTorch tensors.

    This is a host-boundary bridge. The quantum expectation is evaluated through
    SCPN's deterministic parameter-shift rule and the result is converted to
    PyTorch tensors for framework pipelines. It does not claim native PyTorch
    autograd through a quantum simulator.
    """
    torch_module = _load_torch()
    parameter_values = _torch_values_to_numpy(values)
    result: GradientResult = value_and_parameter_shift_grad(
        objective,
        parameter_values,
        parameters=parameters,
        rule=rule,
    )
    shift_terms = len((rule or ParameterShiftRule()).terms)
    gradient = _as_parameter_vector(
        "PyTorch parameter-shift gradient",
        result.gradient,
        width=parameter_values.size,
    )
    return PhaseTorchParameterShiftResult(
        value=float(result.value),
        gradient=gradient,
        torch_value=_torch_tensor(torch_module, np.asarray(result.value, dtype=np.float64)),
        torch_gradient=_torch_tensor(torch_module, gradient),
        method=result.method,
        evaluations=result.evaluations,
        host_boundary=True,
        shift_terms=shift_terms,
    )


def torch_bounded_qnn_value_and_grad(
    features: ArrayLike,
    labels: ArrayLike,
    params: ArrayLike | object,
    *,
    tolerance: float = 1e-6,
) -> PhaseTorchQNNGradientResult:
    """Return bounded phase-QNN loss and gradient as NumPy plus PyTorch tensors.

    This route is deliberately narrower than arbitrary PyTorch autograd through
    a quantum simulator. It expresses the bounded phase-QNN gradient in the same
    tensor-ready analytic form used by the parameter-shift classifier and
    compares it against the canonical SCPN parameter-shift gradient before
    returning tensors to PyTorch workflows.
    """
    torch_module = _load_torch()
    parameter_values = _torch_values_to_numpy(params)
    (
        loss,
        gradient,
        reference_gradient,
        max_abs_error,
        l2_error,
        tolerance_value,
        passed,
    ) = _bounded_qnn_loss_gradient_reference(
        features,
        labels,
        parameter_values,
        tolerance=tolerance,
    )
    return PhaseTorchQNNGradientResult(
        loss=float(loss),
        gradient=gradient,
        parameter_shift_gradient=reference_gradient,
        torch_loss=_torch_tensor(torch_module, loss),
        torch_gradient=_torch_tensor(torch_module, gradient),
        torch_parameter_shift_gradient=_torch_tensor(torch_module, reference_gradient),
        max_abs_error=max_abs_error,
        l2_error=l2_error,
        tolerance=tolerance_value,
        passed=passed,
    )


def torch_autograd_qnn_value_and_grad(
    features: ArrayLike,
    labels: ArrayLike,
    params: ArrayLike | object,
    *,
    tolerance: float = 1e-6,
) -> PhaseTorchAutogradQNNGradientResult:
    """Return bounded phase-QNN loss and gradient through ``torch.autograd.Function``.

    This is a native PyTorch autograd route for the bounded phase-QNN surface
    only. The custom backward is the audited bounded analytic gradient, checked
    against the canonical SCPN parameter-shift gradient before the result is
    returned.
    """
    torch_module = _load_torch()
    function_base = _torch_autograd_function(torch_module)
    autograd_grad = _torch_autograd_grad(torch_module)
    feature_matrix = _as_feature_matrix(features)
    label_vector = _as_label_vector(labels, n_samples=feature_matrix.shape[0])
    parameter_values = _torch_values_to_numpy(params)
    if parameter_values.shape != (feature_matrix.shape[1],):
        raise ValueError(
            "params width must match feature width: "
            f"{parameter_values.shape[0]} != {feature_matrix.shape[1]}",
        )
    tolerance_value = _as_non_negative_tolerance(tolerance)
    audit: dict[str, AutogradAuditRecord] = {}

    class _BoundedPhaseQNNFunction(function_base):  # type: ignore[misc, valid-type]
        @staticmethod
        def forward(ctx: Any, parameter_tensor: Any) -> Any:
            raw_params = _torch_values_to_numpy(parameter_tensor)
            (
                loss,
                gradient,
                reference_gradient,
                max_abs_error,
                l2_error,
                _checked_tolerance,
                passed,
            ) = _bounded_qnn_loss_gradient_reference(
                feature_matrix,
                label_vector,
                raw_params,
                tolerance=tolerance_value,
            )
            ctx.gradient = _torch_tensor(torch_module, gradient)
            audit["record"] = (
                float(loss),
                gradient,
                reference_gradient,
                max_abs_error,
                l2_error,
                passed,
            )
            return _torch_tensor(torch_module, loss)

        @staticmethod
        def backward(ctx: Any, grad_output: Any) -> tuple[Any]:
            return (ctx.gradient * grad_output,)

    trainable_params = _torch_trainable_tensor(torch_module, parameter_values)
    torch_loss = _BoundedPhaseQNNFunction.apply(trainable_params)
    torch_gradient = autograd_grad(
        torch_loss,
        trainable_params,
        retain_graph=False,
        create_graph=False,
    )[0]
    try:
        loss, gradient, reference_gradient, max_abs_error, l2_error, passed = audit["record"]
    except KeyError as exc:
        raise RuntimeError("PyTorch autograd forward did not produce audit evidence") from exc
    return PhaseTorchAutogradQNNGradientResult(
        loss=loss,
        gradient=_as_parameter_vector("PyTorch autograd bounded phase-QNN gradient", gradient),
        parameter_shift_gradient=reference_gradient,
        torch_loss=torch_loss,
        torch_gradient=torch_gradient,
        torch_parameter_shift_gradient=_torch_tensor(torch_module, reference_gradient),
        max_abs_error=max_abs_error,
        l2_error=l2_error,
        tolerance=tolerance_value,
        passed=passed,
    )


__all__ = [
    "PhaseTorchAutogradQNNGradientResult",
    "PhaseTorchParameterShiftResult",
    "PhaseTorchQNNGradientResult",
    "is_phase_torch_available",
    "torch_autograd_qnn_value_and_grad",
    "torch_bounded_qnn_value_and_grad",
    "torch_parameter_shift_value_and_grad",
]
