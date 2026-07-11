# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Torch Compatibility, Module, and Training
"""Bounded Torch transforms, modules, and deterministic training evidence.

This one-way leaf owns the bounded phase-QNN ``torch.func`` and ``torch.compile``
compatibility routes, module/layer wrapper, wrapper audit, and compiled training
loop. The public facade injects its active optional-Torch loader.
"""

from __future__ import annotations

from typing import Any, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .qnn_training import parameter_shift_qnn_classifier_gradient
from .torch_bridge_contracts import (
    PhaseTorchCompileCompatibilityResult,
    PhaseTorchFuncCompatibilityResult,
    PhaseTorchModuleWrapperAuditResult,
    PhaseTorchTrainingLoopAuditResult,
)
from .torch_gradients import (
    TorchLoader,
    _as_feature_matrix,
    _as_label_vector,
    _as_non_negative_tolerance,
    _as_parameter_vector,
    _load_torch,
    _torch_tensor,
    _torch_values_to_numpy,
)
from .torch_qnode_transforms import (
    _as_parameter_matrix,
    _torch_batch_to_numpy,
    _torch_compile,
    _torch_func_transforms,
    _torch_scalar_to_float,
)

FloatArray: TypeAlias = NDArray[np.float64]


def _as_positive_learning_rate(value: float) -> float:
    learning_rate = float(value)
    if not np.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("learning_rate must be a positive finite float")
    return learning_rate


def _as_positive_step_count(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("steps must be a positive integer")
    if value <= 0:
        raise ValueError("steps must be a positive integer")
    return value


def _torch_nn_module_and_parameter(torch_module: Any) -> tuple[Any, Any]:
    torch_nn = getattr(torch_module, "nn", None)
    module_base = getattr(torch_nn, "Module", None)
    parameter_cls = getattr(torch_nn, "Parameter", None)
    if torch_nn is None or module_base is None or not callable(parameter_cls):
        raise RuntimeError("PyTorch module does not expose torch.nn.Module and torch.nn.Parameter")
    return module_base, parameter_cls


def _torch_parameter_count(module: Any) -> int:
    parameters = getattr(module, "parameters", None)
    if not callable(parameters):
        return 0
    return sum(1 for _parameter in parameters())


def _torch_bounded_qnn_loss_tensor(
    torch_module: Any,
    feature_tensor: Any,
    label_tensor: Any,
    parameter_tensor: Any,
) -> Any:
    shifted = feature_tensor + parameter_tensor.unsqueeze(0)
    probabilities = 0.5 * (1.0 - torch_module.cos(shifted))
    predictions = torch_module.mean(probabilities, dim=1)
    residual = predictions - label_tensor
    return torch_module.mean(residual * residual)


def run_torch_func_compatibility_audit(
    *,
    features: ArrayLike,
    labels: ArrayLike,
    params: ArrayLike | object,
    params_batch: ArrayLike | object,
    tolerance: float = 1e-6,
    _torch_loader: TorchLoader = _load_torch,
) -> PhaseTorchFuncCompatibilityResult:
    """Audit bounded phase-QNN compatibility with ``torch.func`` transforms."""
    torch_module = _torch_loader()
    torch_func_grad, torch_func_vmap, torch_func_jacrev = _torch_func_transforms(torch_module)
    feature_matrix = _as_feature_matrix(features)
    label_vector = _as_label_vector(labels, n_samples=feature_matrix.shape[0])
    parameter_values = _torch_values_to_numpy(params)
    if parameter_values.shape != (feature_matrix.shape[1],):
        raise ValueError(
            "params width must match feature width: "
            f"{parameter_values.shape[0]} != {feature_matrix.shape[1]}",
        )
    parameter_batch = _torch_batch_to_numpy(params_batch)
    parameter_batch = _as_parameter_matrix(
        "params_batch",
        parameter_batch,
        width=feature_matrix.shape[1],
    )
    tolerance_value = _as_non_negative_tolerance(tolerance)
    feature_tensor = _torch_tensor(torch_module, feature_matrix)
    label_tensor = _torch_tensor(torch_module, label_vector)

    def loss_fn(parameter_tensor: Any) -> Any:
        return _torch_bounded_qnn_loss_tensor(
            torch_module,
            feature_tensor,
            label_tensor,
            parameter_tensor,
        )

    grad_fn = torch_func_grad(loss_fn)
    vmap_grad_fn = torch_func_vmap(grad_fn)
    jacrev_fn = torch_func_jacrev(loss_fn)
    torch_params = _torch_tensor(torch_module, parameter_values)
    torch_params_batch = _torch_tensor(torch_module, parameter_batch)
    torch_grad_gradient = grad_fn(torch_params)
    torch_vmap_gradients = vmap_grad_fn(torch_params_batch)
    torch_jacrev_gradient = jacrev_fn(torch_params)
    grad_gradient = _torch_values_to_numpy(torch_grad_gradient)
    vmap_gradients = _torch_batch_to_numpy(torch_vmap_gradients)
    jacrev_gradient = _torch_values_to_numpy(torch_jacrev_gradient)
    parameter_shift_gradient = parameter_shift_qnn_classifier_gradient(
        feature_matrix,
        label_vector,
        parameter_values,
    )
    parameter_shift_gradient = _as_parameter_vector(
        "SCPN bounded phase-QNN parameter-shift gradient",
        parameter_shift_gradient,
        width=parameter_values.size,
    )
    parameter_shift_batch_gradients = np.vstack(
        [
            parameter_shift_qnn_classifier_gradient(feature_matrix, label_vector, row)
            for row in parameter_batch
        ],
    )
    deltas = (
        grad_gradient - parameter_shift_gradient,
        jacrev_gradient - parameter_shift_gradient,
        (vmap_gradients - parameter_shift_batch_gradients).reshape(-1),
    )
    flat_delta = np.concatenate([delta.reshape(-1) for delta in deltas])
    max_abs_error = float(np.max(np.abs(flat_delta))) if flat_delta.size else 0.0
    l2_error = float(np.linalg.norm(flat_delta))
    return PhaseTorchFuncCompatibilityResult(
        grad_gradient=grad_gradient,
        vmap_gradients=vmap_gradients,
        jacrev_gradient=jacrev_gradient,
        parameter_shift_gradient=parameter_shift_gradient,
        parameter_shift_batch_gradients=parameter_shift_batch_gradients,
        torch_grad_gradient=torch_grad_gradient,
        torch_vmap_gradients=torch_vmap_gradients,
        torch_jacrev_gradient=torch_jacrev_gradient,
        max_abs_error=max_abs_error,
        l2_error=l2_error,
        tolerance=tolerance_value,
        passed=bool(max_abs_error <= tolerance_value),
        func_grad_supported=True,
        func_vmap_supported=True,
        func_jacrev_supported=True,
    )


def run_torch_compile_compatibility_audit(
    *,
    features: ArrayLike,
    labels: ArrayLike,
    params: ArrayLike | object,
    tolerance: float = 1e-6,
    fullgraph: bool = True,
    dynamic: bool = False,
    _torch_loader: TorchLoader = _load_torch,
) -> PhaseTorchCompileCompatibilityResult:
    """Audit bounded phase-QNN compatibility with ``torch.compile``."""
    torch_module = _torch_loader()
    compile_fn = _torch_compile(torch_module)
    torch_func_grad, _torch_func_vmap, _torch_func_jacrev = _torch_func_transforms(torch_module)
    feature_matrix = _as_feature_matrix(features)
    label_vector = _as_label_vector(labels, n_samples=feature_matrix.shape[0])
    parameter_values = _torch_values_to_numpy(params)
    if parameter_values.shape != (feature_matrix.shape[1],):
        raise ValueError(
            "params width must match feature width: "
            f"{parameter_values.shape[0]} != {feature_matrix.shape[1]}",
        )
    tolerance_value = _as_non_negative_tolerance(tolerance)
    feature_tensor = _torch_tensor(torch_module, feature_matrix)
    label_tensor = _torch_tensor(torch_module, label_vector)

    def loss_fn(parameter_tensor: Any) -> Any:
        return _torch_bounded_qnn_loss_tensor(
            torch_module,
            feature_tensor,
            label_tensor,
            parameter_tensor,
        )

    grad_fn = torch_func_grad(loss_fn)
    compiled_grad_fn = compile_fn(grad_fn, fullgraph=bool(fullgraph), dynamic=bool(dynamic))
    torch_params = _torch_tensor(torch_module, parameter_values)
    torch_gradient = compiled_grad_fn(torch_params)
    gradient = _torch_values_to_numpy(torch_gradient)
    parameter_shift_gradient = parameter_shift_qnn_classifier_gradient(
        feature_matrix,
        label_vector,
        parameter_values,
    )
    parameter_shift_gradient = _as_parameter_vector(
        "SCPN bounded phase-QNN parameter-shift gradient",
        parameter_shift_gradient,
        width=parameter_values.size,
    )
    delta = gradient - parameter_shift_gradient
    max_abs_error = float(np.max(np.abs(delta))) if delta.size else 0.0
    l2_error = float(np.linalg.norm(delta))
    return PhaseTorchCompileCompatibilityResult(
        gradient=gradient,
        parameter_shift_gradient=parameter_shift_gradient,
        torch_gradient=torch_gradient,
        max_abs_error=max_abs_error,
        l2_error=l2_error,
        tolerance=tolerance_value,
        passed=bool(max_abs_error <= tolerance_value),
        torch_compile_supported=True,
        compiled_loss_supported=True,
        compiled_gradient_supported=True,
        fullgraph=bool(fullgraph),
        dynamic=bool(dynamic),
    )


def torch_bounded_qnn_module(
    *,
    features: ArrayLike,
    labels: ArrayLike,
    initial_params: ArrayLike | object,
    trainable: bool = True,
    _torch_loader: TorchLoader = _load_torch,
) -> Any:
    """Return a PyTorch ``nn.Module`` wrapper for the bounded phase-QNN loss."""
    torch_module = _torch_loader()
    module_base, parameter_cls = _torch_nn_module_and_parameter(torch_module)
    feature_matrix = _as_feature_matrix(features)
    label_vector = _as_label_vector(labels, n_samples=feature_matrix.shape[0])
    parameter_values = _torch_values_to_numpy(initial_params)
    if parameter_values.shape != (feature_matrix.shape[1],):
        raise ValueError(
            "initial_params width must match feature width: "
            f"{parameter_values.shape[0]} != {feature_matrix.shape[1]}",
        )
    feature_tensor = _torch_tensor(torch_module, feature_matrix)
    label_tensor = _torch_tensor(torch_module, label_vector)
    parameter_tensor = _torch_tensor(torch_module, parameter_values)

    class _BoundedPhaseQNNModule(module_base):  # type: ignore[misc, valid-type]
        def __init__(self) -> None:
            super().__init__()
            register_buffer = getattr(self, "register_buffer", None)
            if callable(register_buffer):
                register_buffer("features", feature_tensor)
                register_buffer("labels", label_tensor)
            else:
                self.features = feature_tensor
                self.labels = label_tensor
            self.params = parameter_cls(parameter_tensor, requires_grad=bool(trainable))
            self.feature_width = int(feature_matrix.shape[1])
            self.host_boundary = False
            self.native_framework_autodiff = True
            self.claim_boundary = "bounded_torch_module_layer_wrapper"

        def forward(self, params: Any | None = None) -> Any:
            parameter_source = self.params if params is None else params
            return _torch_bounded_qnn_loss_tensor(
                torch_module,
                self.features,
                self.labels,
                parameter_source,
            )

        def parameter_shift_gradient(self, params: Any | None = None) -> FloatArray:
            parameter_source = self.params if params is None else params
            raw_params = _torch_values_to_numpy(parameter_source)
            raw_params = _as_parameter_vector(
                "PyTorch bounded phase-QNN module parameters",
                raw_params,
                width=feature_matrix.shape[1],
            )
            reference_gradient = parameter_shift_qnn_classifier_gradient(
                feature_matrix,
                label_vector,
                raw_params,
            )
            return _as_parameter_vector(
                "SCPN bounded phase-QNN parameter-shift gradient",
                reference_gradient,
                width=feature_matrix.shape[1],
            )

    return _BoundedPhaseQNNModule()


def torch_bounded_qnn_layer(
    *,
    features: ArrayLike,
    labels: ArrayLike,
    initial_params: ArrayLike | object,
    trainable: bool = True,
    _torch_loader: TorchLoader = _load_torch,
) -> Any:
    """Return the bounded phase-QNN wrapper using layer-oriented naming."""
    return torch_bounded_qnn_module(
        features=features,
        labels=labels,
        initial_params=initial_params,
        trainable=trainable,
        _torch_loader=_torch_loader,
    )


def run_torch_module_wrapper_audit(
    *,
    features: ArrayLike,
    labels: ArrayLike,
    initial_params: ArrayLike | object,
    tolerance: float = 1e-6,
    _torch_loader: TorchLoader = _load_torch,
) -> PhaseTorchModuleWrapperAuditResult:
    """Audit bounded phase-QNN PyTorch module/layer wrapper gradients."""
    torch_module = _torch_loader()
    torch_func_grad, _torch_func_vmap, _torch_func_jacrev = _torch_func_transforms(torch_module)
    tolerance_value = _as_non_negative_tolerance(tolerance)
    module = torch_bounded_qnn_module(
        features=features,
        labels=labels,
        initial_params=initial_params,
        trainable=True,
        _torch_loader=_torch_loader,
    )
    torch_params = module.params
    torch_loss = module()

    def loss_fn(parameter_tensor: Any) -> Any:
        return module(parameter_tensor)

    grad_fn = torch_func_grad(loss_fn)
    torch_gradient = grad_fn(torch_params)
    gradient = _torch_values_to_numpy(torch_gradient)
    parameter_shift_gradient = module.parameter_shift_gradient(torch_params)
    delta = gradient - parameter_shift_gradient
    max_abs_error = float(np.max(np.abs(delta))) if delta.size else 0.0
    l2_error = float(np.linalg.norm(delta))
    return PhaseTorchModuleWrapperAuditResult(
        loss=_torch_scalar_to_float(torch_loss),
        gradient=gradient,
        parameter_shift_gradient=parameter_shift_gradient,
        torch_module=module,
        torch_loss=torch_loss,
        torch_gradient=torch_gradient,
        max_abs_error=max_abs_error,
        l2_error=l2_error,
        tolerance=tolerance_value,
        passed=bool(max_abs_error <= tolerance_value),
        module_wrapper_supported=True,
        layer_wrapper_supported=True,
        trainable_parameters=_torch_parameter_count(module),
    )


def run_torch_training_loop_audit(
    *,
    features: ArrayLike,
    labels: ArrayLike,
    initial_params: ArrayLike | object,
    learning_rate: float = 0.1,
    steps: int = 4,
    tolerance: float = 1e-6,
    fullgraph: bool = True,
    dynamic: bool = False,
    _torch_loader: TorchLoader = _load_torch,
) -> PhaseTorchTrainingLoopAuditResult:
    """Audit a bounded PyTorch module training loop against SCPN references.

    The loop uses the bounded phase-QNN ``nn.Module`` wrapper, compiles its loss
    callable with ``torch.compile``, obtains gradients through ``torch.func``,
    and applies deterministic gradient-descent updates. It is local functional
    correctness evidence only; CUDA, provider, finite-shot, hardware, isolated
    benchmark, and performance promotion claims remain outside this route.
    """
    torch_module = _torch_loader()
    compile_fn = _torch_compile(torch_module)
    torch_func_grad, _torch_func_vmap, _torch_func_jacrev = _torch_func_transforms(torch_module)
    del _torch_func_vmap, _torch_func_jacrev
    feature_matrix = _as_feature_matrix(features)
    label_vector = _as_label_vector(labels, n_samples=feature_matrix.shape[0])
    parameter_values = _torch_values_to_numpy(initial_params)
    parameter_values = _as_parameter_vector(
        "initial_params",
        parameter_values,
        width=feature_matrix.shape[1],
    )
    tolerance_value = _as_non_negative_tolerance(tolerance)
    learning_rate_value = _as_positive_learning_rate(learning_rate)
    step_count = _as_positive_step_count(steps)
    module = torch_bounded_qnn_module(
        features=feature_matrix,
        labels=label_vector,
        initial_params=parameter_values,
        trainable=True,
        _torch_loader=_torch_loader,
    )

    def loss_fn(parameter_tensor: Any) -> Any:
        return module(parameter_tensor)

    compiled_loss_fn = compile_fn(loss_fn, fullgraph=bool(fullgraph), dynamic=bool(dynamic))
    grad_fn = torch_func_grad(loss_fn)
    compiled_grad_fn = compile_fn(grad_fn, fullgraph=bool(fullgraph), dynamic=bool(dynamic))
    current_params = parameter_values.copy()
    loss_values: list[float] = []
    gradient_values: list[FloatArray] = []
    gradient_deltas: list[FloatArray] = []
    for _index in range(step_count):
        torch_params = _torch_tensor(torch_module, current_params)
        torch_loss = compiled_loss_fn(torch_params)
        torch_gradient = compiled_grad_fn(torch_params)
        loss_values.append(_torch_scalar_to_float(torch_loss))
        gradient = _as_parameter_vector(
            "PyTorch training-loop gradient",
            _torch_values_to_numpy(torch_gradient),
            width=parameter_values.size,
        )
        reference_gradient = parameter_shift_qnn_classifier_gradient(
            feature_matrix,
            label_vector,
            current_params,
        )
        reference_gradient = _as_parameter_vector(
            "SCPN bounded phase-QNN parameter-shift gradient",
            reference_gradient,
            width=parameter_values.size,
        )
        gradient_values.append(gradient)
        gradient_deltas.append(gradient - reference_gradient)
        current_params = current_params - learning_rate_value * gradient

    final_torch_params = _torch_tensor(torch_module, current_params)
    final_torch_loss = compiled_loss_fn(final_torch_params)
    final_torch_gradient = compiled_grad_fn(final_torch_params)
    final_gradient = _as_parameter_vector(
        "PyTorch training-loop final gradient",
        _torch_values_to_numpy(final_torch_gradient),
        width=parameter_values.size,
    )
    parameter_shift_final_gradient = parameter_shift_qnn_classifier_gradient(
        feature_matrix,
        label_vector,
        current_params,
    )
    parameter_shift_final_gradient = _as_parameter_vector(
        "SCPN bounded phase-QNN final parameter-shift gradient",
        parameter_shift_final_gradient,
        width=parameter_values.size,
    )
    gradient_deltas.append(final_gradient - parameter_shift_final_gradient)
    loss_values.append(_torch_scalar_to_float(final_torch_loss))
    flat_delta = np.concatenate([delta.reshape(-1) for delta in gradient_deltas])
    max_abs_gradient_error = float(np.max(np.abs(flat_delta))) if flat_delta.size else 0.0
    l2_gradient_error = float(np.linalg.norm(flat_delta))
    loss_history = _as_parameter_vector("PyTorch training-loop loss history", loss_values)
    gradient_history = _as_parameter_matrix(
        "PyTorch training-loop gradient history",
        np.vstack(gradient_values),
        width=parameter_values.size,
    )
    passed = bool(
        max_abs_gradient_error <= tolerance_value
        and np.all(np.isfinite(loss_history))
        and float(loss_history[-1]) <= float(loss_history[0]) + tolerance_value
    )
    return PhaseTorchTrainingLoopAuditResult(
        initial_params=parameter_values,
        final_params=current_params,
        loss_history=loss_history,
        gradient_history=gradient_history,
        final_gradient=final_gradient,
        parameter_shift_final_gradient=parameter_shift_final_gradient,
        max_abs_gradient_error=max_abs_gradient_error,
        l2_gradient_error=l2_gradient_error,
        tolerance=tolerance_value,
        passed=passed,
        steps=step_count,
        learning_rate=learning_rate_value,
        torch_module=module,
        torch_final_loss=final_torch_loss,
        torch_final_gradient=final_torch_gradient,
        module_wrapper_supported=True,
        func_grad_supported=True,
        torch_compile_supported=True,
        compiled_loss_supported=True,
        parameter_update_supported=bool(not np.allclose(current_params, parameter_values)),
    )
