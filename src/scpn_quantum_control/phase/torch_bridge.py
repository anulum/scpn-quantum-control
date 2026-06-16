# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Phase PyTorch Bridge
"""Optional PyTorch interop for phase parameter-shift gradients."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
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


@dataclass(frozen=True)
class PhaseTorchFuncCompatibilityResult:
    """Bounded phase-QNN compatibility evidence for ``torch.func`` transforms."""

    grad_gradient: FloatArray
    vmap_gradients: FloatArray
    jacrev_gradient: FloatArray
    parameter_shift_gradient: FloatArray
    parameter_shift_batch_gradients: FloatArray
    torch_grad_gradient: Any
    torch_vmap_gradients: Any
    torch_jacrev_gradient: Any
    max_abs_error: float
    l2_error: float
    tolerance: float
    passed: bool
    func_grad_supported: bool
    func_vmap_supported: bool
    func_jacrev_supported: bool
    native_framework_autodiff: bool = True
    host_boundary: bool = False
    claim_boundary: str = "bounded_torch_func_compatibility"

    def to_dict(self) -> dict[str, object]:
        """Return JSON-serialisable PyTorch ``torch.func`` evidence metadata."""
        return {
            "grad_gradient": self.grad_gradient.copy(),
            "vmap_gradients": self.vmap_gradients.copy(),
            "jacrev_gradient": self.jacrev_gradient.copy(),
            "parameter_shift_gradient": self.parameter_shift_gradient.copy(),
            "parameter_shift_batch_gradients": self.parameter_shift_batch_gradients.copy(),
            "max_abs_error": self.max_abs_error,
            "l2_error": self.l2_error,
            "tolerance": self.tolerance,
            "passed": self.passed,
            "func_grad_supported": self.func_grad_supported,
            "func_vmap_supported": self.func_vmap_supported,
            "func_jacrev_supported": self.func_jacrev_supported,
            "native_framework_autodiff": self.native_framework_autodiff,
            "host_boundary": self.host_boundary,
            "claim_boundary": self.claim_boundary,
            "torch_grad_gradient_type": type(self.torch_grad_gradient).__name__,
            "torch_vmap_gradients_type": type(self.torch_vmap_gradients).__name__,
            "torch_jacrev_gradient_type": type(self.torch_jacrev_gradient).__name__,
        }


@dataclass(frozen=True)
class PhaseTorchCompileCompatibilityResult:
    """Bounded phase-QNN compatibility evidence for ``torch.compile``."""

    gradient: FloatArray
    parameter_shift_gradient: FloatArray
    torch_gradient: Any
    max_abs_error: float
    l2_error: float
    tolerance: float
    passed: bool
    torch_compile_supported: bool
    compiled_loss_supported: bool
    compiled_gradient_supported: bool
    fullgraph: bool
    dynamic: bool
    native_framework_autodiff: bool = True
    host_boundary: bool = False
    claim_boundary: str = "bounded_torch_compile_compatibility"

    def to_dict(self) -> dict[str, object]:
        """Return JSON-serialisable PyTorch compile evidence metadata."""
        return {
            "gradient": self.gradient.copy(),
            "parameter_shift_gradient": self.parameter_shift_gradient.copy(),
            "max_abs_error": self.max_abs_error,
            "l2_error": self.l2_error,
            "tolerance": self.tolerance,
            "passed": self.passed,
            "torch_compile_supported": self.torch_compile_supported,
            "compiled_loss_supported": self.compiled_loss_supported,
            "compiled_gradient_supported": self.compiled_gradient_supported,
            "fullgraph": self.fullgraph,
            "dynamic": self.dynamic,
            "native_framework_autodiff": self.native_framework_autodiff,
            "host_boundary": self.host_boundary,
            "claim_boundary": self.claim_boundary,
            "torch_gradient_type": type(self.torch_gradient).__name__,
        }


@dataclass(frozen=True)
class PhaseTorchModuleWrapperAuditResult:
    """Bounded phase-QNN evidence for PyTorch module/layer wrappers."""

    loss: float
    gradient: FloatArray
    parameter_shift_gradient: FloatArray
    torch_module: Any
    torch_loss: Any
    torch_gradient: Any
    max_abs_error: float
    l2_error: float
    tolerance: float
    passed: bool
    module_wrapper_supported: bool
    layer_wrapper_supported: bool
    trainable_parameters: int
    native_framework_autodiff: bool = True
    host_boundary: bool = False
    claim_boundary: str = "bounded_torch_module_layer_wrapper"

    def to_dict(self) -> dict[str, object]:
        """Return JSON-serialisable PyTorch module-wrapper evidence metadata."""
        return {
            "loss": self.loss,
            "gradient": self.gradient.copy(),
            "parameter_shift_gradient": self.parameter_shift_gradient.copy(),
            "max_abs_error": self.max_abs_error,
            "l2_error": self.l2_error,
            "tolerance": self.tolerance,
            "passed": self.passed,
            "module_wrapper_supported": self.module_wrapper_supported,
            "layer_wrapper_supported": self.layer_wrapper_supported,
            "trainable_parameters": self.trainable_parameters,
            "native_framework_autodiff": self.native_framework_autodiff,
            "host_boundary": self.host_boundary,
            "claim_boundary": self.claim_boundary,
            "torch_module_type": type(self.torch_module).__name__,
            "torch_loss_type": type(self.torch_loss).__name__,
            "torch_gradient_type": type(self.torch_gradient).__name__,
        }


@dataclass(frozen=True)
class PhaseTorchLiveOverlayEvidence:
    """Validated live CPU-overlay PyTorch external-comparison evidence."""

    artifact_id: str
    artifact_path: str
    classification: str
    torch_version: str
    value_error: float
    gradient_error: float
    runtime_seconds: float
    memory_peak_bytes: int
    batching_support: str
    transform_support: str
    claim_boundary: str
    promotion_ready: bool
    passed: bool = True

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready live-overlay evidence metadata."""
        return {
            "artifact_id": self.artifact_id,
            "artifact_path": self.artifact_path,
            "classification": self.classification,
            "torch_version": self.torch_version,
            "value_error": self.value_error,
            "gradient_error": self.gradient_error,
            "runtime_seconds": self.runtime_seconds,
            "memory_peak_bytes": self.memory_peak_bytes,
            "batching_support": self.batching_support,
            "transform_support": self.transform_support,
            "claim_boundary": self.claim_boundary,
            "promotion_ready": self.promotion_ready,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class PhaseTorchMaturityAuditResult:
    """Aggregate PyTorch maturity evidence and explicit provider-parity blockers."""

    bounded_model_ready: bool
    ready_for_provider_exceedance: bool
    evidence: dict[str, object]
    required_capabilities: dict[str, str]
    open_gaps: tuple[str, ...]
    claim_boundary: str = "bounded_torch_provider_maturity_audit"

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready PyTorch maturity evidence."""
        return {
            "bounded_model_ready": self.bounded_model_ready,
            "ready_for_provider_exceedance": self.ready_for_provider_exceedance,
            "evidence": {
                name: _json_ready(_result_to_dict(result))
                for name, result in self.evidence.items()
            },
            "required_capabilities": dict(self.required_capabilities),
            "open_gaps": list(self.open_gaps),
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class PhaseTorchPhaseQNodeLoweringRoute:
    """One PyTorch lowering route in the registered Phase-QNode parity matrix."""

    name: str
    status: str
    reason: str
    requires: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready route metadata."""
        return {
            "name": self.name,
            "status": self.status,
            "reason": self.reason,
            "requires": list(self.requires),
        }


@dataclass(frozen=True)
class PhaseTorchPhaseQNodeLoweringMatrixResult:
    """Fail-closed PyTorch parity matrix for arbitrary registered Phase-QNodes."""

    routes: tuple[PhaseTorchPhaseQNodeLoweringRoute, ...]
    claim_boundary: str = "bounded_torch_phase_qnode_lowering_matrix"

    @property
    def bounded_qnn_routes_ready(self) -> bool:
        """Return whether the bounded QNN Torch routes are declared ready."""
        return all(
            route.status == "passed"
            for route in self.routes
            if route.name.startswith("bounded_qnn_")
        )

    @property
    def arbitrary_phase_qnode_lowering_ready(self) -> bool:
        """Return whether arbitrary registered Phase-QNode lowering is ready."""
        return all(
            route.status == "passed"
            for route in self.routes
            if route.name.startswith("registered_phase_qnode_")
        )

    @property
    def ready_for_provider_exceedance(self) -> bool:
        """Return whether this matrix permits PyTorch provider-exceedance claims."""
        return all(route.status == "passed" for route in self.routes)

    @property
    def open_gaps(self) -> tuple[str, ...]:
        """Return routes that still block PyTorch provider-exceedance claims."""
        return tuple(route.name for route in self.routes if route.status != "passed")

    def route_status(self, name: str) -> str:
        """Return the status for a named route, failing closed on unknown names."""
        for route in self.routes:
            if route.name == name:
                return route.status
        raise KeyError(f"unknown PyTorch Phase-QNode lowering route: {name}")

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready PyTorch Phase-QNode lowering parity metadata."""
        return {
            "bounded_qnn_routes_ready": self.bounded_qnn_routes_ready,
            "arbitrary_phase_qnode_lowering_ready": self.arbitrary_phase_qnode_lowering_ready,
            "ready_for_provider_exceedance": self.ready_for_provider_exceedance,
            "routes": {route.name: route.to_dict() for route in self.routes},
            "open_gaps": list(self.open_gaps),
            "claim_boundary": self.claim_boundary,
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


def run_torch_phase_qnode_lowering_matrix() -> PhaseTorchPhaseQNodeLoweringMatrixResult:
    """Return the PyTorch parity matrix for registered Phase-QNode lowering.

    The current PyTorch surface is production-grade for the bounded QNN routes
    listed here, but arbitrary registered Phase-QNode lowering is intentionally
    blocked until Torch lowering rules, provider safety, hardware evidence, and
    isolated benchmark artefacts exist.
    """

    routes = (
        PhaseTorchPhaseQNodeLoweringRoute(
            name="bounded_qnn_analytic_tensor",
            status="passed",
            reason="bounded phase-QNN analytic tensor value-and-gradient is implemented",
        ),
        PhaseTorchPhaseQNodeLoweringRoute(
            name="bounded_qnn_custom_autograd",
            status="passed",
            reason="bounded phase-QNN custom torch.autograd.Function route is implemented",
        ),
        PhaseTorchPhaseQNodeLoweringRoute(
            name="bounded_qnn_torch_func",
            status="passed",
            reason="bounded torch.func grad/vmap/jacrev compatibility is implemented",
        ),
        PhaseTorchPhaseQNodeLoweringRoute(
            name="bounded_qnn_torch_compile",
            status="passed",
            reason="bounded torch.compile gradient compatibility is implemented",
        ),
        PhaseTorchPhaseQNodeLoweringRoute(
            name="bounded_qnn_module_layer_wrapper",
            status="passed",
            reason="bounded nn.Module and layer wrappers are implemented",
        ),
        PhaseTorchPhaseQNodeLoweringRoute(
            name="registered_phase_qnode_statevector_lowering",
            status="blocked",
            reason="arbitrary registered Phase-QNode circuits do not yet lower into Torch graphs",
            requires=(
                "torch_fx_lowering_rules",
                "gate_observable_coverage_matrix",
                "statevector_gradient_parity_artifact",
            ),
        ),
        PhaseTorchPhaseQNodeLoweringRoute(
            name="registered_phase_qnode_finite_shot_lowering",
            status="blocked",
            reason="finite-shot Torch lowering needs uncertainty and sampler provenance",
            requires=(
                "shot_policy",
                "rng_seed_provenance",
                "uncertainty_artifact",
            ),
        ),
        PhaseTorchPhaseQNodeLoweringRoute(
            name="registered_phase_qnode_provider_lowering",
            status="blocked",
            reason="provider callbacks are not Torch compiler/autograd-safe yet",
            requires=(
                "provider_allowlist",
                "callback_transform_safety_audit",
                "provider_execution_artifact",
            ),
        ),
        PhaseTorchPhaseQNodeLoweringRoute(
            name="registered_phase_qnode_hardware_lowering",
            status="blocked",
            reason="live hardware lowering requires explicit ticketed execution evidence",
            requires=(
                "live_ticket",
                "provider_allowlist",
                "shot_budget",
                "hardware_evidence_id",
            ),
        ),
        PhaseTorchPhaseQNodeLoweringRoute(
            name="registered_phase_qnode_dynamic_circuit_lowering",
            status="blocked",
            reason="mid-circuit measurement and feedback are outside the Torch lowering boundary",
            requires=(
                "dynamic_circuit_semantics",
                "classical_feedback_contract",
                "gradient_policy",
            ),
        ),
        PhaseTorchPhaseQNodeLoweringRoute(
            name="isolated_benchmark_artifact",
            status="blocked",
            reason="provider-exceedance promotion needs isolated benchmark evidence",
            requires=("isolated_affinity_benchmark_id",),
        ),
    )
    return PhaseTorchPhaseQNodeLoweringMatrixResult(routes=routes)


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


def _as_parameter_matrix(name: str, values: object, *, width: int | None = None) -> FloatArray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must not be empty")
    if width is not None and matrix.shape[1] != width:
        raise ValueError(f"{name} width must be {width}, got {matrix.shape[1]}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
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


def _torch_batch_to_numpy(values: object) -> FloatArray:
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
    return _as_parameter_matrix("values", candidate)


def _torch_matrix_to_numpy(name: str, values: object) -> FloatArray:
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
    return _as_parameter_matrix(name, candidate)


def _torch_scalar_to_float(values: object) -> float:
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
    scalar = np.asarray(candidate, dtype=float)
    if scalar.shape not in ((), (1,)):
        raise ValueError(f"PyTorch scalar value must be scalar-like, got {scalar.shape}")
    value = float(scalar.reshape(-1)[0])
    if not np.isfinite(value):
        raise ValueError("PyTorch scalar value must be finite")
    return value


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


def _torch_func_transforms(torch_module: Any) -> tuple[Any, Any, Any]:
    torch_func = getattr(torch_module, "func", None)
    grad = getattr(torch_func, "grad", None)
    vmap = getattr(torch_func, "vmap", None)
    jacrev = getattr(torch_func, "jacrev", None)
    if torch_func is None or not callable(grad) or not callable(vmap) or not callable(jacrev):
        raise RuntimeError("PyTorch module does not expose torch.func.grad/vmap/jacrev")
    return grad, vmap, jacrev


def _torch_compile(torch_module: Any) -> Any:
    compile_fn = getattr(torch_module, "compile", None)
    if not callable(compile_fn):
        raise RuntimeError("PyTorch module does not expose torch.compile")
    return compile_fn


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


def run_torch_func_compatibility_audit(
    *,
    features: ArrayLike,
    labels: ArrayLike,
    params: ArrayLike | object,
    params_batch: ArrayLike | object,
    tolerance: float = 1e-6,
) -> PhaseTorchFuncCompatibilityResult:
    """Audit bounded phase-QNN compatibility with ``torch.func`` transforms."""
    torch_module = _load_torch()
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
) -> PhaseTorchCompileCompatibilityResult:
    """Audit bounded phase-QNN compatibility with ``torch.compile``."""
    torch_module = _load_torch()
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
) -> Any:
    """Return a PyTorch ``nn.Module`` wrapper for the bounded phase-QNN loss."""
    torch_module = _load_torch()
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
) -> Any:
    """Return the bounded phase-QNN wrapper using layer-oriented naming."""
    return torch_bounded_qnn_module(
        features=features,
        labels=labels,
        initial_params=initial_params,
        trainable=trainable,
    )


def run_torch_module_wrapper_audit(
    *,
    features: ArrayLike,
    labels: ArrayLike,
    initial_params: ArrayLike | object,
    tolerance: float = 1e-6,
) -> PhaseTorchModuleWrapperAuditResult:
    """Audit bounded phase-QNN PyTorch module/layer wrapper gradients."""
    torch_module = _load_torch()
    torch_func_grad, _torch_func_vmap, _torch_func_jacrev = _torch_func_transforms(torch_module)
    tolerance_value = _as_non_negative_tolerance(tolerance)
    module = torch_bounded_qnn_module(
        features=features,
        labels=labels,
        initial_params=initial_params,
        trainable=True,
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


def run_torch_maturity_audit(
    *,
    features: ArrayLike,
    labels: ArrayLike,
    params: ArrayLike | object,
    params_batch: ArrayLike | object,
    tolerance: float = 1e-6,
    fullgraph: bool = True,
    dynamic: bool = False,
    live_overlay_artifact_path: str | Path | None = None,
) -> PhaseTorchMaturityAuditResult:
    """Aggregate bounded PyTorch evidence and provider-level parity blockers.

    The audit records the bounded phase-QNN routes that are implemented today:
    tensor-ready analytic gradients, custom autograd, ``torch.func`` transforms,
    ``torch.compile``, and module/layer wrappers. It deliberately keeps broader
    PyTorch-provider maturity blocked until arbitrary Phase-QNode lowering, full
    compiler/autograd integration, live overlay evidence, and isolated benchmark
    artefacts are present.
    """

    tolerance_value = _as_non_negative_tolerance(tolerance)
    feature_matrix = _as_feature_matrix(features)
    label_vector = _as_label_vector(labels, n_samples=feature_matrix.shape[0])
    parameter_values = _as_parameter_vector(
        "params",
        _torch_values_to_numpy(params),
        width=feature_matrix.shape[1],
    )
    parameter_batch = _as_parameter_matrix(
        "params_batch",
        _torch_matrix_to_numpy("params_batch", params_batch),
        width=feature_matrix.shape[1],
    )

    analytic_tensor = torch_bounded_qnn_value_and_grad(
        feature_matrix,
        label_vector,
        parameter_values,
        tolerance=tolerance_value,
    )
    custom_autograd = torch_autograd_qnn_value_and_grad(
        feature_matrix,
        label_vector,
        parameter_values,
        tolerance=tolerance_value,
    )
    torch_func = run_torch_func_compatibility_audit(
        features=feature_matrix,
        labels=label_vector,
        params=parameter_values,
        params_batch=parameter_batch,
        tolerance=tolerance_value,
    )
    torch_compile = run_torch_compile_compatibility_audit(
        features=feature_matrix,
        labels=label_vector,
        params=parameter_values,
        tolerance=tolerance_value,
        fullgraph=fullgraph,
        dynamic=dynamic,
    )
    module_layer_wrapper = run_torch_module_wrapper_audit(
        features=feature_matrix,
        labels=label_vector,
        initial_params=parameter_values,
        tolerance=tolerance_value,
    )
    live_overlay = (
        _load_torch_live_overlay_evidence(live_overlay_artifact_path)
        if live_overlay_artifact_path is not None
        else None
    )

    evidence: dict[str, object] = {
        "analytic_tensor": analytic_tensor,
        "custom_autograd": custom_autograd,
        "torch_func": torch_func,
        "torch_compile": torch_compile,
        "module_layer_wrapper": module_layer_wrapper,
        "phase_qnode_lowering_matrix": run_torch_phase_qnode_lowering_matrix(),
    }
    if live_overlay is not None:
        evidence["live_overlay"] = live_overlay
    bounded_model_ready = all(
        bool(getattr(result, "passed", False))
        for name, result in evidence.items()
        if name != "phase_qnode_lowering_matrix"
    )
    lowering_matrix = evidence["phase_qnode_lowering_matrix"]
    assert isinstance(lowering_matrix, PhaseTorchPhaseQNodeLoweringMatrixResult)
    required_capabilities = {
        "analytic_tensor": "passed" if analytic_tensor.passed else "failed",
        "custom_autograd": "passed" if custom_autograd.passed else "failed",
        "torch_func": "passed" if torch_func.passed else "failed",
        "torch_compile": "passed" if torch_compile.passed else "failed",
        "module_layer_wrapper": "passed" if module_layer_wrapper.passed else "failed",
        "live_overlay_execution": "passed" if live_overlay is not None else "blocked",
        "arbitrary_phase_qnode_torch_lowering": "blocked",
        "full_compiler_autograd_integration": "blocked",
        "promotion_grade_isolated_benchmarks": "blocked",
    }
    required_capabilities.update(
        {
            f"phase_qnode_lowering:{route.name}": route.status
            for route in lowering_matrix.routes
            if route.status != "passed"
        }
    )
    open_gaps = tuple(name for name, status in required_capabilities.items() if status != "passed")
    return PhaseTorchMaturityAuditResult(
        bounded_model_ready=bounded_model_ready,
        ready_for_provider_exceedance=bounded_model_ready and not open_gaps,
        evidence=evidence,
        required_capabilities=required_capabilities,
        open_gaps=open_gaps,
    )


def _load_torch_live_overlay_evidence(
    artifact_path: str | Path,
) -> PhaseTorchLiveOverlayEvidence:
    path = Path(artifact_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("PyTorch live overlay artefact must be a JSON object")
    classification = _required_str(payload, "classification")
    if classification != "functional_non_isolated":
        raise ValueError("PyTorch live overlay artefact must be functional_non_isolated")
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("PyTorch live overlay artefact must include rows")
    pytorch_rows = [
        row
        for row in rows
        if isinstance(row, dict)
        and row.get("backend") == "pytorch"
        and row.get("status") == "success"
    ]
    if not pytorch_rows:
        raise ValueError("PyTorch live overlay artefact requires a successful PyTorch row")
    row = pytorch_rows[0]
    dependency_versions = row.get("dependency_versions")
    if not isinstance(dependency_versions, dict):
        raise ValueError("successful PyTorch row must include dependency_versions")
    torch_version = dependency_versions.get("torch")
    if not isinstance(torch_version, str) or not torch_version:
        raise ValueError("successful PyTorch row must include a torch dependency version")
    return PhaseTorchLiveOverlayEvidence(
        artifact_id=_required_str(payload, "artifact_id"),
        artifact_path=str(path),
        classification=classification,
        torch_version=torch_version,
        value_error=_required_float(row, "value_error"),
        gradient_error=_required_float(row, "gradient_error"),
        runtime_seconds=_required_float(row, "runtime_seconds"),
        memory_peak_bytes=_required_int(row, "memory_peak_bytes"),
        batching_support=_required_str(row, "batching_support"),
        transform_support=_required_str(row, "transform_support"),
        claim_boundary=_required_str(row, "claim_boundary"),
        promotion_ready=bool(payload.get("promotion_ready", False)),
    )


def _required_str(payload: dict[Any, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _required_float(payload: dict[Any, Any], key: str) -> float:
    value = payload.get(key)
    if not isinstance(value, int | float):
        raise ValueError(f"{key} must be numeric")
    return float(value)


def _required_int(payload: dict[Any, Any], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return int(value)


def _result_to_dict(result: object) -> object:
    to_dict = getattr(result, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    return result


def _json_ready(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


__all__ = [
    "PhaseTorchAutogradQNNGradientResult",
    "PhaseTorchCompileCompatibilityResult",
    "PhaseTorchFuncCompatibilityResult",
    "PhaseTorchLiveOverlayEvidence",
    "PhaseTorchMaturityAuditResult",
    "PhaseTorchModuleWrapperAuditResult",
    "PhaseTorchParameterShiftResult",
    "PhaseTorchPhaseQNodeLoweringMatrixResult",
    "PhaseTorchPhaseQNodeLoweringRoute",
    "PhaseTorchQNNGradientResult",
    "is_phase_torch_available",
    "run_torch_compile_compatibility_audit",
    "run_torch_func_compatibility_audit",
    "run_torch_maturity_audit",
    "run_torch_module_wrapper_audit",
    "run_torch_phase_qnode_lowering_matrix",
    "torch_autograd_qnn_value_and_grad",
    "torch_bounded_qnn_value_and_grad",
    "torch_bounded_qnn_layer",
    "torch_bounded_qnn_module",
    "torch_parameter_shift_value_and_grad",
]
