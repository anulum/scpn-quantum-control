# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Torch Result Contracts
"""Immutable NumPy/stdlib result contracts for the optional Torch bridge.

This dependency-free leaf owns structured evidence records only. Optional
Torch loading and all executable framework routes remain outside this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]


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
class PhaseTorchTrainingLoopAuditResult:
    """Bounded PyTorch training-loop parity evidence for phase-QNN modules."""

    initial_params: FloatArray
    final_params: FloatArray
    loss_history: FloatArray
    gradient_history: FloatArray
    final_gradient: FloatArray
    parameter_shift_final_gradient: FloatArray
    max_abs_gradient_error: float
    l2_gradient_error: float
    tolerance: float
    passed: bool
    steps: int
    learning_rate: float
    torch_module: Any
    torch_final_loss: Any
    torch_final_gradient: Any
    module_wrapper_supported: bool
    func_grad_supported: bool
    torch_compile_supported: bool
    compiled_loss_supported: bool
    parameter_update_supported: bool
    native_framework_autodiff: bool = True
    host_boundary: bool = False
    claim_boundary: str = "bounded_torch_training_loop_parity"

    @property
    def initial_loss(self) -> float:
        """Return the first recorded training loss."""
        return float(self.loss_history[0])

    @property
    def final_loss(self) -> float:
        """Return the final recorded training loss."""
        return float(self.loss_history[-1])

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready PyTorch training-loop evidence metadata."""
        return {
            "initial_params": self.initial_params.tolist(),
            "final_params": self.final_params.tolist(),
            "loss_history": self.loss_history.tolist(),
            "gradient_history": self.gradient_history.tolist(),
            "final_gradient": self.final_gradient.tolist(),
            "parameter_shift_final_gradient": self.parameter_shift_final_gradient.tolist(),
            "max_abs_gradient_error": self.max_abs_gradient_error,
            "l2_gradient_error": self.l2_gradient_error,
            "tolerance": self.tolerance,
            "passed": self.passed,
            "steps": self.steps,
            "learning_rate": self.learning_rate,
            "initial_loss": self.initial_loss,
            "final_loss": self.final_loss,
            "module_wrapper_supported": self.module_wrapper_supported,
            "func_grad_supported": self.func_grad_supported,
            "torch_compile_supported": self.torch_compile_supported,
            "compiled_loss_supported": self.compiled_loss_supported,
            "parameter_update_supported": self.parameter_update_supported,
            "native_framework_autodiff": self.native_framework_autodiff,
            "host_boundary": self.host_boundary,
            "claim_boundary": self.claim_boundary,
            "torch_module_type": type(self.torch_module).__name__,
            "torch_final_loss_type": type(self.torch_final_loss).__name__,
            "torch_final_gradient_type": type(self.torch_final_gradient).__name__,
        }


@dataclass(frozen=True)
class PhaseTorchPhaseQNodeStatevectorResult:
    """Native PyTorch autograd evidence for a registered local Phase-QNode."""

    value: float
    gradient: FloatArray
    state: NDArray[np.complex128]
    parameter_shift_value: float
    parameter_shift_gradient: FloatArray
    torch_value: Any
    torch_gradient: Any
    max_abs_error: float
    l2_error: float
    tolerance: float
    passed: bool
    native_framework_autodiff: bool
    host_boundary: bool
    method: str = "torch_native_registered_phase_qnode_statevector_value_and_grad"
    claim_boundary: str = "registered_phase_qnode_torch_statevector_lowering"

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready registered Phase-QNode PyTorch lowering evidence."""
        return {
            "value": self.value,
            "gradient": self.gradient.tolist(),
            "state_real": self.state.real.tolist(),
            "state_imag": self.state.imag.tolist(),
            "parameter_shift_value": self.parameter_shift_value,
            "parameter_shift_gradient": self.parameter_shift_gradient.tolist(),
            "max_abs_error": self.max_abs_error,
            "l2_error": self.l2_error,
            "tolerance": self.tolerance,
            "passed": self.passed,
            "native_framework_autodiff": self.native_framework_autodiff,
            "host_boundary": self.host_boundary,
            "method": self.method,
            "claim_boundary": self.claim_boundary,
            "torch_value_type": type(self.torch_value).__name__,
            "torch_gradient_type": type(self.torch_gradient).__name__,
        }


@dataclass(frozen=True)
class PhaseTorchPhaseQNodeTransformResult:
    """Native ``torch.func`` evidence for registered local Phase-QNode circuits."""

    value: float
    gradient: FloatArray
    jacrev_gradient: FloatArray
    vmap_gradients: FloatArray
    parameter_shift_value: float
    parameter_shift_gradient: FloatArray
    parameter_shift_batch_gradients: FloatArray
    torch_value: Any
    torch_gradient: Any
    torch_jacrev_gradient: Any
    torch_vmap_gradients: Any
    max_abs_error: float
    l2_error: float
    tolerance: float
    passed: bool
    func_grad_supported: bool
    func_vmap_supported: bool
    func_jacrev_supported: bool
    native_framework_autodiff: bool = True
    host_boundary: bool = False
    method: str = "torch_native_registered_phase_qnode_func_transform_audit"
    claim_boundary: str = "registered_phase_qnode_torch_func_transform_lowering"

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready registered Phase-QNode PyTorch transform evidence."""
        return {
            "value": self.value,
            "gradient": self.gradient.tolist(),
            "jacrev_gradient": self.jacrev_gradient.tolist(),
            "vmap_gradients": self.vmap_gradients.tolist(),
            "parameter_shift_value": self.parameter_shift_value,
            "parameter_shift_gradient": self.parameter_shift_gradient.tolist(),
            "parameter_shift_batch_gradients": self.parameter_shift_batch_gradients.tolist(),
            "max_abs_error": self.max_abs_error,
            "l2_error": self.l2_error,
            "tolerance": self.tolerance,
            "passed": self.passed,
            "func_grad_supported": self.func_grad_supported,
            "func_vmap_supported": self.func_vmap_supported,
            "func_jacrev_supported": self.func_jacrev_supported,
            "native_framework_autodiff": self.native_framework_autodiff,
            "host_boundary": self.host_boundary,
            "method": self.method,
            "claim_boundary": self.claim_boundary,
            "torch_value_type": type(self.torch_value).__name__,
            "torch_gradient_type": type(self.torch_gradient).__name__,
            "torch_jacrev_gradient_type": type(self.torch_jacrev_gradient).__name__,
            "torch_vmap_gradients_type": type(self.torch_vmap_gradients).__name__,
        }


@dataclass(frozen=True)
class PhaseTorchPhaseQNodeCompileResult:
    """Native ``torch.compile`` evidence for registered local Phase-QNode circuits."""

    value: float
    gradient: FloatArray
    parameter_shift_value: float
    parameter_shift_gradient: FloatArray
    torch_value: Any
    torch_gradient: Any
    max_abs_error: float
    l2_error: float
    tolerance: float
    passed: bool
    torch_compile_supported: bool
    compiled_value_supported: bool
    compiled_gradient_supported: bool
    fullgraph: bool
    dynamic: bool
    native_framework_autodiff: bool = True
    host_boundary: bool = False
    method: str = "torch_native_registered_phase_qnode_compile_audit"
    claim_boundary: str = "registered_phase_qnode_torch_compile_lowering"

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready registered Phase-QNode PyTorch compile evidence."""
        return {
            "value": self.value,
            "gradient": self.gradient.tolist(),
            "parameter_shift_value": self.parameter_shift_value,
            "parameter_shift_gradient": self.parameter_shift_gradient.tolist(),
            "max_abs_error": self.max_abs_error,
            "l2_error": self.l2_error,
            "tolerance": self.tolerance,
            "passed": self.passed,
            "torch_compile_supported": self.torch_compile_supported,
            "compiled_value_supported": self.compiled_value_supported,
            "compiled_gradient_supported": self.compiled_gradient_supported,
            "fullgraph": self.fullgraph,
            "dynamic": self.dynamic,
            "native_framework_autodiff": self.native_framework_autodiff,
            "host_boundary": self.host_boundary,
            "method": self.method,
            "claim_boundary": self.claim_boundary,
            "torch_value_type": type(self.torch_value).__name__,
            "torch_gradient_type": type(self.torch_gradient).__name__,
        }


@dataclass(frozen=True)
class PhaseTorchCompileBoundaryRoute:
    """One classified PyTorch compiler boundary for registered Phase-QNode routes."""

    name: str
    status: str
    reason: str
    execution_passed: bool
    fullgraph: bool
    dynamic: bool
    requires: tuple[str, ...] = ()
    value: float | None = None
    max_abs_reference_error: float | None = None
    exception_type: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready compiler-boundary route metadata."""
        return {
            "name": self.name,
            "status": self.status,
            "reason": self.reason,
            "execution_passed": self.execution_passed,
            "fullgraph": self.fullgraph,
            "dynamic": self.dynamic,
            "requires": list(self.requires),
            "value": self.value,
            "max_abs_reference_error": self.max_abs_reference_error,
            "exception_type": self.exception_type,
        }


@dataclass(frozen=True)
class PhaseTorchCompileBoundaryAuditResult:
    """Fail-closed PyTorch compiler-boundary evidence for registered Phase-QNodes."""

    routes: tuple[PhaseTorchCompileBoundaryRoute, ...]
    non_fullgraph_value: float
    non_fullgraph_gradient: FloatArray
    parameter_shift_value: float
    parameter_shift_gradient: FloatArray
    max_abs_reference_error: float
    tolerance: float
    torch_version: str
    passed: bool
    persistent_export_claim: bool
    provider_claim: bool
    performance_claim: bool
    method: str = "torch_registered_phase_qnode_compile_boundary_audit"
    claim_boundary: str = (
        "registered Phase-QNode PyTorch compile-boundary diagnostic for local "
        "CPU non-fullgraph execution only; dynamic-shape, fullgraph compiled-frame, "
        "AOTAutograd, torch.export persistent export, provider, hardware, CUDA, "
        "isolated benchmark, and performance promotion remain blocked, with no "
        "persistent export claim"
    )

    @property
    def non_fullgraph_passed(self) -> bool:
        """Return whether the non-fullgraph execution baseline passed."""
        return self.route_status("non_fullgraph_compile") == "passed"

    @property
    def open_gaps(self) -> tuple[str, ...]:
        """Return compile-boundary routes that remain blocked."""
        return tuple(route.name for route in self.routes if route.status != "passed")

    def route_status(self, name: str) -> str:
        """Return the status for a named compile-boundary route."""
        for route in self.routes:
            if route.name == name:
                return route.status
        raise KeyError(f"unknown PyTorch compile-boundary route: {name}")

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready compile-boundary audit evidence."""
        return {
            "routes": {route.name: route.to_dict() for route in self.routes},
            "open_gaps": list(self.open_gaps),
            "non_fullgraph_value": self.non_fullgraph_value,
            "non_fullgraph_gradient": self.non_fullgraph_gradient.tolist(),
            "parameter_shift_value": self.parameter_shift_value,
            "parameter_shift_gradient": self.parameter_shift_gradient.tolist(),
            "max_abs_reference_error": self.max_abs_reference_error,
            "tolerance": self.tolerance,
            "torch_version": self.torch_version,
            "passed": self.passed,
            "persistent_export_claim": self.persistent_export_claim,
            "provider_claim": self.provider_claim,
            "performance_claim": self.performance_claim,
            "method": self.method,
            "claim_boundary": self.claim_boundary,
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
class PhaseTorchEcosystemMaturityRoute:
    """One PyTorch ecosystem capability route and its claim boundary."""

    name: str
    status: str
    reason: str
    requires: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready PyTorch ecosystem route metadata."""
        return {
            "name": self.name,
            "status": self.status,
            "reason": self.reason,
            "requires": list(self.requires),
        }


@dataclass(frozen=True)
class PhaseTorchEcosystemMaturityAuditResult:
    """PyTorch module, transform, compiler, and device maturity evidence."""

    torch_version: str
    cuda_available: bool
    cuda_device_count: int
    cuda_device_names: tuple[str, ...]
    routes: tuple[PhaseTorchEcosystemMaturityRoute, ...]
    claim_boundary: str = "torch_ecosystem_device_maturity_audit"

    @property
    def open_gaps(self) -> tuple[str, ...]:
        """Return PyTorch ecosystem routes that remain blocked."""
        return tuple(route.name for route in self.routes if route.status != "passed")

    @property
    def ready_for_provider_exceedance(self) -> bool:
        """Return whether PyTorch ecosystem evidence permits provider exceedance."""
        return not self.open_gaps

    def route_status(self, name: str) -> str:
        """Return the status for a named PyTorch ecosystem route."""
        for route in self.routes:
            if route.name == name:
                return route.status
        raise KeyError(f"unknown PyTorch ecosystem maturity route: {name}")

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready PyTorch ecosystem maturity evidence."""
        return {
            "torch_version": self.torch_version,
            "cuda_available": self.cuda_available,
            "cuda_device_count": self.cuda_device_count,
            "cuda_device_names": list(self.cuda_device_names),
            "routes": {route.name: route.to_dict() for route in self.routes},
            "open_gaps": list(self.open_gaps),
            "ready_for_provider_exceedance": self.ready_for_provider_exceedance,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class PhaseTorchCloudValidationRunSpec:
    """Cloud validation batch plan for PyTorch compiler and device promotion.

    The plan records why the local workstation is or is not suitable for CUDA
    validation, which PyTorch Phase-QNode routes remain blocked locally, and the
    exact artefact classes that must be produced on a compatible cloud runner
    before any provider-exceedance or accelerator-performance claim is made.
    """

    runner: str
    local_execution_status: str
    local_skip_reason: str
    torch_version: str
    cuda_available: bool
    cuda_device_count: int
    cuda_device_names: tuple[str, ...]
    blocked_local_routes: tuple[str, ...]
    required_artifacts: tuple[str, ...]
    required_environment: dict[str, object]
    commands: tuple[str, ...]
    ready_for_cloud_dispatch: bool
    claim_boundary: str = "torch_cloud_validation_batch_plan"

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready PyTorch cloud validation scheduling metadata."""
        return {
            "runner": self.runner,
            "local_execution_status": self.local_execution_status,
            "local_skip_reason": self.local_skip_reason,
            "torch_version": self.torch_version,
            "cuda_available": self.cuda_available,
            "cuda_device_count": self.cuda_device_count,
            "cuda_device_names": list(self.cuda_device_names),
            "blocked_local_routes": list(self.blocked_local_routes),
            "required_artifacts": list(self.required_artifacts),
            "required_environment": dict(self.required_environment),
            "commands": list(self.commands),
            "ready_for_cloud_dispatch": self.ready_for_cloud_dispatch,
            "claim_boundary": self.claim_boundary,
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
    """Fail-closed PyTorch parity matrix for registered Phase-QNode routes."""

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
    "PhaseTorchCloudValidationRunSpec",
    "PhaseTorchCompileBoundaryAuditResult",
    "PhaseTorchCompileBoundaryRoute",
    "PhaseTorchCompileCompatibilityResult",
    "PhaseTorchEcosystemMaturityAuditResult",
    "PhaseTorchEcosystemMaturityRoute",
    "PhaseTorchFuncCompatibilityResult",
    "PhaseTorchLiveOverlayEvidence",
    "PhaseTorchMaturityAuditResult",
    "PhaseTorchModuleWrapperAuditResult",
    "PhaseTorchParameterShiftResult",
    "PhaseTorchPhaseQNodeCompileResult",
    "PhaseTorchPhaseQNodeLoweringMatrixResult",
    "PhaseTorchPhaseQNodeLoweringRoute",
    "PhaseTorchPhaseQNodeStatevectorResult",
    "PhaseTorchPhaseQNodeTransformResult",
    "PhaseTorchQNNGradientResult",
    "PhaseTorchTrainingLoopAuditResult",
]
