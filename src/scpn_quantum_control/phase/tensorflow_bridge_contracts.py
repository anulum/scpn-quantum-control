# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — TensorFlow Bridge Contracts
"""TensorFlow bridge result, compatibility, lowering, and maturity records.

This dependency-free declaration leaf owns immutable records, the shared float
array alias, and JSON-ready result serialization. It contains no TensorFlow
loading, gradient execution, compatibility, lowering execution, provider,
hardware, benchmark, or publication orchestration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]


@dataclass(frozen=True)
class PhaseTensorFlowParameterShiftResult:
    """Result from the optional TensorFlow phase parameter-shift bridge."""

    value: float
    gradient: FloatArray
    tensorflow_value: Any
    tensorflow_gradient: Any
    method: str
    evaluations: int
    host_boundary: bool
    shift_terms: int = 1

    def to_dict(self) -> dict[str, object]:
        """Return JSON-serialisable TensorFlow interop metadata."""
        return {
            "value": self.value,
            "gradient": self.gradient.copy(),
            "method": self.method,
            "evaluations": self.evaluations,
            "host_boundary": self.host_boundary,
            "shift_terms": self.shift_terms,
            "tensorflow_value_type": type(self.tensorflow_value).__name__,
            "tensorflow_gradient_type": type(self.tensorflow_gradient).__name__,
        }


@dataclass(frozen=True)
class PhaseTensorFlowQNNGradientResult:
    """Tensor-ready bounded phase-QNN gradient evidence for TensorFlow workflows."""

    loss: float
    gradient: FloatArray
    parameter_shift_gradient: FloatArray
    tensorflow_loss: Any
    tensorflow_gradient: Any
    tensorflow_parameter_shift_gradient: Any
    max_abs_error: float
    l2_error: float
    tolerance: float
    passed: bool
    method: str = "tensorflow_bounded_phase_qnn_analytic_value_and_grad"
    host_boundary: bool = False
    native_framework_autodiff: bool = False
    analytic_framework_gradient: bool = True

    def to_dict(self) -> dict[str, object]:
        """Return JSON-serialisable TensorFlow bounded-QNN gradient metadata."""
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
            "tensorflow_loss_type": type(self.tensorflow_loss).__name__,
            "tensorflow_gradient_type": type(self.tensorflow_gradient).__name__,
            "tensorflow_parameter_shift_gradient_type": type(
                self.tensorflow_parameter_shift_gradient,
            ).__name__,
        }


@dataclass(frozen=True)
class PhaseTensorFlowGradientTapeCompatibilityResult:
    """Bounded phase-QNN compatibility evidence for TensorFlow ``GradientTape``."""

    loss: float
    gradient: FloatArray
    parameter_shift_gradient: FloatArray
    tensorflow_loss: Any
    tensorflow_gradient: Any
    max_abs_error: float
    l2_error: float
    tolerance: float
    passed: bool
    gradient_tape_supported: bool
    native_framework_autodiff: bool = True
    host_boundary: bool = False
    claim_boundary: str = "bounded_tensorflow_gradient_tape_compatibility"

    def to_dict(self) -> dict[str, object]:
        """Return JSON-serialisable TensorFlow ``GradientTape`` evidence."""
        return {
            "loss": self.loss,
            "gradient": self.gradient.copy(),
            "parameter_shift_gradient": self.parameter_shift_gradient.copy(),
            "max_abs_error": self.max_abs_error,
            "l2_error": self.l2_error,
            "tolerance": self.tolerance,
            "passed": self.passed,
            "gradient_tape_supported": self.gradient_tape_supported,
            "native_framework_autodiff": self.native_framework_autodiff,
            "host_boundary": self.host_boundary,
            "claim_boundary": self.claim_boundary,
            "tensorflow_loss_type": type(self.tensorflow_loss).__name__,
            "tensorflow_gradient_type": type(self.tensorflow_gradient).__name__,
        }


@dataclass(frozen=True)
class PhaseTensorFlowFunctionCompatibilityResult:
    """Bounded phase-QNN compatibility evidence for TensorFlow ``tf.function``."""

    loss: float
    gradient: FloatArray
    parameter_shift_gradient: FloatArray
    tensorflow_loss: Any
    tensorflow_gradient: Any
    max_abs_error: float
    l2_error: float
    tolerance: float
    passed: bool
    function_supported: bool
    gradient_tape_supported: bool
    native_framework_autodiff: bool = True
    host_boundary: bool = False
    claim_boundary: str = "bounded_tensorflow_function_compatibility"

    def to_dict(self) -> dict[str, object]:
        """Return JSON-serialisable TensorFlow ``tf.function`` evidence."""
        return {
            "loss": self.loss,
            "gradient": self.gradient.copy(),
            "parameter_shift_gradient": self.parameter_shift_gradient.copy(),
            "max_abs_error": self.max_abs_error,
            "l2_error": self.l2_error,
            "tolerance": self.tolerance,
            "passed": self.passed,
            "function_supported": self.function_supported,
            "gradient_tape_supported": self.gradient_tape_supported,
            "native_framework_autodiff": self.native_framework_autodiff,
            "host_boundary": self.host_boundary,
            "claim_boundary": self.claim_boundary,
            "tensorflow_loss_type": type(self.tensorflow_loss).__name__,
            "tensorflow_gradient_type": type(self.tensorflow_gradient).__name__,
        }


@dataclass(frozen=True)
class PhaseTensorFlowXLACompatibilityResult:
    """Bounded phase-QNN compatibility evidence for TensorFlow XLA JIT."""

    loss: float
    gradient: FloatArray
    parameter_shift_gradient: FloatArray
    tensorflow_loss: Any
    tensorflow_gradient: Any
    max_abs_error: float
    l2_error: float
    tolerance: float
    passed: bool
    function_supported: bool
    gradient_tape_supported: bool
    xla_compile_requested: bool
    native_framework_autodiff: bool = True
    host_boundary: bool = False
    claim_boundary: str = "bounded_tensorflow_xla_compatibility"

    def to_dict(self) -> dict[str, object]:
        """Return JSON-serialisable TensorFlow XLA compatibility evidence."""
        return {
            "loss": self.loss,
            "gradient": self.gradient.copy(),
            "parameter_shift_gradient": self.parameter_shift_gradient.copy(),
            "max_abs_error": self.max_abs_error,
            "l2_error": self.l2_error,
            "tolerance": self.tolerance,
            "passed": self.passed,
            "function_supported": self.function_supported,
            "gradient_tape_supported": self.gradient_tape_supported,
            "xla_compile_requested": self.xla_compile_requested,
            "native_framework_autodiff": self.native_framework_autodiff,
            "host_boundary": self.host_boundary,
            "claim_boundary": self.claim_boundary,
            "tensorflow_loss_type": type(self.tensorflow_loss).__name__,
            "tensorflow_gradient_type": type(self.tensorflow_gradient).__name__,
        }


@dataclass(frozen=True)
class PhaseTensorFlowKerasLayerWrapperAuditResult:
    """Bounded phase-QNN compatibility evidence for a TensorFlow Keras layer."""

    loss: float
    gradient: FloatArray
    parameter_shift_gradient: FloatArray
    tensorflow_layer: Any
    tensorflow_loss: Any
    tensorflow_gradient: Any
    max_abs_error: float
    l2_error: float
    tolerance: float
    passed: bool
    keras_layer_supported: bool
    gradient_tape_supported: bool
    trainable_parameters: int
    native_framework_autodiff: bool = True
    host_boundary: bool = False
    claim_boundary: str = "bounded_tensorflow_keras_layer_wrapper"

    def to_dict(self) -> dict[str, object]:
        """Return JSON-serialisable TensorFlow Keras layer-wrapper evidence."""
        return {
            "loss": self.loss,
            "gradient": self.gradient.copy(),
            "parameter_shift_gradient": self.parameter_shift_gradient.copy(),
            "max_abs_error": self.max_abs_error,
            "l2_error": self.l2_error,
            "tolerance": self.tolerance,
            "passed": self.passed,
            "keras_layer_supported": self.keras_layer_supported,
            "gradient_tape_supported": self.gradient_tape_supported,
            "trainable_parameters": self.trainable_parameters,
            "native_framework_autodiff": self.native_framework_autodiff,
            "host_boundary": self.host_boundary,
            "claim_boundary": self.claim_boundary,
            "tensorflow_layer_type": type(self.tensorflow_layer).__name__,
            "tensorflow_loss_type": type(self.tensorflow_loss).__name__,
            "tensorflow_gradient_type": type(self.tensorflow_gradient).__name__,
        }


@dataclass(frozen=True)
class PhaseTensorFlowMaturityAuditResult:
    """Aggregate TensorFlow maturity evidence and explicit provider blockers."""

    bounded_model_ready: bool
    ready_for_provider_exceedance: bool
    evidence: dict[str, object]
    required_capabilities: dict[str, str]
    open_gaps: tuple[str, ...]
    claim_boundary: str = "bounded_tensorflow_provider_maturity_audit"

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready TensorFlow maturity evidence."""
        return {
            "bounded_model_ready": self.bounded_model_ready,
            "ready_for_provider_exceedance": self.ready_for_provider_exceedance,
            "evidence": {name: _result_to_dict(result) for name, result in self.evidence.items()},
            "required_capabilities": dict(self.required_capabilities),
            "open_gaps": list(self.open_gaps),
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class PhaseTensorFlowPhaseQNodeLoweringRoute:
    """One TensorFlow route in the registered Phase-QNode parity matrix."""

    name: str
    status: str
    reason: str
    requires: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready TensorFlow route metadata."""
        return {
            "name": self.name,
            "status": self.status,
            "reason": self.reason,
            "requires": list(self.requires),
        }


@dataclass(frozen=True)
class PhaseTensorFlowPhaseQNodeLoweringMatrixResult:
    """Fail-closed TensorFlow parity matrix for arbitrary registered Phase-QNodes."""

    routes: tuple[PhaseTensorFlowPhaseQNodeLoweringRoute, ...]
    claim_boundary: str = "bounded_tensorflow_phase_qnode_lowering_matrix"

    @property
    def bounded_qnn_routes_ready(self) -> bool:
        """Return whether the bounded TensorFlow QNN routes are declared ready."""
        return all(
            route.status == "passed"
            for route in self.routes
            if route.name.startswith("bounded_qnn_")
        )

    @property
    def arbitrary_phase_qnode_lowering_ready(self) -> bool:
        """Return whether arbitrary registered Phase-QNode TensorFlow lowering is ready."""
        return all(
            route.status == "passed"
            for route in self.routes
            if route.name.startswith("registered_phase_qnode_")
        )

    @property
    def ready_for_provider_exceedance(self) -> bool:
        """Return whether this matrix permits TensorFlow provider-exceedance claims."""
        return all(route.status == "passed" for route in self.routes)

    @property
    def open_gaps(self) -> tuple[str, ...]:
        """Return routes that still block TensorFlow provider-exceedance claims."""
        return tuple(route.name for route in self.routes if route.status != "passed")

    def route_status(self, name: str) -> str:
        """Return the status for a named route, failing closed on unknown names."""
        for route in self.routes:
            if route.name == name:
                return route.status
        raise KeyError(f"unknown TensorFlow Phase-QNode lowering route: {name}")

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready TensorFlow Phase-QNode lowering parity metadata."""
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


__all__ = [
    "PhaseTensorFlowParameterShiftResult",
    "PhaseTensorFlowQNNGradientResult",
    "PhaseTensorFlowGradientTapeCompatibilityResult",
    "PhaseTensorFlowFunctionCompatibilityResult",
    "PhaseTensorFlowXLACompatibilityResult",
    "PhaseTensorFlowKerasLayerWrapperAuditResult",
    "PhaseTensorFlowMaturityAuditResult",
    "PhaseTensorFlowPhaseQNodeLoweringRoute",
    "PhaseTensorFlowPhaseQNodeLoweringMatrixResult",
]
