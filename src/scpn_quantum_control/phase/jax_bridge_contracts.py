# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — JAX Bridge Result Contracts
"""Immutable result contracts for the optional phase JAX bridge.

This NumPy/stdlib leaf owns serialization-ready evidence records while the
``jax_bridge`` compatibility module owns optional JAX loading and execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]


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
class PhaseJAXPhaseQNodeStatevectorResult:
    """Native JAX autodiff evidence for a registered local Phase-QNode."""

    value: float
    gradient: FloatArray
    state: NDArray[np.complex128]
    parameter_shift_value: float
    parameter_shift_gradient: FloatArray
    max_abs_error: float
    l2_error: float
    tolerance: float
    passed: bool
    native_framework_autodiff: bool
    host_callback: bool
    jit_requested: bool
    jitted: bool
    method: str = "jax_native_registered_phase_qnode_statevector_value_and_grad"
    claim_boundary: str = "registered_phase_qnode_jax_statevector_lowering"

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready registered Phase-QNode JAX lowering evidence."""
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
            "host_callback": self.host_callback,
            "jit_requested": self.jit_requested,
            "jitted": self.jitted,
            "method": self.method,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class PhaseJAXPhaseQNodeNativeTransformResult:
    """Native JAX transform evidence for a registered local Phase-QNode."""

    value: float
    gradient: FloatArray
    value_and_grad_value: float
    value_and_grad_gradient: FloatArray
    jacfwd_gradient: FloatArray
    jacrev_gradient: FloatArray
    hessian: FloatArray
    jvp_value: float
    jvp_tangent: float
    vjp_value: float
    vjp_cotangent_gradient: FloatArray
    vmap_values: FloatArray
    vmap_gradients: FloatArray
    parameter_shift_value: float
    parameter_shift_gradient: FloatArray
    tangent: FloatArray
    batch_params: FloatArray
    batch_parameter_shift_gradients: FloatArray
    max_abs_gradient_error: float
    max_abs_transform_error: float
    max_abs_hessian_symmetry_error: float
    tolerance: float
    passed: bool
    native_framework_autodiff: bool
    host_callback: bool
    jit_value_and_grad: bool
    vmap_value_and_grad: bool
    transform_names: tuple[str, ...]
    method: str = "jax_native_registered_phase_qnode_transform_audit"
    claim_boundary: str = "registered_phase_qnode_jax_native_transform_lowering"

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready registered Phase-QNode JAX transform evidence."""
        return {
            "value": self.value,
            "gradient": self.gradient.tolist(),
            "value_and_grad_value": self.value_and_grad_value,
            "value_and_grad_gradient": self.value_and_grad_gradient.tolist(),
            "jacfwd_gradient": self.jacfwd_gradient.tolist(),
            "jacrev_gradient": self.jacrev_gradient.tolist(),
            "hessian": self.hessian.tolist(),
            "jvp_value": self.jvp_value,
            "jvp_tangent": self.jvp_tangent,
            "vjp_value": self.vjp_value,
            "vjp_cotangent_gradient": self.vjp_cotangent_gradient.tolist(),
            "vmap_values": self.vmap_values.tolist(),
            "vmap_gradients": self.vmap_gradients.tolist(),
            "parameter_shift_value": self.parameter_shift_value,
            "parameter_shift_gradient": self.parameter_shift_gradient.tolist(),
            "tangent": self.tangent.tolist(),
            "batch_params": self.batch_params.tolist(),
            "batch_parameter_shift_gradients": self.batch_parameter_shift_gradients.tolist(),
            "max_abs_gradient_error": self.max_abs_gradient_error,
            "max_abs_transform_error": self.max_abs_transform_error,
            "max_abs_hessian_symmetry_error": self.max_abs_hessian_symmetry_error,
            "tolerance": self.tolerance,
            "passed": self.passed,
            "native_framework_autodiff": self.native_framework_autodiff,
            "host_callback": self.host_callback,
            "jit_value_and_grad": self.jit_value_and_grad,
            "vmap_value_and_grad": self.vmap_value_and_grad,
            "transform_names": list(self.transform_names),
            "method": self.method,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class PhaseJAXPhaseQNodePyTreeTransformResult:
    """Native JAX PyTree transform evidence for a registered local Phase-QNode."""

    value: float
    gradient: FloatArray
    gradient_pytree: object
    value_and_grad_value: float
    value_and_grad_gradient: FloatArray
    value_and_grad_gradient_pytree: object
    jacfwd_gradient: FloatArray
    jacrev_gradient: FloatArray
    hessian: FloatArray
    hessian_pytree: object
    jvp_value: float
    jvp_tangent: float
    vjp_value: float
    vjp_cotangent_gradient: FloatArray
    vmap_values: FloatArray
    vmap_gradients: FloatArray
    parameter_shift_value: float
    parameter_shift_gradient: FloatArray
    parameter_vector: FloatArray
    tangent: FloatArray
    batch_params: FloatArray
    batch_parameter_shift_gradients: FloatArray
    leaf_shapes: tuple[tuple[int, ...], ...]
    max_abs_gradient_error: float
    max_abs_transform_error: float
    max_abs_hessian_symmetry_error: float
    tolerance: float
    passed: bool
    native_framework_autodiff: bool
    host_callback: bool
    jit_value_and_grad: bool
    vmap_value_and_grad: bool
    transform_names: tuple[str, ...]
    method: str = "jax_native_registered_phase_qnode_pytree_transform_audit"
    claim_boundary: str = "registered_phase_qnode_jax_pytree_transform_lowering"

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready registered Phase-QNode JAX PyTree evidence."""
        return {
            "value": self.value,
            "gradient": self.gradient.tolist(),
            "gradient_pytree": _json_ready_pytree(self.gradient_pytree),
            "value_and_grad_value": self.value_and_grad_value,
            "value_and_grad_gradient": self.value_and_grad_gradient.tolist(),
            "value_and_grad_gradient_pytree": _json_ready_pytree(
                self.value_and_grad_gradient_pytree
            ),
            "jacfwd_gradient": self.jacfwd_gradient.tolist(),
            "jacrev_gradient": self.jacrev_gradient.tolist(),
            "hessian": self.hessian.tolist(),
            "hessian_pytree": _json_ready_pytree(self.hessian_pytree),
            "jvp_value": self.jvp_value,
            "jvp_tangent": self.jvp_tangent,
            "vjp_value": self.vjp_value,
            "vjp_cotangent_gradient": self.vjp_cotangent_gradient.tolist(),
            "vmap_values": self.vmap_values.tolist(),
            "vmap_gradients": self.vmap_gradients.tolist(),
            "parameter_shift_value": self.parameter_shift_value,
            "parameter_shift_gradient": self.parameter_shift_gradient.tolist(),
            "parameter_vector": self.parameter_vector.tolist(),
            "tangent": self.tangent.tolist(),
            "batch_params": self.batch_params.tolist(),
            "batch_parameter_shift_gradients": self.batch_parameter_shift_gradients.tolist(),
            "leaf_shapes": [list(shape) for shape in self.leaf_shapes],
            "max_abs_gradient_error": self.max_abs_gradient_error,
            "max_abs_transform_error": self.max_abs_transform_error,
            "max_abs_hessian_symmetry_error": self.max_abs_hessian_symmetry_error,
            "tolerance": self.tolerance,
            "passed": self.passed,
            "native_framework_autodiff": self.native_framework_autodiff,
            "host_callback": self.host_callback,
            "jit_value_and_grad": self.jit_value_and_grad,
            "vmap_value_and_grad": self.vmap_value_and_grad,
            "transform_names": list(self.transform_names),
            "method": self.method,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class PhaseJAXPhaseQNodeShardingTransformResult:
    """Native JAX PMAP evidence for registered local Phase-QNode batches."""

    values: FloatArray
    gradients: FloatArray
    parameter_shift_values: FloatArray
    parameter_shift_gradients: FloatArray
    batch_params: FloatArray
    batch_size: int
    local_device_count: int
    device_descriptions: tuple[str, ...]
    sharding_mode: str
    max_abs_value_error: float
    max_abs_gradient_error: float
    tolerance: float
    passed: bool
    native_framework_autodiff: bool
    host_callback: bool
    pmapped: bool
    method: str = "jax_native_registered_phase_qnode_pmap_sharding_audit"
    claim_boundary: str = "registered_phase_qnode_jax_pmap_sharding_lowering"

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready registered Phase-QNode JAX PMAP evidence."""
        return {
            "values": self.values.tolist(),
            "gradients": self.gradients.tolist(),
            "parameter_shift_values": self.parameter_shift_values.tolist(),
            "parameter_shift_gradients": self.parameter_shift_gradients.tolist(),
            "batch_params": self.batch_params.tolist(),
            "batch_size": self.batch_size,
            "local_device_count": self.local_device_count,
            "device_descriptions": list(self.device_descriptions),
            "sharding_mode": self.sharding_mode,
            "max_abs_value_error": self.max_abs_value_error,
            "max_abs_gradient_error": self.max_abs_gradient_error,
            "tolerance": self.tolerance,
            "passed": self.passed,
            "native_framework_autodiff": self.native_framework_autodiff,
            "host_callback": self.host_callback,
            "pmapped": self.pmapped,
            "method": self.method,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class PhaseJAXPhaseQNodeAOTExportResult:
    """JAX AOT/export diagnostic for a registered local Phase-QNode value route."""

    value: float
    compiled_value: float
    deserialized_value: float
    parameter_shift_value: float
    max_abs_value_error: float
    tolerance: float
    passed: bool
    lowered: bool
    compiled: bool
    exported: bool
    serialized: bool
    deserialized_call: bool
    host_callback: bool
    parameter_shape: tuple[int, ...]
    parameter_dtype: str
    compiler_ir_dialects: tuple[str, ...]
    lowered_text_bytes: int
    mlir_module_bytes: int
    serialized_bytes: int
    export_platforms: tuple[str, ...]
    calling_convention_version: int
    minimum_supported_calling_convention_version: int
    maximum_supported_calling_convention_version: int
    disabled_safety_checks: tuple[str, ...]
    uses_global_constants: bool
    persistent_export_claim: bool = False
    method: str = "jax_registered_phase_qnode_aot_export_audit"
    claim_boundary: str = "registered_phase_qnode_jax_aot_export_diagnostic"

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready registered Phase-QNode JAX AOT/export evidence."""
        return {
            "value": self.value,
            "compiled_value": self.compiled_value,
            "deserialized_value": self.deserialized_value,
            "parameter_shift_value": self.parameter_shift_value,
            "max_abs_value_error": self.max_abs_value_error,
            "tolerance": self.tolerance,
            "passed": self.passed,
            "lowered": self.lowered,
            "compiled": self.compiled,
            "exported": self.exported,
            "serialized": self.serialized,
            "deserialized_call": self.deserialized_call,
            "host_callback": self.host_callback,
            "parameter_shape": list(self.parameter_shape),
            "parameter_dtype": self.parameter_dtype,
            "compiler_ir_dialects": list(self.compiler_ir_dialects),
            "lowered_text_bytes": self.lowered_text_bytes,
            "mlir_module_bytes": self.mlir_module_bytes,
            "serialized_bytes": self.serialized_bytes,
            "export_platforms": list(self.export_platforms),
            "calling_convention_version": self.calling_convention_version,
            "minimum_supported_calling_convention_version": (
                self.minimum_supported_calling_convention_version
            ),
            "maximum_supported_calling_convention_version": (
                self.maximum_supported_calling_convention_version
            ),
            "disabled_safety_checks": list(self.disabled_safety_checks),
            "uses_global_constants": self.uses_global_constants,
            "persistent_export_claim": self.persistent_export_claim,
            "method": self.method,
            "claim_boundary": self.claim_boundary,
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


@dataclass(frozen=True)
class PhaseJAXMaturityAuditResult:
    """Aggregate JAX maturity evidence and explicit provider-parity blockers."""

    bounded_model_ready: bool
    ready_for_provider_exceedance: bool
    evidence: dict[str, object]
    required_capabilities: dict[str, str]
    open_gaps: tuple[str, ...]
    claim_boundary: str = "bounded_jax_provider_maturity_audit"

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready JAX maturity evidence."""
        return {
            "bounded_model_ready": self.bounded_model_ready,
            "ready_for_provider_exceedance": self.ready_for_provider_exceedance,
            "evidence": {name: _result_to_dict(result) for name, result in self.evidence.items()},
            "required_capabilities": dict(self.required_capabilities),
            "open_gaps": list(self.open_gaps),
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class PhaseJAXNestedTransformRoute:
    """One bounded JAX nested-transform route or explicit blocker."""

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
class PhaseJAXNestedTransformAlgebraResult:
    """Bounded JAX nested-transform evidence plus provider-exceedance blockers."""

    jit_under_vmap_gradients: FloatArray
    jit_vmap_gradients: FloatArray
    pytree_gradient_vector: FloatArray
    parameter_shift_batch_gradients: FloatArray
    parameter_shift_pytree_gradient: FloatArray
    max_abs_error: float
    l2_error: float
    tolerance: float
    routes: tuple[PhaseJAXNestedTransformRoute, ...]
    claim_boundary: str = "bounded_jax_nested_transform_algebra"

    @property
    def passed(self) -> bool:
        """Return whether every implemented bounded transform check agrees."""
        return self.max_abs_error <= self.tolerance

    @property
    def bounded_transform_algebra_ready(self) -> bool:
        """Return whether bounded JAX transform routes pass."""
        return self.passed and all(
            route.status == "passed"
            for route in self.routes
            if route.name.startswith(("jit_", "pytree_", "vmap_"))
        )

    @property
    def ready_for_provider_exceedance(self) -> bool:
        """Return whether this audit permits JAX provider-exceedance claims."""
        return self.bounded_transform_algebra_ready and all(
            route.status == "passed" for route in self.routes
        )

    @property
    def open_gaps(self) -> tuple[str, ...]:
        """Return non-passing transform routes."""
        return tuple(route.name for route in self.routes if route.status != "passed")

    def route_status(self, name: str) -> str:
        """Return a route status, failing closed on unknown routes."""
        for route in self.routes:
            if route.name == name:
                return route.status
        raise KeyError(f"unknown JAX nested-transform route: {name}")

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready JAX nested-transform evidence."""
        return {
            "jit_under_vmap_gradients": self.jit_under_vmap_gradients.copy(),
            "jit_vmap_gradients": self.jit_vmap_gradients.copy(),
            "pytree_gradient_vector": self.pytree_gradient_vector.copy(),
            "parameter_shift_batch_gradients": self.parameter_shift_batch_gradients.copy(),
            "parameter_shift_pytree_gradient": self.parameter_shift_pytree_gradient.copy(),
            "max_abs_error": self.max_abs_error,
            "l2_error": self.l2_error,
            "tolerance": self.tolerance,
            "passed": self.passed,
            "bounded_transform_algebra_ready": self.bounded_transform_algebra_ready,
            "ready_for_provider_exceedance": self.ready_for_provider_exceedance,
            "routes": {route.name: route.to_dict() for route in self.routes},
            "open_gaps": list(self.open_gaps),
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class PhaseJAXCloudValidationRunSpec:
    """Cloud validation batch plan for JAX device and sharding promotion.

    The plan records local JAX device evidence, why local GTX 1060 or
    single-device runs cannot promote accelerator routes, and the exact
    artefacts that a compatible cloud runner must produce before any JAX GPU,
    multi-device, or accelerator-performance claim is made.
    """

    runner: str
    local_execution_status: str
    local_skip_reason: str
    accelerator_backend: str
    local_device_count: int
    device_descriptions: tuple[str, ...]
    blocked_local_routes: tuple[str, ...]
    required_artifacts: tuple[str, ...]
    required_environment: dict[str, object]
    commands: tuple[str, ...]
    ready_for_cloud_dispatch: bool
    claim_boundary: str = "jax_cloud_validation_batch_plan"

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready JAX cloud validation scheduling metadata."""
        return {
            "runner": self.runner,
            "local_execution_status": self.local_execution_status,
            "local_skip_reason": self.local_skip_reason,
            "accelerator_backend": self.accelerator_backend,
            "local_device_count": self.local_device_count,
            "device_descriptions": list(self.device_descriptions),
            "blocked_local_routes": list(self.blocked_local_routes),
            "required_artifacts": list(self.required_artifacts),
            "required_environment": dict(self.required_environment),
            "commands": list(self.commands),
            "ready_for_cloud_dispatch": self.ready_for_cloud_dispatch,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class PhaseJAXPhaseQNodeLoweringRoute:
    """One JAX lowering route in the registered Phase-QNode parity matrix."""

    name: str
    status: str
    reason: str
    host_callback: bool
    requires: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready route metadata."""
        return {
            "name": self.name,
            "status": self.status,
            "reason": self.reason,
            "host_callback": self.host_callback,
            "requires": list(self.requires),
        }


@dataclass(frozen=True)
class PhaseJAXPhaseQNodeLoweringMatrixResult:
    """Fail-closed JAX parity matrix for registered Phase-QNode lowering."""

    routes: tuple[PhaseJAXPhaseQNodeLoweringRoute, ...]
    claim_boundary: str = "bounded_jax_phase_qnode_lowering_matrix"

    @property
    def bounded_no_host_callback_routes_ready(self) -> bool:
        """Return whether bounded JAX QNN routes pass without host callbacks."""
        return all(
            route.status == "passed" and not route.host_callback
            for route in self.routes
            if route.name.startswith("bounded_qnn_")
        )

    @property
    def arbitrary_phase_qnode_lowering_ready(self) -> bool:
        """Return whether arbitrary registered Phase-QNode JAX lowering is ready."""
        return all(
            route.status == "passed" and not route.host_callback
            for route in self.routes
            if route.name.startswith("registered_phase_qnode_")
            and route.name.endswith("_lowering")
            and route.name
            not in {
                "registered_phase_qnode_finite_shot_lowering",
                "registered_phase_qnode_provider_lowering",
                "registered_phase_qnode_hardware_lowering",
                "registered_phase_qnode_dynamic_circuit_lowering",
            }
        )

    @property
    def ready_for_provider_exceedance(self) -> bool:
        """Return whether this matrix permits JAX provider-exceedance claims."""
        return self.bounded_no_host_callback_routes_ready and all(
            route.status == "passed" and not route.host_callback for route in self.routes
        )

    @property
    def open_gaps(self) -> tuple[str, ...]:
        """Return routes that still block JAX provider-exceedance claims."""
        return tuple(route.name for route in self.routes if route.status != "passed")

    def route_status(self, name: str) -> str:
        """Return the status for a named route, failing closed on unknown names."""
        for route in self.routes:
            if route.name == name:
                return route.status
        raise KeyError(f"unknown JAX Phase-QNode lowering route: {name}")

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready JAX Phase-QNode lowering parity metadata."""
        return {
            "bounded_no_host_callback_routes_ready": (self.bounded_no_host_callback_routes_ready),
            "arbitrary_phase_qnode_lowering_ready": self.arbitrary_phase_qnode_lowering_ready,
            "ready_for_provider_exceedance": self.ready_for_provider_exceedance,
            "routes": {route.name: route.to_dict() for route in self.routes},
            "open_gaps": list(self.open_gaps),
            "claim_boundary": self.claim_boundary,
        }


def _json_ready_pytree(tree: object) -> object:
    if isinstance(tree, dict):
        return {str(key): _json_ready_pytree(value) for key, value in tree.items()}
    if isinstance(tree, tuple):
        return [_json_ready_pytree(value) for value in tree]
    if isinstance(tree, list):
        return [_json_ready_pytree(value) for value in tree]
    return np.asarray(tree, dtype=float).tolist()


def _result_to_dict(result: object) -> object:
    to_dict = getattr(result, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    return result


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
    "PhaseJAXPhaseQNodeAOTExportResult",
    "PhaseJAXPhaseQNodeLoweringMatrixResult",
    "PhaseJAXPhaseQNodeLoweringRoute",
    "PhaseJAXPhaseQNodeNativeTransformResult",
    "PhaseJAXPhaseQNodePyTreeTransformResult",
    "PhaseJAXPhaseQNodeShardingTransformResult",
    "PhaseJAXPhaseQNodeStatevectorResult",
    "PhaseJAXPyTreeCompatibilityResult",
    "PhaseJAXShardingCompatibilityResult",
    "PhaseJAXVMAPCompatibilityResult",
]
