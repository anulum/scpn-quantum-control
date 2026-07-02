# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Bounded QNN Framework Bridge Matrix
"""Fail-closed support matrix for bounded phase-QNN framework bridges."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

EVIDENCE_CLASS = "bounded_qnn_framework_bridge_matrix"
CLAIM_BOUNDARY = (
    "declares bounded phase-QNN bridge support only; this is not arbitrary "
    "framework autodiff through simulator kernels, not device-placement "
    "certification, and not live provider hardware gradient execution"
)


@dataclass(frozen=True)
class BoundedQNNFrameworkBridgeCapability:
    """One bounded phase-QNN framework bridge capability declaration."""

    framework: str
    public_api: str
    gradient_route: str
    optional_dependency: str | None
    implemented: bool
    runtime_dependency_required: bool
    native_framework_autodiff: bool
    tensor_output: bool
    host_boundary: bool
    analytic_framework_gradient: bool
    fail_closed_reason: str | None
    evidence_class: str = EVIDENCE_CLASS
    claim_boundary: str = CLAIM_BOUNDARY

    @property
    def supported(self) -> bool:
        """Return whether the declared bridge route is implemented."""
        return self.implemented and self.fail_closed_reason is None

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready bridge capability metadata."""
        return {
            "framework": self.framework,
            "public_api": self.public_api,
            "gradient_route": self.gradient_route,
            "optional_dependency": self.optional_dependency,
            "implemented": self.implemented,
            "runtime_dependency_required": self.runtime_dependency_required,
            "supported": self.supported,
            "native_framework_autodiff": self.native_framework_autodiff,
            "tensor_output": self.tensor_output,
            "host_boundary": self.host_boundary,
            "analytic_framework_gradient": self.analytic_framework_gradient,
            "fail_closed_reason": self.fail_closed_reason,
            "evidence_class": self.evidence_class,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class BoundedQNNFrameworkBridgeMatrixResult:
    """Auditable bounded phase-QNN framework bridge matrix."""

    capabilities: tuple[BoundedQNNFrameworkBridgeCapability, ...]
    evidence_class: str = EVIDENCE_CLASS
    claim_boundary: str = CLAIM_BOUNDARY

    @property
    def framework_count(self) -> int:
        """Return the number of declared bridge rows."""
        return len(self.capabilities)

    @property
    def supported_count(self) -> int:
        """Return the number of implemented bounded bridge routes."""
        return sum(1 for capability in self.capabilities if capability.supported)

    @property
    def fail_closed_count(self) -> int:
        """Return the number of explicitly blocked routes."""
        return self.framework_count - self.supported_count

    @property
    def native_framework_autodiff_count(self) -> int:
        """Return the number of implemented native-autodiff bridge routes."""
        return sum(
            1
            for capability in self.capabilities
            if capability.supported and capability.native_framework_autodiff
        )

    @property
    def tensor_output_count(self) -> int:
        """Return the number of implemented tensor-output bridge routes."""
        return sum(
            1
            for capability in self.capabilities
            if capability.supported and capability.tensor_output
        )

    @property
    def host_boundary_count(self) -> int:
        """Return the number of implemented routes that cross a host boundary."""
        return sum(
            1
            for capability in self.capabilities
            if capability.supported and capability.host_boundary
        )

    @property
    def passed(self) -> bool:
        """Return whether every supported row has a concrete public API."""
        return all(
            bool(capability.public_api) and bool(capability.gradient_route)
            for capability in self.capabilities
            if capability.supported
        )

    def capability_by_framework(self, framework: str) -> BoundedQNNFrameworkBridgeCapability:
        """Return one bridge capability by framework name."""
        normalized = _as_framework_name(framework)
        for capability in self.capabilities:
            if capability.framework == normalized:
                return capability
        raise KeyError(f"unknown bounded QNN framework bridge: {framework}")

    def assert_supported(self, framework: str) -> BoundedQNNFrameworkBridgeCapability:
        """Return a supported capability or raise a targeted fail-closed error."""
        capability = self.capability_by_framework(framework)
        if not capability.supported:
            reason = capability.fail_closed_reason or "bridge route is not implemented"
            raise RuntimeError(
                f"bounded QNN framework bridge {framework!r} is unsupported: {reason}"
            )
        return capability

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready matrix evidence."""
        return {
            "passed": self.passed,
            "framework_count": self.framework_count,
            "supported_count": self.supported_count,
            "fail_closed_count": self.fail_closed_count,
            "native_framework_autodiff_count": self.native_framework_autodiff_count,
            "tensor_output_count": self.tensor_output_count,
            "host_boundary_count": self.host_boundary_count,
            "evidence_class": self.evidence_class,
            "claim_boundary": self.claim_boundary,
            "capabilities": [capability.to_dict() for capability in self.capabilities],
        }


def _as_framework_name(name: str) -> str:
    normalized = str(name).strip().lower()
    if not normalized:
        raise ValueError("framework name must be non-empty")
    if any(character.isspace() for character in normalized):
        raise ValueError("framework name must not contain whitespace")
    return normalized


def _default_capabilities() -> tuple[BoundedQNNFrameworkBridgeCapability, ...]:
    return (
        BoundedQNNFrameworkBridgeCapability(
            framework="jax",
            public_api="jax_native_qnn_value_and_grad,jax_custom_vjp_qnn_value_and_grad",
            gradient_route="native_bounded_phase_qnn_value_and_grad,jax_custom_vjp_bounded_phase_qnn_value_and_grad",
            optional_dependency="jax",
            implemented=True,
            runtime_dependency_required=True,
            native_framework_autodiff=True,
            tensor_output=False,
            host_boundary=False,
            analytic_framework_gradient=False,
            fail_closed_reason=None,
        ),
        BoundedQNNFrameworkBridgeCapability(
            framework="pytorch",
            public_api=(
                "torch_bounded_qnn_value_and_grad,torch_autograd_qnn_value_and_grad,"
                "run_torch_func_compatibility_audit,run_torch_compile_compatibility_audit,"
                "torch_bounded_qnn_module,torch_bounded_qnn_layer,run_torch_module_wrapper_audit,"
                "validate_torch_bounded_qnn_state_dict,run_torch_module_state_audit,"
                "run_torch_module_device_state_audit,run_torch_module_checkpoint_audit"
            ),
            gradient_route=(
                "bounded_phase_qnn_tensor_analytic_gradient,"
                "torch_bounded_phase_qnn_custom_autograd_function,"
                "bounded_torch_func_grad_vmap_jacrev,bounded_torch_compile_gradient,"
                "bounded_torch_module_layer_wrapper_gradient,"
                "bounded_torch_module_optimizer_state_replay,"
                "bounded_torch_module_device_state_replay,"
                "bounded_torch_module_checkpoint_replay"
            ),
            optional_dependency="torch",
            implemented=True,
            runtime_dependency_required=True,
            native_framework_autodiff=True,
            tensor_output=True,
            host_boundary=False,
            analytic_framework_gradient=True,
            fail_closed_reason=None,
        ),
        BoundedQNNFrameworkBridgeCapability(
            framework="tensorflow",
            public_api=(
                "tensorflow_bounded_qnn_value_and_grad,"
                "run_tensorflow_gradient_tape_compatibility_audit,"
                "run_tensorflow_function_compatibility_audit,"
                "run_tensorflow_xla_compatibility_audit,"
                "tensorflow_bounded_qnn_keras_layer,"
                "run_tensorflow_keras_layer_wrapper_audit,"
                "run_tensorflow_maintenance_decision"
            ),
            gradient_route=(
                "bounded_phase_qnn_tensor_analytic_gradient,"
                "bounded_tensorflow_gradient_tape_gradient,"
                "bounded_tensorflow_function_gradient,"
                "bounded_tensorflow_xla_gradient,"
                "bounded_tensorflow_keras_layer_gradient,"
                "tensorflow_compatibility_only_decision"
            ),
            optional_dependency="tensorflow",
            implemented=True,
            runtime_dependency_required=True,
            native_framework_autodiff=True,
            tensor_output=True,
            host_boundary=False,
            analytic_framework_gradient=True,
            fail_closed_reason=None,
        ),
        BoundedQNNFrameworkBridgeCapability(
            framework="generic_simulator_autodiff",
            public_api="",
            gradient_route="arbitrary_framework_autodiff_through_simulator",
            optional_dependency=None,
            implemented=False,
            runtime_dependency_required=False,
            native_framework_autodiff=False,
            tensor_output=False,
            host_boundary=False,
            analytic_framework_gradient=False,
            fail_closed_reason=(
                "arbitrary simulator kernels do not yet expose a framework-native "
                "differentiable execution path"
            ),
        ),
        BoundedQNNFrameworkBridgeCapability(
            framework="provider_hardware_gradient",
            public_api="",
            gradient_route="live_provider_hardware_qnn_gradient",
            optional_dependency=None,
            implemented=False,
            runtime_dependency_required=True,
            native_framework_autodiff=False,
            tensor_output=False,
            host_boundary=True,
            analytic_framework_gradient=False,
            fail_closed_reason=(
                "live hardware QNN gradients require provider job policy, shot "
                "ledger, uncertainty records, and explicit hardware approval"
            ),
        ),
    )


def run_bounded_qnn_framework_bridge_matrix(
    *,
    frameworks: Iterable[str] | None = None,
) -> BoundedQNNFrameworkBridgeMatrixResult:
    """Return bounded phase-QNN bridge support rows for selected frameworks."""
    capabilities = {capability.framework: capability for capability in _default_capabilities()}
    if frameworks is None:
        return BoundedQNNFrameworkBridgeMatrixResult(capabilities=tuple(capabilities.values()))

    selected: list[BoundedQNNFrameworkBridgeCapability] = []
    for framework in frameworks:
        normalized = _as_framework_name(framework)
        if normalized not in capabilities:
            raise ValueError(f"unknown bounded QNN framework bridge: {framework}")
        selected.append(capabilities[normalized])
    return BoundedQNNFrameworkBridgeMatrixResult(capabilities=tuple(selected))


def assert_bounded_qnn_framework_bridge_supported(
    framework: str,
) -> BoundedQNNFrameworkBridgeCapability:
    """Fail closed unless a bounded phase-QNN framework bridge is implemented."""
    return run_bounded_qnn_framework_bridge_matrix(frameworks=(framework,)).assert_supported(
        framework,
    )


__all__ = [
    "BoundedQNNFrameworkBridgeCapability",
    "BoundedQNNFrameworkBridgeMatrixResult",
    "assert_bounded_qnn_framework_bridge_supported",
    "run_bounded_qnn_framework_bridge_matrix",
]
