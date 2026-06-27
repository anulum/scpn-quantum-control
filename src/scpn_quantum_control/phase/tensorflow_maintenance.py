# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — TensorFlow maintenance decision.
"""TensorFlow maintenance decision for differentiable framework parity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

TensorFlowMaintenanceDecision = Literal["maintained", "compatibility_only", "blocked"]
TensorFlowMaintenanceStrategy: TypeAlias = Literal["compatibility_only"]

TENSORFLOW_MAINTENANCE_CLAIM_BOUNDARY: Final[str] = (
    "TensorFlow remains a bounded compatibility surface for declared phase-QNN "
    "tensor, GradientTape, tf.function, XLA-request, and Keras-layer routes; "
    "this decision does not promote arbitrary Phase-QNode TensorFlow lowering, "
    "full graph autodiff through simulators, provider callbacks, hardware "
    "execution, broad Graph/XLA parity, or performance claims"
)


@dataclass(frozen=True)
class PhaseTensorFlowMaintenanceRoute:
    """One TensorFlow route decision in the framework-parity maintenance ledger."""

    name: str
    decision: TensorFlowMaintenanceDecision
    evidence: tuple[str, ...]
    blocked_reasons: tuple[str, ...]
    required_before_promotion: tuple[str, ...]
    claim_boundary: str = TENSORFLOW_MAINTENANCE_CLAIM_BOUNDARY

    @property
    def fail_closed(self) -> bool:
        """Return whether the route is intentionally non-promotional."""
        return self.decision in {"compatibility_only", "blocked"}

    def __post_init__(self) -> None:
        """Validate route identity, evidence, blockers, and promotion requirements."""
        if not self.name:
            raise ValueError("TensorFlow maintenance route name must be non-empty")
        if self.decision not in {"maintained", "compatibility_only", "blocked"}:
            raise ValueError("unknown TensorFlow maintenance decision")
        if not self.evidence or any(not item for item in self.evidence):
            raise ValueError("TensorFlow maintenance route evidence must be non-empty")
        if any(not item for item in self.blocked_reasons):
            raise ValueError("TensorFlow maintenance route blockers must be non-empty")
        if any(not item for item in self.required_before_promotion):
            raise ValueError("TensorFlow maintenance promotion requirements must be non-empty")
        if self.decision == "blocked" and not self.blocked_reasons:
            raise ValueError("blocked TensorFlow maintenance routes require blockers")
        if self.decision == "maintained" and self.blocked_reasons:
            raise ValueError("maintained TensorFlow maintenance routes cannot carry blockers")
        if not self.claim_boundary:
            raise ValueError("TensorFlow maintenance route claim boundary must be non-empty")

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready route metadata."""
        return {
            "name": self.name,
            "decision": self.decision,
            "fail_closed": self.fail_closed,
            "evidence": list(self.evidence),
            "blocked_reasons": list(self.blocked_reasons),
            "required_before_promotion": list(self.required_before_promotion),
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class PhaseTensorFlowMaintenanceReport:
    """TensorFlow framework-parity maintenance decision report."""

    strategy: TensorFlowMaintenanceStrategy
    routes: tuple[PhaseTensorFlowMaintenanceRoute, ...]
    claim_boundary: str = TENSORFLOW_MAINTENANCE_CLAIM_BOUNDARY

    @property
    def compatibility_only(self) -> bool:
        """Return whether TensorFlow is scoped as a compatibility-only surface."""
        return self.strategy == "compatibility_only"

    @property
    def graph_xla_parity_promoted(self) -> bool:
        """Return whether broad TensorFlow Graph/XLA parity is promoted."""
        return False

    @property
    def maintained_compatibility_routes(self) -> tuple[str, ...]:
        """Return bounded TensorFlow routes kept under maintenance."""
        return tuple(route.name for route in self.routes if route.decision == "compatibility_only")

    @property
    def blocked_routes(self) -> tuple[str, ...]:
        """Return TensorFlow routes that remain blocked."""
        return tuple(route.name for route in self.routes if route.decision == "blocked")

    @property
    def ready_for_provider_exceedance(self) -> bool:
        """Return whether TensorFlow can support provider-exceedance claims."""
        return False

    @property
    def stale_claim_blockers(self) -> tuple[str, ...]:
        """Return claim families that must stay blocked in public surfaces."""
        return (
            "arbitrary_phase_qnode_tensorflow_lowering",
            "full_tensorflow_graph_autodiff_through_simulators",
            "broad_graph_xla_parity",
            "provider_callbacks",
            "hardware_gradients",
            "isolated_benchmark_promotion",
        )

    def __post_init__(self) -> None:
        """Validate the TensorFlow decision report."""
        if self.strategy != "compatibility_only":
            raise ValueError("TensorFlow maintenance strategy must be compatibility_only")
        if not self.routes:
            raise ValueError("TensorFlow maintenance report requires routes")
        names = [route.name for route in self.routes]
        if len(set(names)) != len(names):
            raise ValueError("TensorFlow maintenance route names must be unique")
        if not self.claim_boundary:
            raise ValueError("TensorFlow maintenance report claim boundary must be non-empty")

    def route(self, name: str) -> PhaseTensorFlowMaintenanceRoute:
        """Return one named route or fail closed on unknown route names."""
        for route in self.routes:
            if route.name == name:
                return route
        raise KeyError(f"unknown TensorFlow maintenance route: {name}")

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready TensorFlow maintenance decision metadata."""
        return {
            "strategy": self.strategy,
            "compatibility_only": self.compatibility_only,
            "graph_xla_parity_promoted": self.graph_xla_parity_promoted,
            "ready_for_provider_exceedance": self.ready_for_provider_exceedance,
            "maintained_compatibility_routes": list(self.maintained_compatibility_routes),
            "blocked_routes": list(self.blocked_routes),
            "stale_claim_blockers": list(self.stale_claim_blockers),
            "routes": {route.name: route.to_dict() for route in self.routes},
            "claim_boundary": self.claim_boundary,
        }


def run_tensorflow_maintenance_decision() -> PhaseTensorFlowMaintenanceReport:
    """Return the TensorFlow framework-parity maintenance decision."""
    return PhaseTensorFlowMaintenanceReport(
        strategy="compatibility_only",
        routes=(
            _compatibility_route(
                "bounded_qnn_tensor_gradient",
                ("tensorflow_bounded_qnn_value_and_grad", "PhaseTensorFlowQNNGradientResult"),
            ),
            _compatibility_route(
                "bounded_gradient_tape",
                (
                    "run_tensorflow_gradient_tape_compatibility_audit",
                    "PhaseTensorFlowGradientTapeCompatibilityResult",
                ),
            ),
            _compatibility_route(
                "bounded_tf_function",
                (
                    "run_tensorflow_function_compatibility_audit",
                    "PhaseTensorFlowFunctionCompatibilityResult",
                ),
            ),
            _compatibility_route(
                "bounded_xla_request",
                (
                    "run_tensorflow_xla_compatibility_audit",
                    "PhaseTensorFlowXLACompatibilityResult",
                ),
            ),
            _compatibility_route(
                "bounded_keras_layer",
                (
                    "tensorflow_bounded_qnn_keras_layer",
                    "run_tensorflow_keras_layer_wrapper_audit",
                ),
            ),
            _blocked_route(
                "arbitrary_phase_qnode_tensorflow_lowering",
                "registered Phase-QNode TensorFlow lowering is not implemented",
                (
                    "native_tensorflow_gate_lowering_rules",
                    "statevector_gradient_parity_artifact",
                    "shape_dtype_policy",
                ),
            ),
            _blocked_route(
                "full_graph_autodiff_through_simulator",
                "full TensorFlow graph autodiff through arbitrary simulators is not implemented",
                (
                    "graph_autodiff_contract",
                    "simulator_lowering_rules",
                    "gradient_tape_parity_artifact",
                ),
            ),
            _blocked_route(
                "provider_callback_graph_route",
                "provider callbacks are not safe TensorFlow graph routes",
                (
                    "provider_allowlist",
                    "callback_transform_safety_audit",
                    "provider_execution_artifact",
                ),
            ),
            _blocked_route(
                "hardware_gradient_route",
                "hardware TensorFlow gradient claims require live-ticketed evidence",
                ("live_ticket", "raw_count_replay", "calibration_snapshot"),
            ),
            _blocked_route(
                "isolated_benchmark_promotion",
                "TensorFlow performance promotion requires isolated benchmark artefacts",
                ("isolated_affinity_benchmark_id", "host_load_record", "device_placement_record"),
            ),
        ),
    )


def _compatibility_route(
    name: str,
    evidence: tuple[str, ...],
) -> PhaseTensorFlowMaintenanceRoute:
    return PhaseTensorFlowMaintenanceRoute(
        name=name,
        decision="compatibility_only",
        evidence=evidence,
        blocked_reasons=(
            "maintained as bounded compatibility evidence only",
            "does not promote broad TensorFlow Graph/XLA parity",
        ),
        required_before_promotion=(
            "same-circuit framework conformance table",
            "dependency-lock evidence",
            "isolated benchmark artefact for performance claims",
        ),
    )


def _blocked_route(
    name: str,
    reason: str,
    required_before_promotion: tuple[str, ...],
) -> PhaseTensorFlowMaintenanceRoute:
    return PhaseTensorFlowMaintenanceRoute(
        name=name,
        decision="blocked",
        evidence=("fail_closed_maintenance_decision",),
        blocked_reasons=(reason,),
        required_before_promotion=required_before_promotion,
    )


__all__ = [
    "TENSORFLOW_MAINTENANCE_CLAIM_BOUNDARY",
    "PhaseTensorFlowMaintenanceReport",
    "PhaseTensorFlowMaintenanceRoute",
    "TensorFlowMaintenanceDecision",
    "TensorFlowMaintenanceStrategy",
    "run_tensorflow_maintenance_decision",
]
