# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Quantum Control Systems
"""Synchronisation-control systems and no-submit campaign scaffolds.

The package covers Kuramoto-XY feedback, FRC schedule scoring, QAOA-MPC,
VQ proxy solvers, disruption classification, topological optimisation, and
Petri-net scheduling. It is not a generic pulse-control or hardware drift
compensation surface.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .adaptive_branching import (
        AdaptiveBranchDecision,
        AdaptiveBranchingConfig,
        AdaptiveBranchingReadiness,
        build_adaptive_branch_table,
        classify_branch_state,
        estimate_branching_readiness,
        required_s8_dynamic_features,
        s8_adaptive_branching_markdown,
        s8_adaptive_branching_payload,
    )
    from .closed_loop_analysis import (
        ClosedLoopControlEvidence,
        ClosedLoopExecutionDecision,
        ClosedLoopExecutionPolicy,
        ClosedLoopLatencyBudget,
        ClosedLoopLatencyReport,
        ClosedLoopPublicationPackage,
        ControlPerformance,
        ExecutionMode,
        ResponseClass,
        analyse_closed_loop_response,
        build_closed_loop_publication_package,
        evaluate_closed_loop_policy,
        measure_closed_loop_latency_budget,
        run_closed_loop_control,
    )
    from .frc_pulsed_qaoa import (
        FRCScheduleResult,
        classical_sqp_schedule,
        optimal_schedule,
        solve_frc_pulsed_qaoa,
    )
    from .q_disruption import QuantumDisruptionClassifier
    from .q_disruption_iter import (
        DisruptionBenchmark,
        ITERFeatureSpec,
        generate_synthetic_iter_data,
        normalize_iter_features,
        scpn_control_bridge_dependency_contract,
        validate_scpn_control_bridge_dependency_contract,
    )
    from .qaoa_mpc import QAOA_MPC
    from .qaoa_pulsed_cost import (
        FRCPlasmaSurrogate,
        FRCQAOAObjective,
        decode_schedule_to_field,
        frc_pulsed_shot_cost,
    )
    from .qpetri import QuantumPetriCampaignReport, QuantumPetriNet, QuantumPetriStepReport
    from .realtime_feedback import (
        FeedbackStep,
        RealtimeFeedbackConfig,
        RealtimeSyncFeedbackController,
        build_monitored_feedback_circuit,
        feedback_policy_numpy,
    )
    from .structured_ansatz import StructuredAnsatz
    from .vqls_gs import VQLS_GradShafranov, VQLSGradShafranovResult

_PUBLIC_EXPORTS: dict[str, tuple[str, str | None]] = {
    "AdaptiveBranchDecision": (
        "scpn_quantum_control.control.adaptive_branching",
        "AdaptiveBranchDecision",
    ),
    "AdaptiveBranchingConfig": (
        "scpn_quantum_control.control.adaptive_branching",
        "AdaptiveBranchingConfig",
    ),
    "AdaptiveBranchingReadiness": (
        "scpn_quantum_control.control.adaptive_branching",
        "AdaptiveBranchingReadiness",
    ),
    "build_adaptive_branch_table": (
        "scpn_quantum_control.control.adaptive_branching",
        "build_adaptive_branch_table",
    ),
    "classify_branch_state": (
        "scpn_quantum_control.control.adaptive_branching",
        "classify_branch_state",
    ),
    "estimate_branching_readiness": (
        "scpn_quantum_control.control.adaptive_branching",
        "estimate_branching_readiness",
    ),
    "required_s8_dynamic_features": (
        "scpn_quantum_control.control.adaptive_branching",
        "required_s8_dynamic_features",
    ),
    "s8_adaptive_branching_markdown": (
        "scpn_quantum_control.control.adaptive_branching",
        "s8_adaptive_branching_markdown",
    ),
    "s8_adaptive_branching_payload": (
        "scpn_quantum_control.control.adaptive_branching",
        "s8_adaptive_branching_payload",
    ),
    "ClosedLoopControlEvidence": (
        "scpn_quantum_control.control.closed_loop_analysis",
        "ClosedLoopControlEvidence",
    ),
    "ClosedLoopExecutionDecision": (
        "scpn_quantum_control.control.closed_loop_analysis",
        "ClosedLoopExecutionDecision",
    ),
    "ClosedLoopExecutionPolicy": (
        "scpn_quantum_control.control.closed_loop_analysis",
        "ClosedLoopExecutionPolicy",
    ),
    "ClosedLoopLatencyBudget": (
        "scpn_quantum_control.control.closed_loop_analysis",
        "ClosedLoopLatencyBudget",
    ),
    "ClosedLoopLatencyReport": (
        "scpn_quantum_control.control.closed_loop_analysis",
        "ClosedLoopLatencyReport",
    ),
    "ClosedLoopPublicationPackage": (
        "scpn_quantum_control.control.closed_loop_analysis",
        "ClosedLoopPublicationPackage",
    ),
    "ControlPerformance": (
        "scpn_quantum_control.control.closed_loop_analysis",
        "ControlPerformance",
    ),
    "ExecutionMode": ("scpn_quantum_control.control.closed_loop_analysis", "ExecutionMode"),
    "ResponseClass": ("scpn_quantum_control.control.closed_loop_analysis", "ResponseClass"),
    "analyse_closed_loop_response": (
        "scpn_quantum_control.control.closed_loop_analysis",
        "analyse_closed_loop_response",
    ),
    "build_closed_loop_publication_package": (
        "scpn_quantum_control.control.closed_loop_analysis",
        "build_closed_loop_publication_package",
    ),
    "evaluate_closed_loop_policy": (
        "scpn_quantum_control.control.closed_loop_analysis",
        "evaluate_closed_loop_policy",
    ),
    "measure_closed_loop_latency_budget": (
        "scpn_quantum_control.control.closed_loop_analysis",
        "measure_closed_loop_latency_budget",
    ),
    "run_closed_loop_control": (
        "scpn_quantum_control.control.closed_loop_analysis",
        "run_closed_loop_control",
    ),
    "FRCScheduleResult": ("scpn_quantum_control.control.frc_pulsed_qaoa", "FRCScheduleResult"),
    "classical_sqp_schedule": (
        "scpn_quantum_control.control.frc_pulsed_qaoa",
        "classical_sqp_schedule",
    ),
    "optimal_schedule": ("scpn_quantum_control.control.frc_pulsed_qaoa", "optimal_schedule"),
    "solve_frc_pulsed_qaoa": (
        "scpn_quantum_control.control.frc_pulsed_qaoa",
        "solve_frc_pulsed_qaoa",
    ),
    "QuantumDisruptionClassifier": (
        "scpn_quantum_control.control.q_disruption",
        "QuantumDisruptionClassifier",
    ),
    "DisruptionBenchmark": (
        "scpn_quantum_control.control.q_disruption_iter",
        "DisruptionBenchmark",
    ),
    "ITERFeatureSpec": ("scpn_quantum_control.control.q_disruption_iter", "ITERFeatureSpec"),
    "generate_synthetic_iter_data": (
        "scpn_quantum_control.control.q_disruption_iter",
        "generate_synthetic_iter_data",
    ),
    "normalize_iter_features": (
        "scpn_quantum_control.control.q_disruption_iter",
        "normalize_iter_features",
    ),
    "scpn_control_bridge_dependency_contract": (
        "scpn_quantum_control.control.q_disruption_iter",
        "scpn_control_bridge_dependency_contract",
    ),
    "validate_scpn_control_bridge_dependency_contract": (
        "scpn_quantum_control.control.q_disruption_iter",
        "validate_scpn_control_bridge_dependency_contract",
    ),
    "QAOA_MPC": ("scpn_quantum_control.control.qaoa_mpc", "QAOA_MPC"),
    "FRCPlasmaSurrogate": ("scpn_quantum_control.control.qaoa_pulsed_cost", "FRCPlasmaSurrogate"),
    "FRCQAOAObjective": ("scpn_quantum_control.control.qaoa_pulsed_cost", "FRCQAOAObjective"),
    "decode_schedule_to_field": (
        "scpn_quantum_control.control.qaoa_pulsed_cost",
        "decode_schedule_to_field",
    ),
    "frc_pulsed_shot_cost": (
        "scpn_quantum_control.control.qaoa_pulsed_cost",
        "frc_pulsed_shot_cost",
    ),
    "QuantumPetriCampaignReport": (
        "scpn_quantum_control.control.qpetri",
        "QuantumPetriCampaignReport",
    ),
    "QuantumPetriNet": ("scpn_quantum_control.control.qpetri", "QuantumPetriNet"),
    "QuantumPetriStepReport": ("scpn_quantum_control.control.qpetri", "QuantumPetriStepReport"),
    "FeedbackStep": ("scpn_quantum_control.control.realtime_feedback", "FeedbackStep"),
    "RealtimeFeedbackConfig": (
        "scpn_quantum_control.control.realtime_feedback",
        "RealtimeFeedbackConfig",
    ),
    "RealtimeSyncFeedbackController": (
        "scpn_quantum_control.control.realtime_feedback",
        "RealtimeSyncFeedbackController",
    ),
    "build_monitored_feedback_circuit": (
        "scpn_quantum_control.control.realtime_feedback",
        "build_monitored_feedback_circuit",
    ),
    "feedback_policy_numpy": (
        "scpn_quantum_control.control.realtime_feedback",
        "feedback_policy_numpy",
    ),
    "StructuredAnsatz": ("scpn_quantum_control.control.structured_ansatz", "StructuredAnsatz"),
    "VQLS_GradShafranov": ("scpn_quantum_control.control.vqls_gs", "VQLS_GradShafranov"),
    "VQLSGradShafranovResult": ("scpn_quantum_control.control.vqls_gs", "VQLSGradShafranovResult"),
}


def __getattr__(name: str) -> Any:
    """Resolve and cache a public export from its original owning module.

    Parameters
    ----------
    name
        Public export requested through this package.

    Returns
    -------
    Any
        Original object, including module-valued exports.

    Raises
    ------
    AttributeError
        If the name is undeclared or the original module lacks its attribute.
    ImportError
        If the owning module cannot be imported.

    """
    target = _PUBLIC_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    origin = import_module(target[0])
    value = origin if target[1] is None else getattr(origin, target[1])
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """List cached and deferred names for inspection tools.

    Returns
    -------
    list[str]
        Sorted package namespace and declared lazy export names.

    """
    return sorted(set(globals()) | set(_PUBLIC_EXPORTS))


__all__ = [
    "ClosedLoopControlEvidence",
    "ClosedLoopExecutionDecision",
    "ClosedLoopExecutionPolicy",
    "ClosedLoopLatencyBudget",
    "ClosedLoopLatencyReport",
    "ClosedLoopPublicationPackage",
    "ControlPerformance",
    "ExecutionMode",
    "ResponseClass",
    "analyse_closed_loop_response",
    "build_closed_loop_publication_package",
    "evaluate_closed_loop_policy",
    "measure_closed_loop_latency_budget",
    "run_closed_loop_control",
    "StructuredAnsatz",
    "QAOA_MPC",
    "FRCPlasmaSurrogate",
    "FRCQAOAObjective",
    "FRCScheduleResult",
    "classical_sqp_schedule",
    "decode_schedule_to_field",
    "frc_pulsed_shot_cost",
    "optimal_schedule",
    "solve_frc_pulsed_qaoa",
    "VQLS_GradShafranov",
    "VQLSGradShafranovResult",
    "QuantumPetriNet",
    "QuantumPetriStepReport",
    "QuantumPetriCampaignReport",
    "QuantumDisruptionClassifier",
    "AdaptiveBranchDecision",
    "AdaptiveBranchingConfig",
    "AdaptiveBranchingReadiness",
    "build_adaptive_branch_table",
    "classify_branch_state",
    "estimate_branching_readiness",
    "required_s8_dynamic_features",
    "s8_adaptive_branching_markdown",
    "s8_adaptive_branching_payload",
    "DisruptionBenchmark",
    "ITERFeatureSpec",
    "generate_synthetic_iter_data",
    "normalize_iter_features",
    "scpn_control_bridge_dependency_contract",
    "validate_scpn_control_bridge_dependency_contract",
    "FeedbackStep",
    "RealtimeFeedbackConfig",
    "RealtimeSyncFeedbackController",
    "build_monitored_feedback_circuit",
    "feedback_policy_numpy",
]
