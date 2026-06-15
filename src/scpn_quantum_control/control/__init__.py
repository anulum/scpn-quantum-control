# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Quantum Control Systems
"""Quantum control systems: QAOA-MPC, VQ linear solvers, disruption classification,
topological optimisation, and Petri net scheduling.
"""

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
