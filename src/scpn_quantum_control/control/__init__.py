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
from .q_disruption import QuantumDisruptionClassifier
from .q_disruption_iter import (
    DisruptionBenchmark,
    ITERFeatureSpec,
    generate_synthetic_iter_data,
    normalize_iter_features,
)
from .qaoa_mpc import QAOA_MPC
from .qpetri import QuantumPetriNet
from .realtime_feedback import (
    FeedbackStep,
    RealtimeFeedbackConfig,
    RealtimeSyncFeedbackController,
    build_monitored_feedback_circuit,
    feedback_policy_numpy,
)
from .structured_ansatz import StructuredAnsatz
from .vqls_gs import VQLS_GradShafranov

__all__ = [
    "StructuredAnsatz",
    "QAOA_MPC",
    "VQLS_GradShafranov",
    "QuantumPetriNet",
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
    "FeedbackStep",
    "RealtimeFeedbackConfig",
    "RealtimeSyncFeedbackController",
    "build_monitored_feedback_circuit",
    "feedback_policy_numpy",
]
