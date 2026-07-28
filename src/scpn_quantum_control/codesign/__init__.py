# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — quantum-classical co-design package
"""Public simulator-first quantum-classical co-design surface (BL-33)."""

from .adapters import (
    ControlAdapterEvidence,
    consume_cosimulation_port,
    consume_qaoa_mpc_port,
    consume_realtime_feedback_port,
    observer_inputs_from_products,
)
from .components import (
    ExponentialOrderEstimator,
    GradientFeedbackController,
    OpenSystemObjectiveConfig,
    PhaseObjectiveSimulator,
)
from .contracts import (
    CODESIGN_CLAIM_BOUNDARY,
    CODESIGN_SCHEMA,
    BackendCapabilities,
    CoDesignMode,
    ControllerProposal,
    GradientPlanRecord,
    LatencyDecision,
    LoopStepInput,
    LoopStepOutput,
    ObserverInputs,
    PlasmaObjectiveTemplate,
    QuantumEvaluation,
    SafetyAction,
    SafetyDecision,
    StaleGradientAction,
    StateEstimate,
    plasma_objective_templates,
)
from .evidence import (
    EVIDENCE_CLASSIFICATION,
    EVIDENCE_SCHEMA,
    FunctionalEvidence,
    build_demo_loop,
    demo_inputs,
    run_functional_evidence,
    validate_functional_evidence,
    write_functional_evidence,
)
from .loop import CoDesignLoop
from .policies import LatencyPolicy, SafetyEnvelope
from .replay import REPLAY_SCHEMA, ReplayTrace, record_replay_trace, verify_replay_trace

__all__ = [
    "CODESIGN_CLAIM_BOUNDARY",
    "CODESIGN_SCHEMA",
    "EVIDENCE_CLASSIFICATION",
    "EVIDENCE_SCHEMA",
    "REPLAY_SCHEMA",
    "BackendCapabilities",
    "CoDesignLoop",
    "CoDesignMode",
    "ControlAdapterEvidence",
    "ControllerProposal",
    "ExponentialOrderEstimator",
    "FunctionalEvidence",
    "GradientFeedbackController",
    "GradientPlanRecord",
    "LatencyDecision",
    "LatencyPolicy",
    "LoopStepInput",
    "LoopStepOutput",
    "ObserverInputs",
    "OpenSystemObjectiveConfig",
    "PhaseObjectiveSimulator",
    "PlasmaObjectiveTemplate",
    "QuantumEvaluation",
    "ReplayTrace",
    "SafetyAction",
    "SafetyDecision",
    "SafetyEnvelope",
    "StaleGradientAction",
    "StateEstimate",
    "build_demo_loop",
    "consume_cosimulation_port",
    "consume_qaoa_mpc_port",
    "consume_realtime_feedback_port",
    "demo_inputs",
    "observer_inputs_from_products",
    "plasma_objective_templates",
    "record_replay_trace",
    "run_functional_evidence",
    "validate_functional_evidence",
    "verify_replay_trace",
    "write_functional_evidence",
]
