# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — quantum-classical co-design package
"""Public simulator-first quantum-classical co-design surface."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .adapters import (
        ControlAdapterEvidence,
        adaptive_fim_proposal_port,
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

_PUBLIC_EXPORTS: dict[str, tuple[str, str | None]] = {
    "ControlAdapterEvidence": ("scpn_quantum_control.codesign.adapters", "ControlAdapterEvidence"),
    "adaptive_fim_proposal_port": (
        "scpn_quantum_control.codesign.adapters",
        "adaptive_fim_proposal_port",
    ),
    "consume_cosimulation_port": (
        "scpn_quantum_control.codesign.adapters",
        "consume_cosimulation_port",
    ),
    "consume_qaoa_mpc_port": ("scpn_quantum_control.codesign.adapters", "consume_qaoa_mpc_port"),
    "consume_realtime_feedback_port": (
        "scpn_quantum_control.codesign.adapters",
        "consume_realtime_feedback_port",
    ),
    "observer_inputs_from_products": (
        "scpn_quantum_control.codesign.adapters",
        "observer_inputs_from_products",
    ),
    "ExponentialOrderEstimator": (
        "scpn_quantum_control.codesign.components",
        "ExponentialOrderEstimator",
    ),
    "GradientFeedbackController": (
        "scpn_quantum_control.codesign.components",
        "GradientFeedbackController",
    ),
    "OpenSystemObjectiveConfig": (
        "scpn_quantum_control.codesign.components",
        "OpenSystemObjectiveConfig",
    ),
    "PhaseObjectiveSimulator": (
        "scpn_quantum_control.codesign.components",
        "PhaseObjectiveSimulator",
    ),
    "CODESIGN_CLAIM_BOUNDARY": (
        "scpn_quantum_control.codesign.contracts",
        "CODESIGN_CLAIM_BOUNDARY",
    ),
    "CODESIGN_SCHEMA": ("scpn_quantum_control.codesign.contracts", "CODESIGN_SCHEMA"),
    "BackendCapabilities": ("scpn_quantum_control.codesign.contracts", "BackendCapabilities"),
    "CoDesignMode": ("scpn_quantum_control.codesign.contracts", "CoDesignMode"),
    "ControllerProposal": ("scpn_quantum_control.codesign.contracts", "ControllerProposal"),
    "GradientPlanRecord": ("scpn_quantum_control.codesign.contracts", "GradientPlanRecord"),
    "LatencyDecision": ("scpn_quantum_control.codesign.contracts", "LatencyDecision"),
    "LoopStepInput": ("scpn_quantum_control.codesign.contracts", "LoopStepInput"),
    "LoopStepOutput": ("scpn_quantum_control.codesign.contracts", "LoopStepOutput"),
    "ObserverInputs": ("scpn_quantum_control.codesign.contracts", "ObserverInputs"),
    "PlasmaObjectiveTemplate": (
        "scpn_quantum_control.codesign.contracts",
        "PlasmaObjectiveTemplate",
    ),
    "QuantumEvaluation": ("scpn_quantum_control.codesign.contracts", "QuantumEvaluation"),
    "SafetyAction": ("scpn_quantum_control.codesign.contracts", "SafetyAction"),
    "SafetyDecision": ("scpn_quantum_control.codesign.contracts", "SafetyDecision"),
    "StaleGradientAction": ("scpn_quantum_control.codesign.contracts", "StaleGradientAction"),
    "StateEstimate": ("scpn_quantum_control.codesign.contracts", "StateEstimate"),
    "plasma_objective_templates": (
        "scpn_quantum_control.codesign.contracts",
        "plasma_objective_templates",
    ),
    "EVIDENCE_CLASSIFICATION": (
        "scpn_quantum_control.codesign.evidence",
        "EVIDENCE_CLASSIFICATION",
    ),
    "EVIDENCE_SCHEMA": ("scpn_quantum_control.codesign.evidence", "EVIDENCE_SCHEMA"),
    "FunctionalEvidence": ("scpn_quantum_control.codesign.evidence", "FunctionalEvidence"),
    "build_demo_loop": ("scpn_quantum_control.codesign.evidence", "build_demo_loop"),
    "demo_inputs": ("scpn_quantum_control.codesign.evidence", "demo_inputs"),
    "run_functional_evidence": (
        "scpn_quantum_control.codesign.evidence",
        "run_functional_evidence",
    ),
    "validate_functional_evidence": (
        "scpn_quantum_control.codesign.evidence",
        "validate_functional_evidence",
    ),
    "write_functional_evidence": (
        "scpn_quantum_control.codesign.evidence",
        "write_functional_evidence",
    ),
    "CoDesignLoop": ("scpn_quantum_control.codesign.loop", "CoDesignLoop"),
    "LatencyPolicy": ("scpn_quantum_control.codesign.policies", "LatencyPolicy"),
    "SafetyEnvelope": ("scpn_quantum_control.codesign.policies", "SafetyEnvelope"),
    "REPLAY_SCHEMA": ("scpn_quantum_control.codesign.replay", "REPLAY_SCHEMA"),
    "ReplayTrace": ("scpn_quantum_control.codesign.replay", "ReplayTrace"),
    "record_replay_trace": ("scpn_quantum_control.codesign.replay", "record_replay_trace"),
    "verify_replay_trace": ("scpn_quantum_control.codesign.replay", "verify_replay_trace"),
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
    "adaptive_fim_proposal_port",
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
