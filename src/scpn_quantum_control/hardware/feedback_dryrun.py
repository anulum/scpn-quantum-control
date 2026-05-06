# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Feedback provider dry-runs
"""No-submit provider dry-run payloads for S1 feedback jobs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from .feedback_submission import FeedbackSubmissionPackage

DryRunProvider = Literal["ibm_runtime", "openqasm3_gate", "analog_native_review"]


@dataclass(frozen=True)
class FeedbackDryRunPayload:
    """Reviewable provider payload that never submits hardware jobs."""

    provider: DryRunProvider
    submission_enabled: bool
    payload: dict[str, Any]
    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.submission_enabled:
            raise ValueError("dry-run payloads must not enable submission")

    def to_dict(self) -> dict[str, Any]:
        """Serialise the dry-run payload."""
        return {
            "provider": self.provider,
            "submission_enabled": self.submission_enabled,
            "payload": self.payload,
            "warnings": list(self.warnings),
        }


def build_ibm_runtime_dry_run(package: FeedbackSubmissionPackage) -> FeedbackDryRunPayload:
    """Build a no-submit IBM Runtime dry-run payload."""
    return FeedbackDryRunPayload(
        provider="ibm_runtime",
        submission_enabled=False,
        payload={
            "runtime_program": "sampler",
            "backend_family": "ibm_heron_dynamic_circuits",
            "experiment_id": package.experiment_id,
            "circuits": package.budget.circuits,
            "shots_per_circuit": package.budget.shots_per_circuit,
            "repetitions": package.budget.repetitions,
            "circuit_summary": {
                "n_qubits": package.circuit.n_qubits,
                "n_clbits": package.circuit.n_clbits,
                "depth": package.circuit.depth,
                "operation_counts": dict(package.circuit.operation_counts),
                "has_mid_circuit_measurement": package.circuit.has_mid_circuit_measurement,
                "has_conditional_control": package.circuit.has_conditional_control,
            },
            "required_backend_features": [
                "mid_circuit_measurement",
                "conditional_reset",
                "conditional_rotation",
                "dynamic_circuit_control_flow",
            ],
            "pre_submit_checks": [
                "confirm backend supports dynamic circuits on the selected instance",
                "transpile live circuit and record depth plus operation counts",
                "confirm QPU budget approval",
                "write raw counts and job metadata before analysis",
            ],
        },
        warnings=(
            "No IBM credentials are read by this dry-run payload.",
            "Queue time and live calibration are not represented.",
            "Submission requires an explicit approval-gated hardware scheduler.",
        ),
    )


def build_openqasm3_gate_dry_run(package: FeedbackSubmissionPackage) -> FeedbackDryRunPayload:
    """Build a provider-neutral OpenQASM 3 style dry-run payload."""
    return FeedbackDryRunPayload(
        provider="openqasm3_gate",
        submission_enabled=False,
        payload={
            "schema": "openqasm3_dynamic_feedback_dry_run_v1",
            "experiment_id": package.experiment_id,
            "program_requirements": {
                "qubits": package.circuit.n_qubits,
                "classical_bits": package.circuit.n_clbits,
                "mid_circuit_measurement": package.circuit.has_mid_circuit_measurement,
                "conditionals": package.circuit.has_conditional_control,
                "rounds": package.circuit.n_rounds,
            },
            "budget": {
                "circuits": package.budget.circuits,
                "shots_per_circuit": package.budget.shots_per_circuit,
                "repetitions": package.budget.repetitions,
                "estimated_execution_seconds": package.budget.estimated_execution_seconds,
            },
            "semantic_controls": [
                "monitor qubit measurement",
                "conditional monitor reset",
                "conditional system-qubit rotation",
                "matched open-loop control arm",
            ],
        },
        warnings=(
            "This payload is an interchange description, not a provider submission.",
            "Provider-specific OpenQASM 3 syntax support must be checked separately.",
        ),
    )


def build_analog_native_review_payload(
    package: FeedbackSubmissionPackage,
) -> FeedbackDryRunPayload:
    """Build a review payload for analogue/native feedback reformulation."""
    return FeedbackDryRunPayload(
        provider="analog_native_review",
        submission_enabled=False,
        payload={
            "schema": "analog_native_feedback_review_v1",
            "experiment_id": package.experiment_id,
            "native_candidate": "open_loop_or_global_feedback_xy",
            "dynamic_circuit_features_not_portable": [
                "mid_circuit_measurement",
                "conditional_reset",
                "conditional_rotation",
            ],
            "required_reformulation": [
                "native XY Hamiltonian schedule",
                "global feedback parameter schedule or batch-level adaptive update",
                "separate matched open-loop analogue control",
                "platform-specific observable mapping for synchronisation R",
            ],
        },
        warnings=(
            "Analogue targets are scientifically relevant but do not execute the same dynamic-circuit payload.",
            "A separate native-feedback dossier is required before analogue submission.",
        ),
    )


def build_s1_feedback_dry_run_bundle(
    package: FeedbackSubmissionPackage,
) -> tuple[FeedbackDryRunPayload, ...]:
    """Build the default no-submit dry-run bundle for S1."""
    return (
        build_ibm_runtime_dry_run(package),
        build_openqasm3_gate_dry_run(package),
        build_analog_native_review_payload(package),
    )
