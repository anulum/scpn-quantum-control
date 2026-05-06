# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for feedback submission readiness
"""Tests for provider-neutral S1 feedback submission packages."""

from __future__ import annotations

import numpy as np

from scpn_quantum_control.control.realtime_feedback import (
    RealtimeFeedbackConfig,
    RealtimeSyncFeedbackController,
)
from scpn_quantum_control.hardware.feedback_submission import (
    FeedbackBudgetEstimate,
    FeedbackPlatformCapability,
    assess_platform_readiness,
    build_s1_feedback_submission_package,
)


def _controller() -> RealtimeSyncFeedbackController:
    return RealtimeSyncFeedbackController(
        np.array([[0.0, 0.25], [0.25, 0.0]], dtype=np.float64),
        np.array([0.1, 0.4], dtype=np.float64),
        config=RealtimeFeedbackConfig(base_dt=0.02, trotter_steps=1, measurement_shots=32),
    )


def test_s1_feedback_submission_package_marks_dynamic_backends_ready() -> None:
    package = build_s1_feedback_submission_package(
        _controller(),
        n_rounds=2,
        shots_per_circuit=512,
        repetitions=8,
        estimated_seconds_per_circuit=0.75,
    )

    data = package.to_dict()

    assert package.budget.total_reserved_seconds == 6.0
    assert package.circuit.n_qubits == 3
    assert package.circuit.n_clbits == 4
    assert package.circuit.has_mid_circuit_measurement is True
    assert package.circuit.has_conditional_control is True
    assert "IBM Heron dynamic-circuit backend" in package.ready_platforms
    assert "Local statevector simulator" in package.ready_platforms
    assert data["budget"]["shots_per_circuit"] == 512
    assert data["platform_readiness"][0]["payload"]["repetitions"] == 8
    assert data["dossier"]["job_id"] == "s1_dynamic_feedback_readiness"
    assert "falsification_condition" in data["dossier"]
    assert "## Claim Boundary" in package.dossier.to_markdown()


def test_s1_feedback_submission_package_separates_analog_targets_for_review() -> None:
    package = build_s1_feedback_submission_package(_controller(), n_rounds=1)
    decisions = {decision.platform.name: decision for decision in package.platform_readiness}

    neutral_atom = decisions["Neutral-atom analogue XY target"]

    assert neutral_atom.status == "manual_review"
    assert "payload requires mid-circuit measurement" in neutral_atom.reasons
    assert "payload requires conditional rotations" in neutral_atom.reasons


def test_platform_readiness_blocks_insufficient_gate_backend() -> None:
    package = build_s1_feedback_submission_package(_controller(), n_rounds=1)
    tiny_backend = FeedbackPlatformCapability(
        name="tiny gate target",
        kind="gate_based_dynamic_circuits",
        max_qubits=1,
        supports_mid_circuit_measurement=False,
        supports_conditional_reset=False,
        supports_conditional_rotation=False,
        supports_cross_shot_batches=True,
    )

    decision = assess_platform_readiness(tiny_backend, package.circuit, package.budget)

    assert decision.status == "blocked"
    assert "requires 3 qubits but platform declares 1" in decision.reasons
    assert "payload requires mid-circuit measurement" in decision.reasons


def test_feedback_budget_total_includes_queue_and_calibration_seconds() -> None:
    budget = FeedbackBudgetEstimate(
        circuits=2,
        shots_per_circuit=128,
        repetitions=3,
        estimated_execution_seconds=5.0,
        queue_seconds=7.0,
        calibration_seconds=11.0,
    )

    assert budget.total_reserved_seconds == 23.0
