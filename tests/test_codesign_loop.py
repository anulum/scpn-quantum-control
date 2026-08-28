# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — co-design loop tests
"""End-to-end simulator tests for the public co-design loop."""

from __future__ import annotations

import pytest

from scpn_quantum_control.codesign import (
    CoDesignLoop,
    CoDesignMode,
    ExponentialOrderEstimator,
    GradientFeedbackController,
    LatencyPolicy,
    LoopStepInput,
    ObserverInputs,
    PhaseObjectiveSimulator,
    SafetyAction,
    SafetyEnvelope,
    StaleGradientAction,
)
from scpn_quantum_control.control.closed_loop_analysis import ClosedLoopExecutionPolicy


def _loop(*, missing_steps: frozenset[int] = frozenset(), max_age_ms: float = 5.0) -> CoDesignLoop:
    return CoDesignLoop(
        estimator=ExponentialOrderEstimator(alpha=0.5),
        evaluator=PhaseObjectiveSimulator(
            policy=ClosedLoopExecutionPolicy(round_budget=8),
            missing_gradient_steps=missing_steps,
        ),
        controller=GradientFeedbackController(learning_rate=0.1, feedback_gain=0.5),
        latency_policy=LatencyPolicy(
            max_age_ms=max_age_ms,
            on_stale=StaleGradientAction.HOLD,
            on_missing=StaleGradientAction.ABORT,
        ),
        safety_envelope=SafetyEnvelope(3.2, 0.5, 2.0),
    )


def _input(step: int, mode: CoDesignMode, *, apply_at_ms: float | None = None) -> LoopStepInput:
    observed = float(step * 10)
    return LoopStepInput(
        step=step,
        observed_at_ms=observed,
        apply_at_ms=observed + 2.0 if apply_at_ms is None else apply_at_ms,
        parameters=(0.0, 0.6, 1.2),
        measurement=0.2,
        target_order_parameter=0.9,
        mode=mode,
    )


def test_classical_to_quantum_step_uses_external_measurement() -> None:
    """Drive the estimator from a caller-supplied classical measurement."""
    output = _loop().step(_input(0, CoDesignMode.CLASSICAL_TO_QUANTUM))

    assert output.estimate.order_parameter == 0.2
    assert output.evaluation is not None
    assert output.proposal is not None
    assert output.safety.action in {SafetyAction.ALLOW, SafetyAction.CLAMP}
    assert output.safety.applied_parameters != (0.0, 0.6, 1.2)
    assert output.to_dict()["mode"] == "classical_to_quantum"


def test_quantum_to_classical_step_uses_simulator_observation() -> None:
    """Drive the estimator from the local simulator observation."""
    output = _loop().step(_input(0, CoDesignMode.QUANTUM_TO_CLASSICAL))

    assert output.evaluation is not None
    assert output.estimate.order_parameter == pytest.approx(output.evaluation.order_parameter)
    assert output.estimate.order_parameter != 0.2


def test_hybrid_trace_is_stable_under_bounded_measurement_noise() -> None:
    """Keep a bounded noisy hybrid trace inside its safety envelope."""
    inputs = tuple(
        LoopStepInput(
            step=index,
            observed_at_ms=float(index * 10),
            apply_at_ms=float(index * 10 + 2),
            parameters=(0.0, 0.6, 1.2),
            measurement=measurement,
            target_order_parameter=0.9,
            mode=CoDesignMode.HYBRID_REPLAY,
        )
        for index, measurement in enumerate((0.45, 0.55, 0.48, 0.52))
    )
    outputs = _loop().run(inputs)

    assert len(outputs) == 4
    assert all(output.safety.action is not SafetyAction.ABORT for output in outputs)
    assert outputs[-1].estimate.order_parameter == pytest.approx(0.505)


def test_missing_gradient_aborts_and_stops_trace() -> None:
    """Abort at a missing update and stop subsequent loop work."""
    inputs = (
        _input(0, CoDesignMode.HYBRID_REPLAY),
        _input(1, CoDesignMode.HYBRID_REPLAY),
        _input(2, CoDesignMode.HYBRID_REPLAY),
    )
    outputs = _loop(missing_steps=frozenset({1})).run(inputs)

    assert len(outputs) == 2
    assert outputs[-1].evaluation is None
    assert outputs[-1].proposal is None
    assert outputs[-1].safety.action is SafetyAction.ABORT
    assert outputs[-1].safety.applied_parameters == inputs[1].parameters


def test_stale_gradient_holds_previous_parameters() -> None:
    """Hold current parameters when the logical gradient age is stale."""
    output = _loop(max_age_ms=0.5).step(
        _input(0, CoDesignMode.CLASSICAL_TO_QUANTUM, apply_at_ms=2.0)
    )

    assert output.latency.stale is True
    assert output.safety.action is SafetyAction.HOLD
    assert output.safety.applied_parameters == (0.0, 0.6, 1.2)


def test_identity_observer_aborts_controller_application() -> None:
    """Give an identity-observer abort decision precedence over the proposal."""
    output = _loop().step(
        _input(0, CoDesignMode.CLASSICAL_TO_QUANTUM),
        observers=ObserverInputs(identity_action="abort", identity_reason="threshold trip"),
    )

    assert output.evaluation is not None
    assert output.safety.action is SafetyAction.ABORT
    assert output.safety.applied_parameters == (0.0, 0.6, 1.2)


def test_run_validates_sequence_contracts() -> None:
    """Reject empty inputs and observer sequences with wrong cardinality."""
    loop = _loop()
    with pytest.raises(ValueError, match="at least one"):
        loop.run(())
    with pytest.raises(ValueError, match="observer sequence"):
        loop.run((_input(0, CoDesignMode.HYBRID_REPLAY),), observers=())
