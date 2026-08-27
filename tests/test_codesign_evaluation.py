# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — co-design component tests
"""Real phase-objective tests for co-design estimator, evaluator, and controller."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from scpn_quantum_control.codesign.components import (
    ExponentialOrderEstimator,
    GradientFeedbackController,
    OpenSystemObjectiveConfig,
    PhaseObjectiveSimulator,
    component_claim_boundary,
)
from scpn_quantum_control.codesign.contracts import BackendCapabilities, LoopStepInput
from scpn_quantum_control.control.closed_loop_analysis import ClosedLoopExecutionPolicy
from scpn_quantum_control.phase.open_system_objectives import (
    default_open_system_objective_cases,
)
from scpn_quantum_control.phase.synchronisation_objectives import (
    build_synchronisation_objective,
    kuramoto_order_parameter,
)


def _input(parameters: tuple[float, ...] = (0.0, 0.5, 1.0)) -> LoopStepInput:
    return LoopStepInput(
        step=0,
        observed_at_ms=10.0,
        apply_at_ms=13.0,
        parameters=parameters,
        measurement=0.7,
        target_order_parameter=0.9,
    )


def test_estimator_assimilates_samples_without_inventing_a_prior() -> None:
    """Initialise from the first sample and assimilate the next sample."""
    estimator = ExponentialOrderEstimator(alpha=0.25)

    first = estimator.update(0.8)
    second = estimator.update(0.4)

    assert first.order_parameter == 0.8
    assert first.innovation == 0.0
    assert second.order_parameter == pytest.approx(0.7)
    assert second.innovation == pytest.approx(-0.4)
    assert second.sample_count == 2


def test_estimator_uses_explicit_prior_and_validates_inputs() -> None:
    """Apply an explicit prior and reject invalid estimator inputs."""
    estimator = ExponentialOrderEstimator(alpha=1.0, initial_order_parameter=0.2)

    assert estimator.update(0.9).innovation == pytest.approx(0.7)
    with pytest.raises(ValueError, match="alpha"):
        ExponentialOrderEstimator(alpha=0.0)
    with pytest.raises(ValueError, match="initial_order_parameter"):
        ExponentialOrderEstimator(alpha=0.5, initial_order_parameter=2.0)
    with pytest.raises(ValueError, match="measurement"):
        estimator.update(float("nan"))


def test_simulator_matches_existing_phase_objective_and_planner() -> None:
    """Match the existing objective while attaching a local governed plan."""
    step_input = _input()
    simulator = PhaseObjectiveSimulator(policy=ClosedLoopExecutionPolicy())
    result = simulator.evaluate(step_input)

    assert result is not None
    objective = build_synchronisation_objective(3, order_parameter_target=0.9)
    reference = objective.evaluate(np.asarray(step_input.parameters, dtype=np.float64))
    assert result.objective_value == pytest.approx(reference.value)
    assert result.gradient == pytest.approx(reference.gradient)
    assert result.order_parameter == pytest.approx(kuramoto_order_parameter(step_input.parameters))
    assert result.gradient_source == "exact_phase_objective"
    assert result.gradient_plan.supported is True
    assert result.gradient_plan.selected_method == "parameter_shift"
    assert result.gradient_plan.requires_hardware_approval is False
    assert result.evaluated_at_ms == 11.0
    assert result.to_dict()["backend_status"] == "local_simulator"
    assert "no live QPU" in component_claim_boundary()


def test_simulator_models_missing_updates_deterministically() -> None:
    """Model a configured missing simulator update without hidden reuse."""
    simulator = PhaseObjectiveSimulator(
        policy=ClosedLoopExecutionPolicy(),
        missing_gradient_steps=frozenset({0}),
    )

    assert simulator.evaluate(_input()) is None


def test_simulator_refuses_an_unsupported_planner_route() -> None:
    """Refuse a governed route that is diagnostic rather than promoted."""
    simulator = PhaseObjectiveSimulator(
        policy=ClosedLoopExecutionPolicy(),
        planner_method="finite_difference",
    )

    with pytest.raises(RuntimeError, match="did not produce"):
        simulator.evaluate(_input())


def test_simulator_refuses_hardware_and_invalid_capabilities() -> None:
    """Reject hardware and internally inconsistent simulator capabilities."""
    with pytest.raises(ValueError, match="simulator-only"):
        PhaseObjectiveSimulator(
            policy=ClosedLoopExecutionPolicy(),
            capabilities=BackendCapabilities("provider", False, False, 10.0, hardware=True),
        )
    with pytest.raises(ValueError, match="logical_latency"):
        PhaseObjectiveSimulator(policy=ClosedLoopExecutionPolicy(), logical_latency_ms=-1.0)
    with pytest.raises(ValueError, match="exceeds"):
        PhaseObjectiveSimulator(
            policy=ClosedLoopExecutionPolicy(),
            capabilities=BackendCapabilities("sim", False, False, 1.0),
            logical_latency_ms=2.0,
        )
    with pytest.raises(ValueError, match="non-negative"):
        PhaseObjectiveSimulator(
            policy=ClosedLoopExecutionPolicy(),
            missing_gradient_steps=frozenset({-1}),
        )

    hardware_policy = ClosedLoopExecutionPolicy(
        allow_hardware=True,
        live_ticket="test-ticket",
        backend_allowlist=("statevector",),
    )
    with pytest.raises(PermissionError, match="refuses hardware"):
        PhaseObjectiveSimulator(policy=hardware_policy).evaluate(_input())


def test_optional_open_system_path_reuses_bounded_objective_suite() -> None:
    """Compose the actual bounded Lindblad objective at current inputs."""
    case = default_open_system_objective_cases()[0]
    simulator = PhaseObjectiveSimulator(
        policy=ClosedLoopExecutionPolicy(),
        open_system=OpenSystemObjectiveConfig(case, "lindblad_density", 0.1),
    )
    step_input = _input(tuple(float(value) for value in case.initial_params))
    result = simulator.evaluate(step_input)

    assert result is not None
    assert result.open_system_backend == "lindblad_density"
    assert "open_system:lindblad_density" in result.objective_terms
    assert np.all(np.isfinite(result.gradient))
    assert simulator.open_system is not None
    with pytest.raises(ValueError, match="weight"):
        replace(simulator.open_system, weight=-1.0)


def test_zero_weight_open_system_path_remains_inactive() -> None:
    """Leave the optional open-system augmentation inactive at zero weight."""
    case = default_open_system_objective_cases()[0]
    simulator = PhaseObjectiveSimulator(
        policy=ClosedLoopExecutionPolicy(),
        open_system=OpenSystemObjectiveConfig(case, "lindblad_density", 0.0),
    )
    result = simulator.evaluate(_input())

    assert result is not None
    assert result.open_system_backend is None
    assert all(not term.startswith("open_system:") for term in result.objective_terms)


def test_open_system_width_mismatch_fails_closed() -> None:
    """Reject dimensional drift between open-system and loop parameters."""
    case = default_open_system_objective_cases()[0]
    simulator = PhaseObjectiveSimulator(
        policy=ClosedLoopExecutionPolicy(),
        open_system=OpenSystemObjectiveConfig(case, "lindblad_density", 0.1),
    )
    with pytest.raises(ValueError, match="width"):
        simulator.evaluate(_input((0.0, 0.5, 1.0)))


def test_controller_applies_gradient_and_estimator_gain() -> None:
    """Apply exact gradient direction with estimator-dependent gain."""
    estimate = ExponentialOrderEstimator(alpha=1.0).update(0.5)
    controller = GradientFeedbackController(learning_rate=0.2, feedback_gain=0.5)
    proposal = controller.propose((1.0, -1.0), (0.5, -0.25), estimate, 0.9)

    assert proposal.gain_scale == pytest.approx(1.2)
    assert proposal.update == pytest.approx((-0.12, 0.06))
    assert proposal.parameters == pytest.approx((0.88, -0.94))


@pytest.mark.parametrize(
    ("controller", "args", "message"),
    [
        (GradientFeedbackController, {"learning_rate": 0.0}, "learning_rate"),
        (GradientFeedbackController, {"learning_rate": 0.1, "feedback_gain": -1.0}, "gains"),
    ],
)
def test_controller_validates_configuration(
    controller: type[GradientFeedbackController], args: dict[str, float], message: str
) -> None:
    """Reject non-positive or negative controller configuration."""
    with pytest.raises(ValueError, match=message):
        controller(**args)


def test_controller_rejects_invalid_runtime_vectors() -> None:
    """Reject mismatched, non-finite, or out-of-range runtime inputs."""
    controller = GradientFeedbackController(learning_rate=0.1)
    estimate = ExponentialOrderEstimator(alpha=1.0).update(0.5)

    with pytest.raises(ValueError, match="matching"):
        controller.propose((0.0,), (0.0, 1.0), estimate, 0.8)
    with pytest.raises(ValueError, match="finite"):
        controller.propose((0.0,), (float("nan"),), estimate, 0.8)
    with pytest.raises(ValueError, match="target_order_parameter"):
        controller.propose((0.0,), (0.0,), estimate, 2.0)
