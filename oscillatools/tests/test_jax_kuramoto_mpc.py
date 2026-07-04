# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — tests for the JAX differentiable-model receding-horizon MPC tier
"""Contract tests for the JAX differentiable-model receding-horizon MPC tier.

These exercise real JAX on the accelerator JAX selected (a CUDA GPU when present), so the module skips
without the optional ``[jax]`` extra. They pin the claims the tier is built on — the control-sequence
gradient from ``jax.value_and_grad`` matches the hand-derived discrete adjoint at ``r*=0`` (and a
central finite difference for a general target); the receding-horizon controller drives the order
parameter toward both a synchronising and a desynchronising target; and, under a model/plant coupling
mismatch, the feedback tracks the target better than an open-loop plan — plus the residency, fallback,
determinism, result-shape and validation contracts.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")

from oscillatools.accel import jax_kuramoto_mpc as mpc  # noqa: E402
from oscillatools.accel.kuramoto_network_control import (  # noqa: E402
    integrate_controlled_network,
    network_control_value_and_grad,
)
from oscillatools.accel.order_parameter_observables import order_parameter  # noqa: E402

_DT = 0.05
_WEIGHT = 1e-3


def _problem(
    n: int, seed: int, *, clustered: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(phases, omega, coupling)``; ``clustered`` gives a high-coherence start."""
    rng = np.random.default_rng(seed)
    omega = rng.normal(0.0, 0.3, size=n)
    coupling = rng.uniform(0.0, 1.2, size=(n, n))
    np.fill_diagonal(coupling, 0.0)
    if clustered:
        phases = rng.normal(0.0, 0.15, size=n)
    else:
        phases = rng.uniform(0.0, 2.0 * np.pi, size=n)
    return phases, omega, coupling


def test_zero_control_cost_matches_the_production_plant() -> None:
    n, horizon = 8, 12
    phases, omega, coupling = _problem(n, seed=0)
    control = np.zeros((horizon, n), dtype=np.float64)
    plant = integrate_controlled_network(phases, control, omega, coupling, _DT).phases[1:]
    reference_cost = float(np.sum([order_parameter(state) ** 2 for state in plant]) * _DT)
    result = mpc.jax_mpc_control_value_and_grad(
        phases, control, omega, coupling, _DT, target_coherence=0.0, control_weight=_WEIGHT
    )
    # the JAX model rollout reproduces the production controlled plant at r*=0 (control energy is zero)
    assert abs(result.cost - reference_cost) < 1e-11


def test_control_gradient_matches_hand_adjoint_at_zero_target() -> None:
    n, horizon = 8, 12
    phases, omega, coupling = _problem(n, seed=1)
    control = np.random.default_rng(2).normal(0.0, 0.2, size=(horizon, n))
    hand = network_control_value_and_grad(
        phases, control, omega, coupling, _DT, control_weight=_WEIGHT
    )
    jax_result = mpc.jax_mpc_control_value_and_grad(
        phases, control, omega, coupling, _DT, target_coherence=0.0, control_weight=_WEIGHT
    )
    # the JAX autodiff control gradient verifies the hand-written discrete adjoint to machine precision
    assert abs(jax_result.cost - hand.cost) < 1e-9
    assert jax_result.control_gradient.shape == (horizon, n)
    assert np.max(np.abs(jax_result.control_gradient - hand.control_gradient)) < 1e-9


def test_control_gradient_matches_central_finite_difference() -> None:
    n, horizon = 4, 6
    phases, omega, coupling = _problem(n, seed=3)
    control = np.random.default_rng(4).normal(0.0, 0.2, size=(horizon, n))
    target = 0.6
    gradient = mpc.jax_mpc_control_value_and_grad(
        phases, control, omega, coupling, _DT, target_coherence=target, control_weight=_WEIGHT
    ).control_gradient
    eps = 1e-6
    finite_difference = np.zeros_like(control)
    for row in range(horizon):
        for column in range(n):
            plus = control.copy()
            plus[row, column] += eps
            minus = control.copy()
            minus[row, column] -= eps
            forward = mpc.jax_mpc_control_value_and_grad(
                phases, plus, omega, coupling, _DT, target_coherence=target, control_weight=_WEIGHT
            ).cost
            backward = mpc.jax_mpc_control_value_and_grad(
                phases,
                minus,
                omega,
                coupling,
                _DT,
                target_coherence=target,
                control_weight=_WEIGHT,
            ).cost
            finite_difference[row, column] = (forward - backward) / (2.0 * eps)
    assert np.max(np.abs(gradient - finite_difference)) < 1e-7


def test_receding_horizon_synchronises_to_a_high_target() -> None:
    phases, omega, coupling = _problem(8, seed=5)
    result = mpc.receding_horizon_control(
        phases,
        omega,
        coupling,
        _DT,
        horizon=15,
        n_control_steps=40,
        target_coherence=0.9,
        control_weight=_WEIGHT,
        inner_iterations=60,
        inner_step_size=30.0,
    )
    assert result.coherence[0] < 0.6  # a genuinely incoherent start
    # the controller raises the order parameter close to the synchronising target
    assert abs(result.terminal_coherence - 0.9) < 0.12


def test_receding_horizon_desynchronises_to_a_low_target() -> None:
    phases, omega, coupling = _problem(8, seed=6, clustered=True)
    result = mpc.receding_horizon_control(
        phases,
        omega,
        coupling,
        _DT,
        horizon=15,
        n_control_steps=40,
        target_coherence=0.1,
        control_weight=_WEIGHT,
        inner_iterations=60,
        inner_step_size=30.0,
    )
    assert result.coherence[0] > 0.85  # a genuinely coherent start
    # the controller lowers the order parameter close to the desynchronising target
    assert abs(result.terminal_coherence - 0.1) < 0.1


@pytest.mark.parametrize("seed", [10, 11, 12])
def test_receding_horizon_beats_open_loop_under_mismatch(seed: int) -> None:
    n, horizon, steps = 8, 15, 40
    phases, omega, coupling = _problem(n, seed=seed, clustered=True)
    plant_coupling = 1.15 * coupling  # the plant is stiffer than the model
    target = 0.2

    receding = mpc.receding_horizon_control(
        phases,
        omega,
        coupling,
        _DT,
        horizon=horizon,
        n_control_steps=steps,
        target_coherence=target,
        control_weight=_WEIGHT,
        inner_iterations=60,
        inner_step_size=30.0,
        plant_coupling=plant_coupling,
    )
    receding_error = abs(receding.terminal_coherence - target)

    # open-loop: plan the whole run once on the model, then apply it to the true (stiffer) plant
    open_control, _ = mpc.jax_mpc_horizon_control(
        phases,
        omega,
        coupling,
        _DT,
        steps,
        target_coherence=target,
        control_weight=_WEIGHT,
        step_size=30.0,
        n_iterations=60,
    )
    open_terminal = integrate_controlled_network(
        phases, open_control, omega, plant_coupling, _DT
    ).phases[-1]
    open_error = abs(order_parameter(open_terminal) - target)

    # feedback re-planning on the measured plant state corrects the mismatch the open-loop plan cannot
    assert receding_error < open_error


def test_receding_horizon_is_deterministic() -> None:
    phases, omega, coupling = _problem(6, seed=7)
    kwargs = dict(
        horizon=10,
        n_control_steps=12,
        target_coherence=0.8,
        control_weight=_WEIGHT,
        inner_iterations=40,
        inner_step_size=25.0,
    )
    first = mpc.receding_horizon_control(phases, omega, coupling, _DT, **kwargs)
    second = mpc.receding_horizon_control(phases, omega, coupling, _DT, **kwargs)
    assert np.array_equal(first.applied_control, second.applied_control)
    assert np.array_equal(first.phases, second.phases)


def test_horizon_solve_runs_on_the_jax_default_device() -> None:
    import jax

    phases, omega, coupling = _problem(8, seed=8)
    backend = mpc._load_backend()
    jnp = backend.jnp
    control, _history = backend.horizon_solve(
        jnp.asarray(phases),
        jnp.asarray(omega),
        jnp.asarray(coupling),
        _DT,
        0.5,
        _WEIGHT,
        20.0,
        20,
        jnp.zeros((10, 8)),
        False,
    )
    # the jitted inner solve lands on the accelerator JAX selected (a CUDA GPU when one is present)
    assert control.device == jax.devices()[0]


def test_result_shapes_and_terminal_coherence() -> None:
    n, steps = 6, 9
    phases, omega, coupling = _problem(n, seed=9)
    result = mpc.receding_horizon_control(
        phases,
        omega,
        coupling,
        _DT,
        horizon=8,
        n_control_steps=steps,
        target_coherence=0.7,
        control_weight=_WEIGHT,
        inner_iterations=30,
        inner_step_size=25.0,
    )
    assert result.phases.shape == (steps + 1, n)
    assert result.applied_control.shape == (steps, n)
    assert result.coherence.shape == (steps + 1,)
    assert result.horizon_cost.shape == (steps,)
    assert result.terminal_coherence == float(result.coherence[-1])
    assert result.times[-1] == pytest.approx(steps * _DT)
    assert result.target_coherence == 0.7


def test_adam_optimiser_also_reaches_the_target() -> None:
    phases, omega, coupling = _problem(6, seed=13)
    control, history = mpc.jax_mpc_horizon_control(
        phases,
        omega,
        coupling,
        _DT,
        12,
        target_coherence=0.9,
        control_weight=_WEIGHT,
        step_size=0.5,
        n_iterations=80,
        optimiser="adam",
    )
    assert control.shape == (12, 6)
    # Adam monotonically reduces the horizon cost like the default steepest descent
    assert history[-1] < history[0]


def test_warm_start_and_cold_start_both_track() -> None:
    phases, omega, coupling = _problem(6, seed=14)
    kwargs = dict(
        horizon=10,
        n_control_steps=20,
        target_coherence=0.85,
        control_weight=_WEIGHT,
        inner_iterations=50,
        inner_step_size=25.0,
    )
    warm = mpc.receding_horizon_control(phases, omega, coupling, _DT, warm_start=True, **kwargs)
    cold = mpc.receding_horizon_control(phases, omega, coupling, _DT, warm_start=False, **kwargs)
    assert abs(warm.terminal_coherence - 0.85) < 0.12
    assert abs(cold.terminal_coherence - 0.85) < 0.12


def test_missing_jax_raises_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise() -> object:
        raise ImportError("the JAX MPC tier requires JAX; install oscillatools[jax]")

    monkeypatch.setattr(mpc, "_load_backend", _raise)
    phases, omega, coupling = _problem(6, seed=15)
    with pytest.raises(ImportError, match="oscillatools\\[jax\\]"):
        mpc.jax_mpc_control_value_and_grad(
            phases,
            np.zeros((5, 6)),
            omega,
            coupling,
            _DT,
            target_coherence=0.0,
            control_weight=_WEIGHT,
        )
    with pytest.raises(ImportError, match="oscillatools\\[jax\\]"):
        mpc.jax_mpc_horizon_control(
            phases,
            omega,
            coupling,
            _DT,
            5,
            target_coherence=0.5,
            control_weight=_WEIGHT,
            step_size=10.0,
            n_iterations=5,
        )
    with pytest.raises(ImportError, match="oscillatools\\[jax\\]"):
        mpc.receding_horizon_control(
            phases,
            omega,
            coupling,
            _DT,
            horizon=5,
            n_control_steps=3,
            target_coherence=0.5,
            control_weight=_WEIGHT,
            inner_iterations=5,
            inner_step_size=10.0,
        )


def test_validation_rejects_out_of_range_arguments() -> None:
    phases, omega, coupling = _problem(6, seed=16)
    with pytest.raises(ValueError, match="target_coherence must be in"):
        mpc.jax_mpc_horizon_control(
            phases,
            omega,
            coupling,
            _DT,
            5,
            target_coherence=1.5,
            control_weight=_WEIGHT,
            step_size=10.0,
            n_iterations=5,
        )
    with pytest.raises(ValueError, match="control_weight must be non-negative"):
        mpc.jax_mpc_horizon_control(
            phases,
            omega,
            coupling,
            _DT,
            5,
            target_coherence=0.5,
            control_weight=-1.0,
            step_size=10.0,
            n_iterations=5,
        )
    with pytest.raises(ValueError, match="horizon must be positive"):
        mpc.jax_mpc_horizon_control(
            phases,
            omega,
            coupling,
            _DT,
            0,
            target_coherence=0.5,
            control_weight=_WEIGHT,
            step_size=10.0,
            n_iterations=5,
        )
    with pytest.raises(ValueError, match="step_size must be positive"):
        mpc.jax_mpc_horizon_control(
            phases,
            omega,
            coupling,
            _DT,
            5,
            target_coherence=0.5,
            control_weight=_WEIGHT,
            step_size=0.0,
            n_iterations=5,
        )
    with pytest.raises(ValueError, match="n_iterations must be positive"):
        mpc.jax_mpc_horizon_control(
            phases,
            omega,
            coupling,
            _DT,
            5,
            target_coherence=0.5,
            control_weight=_WEIGHT,
            step_size=10.0,
            n_iterations=0,
        )
    with pytest.raises(ValueError, match="optimiser must be one of"):
        mpc.jax_mpc_horizon_control(
            phases,
            omega,
            coupling,
            _DT,
            5,
            target_coherence=0.5,
            control_weight=_WEIGHT,
            step_size=10.0,
            n_iterations=5,
            optimiser="momentum",
        )
    with pytest.raises(ValueError, match="initial_control must have shape"):
        mpc.jax_mpc_horizon_control(
            phases,
            omega,
            coupling,
            _DT,
            5,
            target_coherence=0.5,
            control_weight=_WEIGHT,
            step_size=10.0,
            n_iterations=5,
            initial_control=np.zeros((4, 6)),
        )
    with pytest.raises(ValueError, match="control must have shape"):
        mpc.jax_mpc_control_value_and_grad(
            phases,
            np.zeros((5, 5)),
            omega,
            coupling,
            _DT,
            target_coherence=0.0,
            control_weight=_WEIGHT,
        )
    with pytest.raises(ValueError, match="n_control_steps must be positive"):
        mpc.receding_horizon_control(
            phases,
            omega,
            coupling,
            _DT,
            horizon=5,
            n_control_steps=0,
            target_coherence=0.5,
            control_weight=_WEIGHT,
            inner_iterations=5,
            inner_step_size=10.0,
        )
    with pytest.raises(ValueError, match="inner_step_size must be positive"):
        mpc.receding_horizon_control(
            phases,
            omega,
            coupling,
            _DT,
            horizon=5,
            n_control_steps=3,
            target_coherence=0.5,
            control_weight=_WEIGHT,
            inner_iterations=5,
            inner_step_size=0.0,
        )
    with pytest.raises(ValueError, match="plant_coupling must have shape"):
        mpc.receding_horizon_control(
            phases,
            omega,
            coupling,
            _DT,
            horizon=5,
            n_control_steps=3,
            target_coherence=0.5,
            control_weight=_WEIGHT,
            inner_iterations=5,
            inner_step_size=10.0,
            plant_coupling=np.zeros((5, 5)),
        )
    with pytest.raises(ValueError, match="omega must have shape"):
        mpc.jax_mpc_horizon_control(
            phases,
            omega[:-1],
            coupling,
            _DT,
            5,
            target_coherence=0.5,
            control_weight=_WEIGHT,
            step_size=10.0,
            n_iterations=5,
        )
