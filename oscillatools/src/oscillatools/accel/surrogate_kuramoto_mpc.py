# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# oscillatools — receding-horizon Kuramoto MPC on a learned control-conditioned surrogate model
r"""A receding-horizon Kuramoto MPC that plans on a learned control-conditioned surrogate model.

The shipped receding-horizon controller
(:func:`~oscillatools.accel.jax_kuramoto_mpc.receding_horizon_control`) plans on the *true* Kuramoto
rollout — every horizon evaluation integrates the RK4 dynamics. A learned-surrogate variant replaces
the predictive model with a compact network so planning queries a single forward pass instead of the
rollout, at the cost of the surrogate's model error. This module builds that variant honestly and, in the
same call, reports how much closed-loop control quality it costs against the true-model controller.

The 8.5 DeepONet (:mod:`~oscillatools.neural_operator`) forecasts the *autonomous* solution map
``θ(0) ↦ θ(t)`` and carries no control input, so it cannot serve as the predictive model of a controlled
horizon. This tier therefore learns a **control-conditioned one-step surrogate**

.. math::

    \Phi_\psi(\theta, u) \approx \theta_{t+1} = \text{plant}(\theta, u),

trained to match the production controlled step
:func:`~oscillatools.accel.kuramoto_network_control.integrate_controlled_network` (the same plant the
closed loop is applied to). The surrogate is a small perceptron on the circle-respecting features
``(\cos\theta, \sin\theta, u)`` predicting the wrapped phase increment ``\Delta\theta``, so the phases may
drift freely while the features stay periodic.

Planning with the surrogate reuses the coherence-tracking objective of the shipped MPC,

.. math::

    J = \sum_t (r(\theta_{t+1}) - r^\star)^2\,\mathrm dt + w \sum_t \lVert u_t\rVert^2\,\mathrm dt ,

but rolls the horizon out through ``\Phi_\psi`` — differentiable end-to-end, so the control is optimised by
the same gradient descent. The receding-horizon loop applies the first planned control to the **true**
plant and re-plans from the measured state, exactly as the shipped controller does; the surrogate is thus
a model/plant mismatch whose feedback the receding horizon partly corrects.

Honest boundary (NV3 discipline): the surrogate is **not** a free substitute. Its one-step error is small
only over the control range it was trained on — a planner that drives it outside that range mispredicts,
so the closed-loop coherence it reaches falls at or below the true-model controller's. The cost is set by
the surrogate's accuracy over the controls the planner actually applies: near zero for a faithful
surrogate, larger when the planner leaves its training coverage. :func:`compare_surrogate_control` runs
both controllers on the same problem and returns the coherence gap and control energies so the cost is
reported, never hidden.

This tier is **opt-in** and requires JAX (``oscillatools[jax]``); it raises :class:`ImportError` with an
install hint when JAX is absent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .jax_kuramoto_mpc import RecedingHorizonResult, _validate_objective, receding_horizon_control
from .kuramoto_network_control import integrate_controlled_network
from .order_parameter_observables import order_parameter

_Layer = tuple[NDArray[np.float64], NDArray[np.float64]]
_Parameters = tuple[_Layer, ...]


@dataclass(frozen=True)
class SurrogateStepModel:
    """A trained control-conditioned one-step Kuramoto surrogate ``Φ_ψ(θ, u)``.

    Attributes
    ----------
    parameters : tuple
        The perceptron weights as a tuple of ``(weight, bias)`` layers, mapping the features
        ``(cos θ, sin θ, u)`` to the wrapped phase increment ``Δθ``.
    omega : numpy.ndarray
        The natural frequencies ``ω`` of the plant the surrogate was fitted to (length ``N``).
    coupling : numpy.ndarray
        The plant coupling matrix ``K`` (shape ``(N, N)``).
    dt : float
        The plant step the surrogate advances by.
    control_scale : float
        The standard deviation of the control perturbations the surrogate was trained over.
    training_loss : float
        The final mean-squared one-step increment error on the training sample.
    """

    parameters: _Parameters
    omega: NDArray[np.float64]
    coupling: NDArray[np.float64]
    dt: float
    control_scale: float
    training_loss: float


@dataclass(frozen=True)
class SurrogateControlComparison:
    """The closed-loop quality of the surrogate controller against the true-model controller.

    Attributes
    ----------
    surrogate_terminal_coherence : float
        The final order parameter reached by the surrogate-planned controller on the true plant.
    true_model_terminal_coherence : float
        The final order parameter reached by the true-model controller on the same plant.
    coherence_gap : float
        ``true_model_terminal_coherence − surrogate_terminal_coherence`` — the closed-loop cost of the
        surrogate's model error (positive when the surrogate tracks the target less well; near zero when
        the surrogate is faithful over the controls the planner applies).
    surrogate_control_energy : float
        The total applied control energy ``Σ_k ‖u_k‖² dt`` of the surrogate controller.
    true_model_control_energy : float
        The total applied control energy of the true-model controller.
    target_coherence : float
        The tracked target order parameter ``r*``.
    n_control_steps : int
        The number of closed-loop control steps both controllers ran.
    """

    surrogate_terminal_coherence: float
    true_model_terminal_coherence: float
    coherence_gap: float
    surrogate_control_energy: float
    true_model_control_energy: float
    target_coherence: float
    n_control_steps: int


@dataclass(frozen=True)
class _SurrogateBackend:
    """The cached JAX backend: the surrogate step, the horizon solver and the fitting gradient."""

    jax: Any
    jnp: Any
    surrogate_step: Any
    horizon_solve: Any
    loss_and_grad: Any


_BACKEND: _SurrogateBackend | None = None


def _load_backend() -> _SurrogateBackend:
    """Return the cached JAX surrogate-MPC backend, building it on first use.

    Enables 64-bit precision and JIT-compiles the surrogate one-step map, the fixed-iteration horizon
    solver that plans through it, and the mean-squared fitting gradient.

    Raises
    ------
    ImportError
        If JAX is not installed.
    """
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as error:  # pragma: no cover - exercised only without the optional extra
        raise ImportError(
            "the surrogate-MPC tier requires JAX; install oscillatools[jax]"
        ) from error
    jax.config.update("jax_enable_x64", True)

    def network(params: Any, features: Any) -> Any:
        activation = features
        for weight, bias in params[:-1]:
            activation = jnp.tanh(activation @ weight + bias)
        weight, bias = params[-1]
        return activation @ weight + bias

    def increment(params: Any, theta: Any, control: Any) -> Any:
        features = jnp.concatenate([jnp.cos(theta), jnp.sin(theta), control])
        return network(params, features)

    def surrogate_step(params: Any, theta: Any, control: Any) -> Any:
        return theta + increment(params, theta, control)

    def rollout(theta0: Any, control: Any, params: Any) -> Any:
        def body(theta: Any, step_control: Any) -> tuple[Any, Any]:
            advanced = surrogate_step(params, theta, step_control)
            return advanced, advanced

        _, states = jax.lax.scan(body, theta0, control)
        return states

    def coherence(states: Any) -> Any:
        cosine = jnp.mean(jnp.cos(states), axis=1)
        sine = jnp.mean(jnp.sin(states), axis=1)
        return jnp.sqrt(cosine * cosine + sine * sine)

    def tracking_cost(
        theta0: Any, control: Any, params: Any, target: float, weight: float, dt: float
    ) -> Any:
        states = rollout(theta0, control, params)
        radius = coherence(states)
        return jnp.sum((radius - target) ** 2) * dt + weight * jnp.sum(control**2) * dt

    control_value_and_grad = jax.value_and_grad(tracking_cost, argnums=1)

    def horizon_solve(
        theta0: Any,
        params: Any,
        target: float,
        weight: float,
        dt: float,
        step_size: float,
        n_iterations: int,
        initial_control: Any,
    ) -> tuple[Any, Any]:
        def gd_body(control: Any, _: Any) -> tuple[Any, Any]:
            cost, gradient = control_value_and_grad(theta0, control, params, target, weight, dt)
            return control - step_size * gradient, cost

        control, history = jax.lax.scan(gd_body, initial_control, None, length=n_iterations)
        final_cost = tracking_cost(theta0, control, params, target, weight, dt)
        return control, jnp.concatenate([history, final_cost[None]])

    def fitting_loss(params: Any, features: Any, targets: Any) -> Any:
        predicted = jax.vmap(lambda row: network(params, row))(features)
        residual = predicted - targets
        return jnp.mean(jnp.sum(residual * residual, axis=1))

    _BACKEND = _SurrogateBackend(
        jax=jax,
        jnp=jnp,
        surrogate_step=jax.jit(surrogate_step),
        horizon_solve=jax.jit(horizon_solve, static_argnums=(6,)),
        loss_and_grad=jax.jit(jax.value_and_grad(fitting_loss)),
    )
    return _BACKEND


def _validate_plant(omega: NDArray[np.float64], coupling: NDArray[np.float64], dt: float) -> int:
    """Validate the plant and return the oscillator count ``N``."""
    if omega.ndim != 1 or omega.size < 2:
        raise ValueError("omega must be a one-dimensional array of length at least two")
    count = int(omega.size)
    if coupling.shape != (count, count):
        raise ValueError(f"coupling must have shape ({count}, {count}), got {coupling.shape}")
    if not (np.all(np.isfinite(omega)) and np.all(np.isfinite(coupling))):
        raise ValueError("omega and coupling must be finite")
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    return count


def _init_parameters(
    rng: np.random.Generator, input_dim: int, hidden_layers: tuple[int, ...], output_dim: int
) -> list[_Layer]:
    """Glorot-uniform initialisation of the perceptron ``[input, *hidden, output]``."""
    dimensions = [input_dim, *hidden_layers, output_dim]
    layers: list[_Layer] = []
    for fan_in, fan_out in zip(dimensions[:-1], dimensions[1:], strict=True):
        limit = math.sqrt(6.0 / (fan_in + fan_out))
        weight = rng.uniform(-limit, limit, size=(fan_in, fan_out))
        bias = np.zeros(fan_out, dtype=np.float64)
        layers.append((np.ascontiguousarray(weight, dtype=np.float64), bias))
    return layers


def _as_jax_parameters(backend: _SurrogateBackend, parameters: _Parameters) -> Any:
    """Convert stored ``(weight, bias)`` layers to a JAX pytree."""
    jnp = backend.jnp
    return tuple((jnp.asarray(weight), jnp.asarray(bias)) for weight, bias in parameters)


def _as_numpy_parameters(parameters: Any) -> _Parameters:
    """Convert a JAX pytree of layers back to stored NumPy ``(weight, bias)`` layers."""
    return tuple(
        (np.asarray(weight, dtype=np.float64), np.asarray(bias, dtype=np.float64))
        for weight, bias in parameters
    )


def _adam(
    backend: _SurrogateBackend,
    params: Any,
    grads: Any,
    first: Any,
    second: Any,
    step: int,
    learning_rate: float,
) -> tuple[Any, Any, Any]:
    """One Adam update over the parameter pytree."""
    jnp = backend.jnp
    tree_map = backend.jax.tree_util.tree_map
    first = tree_map(lambda moment, grad: 0.9 * moment + 0.1 * grad, first, grads)
    second = tree_map(lambda moment, grad: 0.999 * moment + 0.001 * grad * grad, second, grads)
    bias_first = 1.0 - 0.9**step
    bias_second = 1.0 - 0.999**step
    params = tree_map(
        lambda weight, moment, raw: (
            weight - learning_rate * (moment / bias_first) / (jnp.sqrt(raw / bias_second) + 1e-8)
        ),
        params,
        first,
        second,
    )
    return params, first, second


def _plant_increment(
    states: NDArray[np.float64],
    controls: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    dt: float,
) -> NDArray[np.float64]:
    """The wrapped one-step increments ``Δθ`` of the true plant for a batch of ``(θ, u)`` pairs."""
    increments = np.empty_like(states)
    for index in range(states.shape[0]):
        advanced = integrate_controlled_network(
            states[index], controls[index][None, :], omega, coupling, dt
        ).terminal_phases
        increments[index] = advanced - states[index]
    return np.asarray((increments + np.pi) % (2.0 * np.pi) - np.pi, dtype=np.float64)


def fit_surrogate_step_model(
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    dt: float,
    *,
    control_scale: float = 0.75,
    hidden_layers: tuple[int, ...] = (64, 64),
    learning_rate: float = 5e-3,
    iterations: int = 400,
    sample_size: int = 2048,
    seed: int = 0,
) -> SurrogateStepModel:
    r"""Learn a control-conditioned one-step surrogate ``Φ_ψ(θ, u)`` of the true controlled plant.

    Samples random phase configurations and controls, records the true plant's wrapped one-step
    increment ``Δθ`` from :func:`~oscillatools.accel.kuramoto_network_control.integrate_controlled_network`,
    and fits a perceptron on the features ``(cos θ, sin θ, u)`` to that increment by full-batch Adam.

    Parameters
    ----------
    omega : numpy.ndarray
        The plant natural frequencies ``ω`` (length ``N ≥ 2``).
    coupling : numpy.ndarray
        The plant coupling matrix ``K`` (shape ``(N, N)``).
    dt : float
        The plant step to advance by (``> 0``).
    control_scale : float, optional
        The standard deviation of the Gaussian control perturbations sampled during training
        (``> 0``, default ``0.75``) — set it near the control magnitudes the MPC applies.
    hidden_layers : tuple of int, optional
        The perceptron hidden widths (default ``(64, 64)``); each must be positive.
    learning_rate : float, optional
        The Adam step size (``> 0``, default ``5e-3``).
    iterations : int, optional
        The number of full-batch Adam steps (``≥ 1``, default ``400``).
    sample_size : int, optional
        The number of ``(θ, u)`` training pairs (``≥ 1``, default ``2048``).
    seed : int, optional
        The seed for the deterministic sampling and initialisation (default ``0``).

    Returns
    -------
    SurrogateStepModel
        The trained surrogate and its final training loss.

    Raises
    ------
    ValueError
        If any argument falls outside its documented bound.
    ImportError
        If JAX is not installed.
    """
    frequencies = np.ascontiguousarray(omega, dtype=np.float64)
    matrix = np.ascontiguousarray(coupling, dtype=np.float64)
    count = _validate_plant(frequencies, matrix, dt)
    if control_scale <= 0.0:
        raise ValueError(f"control_scale must be positive, got {control_scale}")
    if any(width < 1 for width in hidden_layers) or len(hidden_layers) < 1:
        raise ValueError("hidden_layers must be a non-empty tuple of positive widths")
    if learning_rate <= 0.0:
        raise ValueError(f"learning_rate must be positive, got {learning_rate}")
    if iterations < 1:
        raise ValueError(f"iterations must be positive, got {iterations}")
    if sample_size < 1:
        raise ValueError(f"sample_size must be positive, got {sample_size}")

    rng = np.random.default_rng(seed)
    states = rng.uniform(0.0, 2.0 * np.pi, size=(sample_size, count))
    controls = rng.normal(0.0, control_scale, size=(sample_size, count))
    increments = _plant_increment(states, controls, frequencies, matrix, dt)
    features = np.concatenate([np.cos(states), np.sin(states), controls], axis=1)

    backend = _load_backend()
    jnp = backend.jnp
    feature_batch = jnp.asarray(features)
    target_batch = jnp.asarray(increments)
    params = _as_jax_parameters(
        backend, tuple(_init_parameters(rng, 3 * count, hidden_layers, count))
    )
    zeros = backend.jax.tree_util.tree_map(jnp.zeros_like, params)
    first, second = zeros, zeros
    loss = 0.0
    for step in range(iterations):
        loss_value, gradient = backend.loss_and_grad(params, feature_batch, target_batch)
        params, first, second = _adam(
            backend, params, gradient, first, second, step + 1, learning_rate
        )
        loss = float(loss_value)

    return SurrogateStepModel(
        parameters=_as_numpy_parameters(params),
        omega=frequencies,
        coupling=matrix,
        dt=float(dt),
        control_scale=float(control_scale),
        training_loss=loss,
    )


def surrogate_step(
    model: SurrogateStepModel, phases: NDArray[np.float64], control: NDArray[np.float64]
) -> NDArray[np.float64]:
    r"""Advance the phases one step through the learned surrogate ``Φ_ψ(θ, u)``.

    Parameters
    ----------
    model : SurrogateStepModel
        A surrogate from :func:`fit_surrogate_step_model`.
    phases : numpy.ndarray
        The phase configuration ``θ`` (length ``N``).
    control : numpy.ndarray
        The applied control ``u`` (length ``N``).

    Returns
    -------
    numpy.ndarray
        The predicted next phases ``θ + Φ_ψ(θ, u)`` (length ``N``).

    Raises
    ------
    ValueError
        If ``phases`` or ``control`` has the wrong shape.
    ImportError
        If JAX is not installed.
    """
    count = model.omega.size
    theta = np.ascontiguousarray(phases, dtype=np.float64)
    drive = np.ascontiguousarray(control, dtype=np.float64)
    if theta.shape != (count,):
        raise ValueError(f"phases must have shape ({count},), got {theta.shape}")
    if drive.shape != (count,):
        raise ValueError(f"control must have shape ({count},), got {drive.shape}")
    backend = _load_backend()
    jnp = backend.jnp
    params = _as_jax_parameters(backend, model.parameters)
    advanced = backend.surrogate_step(params, jnp.asarray(theta), jnp.asarray(drive))
    return np.asarray(advanced, dtype=np.float64)


def surrogate_receding_horizon_control(
    model: SurrogateStepModel,
    phases: NDArray[np.float64],
    *,
    horizon: int,
    n_control_steps: int,
    target_coherence: float,
    control_weight: float,
    inner_iterations: int,
    inner_step_size: float,
    warm_start: bool = True,
) -> RecedingHorizonResult:
    r"""Run a receding-horizon MPC that plans on the surrogate and drives the true plant.

    At each control step the horizon control is optimised by gradient descent through the surrogate
    rollout ``Φ_ψ``, the first control is applied to the true plant
    (:func:`~oscillatools.accel.kuramoto_network_control.integrate_controlled_network` with the model's
    ``ω``/``K``), the order parameter is re-measured and the plan is re-solved (warm-started by shifting
    the previous solution). The plant stepping is identical to the true-model controller, so the only
    difference is that the plan is made on the surrogate rather than the true rollout.

    Parameters
    ----------
    model : SurrogateStepModel
        A surrogate from :func:`fit_surrogate_step_model`.
    phases : numpy.ndarray
        The initial phases ``θ(0)`` (length ``N``).
    horizon : int
        The planning-horizon length (``≥ 1``).
    n_control_steps : int
        The number of closed-loop control steps (``≥ 1``).
    target_coherence : float
        The target order parameter ``r*`` in ``[0, 1]``.
    control_weight : float
        The control-energy weight ``w`` (``≥ 0``).
    inner_iterations : int
        The inner-solve descent iterations per replan (``≥ 1``).
    inner_step_size : float
        The inner-solve step size (``> 0``).
    warm_start : bool, optional
        Whether to warm-start each replan by shifting the previous solution; defaults to ``True``.

    Returns
    -------
    RecedingHorizonResult
        The closed-loop plant trajectory, applied control, coherence and per-step horizon cost.

    Raises
    ------
    ValueError
        If any argument falls outside its documented bound.
    ImportError
        If JAX is not installed.
    """
    count = model.omega.size
    state = np.ascontiguousarray(phases, dtype=np.float64)
    if state.shape != (count,):
        raise ValueError(f"phases must have shape ({count},), got {state.shape}")
    _validate_objective(target_coherence, control_weight)
    if horizon < 1:
        raise ValueError(f"horizon must be positive, got {horizon}")
    if n_control_steps < 1:
        raise ValueError(f"n_control_steps must be positive, got {n_control_steps}")
    if inner_iterations < 1:
        raise ValueError(f"inner_iterations must be positive, got {inner_iterations}")
    if inner_step_size <= 0.0:
        raise ValueError(f"inner_step_size must be positive, got {inner_step_size}")

    backend = _load_backend()
    jnp = backend.jnp
    params = _as_jax_parameters(backend, model.parameters)

    trajectory = np.empty((n_control_steps + 1, count), dtype=np.float64)
    applied = np.empty((n_control_steps, count), dtype=np.float64)
    coherence_series = np.empty(n_control_steps + 1, dtype=np.float64)
    horizon_cost = np.empty(n_control_steps, dtype=np.float64)
    trajectory[0] = state
    coherence_series[0] = order_parameter(state)
    warm = np.zeros((horizon, count), dtype=np.float64)

    for step in range(n_control_steps):
        control, history = backend.horizon_solve(
            jnp.asarray(state),
            params,
            float(target_coherence),
            float(control_weight),
            float(model.dt),
            float(inner_step_size),
            int(inner_iterations),
            jnp.asarray(warm),
        )
        control = np.asarray(control, dtype=np.float64)
        first_control = control[0]
        state = integrate_controlled_network(
            state, first_control[None, :], model.omega, model.coupling, model.dt
        ).terminal_phases
        applied[step] = first_control
        trajectory[step + 1] = state
        coherence_series[step + 1] = order_parameter(state)
        horizon_cost[step] = float(np.asarray(history)[-1])
        if warm_start:
            warm = np.vstack([control[1:], np.zeros((1, count), dtype=np.float64)])
        else:
            warm = np.zeros((horizon, count), dtype=np.float64)

    times = model.dt * np.arange(n_control_steps + 1, dtype=np.float64)
    return RecedingHorizonResult(
        times=times,
        phases=trajectory,
        applied_control=applied,
        coherence=coherence_series,
        horizon_cost=horizon_cost,
        target_coherence=float(target_coherence),
    )


def compare_surrogate_control(
    model: SurrogateStepModel,
    phases: NDArray[np.float64],
    *,
    horizon: int,
    n_control_steps: int,
    target_coherence: float,
    control_weight: float,
    inner_iterations: int,
    inner_step_size: float,
    warm_start: bool = True,
) -> SurrogateControlComparison:
    r"""Compare the surrogate-planned controller with the true-model controller on the same problem.

    Runs :func:`~oscillatools.accel.jax_kuramoto_mpc.receding_horizon_control` (planning on the true
    rollout) and :func:`surrogate_receding_horizon_control` (planning on the surrogate) from the same
    initial state against the same plant, and returns the closed-loop coherence gap and control energies.
    The gap is the honest cost of the surrogate's model error over the controls the planner applies: near
    zero for a faithful surrogate, larger when the planner leaves the surrogate's training coverage. It is
    reported rather than hidden.

    Parameters
    ----------
    model : SurrogateStepModel
        A surrogate from :func:`fit_surrogate_step_model`.
    phases : numpy.ndarray
        The initial phases ``θ(0)`` (length ``N``).
    horizon, n_control_steps, target_coherence, control_weight, inner_iterations, inner_step_size, warm_start
        As for :func:`surrogate_receding_horizon_control`; the same schedule drives both controllers.

    Returns
    -------
    SurrogateControlComparison
        The surrogate and true-model terminal coherences, the coherence gap, and the control energies.

    Raises
    ------
    ValueError
        If any argument falls outside its documented bound.
    ImportError
        If JAX is not installed.
    """
    true_result = receding_horizon_control(
        phases,
        model.omega,
        model.coupling,
        model.dt,
        horizon=horizon,
        n_control_steps=n_control_steps,
        target_coherence=target_coherence,
        control_weight=control_weight,
        inner_iterations=inner_iterations,
        inner_step_size=inner_step_size,
        warm_start=warm_start,
    )
    surrogate_result = surrogate_receding_horizon_control(
        model,
        phases,
        horizon=horizon,
        n_control_steps=n_control_steps,
        target_coherence=target_coherence,
        control_weight=control_weight,
        inner_iterations=inner_iterations,
        inner_step_size=inner_step_size,
        warm_start=warm_start,
    )
    surrogate_energy = float(np.sum(surrogate_result.applied_control**2) * model.dt)
    true_energy = float(np.sum(true_result.applied_control**2) * model.dt)
    return SurrogateControlComparison(
        surrogate_terminal_coherence=surrogate_result.terminal_coherence,
        true_model_terminal_coherence=true_result.terminal_coherence,
        coherence_gap=true_result.terminal_coherence - surrogate_result.terminal_coherence,
        surrogate_control_energy=surrogate_energy,
        true_model_control_energy=true_energy,
        target_coherence=float(target_coherence),
        n_control_steps=int(n_control_steps),
    )


__all__ = [
    "SurrogateControlComparison",
    "SurrogateStepModel",
    "compare_surrogate_control",
    "fit_surrogate_step_model",
    "surrogate_receding_horizon_control",
    "surrogate_step",
]
