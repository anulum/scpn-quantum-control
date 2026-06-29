# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Differentiable-simulation policy learning for desynchronisation
r"""Learning a feedback policy on the differentiable Kuramoto simulator (analytic policy gradient).

The open-loop controllers in this toolkit optimise a *control time series* for one trajectory. A
*policy* instead is a feedback law ``u = π_p(θ)`` — a function of the current state — and so transfers
to initial conditions it was never trained on. This module learns such a policy by the *analytic
policy gradient*: it differentiates the cumulative reward of a closed-loop rollout straight through
the differentiable Kuramoto integrator (forward-mode sensitivity through the RK4 map), which is the
model-based reinforcement-learning route — using the differentiable model in place of model-free
sampling.

Honest scope: model-free actor–critic methods (PPO, TD3) are the dominant, validated route for
synchrony-suppression control and need no differentiable model; the advantage of differentiating the
model is a *bet*, strongest for initial-condition-sensitive problems, not a guaranteed win over
model-free RL. This module ships the differentiable-policy-gradient capability — exact gradients and a
policy that generalises across initial conditions — without claiming it beats model-free RL.

The policy is a learnable mean-field harmonic feedback

.. math::

    u_i(θ; a, b) = \sum_{m=1}^{M}\frac1N\sum_l
        \bigl[a_m\sin\!\bigl(m(θ_l - θ_i)\bigr) + b_m\cos\!\bigl(m(θ_l - θ_i)\bigr)\bigr]

(a learnable Daido-type coupling: ``a_1 < 0`` is the desynchronising repulsion), trained to minimise
the rollout's order-parameter (desynchronisation) cost plus a gain penalty. It adds no compute kernel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .networked_kuramoto import networked_kuramoto_force, networked_kuramoto_jacobian
from .order_parameter_observables import order_parameter, order_parameter_gradient


@dataclass(frozen=True)
class DesynchronisingPolicy:
    """A learned mean-field harmonic feedback policy.

    Attributes
    ----------
    sine_gains : numpy.ndarray
        The ``(M,)`` sine harmonic gains ``a_m``.
    cosine_gains : numpy.ndarray
        The ``(M,)`` cosine harmonic gains ``b_m``.
    """

    sine_gains: NDArray[np.float64]
    cosine_gains: NDArray[np.float64]

    def control(self, phases: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate the feedback control ``u(θ)`` at the given phases."""
        return _policy_control(
            np.ascontiguousarray(phases, dtype=np.float64), self.sine_gains, self.cosine_gains
        )


@dataclass(frozen=True)
class PolicyRolloutGradients:
    """The closed-loop rollout cost and its gradients with respect to the policy gains.

    Attributes
    ----------
    cost : float
        The rollout cost ``Σ_k r(θ_k)² dt + λ(‖a‖² + ‖b‖²)``.
    sine_gradient, cosine_gradient : numpy.ndarray
        The ``(M,)`` gradients with respect to the sine and cosine gains.
    """

    cost: float
    sine_gradient: NDArray[np.float64]
    cosine_gradient: NDArray[np.float64]


def _policy_control(
    phases: NDArray[np.float64], sine: NDArray[np.float64], cosine: NDArray[np.float64]
) -> NDArray[np.float64]:
    """The mean-field harmonic feedback ``u(θ)``."""
    difference = phases[None, :] - phases[:, None]
    control = np.zeros(phases.size, dtype=np.float64)
    for index in range(sine.size):
        harmonic = index + 1
        control = control + np.mean(
            sine[index] * np.sin(harmonic * difference)
            + cosine[index] * np.cos(harmonic * difference),
            axis=1,
        )
    return np.asarray(control, dtype=np.float64)


def _policy_state_jacobian(
    phases: NDArray[np.float64], sine: NDArray[np.float64], cosine: NDArray[np.float64]
) -> NDArray[np.float64]:
    """The closed-loop Jacobian ``∂u/∂θ`` of the policy."""
    count = phases.size
    difference = phases[None, :] - phases[:, None]
    jacobian = np.zeros((count, count), dtype=np.float64)
    for index in range(sine.size):
        harmonic = index + 1
        weight = (
            harmonic
            / count
            * (
                sine[index] * np.cos(harmonic * difference)
                - cosine[index] * np.sin(harmonic * difference)
            )
        )
        jacobian = jacobian + weight
        np.fill_diagonal(jacobian, np.diag(jacobian) - weight.sum(axis=1))
    return jacobian


def _policy_parameter_jacobian(
    phases: NDArray[np.float64], n_harmonics: int
) -> NDArray[np.float64]:
    """The Jacobian ``∂u/∂[a; b]`` of the policy (shape ``(N, 2M)``)."""
    difference = phases[None, :] - phases[:, None]
    columns = [np.mean(np.sin((m + 1) * difference), axis=1) for m in range(n_harmonics)]
    columns.extend(np.mean(np.cos((m + 1) * difference), axis=1) for m in range(n_harmonics))
    return np.stack(columns, axis=1)


def _validate(
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    dt: float,
    n_steps: int,
    parameter_penalty: float,
    count: int,
) -> None:
    if omega.shape != (count,):
        raise ValueError(f"omega must have shape ({count},), got {omega.shape}")
    if coupling.shape != (count, count):
        raise ValueError(f"coupling must have shape ({count}, {count}), got {coupling.shape}")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    if n_steps < 1:
        raise ValueError(f"n_steps must be positive, got {n_steps}")
    if parameter_penalty < 0.0:
        raise ValueError(f"parameter_penalty must be non-negative, got {parameter_penalty}")
    if not (np.all(np.isfinite(omega)) and np.all(np.isfinite(coupling))):
        raise ValueError("omega and coupling must be finite")


def policy_rollout_value_and_grad(
    initial_phases: NDArray[np.float64],
    sine_gains: NDArray[np.float64],
    cosine_gains: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    dt: float,
    n_steps: int,
    *,
    parameter_penalty: float,
) -> PolicyRolloutGradients:
    r"""The closed-loop desynchronisation cost and its exact policy gradient by forward sensitivity.

    Rolls the closed-loop ``θ̇ = ω + F(θ) + u(θ)`` out from ``initial_phases`` while propagating the
    forward-mode sensitivity ``∂θ/∂[a; b]`` through the RK4 map, and returns the cost
    ``Σ_k r(θ_k)² dt + λ(‖a‖² + ‖b‖²)`` and its gradient — the analytic policy gradient, matching
    finite differences to machine precision.

    Parameters
    ----------
    initial_phases : numpy.ndarray
        The initial phases ``θ(0)`` (length ``N ≥ 2``).
    sine_gains, cosine_gains : numpy.ndarray
        The ``(M,)`` policy gains.
    omega : numpy.ndarray
        The natural frequencies ``ω`` (length ``N``).
    coupling : numpy.ndarray
        The ``(N, N)`` coupling matrix ``K``.
    dt : float
        The RK4 step (finite, ``> 0``).
    n_steps : int
        The rollout length (``≥ 1``).
    parameter_penalty : float
        The gain-penalty weight ``λ`` (``≥ 0``).

    Returns
    -------
    PolicyRolloutGradients
        The rollout cost and the sine / cosine gain gradients.

    Raises
    ------
    ValueError
        If any argument falls outside its documented bound.
    """
    phases = np.ascontiguousarray(initial_phases, dtype=np.float64)
    sine = np.ascontiguousarray(sine_gains, dtype=np.float64)
    cosine = np.ascontiguousarray(cosine_gains, dtype=np.float64)
    frequencies = np.ascontiguousarray(omega, dtype=np.float64)
    matrix = np.ascontiguousarray(coupling, dtype=np.float64)
    if phases.ndim != 1 or phases.size < 2:
        raise ValueError("initial_phases must be a one-dimensional array of length at least two")
    if sine.ndim != 1 or sine.shape != cosine.shape or sine.size < 1:
        raise ValueError("sine_gains and cosine_gains must be equal-length non-empty 1-D arrays")
    count = phases.size
    harmonics = sine.size
    _validate(frequencies, matrix, dt, n_steps, parameter_penalty, count)
    if not np.all(np.isfinite(phases)):
        raise ValueError("initial_phases must be finite")

    parameters = 2 * harmonics

    def derivative(
        state: NDArray[np.float64], sensitivity: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        field = (
            frequencies
            + networked_kuramoto_force(state, matrix)
            + _policy_control(state, sine, cosine)
        )
        closed_loop = networked_kuramoto_jacobian(state, matrix) + _policy_state_jacobian(
            state, sine, cosine
        )
        parameter_jacobian = _policy_parameter_jacobian(state, harmonics)
        return field, closed_loop @ sensitivity + parameter_jacobian

    state: NDArray[np.float64] = phases
    sensitivity: NDArray[np.float64] = np.zeros((count, parameters), dtype=np.float64)
    cost = 0.0
    gradient = np.zeros(parameters, dtype=np.float64)
    for _ in range(n_steps):
        k1, s1 = derivative(state, sensitivity)
        k2, s2 = derivative(state + 0.5 * dt * k1, sensitivity + 0.5 * dt * s1)
        k3, s3 = derivative(state + 0.5 * dt * k2, sensitivity + 0.5 * dt * s2)
        k4, s4 = derivative(state + dt * k3, sensitivity + dt * s3)
        state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        sensitivity = sensitivity + (dt / 6.0) * (s1 + 2.0 * s2 + 2.0 * s3 + s4)
        radius = order_parameter(state)
        cost += radius * radius * dt
        gradient += 2.0 * radius * (order_parameter_gradient(state) @ sensitivity) * dt

    penalty = parameter_penalty * (float(np.sum(sine**2)) + float(np.sum(cosine**2)))
    cost += penalty
    gradient += 2.0 * parameter_penalty * np.concatenate([sine, cosine])
    return PolicyRolloutGradients(
        cost=cost,
        sine_gradient=np.ascontiguousarray(gradient[:harmonics], dtype=np.float64),
        cosine_gradient=np.ascontiguousarray(gradient[harmonics:], dtype=np.float64),
    )


def learn_desynchronising_policy(
    initial_phases_batch: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    dt: float,
    n_steps: int,
    n_harmonics: int,
    *,
    parameter_penalty: float,
    learning_rate: float,
    n_iterations: int,
) -> tuple[DesynchronisingPolicy, NDArray[np.float64]]:
    r"""Learn a desynchronising feedback policy by analytic-policy-gradient descent over a batch.

    Averages the rollout policy gradient over a batch of initial conditions (so the learned feedback
    generalises across them) and applies ``n_iterations`` gradient-descent updates.

    Parameters
    ----------
    initial_phases_batch : numpy.ndarray
        The ``(B, N)`` batch of initial conditions to train across.
    omega, coupling, dt, n_steps, parameter_penalty
        As for :func:`policy_rollout_value_and_grad`.
    n_harmonics : int
        The number ``M`` of policy harmonics (``≥ 1``).
    learning_rate : float
        The gradient-descent step size (``> 0``).
    n_iterations : int
        The number of descent iterations (``≥ 1``).

    Returns
    -------
    tuple
        The learned :class:`DesynchronisingPolicy` and the ``(n_iterations + 1,)`` mean-cost history.

    Raises
    ------
    ValueError
        If any argument falls outside its documented bound.
    """
    batch = np.ascontiguousarray(initial_phases_batch, dtype=np.float64)
    if batch.ndim != 2 or batch.shape[0] < 1 or batch.shape[1] < 2:
        raise ValueError("initial_phases_batch must be a (B >= 1, N >= 2) array")
    if n_harmonics < 1:
        raise ValueError(f"n_harmonics must be positive, got {n_harmonics}")
    if learning_rate <= 0.0:
        raise ValueError(f"learning_rate must be positive, got {learning_rate}")
    if n_iterations < 1:
        raise ValueError(f"n_iterations must be positive, got {n_iterations}")

    sine: NDArray[np.float64] = np.zeros(n_harmonics, dtype=np.float64)
    cosine: NDArray[np.float64] = np.zeros(n_harmonics, dtype=np.float64)
    history = np.empty(n_iterations + 1, dtype=np.float64)
    for iteration in range(n_iterations + 1):
        total_cost = 0.0
        sine_gradient = np.zeros(n_harmonics, dtype=np.float64)
        cosine_gradient = np.zeros(n_harmonics, dtype=np.float64)
        for initial in batch:
            result = policy_rollout_value_and_grad(
                initial,
                sine,
                cosine,
                omega,
                coupling,
                dt,
                n_steps,
                parameter_penalty=parameter_penalty,
            )
            total_cost += result.cost
            sine_gradient += result.sine_gradient
            cosine_gradient += result.cosine_gradient
        history[iteration] = total_cost / batch.shape[0]
        if iteration < n_iterations:
            sine = sine - learning_rate * sine_gradient / batch.shape[0]
            cosine = cosine - learning_rate * cosine_gradient / batch.shape[0]
    return DesynchronisingPolicy(sine_gains=sine, cosine_gains=cosine), history


__all__ = [
    "DesynchronisingPolicy",
    "PolicyRolloutGradients",
    "learn_desynchronising_policy",
    "policy_rollout_value_and_grad",
]
