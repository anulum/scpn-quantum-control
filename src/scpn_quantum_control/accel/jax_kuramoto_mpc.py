# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — JAX differentiable-model receding-horizon MPC for the oscillator network
r"""A JAX differentiable-model receding-horizon model-predictive controller for the Kuramoto network.

Model-predictive control repeatedly plans a finite-horizon control from the current measured state,
applies the first control, then re-plans from the new measurement. This module builds an MPC whose
*predictive model* is the differentiable JAX Kuramoto rollout of the shipped JAX backend: the
finite-horizon optimal control is found by gradient descent on the control sequence, with the gradient
supplied by :func:`jax.value_and_grad` of the horizon objective, and the *plant* that the chosen control
is applied to is the production controlled integrator
:func:`~scpn_quantum_control.accel.kuramoto_network_control.integrate_controlled_network`.

The control convention is the additive per-oscillator drive of
:mod:`~scpn_quantum_control.accel.kuramoto_network_control`:

.. math::

    \dot\theta_i = \omega_i + \sum_j K_{ij}\sin(\theta_j - \theta_i) + u_i(t),

and the horizon objective is a coherence-tracking functional

.. math::

    J = \sum_t (r(\theta_{t+1}) - r^\star)^2\,\mathrm dt + w \sum_t \lVert u_t\rVert^2\,\mathrm dt ,

which drives the order parameter ``r`` toward a target ``r^\star`` (``target_coherence``) at least control
cost. The ``r^\star = 0`` case is exactly the desynchronisation objective of
:func:`~scpn_quantum_control.accel.kuramoto_network_control.network_control_value_and_grad`, so the JAX
autodiff control gradient is verified against that hand-written discrete adjoint to machine precision —
the autodiff tier both reproduces the hand-derived gradient and generalises it to any target ``r^\star``.

Because the JAX model and the NumPy plant are kept separate, the receding-horizon controller re-plans on
the *measured* plant state each step: when the plant differs from the model (a coupling mismatch), the
feedback corrects for the mismatch that an open-loop plan cannot — a property of receding-horizon control.

This tier is **opt-in** and requires JAX (``scpn-quantum-control[jax]``); it raises :class:`ImportError`
with an install hint when JAX is absent. It differentiates the model *rollout*; differentiating *through*
the inner optimiser's argmin (the implicit-function-theorem / KKT sensitivity of a learned MPC), a
neural-Lyapunov certificate, and a learned-surrogate predictive model are separate, later concerns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .kuramoto_network_control import integrate_controlled_network
from .order_parameter_observables import order_parameter

_OPTIMISERS = ("gd", "adam")


@dataclass(frozen=True)
class MpcControlGradients:
    """The coherence-tracking objective value and its gradient with respect to the control series.

    Attributes
    ----------
    cost : float
        The objective ``Σ_t (r(θ_{t+1}) − r*)² dt + w Σ_t ‖u_t‖² dt``.
    control_gradient : numpy.ndarray
        The ``(horizon, N)`` gradient with respect to the control series; a steepest-descent step is
        ``u ← u − η · control_gradient``.
    """

    cost: float
    control_gradient: NDArray[np.float64]


@dataclass(frozen=True)
class RecedingHorizonResult:
    """The closed-loop outcome of a receding-horizon MPC run on the true plant.

    Attributes
    ----------
    times : numpy.ndarray
        The ``(n_control_steps + 1,)`` sample times ``0, dt, …, n_control_steps·dt``.
    phases : numpy.ndarray
        The ``(n_control_steps + 1, N)`` closed-loop plant phase trajectory.
    applied_control : numpy.ndarray
        The ``(n_control_steps, N)`` first-control ``u_0`` applied at each control step.
    coherence : numpy.ndarray
        The order-parameter series ``r(θ_k)`` at every sample (length ``n_control_steps + 1``).
    horizon_cost : numpy.ndarray
        The final inner-solve horizon cost at each replan (length ``n_control_steps``).
    target_coherence : float
        The tracked target order parameter ``r*``.
    """

    times: NDArray[np.float64]
    phases: NDArray[np.float64]
    applied_control: NDArray[np.float64]
    coherence: NDArray[np.float64]
    horizon_cost: NDArray[np.float64]
    target_coherence: float

    @property
    def terminal_coherence(self) -> float:
        """The order parameter of the final closed-loop state."""
        return float(self.coherence[-1])


@dataclass(frozen=True)
class _MpcBackend:
    """The cached JAX backend: the JIT-compiled control-gradient seam and inner horizon solver."""

    jax: Any
    jnp: Any
    value_and_grad: Any
    horizon_solve: Any


_BACKEND: _MpcBackend | None = None


def _load_backend() -> _MpcBackend:
    """Return the cached JAX MPC backend, building it on first use.

    Enables 64-bit precision (a global JAX flag, set lazily) and JIT-compiles the control-sequence
    value-and-gradient seam and the fixed-iteration inner horizon solver (with the iteration count and
    the optimiser choice as static arguments so JAX caches one compiled kernel per configuration).

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
            "the JAX MPC tier requires JAX; install scpn-quantum-control[jax]"
        ) from error
    jax.config.update("jax_enable_x64", True)

    def force(theta: Any, coupling: Any) -> Any:
        difference = theta[None, :] - theta[:, None]
        return jnp.sum(coupling * jnp.sin(difference), axis=1)

    def rk4_step(theta: Any, control: Any, omega: Any, coupling: Any, dt: float) -> Any:
        # the control is held constant across all four stages, matching the production plant step
        k1 = omega + force(theta, coupling) + control
        k2 = omega + force(theta + 0.5 * dt * k1, coupling) + control
        k3 = omega + force(theta + 0.5 * dt * k2, coupling) + control
        k4 = omega + force(theta + dt * k3, coupling) + control
        return theta + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def rollout(theta0: Any, control: Any, omega: Any, coupling: Any, dt: float) -> Any:
        def body(theta: Any, step_control: Any) -> tuple[Any, Any]:
            advanced = rk4_step(theta, step_control, omega, coupling, dt)
            return advanced, advanced

        _, stepped = jax.lax.scan(body, theta0, control)
        return stepped  # (horizon, N) = θ_1 … θ_horizon

    def coherence(states: Any) -> Any:
        cosine = jnp.mean(jnp.cos(states), axis=1)
        sine = jnp.mean(jnp.sin(states), axis=1)
        return jnp.sqrt(cosine * cosine + sine * sine)

    def tracking_cost(
        theta0: Any,
        control: Any,
        omega: Any,
        coupling: Any,
        dt: float,
        target: float,
        weight: float,
    ) -> Any:
        states = rollout(theta0, control, omega, coupling, dt)
        radius = coherence(states)
        return jnp.sum((radius - target) ** 2) * dt + weight * jnp.sum(control**2) * dt

    value_and_grad = jax.value_and_grad(tracking_cost, argnums=1)

    def horizon_solve(
        theta0: Any,
        omega: Any,
        coupling: Any,
        dt: float,
        target: float,
        weight: float,
        step_size: float,
        n_iterations: int,
        initial_control: Any,
        use_adam: bool,
    ) -> tuple[Any, Any]:
        def gd_body(control: Any, _: Any) -> tuple[Any, Any]:
            cost, gradient = value_and_grad(theta0, control, omega, coupling, dt, target, weight)
            return control - step_size * gradient, cost

        def adam_body(carry: Any, _: Any) -> tuple[Any, Any]:
            control, first_moment, second_moment, count = carry
            cost, gradient = value_and_grad(theta0, control, omega, coupling, dt, target, weight)
            count = count + 1.0
            first_moment = 0.9 * first_moment + 0.1 * gradient
            second_moment = 0.999 * second_moment + 0.001 * gradient**2
            corrected_first = first_moment / (1.0 - 0.9**count)
            corrected_second = second_moment / (1.0 - 0.999**count)
            updated = control - step_size * corrected_first / (jnp.sqrt(corrected_second) + 1e-8)
            return (updated, first_moment, second_moment, count), cost

        if use_adam:
            zeros = jnp.zeros_like(initial_control)
            (control, _, _, _), history = jax.lax.scan(
                adam_body, (initial_control, zeros, zeros, 0.0), None, length=n_iterations
            )
        else:
            control, history = jax.lax.scan(gd_body, initial_control, None, length=n_iterations)
        final_cost = tracking_cost(theta0, control, omega, coupling, dt, target, weight)
        return control, jnp.concatenate([history, final_cost[None]])

    _BACKEND = _MpcBackend(
        jax=jax,
        jnp=jnp,
        value_and_grad=jax.jit(value_and_grad),
        horizon_solve=jax.jit(horizon_solve, static_argnums=(7, 9)),
    )
    return _BACKEND


def _validate_problem(
    phases: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    dt: float,
) -> int:
    """Validate the network problem and return the oscillator count ``N``."""
    count = phases.size
    if phases.ndim != 1 or count < 1:
        raise ValueError("phases must be a non-empty one-dimensional array")
    if omega.shape != (count,):
        raise ValueError(f"omega must have shape ({count},), got {omega.shape}")
    if coupling.shape != (count, count):
        raise ValueError(f"coupling must have shape ({count}, {count}), got {coupling.shape}")
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    return count


def _validate_objective(target_coherence: float, control_weight: float) -> None:
    """Validate the objective parameters shared by every entry point."""
    if not 0.0 <= target_coherence <= 1.0:
        raise ValueError(f"target_coherence must be in [0, 1], got {target_coherence}")
    if control_weight < 0.0:
        raise ValueError(f"control_weight must be non-negative, got {control_weight}")


def _use_adam(optimiser: str) -> bool:
    """Map the optimiser name to the static Adam flag, rejecting unknown names."""
    if optimiser not in _OPTIMISERS:
        raise ValueError(f"optimiser must be one of {_OPTIMISERS}, got {optimiser!r}")
    return optimiser == "adam"


def jax_mpc_control_value_and_grad(
    phases: NDArray[np.float64],
    control: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    dt: float,
    *,
    target_coherence: float,
    control_weight: float,
) -> MpcControlGradients:
    r"""Coherence-tracking objective value and its control-sequence gradient on the JAX tier.

    Evaluates ``J = Σ_t (r(θ_{t+1}) − r*)² dt + w Σ_t ‖u_t‖² dt`` for the controlled rollout
    ``θ̇ = ω + F(θ) + u`` and returns its gradient with respect to the whole ``(horizon, N)`` control
    series from :func:`jax.value_and_grad`. At ``target_coherence = 0`` this is exactly the objective of
    :func:`~scpn_quantum_control.accel.kuramoto_network_control.network_control_value_and_grad`, whose
    hand-derived discrete adjoint it reproduces to machine precision.

    Parameters
    ----------
    phases : numpy.ndarray
        The initial phases ``θ(0)`` (length ``N``).
    control : numpy.ndarray
        The ``(horizon, N)`` control series ``u_t``.
    omega : numpy.ndarray
        The natural frequencies ``ω`` (length ``N``).
    coupling : numpy.ndarray
        The ``(N, N)`` coupling matrix ``K``.
    dt : float
        The RK4 step (``> 0``).
    target_coherence : float
        The target order parameter ``r*`` in ``[0, 1]``.
    control_weight : float
        The control-energy weight ``w`` (``≥ 0``).

    Returns
    -------
    MpcControlGradients
        The cost and the ``(horizon, N)`` control gradient.

    Raises
    ------
    ValueError
        If any argument falls outside its documented bound.
    ImportError
        If JAX is not installed.
    """
    state = np.ascontiguousarray(phases, dtype=np.float64)
    series = np.ascontiguousarray(control, dtype=np.float64)
    frequencies = np.ascontiguousarray(omega, dtype=np.float64)
    matrix = np.ascontiguousarray(coupling, dtype=np.float64)
    count = _validate_problem(state, frequencies, matrix, dt)
    if series.ndim != 2 or series.shape[1] != count or series.shape[0] < 1:
        raise ValueError(f"control must have shape (horizon, {count}), got {series.shape}")
    _validate_objective(target_coherence, control_weight)
    backend = _load_backend()
    jnp = backend.jnp
    cost, gradient = backend.value_and_grad(
        jnp.asarray(state),
        jnp.asarray(series),
        jnp.asarray(frequencies),
        jnp.asarray(matrix),
        float(dt),
        float(target_coherence),
        float(control_weight),
    )
    return MpcControlGradients(
        cost=float(cost), control_gradient=np.asarray(gradient, dtype=np.float64)
    )


def jax_mpc_horizon_control(
    phases: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    dt: float,
    horizon: int,
    *,
    target_coherence: float,
    control_weight: float,
    step_size: float,
    n_iterations: int,
    initial_control: NDArray[np.float64] | None = None,
    optimiser: str = "gd",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Solve the finite-horizon optimal control by gradient descent on the JAX model.

    Runs ``n_iterations`` descent updates on the ``(horizon, N)`` control series to minimise the
    coherence-tracking objective, returning the optimised control and the cost history. The default
    ``optimiser="gd"`` is steepest descent (``u ← u − η ∇_u J``); ``"adam"`` is the adaptive-moment
    alternative for stiffer horizons.

    Parameters
    ----------
    phases, omega, coupling, dt
        As for :func:`jax_mpc_control_value_and_grad`.
    horizon : int
        The control-horizon length (``≥ 1``).
    target_coherence : float
        The target order parameter ``r*`` in ``[0, 1]``.
    control_weight : float
        The control-energy weight ``w`` (``≥ 0``).
    step_size : float
        The optimiser step size ``η`` (``> 0``).
    n_iterations : int
        The number of descent iterations (``≥ 1``).
    initial_control : numpy.ndarray, optional
        The starting ``(horizon, N)`` control; defaults to zeros (uncontrolled).
    optimiser : str, optional
        ``"gd"`` (default) or ``"adam"``.

    Returns
    -------
    tuple of numpy.ndarray
        The optimised ``(horizon, N)`` control and the ``(n_iterations + 1,)`` cost history.

    Raises
    ------
    ValueError
        If any argument falls outside its documented bound.
    ImportError
        If JAX is not installed.
    """
    state = np.ascontiguousarray(phases, dtype=np.float64)
    frequencies = np.ascontiguousarray(omega, dtype=np.float64)
    matrix = np.ascontiguousarray(coupling, dtype=np.float64)
    count = _validate_problem(state, frequencies, matrix, dt)
    _validate_objective(target_coherence, control_weight)
    if horizon < 1:
        raise ValueError(f"horizon must be positive, got {horizon}")
    if step_size <= 0.0:
        raise ValueError(f"step_size must be positive, got {step_size}")
    if n_iterations < 1:
        raise ValueError(f"n_iterations must be positive, got {n_iterations}")
    use_adam = _use_adam(optimiser)
    if initial_control is None:
        start = np.zeros((horizon, count), dtype=np.float64)
    else:
        start = np.ascontiguousarray(initial_control, dtype=np.float64)
        if start.shape != (horizon, count):
            raise ValueError(
                f"initial_control must have shape ({horizon}, {count}), got {start.shape}"
            )
    backend = _load_backend()
    jnp = backend.jnp
    control, history = backend.horizon_solve(
        jnp.asarray(state),
        jnp.asarray(frequencies),
        jnp.asarray(matrix),
        float(dt),
        float(target_coherence),
        float(control_weight),
        float(step_size),
        int(n_iterations),
        jnp.asarray(start),
        use_adam,
    )
    return (
        np.asarray(control, dtype=np.float64),
        np.asarray(history, dtype=np.float64),
    )


def receding_horizon_control(
    phases: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    dt: float,
    *,
    horizon: int,
    n_control_steps: int,
    target_coherence: float,
    control_weight: float,
    inner_iterations: int,
    inner_step_size: float,
    plant_coupling: NDArray[np.float64] | None = None,
    warm_start: bool = True,
    optimiser: str = "gd",
) -> RecedingHorizonResult:
    r"""Run a receding-horizon MPC: plan on the JAX model, apply the first control to the true plant.

    At each of ``n_control_steps`` control steps the controller solves the finite-horizon optimal
    control from the current *measured* state on the JAX model coupling, applies the first control
    ``u_0`` by advancing the true plant one step through
    :func:`~scpn_quantum_control.accel.kuramoto_network_control.integrate_controlled_network` with
    ``plant_coupling``, re-measures the order parameter and re-plans (warm-started by shifting the
    previous solution). When ``plant_coupling`` differs from ``coupling`` the feedback corrects the
    model/plant mismatch that an open-loop plan cannot.

    Parameters
    ----------
    phases, omega, coupling, dt
        As for :func:`jax_mpc_horizon_control`; ``coupling`` is the **model** matrix.
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
    plant_coupling : numpy.ndarray, optional
        The **plant** coupling matrix; defaults to ``coupling`` (no model/plant mismatch).
    warm_start : bool, optional
        Whether to warm-start each replan by shifting the previous solution; defaults to ``True``.
    optimiser : str, optional
        ``"gd"`` (default) or ``"adam"``.

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
    state = np.ascontiguousarray(phases, dtype=np.float64)
    frequencies = np.ascontiguousarray(omega, dtype=np.float64)
    matrix = np.ascontiguousarray(coupling, dtype=np.float64)
    count = _validate_problem(state, frequencies, matrix, dt)
    _validate_objective(target_coherence, control_weight)
    if horizon < 1:
        raise ValueError(f"horizon must be positive, got {horizon}")
    if n_control_steps < 1:
        raise ValueError(f"n_control_steps must be positive, got {n_control_steps}")
    if inner_iterations < 1:
        raise ValueError(f"inner_iterations must be positive, got {inner_iterations}")
    if inner_step_size <= 0.0:
        raise ValueError(f"inner_step_size must be positive, got {inner_step_size}")
    _use_adam(optimiser)
    if plant_coupling is None:
        plant_matrix = matrix
    else:
        plant_matrix = np.ascontiguousarray(plant_coupling, dtype=np.float64)
        if plant_matrix.shape != (count, count):
            raise ValueError(
                f"plant_coupling must have shape ({count}, {count}), got {plant_matrix.shape}"
            )

    trajectory = np.empty((n_control_steps + 1, count), dtype=np.float64)
    applied = np.empty((n_control_steps, count), dtype=np.float64)
    coherence_series = np.empty(n_control_steps + 1, dtype=np.float64)
    horizon_cost = np.empty(n_control_steps, dtype=np.float64)
    trajectory[0] = state
    coherence_series[0] = order_parameter(state)
    warm: NDArray[np.float64] = np.zeros((horizon, count), dtype=np.float64)

    for step in range(n_control_steps):
        control, history = jax_mpc_horizon_control(
            state,
            frequencies,
            matrix,
            dt,
            horizon,
            target_coherence=target_coherence,
            control_weight=control_weight,
            step_size=inner_step_size,
            n_iterations=inner_iterations,
            initial_control=warm,
            optimiser=optimiser,
        )
        first_control = control[0]
        state = integrate_controlled_network(
            state, first_control[None, :], frequencies, plant_matrix, dt
        ).terminal_phases
        applied[step] = first_control
        trajectory[step + 1] = state
        coherence_series[step + 1] = order_parameter(state)
        horizon_cost[step] = history[-1]
        if warm_start:
            warm = np.vstack([control[1:], np.zeros((1, count), dtype=np.float64)])
        else:
            warm = np.zeros((horizon, count), dtype=np.float64)

    times = dt * np.arange(n_control_steps + 1, dtype=np.float64)
    return RecedingHorizonResult(
        times=times,
        phases=trajectory,
        applied_control=applied,
        coherence=coherence_series,
        horizon_cost=horizon_cost,
        target_coherence=float(target_coherence),
    )


__all__ = [
    "MpcControlGradients",
    "RecedingHorizonResult",
    "jax_mpc_control_value_and_grad",
    "jax_mpc_horizon_control",
    "receding_horizon_control",
]
