# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — implicit-function-theorem sensitivity of the Kuramoto MPC optimum
r"""Differentiating *through* the Kuramoto MPC optimum by the implicit function theorem.

The shipped receding-horizon controller in :mod:`~oscillatools.accel.jax_kuramoto_mpc` finds the
finite-horizon optimal control by unrolling a gradient-descent solve and differentiates the horizon
*rollout*. This module differentiates the *argmin* itself: given the optimal control ``u*(θ)`` that
satisfies the first-order stationarity condition of the coherence-tracking objective, it returns the
sensitivity of ``u*`` to the problem parameters ``θ`` — the initial phases, the natural frequencies, and
the coupling matrix — without back-propagating through a single solver iteration.

The construction is the OptNet / differentiable-MPC implicit differentiation (Amos & Kolter,
*OptNet: Differentiable Optimization as a Layer in Neural Networks*, ICML 2017; Amos, Rodriguez, Sacks,
Boots & Kolter, *Differentiable MPC for End-to-end Planning and Control*, NeurIPS 2018), specialised to
the unconstrained horizon problem. The horizon objective is the coherence tracker of
:func:`~oscillatools.accel.jax_kuramoto_mpc.jax_mpc_control_value_and_grad`,

.. math::

    J(u;\theta) = \sum_t \bigl(r(\theta_{t+1}) - r^\star\bigr)^2\,\mathrm dt + w\sum_t\lVert u_t\rVert^2\,\mathrm dt ,

whose control has no explicit constraints, so the Karush–Kuhn–Tucker system collapses to the plain
stationarity residual

.. math::

    g(u,\theta) = \nabla_u J(u,\theta) = 0 .

At the optimum ``u*`` the implicit function theorem gives the exact parameter Jacobian

.. math::

    \frac{\partial u^\star}{\partial\theta}
        = -\Bigl(\frac{\partial g}{\partial u}\Bigr)^{-1}\frac{\partial g}{\partial\theta}
        = -H^{-1}\,\frac{\partial g}{\partial\theta},
    \qquad H \equiv \nabla^2_{uu} J ,

with ``H`` the (symmetric) Hessian of the objective in the control — the "KKT matrix" of the
unconstrained problem. The reverse-mode pullback of a cotangent ``ū`` on ``u*`` is therefore

.. math::

    \bar\theta = -\Bigl(\frac{\partial g}{\partial\theta}\Bigr)^{\!\top} H^{-1}\,\bar u
        = -\Bigl(\frac{\partial g}{\partial\theta}\Bigr)^{\!\top} w ,
    \qquad H\,w = \bar u ,

a single structured linear solve of the KKT matrix followed by one vector–Jacobian product of the
stationarity residual against the adjoint ``w`` — at fixed memory and independent of the iteration count
that produced ``u*``. The exact ``∂g/∂u`` and ``∂g/∂θ`` come from the differentiable JAX dynamics of the
shipped backend, so no new kernel is introduced (the module is Python orchestration over the accelerated
force/Jacobian tiers).

This tier is **opt-in** and requires JAX (``oscillatools[jax]``); it raises :class:`ImportError` with an
install hint when JAX is absent, through the shared backend loader.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .jax_kuramoto_mpc import (
    _load_backend,
    _use_adam,
    _validate_objective,
    _validate_problem,
)


@dataclass(frozen=True)
class MpcOptimumSensitivity:
    """The MPC optimum and the implicit-function-theorem pullback of a control cotangent.

    Attributes
    ----------
    optimal_control : numpy.ndarray
        The converged ``(horizon, N)`` optimal control ``u*`` that the sensitivity is taken at.
    stationarity_residual : float
        The Euclidean norm ``‖∇_u J(u*, θ)‖`` at the returned optimum — the degree to which the
        first-order optimality condition is met, and hence the regime in which the implicit-function
        gradient is exact. A well-converged solve drives this toward zero.
    phases_cotangent : numpy.ndarray
        The pullback ``∂L/∂θ(0)`` of the seed control cotangent through ``u*(θ(0))`` (length ``N``).
    omega_cotangent : numpy.ndarray
        The pullback ``∂L/∂ω`` of the seed control cotangent through ``u*(ω)`` (length ``N``).
    coupling_cotangent : numpy.ndarray
        The pullback ``∂L/∂K`` of the seed control cotangent through ``u*(K)`` (shape ``(N, N)``).
    """

    optimal_control: NDArray[np.float64]
    stationarity_residual: float
    phases_cotangent: NDArray[np.float64]
    omega_cotangent: NDArray[np.float64]
    coupling_cotangent: NDArray[np.float64]


def _make_implicit_plan(
    backend: Any,
    *,
    dt: float,
    target: float,
    weight: float,
    step_size: float,
    n_iterations: int,
    use_adam: bool,
    initial_control: Any,
) -> Any:
    r"""Build the JAX custom-VJP plan map ``(θ(0), ω, K) → u*`` with the IFT adjoint.

    The primal solves the horizon problem with the shipped gradient-descent solver; the custom backward
    replaces differentiation-through-iterations with one KKT linear solve ``H w = ū`` and one
    vector–Jacobian product ``−(∂g/∂θ)ᵀ w`` of the stationarity residual, per the implicit function
    theorem. ``dt``, the objective weights, the solver schedule and the warm-start control are baked in
    as constants, so the returned map is differentiated only with respect to the three dynamics
    parameters.
    """
    jax = backend.jax
    jnp = backend.jnp
    value_and_grad = backend.value_and_grad
    horizon_solve = backend.horizon_solve

    def stationarity(control: Any, phases: Any, omega: Any, coupling: Any) -> Any:
        """The control-gradient ``∇_u J`` whose root is the horizon optimum."""
        return value_and_grad(phases, control, omega, coupling, dt, target, weight)[1]

    def primal(phases: Any, omega: Any, coupling: Any) -> Any:
        control, _ = horizon_solve(
            phases,
            omega,
            coupling,
            dt,
            target,
            weight,
            step_size,
            n_iterations,
            initial_control,
            use_adam,
        )
        return control

    plan = jax.custom_vjp(primal)

    def plan_fwd(phases: Any, omega: Any, coupling: Any) -> tuple[Any, Any]:
        optimum = primal(phases, omega, coupling)
        return optimum, (optimum, phases, omega, coupling)

    def plan_bwd(residual: Any, control_bar: Any) -> tuple[Any, Any, Any]:
        optimum, phases, omega, coupling = residual
        size = optimum.size

        def residual_in_control(control: Any) -> Any:
            return stationarity(control, phases, omega, coupling)

        kkt = jax.jacfwd(residual_in_control)(optimum).reshape(size, size)
        adjoint = jnp.linalg.solve(kkt, control_bar.reshape(size)).reshape(optimum.shape)
        _, pull_parameters = jax.vjp(
            lambda p, o, c: stationarity(optimum, p, o, c), phases, omega, coupling
        )
        phases_bar, omega_bar, coupling_bar = pull_parameters(adjoint)
        return (-phases_bar, -omega_bar, -coupling_bar)

    plan.defvjp(plan_fwd, plan_bwd)
    return plan


def _cotangent_horizon(control_cotangent: NDArray[np.float64], count: int) -> int:
    """Validate the control cotangent against the oscillator count and read off its horizon."""
    if control_cotangent.ndim != 2 or control_cotangent.shape[1] != count:
        raise ValueError(
            f"control_cotangent must have shape (horizon, {count}), got {control_cotangent.shape}"
        )
    return int(control_cotangent.shape[0])


def mpc_optimum_parameter_sensitivity(
    phases: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    dt: float,
    control_cotangent: NDArray[np.float64],
    *,
    target_coherence: float,
    control_weight: float,
    step_size: float,
    n_iterations: int,
    optimiser: str = "gd",
    initial_control: NDArray[np.float64] | None = None,
) -> MpcOptimumSensitivity:
    r"""Pull a control cotangent back through the MPC optimum by the implicit function theorem.

    Solves the finite-horizon problem for the optimal control ``u*`` and returns the reverse-mode
    sensitivity ``∂L/∂θ`` of any downstream scalar ``L(u*)`` whose control cotangent
    ``∂L/∂u* = control_cotangent`` is supplied, for the three dynamics parameters ``θ ∈ {θ(0), ω, K}``.
    The gradient is the exact implicit-function-theorem adjoint — a KKT linear solve of the objective
    Hessian in the control followed by one vector–Jacobian product of the stationarity residual — so it
    is independent of the number of solver iterations and never differentiates through them.

    Parameters
    ----------
    phases : numpy.ndarray
        The initial phases ``θ(0)`` (length ``N``).
    omega : numpy.ndarray
        The natural frequencies ``ω`` (length ``N``).
    coupling : numpy.ndarray
        The coupling matrix ``K`` (shape ``(N, N)``).
    dt : float
        The integration step (``> 0``).
    control_cotangent : numpy.ndarray
        The seed cotangent ``∂L/∂u*`` on the optimum, shape ``(horizon, N)``; seeding it with ``u*``
        itself yields the gradient of the plan energy ``½‖u*‖²``.
    target_coherence : float
        The target order parameter ``r*`` in ``[0, 1]``.
    control_weight : float
        The control-energy weight ``w`` (``≥ 0``).
    step_size : float
        The solver step size ``η`` (``> 0``).
    n_iterations : int
        The number of solver iterations used to reach ``u*`` (``≥ 1``); larger values tighten the
        optimality residual and hence the exactness of the implicit-function gradient.
    optimiser : str, optional
        ``"gd"`` (default) or ``"adam"``.
    initial_control : numpy.ndarray, optional
        The warm-start ``(horizon, N)`` control; defaults to zeros (uncontrolled).

    Returns
    -------
    MpcOptimumSensitivity
        The optimum, its stationarity residual, and the pulled-back cotangents for ``θ(0)``, ``ω`` and
        ``K``.

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
    cotangent = np.ascontiguousarray(control_cotangent, dtype=np.float64)
    count = _validate_problem(state, frequencies, matrix, dt)
    _validate_objective(target_coherence, control_weight)
    horizon = _cotangent_horizon(cotangent, count)
    if step_size <= 0.0:
        raise ValueError(f"step_size must be positive, got {step_size}")
    if n_iterations < 1:
        raise ValueError(f"n_iterations must be positive, got {n_iterations}")
    use_adam = _use_adam(optimiser)
    start: NDArray[np.float64]
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
    plan = _make_implicit_plan(
        backend,
        dt=float(dt),
        target=float(target_coherence),
        weight=float(control_weight),
        step_size=float(step_size),
        n_iterations=int(n_iterations),
        use_adam=use_adam,
        initial_control=jnp.asarray(start),
    )
    optimum, pullback = backend.jax.vjp(
        plan, jnp.asarray(state), jnp.asarray(frequencies), jnp.asarray(matrix)
    )
    phases_bar, omega_bar, coupling_bar = pullback(jnp.asarray(cotangent))
    residual_vector = backend.value_and_grad(
        jnp.asarray(state),
        optimum,
        jnp.asarray(frequencies),
        jnp.asarray(matrix),
        float(dt),
        float(target_coherence),
        float(control_weight),
    )[1]
    return MpcOptimumSensitivity(
        optimal_control=np.asarray(optimum, dtype=np.float64),
        stationarity_residual=float(np.linalg.norm(np.asarray(residual_vector, dtype=np.float64))),
        phases_cotangent=np.asarray(phases_bar, dtype=np.float64),
        omega_cotangent=np.asarray(omega_bar, dtype=np.float64),
        coupling_cotangent=np.asarray(coupling_bar, dtype=np.float64),
    )


def mpc_plan_energy_gradient(
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
    optimiser: str = "gd",
    initial_control: NDArray[np.float64] | None = None,
) -> MpcOptimumSensitivity:
    r"""Gradient of the optimal plan energy ``½‖u*‖²`` with respect to the dynamics parameters.

    A concrete instance of :func:`mpc_optimum_parameter_sensitivity`: it solves for ``u*`` and returns
    how the control effort the controller must expend responds to the initial phases, the natural
    frequencies and the coupling — using the implicit-function-theorem adjoint seeded with the cotangent
    ``∂(½‖u*‖²)/∂u* = u*``. This answers "which problem parameters make the horizon cheaper or more
    expensive to steer" with an exact, iteration-independent gradient.

    Parameters
    ----------
    phases, omega, coupling, dt
        As for :func:`mpc_optimum_parameter_sensitivity`.
    horizon : int
        The control-horizon length (``≥ 1``).
    target_coherence, control_weight, step_size, n_iterations, optimiser, initial_control
        As for :func:`mpc_optimum_parameter_sensitivity`.

    Returns
    -------
    MpcOptimumSensitivity
        The optimum and the plan-energy gradient with respect to ``θ(0)``, ``ω`` and ``K``.

    Raises
    ------
    ValueError
        If any argument falls outside its documented bound.
    ImportError
        If JAX is not installed.
    """
    optimum = jax_mpc_optimum(
        phases,
        omega,
        coupling,
        dt,
        horizon,
        target_coherence=target_coherence,
        control_weight=control_weight,
        step_size=step_size,
        n_iterations=n_iterations,
        optimiser=optimiser,
        initial_control=initial_control,
    )
    return mpc_optimum_parameter_sensitivity(
        phases,
        omega,
        coupling,
        dt,
        optimum,
        target_coherence=target_coherence,
        control_weight=control_weight,
        step_size=step_size,
        n_iterations=n_iterations,
        optimiser=optimiser,
        initial_control=initial_control,
    )


def jax_mpc_optimum(
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
    optimiser: str = "gd",
    initial_control: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    r"""Solve the finite-horizon problem and return the optimal control ``u*``.

    The forward solve of the implicit-differentiation tier: it runs the shipped gradient-descent horizon
    solver and returns only the converged ``(horizon, N)`` optimum, discarding the cost history. It is
    the point at which :func:`mpc_optimum_parameter_sensitivity` takes the parameter derivative.

    Parameters
    ----------
    phases, omega, coupling, dt, horizon
        The network problem and control-horizon length.
    target_coherence, control_weight, step_size, n_iterations, optimiser, initial_control
        The objective and solver schedule, as for :func:`mpc_optimum_parameter_sensitivity`.

    Returns
    -------
    numpy.ndarray
        The optimised ``(horizon, N)`` control ``u*``.

    Raises
    ------
    ValueError
        If any argument falls outside its documented bound.
    ImportError
        If JAX is not installed.
    """
    from .jax_kuramoto_mpc import jax_mpc_horizon_control

    optimum, _ = jax_mpc_horizon_control(
        phases,
        omega,
        coupling,
        dt,
        horizon,
        target_coherence=target_coherence,
        control_weight=control_weight,
        step_size=step_size,
        n_iterations=n_iterations,
        initial_control=initial_control,
        optimiser=optimiser,
    )
    return optimum
