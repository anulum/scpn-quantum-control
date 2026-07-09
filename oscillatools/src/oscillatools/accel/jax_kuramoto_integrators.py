# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — JAX Kuramoto integrator parity tier
r"""Opt-in JAX trajectories for the non-RK4 networked-Kuramoto integrator family.

This module complements :mod:`oscillatools.accel.jax_kuramoto`, which owns the RK4 autodiff and
batched-ensemble surface. The functions here provide explicit JAX counterparts for the remaining
networked-Kuramoto trajectory integrators: fixed-step Euler, adaptive Dormand-Prince, inertial RK4,
symplectic inertial, and seeded noisy Euler-Maruyama. They are directly callable opt-in paths and are
not members of the default Rust → Julia → NumPy dispatch chains.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .diff_kuramoto_dopri import (
    _A as _DOPRI_A,
)
from .diff_kuramoto_dopri import (
    _B4 as _DOPRI_B4,
)
from .diff_kuramoto_dopri import (
    _B5 as _DOPRI_B5,
)
from .diff_kuramoto_dopri import (
    DopriTrajectory,
)
from .diff_kuramoto_dopri import (
    _validate_state as _validate_dopri_state,
)
from .diff_kuramoto_rk4 import _validate_forward
from .kuramoto_inertial import InertialTrajectory
from .kuramoto_noisy import NoisyKuramotoRun
from .networked_inertial import _validate_state as _validate_inertial_state
from .networked_noisy import _validate_noisy_state
from .networked_symplectic_inertial import (
    _validate_state as _validate_symplectic_inertial_state,
)


@dataclass(frozen=True)
class _IntegratorBackend:
    """Cached JAX modules and JIT-compiled fixed-shape trajectory kernels."""

    jax: Any
    jnp: Any
    euler_trajectory: Any
    inertial_trajectory: Any
    symplectic_inertial_trajectory: Any
    noisy_trajectory: Any


_BACKEND: _IntegratorBackend | None = None


def _load_backend() -> _IntegratorBackend:
    """Return the cached JAX integrator backend, building it on first use.

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
    except ImportError as error:
        raise ImportError(
            "the JAX Kuramoto integrator tier requires JAX; install oscillatools[jax]"
        ) from error
    jax.config.update("jax_enable_x64", True)

    def force(theta: Any, coupling: Any) -> Any:
        difference = theta[None, :] - theta[:, None]
        return jnp.sum(coupling * jnp.sin(difference), axis=1)

    def euler_step(theta: Any, omega: Any, coupling: Any, dt: float) -> Any:
        return theta + dt * (omega + force(theta, coupling))

    def euler_solve(theta0: Any, omega: Any, coupling: Any, dt: float, n_steps: int) -> Any:
        def body(carry: Any, _: Any) -> tuple[Any, Any]:
            advanced = euler_step(carry, omega, coupling, dt)
            return advanced, advanced

        if n_steps == 0:
            return theta0[None, :]
        _, stepped = jax.lax.scan(body, theta0, None, length=n_steps)
        return jnp.concatenate([theta0[None, :], stepped], axis=0)

    def inertial_vector(
        theta: Any, velocity: Any, omega: Any, coupling: Any, mass: float, damping: float
    ) -> Any:
        acceleration = (omega + force(theta, coupling) - damping * velocity) / mass
        return velocity, acceleration

    def inertial_step(
        theta: Any,
        velocity: Any,
        omega: Any,
        coupling: Any,
        mass: float,
        damping: float,
        dt: float,
    ) -> tuple[Any, Any]:
        k1_theta, k1_velocity = inertial_vector(theta, velocity, omega, coupling, mass, damping)
        k2_theta, k2_velocity = inertial_vector(
            theta + 0.5 * dt * k1_theta,
            velocity + 0.5 * dt * k1_velocity,
            omega,
            coupling,
            mass,
            damping,
        )
        k3_theta, k3_velocity = inertial_vector(
            theta + 0.5 * dt * k2_theta,
            velocity + 0.5 * dt * k2_velocity,
            omega,
            coupling,
            mass,
            damping,
        )
        k4_theta, k4_velocity = inertial_vector(
            theta + dt * k3_theta,
            velocity + dt * k3_velocity,
            omega,
            coupling,
            mass,
            damping,
        )
        next_theta = theta + (dt / 6.0) * (k1_theta + 2.0 * k2_theta + 2.0 * k3_theta + k4_theta)
        next_velocity = velocity + (dt / 6.0) * (
            k1_velocity + 2.0 * k2_velocity + 2.0 * k3_velocity + k4_velocity
        )
        return next_theta, next_velocity

    def inertial_solve(
        theta0: Any,
        velocities: Any,
        omega: Any,
        coupling: Any,
        mass: float,
        damping: float,
        dt: float,
        n_steps: int,
    ) -> tuple[Any, Any]:
        def body(carry: tuple[Any, Any], _: Any) -> tuple[tuple[Any, Any], tuple[Any, Any]]:
            theta, velocity = carry
            next_theta, next_velocity = inertial_step(
                theta, velocity, omega, coupling, mass, damping, dt
            )
            return (next_theta, next_velocity), (next_theta, next_velocity)

        _, (theta_steps, velocity_steps) = jax.lax.scan(
            body, (theta0, velocities), None, length=n_steps
        )
        return (
            jnp.concatenate([theta0[None, :], theta_steps], axis=0),
            jnp.concatenate([velocities[None, :], velocity_steps], axis=0),
        )

    def symplectic_step(
        theta: Any,
        velocity: Any,
        omega: Any,
        coupling: Any,
        mass: float,
        damping: float,
        dt: float,
    ) -> tuple[Any, Any]:
        decay = jnp.exp(-0.5 * damping * dt / mass)
        half_velocity = decay * velocity + 0.5 * dt * (omega + force(theta, coupling)) / mass
        next_theta = theta + dt * half_velocity
        kicked = half_velocity + 0.5 * dt * (omega + force(next_theta, coupling)) / mass
        return next_theta, decay * kicked

    def symplectic_solve(
        theta0: Any,
        velocities: Any,
        omega: Any,
        coupling: Any,
        mass: float,
        damping: float,
        dt: float,
        n_steps: int,
    ) -> tuple[Any, Any]:
        def body(carry: tuple[Any, Any], _: Any) -> tuple[tuple[Any, Any], tuple[Any, Any]]:
            theta, velocity = carry
            next_theta, next_velocity = symplectic_step(
                theta, velocity, omega, coupling, mass, damping, dt
            )
            return (next_theta, next_velocity), (next_theta, next_velocity)

        _, (theta_steps, velocity_steps) = jax.lax.scan(
            body, (theta0, velocities), None, length=n_steps
        )
        return (
            jnp.concatenate([theta0[None, :], theta_steps], axis=0),
            jnp.concatenate([velocities[None, :], velocity_steps], axis=0),
        )

    def noisy_solve(
        theta0: Any,
        omega: Any,
        coupling: Any,
        diffusion: float,
        dt: float,
        increments: Any,
    ) -> tuple[Any, Any]:
        scale = jnp.sqrt(2.0 * diffusion * dt)

        def body(theta: Any, increment: Any) -> tuple[Any, Any]:
            advanced = euler_step(theta, omega, coupling, dt) + scale * increment
            coherence = jnp.abs(jnp.mean(jnp.exp(1j * advanced)))
            return advanced, coherence

        terminal, series = jax.lax.scan(body, theta0, increments)
        return series, terminal

    _BACKEND = _IntegratorBackend(
        jax=jax,
        jnp=jnp,
        euler_trajectory=jax.jit(euler_solve, static_argnums=(4,)),
        inertial_trajectory=jax.jit(inertial_solve, static_argnums=(7,)),
        symplectic_inertial_trajectory=jax.jit(symplectic_solve, static_argnums=(7,)),
        noisy_trajectory=jax.jit(noisy_solve),
    )
    return _BACKEND


def _time_grid(dt: float, n_steps: int) -> NDArray[np.float64]:
    """Return the ``(n_steps + 1,)`` fixed-step time grid."""
    return np.arange(n_steps + 1, dtype=np.float64) * float(dt)


def _validate_fixed_step_params(
    mass: float | None,
    damping: float,
    dt: float,
    n_steps: int,
) -> None:
    """Validate the shared fixed-step range constraints."""
    if mass is not None and mass <= 0.0:
        raise ValueError(f"mass must be positive, got {mass}")
    if damping < 0.0:
        raise ValueError(f"damping must be non-negative, got {damping}")
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    if n_steps < 1:
        raise ValueError(f"n_steps must be positive, got {n_steps}")


def _jax_force(theta: Any, coupling: Any, jnp: Any) -> Any:
    """Return the JAX networked force used by the adaptive DOPRI loop."""
    difference = theta[None, :] - theta[:, None]
    return jnp.sum(coupling * jnp.sin(difference), axis=1)


def _jax_dopri_stage_derivatives(
    phases: Any,
    frequencies: Any,
    coupling: Any,
    step: float,
    jnp: Any,
) -> list[Any]:
    """Return the seven Dormand-Prince stage derivatives as JAX arrays."""
    derivatives: list[Any] = []
    for stage, coefficients in enumerate(_DOPRI_A):
        increment = phases
        for previous in range(stage):
            increment = increment + step * coefficients[previous] * derivatives[previous]
        derivatives.append(frequencies + _jax_force(increment, coupling, jnp))
    return derivatives


def jax_kuramoto_euler_trajectory(
    theta0: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    dt: float,
    n_steps: int,
) -> NDArray[np.float64]:
    r"""Forward networked-Kuramoto Euler trajectory evaluated on the JAX tier.

    Integrates :math:`\dot\theta = \omega + F(\theta)` by the same fixed-step Euler rule as
    :func:`~oscillatools.accel.diff_kuramoto_euler.kuramoto_euler_trajectory`, but expressed as a
    JAX scan so the full trajectory runs on the selected JAX device.

    Parameters
    ----------
    theta0 : numpy.ndarray
        The initial phases ``θ_0`` (length ``N``).
    omega : numpy.ndarray
        The natural frequencies ``ω`` (length ``N``).
    coupling : numpy.ndarray
        The ``(N, N)`` coupling matrix ``K``.
    dt : float
        The Euler step size.
    n_steps : int
        The number of Euler steps (``≥ 0``).

    Returns
    -------
    numpy.ndarray
        The ``(n_steps + 1, N)`` phase trajectory.

    Raises
    ------
    ValueError
        If the state shapes are inconsistent or ``n_steps`` is negative.
    ImportError
        If JAX is not installed.
    """
    phases, frequencies, matrix = _validate_forward(theta0, omega, coupling, n_steps)
    backend = _load_backend()
    jnp = backend.jnp
    result = backend.euler_trajectory(
        jnp.asarray(phases), jnp.asarray(frequencies), jnp.asarray(matrix), float(dt), int(n_steps)
    )
    return np.asarray(result, dtype=np.float64)


def jax_kuramoto_dopri_trajectory(
    theta0: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    *,
    t_end: float,
    rtol: float = 1e-6,
    atol: float = 1e-9,
    first_step: float = 0.0,
    max_steps: int = 100_000,
    safety: float = 0.9,
    min_factor: float = 0.2,
    max_factor: float = 5.0,
) -> DopriTrajectory:
    r"""Error-controlled Dormand-Prince RK45 trajectory evaluated with JAX stage arithmetic.

    This is the opt-in JAX counterpart of
    :func:`~oscillatools.accel.diff_kuramoto_dopri.kuramoto_dopri_trajectory`. The adaptive
    accept/reject controller stays on the host so the returned trajectory keeps its natural variable
    length; each accepted or rejected step evaluates the seven Dormand-Prince stages as float64 JAX
    arrays on the selected device.

    Parameters
    ----------
    theta0 : numpy.ndarray
        One-dimensional array of ``N`` initial phases in radians.
    omega : numpy.ndarray
        One-dimensional array of ``N`` natural frequencies.
    coupling : numpy.ndarray
        Two-dimensional ``(N, N)`` coupling matrix ``K``.
    t_end : float
        The final integration time; must be strictly positive.
    rtol, atol : float, optional
        Relative and absolute error tolerances. Both must be strictly positive.
    first_step : float, optional
        Initial step size. If not positive, ``t_end / 100`` is used.
    max_steps : int, optional
        Maximum accepted-step budget.
    safety, min_factor, max_factor : float, optional
        Elementary step-controller safety and growth clamps.

    Returns
    -------
    DopriTrajectory
        Accepted times, phases and realised step sizes.

    Raises
    ------
    ValueError
        If shapes or integration parameters are invalid, or the budget is exhausted.
    ImportError
        If JAX is not installed.
    """
    phases, frequencies, matrix = _validate_dopri_state(theta0, omega, coupling)
    if t_end <= 0.0:
        raise ValueError(f"t_end must be strictly positive, got {t_end}")
    if rtol <= 0.0 or atol <= 0.0:
        raise ValueError(f"rtol and atol must be strictly positive, got {rtol} and {atol}")
    if max_steps < 1:
        raise ValueError(f"max_steps must be positive, got {max_steps}")
    if safety <= 0.0 or min_factor <= 0.0 or max_factor <= 0.0:
        raise ValueError("safety, min_factor and max_factor must be positive")
    if min_factor > max_factor:
        raise ValueError("min_factor must be less than or equal to max_factor")

    backend = _load_backend()
    jnp = backend.jnp
    current = jnp.asarray(phases)
    frequency_values = jnp.asarray(frequencies)
    matrix_values = jnp.asarray(matrix)
    step = float(first_step if first_step > 0.0 else t_end / 100.0)
    time = 0.0
    times: list[float] = [0.0]
    path: list[NDArray[np.float64]] = [phases.copy()]
    realised: list[float] = []

    while time < t_end - 1e-14 and len(realised) < max_steps:
        if time + step > t_end:
            step = t_end - time
        derivatives = _jax_dopri_stage_derivatives(
            current, frequency_values, matrix_values, step, jnp
        )
        stacked = jnp.stack(derivatives)
        proposal = current + step * jnp.tensordot(jnp.asarray(_DOPRI_B5), stacked, axes=1)
        error_vector = step * jnp.tensordot(jnp.asarray(_DOPRI_B5 - _DOPRI_B4), stacked, axes=1)
        scale = atol + rtol * jnp.maximum(jnp.abs(current), jnp.abs(proposal))
        error = float(jnp.sqrt(jnp.mean((error_vector / scale) ** 2)))
        if error <= 1.0:
            time += step
            current = proposal
            times.append(time)
            path.append(np.asarray(current, dtype=np.float64).copy())
            realised.append(step)
        factor = safety * (error + 1e-300) ** (-0.2)
        step *= min(max_factor, max(min_factor, factor))

    if time < t_end - 1e-9:
        raise ValueError(f"integration exceeded max_steps={max_steps} before reaching t_end")

    return DopriTrajectory(
        times=np.ascontiguousarray(times, dtype=np.float64),
        phases=np.ascontiguousarray(path, dtype=np.float64),
        steps=np.ascontiguousarray(realised, dtype=np.float64),
    )


def jax_networked_inertial_trajectory(
    theta0: NDArray[np.float64],
    velocities: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    mass: float,
    *,
    damping: float = 1.0,
    dt: float,
    n_steps: int,
) -> InertialTrajectory:
    r"""Fixed-step RK4 inertial networked-Kuramoto trajectory on the JAX tier.

    Advances ``m θ̈ + γ θ̇ = ω + F(θ)`` over the concatenated ``(θ, v)`` state, using the same
    networked-force RK4 arithmetic as
    :func:`~oscillatools.accel.networked_inertial.networked_inertial_trajectory`.

    Parameters
    ----------
    theta0 : numpy.ndarray
        One-dimensional array of ``N`` initial phases.
    velocities : numpy.ndarray
        One-dimensional array of ``N`` initial velocities.
    omega : numpy.ndarray
        One-dimensional array of ``N`` natural frequencies / power injections.
    coupling : numpy.ndarray
        Two-dimensional ``(N, N)`` coupling matrix ``K``.
    mass : float
        The inertia ``m``; must be strictly positive.
    damping : float, optional
        The damping ``γ``; must be non-negative.
    dt : float
        Fixed RK4 step size; must be strictly positive.
    n_steps : int
        Number of RK4 steps; must be positive.

    Returns
    -------
    InertialTrajectory
        The sampled phase and velocity trajectory.

    Raises
    ------
    ValueError
        If state shapes or range constraints are invalid.
    ImportError
        If JAX is not installed.
    """
    phases, speed, frequencies, matrix = _validate_inertial_state(
        theta0, velocities, omega, coupling
    )
    _validate_fixed_step_params(mass, damping, dt, n_steps)
    backend = _load_backend()
    jnp = backend.jnp
    path, velocity_history = backend.inertial_trajectory(
        jnp.asarray(phases),
        jnp.asarray(speed),
        jnp.asarray(frequencies),
        jnp.asarray(matrix),
        float(mass),
        float(damping),
        float(dt),
        int(n_steps),
    )
    return InertialTrajectory(
        times=_time_grid(dt, n_steps),
        phases=np.asarray(path, dtype=np.float64),
        velocities=np.asarray(velocity_history, dtype=np.float64),
        mass=float(mass),
        damping=float(damping),
    )


def jax_networked_symplectic_inertial_trajectory(
    theta0: NDArray[np.float64],
    velocities: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    mass: float,
    *,
    damping: float = 1.0,
    dt: float,
    n_steps: int,
) -> InertialTrajectory:
    r"""Velocity-Verlet inertial networked-Kuramoto trajectory on the JAX tier.

    Advances ``m θ̈ + γ θ̇ = ω + F(θ)`` by the same damped symplectic splitting as
    :func:`~oscillatools.accel.networked_symplectic_inertial.networked_symplectic_inertial_trajectory`.
    At ``damping = 0`` this is the velocity-Verlet Hamiltonian limit.

    Parameters
    ----------
    theta0 : numpy.ndarray
        One-dimensional array of ``N`` initial phases.
    velocities : numpy.ndarray
        One-dimensional array of ``N`` initial velocities.
    omega : numpy.ndarray
        One-dimensional array of ``N`` natural frequencies / power injections.
    coupling : numpy.ndarray
        Two-dimensional ``(N, N)`` coupling matrix ``K``.
    mass : float
        The inertia ``m``; must be strictly positive.
    damping : float, optional
        The damping ``γ``; must be non-negative.
    dt : float
        Fixed Verlet step size; must be strictly positive.
    n_steps : int
        Number of Verlet steps; must be positive.

    Returns
    -------
    InertialTrajectory
        The sampled phase and velocity trajectory.

    Raises
    ------
    ValueError
        If state shapes or range constraints are invalid.
    ImportError
        If JAX is not installed.
    """
    phases, speed, frequencies, matrix = _validate_symplectic_inertial_state(
        theta0, velocities, omega, coupling
    )
    _validate_fixed_step_params(mass, damping, dt, n_steps)
    backend = _load_backend()
    jnp = backend.jnp
    path, velocity_history = backend.symplectic_inertial_trajectory(
        jnp.asarray(phases),
        jnp.asarray(speed),
        jnp.asarray(frequencies),
        jnp.asarray(matrix),
        float(mass),
        float(damping),
        float(dt),
        int(n_steps),
    )
    return InertialTrajectory(
        times=_time_grid(dt, n_steps),
        phases=np.asarray(path, dtype=np.float64),
        velocities=np.asarray(velocity_history, dtype=np.float64),
        mass=float(mass),
        damping=float(damping),
    )


def jax_networked_noisy_trajectory(
    theta0: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    diffusion: float,
    *,
    dt: float,
    n_steps: int,
    seed: int,
    settle_steps: int | None = None,
) -> NoisyKuramotoRun:
    r"""Seeded Euler-Maruyama noisy networked-Kuramoto trajectory on the JAX tier.

    The standard-normal Wiener increments are drawn once with
    :class:`numpy.random.Generator`, matching
    :func:`~oscillatools.accel.networked_noisy.networked_noisy_trajectory`; the drift, noise update
    and order-parameter series are then evaluated by a JAX scan.

    Parameters
    ----------
    theta0 : numpy.ndarray
        One-dimensional array of ``N`` initial phases.
    omega : numpy.ndarray
        One-dimensional array of ``N`` natural frequencies.
    coupling : numpy.ndarray
        Two-dimensional ``(N, N)`` coupling matrix ``K``.
    diffusion : float
        The diffusion / noise intensity ``D``; must be non-negative.
    dt : float
        Euler-Maruyama step size; must be strictly positive.
    n_steps : int
        Number of stochastic steps; must be positive.
    seed : int
        Seed for the standard-normal increment stream.
    settle_steps : int, optional
        Trailing order-parameter window for the stationary estimate.

    Returns
    -------
    NoisyKuramotoRun
        Order-parameter series, terminal phases and settle-window statistics.

    Raises
    ------
    ValueError
        If state shapes or range constraints are invalid.
    ImportError
        If JAX is not installed.
    """
    theta, frequencies, matrix, count = _validate_noisy_state(theta0, omega, coupling)
    if diffusion < 0.0:
        raise ValueError(f"diffusion must be non-negative, got {diffusion}")
    _validate_fixed_step_params(None, 0.0, dt, n_steps)
    resolved_settle = settle_steps if settle_steps is not None else max(1, n_steps // 2)
    if not 1 <= resolved_settle <= n_steps:
        raise ValueError(f"settle_steps must be in [1, {n_steps}], got {resolved_settle}")
    noise: NDArray[np.float64] = np.random.default_rng(seed).standard_normal((n_steps, count))
    backend = _load_backend()
    jnp = backend.jnp
    series, terminal = backend.noisy_trajectory(
        jnp.asarray(theta),
        jnp.asarray(frequencies),
        jnp.asarray(matrix),
        float(diffusion),
        float(dt),
        jnp.asarray(noise),
    )
    order_series = np.asarray(series, dtype=np.float64)
    terminal_phases = np.asarray(terminal, dtype=np.float64)
    settle = order_series[n_steps - resolved_settle :]
    return NoisyKuramotoRun(
        order_parameter_series=order_series,
        terminal_phases=terminal_phases,
        mean_order_parameter=float(settle.mean()),
        order_parameter_std=float(settle.std()),
        diffusion=float(diffusion),
        settle_steps=resolved_settle,
    )


__all__ = [
    "jax_kuramoto_dopri_trajectory",
    "jax_kuramoto_euler_trajectory",
    "jax_networked_inertial_trajectory",
    "jax_networked_noisy_trajectory",
    "jax_networked_symplectic_inertial_trajectory",
]
