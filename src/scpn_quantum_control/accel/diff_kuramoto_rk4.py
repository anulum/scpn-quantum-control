# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Differentiable networked-Kuramoto RK4 integrator and adjoint
"""Differentiable networked-Kuramoto fourth-order Runge–Kutta integrator and its adjoint.

Forward classical RK4 integration of the networked Kuramoto dynamics ``θ̇ = ω + F(θ)`` with
``F_j(θ) = Σ_k K_jk sin(θ_k − θ_j)``, recording the full phase trajectory. Each step evaluates
the four stages ``k1 = f(θ_n)``, ``k2 = f(θ_n + ½dt·k1)``, ``k3 = f(θ_n + ½dt·k2)``,
``k4 = f(θ_n + dt·k3)`` (``f = ω + F``) and advances by ``θ_{n+1} = θ_n + (dt/6)(k1 + 2k2 +
2k3 + k4)`` — fourth-order accurate, where the Euler scheme is first-order. The reverse-mode
adjoint backpropagates a terminal cotangent through the four stages, recomputing the stage
states from the stored trajectory, and returns the gradients of a scalar objective with respect
to the initial phases, the natural frequencies and the coupling matrix. It is the higher-order
companion of :mod:`~scpn_quantum_control.accel.diff_kuramoto_euler` for differentiable
simulation, optimal coupling design, pinning control and learning ``K`` from data.

Multi-language (Rust → Julia → Python floor) implementations dispatched through
:class:`~scpn_quantum_control.accel.dispatcher.MultiLangDispatcher`.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from . import dispatcher
from .dispatcher import MultiLangDispatcher, register_dispatcher

_VjpResult = tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]


def _validate_forward(
    theta0: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    n_steps: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return contiguous ``(theta0, omega, coupling)`` after shape validation.

    Raises
    ------
    ValueError
        If ``omega`` or ``coupling`` does not match ``theta0``'s order, or ``n_steps`` < 0.
    """
    phases = np.ascontiguousarray(theta0, dtype=np.float64)
    frequencies = np.ascontiguousarray(omega, dtype=np.float64)
    matrix = np.ascontiguousarray(coupling, dtype=np.float64)
    count = phases.size
    if frequencies.shape != (count,):
        raise ValueError(f"omega must have shape ({count},), got {frequencies.shape}")
    if matrix.shape != (count, count):
        raise ValueError(f"coupling must have shape ({count}, {count}), got {matrix.shape}")
    if n_steps < 0:
        raise ValueError(f"n_steps must be non-negative, got {n_steps}")
    return phases, frequencies, matrix


def _force(phases: NDArray[np.float64], coupling: NDArray[np.float64]) -> NDArray[np.float64]:
    """Networked coupling force ``F_j = Σ_k K_jk sin(θ_k − θ_j)``."""
    difference = phases[None, :] - phases[:, None]
    return np.asarray((coupling * np.sin(difference)).sum(axis=1), dtype=np.float64)


def _rust_kuramoto_rk4_trajectory(
    theta0: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    dt: float,
    n_steps: int,
) -> NDArray[np.float64]:
    phases, frequencies, matrix = _validate_forward(theta0, omega, coupling, n_steps)
    engine = dispatcher.optional_rust_engine()
    if engine is None:
        raise ModuleNotFoundError("scpn_quantum_engine")
    rust_forward = getattr(engine, "kuramoto_rk4_trajectory", None)
    if not callable(rust_forward):
        raise ImportError("scpn_quantum_engine.kuramoto_rk4_trajectory is unavailable")

    return np.asarray(
        rust_forward(phases, frequencies, matrix, float(dt), int(n_steps)), dtype=np.float64
    )


def _julia_kuramoto_rk4_trajectory(
    theta0: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    dt: float,
    n_steps: int,
) -> NDArray[np.float64]:
    phases, frequencies, matrix = _validate_forward(theta0, omega, coupling, n_steps)
    from .julia import kuramoto_rk4_trajectory as julia_forward

    return julia_forward(phases, frequencies, matrix, float(dt), int(n_steps))


def _python_kuramoto_rk4_trajectory(
    theta0: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    dt: float,
    n_steps: int,
) -> NDArray[np.float64]:
    # Correctness floor — classical RK4 of θ̇ = ω + F(θ). Returns the (n_steps + 1, N) phase
    # trajectory including the initial state.
    phases, frequencies, matrix = _validate_forward(theta0, omega, coupling, n_steps)
    count = phases.size
    trajectory = np.zeros((n_steps + 1, count), dtype=np.float64)
    trajectory[0] = phases
    current = phases
    half = 0.5 * dt
    for step in range(n_steps):
        k1 = frequencies + _force(current, matrix)
        k2 = frequencies + _force(current + half * k1, matrix)
        k3 = frequencies + _force(current + half * k2, matrix)
        k4 = frequencies + _force(current + dt * k3, matrix)
        current = current + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        trajectory[step + 1] = current
    return np.ascontiguousarray(trajectory, dtype=np.float64)


_KURAMOTO_RK4_TRAJECTORY_CHAIN: list[
    tuple[
        str,
        Callable[
            [
                NDArray[np.float64],
                NDArray[np.float64],
                NDArray[np.float64],
                float,
                int,
            ],
            NDArray[np.float64],
        ],
    ]
] = [
    ("rust", _rust_kuramoto_rk4_trajectory),
    ("julia", _julia_kuramoto_rk4_trajectory),
    ("python", _python_kuramoto_rk4_trajectory),
]


def _validate_vjp(
    trajectory: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    cotangent: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return contiguous ``(trajectory, omega, coupling, cotangent)`` after shape validation.

    Raises
    ------
    ValueError
        If ``trajectory`` is not two-dimensional, or ``omega``/``coupling``/``cotangent`` do not
        match its oscillator count.
    """
    path = np.ascontiguousarray(trajectory, dtype=np.float64)
    frequencies = np.ascontiguousarray(omega, dtype=np.float64)
    matrix = np.ascontiguousarray(coupling, dtype=np.float64)
    seed = np.ascontiguousarray(cotangent, dtype=np.float64)
    if path.ndim != 2:
        raise ValueError(f"trajectory must be two-dimensional, got shape {path.shape}")
    count = path.shape[1]
    if frequencies.shape != (count,):
        raise ValueError(f"omega must have shape ({count},), got {frequencies.shape}")
    if matrix.shape != (count, count):
        raise ValueError(f"coupling must have shape ({count}, {count}), got {matrix.shape}")
    if seed.shape != (count,):
        raise ValueError(f"cotangent must have shape ({count},), got {seed.shape}")
    return path, frequencies, matrix, seed


def _jacobian(phases: NDArray[np.float64], coupling: NDArray[np.float64]) -> NDArray[np.float64]:
    """Networked stability Jacobian ``J_jl = K_jl cos(θ_l − θ_j)`` (zero-row-sum diagonal)."""
    difference = phases[None, :] - phases[:, None]
    jacobian = np.asarray(coupling * np.cos(difference), dtype=np.float64)
    np.fill_diagonal(jacobian, 0.0)
    np.fill_diagonal(jacobian, -jacobian.sum(axis=1))
    return jacobian


def _coupling_gradient_contribution(
    stage: NDArray[np.float64], stage_cotangent: NDArray[np.float64]
) -> NDArray[np.float64]:
    """``∂L/∂K_pq`` contribution of one stage: ``stage_cotangent_p sin(s_q − s_p)``."""
    difference = stage[None, :] - stage[:, None]
    return np.asarray(stage_cotangent[:, None] * np.sin(difference), dtype=np.float64)


def _rust_kuramoto_rk4_vjp(
    trajectory: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    dt: float,
    cotangent: NDArray[np.float64],
) -> _VjpResult:
    path, frequencies, matrix, seed = _validate_vjp(trajectory, omega, coupling, cotangent)
    engine = dispatcher.optional_rust_engine()
    if engine is None:
        raise ModuleNotFoundError("scpn_quantum_engine")
    rust_vjp = getattr(engine, "kuramoto_rk4_vjp", None)
    if not callable(rust_vjp):
        raise ImportError("scpn_quantum_engine.kuramoto_rk4_vjp is unavailable")

    grad_theta0, grad_omega, grad_coupling = rust_vjp(path, frequencies, matrix, float(dt), seed)
    return (
        np.asarray(grad_theta0, dtype=np.float64),
        np.asarray(grad_omega, dtype=np.float64),
        np.asarray(grad_coupling, dtype=np.float64),
    )


def _julia_kuramoto_rk4_vjp(
    trajectory: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    dt: float,
    cotangent: NDArray[np.float64],
) -> _VjpResult:
    path, frequencies, matrix, seed = _validate_vjp(trajectory, omega, coupling, cotangent)
    from .julia import kuramoto_rk4_vjp as julia_vjp

    grad_theta0, grad_omega, grad_coupling = julia_vjp(path, frequencies, matrix, float(dt), seed)
    return (
        np.asarray(grad_theta0, dtype=np.float64),
        np.asarray(grad_omega, dtype=np.float64),
        np.asarray(grad_coupling, dtype=np.float64),
    )


def _python_kuramoto_rk4_vjp(
    trajectory: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    dt: float,
    cotangent: NDArray[np.float64],
) -> _VjpResult:
    # Correctness floor — reverse-mode adjoint through the four RK4 stages. Each step's stages
    # are recomputed from the stored θ_n; the terminal cotangent is backpropagated through the
    # combination θ_{n+1} = θ_n + (dt/6)(k1 + 2k2 + 2k3 + k4) and the stage chain
    # s2 = θ_n + ½dt·k1, s3 = θ_n + ½dt·k2, s4 = θ_n + dt·k3, accumulating ∂L/∂ω from the stage
    # cotangents and ∂L/∂K from each stage's ∂F/∂K. Returns (∂L/∂θ₀, ∂L/∂ω, ∂L/∂K).
    path, frequencies, matrix, seed = _validate_vjp(trajectory, omega, coupling, cotangent)
    n_steps = path.shape[0] - 1
    count = path.shape[1]
    half = 0.5 * dt
    adjoint = seed.copy()
    grad_omega = np.zeros(count, dtype=np.float64)
    grad_coupling = np.zeros((count, count), dtype=np.float64)
    for step in range(n_steps - 1, -1, -1):
        phases = path[step]
        k1 = frequencies + _force(phases, matrix)
        stage2 = phases + half * k1
        k2 = frequencies + _force(stage2, matrix)
        stage3 = phases + half * k2
        k3 = frequencies + _force(stage3, matrix)
        stage4 = phases + dt * k3
        stage_cotangents = [
            (dt / 6.0) * adjoint,
            (dt / 3.0) * adjoint,
            (dt / 3.0) * adjoint,
            (dt / 6.0) * adjoint,
        ]
        next_adjoint = adjoint.copy()
        backward4 = _jacobian(stage4, matrix).T @ stage_cotangents[3]
        next_adjoint = next_adjoint + backward4
        stage_cotangents[2] = stage_cotangents[2] + dt * backward4
        backward3 = _jacobian(stage3, matrix).T @ stage_cotangents[2]
        next_adjoint = next_adjoint + backward3
        stage_cotangents[1] = stage_cotangents[1] + half * backward3
        backward2 = _jacobian(stage2, matrix).T @ stage_cotangents[1]
        next_adjoint = next_adjoint + backward2
        stage_cotangents[0] = stage_cotangents[0] + half * backward2
        backward1 = _jacobian(phases, matrix).T @ stage_cotangents[0]
        next_adjoint = next_adjoint + backward1
        for stage, stage_cotangent in zip(
            (phases, stage2, stage3, stage4), stage_cotangents, strict=True
        ):
            grad_omega += stage_cotangent
            grad_coupling += _coupling_gradient_contribution(stage, stage_cotangent)
        adjoint = next_adjoint
    return (
        np.ascontiguousarray(adjoint, dtype=np.float64),
        np.ascontiguousarray(grad_omega, dtype=np.float64),
        np.ascontiguousarray(grad_coupling, dtype=np.float64),
    )


# The VJP chain mirrors the forward chain (Rust → Julia → Python floor). Their micro-benchmark
# (on a dense random coupling matrix over a fixed step budget) is recorded in
# ``docs/benchmarks/diff_kuramoto_rk4_tiers.json``; rerun
# ``python scripts/bench_diff_kuramoto_rk4_tiers.py`` when these chains are edited.
_KURAMOTO_RK4_VJP_CHAIN: list[
    tuple[
        str,
        Callable[
            [
                NDArray[np.float64],
                NDArray[np.float64],
                NDArray[np.float64],
                float,
                NDArray[np.float64],
            ],
            _VjpResult,
        ],
    ]
] = [
    ("rust", _rust_kuramoto_rk4_vjp),
    ("julia", _julia_kuramoto_rk4_vjp),
    ("python", _python_kuramoto_rk4_vjp),
]


_kuramoto_rk4_trajectory_dispatcher = MultiLangDispatcher(_KURAMOTO_RK4_TRAJECTORY_CHAIN)
_kuramoto_rk4_vjp_dispatcher = MultiLangDispatcher(_KURAMOTO_RK4_VJP_CHAIN)


def kuramoto_rk4_trajectory(
    theta0: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    dt: float,
    n_steps: int,
) -> NDArray[np.float64]:
    r"""Forward networked-Kuramoto RK4 trajectory with multi-language dispatch.

    Integrates :math:`\dot\theta = \omega + F(\theta)` with
    :math:`F_j = \sum_k K_{jk}\sin(\theta_k - \theta_j)` by classical fourth-order Runge–Kutta
    for ``n_steps`` steps, returning the full phase trajectory (the initial state plus every
    step). Fourth-order accurate where
    :func:`~scpn_quantum_control.accel.diff_kuramoto_euler.kuramoto_euler_trajectory` is
    first-order; the trajectory is the forward record :func:`kuramoto_rk4_vjp` consumes.

    Parameters
    ----------
    theta0 : numpy.ndarray
        One-dimensional array of ``N`` initial phases in radians.
    omega : numpy.ndarray
        One-dimensional array of ``N`` natural frequencies.
    coupling : numpy.ndarray
        Two-dimensional ``(N, N)`` coupling matrix ``K``.
    dt : float
        The RK4 step size.
    n_steps : int
        The number of integration steps (non-negative).

    Returns
    -------
    numpy.ndarray
        Two-dimensional ``(n_steps + 1, N)`` float64 trajectory; row 0 is ``theta0``.

    Raises
    ------
    ValueError
        If ``omega``/``coupling`` shapes do not match ``theta0`` or ``n_steps`` is negative.

    Notes
    -----
    Chain (measured fastest first): Rust → Julia → Python floor. The served tier is
    recorded on :func:`last_kuramoto_rk4_trajectory_tier_used`.
    """
    return np.asarray(
        _kuramoto_rk4_trajectory_dispatcher(theta0, omega, coupling, dt, n_steps),
        dtype=np.float64,
    )


def kuramoto_rk4_vjp(
    trajectory: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    dt: float,
    cotangent: NDArray[np.float64],
) -> _VjpResult:
    r"""Reverse-mode adjoint of the networked-Kuramoto RK4 integrator.

    Given the forward ``trajectory`` from :func:`kuramoto_rk4_trajectory` and a cotangent
    :math:`\lambda_N = \partial L/\partial \theta_N`, backpropagates through each step's four
    Runge–Kutta stages and returns the gradients of the scalar objective ``L`` with respect to
    the initial phases, the natural frequencies and the coupling matrix.

    Parameters
    ----------
    trajectory : numpy.ndarray
        Two-dimensional ``(n_steps + 1, N)`` forward trajectory.
    omega : numpy.ndarray
        One-dimensional ``(N,)`` natural frequencies used in the forward pass.
    coupling : numpy.ndarray
        Two-dimensional ``(N, N)`` coupling matrix ``K`` used in the forward pass.
    dt : float
        The RK4 step size used in the forward pass.
    cotangent : numpy.ndarray
        One-dimensional ``(N,)`` cotangent on the final phase, ``∂L/∂θ_N``.

    Returns
    -------
    tuple of numpy.ndarray
        ``(grad_theta0, grad_omega, grad_coupling)`` with shapes ``(N,)``, ``(N,)`` and
        ``(N, N)``.

    Raises
    ------
    ValueError
        If ``trajectory`` is not two-dimensional or ``omega``/``coupling``/``cotangent`` shapes
        are inconsistent with it.

    Notes
    -----
    Chain (measured fastest first): Rust → Julia → Python floor. The served tier is
    recorded on :func:`last_kuramoto_rk4_vjp_tier_used`.
    """
    grad_theta0, grad_omega, grad_coupling = _kuramoto_rk4_vjp_dispatcher(
        trajectory, omega, coupling, dt, cotangent
    )
    return (
        np.asarray(grad_theta0, dtype=np.float64),
        np.asarray(grad_omega, dtype=np.float64),
        np.asarray(grad_coupling, dtype=np.float64),
    )


def last_kuramoto_rk4_trajectory_tier_used() -> str | None:
    """Return the tier that served the most recent ``kuramoto_rk4_trajectory``."""
    return _kuramoto_rk4_trajectory_dispatcher.last_tier


def last_kuramoto_rk4_vjp_tier_used() -> str | None:
    """Return the tier that served the most recent ``kuramoto_rk4_vjp``."""
    return _kuramoto_rk4_vjp_dispatcher.last_tier


# Register the dispatchers for name-keyed ``dispatch`` lookups.
register_dispatcher("kuramoto_rk4_trajectory", _kuramoto_rk4_trajectory_dispatcher)
register_dispatcher("kuramoto_rk4_vjp", _kuramoto_rk4_vjp_dispatcher)


__all__ = [
    "kuramoto_rk4_trajectory",
    "kuramoto_rk4_vjp",
    "last_kuramoto_rk4_trajectory_tier_used",
    "last_kuramoto_rk4_vjp_tier_used",
]
