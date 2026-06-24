# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Adaptive Dormand–Prince RK45 Kuramoto integrator with a fixed-grid adjoint
r"""Adaptive Dormand–Prince (RK45) Kuramoto integrator with a fixed-grid discrete adjoint.

Integrates the networked phase flow :math:`\dot\theta = \omega + F(\theta)` with
:math:`F_j = \sum_k K_{jk}\sin(\theta_k - \theta_j)` by the Dormand–Prince embedded
Runge–Kutta pair: a fifth-order solution is propagated and the embedded fourth-order solution
gives a per-step error estimate that drives the step size. Each step is accepted only when the
scaled error :math:`\sqrt{\langle (e/(\text{atol}+\text{rtol}\,|\theta|))^2\rangle}` is at most
one, so the tolerance is honoured on every accepted step; the next step is rescaled by the
standard elementary controller :math:`h \leftarrow h\,\min(f_{\max}, \max(f_{\min}, s\,
\text{err}^{-1/5}))`.

The reverse mode is the *fixed-grid* adjoint: the error-controlled forward fixes the realised
step sequence, and the discrete adjoint backpropagates through exactly those steps — through the
seven Dormand–Prince stages of each — so the returned gradients are the exact transpose of the
realised forward map (gradient parity with a finite difference). It returns the gradients of a
terminal cotangent with respect to the initial phases, the natural frequencies and the coupling
matrix.

Unlike the fixed-step :mod:`~scpn_quantum_control.accel.diff_kuramoto_rk4`, the adaptive control
and the realised-grid adjoint are inherently sequential, so this is a pure-Python orchestration
layer — it adds no compute kernel (the engine stays at its current PyO3 surface). For a
Rust-accelerated fixed-step trajectory use
:func:`~scpn_quantum_control.accel.diff_kuramoto_rk4.kuramoto_rk4_trajectory`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

_VjpResult = tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]

# Dormand–Prince (DOPRI5) Butcher tableau. ``_A`` is lower-triangular row by row; ``_B5`` is the
# propagated fifth-order weight row and ``_ERROR`` the fifth-minus-fourth-order error weights.
_A: tuple[tuple[float, ...], ...] = (
    (),
    (1 / 5,),
    (3 / 40, 9 / 40),
    (44 / 45, -56 / 15, 32 / 9),
    (19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729),
    (9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656),
    (35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84),
)
_B5 = np.array([35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0.0])
_B4 = np.array([5179 / 57600, 0.0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40])
_ERROR = _B5 - _B4
_STAGE_COUNT = 7
_ERROR_EXPONENT = 0.2  # 1/(order + 1) with the embedded fourth-order estimate


def _force(phases: NDArray[np.float64], coupling: NDArray[np.float64]) -> NDArray[np.float64]:
    """Networked coupling force ``F_j = Σ_k K_jk sin(θ_k − θ_j)``."""
    difference = phases[None, :] - phases[:, None]
    return np.asarray((coupling * np.sin(difference)).sum(axis=1), dtype=np.float64)


def _jacobian(phases: NDArray[np.float64], coupling: NDArray[np.float64]) -> NDArray[np.float64]:
    """Jacobian ``∂F_j/∂θ_l`` of the networked force; the diagonal excludes the self term."""
    weighted = coupling * np.cos(phases[None, :] - phases[:, None])
    jacobian = weighted.copy()
    diagonal = weighted.sum(axis=1) - np.diag(weighted)
    np.fill_diagonal(jacobian, -diagonal)
    return np.asarray(jacobian, dtype=np.float64)


def _coupling_gradient_contribution(
    phases: NDArray[np.float64], cotangent: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Contribution ``∂L/∂K_jl = c_j sin(θ_l − θ_j)`` of one force cotangent."""
    return np.asarray(
        cotangent[:, None] * np.sin(phases[None, :] - phases[:, None]), dtype=np.float64
    )


def _validate_state(
    theta0: NDArray[np.float64], omega: NDArray[np.float64], coupling: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return contiguous ``(theta0, omega, coupling)`` after shape validation."""
    phases = np.ascontiguousarray(theta0, dtype=np.float64)
    frequencies = np.ascontiguousarray(omega, dtype=np.float64)
    matrix = np.ascontiguousarray(coupling, dtype=np.float64)
    if phases.ndim != 1 or phases.size == 0:
        raise ValueError("theta0 must be a non-empty one-dimensional array")
    if frequencies.shape != phases.shape:
        raise ValueError(f"omega must match theta0 shape {phases.shape}, got {frequencies.shape}")
    count = phases.size
    if matrix.shape != (count, count):
        raise ValueError(f"coupling must be a square matrix of order {count}, got {matrix.shape}")
    return phases, frequencies, matrix


def _stage_derivatives(
    phases: NDArray[np.float64],
    frequencies: NDArray[np.float64],
    coupling: NDArray[np.float64],
    step: float,
) -> NDArray[np.float64]:
    """Return the ``(7, N)`` Dormand–Prince stage derivatives at one step."""
    derivatives = np.empty((_STAGE_COUNT, phases.size), dtype=np.float64)
    for stage in range(_STAGE_COUNT):
        increment = phases.copy()
        for previous in range(stage):
            increment = increment + step * _A[stage][previous] * derivatives[previous]
        derivatives[stage] = frequencies + _force(increment, coupling)
    return derivatives


@dataclass(frozen=True)
class DopriTrajectory:
    """The realised adaptive Dormand–Prince trajectory.

    Attributes
    ----------
    times : numpy.ndarray
        The ``(M + 1,)`` accepted times, starting at ``0``.
    phases : numpy.ndarray
        The ``(M + 1, N)`` phases at the accepted times; row ``0`` is the initial state.
    steps : numpy.ndarray
        The ``(M,)`` realised step sizes — the fixed grid the adjoint reverses over.
    """

    times: NDArray[np.float64]
    phases: NDArray[np.float64]
    steps: NDArray[np.float64]

    @property
    def terminal_phases(self) -> NDArray[np.float64]:
        """The phases at the final accepted time."""
        return np.ascontiguousarray(self.phases[-1], dtype=np.float64)


def kuramoto_dopri_trajectory(
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
    r"""Error-controlled Dormand–Prince RK45 integration of the networked phase flow.

    Integrates :math:`\dot\theta = \omega + F(\theta)` from ``0`` to ``t_end`` with adaptive step
    sizes, accepting a step only when the scaled embedded-error estimate is at most one, so the
    requested tolerance is honoured on every accepted step.

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
        The relative and absolute error tolerances; must be strictly positive. Default ``1e-6``
        and ``1e-9``.
    first_step : float, optional
        The initial step size. If not positive, a conservative ``t_end / 100`` guess is used and
        the controller adapts from there. Defaults to ``0.0`` (auto).
    max_steps : int, optional
        The maximum number of attempted steps before raising. Defaults to ``100000``.
    safety, min_factor, max_factor : float, optional
        The elementary step-controller safety factor and its growth/shrink clamps. Defaults
        ``0.9``, ``0.2`` and ``5.0``.

    Returns
    -------
    DopriTrajectory
        The accepted times, phases and realised step sizes.

    Raises
    ------
    ValueError
        If the state shapes are inconsistent, ``t_end``/``rtol``/``atol`` are not positive, or the
        integration exceeds ``max_steps``.
    """
    phases, frequencies, matrix = _validate_state(theta0, omega, coupling)
    if t_end <= 0.0:
        raise ValueError(f"t_end must be strictly positive, got {t_end}")
    if rtol <= 0.0 or atol <= 0.0:
        raise ValueError(f"rtol and atol must be strictly positive, got {rtol} and {atol}")

    step = first_step if first_step > 0.0 else t_end / 100.0
    time = 0.0
    current = phases
    times = [0.0]
    path = [current.copy()]
    realised: list[float] = []

    attempts = 0
    while time < t_end - 1e-14:
        if attempts >= max_steps:
            raise ValueError(f"integration exceeded max_steps={max_steps} before reaching t_end")
        attempts += 1
        if time + step > t_end:
            step = t_end - time
        derivatives = _stage_derivatives(current, frequencies, matrix, step)
        proposal = current + step * (_B5 @ derivatives)
        error_vector = step * (_ERROR @ derivatives)
        scale = atol + rtol * np.maximum(np.abs(current), np.abs(proposal))
        error = float(np.sqrt(np.mean((error_vector / scale) ** 2)))
        if error <= 1.0:
            time += step
            current = proposal
            times.append(time)
            path.append(current.copy())
            realised.append(step)
        factor = safety * (error + 1e-300) ** (-_ERROR_EXPONENT)
        step = step * min(max_factor, max(min_factor, factor))

    return DopriTrajectory(
        times=np.ascontiguousarray(times, dtype=np.float64),
        phases=np.ascontiguousarray(path, dtype=np.float64),
        steps=np.ascontiguousarray(realised, dtype=np.float64),
    )


def _validate_vjp(
    phases: NDArray[np.float64],
    steps: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    cotangent: NDArray[np.float64],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Return contiguous adjoint inputs after validating their shapes."""
    path = np.ascontiguousarray(phases, dtype=np.float64)
    realised = np.ascontiguousarray(steps, dtype=np.float64)
    if path.ndim != 2 or path.shape[0] < 2:
        raise ValueError("phases must be a (M+1, N) trajectory with at least one step")
    count = path.shape[1]
    if realised.ndim != 1 or realised.size != path.shape[0] - 1:
        raise ValueError(
            f"steps must be one-dimensional of length {path.shape[0] - 1}, got shape {realised.shape}"
        )
    frequencies = np.ascontiguousarray(omega, dtype=np.float64)
    matrix = np.ascontiguousarray(coupling, dtype=np.float64)
    if frequencies.shape != (count,):
        raise ValueError(f"omega must have shape {(count,)}, got {frequencies.shape}")
    if matrix.shape != (count, count):
        raise ValueError(f"coupling must be a square matrix of order {count}, got {matrix.shape}")
    seed = np.ascontiguousarray(cotangent, dtype=np.float64)
    if seed.shape != (count,):
        raise ValueError(f"cotangent must have shape {(count,)}, got {seed.shape}")
    return path, realised, frequencies, matrix, seed


def kuramoto_dopri_vjp(
    phases: NDArray[np.float64],
    steps: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    cotangent: NDArray[np.float64],
) -> _VjpResult:
    r"""Fixed-grid discrete adjoint of the Dormand–Prince integration.

    Backpropagates a terminal cotangent through the realised step grid — through the seven
    Dormand–Prince stages of every step — to return the gradients of the loss with respect to the
    initial phases, the natural frequencies and the coupling matrix. Because the step grid is the
    one the forward error control realised, the result is the exact transpose of the realised
    forward map (gradient parity).

    Parameters
    ----------
    phases : numpy.ndarray
        The ``(M + 1, N)`` accepted trajectory from :func:`kuramoto_dopri_trajectory`.
    steps : numpy.ndarray
        The ``(M,)`` realised step sizes from the same trajectory.
    omega : numpy.ndarray
        The ``(N,)`` natural frequencies used in the forward integration.
    coupling : numpy.ndarray
        The ``(N, N)`` coupling matrix used in the forward integration.
    cotangent : numpy.ndarray
        The ``(N,)`` cotangent on the terminal phases (``∂L/∂θ_T``).

    Returns
    -------
    tuple of numpy.ndarray
        ``(∂L/∂θ₀, ∂L/∂ω, ∂L/∂K)`` with shapes ``(N,)``, ``(N,)`` and ``(N, N)``.

    Raises
    ------
    ValueError
        If the trajectory, steps, state or cotangent shapes are inconsistent.
    """
    path, realised, frequencies, matrix, seed = _validate_vjp(
        phases, steps, omega, coupling, cotangent
    )
    count = path.shape[1]
    adjoint = seed.copy()
    grad_omega = np.zeros(count, dtype=np.float64)
    grad_coupling = np.zeros((count, count), dtype=np.float64)

    for index in range(realised.size - 1, -1, -1):
        node = path[index]
        step = float(realised[index])
        derivatives = _stage_derivatives(node, frequencies, matrix, step)
        stage_points = []
        for stage in range(_STAGE_COUNT):
            point = node.copy()
            for previous in range(stage):
                point = point + step * _A[stage][previous] * derivatives[previous]
            stage_points.append(point)

        stage_cotangents = [step * _B5[stage] * adjoint for stage in range(_STAGE_COUNT)]
        node_adjoint = adjoint.copy()
        for stage in range(_STAGE_COUNT - 1, -1, -1):
            point = stage_points[stage]
            point_adjoint = _jacobian(point, matrix).T @ stage_cotangents[stage]
            node_adjoint = node_adjoint + point_adjoint
            for previous in range(stage):
                stage_cotangents[previous] = (
                    stage_cotangents[previous] + step * _A[stage][previous] * point_adjoint
                )
            grad_omega += stage_cotangents[stage]
            grad_coupling += _coupling_gradient_contribution(point, stage_cotangents[stage])
        adjoint = node_adjoint

    return (
        np.ascontiguousarray(adjoint, dtype=np.float64),
        np.ascontiguousarray(grad_omega, dtype=np.float64),
        np.ascontiguousarray(grad_coupling, dtype=np.float64),
    )


__all__ = [
    "DopriTrajectory",
    "kuramoto_dopri_trajectory",
    "kuramoto_dopri_vjp",
]
