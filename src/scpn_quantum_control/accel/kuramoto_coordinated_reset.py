# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Coordinated-reset desynchronisation control (anti-control)
r"""Coordinated-reset desynchronisation control of the networked Kuramoto model.

Every control routine in the toolkit so far drives a population *towards* synchrony. Coordinated
reset (Tass) is the opposite — *anti-control*: it breaks a pathologically synchronised state. The
population is partitioned into ``M`` stimulation sites; over each stimulation cycle the sites are
reset one at a time, each to its own phase ``φ_m = 2π m / M`` spread around the circle, so the
sub-populations are driven apart and the Kuramoto order parameter ``r`` collapses. Between resets
the networked Kuramoto coupling pulls the phases back together, so the reset is repeated to hold
the desynchronised state.

On a *plastic* (Hebbian) substrate the effect outlasts the stimulation: while the phases are held
apart, the co-evolving coupling relaxes towards ``K_{ij} = cos(θ_j − θ_i)`` — a desynchronising
connectivity — so when stimulation stops the rewired network no longer re-synchronises. This is
the carryover / after-effect of coordinated reset, reproduced here on the adaptive integrator's
Hebbian rule.

A desynchronising objective (the terminal order parameter, to be *minimised*) is differentiable
through the controlled flow: the forward-mode sensitivity of the static-coupling protocol gives
``∂r/∂{θ_0, ω, K}``, with the reset operation correctly zeroing the sensitivity of the reset
oscillators (their state is overwritten, erasing the initial-condition dependence — the
mechanism by which coordinated reset works).

.. warning::

   Coordinated-reset desynchronisation is **research-grade, not clinically proven**. The
   modelling here is an idealised phase-reset protocol; the clinical evidence for coordinated-reset
   deep-brain / vibrotactile stimulation is from non-human primates and small open-label pilots,
   with no powered randomised controlled trial, and a first-in-human vibrotactile study reported no
   significant blinded UPDRS-III change. This module is a dynamical-systems tool, not a therapy.

This is an analysis/control layer over the synchronisation dynamics; it composes the polyglot
networked force and Jacobian and the Hebbian plasticity rule, and adds no compute kernel.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .kuramoto_adaptive import hebbian_plasticity_rate
from .networked_kuramoto import networked_kuramoto_force, networked_kuramoto_jacobian
from .order_parameter_observables import order_parameter

#: A terminal objective ``L(θ_N)`` on the final phases of a coordinated-reset run.
CoordinatedResetObjective = Callable[[NDArray[np.float64]], float]

#: The gradient of a :data:`CoordinatedResetObjective`, returning ``∂L/∂θ_N``.
CoordinatedResetObjectiveGrad = Callable[[NDArray[np.float64]], NDArray[np.float64]]


def coordinated_reset_sites(n_oscillators: int, n_sites: int) -> NDArray[np.int_]:
    r"""Assign oscillators to ``n_sites`` stimulation sites round-robin (``j → j mod M``).

    Parameters
    ----------
    n_oscillators : int
        The number of oscillators ``N`` (``≥ n_sites``).
    n_sites : int
        The number of stimulation sites ``M`` (``≥ 2``).

    Returns
    -------
    numpy.ndarray
        The integer site index of each oscillator (length ``N``).

    Raises
    ------
    ValueError
        If ``n_sites < 2`` or ``n_oscillators < n_sites``.
    """
    if n_sites < 2:
        raise ValueError(f"n_sites must be at least 2, got {n_sites}")
    if n_oscillators < n_sites:
        raise ValueError(f"n_oscillators ({n_oscillators}) must be at least n_sites ({n_sites})")
    return np.arange(n_oscillators, dtype=np.int_) % n_sites


def coordinated_reset_phases(n_sites: int) -> NDArray[np.float64]:
    r"""Return the spread reset target phases ``φ_m = 2π m / M`` for ``M`` sites."""
    return np.asarray(2.0 * np.pi * np.arange(n_sites) / n_sites, dtype=np.float64)


@dataclass(frozen=True)
class CoordinatedResetTrajectory:
    """A coordinated-reset run of the networked Kuramoto model.

    Attributes
    ----------
    order_parameter_series : numpy.ndarray
        The Kuramoto order parameter ``r`` sampled after every RK4 step (stimulation then free
        evolution).
    terminal_phases : numpy.ndarray
        The final phase vector.
    terminal_coupling : numpy.ndarray
        The final coupling matrix (rewired when ``plasticity_rate > 0``, else the input coupling).
    site_assignment : numpy.ndarray
        The oscillator-to-site assignment used.
    stimulation_steps : int
        The number of RK4 steps performed during stimulation (the rest are free evolution).
    """

    order_parameter_series: NDArray[np.float64]
    terminal_phases: NDArray[np.float64]
    terminal_coupling: NDArray[np.float64]
    site_assignment: NDArray[np.int_]
    stimulation_steps: int

    @property
    def terminal_order_parameter(self) -> float:
        """The final Kuramoto order parameter."""
        return float(self.order_parameter_series[-1])


def _validate(
    phases: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    n_sites: int,
    dt: float,
    steps_per_pulse: int,
    n_cycles: int,
    plasticity_rate: float,
    free_cycles: int,
) -> int:
    """Validate the coordinated-reset problem and return the oscillator count."""
    if phases.ndim != 1 or phases.size < 1:
        raise ValueError("initial_phases must be a non-empty one-dimensional array")
    count = int(phases.size)
    if omega.shape != (count,):
        raise ValueError(f"omega must have shape {(count,)}, got {omega.shape}")
    if coupling.shape != (count, count):
        raise ValueError(f"coupling must have shape {(count, count)}, got {coupling.shape}")
    if n_sites < 2 or n_sites > count:
        raise ValueError(f"n_sites must satisfy 2 <= n_sites <= {count}, got {n_sites}")
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    if steps_per_pulse < 1:
        raise ValueError(f"steps_per_pulse must be positive, got {steps_per_pulse}")
    if n_cycles < 1:
        raise ValueError(f"n_cycles must be positive, got {n_cycles}")
    if plasticity_rate < 0.0:
        raise ValueError(f"plasticity_rate must be non-negative, got {plasticity_rate}")
    if free_cycles < 0:
        raise ValueError(f"free_cycles must be non-negative, got {free_cycles}")
    return count


def _kuramoto_step(
    phases: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    dt: float,
) -> NDArray[np.float64]:
    """Advance the networked Kuramoto phases by one RK4 step."""

    def field(theta: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(omega + networked_kuramoto_force(theta, coupling), dtype=np.float64)

    k1 = field(phases)
    k2 = field(phases + 0.5 * dt * k1)
    k3 = field(phases + 0.5 * dt * k2)
    k4 = field(phases + dt * k3)
    return np.asarray(phases + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4), dtype=np.float64)


def integrate_coordinated_reset(
    initial_phases: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    *,
    n_sites: int,
    dt: float,
    steps_per_pulse: int,
    n_cycles: int,
    plasticity_rate: float = 0.0,
    free_cycles: int = 0,
    free_perturbation: float = 0.0,
    seed: int = 0,
) -> CoordinatedResetTrajectory:
    r"""Integrate a coordinated-reset desynchronisation protocol on the networked Kuramoto model.

    Over each of ``n_cycles`` stimulation cycles every one of the ``n_sites`` sites is reset, one at
    a time, to its spread phase ``φ_m = 2π m / n_sites`` and the networked Kuramoto flow is then
    advanced ``steps_per_pulse`` RK4 steps. When ``plasticity_rate > 0`` the coupling co-evolves by
    the Hebbian rule ``K̇ = ε(cos(θ_j − θ_i) − K)`` throughout, so a desynchronising connectivity is
    learned. After stimulation the system evolves freely for ``free_cycles`` cycles' worth of steps
    (no resets), exposing the carryover: with a rewired plastic coupling the desynchronised state
    persists, whereas a static synchronising coupling re-synchronises.

    Parameters
    ----------
    initial_phases : numpy.ndarray
        The initial phases ``θ(0)`` (length ``N``).
    omega : numpy.ndarray
        The natural frequencies ``ω`` (length ``N``).
    coupling : numpy.ndarray
        The ``(N, N)`` networked coupling matrix ``K``.
    n_sites : int
        The number of stimulation sites ``M`` (``2 ≤ M ≤ N``).
    dt : float
        The RK4 step (``> 0``).
    steps_per_pulse : int
        The number of RK4 steps between consecutive site resets (``≥ 1``).
    n_cycles : int
        The number of stimulation cycles (``≥ 1``).
    plasticity_rate : float, optional
        The Hebbian plasticity rate ``ε`` (``≥ 0``); ``0`` keeps the coupling static.
    free_cycles : int, optional
        The number of post-stimulation free-evolution cycles (``≥ 0``).
    free_perturbation : float, optional
        The standard deviation of a one-off Gaussian phase kick applied at the start of free
        evolution (``≥ 0``); probes the basin stability of the desynchronised state. A static
        synchronising coupling re-synchronises after the kick; a rewired plastic coupling does not.
    seed : int, optional
        The seed of the free-evolution perturbation generator.

    Returns
    -------
    CoordinatedResetTrajectory
        The order-parameter series, terminal phases and (rewired) coupling.

    Raises
    ------
    ValueError
        If any argument falls outside its documented bound.
    """
    phases = np.ascontiguousarray(initial_phases, dtype=np.float64)
    frequencies = np.ascontiguousarray(omega, dtype=np.float64)
    coupling_matrix = np.array(coupling, dtype=np.float64)
    _validate(
        phases,
        frequencies,
        coupling_matrix,
        n_sites,
        dt,
        steps_per_pulse,
        n_cycles,
        plasticity_rate,
        free_cycles,
    )
    if free_perturbation < 0.0:
        raise ValueError(f"free_perturbation must be non-negative, got {free_perturbation}")

    site_assignment = coordinated_reset_sites(phases.size, n_sites)
    targets = coordinated_reset_phases(n_sites)
    series: list[float] = [float(order_parameter(phases))]

    def advance() -> None:
        nonlocal phases, coupling_matrix
        if plasticity_rate > 0.0:
            rate = hebbian_plasticity_rate(
                phases, coupling_matrix, plasticity_rate=plasticity_rate
            )
            coupling_matrix = coupling_matrix + dt * rate
        phases = _kuramoto_step(phases, frequencies, coupling_matrix, dt)
        series.append(float(order_parameter(phases)))

    for _ in range(n_cycles):
        for site_index in range(n_sites):
            phases = phases.copy()
            phases[site_assignment == site_index] = targets[site_index]
            for _ in range(steps_per_pulse):
                advance()
    stimulation_steps = n_cycles * n_sites * steps_per_pulse
    if free_perturbation > 0.0:
        generator = np.random.default_rng(seed)
        phases = phases + free_perturbation * generator.standard_normal(phases.size)
    for _ in range(free_cycles * n_sites * steps_per_pulse):
        advance()

    return CoordinatedResetTrajectory(
        order_parameter_series=np.asarray(series, dtype=np.float64),
        terminal_phases=phases,
        terminal_coupling=coupling_matrix,
        site_assignment=site_assignment,
        stimulation_steps=stimulation_steps,
    )


@dataclass(frozen=True)
class CoordinatedResetGradients:
    """Gradients of a terminal desync objective through the static coordinated-reset flow.

    Attributes
    ----------
    initial_phases : numpy.ndarray
        ``∂L/∂θ_0`` (length ``N``); zero for oscillators whose last action was a reset.
    omega : numpy.ndarray
        ``∂L/∂ω`` (length ``N``).
    coupling : numpy.ndarray
        ``∂L/∂K`` (shape ``(N, N)``).
    """

    initial_phases: NDArray[np.float64]
    omega: NDArray[np.float64]
    coupling: NDArray[np.float64]


def _sensitivity_step(
    phases: NDArray[np.float64],
    sensitivity: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    dt: float,
    count: int,
    omega_start: int,
    coupling_start: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Advance phases and their forward-mode sensitivity through one networked RK4 step."""

    def field(theta: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(omega + networked_kuramoto_force(theta, coupling), dtype=np.float64)

    def tangent(theta: NDArray[np.float64], tan: NDArray[np.float64]) -> NDArray[np.float64]:
        jacobian = networked_kuramoto_jacobian(theta, coupling)
        injection = np.zeros((count, sensitivity.shape[1]), dtype=np.float64)
        injection[:, omega_start : omega_start + count] = np.eye(count)
        phase_delta = theta[np.newaxis, :] - theta[:, np.newaxis]
        sin_delta = np.sin(phase_delta)
        for p in range(count):
            base = coupling_start + p * count
            injection[p, base : base + count] = sin_delta[p]
        return jacobian @ tan + injection

    k1 = field(phases)
    s1 = tangent(phases, sensitivity)
    k2 = field(phases + 0.5 * dt * k1)
    s2 = tangent(phases + 0.5 * dt * k1, sensitivity + 0.5 * dt * s1)
    k3 = field(phases + 0.5 * dt * k2)
    s3 = tangent(phases + 0.5 * dt * k2, sensitivity + 0.5 * dt * s2)
    k4 = field(phases + dt * k3)
    s4 = tangent(phases + dt * k3, sensitivity + dt * s3)
    new_phases = phases + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    new_sensitivity = sensitivity + (dt / 6.0) * (s1 + 2.0 * s2 + 2.0 * s3 + s4)
    return (
        np.asarray(new_phases, dtype=np.float64),
        np.asarray(new_sensitivity, dtype=np.float64),
    )


def coordinated_reset_terminal_value_and_grad(
    initial_phases: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
    *,
    n_sites: int,
    dt: float,
    steps_per_pulse: int,
    n_cycles: int,
    objective: CoordinatedResetObjective,
    objective_grad: CoordinatedResetObjectiveGrad,
) -> tuple[float, CoordinatedResetGradients]:
    r"""Differentiate a terminal desync objective through the static coordinated-reset flow.

    Evaluates ``L(θ_N)`` (the desynchronising objective, e.g. the terminal order parameter to be
    minimised) for the static-coupling protocol and returns its gradients with respect to
    ``θ_0, ω, K`` by forward-mode sensitivity. Each site reset overwrites its oscillators' phases
    with a constant, so the reset zeroes their sensitivity rows — the controlled flow's exact
    derivative.

    Parameters
    ----------
    initial_phases, omega : numpy.ndarray
        The initial phases and natural frequencies (length ``N``).
    coupling : numpy.ndarray
        The ``(N, N)`` networked coupling matrix ``K`` (static).
    n_sites : int
        The number of stimulation sites ``M`` (``2 ≤ M ≤ N``).
    dt : float
        The RK4 step (``> 0``).
    steps_per_pulse : int
        The number of RK4 steps between resets (``≥ 1``).
    n_cycles : int
        The number of stimulation cycles (``≥ 1``).
    objective : callable
        The terminal objective ``L(θ_N) → float``.
    objective_grad : callable
        Its gradient ``θ_N → ∂L/∂θ_N`` (length ``N``).

    Returns
    -------
    tuple
        ``(value, CoordinatedResetGradients)``.

    Raises
    ------
    ValueError
        If any argument falls outside its documented bound, or ``objective_grad`` returns a
        cotangent of the wrong shape.
    """
    phases = np.ascontiguousarray(initial_phases, dtype=np.float64)
    frequencies = np.ascontiguousarray(omega, dtype=np.float64)
    coupling_matrix = np.ascontiguousarray(coupling, dtype=np.float64)
    count = _validate(
        phases, frequencies, coupling_matrix, n_sites, dt, steps_per_pulse, n_cycles, 0.0, 0
    )
    width = 2 * count + count * count
    omega_start = count
    coupling_start = 2 * count

    site_assignment = coordinated_reset_sites(count, n_sites)
    targets = coordinated_reset_phases(n_sites)
    sensitivity: NDArray[np.float64] = np.zeros((count, width), dtype=np.float64)
    sensitivity[:, :count] = np.eye(count)

    for _ in range(n_cycles):
        for site_index in range(n_sites):
            mask = site_assignment == site_index
            phases = phases.copy()
            phases[mask] = targets[site_index]
            sensitivity = sensitivity.copy()
            sensitivity[mask] = 0.0  # the reset overwrites the state, erasing its sensitivity
            for _ in range(steps_per_pulse):
                phases, sensitivity = _sensitivity_step(
                    phases,
                    sensitivity,
                    frequencies,
                    coupling_matrix,
                    dt,
                    count,
                    omega_start,
                    coupling_start,
                )

    value = float(objective(phases))
    cotangent = np.ascontiguousarray(objective_grad(phases), dtype=np.float64)
    if cotangent.shape != (count,):
        raise ValueError(
            f"objective_grad must return a ({count},) cotangent, got {cotangent.shape}"
        )
    flat = cotangent @ sensitivity
    return value, CoordinatedResetGradients(
        initial_phases=np.ascontiguousarray(flat[:count], dtype=np.float64),
        omega=np.ascontiguousarray(flat[omega_start:coupling_start], dtype=np.float64),
        coupling=np.ascontiguousarray(
            flat[coupling_start:].reshape(count, count), dtype=np.float64
        ),
    )


__all__ = [
    "CoordinatedResetGradients",
    "CoordinatedResetObjective",
    "CoordinatedResetObjectiveGrad",
    "CoordinatedResetTrajectory",
    "coordinated_reset_phases",
    "coordinated_reset_sites",
    "coordinated_reset_terminal_value_and_grad",
    "integrate_coordinated_reset",
]
