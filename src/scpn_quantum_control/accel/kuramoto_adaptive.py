# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Adaptive (plastic) Kuramoto dynamics, co-evolving phases and coupling
r"""Adaptive (plastic) Kuramoto model — phases and coupling weights co-evolve.

In a plastic network the coupling weights are not fixed: each weight ``K_{ij}`` adapts to the
phase relation it carries, while the phases evolve under the current weights. The two flows run
together,

.. math::

    \dot{θ}_i &= ω_i + \sum_j K_{ij}\,\sin(θ_j - θ_i), \\
    \dot{K}_{ij} &= ε\,\bigl(\cos(θ_j - θ_i) - K_{ij}\bigr),

the second line the Hebbian learning rule of Seliger, Young and Tsimring (2002): the weight
between two oscillators relaxes (at the plasticity rate ``ε``) towards the cosine of their phase
difference, so in-phase pairs grow an excitatory ``K \to +1`` and anti-phase pairs an inhibitory
``K \to -1``. The network therefore *learns* a connectivity that mirrors its own synchronisation
pattern — the Hebbian equilibrium ``K^*_{ij} = \cos(θ_j - θ_i)`` (:func:`hebbian_coupling_equilibrium`).

The joint state ``(θ, K)`` lives in ``ℝ^{N + N^2}`` and its linear stability is governed by the
coupled Jacobian (:func:`hebbian_adaptive_jacobian`), the block matrix

.. math::

    \begin{pmatrix}
      ∂\dot{θ}/∂θ & ∂\dot{θ}/∂K \\
      ∂\dot{K}/∂θ & ∂\dot{K}/∂K
    \end{pmatrix},

whose phase-phase block is the networked Kuramoto Jacobian, whose phase-coupling block injects
the ``\sin`` sensitivities, and whose coupling-coupling block is the decay ``-ε I`` of the
relaxation.

The integrator (:func:`integrate_adaptive_kuramoto`) advances the joint state by a functional
RK4 and is generic over the phase force and the plasticity rule, so it co-evolves any pairing of
a :data:`AdaptivePhaseForce` (e.g. the networked Kuramoto force) with a :data:`PlasticityRule`
(e.g. the Hebbian rule above). This is an analysis layer composing the polyglot networked force
and the accelerated order parameter, so it adds no compute kernel.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .order_parameter_observables import order_parameter

#: A phase force ``F(θ; K)`` of the adaptive model: it maps the current phases ``θ`` and the
#: current coupling matrix ``K`` to the coupling contribution of ``θ̇`` (length ``N``). The
#: networked Kuramoto force is one, ``lambda phases, coupling: networked_kuramoto_force(phases, coupling)``.
AdaptivePhaseForce = Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]

#: A plasticity rule ``K̇(θ, K)``: it maps the current phases and coupling to the time derivative
#: of the coupling matrix (shape ``(N, N)``). The Hebbian rule is one,
#: ``lambda phases, coupling: hebbian_plasticity_rate(phases, coupling, plasticity_rate=eps)``.
PlasticityRule = Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]


def _phase_difference_matrix(phases: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the matrix ``Δ_{ij} = θ_j − θ_i`` of pairwise phase differences."""
    return phases[np.newaxis, :] - phases[:, np.newaxis]


def hebbian_coupling_equilibrium(phases: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Return the Hebbian coupling equilibrium ``K^*_{ij} = cos(θ_j − θ_i)``.

    This is the fixed point of the Hebbian plasticity rule for a frozen phase configuration: the
    learned connectivity that the weights relax towards. For a phase-locked state it is the
    structure the network self-organises into — ``+1`` between in-phase oscillators, ``-1``
    between anti-phase ones.

    Parameters
    ----------
    phases : numpy.ndarray
        The phase vector ``θ`` (one-dimensional, length ``N``).

    Returns
    -------
    numpy.ndarray
        The ``(N, N)`` equilibrium coupling matrix.

    Raises
    ------
    ValueError
        If ``phases`` is not a non-empty one-dimensional array.
    """
    theta = np.ascontiguousarray(phases, dtype=np.float64)
    if theta.ndim != 1 or theta.size < 1:
        raise ValueError("phases must be a non-empty one-dimensional array")
    return np.asarray(np.cos(_phase_difference_matrix(theta)), dtype=np.float64)


def hebbian_plasticity_rate(
    phases: NDArray[np.float64],
    coupling: NDArray[np.float64],
    *,
    plasticity_rate: float,
) -> NDArray[np.float64]:
    r"""Return the Hebbian coupling rate ``K̇_{ij} = ε (cos(θ_j − θ_i) − K_{ij})``.

    The Seliger–Young–Tsimring plasticity rule: every weight relaxes, at rate ``ε``, towards the
    cosine of the phase difference it carries (:func:`hebbian_coupling_equilibrium`).

    Parameters
    ----------
    phases : numpy.ndarray
        The phase vector ``θ`` (one-dimensional, length ``N``).
    coupling : numpy.ndarray
        The current coupling matrix ``K`` (shape ``(N, N)``).
    plasticity_rate : float
        The plasticity rate ``ε`` (``≥ 0``); the inverse of the learning time constant.

    Returns
    -------
    numpy.ndarray
        The ``(N, N)`` coupling time derivative.

    Raises
    ------
    ValueError
        If the shapes are inconsistent or ``plasticity_rate`` is negative.
    """
    theta = np.ascontiguousarray(phases, dtype=np.float64)
    weights = np.ascontiguousarray(coupling, dtype=np.float64)
    _validate_adaptive_state(theta, weights)
    if plasticity_rate < 0.0:
        raise ValueError(f"plasticity_rate must be non-negative, got {plasticity_rate}")
    equilibrium = np.cos(_phase_difference_matrix(theta))
    return np.asarray(plasticity_rate * (equilibrium - weights), dtype=np.float64)


def _validate_adaptive_state(phases: NDArray[np.float64], coupling: NDArray[np.float64]) -> None:
    """Validate that ``phases`` is a non-empty vector and ``coupling`` the matching ``(N, N)``."""
    if phases.ndim != 1 or phases.size < 1:
        raise ValueError("phases must be a non-empty one-dimensional array")
    count = phases.size
    if coupling.shape != (count, count):
        raise ValueError(f"coupling must have shape ({count}, {count}), got {coupling.shape}")


@dataclass(frozen=True)
class AdaptiveTrajectory:
    """A co-evolving trajectory of the adaptive Kuramoto model sampled at every RK4 step.

    Attributes
    ----------
    times : numpy.ndarray
        The ``(n_steps + 1,)`` sample times ``0, dt, …, n_steps·dt``.
    phases : numpy.ndarray
        The ``(n_steps + 1, N)`` phase trajectory ``θ(t)``, left unwrapped.
    couplings : numpy.ndarray
        The ``(n_steps + 1, N, N)`` coupling trajectory ``K(t)``.
    order_parameter_series : numpy.ndarray
        The Kuramoto order parameter ``r(t)`` at every sample (length ``n_steps + 1``).
    """

    times: NDArray[np.float64]
    phases: NDArray[np.float64]
    couplings: NDArray[np.float64]
    order_parameter_series: NDArray[np.float64]

    @property
    def terminal_phases(self) -> NDArray[np.float64]:
        """The final phase vector of the trajectory."""
        return np.asarray(self.phases[-1], dtype=np.float64)

    @property
    def terminal_coupling(self) -> NDArray[np.float64]:
        """The final coupling matrix of the trajectory."""
        return np.asarray(self.couplings[-1], dtype=np.float64)

    def hebbian_equilibrium_gap(self) -> float:
        r"""Return ``max_{ij} |K_{ij}(T) − cos(θ_j(T) − θ_i(T))|`` at the final state.

        The largest deviation of the learned coupling from the Hebbian equilibrium: it tends to
        zero as the plastic network settles onto the learned structure, a scalar measure of how
        far the run is from the Hebbian fixed point.
        """
        gap = self.terminal_coupling - hebbian_coupling_equilibrium(self.terminal_phases)
        return float(np.max(np.abs(gap)))


def adaptive_vector_field(
    phases: NDArray[np.float64],
    coupling: NDArray[np.float64],
    omega: NDArray[np.float64],
    force: AdaptivePhaseForce,
    plasticity: PlasticityRule,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Return the joint vector field ``(θ̇, K̇)`` of the adaptive Kuramoto model.

    The phase velocity is ``θ̇ = ω + F(θ; K)`` for the supplied phase force ``F`` and the coupling
    velocity is ``K̇ = R(θ, K)`` for the supplied plasticity rule ``R``.

    Parameters
    ----------
    phases : numpy.ndarray
        The phase vector ``θ`` (one-dimensional, length ``N``).
    coupling : numpy.ndarray
        The coupling matrix ``K`` (shape ``(N, N)``).
    omega : numpy.ndarray
        The natural frequencies ``ω`` (length ``N``).
    force : callable
        The phase force ``F(θ; K)`` (see :data:`AdaptivePhaseForce`).
    plasticity : callable
        The plasticity rule ``K̇(θ, K)`` (see :data:`PlasticityRule`).

    Returns
    -------
    tuple of numpy.ndarray
        The phase velocity ``θ̇`` (length ``N``) and the coupling velocity ``K̇`` (shape ``(N, N)``).

    Raises
    ------
    ValueError
        If the state shapes or ``omega`` are inconsistent.
    """
    theta = np.ascontiguousarray(phases, dtype=np.float64)
    weights = np.ascontiguousarray(coupling, dtype=np.float64)
    frequencies = np.ascontiguousarray(omega, dtype=np.float64)
    _validate_adaptive_state(theta, weights)
    if frequencies.shape != theta.shape:
        raise ValueError(f"omega must have shape {theta.shape}, got {frequencies.shape}")
    phase_velocity = frequencies + force(theta, weights)
    coupling_velocity = plasticity(theta, weights)
    return np.asarray(phase_velocity, dtype=np.float64), np.asarray(
        coupling_velocity, dtype=np.float64
    )


def hebbian_adaptive_jacobian(
    phases: NDArray[np.float64],
    coupling: NDArray[np.float64],
    *,
    plasticity_rate: float,
) -> NDArray[np.float64]:
    r"""Return the coupled ``(N + N^2) × (N + N^2)`` Jacobian of the networked–Hebbian system.

    The Jacobian of the joint flow ``(θ̇, K̇)`` for the canonical pairing of the networked
    Kuramoto force ``θ̇_i = ω_i + Σ_j K_{ij} sin(θ_j − θ_i)`` with the Hebbian plasticity rule
    ``K̇_{ij} = ε (cos(θ_j − θ_i) − K_{ij})``, with the joint state ordered as
    ``[θ_0, …, θ_{N-1}, K_{00}, K_{01}, …, K_{N-1,N-1}]`` (the coupling flattened row-major). The
    four blocks are

    .. math::

        ∂\dot{θ}_i/∂θ_l &= K_{il}\cos(θ_l - θ_i)\ (l \ne i), \quad
            ∂\dot{θ}_i/∂θ_i = -\sum_{j \ne i} K_{ij}\cos(θ_j - θ_i), \\
        ∂\dot{θ}_i/∂K_{lm} &= δ_{li}\,\sin(θ_m - θ_i), \\
        ∂\dot{K}_{ij}/∂θ_l &= -ε\,\sin(θ_j - θ_i)\,(δ_{lj} - δ_{li}), \quad
        ∂\dot{K}_{ij}/∂K_{lm} = -ε\,δ_{il}δ_{jm}.

    Parameters
    ----------
    phases : numpy.ndarray
        The phase vector ``θ`` (one-dimensional, length ``N``).
    coupling : numpy.ndarray
        The coupling matrix ``K`` (shape ``(N, N)``).
    plasticity_rate : float
        The Hebbian plasticity rate ``ε`` (``≥ 0``).

    Returns
    -------
    numpy.ndarray
        The ``(N + N^2, N + N^2)`` coupled Jacobian.

    Raises
    ------
    ValueError
        If the state shapes are inconsistent or ``plasticity_rate`` is negative.
    """
    theta = np.ascontiguousarray(phases, dtype=np.float64)
    weights = np.ascontiguousarray(coupling, dtype=np.float64)
    _validate_adaptive_state(theta, weights)
    if plasticity_rate < 0.0:
        raise ValueError(f"plasticity_rate must be non-negative, got {plasticity_rate}")
    count = theta.size
    difference = _phase_difference_matrix(theta)  # Δ_{ij} = θ_j − θ_i
    cosine = np.cos(difference)
    sine = np.sin(difference)
    indices = np.arange(count)

    # ∂θ̇/∂θ — networked Kuramoto Jacobian (the j = i term drops because sin(0) = 0).
    weighted_cos = weights * cosine
    phase_phase = weighted_cos.copy()
    off_diagonal_row_sum = weighted_cos.sum(axis=1) - np.diag(weighted_cos)
    np.fill_diagonal(phase_phase, -off_diagonal_row_sum)

    # ∂θ̇_i/∂K_{lm} — non-zero only for l = i, equal to sin(θ_m − θ_i).
    phase_coupling_tensor = np.zeros((count, count, count), dtype=np.float64)
    phase_coupling_tensor[indices, indices, :] = sine
    phase_coupling = phase_coupling_tensor.reshape(count, count * count)

    # ∂K̇_{ij}/∂θ_l — relaxation of cos(θ_j − θ_i) towards K_{ij}.
    coupling_phase_tensor = np.zeros((count, count, count), dtype=np.float64)
    row, col = np.meshgrid(indices, indices, indexing="ij")
    coupling_phase_tensor[row, col, col] = -plasticity_rate * sine
    coupling_phase_tensor[row, col, row] += plasticity_rate * sine
    coupling_phase = coupling_phase_tensor.reshape(count * count, count)

    # ∂K̇/∂K — diagonal decay −ε.
    coupling_coupling = -plasticity_rate * np.eye(count * count, dtype=np.float64)

    top = np.hstack([phase_phase, phase_coupling])
    bottom = np.hstack([coupling_phase, coupling_coupling])
    return np.ascontiguousarray(np.vstack([top, bottom]), dtype=np.float64)


def integrate_adaptive_kuramoto(
    initial_phases: NDArray[np.float64],
    initial_coupling: NDArray[np.float64],
    omega: NDArray[np.float64],
    force: AdaptivePhaseForce,
    plasticity: PlasticityRule,
    *,
    dt: float,
    n_steps: int,
) -> AdaptiveTrajectory:
    r"""Integrate the adaptive Kuramoto model by a functional RK4 on the joint ``(θ, K)`` state.

    Both the phases and the coupling matrix are advanced together by the classical four-stage
    Runge–Kutta rule applied to :func:`adaptive_vector_field`, sampling ``θ``, ``K`` and the
    order parameter at every step. The phases are left unwrapped so the collective frequency is
    recoverable. The update is functional (no in-place mutation of the running state), so
    gradients propagate where ``force`` and ``plasticity`` are differentiable.

    Parameters
    ----------
    initial_phases : numpy.ndarray
        The initial phases ``θ(0)`` (one-dimensional, length ``N``).
    initial_coupling : numpy.ndarray
        The initial coupling matrix ``K(0)`` (shape ``(N, N)``).
    omega : numpy.ndarray
        The natural frequencies ``ω`` (length ``N``).
    force : callable
        The phase force ``F(θ; K)`` (see :data:`AdaptivePhaseForce`).
    plasticity : callable
        The plasticity rule ``K̇(θ, K)`` (see :data:`PlasticityRule`).
    dt : float
        The RK4 time step (``> 0``).
    n_steps : int
        The number of RK4 steps (``≥ 1``); the trajectory has ``n_steps + 1`` samples.

    Returns
    -------
    AdaptiveTrajectory
        The sampled phase, coupling and order-parameter trajectory.

    Raises
    ------
    ValueError
        If the state shapes / ``omega`` are inconsistent or ``dt`` / ``n_steps`` are out of range.
    """
    theta = np.ascontiguousarray(initial_phases, dtype=np.float64)
    weights = np.ascontiguousarray(initial_coupling, dtype=np.float64)
    frequencies = np.ascontiguousarray(omega, dtype=np.float64)
    _validate_adaptive_state(theta, weights)
    if frequencies.shape != theta.shape:
        raise ValueError(f"omega must have shape {theta.shape}, got {frequencies.shape}")
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    if n_steps < 1:
        raise ValueError(f"n_steps must be positive, got {n_steps}")
    count = theta.size

    def rhs(
        state_phases: NDArray[np.float64], state_coupling: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return frequencies + force(state_phases, state_coupling), plasticity(
            state_phases, state_coupling
        )

    phase_history = np.empty((n_steps + 1, count), dtype=np.float64)
    coupling_history = np.empty((n_steps + 1, count, count), dtype=np.float64)
    series = np.empty(n_steps + 1, dtype=np.float64)
    phase_history[0] = theta
    coupling_history[0] = weights
    series[0] = order_parameter(theta)
    for step in range(n_steps):
        p1, c1 = rhs(theta, weights)
        p2, c2 = rhs(theta + 0.5 * dt * p1, weights + 0.5 * dt * c1)
        p3, c3 = rhs(theta + 0.5 * dt * p2, weights + 0.5 * dt * c2)
        p4, c4 = rhs(theta + dt * p3, weights + dt * c3)
        theta = theta + (dt / 6.0) * (p1 + 2.0 * p2 + 2.0 * p3 + p4)
        weights = weights + (dt / 6.0) * (c1 + 2.0 * c2 + 2.0 * c3 + c4)
        phase_history[step + 1] = theta
        coupling_history[step + 1] = weights
        series[step + 1] = order_parameter(theta)
    times = dt * np.arange(n_steps + 1, dtype=np.float64)
    return AdaptiveTrajectory(times, phase_history, coupling_history, series)


__all__ = [
    "AdaptivePhaseForce",
    "AdaptiveTrajectory",
    "PlasticityRule",
    "adaptive_vector_field",
    "hebbian_adaptive_jacobian",
    "hebbian_coupling_equilibrium",
    "hebbian_plasticity_rate",
    "integrate_adaptive_kuramoto",
]
