# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Explosive-synchronisation continuation and hysteresis of the Kuramoto model
r"""Numerical continuation of the Kuramoto order parameter and its explosive-sync hysteresis loop.

A continuation sweep traces the steady-state order parameter ``r`` as the coupling ``K`` is varied
quasi-statically: each coupling on the grid is integrated to its steady state seeded from the
*previous* coupling's terminal phases (natural-parameter continuation), so a branch follows one
solution sheet until that sheet loses stability at a saddle-node and the state falls to the other
sheet. Sweeping the grid upward from an incoherent seed and downward from a coherent seed yields the
two branches of a **hysteresis loop**.

For the pairwise mean-field force ``F_j = K r sin(ψ − θ_j)`` the transition is continuous: the two
branches coincide and the loop has zero width. For the triadic (2-simplex) force
``F_j = K r² sin(2ψ − 2θ_j)`` the ``r²`` gain vanishes linearly at ``r = 0``, so the incoherent
state stays linearly stable while a coherent branch survives down to a lower saddle-node — the
branches separate over a finite coupling window and the transition is **explosive** (abrupt, with
hysteresis), reproducing the Skardal–Arenas higher-order phenomenology. The coherent branch
collapses sharply at a reproducible backward saddle-node ``K_b``; the incoherent branch only
synchronises far above it (a fluctuation-driven forward saddle-node), so the forward transition may
lie beyond the swept range and is reported as ``None`` when no up-crossing is observed.

This is an analysis layer over the synchronisation dynamics: the steady states are reached with a
fixed-step RK4 composing the polyglot mean-field forces and the order parameter is read with the
accelerated :func:`~scpn_quantum_control.accel.order_parameter_observables.order_parameter`, so the
module adds no compute kernel.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .order_parameter_observables import order_parameter
from .triadic_mean_field import triadic_mean_field_force

#: A mean-field coupling force ``F_j(θ; K)`` added to the natural frequencies; it maps the phase
#: vector and the scalar coupling to a force vector of the same length (the signature shared by
#: :func:`~scpn_quantum_control.accel.kuramoto_mean_field.mean_field_force` and
#: :func:`~scpn_quantum_control.accel.triadic_mean_field.triadic_mean_field_force`).
MeanFieldForce = Callable[[NDArray[np.float64], float], NDArray[np.float64]]


@dataclass(frozen=True)
class ContinuationBranch:
    """One branch of a coupling continuation: the steady-state order parameter along the grid.

    Attributes
    ----------
    coupling_values : numpy.ndarray
        The coupling grid in sweep order (ascending or descending).
    order_parameters : numpy.ndarray
        The steady-state order parameter ``r`` at each coupling (settle-window mean), aligned with
        ``coupling_values``.
    terminal_phases : numpy.ndarray
        The ``(grid, N)`` terminal phase vectors reached at each coupling, the continuation seeds.
    direction : str
        ``"ascending"`` or ``"descending"`` — the monotone direction of ``coupling_values``.
    """

    coupling_values: NDArray[np.float64]
    order_parameters: NDArray[np.float64]
    terminal_phases: NDArray[np.float64]
    direction: str

    @property
    def is_ascending(self) -> bool:
        """Whether the coupling grid is swept from low to high coupling."""
        return self.direction == "ascending"


@dataclass(frozen=True)
class HysteresisLoop:
    """The forward and backward branches of a coupling sweep and the hysteresis they enclose.

    Attributes
    ----------
    forward : ContinuationBranch
        The ascending sweep seeded from the incoherent state.
    backward : ContinuationBranch
        The descending sweep seeded from the coherent state.
    branch_separation : numpy.ndarray
        ``|r_backward − r_forward|`` on the shared ascending grid.
    midline : float
        The half-way order parameter ``(r_max + r_min)/2`` over both branches, the threshold used to
        locate the saddle-node transitions.
    forward_transition_coupling : float or None
        The lowest coupling at which the forward branch rises above ``midline``; ``None`` if it never
        does within the swept range (a forward saddle-node beyond the grid).
    backward_transition_coupling : float or None
        The lowest coupling at which the backward branch stands above ``midline`` — the coherent
        branch's collapse point; ``None`` if it never rises above ``midline``.
    hysteresis_width : float
        The coupling extent of the window where the branches are separated by more than the
        tolerance; ``0`` when they coincide.
    max_branch_separation : float
        The largest branch separation over the grid.
    is_hysteretic : bool
        Whether the branches separate by more than the tolerance anywhere on the grid.
    """

    forward: ContinuationBranch
    backward: ContinuationBranch
    branch_separation: NDArray[np.float64]
    midline: float
    forward_transition_coupling: float | None
    backward_transition_coupling: float | None
    hysteresis_width: float
    max_branch_separation: float
    is_hysteretic: bool


def _validate_sweep(
    omega: NDArray[np.float64],
    coupling_grid: NDArray[np.float64],
    initial_phases: NDArray[np.float64],
    dt: float,
    n_steps: int,
    settle_steps: int,
) -> None:
    """Validate the shared inputs of a continuation sweep."""
    if omega.ndim != 1 or omega.size < 1:
        raise ValueError("omega must be a non-empty one-dimensional array")
    if initial_phases.shape != omega.shape:
        raise ValueError(
            f"initial_phases must have shape {omega.shape}, got {initial_phases.shape}"
        )
    if coupling_grid.ndim != 1 or coupling_grid.size < 2:
        raise ValueError("coupling_grid must be a one-dimensional array of at least two couplings")
    steps = np.diff(coupling_grid)
    if not (np.all(steps > 0.0) or np.all(steps < 0.0)):
        raise ValueError("coupling_grid must be strictly monotonic")
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    if n_steps < 1:
        raise ValueError(f"n_steps must be positive, got {n_steps}")
    if not 1 <= settle_steps <= n_steps:
        raise ValueError(f"settle_steps must be in [1, {n_steps}], got {settle_steps}")


def _evolve_to_steady_state(
    initial_phases: NDArray[np.float64],
    omega: NDArray[np.float64],
    force: MeanFieldForce,
    coupling: float,
    dt: float,
    n_steps: int,
    settle_steps: int,
) -> tuple[NDArray[np.float64], float]:
    """Integrate ``θ̇ = ω + force(θ, K)`` by RK4, returning terminal phases and settle-mean ``r``."""
    phases = np.array(initial_phases, dtype=np.float64)
    settle_start = n_steps - settle_steps
    radius_sum = 0.0
    for step in range(n_steps):
        k1 = omega + force(phases, coupling)
        k2 = omega + force(phases + 0.5 * dt * k1, coupling)
        k3 = omega + force(phases + 0.5 * dt * k2, coupling)
        k4 = omega + force(phases + dt * k3, coupling)
        phases = phases + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if step >= settle_start:
            radius_sum += order_parameter(phases)
    return phases, radius_sum / settle_steps


def continuation_sweep(
    omega: NDArray[np.float64],
    force: MeanFieldForce,
    coupling_grid: NDArray[np.float64],
    initial_phases: NDArray[np.float64],
    *,
    dt: float,
    n_steps: int,
    settle_steps: int | None = None,
) -> ContinuationBranch:
    r"""Trace the steady-state order parameter along a coupling grid by natural-parameter continuation.

    Each coupling on ``coupling_grid`` is integrated to its steady state with a fixed-step RK4 of
    ``θ̇ = ω + force(θ, K)``, seeded from the terminal phases of the previous coupling (the first
    from ``initial_phases``). Following one solution sheet this way exposes the saddle-node where the
    sheet loses stability: the order parameter jumps as the state falls to the surviving branch.

    Parameters
    ----------
    omega : numpy.ndarray
        The natural frequencies ``ω`` (one-dimensional, length ``N``).
    force : callable
        The mean-field coupling force ``F_j(θ; K)`` added to ``ω`` (see :data:`MeanFieldForce`).
    coupling_grid : numpy.ndarray
        A strictly monotonic one-dimensional grid of couplings, swept in its given order.
    initial_phases : numpy.ndarray
        The seed phases for the first coupling (length ``N``).
    dt : float
        The RK4 time step (``> 0``).
    n_steps : int
        The number of RK4 steps per coupling (``≥ 1``).
    settle_steps : int, optional
        The number of trailing steps over which the order parameter is averaged to read the steady
        state; defaults to the final quarter of ``n_steps``. Must lie in ``[1, n_steps]``.

    Returns
    -------
    ContinuationBranch
        The order parameter and terminal phases at each coupling, in sweep order.

    Raises
    ------
    ValueError
        If any input is malformed (see the shape and range checks).
    """
    omega = np.ascontiguousarray(omega, dtype=np.float64)
    coupling_grid = np.ascontiguousarray(coupling_grid, dtype=np.float64)
    initial_phases = np.ascontiguousarray(initial_phases, dtype=np.float64)
    resolved_settle = settle_steps if settle_steps is not None else max(1, n_steps // 4)
    _validate_sweep(omega, coupling_grid, initial_phases, dt, n_steps, resolved_settle)

    radii = np.empty(coupling_grid.size, dtype=np.float64)
    terminals = np.empty((coupling_grid.size, omega.size), dtype=np.float64)
    seed = initial_phases
    for index, coupling in enumerate(coupling_grid):
        seed, radius = _evolve_to_steady_state(
            seed, omega, force, float(coupling), dt, n_steps, resolved_settle
        )
        radii[index] = radius
        terminals[index] = seed
    direction = "ascending" if coupling_grid[-1] > coupling_grid[0] else "descending"
    return ContinuationBranch(coupling_grid, radii, terminals, direction)


def _first_crossing(
    grid: NDArray[np.float64], radii: NDArray[np.float64], midline: float
) -> float | None:
    """Return the lowest coupling at which ``radii`` reaches ``midline``; ``None`` if it never does."""
    above = np.flatnonzero(radii >= midline)
    return float(grid[above[0]]) if above.size else None


def hysteresis_loop(
    omega: NDArray[np.float64],
    force: MeanFieldForce,
    coupling_grid: NDArray[np.float64],
    incoherent_phases: NDArray[np.float64],
    coherent_phases: NDArray[np.float64],
    *,
    dt: float,
    n_steps: int,
    settle_steps: int | None = None,
    separation_tolerance: float = 0.1,
) -> HysteresisLoop:
    r"""Sweep a coupling grid up from incoherence and down from coherence to map the hysteresis loop.

    The forward branch continues from ``incoherent_phases`` over the ascending ``coupling_grid``; the
    backward branch continues from ``coherent_phases`` over the reversed grid. Aligning the branches
    on the shared grid gives the separation ``|r_backward − r_forward|``; the saddle-node transitions
    are located where each branch crosses the half-way order parameter ``midline``.

    Parameters
    ----------
    omega : numpy.ndarray
        The natural frequencies ``ω`` (length ``N``).
    force : callable
        The mean-field coupling force (see :data:`MeanFieldForce`).
    coupling_grid : numpy.ndarray
        A strictly **ascending** one-dimensional coupling grid.
    incoherent_phases : numpy.ndarray
        The forward-branch seed (a desynchronised state, ``r ≈ 0``).
    coherent_phases : numpy.ndarray
        The backward-branch seed (a synchronised state, ``r ≈ 1``).
    dt : float
        The RK4 time step (``> 0``).
    n_steps : int
        The number of RK4 steps per coupling (``≥ 1``).
    settle_steps : int, optional
        The trailing-average window; defaults to the final quarter of ``n_steps``.
    separation_tolerance : float, optional
        The minimum branch separation counted as hysteresis (``> 0``).

    Returns
    -------
    HysteresisLoop
        The two branches, their separation, the saddle-node transitions and the loop width.

    Raises
    ------
    ValueError
        If ``coupling_grid`` is not strictly ascending, ``separation_tolerance`` is not positive, or
        any sweep input is malformed.
    """
    coupling_grid = np.ascontiguousarray(coupling_grid, dtype=np.float64)
    if coupling_grid.ndim != 1 or coupling_grid.size < 2 or np.any(np.diff(coupling_grid) <= 0.0):
        raise ValueError("coupling_grid must be a strictly ascending one-dimensional array")
    if separation_tolerance <= 0.0:
        raise ValueError(f"separation_tolerance must be positive, got {separation_tolerance}")

    forward = continuation_sweep(
        omega,
        force,
        coupling_grid,
        incoherent_phases,
        dt=dt,
        n_steps=n_steps,
        settle_steps=settle_steps,
    )
    backward = continuation_sweep(
        omega,
        force,
        coupling_grid[::-1],
        coherent_phases,
        dt=dt,
        n_steps=n_steps,
        settle_steps=settle_steps,
    )

    forward_radii = forward.order_parameters
    backward_radii = backward.order_parameters[::-1]  # realign the descending sweep onto the grid
    midline = 0.5 * (
        float(max(forward_radii.max(), backward_radii.max()))
        + float(min(forward_radii.min(), backward_radii.min()))
    )
    separation = np.abs(backward_radii - forward_radii)
    window = np.flatnonzero(separation > separation_tolerance)
    width = float(coupling_grid[window[-1]] - coupling_grid[window[0]]) if window.size else 0.0
    max_separation = float(separation.max())
    return HysteresisLoop(
        forward=forward,
        backward=backward,
        branch_separation=separation,
        midline=midline,
        forward_transition_coupling=_first_crossing(coupling_grid, forward_radii, midline),
        backward_transition_coupling=_first_crossing(coupling_grid, backward_radii, midline),
        hysteresis_width=width,
        max_branch_separation=max_separation,
        is_hysteretic=max_separation > separation_tolerance,
    )


def triadic_hysteresis_loop(
    omega: NDArray[np.float64],
    coupling_grid: NDArray[np.float64],
    incoherent_phases: NDArray[np.float64],
    coherent_phases: NDArray[np.float64],
    *,
    dt: float,
    n_steps: int,
    settle_steps: int | None = None,
    separation_tolerance: float = 0.1,
) -> HysteresisLoop:
    r"""Map the explosive-synchronisation hysteresis loop of the triadic (2-simplex) Kuramoto model.

    A convenience wrapper that wires the triadic mean-field force ``F_j = K r² sin(2ψ − 2θ_j)`` into
    :func:`hysteresis_loop`. The ``r²`` gain makes the transition explosive: the coherent branch
    survives below the forward onset, so the loop has a finite width.

    Parameters
    ----------
    omega : numpy.ndarray
        The natural frequencies ``ω`` (length ``N``).
    coupling_grid : numpy.ndarray
        A strictly ascending coupling grid.
    incoherent_phases, coherent_phases : numpy.ndarray
        The forward (desynchronised) and backward (synchronised) branch seeds.
    dt : float
        The RK4 time step (``> 0``).
    n_steps : int
        The number of RK4 steps per coupling (``≥ 1``).
    settle_steps : int, optional
        The trailing-average window; defaults to the final quarter of ``n_steps``.
    separation_tolerance : float, optional
        The minimum branch separation counted as hysteresis.

    Returns
    -------
    HysteresisLoop
        The explosive-synchronisation hysteresis loop of the triadic model.
    """
    return hysteresis_loop(
        omega,
        triadic_mean_field_force,
        coupling_grid,
        incoherent_phases,
        coherent_phases,
        dt=dt,
        n_steps=n_steps,
        settle_steps=settle_steps,
        separation_tolerance=separation_tolerance,
    )


__all__ = [
    "ContinuationBranch",
    "HysteresisLoop",
    "MeanFieldForce",
    "continuation_sweep",
    "hysteresis_loop",
    "triadic_hysteresis_loop",
]
